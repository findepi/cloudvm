#!/usr/bin/env python3
"""Unit tests for cloudvm. Run with: python3 -m unittest discover tests"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import botocore.exceptions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cloudvm import _build_info, _cli as cb  # noqa: E402


SAMPLE_CONFIG = """\
# leading comment

Host *
    TCPKeepAlive yes
    ServerAliveInterval 60

Host my-dev-box
    User ubuntu
    # current IP
    Hostname 1.1.1.1
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host my-dev-box-3
    User ubuntu
    # current IP
    Hostname 2.2.2.2
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host my-dev-box-virginia
    User ubuntu
    # current IP
    Hostname 3.3.3.3
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host i-* mi-*
    ProxyCommand sh -c "aws ssm start-session --target %h"
    User ubuntu
    StrictHostKeyChecking no
"""


# noinspection PyPep8Naming,PyAttributeOutsideInit
class SshConfigEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".sshconfig", delete=False)
        self.tmp.write(SAMPLE_CONFIG)
        self.tmp.close()
        self.path = Path(self.tmp.name)
        # Redirect module globals so update_ssh_config writes to the temp file.
        self._orig_config = cb.SSH_CONFIG
        self._orig_backup = cb.SSH_CONFIG_BACKUP
        cb.SSH_CONFIG = self.path
        cb.SSH_CONFIG_BACKUP = self.path.with_suffix(self.path.suffix + ".bak")

    def tearDown(self):
        cb.SSH_CONFIG = self._orig_config
        cb.SSH_CONFIG_BACKUP = self._orig_backup
        for p in (self.path, self.path.with_suffix(self.path.suffix + ".bak")):
            if p.exists():
                p.unlink()

    def _lines(self):
        return self.path.read_text().splitlines(keepends=True)

    # --- find_host_block ---

    def test_finds_exact_host_block(self):
        block = cb.find_host_block(self._lines(), "my-dev-box-3")
        self.assertIsNotNone(block)
        start, end = block
        self.assertIn("my-dev-box-3", self._lines()[start])

    def test_does_not_match_glob_block(self):
        # `Host i-* mi-*` must not match a literal instance id
        self.assertIsNone(cb.find_host_block(self._lines(), "i-0123abc"))
        self.assertIsNone(cb.find_host_block(self._lines(), "mi-xyz"))

    def test_does_not_match_wildcard_block_as_fallback(self):
        # `Host *` is a real glob, not a fallback for any lookup
        self.assertIsNone(cb.find_host_block(self._lines(), "totally-unrelated"))

    def test_finds_one_of_multiple_tokens_on_host_line(self):
        # If a Host line has several aliases, any of them should match
        config = "Host alpha beta gamma\n    Hostname 1.2.3.4\n"
        lines = config.splitlines(keepends=True)
        self.assertEqual(cb.find_host_block(lines, "beta"), (0, 2))
        self.assertEqual(cb.find_host_block(lines, "alpha"), (0, 2))
        self.assertEqual(cb.find_host_block(lines, "gamma"), (0, 2))
        # And exact-only: glob characters don't match
        self.assertIsNone(cb.find_host_block(lines, "alph"))

    # --- update_ssh_config (in-place edit) ---

    def test_updates_hostname_only(self):
        before = self.path.read_text()
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        after = self.path.read_text()
        diff_lines = [(b, a) for b, a in zip(before.splitlines(), after.splitlines()) if b != a]
        # Exactly one line should differ
        self.assertEqual(len(diff_lines), 1)
        b, a = diff_lines[0]
        self.assertIn("2.2.2.2", b)
        self.assertIn("9.9.9.9", a)
        # And the same number of total lines
        self.assertEqual(before.count("\n"), after.count("\n"))

    def test_preserves_indentation_and_comments(self):
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        text = self.path.read_text()
        # `# current IP` comment must still sit immediately above the Hostname
        self.assertIn("    # current IP\n    Hostname 9.9.9.9\n", text)

    def test_idempotent_when_ip_unchanged(self):
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        # Second call should not change the file and should not produce a new backup
        # (we delete the backup first to confirm no second write happened)
        cb.SSH_CONFIG_BACKUP.unlink()
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        self.assertFalse(cb.SSH_CONFIG_BACKUP.exists())

    def test_other_blocks_untouched(self):
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        text = self.path.read_text()
        # The other my-dev-box-* blocks must still hold their original IPs
        self.assertIn("Hostname 1.1.1.1", text)
        self.assertIn("Hostname 3.3.3.3", text)
        # The glob block's content also intact
        self.assertIn("Host i-* mi-*", text)
        self.assertIn("ProxyCommand sh -c", text)

    def test_backup_holds_previous_content(self):
        original = self.path.read_text()
        cb.update_ssh_config("my-dev-box-3", "9.9.9.9")
        self.assertEqual(cb.SSH_CONFIG_BACKUP.read_text(), original)

    def test_injects_hostname_when_missing(self):
        config = "Host bare\n    User someone\n"
        self.path.write_text(config)
        cb.update_ssh_config("bare", "5.5.5.5")
        text = self.path.read_text()
        self.assertIn("Hostname 5.5.5.5", text)
        self.assertIn("User someone", text)

    # --- _similar_block_defaults ---

    def test_similar_block_defaults_picks_longest_prefix(self):
        defaults = cb._similar_block_defaults(self._lines(), "my-dev-box-warsaw")
        self.assertEqual(defaults.get("user"), "ubuntu")
        self.assertEqual(defaults.get("identityfile"), "~/.ssh/id_ed25519")

    def test_similar_block_defaults_skips_globs(self):
        # `Host *` and `Host i-* mi-*` must not count as prefix matches
        defaults = cb._similar_block_defaults(self._lines(), "completely-unrelated-name")
        self.assertEqual(defaults, {})


class RegionGlobTests(unittest.TestCase):
    def test_split_csv_flattens(self):
        self.assertEqual(cb._split_csv(["a,b", "c"]), ["a", "b", "c"])

    def test_split_csv_trims_and_drops_empty(self):
        self.assertEqual(cb._split_csv([" a , b ", "", ",,"]), ["a", "b"])

    def test_match_globs_basic(self):
        regions = ["eu-central-1", "eu-west-1", "us-east-1", "us-west-2", "ap-south-1"]
        self.assertEqual(cb.match_globs(regions, ["eu-central-*"]), ["eu-central-1"])
        self.assertEqual(cb.match_globs(regions, ["us-*"]), ["us-east-1", "us-west-2"])

    def test_match_globs_multiple_patterns_unioned_and_deduped(self):
        regions = ["eu-central-1", "us-east-1", "us-east-2"]
        result = cb.match_globs(regions, ["eu-central-*", "us-east-*", "us-*"])
        # us-east-1 and us-east-2 match both us-east-* and us-*, but should appear once each
        self.assertEqual(result, ["eu-central-1", "us-east-1", "us-east-2"])

    def test_match_globs_no_matches(self):
        self.assertEqual(cb.match_globs(["eu-central-1", "us-east-1"], ["af-*"]), [])

    def test_match_globs_question_mark(self):
        regions = ["us-east-1", "us-east-2", "us-east-11"]
        self.assertEqual(cb.match_globs(regions, ["us-east-?"]), ["us-east-1", "us-east-2"])


class HasWildcardTests(unittest.TestCase):
    def test_literal(self):
        self.assertFalse(cb._has_wildcard("us-east-1"))
        self.assertFalse(cb._has_wildcard(""))

    def test_star(self):
        self.assertTrue(cb._has_wildcard("us-*"))
        self.assertTrue(cb._has_wildcard("*"))

    def test_question_mark(self):
        self.assertTrue(cb._has_wildcard("us-east-?"))

    def test_bracket_class(self):
        self.assertTrue(cb._has_wildcard("us-[ew]ast-1"))


class ResolveRegionsTests(unittest.TestCase):
    """`_resolve_regions` owns the CLI-arg → region-list mapping: split, fallback, expand."""

    def test_none_falls_back_to_effective_region(self):
        with (
            patch.object(cb, "_effective_region", return_value="eu-central-1") as eff,
            patch.object(cb, "list_aws_regions") as lr,
        ):
            self.assertEqual(cb._resolve_regions(None), ["eu-central-1"])
        eff.assert_called_once_with()
        lr.assert_not_called()

    def test_empty_list_falls_back_to_effective_region(self):
        with (
            patch.object(cb, "_effective_region", return_value="eu-central-1"),
            patch.object(cb, "list_aws_regions") as lr,
        ):
            self.assertEqual(cb._resolve_regions([]), ["eu-central-1"])
        lr.assert_not_called()

    def test_all_literal_skips_describe_regions(self):
        with patch.object(cb, "list_aws_regions") as lr:
            self.assertEqual(
                set(cb._resolve_regions(["us-east-1,eu-central-1"])),
                {"us-east-1", "eu-central-1"},
            )
        lr.assert_not_called()

    def test_all_literal_dedupes(self):
        with patch.object(cb, "list_aws_regions") as lr:
            self.assertEqual(
                set(cb._resolve_regions(["us-east-1", "eu-central-1", "us-east-1"])),
                {"us-east-1", "eu-central-1"},
            )
        lr.assert_not_called()

    def test_wildcard_enumerates_and_globs(self):
        with patch.object(cb, "list_aws_regions", return_value=["eu-central-1", "us-east-1", "us-west-2"]) as lr:
            self.assertEqual(cb._resolve_regions(["us-*"]), ["us-east-1", "us-west-2"])
        lr.assert_called_once_with()

    def test_mixed_literal_and_wildcard_enumerates(self):
        # As soon as one pattern has wildcards we must enumerate, so the literal also goes
        # through `match_globs` (validating it exists).
        with patch.object(cb, "list_aws_regions", return_value=["eu-central-1", "us-east-1"]) as lr:
            self.assertEqual(
                sorted(cb._resolve_regions(["us-east-1", "eu-*"])),
                ["eu-central-1", "us-east-1"],
            )
        lr.assert_called_once_with()

    def test_wildcard_no_matches_raises(self):
        with patch.object(cb, "list_aws_regions", return_value=["us-east-1", "eu-central-1"]):
            with self.assertRaises(cb.CloudvmError):
                cb._resolve_regions(["af-*"])


class CmdListTests(unittest.TestCase):
    @staticmethod
    def _args(region):
        return argparse.Namespace(region=region, name="*")

    def test_per_region_calls_are_concurrent(self):
        """ThreadPoolExecutor must actually overlap the calls — verify by holding each call
        on a barrier that only releases once all workers have entered."""
        import threading

        regions_in = ["us-east-1", "eu-central-1", "ap-south-1"]
        barrier = threading.Barrier(len(regions_in), timeout=2.0)

        def slow(name, region):
            barrier.wait()  # blocks until all 3 workers arrive — proves concurrency
            return [[region, "x", "running", "1.1.1.1"]]

        with patch.object(cb, "_list_in_region", side_effect=slow):
            rc = cb.cmd_list(self._args([",".join(regions_in)]))
        self.assertEqual(rc, 0)


class FormatTableTests(unittest.TestCase):
    def test_basic_table(self):
        out = cb.format_table(
            ["name", "status", "public IP"],
            [["alpha", "running", "1.2.3.4"], ["beta", "stopped", "-"]],
        )
        lines = out.splitlines()
        self.assertEqual(lines[0], "name   status   public IP")
        # Header separator should match column widths
        self.assertEqual(lines[1], "-----  -------  ---------")
        self.assertEqual(lines[2], "alpha  running  1.2.3.4  ")
        self.assertEqual(lines[3], "beta   stopped  -        ")

    def test_widens_to_longest_value(self):
        out = cb.format_table(
            ["a", "b"],
            [["short", "x"], ["a-much-longer-value", "yy"]],
        )
        lines = out.splitlines()
        # Widths: a -> 19 (longest data), b -> 2 ("yy"); separator is two spaces
        self.assertEqual(lines[0], "a".ljust(19) + "  " + "b".ljust(2))
        self.assertEqual(lines[2], "short".ljust(19) + "  " + "x".ljust(2))
        self.assertEqual(lines[3], "a-much-longer-value" + "  " + "yy")

    def test_col_wrappers_applied_after_padding(self):
        # Per-column wrappers are applied to data cells after the cell has been padded,
        # so they can safely add zero-width decoration (e.g., ANSI color) without
        # disturbing alignment.
        out = cb.format_table(
            ["name", "status"],
            [["alpha", "running"], ["beta", "stopped"]],
            col_wrappers=[None, lambda s: f"<{s}>"],
        )
        lines = out.splitlines()
        # Column widths: name -> 5 ("alpha"), status -> 7 ("running"/"stopped")
        self.assertEqual(lines[0], "name ".ljust(5) + "  " + "status".ljust(7))
        # "running" is already 7 chars, so ljust(7) is a no-op; wrapper wraps it as-is
        self.assertEqual(lines[2], "alpha" + "  " + "<running>")
        self.assertEqual(lines[3], "beta " + "  " + "<stopped>")

    def test_col_wrappers_default_none_means_no_wrapping(self):
        out = cb.format_table(["a", "b"], [["x", "y"]])
        self.assertEqual(out.splitlines()[2], "x  y")


class HostLineParseTests(unittest.TestCase):
    def test_host_tokens_basic(self):
        self.assertEqual(cb._host_tokens("Host foo\n"), ["foo"])

    def test_host_tokens_multiple(self):
        self.assertEqual(cb._host_tokens("Host foo bar baz\n"), ["foo", "bar", "baz"])

    def test_host_tokens_indented(self):
        # OpenSSH allows leading whitespace on Host lines
        self.assertEqual(cb._host_tokens("  Host indented\n"), ["indented"])

    def test_host_tokens_case_insensitive(self):
        self.assertEqual(cb._host_tokens("host lower\n"), ["lower"])
        self.assertEqual(cb._host_tokens("HOST upper\n"), ["upper"])

    def test_non_host_line_returns_none(self):
        self.assertIsNone(cb._host_tokens("HostName 1.2.3.4\n"))
        self.assertIsNone(cb._host_tokens("    Hostname 1.2.3.4\n"))
        self.assertIsNone(cb._host_tokens("# Host comment\n"))


class VersionStringTests(unittest.TestCase):
    def test_without_build_info(self):
        with patch.object(_build_info, "COMMIT", ""), patch.object(_build_info, "DATE", ""):
            with patch.object(cb, "_installed_version", return_value="1.2.3"):
                self.assertEqual(cb._version_string(), "cloudvm 1.2.3 ( )")

    def test_with_commit_and_date(self):
        with patch.object(_build_info, "COMMIT", "abc123def"), patch.object(_build_info, "DATE", "2026-06-03"):
            with patch.object(cb, "_installed_version", return_value="1.2.3"):
                self.assertEqual(cb._version_string(), "cloudvm 1.2.3 (abc123def 2026-06-03)")


class CompleteRegionTests(unittest.TestCase):
    def test_returns_regions(self):
        with patch.object(cb, "list_aws_regions", return_value=["eu-central-1", "us-east-1"]):
            self.assertEqual(cb._complete_region(prefix="eu-"), ["eu-central-1", "us-east-1"])

    def test_swallows_exceptions(self):
        with patch.object(cb, "list_aws_regions", side_effect=RuntimeError("boom")):
            self.assertEqual(cb._complete_region(prefix="eu-"), [])


class CompleteInstanceNameTests(unittest.TestCase):
    def test_uses_explicit_region_over_configured(self):
        seen = {}

        def fake_list_in_region(name, region):
            seen["region"] = region
            return [[region, "findepi-dev", "running", "1.2.3.4"]]

        with (
            patch.object(cb, "_list_in_region", side_effect=fake_list_in_region),
            patch.object(cb, "_configured_region", return_value="us-east-1"),
        ):
            out = cb._complete_instance_name(prefix="", parsed_args=argparse.Namespace(region="eu-central-1"))
        self.assertEqual(out, ["findepi-dev"])
        self.assertEqual(seen["region"], "eu-central-1")

    def test_falls_back_to_configured_region(self):
        seen = {}

        def fake_list_in_region(name, region):
            seen["region"] = region
            return [[region, "box", "stopped", "-"]]

        with (
            patch.object(cb, "_list_in_region", side_effect=fake_list_in_region),
            patch.object(cb, "_configured_region", return_value="eu-west-1"),
        ):
            out = cb._complete_instance_name(prefix="", parsed_args=argparse.Namespace(region=None))
        self.assertEqual(out, ["box"])
        self.assertEqual(seen["region"], "eu-west-1")

    def test_returns_empty_when_no_region_available(self):
        with (
            patch.object(cb, "_configured_region", return_value=None),
            patch.object(cb, "_list_in_region") as list_mock,
        ):
            out = cb._complete_instance_name(prefix="", parsed_args=argparse.Namespace(region=None))
        self.assertEqual(out, [])
        list_mock.assert_not_called()

    def test_sorts_dedupes_and_drops_empty_names(self):
        rows = [
            ["eu-central-1", "zeta", "running", "1.1.1.1"],
            ["eu-central-1", "alpha", "stopped", "-"],
            ["eu-central-1", "alpha", "running", "2.2.2.2"],
            ["eu-central-1", "", "stopped", "-"],
        ]
        with patch.object(cb, "_list_in_region", return_value=rows):
            out = cb._complete_instance_name(prefix="", parsed_args=argparse.Namespace(region="eu-central-1"))
        self.assertEqual(out, ["alpha", "zeta"])

    def test_swallows_exceptions(self):
        with patch.object(cb, "_list_in_region", side_effect=RuntimeError("boom")):
            out = cb._complete_instance_name(prefix="", parsed_args=argparse.Namespace(region="eu-central-1"))
        self.assertEqual(out, [])

    def test_handles_missing_parsed_args(self):
        with patch.object(cb, "_configured_region", return_value=None):
            self.assertEqual(cb._complete_instance_name(prefix="", parsed_args=None), [])


class IsExpiredCredentialsTests(unittest.TestCase):
    def test_sso_token_load_error(self):
        self.assertTrue(cb._is_expired_credentials(botocore.exceptions.SSOTokenLoadError(error_msg="x")))

    def test_unauthorized_sso_token_error(self):
        self.assertTrue(cb._is_expired_credentials(botocore.exceptions.UnauthorizedSSOTokenError()))

    def test_token_retrieval_error(self):
        self.assertTrue(
            cb._is_expired_credentials(botocore.exceptions.TokenRetrievalError(provider="sso", error_msg="x"))
        )

    def test_client_error_expired_token(self):
        for code in ("ExpiredToken", "ExpiredTokenException", "RequestExpired"):
            with self.subTest(code=code):
                err = botocore.exceptions.ClientError({"Error": {"Code": code, "Message": "m"}}, "DescribeInstances")
                self.assertTrue(cb._is_expired_credentials(err))

    def test_client_error_other_codes_are_not_expired(self):
        for code in ("AccessDenied", "UnauthorizedOperation", "InvalidInstanceID.NotFound"):
            with self.subTest(code=code):
                err = botocore.exceptions.ClientError({"Error": {"Code": code, "Message": "m"}}, "DescribeInstances")
                self.assertFalse(cb._is_expired_credentials(err))

    def test_waiter_error_with_expired_token(self):
        for code in ("ExpiredToken", "ExpiredTokenException", "RequestExpired"):
            with self.subTest(code=code):
                err = botocore.exceptions.WaiterError(
                    name="instance_running",
                    reason=f"An error occurred ({code})",
                    last_response={"Error": {"Code": code, "Message": "m"}},
                )
                self.assertTrue(cb._is_expired_credentials(err))

    def test_waiter_error_with_other_code_is_not_expired(self):
        err = botocore.exceptions.WaiterError(
            name="instance_running",
            reason="Max attempts exceeded",
            last_response={"Reservations": []},
        )
        self.assertFalse(cb._is_expired_credentials(err))

    def test_unrelated_exception(self):
        self.assertFalse(cb._is_expired_credentials(ValueError("nope")))
        self.assertFalse(cb._is_expired_credentials(RuntimeError("nope")))


class SsoLoginTests(unittest.TestCase):
    """Validate that `_sso_login` shells out to `aws sso login` and clears the session cache."""

    def test_invokes_aws_sso_login_subprocess(self):
        # Seed the session/client cache so we can assert it gets cleared.
        cb._session = object()
        cb._clients["us-east-1"] = object()
        with patch.object(cb.subprocess, "run") as run:
            run.return_value = MagicMock(returncode=0)
            cb._sso_login()
        run.assert_called_once_with(["aws", "sso", "login"])
        self.assertIsNone(cb._session)
        self.assertEqual(cb._clients, {})

    def test_raises_when_aws_sso_login_fails(self):
        with patch.object(cb.subprocess, "run") as run:
            run.return_value = MagicMock(returncode=2)
            with self.assertRaises(cb.CloudvmError):
                cb._sso_login()


class AwsCallRetryTests(unittest.TestCase):
    """Validate `_aws_call`: SSO login fires iff the first call raises an expired-token error."""

    def test_no_error_does_not_invoke_sso_login(self):
        fn = MagicMock(return_value="ok")
        with patch.object(cb, "_sso_login") as login:
            self.assertEqual(cb._aws_call(fn), "ok")
        fn.assert_called_once_with()
        login.assert_not_called()

    def test_non_sso_error_propagates_without_login(self):
        boom = RuntimeError("network broke")
        fn = MagicMock(side_effect=boom)
        with patch.object(cb, "_sso_login") as login:
            with self.assertRaises(RuntimeError):
                cb._aws_call(fn)
        fn.assert_called_once_with()
        login.assert_not_called()

    def test_client_error_with_non_expired_code_is_wrapped_without_login(self):
        err = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidInstanceID.NotFound", "Message": "m"}}, "DescribeInstances"
        )
        fn = MagicMock(side_effect=err)
        with patch.object(cb, "_sso_login") as login:
            with self.assertRaises(cb.CloudvmError):
                cb._aws_call(fn)
        fn.assert_called_once_with()
        login.assert_not_called()

    def test_botocore_error_is_wrapped_without_login(self):
        # NoRegionError is a BotoCoreError (not a ClientError) — verifies the BotoCoreError branch.
        err = botocore.exceptions.NoRegionError()
        fn = MagicMock(side_effect=err)
        with patch.object(cb, "_sso_login") as login:
            with self.assertRaises(cb.CloudvmError):
                cb._aws_call(fn)
        fn.assert_called_once_with()
        login.assert_not_called()

    def test_sso_error_triggers_login_and_retries(self):
        first = botocore.exceptions.SSOTokenLoadError(error_msg="expired")
        fn = MagicMock(side_effect=[first, "ok"])
        with patch.object(cb, "_sso_login") as login:
            self.assertEqual(cb._aws_call(fn), "ok")
        self.assertEqual(fn.call_count, 2)
        login.assert_called_once_with()

    def test_expired_token_client_error_triggers_login_and_retries(self):
        err = botocore.exceptions.ClientError({"Error": {"Code": "ExpiredToken", "Message": "m"}}, "DescribeInstances")
        fn = MagicMock(side_effect=[err, {"Reservations": []}])
        with patch.object(cb, "_sso_login") as login:
            self.assertEqual(cb._aws_call(fn), {"Reservations": []})
        self.assertEqual(fn.call_count, 2)
        login.assert_called_once_with()

    def test_retry_only_happens_once(self):
        """If the call still fails after login, the second failure propagates — no infinite loop."""
        err = botocore.exceptions.SSOTokenLoadError(error_msg="still bad")
        fn = MagicMock(side_effect=err)
        with patch.object(cb, "_sso_login") as login:
            with self.assertRaises(cb.CloudvmError):
                cb._aws_call(fn)
        self.assertEqual(fn.call_count, 2)
        login.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
