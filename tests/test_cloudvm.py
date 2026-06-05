#!/usr/bin/env python3
"""Unit tests for cloudvm. Run with: python3 -m unittest discover tests"""

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


def _fake_aws_configure_list(stdout: str):
    """Return a side_effect for run_aws that returns `stdout` for `configure list`."""

    def _run(args, **kwargs):
        assert args[:2] == ["configure", "list"], args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    return _run


CONFIGURE_LIST_CONFIG_FILE = """\
      NAME       : VALUE                    : TYPE             : LOCATION
profile    : <not set>                : None             : None
access_key : <not set>                : None             : None
secret_key : <not set>                : None             : None
region     : us-east-1                : config-file      : ~/.aws/config
"""

CONFIGURE_LIST_ENV = """\
      NAME       : VALUE                    : TYPE             : LOCATION
profile    : <not set>                : None             : None
access_key : <not set>                : None             : None
secret_key : <not set>                : None             : None
region     : eu-central-1             : env              : ['AWS_REGION', 'AWS_DEFAULT_REGION']
"""

CONFIGURE_LIST_NOT_SET = """\
      NAME       : VALUE                    : TYPE             : LOCATION
profile    : <not set>                : None             : None
access_key : <not set>                : None             : None
secret_key : <not set>                : None             : None
region     : <not set>                : None             : None
"""


class ConfiguredRegionTests(unittest.TestCase):
    def test_reads_region_from_config_file(self):
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(CONFIGURE_LIST_CONFIG_FILE)):
            self.assertEqual(cb._configured_region(), "us-east-1")

    def test_reads_region_from_env(self):
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(CONFIGURE_LIST_ENV)):
            self.assertEqual(cb._configured_region(), "eu-central-1")

    def test_returns_none_when_not_set(self):
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(CONFIGURE_LIST_NOT_SET)):
            self.assertIsNone(cb._configured_region())

    def test_accepts_gov_and_china_regions(self):
        for region in ("us-gov-east-1", "us-gov-west-1", "cn-north-1", "cn-northwest-1", "ap-northeast-2"):
            with self.subTest(region=region):
                stdout = CONFIGURE_LIST_CONFIG_FILE.replace("us-east-1", region)
                with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(stdout)):
                    self.assertEqual(cb._configured_region(), region)

    def test_fails_loudly_when_separator_changes(self):
        # Same data but with `|` instead of `:` — old code would silently misread.
        stdout = CONFIGURE_LIST_CONFIG_FILE.replace(":", "|")
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(stdout)):
            with self.assertRaises(cb.CloudvmError):
                cb._configured_region()

    def test_fails_loudly_on_unexpected_value(self):
        stdout = CONFIGURE_LIST_CONFIG_FILE.replace("us-east-1", "EAST")
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(stdout)):
            with self.assertRaises(cb.CloudvmError):
                cb._configured_region()

    def test_fails_when_region_row_missing(self):
        stdout = "\n".join(line for line in CONFIGURE_LIST_CONFIG_FILE.splitlines() if not line.startswith("region"))
        with patch.object(cb, "run_aws", side_effect=_fake_aws_configure_list(stdout)):
            with self.assertRaises(cb.CloudvmError):
                cb._configured_region()


class CompleteRegionTests(unittest.TestCase):
    def test_returns_regions(self):
        with patch.object(cb, "list_aws_regions", return_value=["eu-central-1", "us-east-1"]):
            self.assertEqual(cb._complete_region(prefix="eu-"), ["eu-central-1", "us-east-1"])

    def test_swallows_exceptions(self):
        with patch.object(cb, "list_aws_regions", side_effect=RuntimeError("boom")):
            self.assertEqual(cb._complete_region(prefix="eu-"), [])


if __name__ == "__main__":
    unittest.main()
