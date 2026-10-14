#!/usr/bin/env python3
"""Manage cloud development instances. Currently, AWS EC2 is supported."""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NoReturn

import argcomplete

from cloudvm import _build_info


SSH_CONFIG = Path.home() / ".ssh" / "config"
SSH_CONFIG_BACKUP = Path.home() / ".ssh" / "config.bak"

# How long to wait for a public IP after the instance reports running.
IP_POLL_SECONDS = 60
IP_POLL_INTERVAL = 1.0

# Backoff between start attempts while EC2 reports InsufficientInstanceCapacity (`up --retry`).
CAPACITY_RETRY_INITIAL = 5.0
CAPACITY_RETRY_MAX = 60.0
CAPACITY_RETRY_FACTOR = 1.5
CAPACITY_RETRY_JITTER = 0.2


class CloudvmError(Exception):
    """Expected failures — printed without a traceback."""


# boto3 is imported lazily so `--print-completion` is as fast as possible
_session: Any = None
_clients: dict[str | None, Any] = {}


def _get_session() -> Any:
    """Sole entry point for building the AWS SDK session."""
    global _session
    if _session is None:
        import boto3

        # boto does not respect AWS_REGION while other AWS SDKs do (https://github.com/boto/boto3/issues/3620)
        # work around this boto limitation to keep it an implementation detail
        region_name = os.environ.get("AWS_REGION") or None
        _session = boto3.Session(region_name=region_name)
    return _session


def _reset_session() -> None:
    """Drop the cached session/clients so the next call re-reads credentials.

    Used after `aws sso login` finishes — the on-disk token cache is fresh, but
    a previously-built session may have memoized the failing state.
    """
    global _session
    _session = None
    _clients.clear()


def _ec2(region: str | None = None):
    if region not in _clients:
        kwargs = {"region_name": region} if region else {}
        _clients[region] = _get_session().client("ec2", **kwargs)
    return _clients[region]


_EXPIRED_TOKEN_CODES = ("ExpiredToken", "ExpiredTokenException", "RequestExpired")


def _is_expired_credentials(exc: BaseException) -> bool:
    import botocore.exceptions as bex

    # Botocore exceptions that mean "the cached SSO token isn't usable" — `aws sso login` fixes them.
    if isinstance(exc, (bex.SSOTokenLoadError, bex.UnauthorizedSSOTokenError, bex.TokenRetrievalError)):
        return True
    if isinstance(exc, bex.ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _EXPIRED_TOKEN_CODES
    # A token expiry mid-`wait()` surfaces as ClientError most of the time, but some waiter
    # configurations bubble it up wrapped as WaiterError with the response stashed away.
    if isinstance(exc, bex.WaiterError):
        last = getattr(exc, "last_response", None) or {}
        code = last.get("Error", {}).get("Code", "") if isinstance(last, dict) else ""
        return code in _EXPIRED_TOKEN_CODES
    return False


def _raise_as_cloudvm(exc: BaseException) -> NoReturn:
    """Wrap botocore errors as CloudvmError so `main()` prints them without a traceback.

    Other exceptions are re-raised unchanged.
    """
    import botocore.exceptions as bex

    if isinstance(exc, (bex.BotoCoreError, bex.ClientError)):
        raise CloudvmError(str(exc)) from exc
    raise exc


def _sso_login() -> None:
    print("SSO token expired or missing; running `aws sso login` ...")
    # Inherit stdio so the browser-handoff messages reach the user's terminal.
    result = subprocess.run(["aws", "sso", "login"])
    if result.returncode != 0:
        raise CloudvmError(f"`aws sso login` failed with exit {result.returncode}")
    _reset_session()


def _aws_call(fn: Callable[[], Any]) -> Any:
    """Run a single AWS API call, refreshing SSO once on token errors before retrying.

    Botocore errors are converted to CloudvmError so `main()` prints them as `error: ...`
    instead of a Python traceback. Non-AWS exceptions propagate unchanged.

    Scoping the retry to one call (rather than the whole command) means a token expiry
    encountered mid-command doesn't replay prints or already-applied AWS state changes.
    `fn` should re-resolve its client (via `_ec2(region)`) so the retry picks up the
    rebuilt session after `_sso_login()` clears the cache.
    """
    try:
        return fn()
    except Exception as exc:
        if not _is_expired_credentials(exc):
            _raise_as_cloudvm(exc)
    _sso_login()
    try:
        return fn()
    except Exception as exc:
        _raise_as_cloudvm(exc)


def _hl(text: str) -> str:
    """Highlight a token in blue when stdout is a TTY; pass through otherwise."""
    if not sys.stdout.isatty():
        return text
    return f"\033[38;5;75m{text}\033[0m"


def describe_instance(name: str, region: str | None = None) -> dict:
    """Return the single instance matching tag:Name=<name> (excluding terminated/shutting-down)."""
    resp = _aws_call(
        lambda: _ec2(region).describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [name]},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ],
        )
    )
    instances = [i for r in resp.get("Reservations", []) for i in r.get("Instances", [])]
    if not instances:
        where = f"region {region}" if region else "the current AWS context"
        raise CloudvmError(f"no instance with tag Name={name!r} in {where}")
    if len(instances) > 1:
        ids = ", ".join(i["InstanceId"] for i in instances)
        raise CloudvmError(f"multiple instances match Name={name!r}: {ids}")
    return instances[0]


def ensure_running(instance: dict, region: str | None = None, retry_seconds: float = 0.0) -> bool:
    """Walk the instance from its current state to `running`, transition by transition.

    Each branch handles one state and its own wait, then advances `state` to the next
    state in the progression: stopping -> stopped -> pending -> running.
    start_instances would otherwise refuse a stopping instance with IncorrectInstanceState.

    Returns True if starting had to be retried because EC2 lacked capacity.
    """
    instance_id = instance["InstanceId"]
    state = instance["State"]["Name"]
    retried = False

    if state == "stopping":
        print(f"instance {instance_id} is {_hl('stopping')}; waiting for {_hl('stopped')} ...")
        _aws_call(lambda: _ec2(region).get_waiter("instance_stopped").wait(InstanceIds=[instance_id]))
        state = "stopped"
    if state == "stopped":
        print(f"instance {instance_id} is {_hl('stopped')}; starting ...")
        retried = start_with_capacity_retry(instance, region, retry_seconds)
        state = "pending"
    if state == "pending":
        print(f"instance {instance_id} is {_hl('pending')}; waiting for {_hl('running')} ...")
        _aws_call(lambda: _ec2(region).get_waiter("instance_running").wait(InstanceIds=[instance_id]))
        state = "running"
    if state == "running":
        return retried
    raise CloudvmError(f"instance {instance_id} is in unexpected state {state!r}")


def _aws_error_code(exc: BaseException) -> str | None:
    import botocore.exceptions as bex

    # `_aws_call` wraps botocore errors as CloudvmError; the original is kept as `__cause__`.
    cause = exc.__cause__
    if not isinstance(cause, bex.ClientError):
        return None
    return cause.response.get("Error", {}).get("Code")


def _is_insufficient_capacity(exc: BaseException) -> bool:
    return _aws_error_code(exc) == "InsufficientInstanceCapacity"


def start_with_capacity_retry(instance: dict, region: str | None, retry_seconds: float) -> bool:
    """Start a stopped instance, retrying on InsufficientInstanceCapacity for up to `retry_seconds`.

    Other errors fail immediately. botocore's own retries stay in effect for each attempt,
    so the deadline may be overshot by the duration of one attempt.
    Returns True if at least one attempt failed for lack of capacity.
    """
    instance_id = instance["InstanceId"]
    where = f"{instance.get('InstanceType', '?')} in {instance.get('Placement', {}).get('AvailabilityZone', '?')}"
    start = time.monotonic()
    deadline = start + retry_seconds
    delay = CAPACITY_RETRY_INITIAL
    attempt = 1
    while True:
        try:
            _aws_call(lambda: _ec2(region).start_instances(InstanceIds=[instance_id]))
            return attempt > 1
        except CloudvmError as exc:
            if not _is_insufficient_capacity(exc):
                raise
            now = time.monotonic()
            if now >= deadline:
                if retry_seconds <= 0:
                    raise CloudvmError(f"{exc} (pass --retry DURATION|forever to keep trying)") from exc
                waited = format_duration(now - start)
                name = next((t["Value"] for t in instance.get("Tags", []) if t["Key"] == "Name"), instance_id)
                notify(f"Giving up on {name} ☹️ AWS had no {where} for {waited}")
                raise CloudvmError(f"still no capacity for {where} after {attempt} attempts ({waited})") from exc
            jittered = delay * random.uniform(1 - CAPACITY_RETRY_JITTER, 1 + CAPACITY_RETRY_JITTER)
            sleep_for = min(jittered, CAPACITY_RETRY_MAX, deadline - now)
            print(
                f"insufficient capacity for {where}; retrying in {format_duration(sleep_for)} "
                f"(attempt {attempt}, elapsed {format_duration(now - start)})"
            )
            time.sleep(sleep_for)
            delay = min(delay * CAPACITY_RETRY_FACTOR, CAPACITY_RETRY_MAX)
            attempt += 1


_DURATION_RE = re.compile(r"(\d+)([smhd]?)")
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_retry_duration(value: str) -> float:
    """argparse type for `--retry`: `forever`, or a duration like `90`, `30s`, `15m`, `1h30m`, `2d`."""
    text = value.strip().lower()
    if text == "forever":
        return math.inf
    parts = _DURATION_RE.findall(text)
    if not text or "".join(n + u for n, u in parts) != text:
        raise argparse.ArgumentTypeError(f"invalid duration {value!r}; use e.g. 30m, 2h, 1h30m or 'forever'")
    return float(sum(int(n) * _DURATION_UNITS[u] for n, u in parts))


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def notification_sequence(message: str) -> str:
    """Terminal escape sequence for a desktop notification (OSC 9) followed by a bell.

    Terminals that support OSC 9 (Ghostty, iTerm2, WezTerm, ...) post it as their own notification,
    which also works over ssh; others ignore it. Control characters are dropped from `message` so
    text such as an instance name can't end the sequence early and inject escapes.
    """
    text = _CONTROL_CHARS_RE.sub("", message)
    return f"\x1b]9;{text}\x1b\\\a"


def notify(message: str) -> None:
    """Best-effort attention grab for unattended runs: desktop notification via the terminal, plus a bell.

    Written to /dev/tty rather than stdout so it still reaches the terminal when output is piped.
    """
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(notification_sequence(message))
    except OSError:
        pass


def wait_for_public_ip(instance_id: str, region: str | None = None) -> str:
    deadline = time.monotonic() + IP_POLL_SECONDS
    while True:
        resp = _aws_call(lambda: _ec2(region).describe_instances(InstanceIds=[instance_id]))
        ip = resp["Reservations"][0]["Instances"][0].get("PublicIpAddress")
        if ip:
            return ip
        if time.monotonic() >= deadline:
            raise CloudvmError(f"instance {instance_id} is running but no public IP after {IP_POLL_SECONDS}s")
        time.sleep(IP_POLL_INTERVAL)


# --- ssh_config editing ---

HOST_LINE_RE = re.compile(r"^(\s*)(Host)(\s+)(.+?)\s*$", re.IGNORECASE)
HOSTNAME_LINE_RE = re.compile(r"^(\s*)(HostName|Hostname)(\s+)(\S+)(.*)$", re.IGNORECASE)
COMMENTED_HOSTNAME_LINE_RE = re.compile(r"^(\s*)#\s*(HostName|Hostname)(\s+)(\S+)(.*)$", re.IGNORECASE)
MATCH_LINE_RE = re.compile(r"^\s*Match\s", re.IGNORECASE)


def _host_tokens(line: str) -> list[str] | None:
    m = HOST_LINE_RE.match(line)
    if not m:
        return None
    return m.group(4).split()


def _starts_block(line: str) -> bool:
    """True for a `Host` or `Match` line; either one ends the preceding block."""
    return _host_tokens(line) is not None or MATCH_LINE_RE.match(line) is not None


def find_host_block(lines: list[str], alias: str) -> tuple[int, int] | None:
    """Return (start_idx, end_idx_exclusive) of the Host block whose tokens contain `alias` exactly."""
    block_start = -1
    matched = False
    for i, line in enumerate(lines):
        if not _starts_block(line):
            continue
        if matched:
            return block_start, i
        block_start = i
        tokens = _host_tokens(line)
        matched = tokens is not None and alias in tokens
    if matched:
        return block_start, len(lines)
    return None


def replace_hostname_in_block(lines: list[str], start: int, end: int, ip: str) -> bool:
    """Replace the `Hostname` value inside lines[start:end]. Return True if a line was changed.

    A commented-out `# Hostname X` line — as left by `down --update-ssh` — is uncommented
    and the IP updated in one step, so `up` after `down` round-trips cleanly.
    """
    for i in range(start + 1, end):
        m = HOSTNAME_LINE_RE.match(lines[i]) or COMMENTED_HOSTNAME_LINE_RE.match(lines[i])
        if m:
            new_line = f"{m.group(1)}{m.group(2)}{m.group(3)}{ip}{m.group(5)}"
            if not new_line.endswith("\n") and lines[i].endswith("\n"):
                new_line += "\n"
            if new_line == lines[i]:
                return False
            lines[i] = new_line
            return True
    # Block exists but has no Hostname — inject one with the block's body indent.
    indent = "    "
    for i in range(start + 1, end):
        m = re.match(r"^(\s+)\S", lines[i])
        if m:
            indent = m.group(1)
            break
    insert_at = start + 1
    lines.insert(insert_at, f"{indent}Hostname {ip}\n")
    return True


def comment_out_hostname_in_block(lines: list[str], start: int, end: int) -> bool:
    """Prefix the active `Hostname` line in lines[start:end] with `# `. Return True if a line
    was changed; False if there is no active Hostname (already commented, or missing)."""
    for i in range(start + 1, end):
        m = HOSTNAME_LINE_RE.match(lines[i])
        if m:
            indent = m.group(1)
            rest = lines[i][len(indent) :]
            lines[i] = f"{indent}# {rest}"
            return True
    return False


def _similar_block_defaults(lines: list[str], alias: str) -> dict[str, str]:
    """Pick defaults (User, IdentityFile) from the existing block whose Host token shares the longest prefix with alias."""
    best_prefix = -1
    best_block: tuple[int, int] | None = None
    n = len(lines)
    i = 0
    while i < n:
        tokens = _host_tokens(lines[i])
        if tokens is None:
            i += 1
            continue
        # Find block end.
        j = i + 1
        while j < n and not _starts_block(lines[j]):
            j += 1
        for token in tokens:
            if any(ch in token for ch in "*?!"):
                continue
            shared = 0
            for a, b in zip(alias, token):
                if a != b:
                    break
                shared += 1
            if shared > best_prefix:
                best_prefix = shared
                best_block = (i, j)
        i = j
    defaults: dict[str, str] = {}
    if best_block is None or best_prefix <= 0:
        return defaults
    bs, be = best_block
    for k in range(bs + 1, be):
        m = re.match(r"^\s*(\S+)\s+(.+?)\s*$", lines[k])
        if not m:
            continue
        key = m.group(1).lower()
        if key in ("user", "identityfile") and key not in defaults:
            defaults[key] = m.group(2)
    return defaults


def prompt_new_block(alias: str, ip: str, lines: list[str]) -> str | None:
    """Return the text of the new Host block, or None if the user declines."""
    defaults = _similar_block_defaults(lines, alias)
    print(f"\nNo `Host {alias}` block found in {SSH_CONFIG}.")
    print("Propose to add a new block.\n")

    def ask(prompt: str, default: str | None) -> str:
        suffix = f" [{default}]" if default else ""
        try:
            value = input(f"  {prompt}{suffix}: ").strip()
        except EOFError:
            value = ""
        return value or (default or "")

    user = ask("User", defaults.get("user") or "ubuntu")
    identity_file = ask("IdentityFile", defaults.get("identityfile"))
    block_lines = [f"Host {alias}", f"    User {user}", f"    Hostname {ip}"]
    if identity_file:
        block_lines.append(f"    IdentityFile {identity_file}")
        block_lines.append("    IdentitiesOnly yes")
    block_text = "\n".join(block_lines) + "\n"

    print("\nWould add this block:\n")
    for ln in block_text.splitlines():
        print(f"  {ln}")
    print()
    try:
        answer = input("Add it? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        return None
    return block_text


def _commit_ssh_config(new_text: str, alias: str, expected_hostname: str | None) -> None:
    """Validate `new_text` via `ssh -G`, back up the existing config, then atomically replace it.

    If `expected_hostname` is given, additionally assert that `ssh -G` resolves to it.
    Pass None to skip that check (e.g., when commenting out a Hostname — ssh then falls back
    to the alias literal and the expected resolution depends on the surrounding config).
    """
    tmp = SSH_CONFIG.with_suffix(SSH_CONFIG.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(new_text)
    try:
        validate = subprocess.run(
            ["ssh", "-G", alias, "-F", str(tmp)],
            capture_output=True,
            text=True,
        )
        if validate.returncode != 0:
            raise CloudvmError(f"ssh rejected the new config: {validate.stderr.strip()}")
        if expected_hostname is not None:
            resolved = None
            for line in validate.stdout.splitlines():
                if line.startswith("hostname "):
                    resolved = line.split(" ", 1)[1].strip()
                    break
            if resolved != expected_hostname:
                raise CloudvmError(
                    f"validation: ssh -G resolved hostname to {resolved!r}, expected {expected_hostname!r}"
                )
        shutil.copyfile(SSH_CONFIG, SSH_CONFIG_BACKUP)
        os.replace(tmp, SSH_CONFIG)
    finally:
        if tmp.exists():
            tmp.unlink()


def update_ssh_config(alias: str, ip: str) -> None:
    if not SSH_CONFIG.exists():
        raise CloudvmError(f"{SSH_CONFIG} does not exist; refusing to create it")

    original = SSH_CONFIG.read_text()
    lines = original.splitlines(keepends=True)

    block = find_host_block(lines, alias)
    if block is None:
        new_block = prompt_new_block(alias, ip, lines)
        if new_block is None:
            print("Skipped ssh_config update.")
            return
        sep = "" if not lines or lines[-1].endswith("\n") else "\n"
        leading = "" if not lines or (lines and lines[-1].strip() == "") else "\n"
        new_text = original + sep + leading + new_block
    else:
        start, end = block
        changed = replace_hostname_in_block(lines, start, end, ip)
        if not changed:
            print(f"{SSH_CONFIG}: Hostname for `{alias}` is already {ip}; nothing to do.")
            return
        new_text = "".join(lines)

    _commit_ssh_config(new_text, alias, expected_hostname=ip)
    print(f"Updated {SSH_CONFIG}: Host {alias} -> {ip} (backup at {SSH_CONFIG_BACKUP})")


def disable_ssh_config(alias: str) -> None:
    """Comment out the matching Host block's Hostname so `ssh <alias>` no longer connects to a stale IP.

    No-op when the block doesn't exist or the Hostname is already commented/absent.
    """
    if not SSH_CONFIG.exists():
        raise CloudvmError(f"{SSH_CONFIG} does not exist; refusing to create it")

    lines = SSH_CONFIG.read_text().splitlines(keepends=True)
    block = find_host_block(lines, alias)
    if block is None:
        print(f"No `Host {alias}` block in {SSH_CONFIG}; skipped ssh_config update.")
        return
    start, end = block
    if not comment_out_hostname_in_block(lines, start, end):
        print(f"{SSH_CONFIG}: Hostname for `{alias}` is already disabled; nothing to do.")
        return
    _commit_ssh_config("".join(lines), alias, expected_hostname=None)
    print(f"Updated {SSH_CONFIG}: Host {alias} Hostname disabled (backup at {SSH_CONFIG_BACKUP})")


def remove_ssh_config(alias: str) -> None:
    """Remove `alias` from ~/.ssh/config, for an instance that no longer exists.

    The whole Host block goes when `alias` is its only token; otherwise only the token is dropped
    from the Host line, so the block's other aliases keep working. No-op when the block doesn't exist.
    """
    if not SSH_CONFIG.exists():
        raise CloudvmError(f"{SSH_CONFIG} does not exist; refusing to create it")

    lines = SSH_CONFIG.read_text().splitlines(keepends=True)
    block = find_host_block(lines, alias)
    if block is None:
        print(f"No `Host {alias}` block in {SSH_CONFIG}; skipped ssh_config update.")
        return
    start, end = block
    m = HOST_LINE_RE.match(lines[start])
    others = [token for token in m.group(4).split() if token != alias]
    if others:
        newline = "\n" if lines[start].endswith("\n") else ""
        lines[start] = f"{m.group(1)}{m.group(2)}{m.group(3)}{' '.join(others)}{newline}"
    else:
        # An unindented comment introduces the block that follows it; an indented one belongs to the
        # block before it. So unindented comments directly above the Host line go with the block,
        # and so do indented comments after its last directive.
        head = start
        while head > 0 and lines[head - 1].startswith("#"):
            head -= 1
        cut = start + 1
        for i in range(start + 1, end):
            if lines[i].strip() and not lines[i].startswith("#"):
                cut = i + 1
        while cut < end and not lines[cut].strip():
            cut += 1
        del lines[head:cut]
        if head == len(lines):
            # The block was last; drop the blank separator that preceded it.
            while lines and not lines[-1].strip():
                lines.pop()
    _commit_ssh_config("".join(lines), alias, expected_hostname=None)
    print(f"Updated {SSH_CONFIG}: Host {alias} removed (backup at {SSH_CONFIG_BACKUP})")


def validate_ssh(alias: str) -> None:
    print(f"Validating ssh to {alias} ...")
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=5",
            alias,
            "echo OK",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        stderr = result.stderr.strip()
        print(f"ssh validation returned exit {result.returncode}: {stderr}", file=sys.stderr)


# --- subcommand handlers ---


def cmd_up(args: argparse.Namespace) -> int:
    instance = describe_instance(args.name, args.region)
    retried = ensure_running(instance, args.region, args.retry)
    ip = wait_for_public_ip(instance["InstanceId"], args.region)
    print(f"{args.name} ({instance['InstanceId']}) is {_hl('running')} at {ip}")
    if args.update_ssh:
        alias = args.ssh_alias or args.name
        update_ssh_config(alias, ip)
        validate_ssh(alias)
    if retried:
        notify(f"{args.name} is running at {ip}")
    return 0


def terminates_on_shutdown(instance_id: str, region: str | None = None) -> bool:
    """True if the instance's shutdown behavior is `terminate`.

    EC2 applies that behavior only to an OS-initiated shutdown; an API stop leaves the instance stopped.
    """
    resp = _aws_call(
        lambda: _ec2(region).describe_instance_attribute(
            InstanceId=instance_id, Attribute="instanceInitiatedShutdownBehavior"
        )
    )
    return resp["InstanceInitiatedShutdownBehavior"]["Value"] == "terminate"


def cmd_down(args: argparse.Namespace) -> int:
    instance = describe_instance(args.name, args.region)
    instance_id = instance["InstanceId"]
    state = instance["State"]["Name"]
    terminated = False
    if state == "stopped":
        print(f"{args.name} ({instance_id}) is already {_hl('stopped')}")
    elif state == "stopping":
        print(f"{args.name} ({instance_id}) is already {_hl('stopping')}")
    elif state in ("running", "pending"):
        if terminates_on_shutdown(instance_id, args.region):
            # An API stop would leave the instance stopped, so apply the shutdown behavior explicitly.
            try:
                _aws_call(lambda: _ec2(args.region).terminate_instances(InstanceIds=[instance_id]))
                terminated = True
            except CloudvmError as exc:
                # Termination protection (DisableApiTermination) refuses the terminate; a stop still works.
                if _aws_error_code(exc) != "OperationNotPermitted":
                    raise
                print(f"{args.name} ({instance_id}) cannot be terminated ({exc}); stopping it instead")
        if terminated:
            print(f"{args.name} ({instance_id}) has shutdown behavior {_hl('terminate')}; it is {_hl('shutting-down')}")
        else:
            _aws_call(lambda: _ec2(args.region).stop_instances(InstanceIds=[instance_id]))
            print(f"{args.name} ({instance_id}) is {_hl('stopping')}")
    else:
        raise CloudvmError(f"instance {instance_id} is in unexpected state {state!r}")
    if args.update_ssh:
        alias = args.ssh_alias or args.name
        if terminated:
            remove_ssh_config(alias)
        else:
            disable_ssh_config(alias)
    return 0


def _split_csv(values: list[str]) -> list[str]:
    """Flatten a list of possibly comma-separated values into a list of trimmed tokens."""
    out: list[str] = []
    for v in values:
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out


def match_globs(values: list[str], patterns: list[str]) -> list[str]:
    """Return values matching any of the fnmatch patterns, in input order, deduplicated."""
    seen: set[str] = set()
    matched: list[str] = []
    for v in values:
        if v in seen:
            continue
        if any(fnmatch.fnmatchcase(v, p) for p in patterns):
            seen.add(v)
            matched.append(v)
    return matched


def _has_wildcard(pattern: str) -> bool:
    """True if `pattern` contains any fnmatch metacharacter."""
    return any(c in pattern for c in "*?[")


def _resolve_regions(region_args: list[str] | None) -> list[str]:
    """Resolve the `--region` CLI argument(s) to the concrete list of regions to query.

    `region_args` is the raw argparse value (list because `--region` is repeatable, or
    None when not passed). Comma-separated tokens are split. When nothing is supplied,
    fall back to the session's effective region.

    When any pattern has wildcards, enumerate live regions via `describe_regions` and
    glob-match. Otherwise patterns are literal — return them directly, deduplicated,
    preserving order (skips the API call).
    """
    patterns = _split_csv(region_args) if region_args else []
    if not patterns:
        return [_effective_region()]
    if any(_has_wildcard(p) for p in patterns):
        all_regions = list_aws_regions()
        matched = match_globs(all_regions, patterns)
        if not matched:
            raise CloudvmError(f"no AWS regions match {patterns!r} (available: {len(all_regions)} regions)")
        return matched
    return list(set(patterns))


def list_aws_regions() -> list[str]:
    # `describe-regions` is itself a regional API and needs a region for endpoint resolution.
    # Prefer the session's configured region; us-east-1 is a safe fallback since the response
    # is global metadata.
    region = _get_session().region_name or "us-east-1"
    resp = _aws_call(lambda: _ec2(region).describe_regions())
    return sorted(r["RegionName"] for r in resp["Regions"])


def _configured_region() -> str | None:
    """Return the region boto3 resolved for the current env/profile, or None if unset."""
    return _get_session().region_name


def _effective_region() -> str:
    """Return the AWS region in use."""
    region = _configured_region()
    if region is None:
        raise CloudvmError("no AWS region configured (set AWS_REGION, AWS_DEFAULT_REGION, or `aws configure`)")
    return region


def format_table(
    headers: list[str], rows: list[list[str]], col_wrappers: list[Callable[[str], str] | None] | None = None
) -> str:
    """Format a left-aligned text table. Width is computed from raw cell strings;
    if `col_wrappers[i]` is given, it wraps the already-padded data cell in column i
    (e.g., to add ANSI color codes). Wrappers are not applied to headers."""
    cols = [headers] + [list(map(str, r)) for r in rows]
    widths = [max(len(row[i]) for row in cols) for i in range(len(headers))]
    sep = "  "
    wrappers = col_wrappers or [None] * len(headers)

    def fmt_data_row(row: list[str]) -> str:
        cells = []
        for i, c in enumerate(row):
            padded = str(c).ljust(widths[i])
            w = wrappers[i] if i < len(wrappers) else None
            cells.append(w(padded) if w else padded)
        return sep.join(cells)

    return "\n".join(
        [
            sep.join(h.ljust(w) for h, w in zip(headers, widths)),
            sep.join("-" * w for w in widths),
            *(fmt_data_row(r) for r in rows),
        ]
    )


def _find_instances_in_region(name_patterns: list[str], region: str) -> list[dict]:
    """Return instances matching any of the Name-tag patterns. Each dict has:
    region, instance_id, name, state, public_ip (defaulting to "-").
    Terminated and shutting-down instances are excluded.
    """
    resp = _aws_call(
        lambda: _ec2(region).describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": name_patterns},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ],
        )
    )
    instances: list[dict] = []
    for r in resp.get("Reservations", []):
        for inst in r.get("Instances", []):
            name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "")
            instances.append(
                {
                    "region": region,
                    "instance_id": inst["InstanceId"],
                    "name": name,
                    "state": inst["State"]["Name"],
                    "public_ip": inst.get("PublicIpAddress") or "-",
                }
            )
    return instances


def _list_in_region(name_patterns: list[str], region: str) -> list[list[str]]:
    return [
        [i["region"], i["name"], i["state"], i["public_ip"]] for i in _find_instances_in_region(name_patterns, region)
    ]


def cmd_delete(args: argparse.Namespace) -> int:
    regions = _resolve_regions(args.region)

    with ThreadPoolExecutor(max_workers=len(regions)) as ex:
        instances = [
            inst for chunk in ex.map(lambda r: _find_instances_in_region([args.name], r), regions) for inst in chunk
        ]

    if not instances:
        print("(no instances match)")
        return 0

    instances.sort(key=lambda i: (i["region"], i["name"]))
    rows = [[i["region"], i["name"], i["instance_id"], i["state"], i["public_ip"]] for i in instances]
    print(
        format_table(
            ["region", "name", "instance id", "status", "public IP"],
            rows,
            col_wrappers=[None, None, None, _hl, None],
        )
    )

    if not args.force:
        try:
            answer = input(f"\nDelete {len(instances)} instance(s)? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("aborted")
            return 1

    by_region: dict[str, list[dict]] = {}
    for inst in instances:
        by_region.setdefault(inst["region"], []).append(inst)

    for region, region_instances in by_region.items():
        ids = [i["instance_id"] for i in region_instances]
        _aws_call(lambda: _ec2(region).terminate_instances(InstanceIds=ids))
        for inst in region_instances:
            print(f"{inst['name']} ({inst['instance_id']}) is {_hl('shutting-down')}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    # AWS filter natively supports * and ? wildcards, and ORs multiple Values.
    name_patterns = _split_csv(args.name) if args.name else ["*"]
    regions = _resolve_regions(args.region)

    with ThreadPoolExecutor(max_workers=len(regions)) as ex:
        rows = [row for chunk in ex.map(lambda r: _list_in_region(name_patterns, r), regions) for row in chunk]

    if not rows:
        print("(no instances match)")
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    print(format_table(["region", "name", "status", "public IP"], rows, col_wrappers=[None, None, _hl, None]))
    return 0


def _installed_version() -> str:
    try:
        return version("cloudvm")
    except PackageNotFoundError:
        return "(unknown)"


def _version_string() -> str:
    version_string = _installed_version()
    commit = _build_info.COMMIT
    date = _build_info.DATE
    return f"cloudvm {version_string} ({commit} {date})"


class _PrintCompletionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(argcomplete.shellcode(["cloudvm"], shell=values or "bash"))
        parser.exit()


def _complete_region(prefix, **kwargs):
    # Runs inside the shell completion hook; never raise — surface failures as empty suggestions.
    try:
        return list_aws_regions()
    except Exception:
        return []


def _complete_instance_name(prefix, parsed_args=None, **kwargs):
    try:
        region = getattr(parsed_args, "region", None) or _configured_region()
        if not region:
            return []
        rows = _list_in_region(["*"], region)
        return sorted({row[1] for row in rows if row[1]})
    except Exception:
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudvm",
        description="Manage cloud development instances. Currently AWS EC2 is supported.",
    )
    parser.add_argument("--version", action="version", version=_version_string())
    parser.add_argument(
        "--print-completion",
        action=_PrintCompletionAction,
        nargs="?",
        const="bash",
        choices=("bash", "zsh", "tcsh", "fish"),
        metavar="SHELL",
        help="print shell completion code and exit (default: bash)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="ensure SSO + start instance + print public IP")
    up.add_argument(
        "--name", "-n", required=True, help="EC2 Name tag of the instance"
    ).completer = _complete_instance_name
    up.add_argument(
        "--region", "-r", default=None, help="AWS region; omit to let aws-cli use its own default"
    ).completer = _complete_region
    up.add_argument(
        "--update-ssh", action="store_true", help="update the matching Host block's Hostname in ~/.ssh/config"
    )
    up.add_argument("--ssh-alias", help="ssh_config Host alias to update (defaults to --name)")
    up.add_argument(
        "--retry",
        type=parse_retry_duration,
        default=0.0,
        metavar="DURATION",
        help="keep retrying the start while EC2 reports insufficient capacity, for up to DURATION "
        "(e.g. 30m, 2h, 1h30m) or 'forever'; notifies when the instance finally starts",
    )
    up.set_defaults(func=cmd_up)

    down = sub.add_parser(
        "down",
        help="trigger stop on a running instance (does not wait for fully stopped); "
        "terminates it instead if its shutdown behavior is terminate",
    )
    down.add_argument(
        "--name", "-n", required=True, help="EC2 Name tag of the instance"
    ).completer = _complete_instance_name
    down.add_argument(
        "--region", "-r", default=None, help="AWS region; omit to let aws-cli use its own default"
    ).completer = _complete_region
    down.add_argument(
        "--update-ssh",
        action="store_true",
        help="comment out the matching Host block's Hostname in ~/.ssh/config; "
        "remove the alias instead if the instance is terminated",
    )
    down.add_argument("--ssh-alias", help="ssh_config Host alias to update (defaults to --name)")
    down.set_defaults(func=cmd_down)

    delete = sub.add_parser("delete", help="terminate matching instance(s); asks for confirmation unless --force")
    delete.add_argument(
        "--name", "-n", required=True, help="EC2 Name-tag glob; AWS-native wildcards '*' and '?'"
    ).completer = _complete_instance_name
    delete.add_argument(
        "--region",
        "-r",
        action="append",
        default=None,
        help="region glob(s); repeatable and/or comma-separated. Defaults to AWS_REGION / AWS_DEFAULT_REGION.",
    ).completer = _complete_region
    delete.add_argument("-f", "--force", action="store_true", help="skip the confirmation prompt")
    delete.set_defaults(func=cmd_delete)

    lst = sub.add_parser("list", help="list instances across regions matching name and region globs")
    lst.add_argument(
        "--region",
        "-r",
        action="append",
        default=None,
        help="region glob(s); repeatable and/or comma-separated (e.g. 'eu-central-*,us-*'). "
        "Defaults to AWS_REGION / AWS_DEFAULT_REGION.",
    ).completer = _complete_region
    lst.add_argument(
        "--name",
        "-n",
        action="append",
        default=None,
        help="Name-tag glob(s); repeatable and/or comma-separated; AWS-native wildcards '*' and '?' (default: '*')",
    )
    lst.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    # Progress (e.g. instance state transitions) must show up live even when piped to tee or a log file.
    sys.stdout.reconfigure(line_buffering=True)
    try:
        return args.func(args)
    except CloudvmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
