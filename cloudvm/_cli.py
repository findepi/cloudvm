#!/usr/bin/env python3
"""Manage cloud development instances. Currently, AWS EC2 is supported."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cloudvm import _build_info


SSH_CONFIG = Path.home() / ".ssh" / "config"
SSH_CONFIG_BACKUP = Path.home() / ".ssh" / "config.bak"

# How long to wait for a public IP after the instance reports running.
IP_POLL_SECONDS = 60
IP_POLL_INTERVAL = 1.0


class CloudvmError(Exception):
    """Expected failures — printed without a traceback."""


def run_aws(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["aws", *args]
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise CloudvmError(f"`{' '.join(cmd)}` failed: {stderr or 'exit ' + str(result.returncode)}")
    return result


def _region_args(region: str | None) -> list[str]:
    """Render --region for aws-cli when explicit; empty list otherwise (let aws-cli pick)."""
    return ["--region", region] if region else []


def _hl(text: str) -> str:
    """Highlight a token in blue when stdout is a TTY; pass through otherwise."""
    if not sys.stdout.isatty():
        return text
    return f"\033[38;5;75m{text}\033[0m"


def ensure_sso() -> None:
    """Refresh SSO only when the cached token is actually expired."""
    probe = run_aws(["sts", "get-caller-identity", "--output", "text"], check=False)
    if probe.returncode == 0:
        return
    print("SSO token expired or missing; running `aws sso login` ...")
    # No capture: let the browser-handoff messages reach the user's terminal.
    run_aws(["sso", "login"], capture=False)


def describe_instance(name: str, region: str | None = None) -> dict:
    """Return the single instance matching tag:Name=<name> (excluding terminated/shutting-down)."""
    result = run_aws([
        "ec2", "describe-instances",
        *_region_args(region),
        "--filters",
        f"Name=tag:Name,Values={name}",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--output", "json",
    ])
    data = json.loads(result.stdout)
    instances = [i for r in data.get("Reservations", []) for i in r.get("Instances", [])]
    if not instances:
        where = f"region {region}" if region else "the current AWS context"
        raise CloudvmError(f"no instance with tag Name={name!r} in {where}")
    if len(instances) > 1:
        ids = ", ".join(i["InstanceId"] for i in instances)
        raise CloudvmError(f"multiple instances match Name={name!r}: {ids}")
    return instances[0]


def ensure_running(instance: dict, region: str | None = None) -> None:
    """Walk the instance from its current state to `running`, transition by transition.

    Each branch handles one state and its own wait, then advances `state` to the next
    state in the progression: stopping -> stopped -> pending -> running.
    start-instances would otherwise refuse a stopping instance with IncorrectInstanceState.
    """
    instance_id = instance["InstanceId"]
    state = instance["State"]["Name"]
    region_args = _region_args(region)

    if state == "stopping":
        print(f"instance {instance_id} is {_hl('stopping')}; waiting for {_hl('stopped')} ...")
        run_aws(["ec2", "wait", "instance-stopped", *region_args, "--instance-ids", instance_id])
        state = "stopped"
    if state == "stopped":
        print(f"instance {instance_id} is {_hl('stopped')}; starting ...")
        run_aws(["ec2", "start-instances", *region_args, "--instance-ids", instance_id])
        state = "pending"
    if state == "pending":
        print(f"instance {instance_id} is {_hl('pending')}; waiting for {_hl('running')} ...")
        run_aws(["ec2", "wait", "instance-running", *region_args, "--instance-ids", instance_id])
        state = "running"
    if state == "running":
        return
    raise CloudvmError(f"instance {instance_id} is in unexpected state {state!r}")


def wait_for_public_ip(instance_id: str, region: str | None = None) -> str:
    deadline = time.monotonic() + IP_POLL_SECONDS
    while True:
        result = run_aws([
            "ec2", "describe-instances",
            *_region_args(region),
            "--instance-ids", instance_id,
            "--query", "Reservations[0].Instances[0].PublicIpAddress",
            "--output", "text",
        ])
        ip = result.stdout.strip()
        if ip and ip != "None":
            return ip
        if time.monotonic() >= deadline:
            raise CloudvmError(f"instance {instance_id} is running but no public IP after {IP_POLL_SECONDS}s")
        time.sleep(IP_POLL_INTERVAL)


# --- ssh_config editing ---

HOST_LINE_RE = re.compile(r"^(\s*)(Host)(\s+)(.+?)\s*$", re.IGNORECASE)
HOSTNAME_LINE_RE = re.compile(r"^(\s*)(HostName|Hostname)(\s+)(\S+)(.*)$", re.IGNORECASE)


def _host_tokens(line: str) -> list[str] | None:
    m = HOST_LINE_RE.match(line)
    if not m:
        return None
    return m.group(4).split()


def find_host_block(lines: list[str], alias: str) -> tuple[int, int] | None:
    """Return (start_idx, end_idx_exclusive) of the Host block whose tokens contain `alias` exactly."""
    block_start = -1
    matched = False
    for i, line in enumerate(lines):
        tokens = _host_tokens(line)
        if tokens is None:
            continue
        if matched:
            return block_start, i
        block_start = i
        matched = alias in tokens
    if matched:
        return block_start, len(lines)
    return None


def replace_hostname_in_block(lines: list[str], start: int, end: int, ip: str) -> bool:
    """Replace the `Hostname` value inside lines[start:end]. Return True if a line was changed."""
    for i in range(start + 1, end):
        m = HOSTNAME_LINE_RE.match(lines[i])
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
        while j < n and _host_tokens(lines[j]) is None:
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
        resolved = None
        for line in validate.stdout.splitlines():
            if line.startswith("hostname "):
                resolved = line.split(" ", 1)[1].strip()
                break
        if resolved != ip:
            raise CloudvmError(
                f"validation: ssh -G resolved hostname to {resolved!r}, expected {ip!r}"
            )
        shutil.copyfile(SSH_CONFIG, SSH_CONFIG_BACKUP)
        os.replace(tmp, SSH_CONFIG)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Updated {SSH_CONFIG}: Host {alias} -> {ip} (backup at {SSH_CONFIG_BACKUP})")


def validate_ssh(alias: str) -> None:
    print(f"Validating ssh to {alias} ...")
    result = subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
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
    ensure_sso()
    instance = describe_instance(args.name, args.region)
    ensure_running(instance, args.region)
    ip = wait_for_public_ip(instance["InstanceId"], args.region)
    print(f"{args.name} ({instance['InstanceId']}) is {_hl('running')} at {ip}")
    if args.update_ssh:
        alias = args.ssh_alias or args.name
        update_ssh_config(alias, ip)
        validate_ssh(alias)
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    ensure_sso()
    instance = describe_instance(args.name, args.region)
    instance_id = instance["InstanceId"]
    state = instance["State"]["Name"]
    if state == "stopped":
        print(f"{args.name} ({instance_id}) is already {_hl('stopped')}")
        return 0
    if state == "stopping":
        print(f"{args.name} ({instance_id}) is already {_hl('stopping')}")
        return 0
    if state in ("running", "pending"):
        run_aws(["ec2", "stop-instances", *_region_args(args.region), "--instance-ids", instance_id])
        print(f"{args.name} ({instance_id}) is {_hl('stopping')}")
        return 0
    raise CloudvmError(f"instance {instance_id} is in unexpected state {state!r}")


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


def list_aws_regions() -> list[str]:
    # `describe-regions` is itself a regional API and needs --region. Prefer the user's
    # configured region (likely nearest/cheapest); fall back to us-east-1 when none is set.
    region = _configured_region() or "us-east-1"
    result = run_aws([
        "ec2", "describe-regions",
        "--region", region,
        "--query", "Regions[].RegionName",
        "--output", "json",
    ])
    return sorted(json.loads(result.stdout))


# AWS region names: e.g. us-east-1, eu-central-1, ap-northeast-2, us-gov-east-1, cn-north-1.
_REGION_NAME_RE = re.compile(r"^[a-z]{2,}(?:-[a-z]+)+-\d+$")


def _configured_region() -> str | None:
    """Return the region AWS CLI has resolved for the current env/profile, or None if unset.

    Parses `aws configure list` because, unlike `aws configure get region`, it reflects env
    vars (AWS_REGION, AWS_DEFAULT_REGION) in addition to the config file. Fails loudly if
    the output format changes (column separator or value shape), rather than silently
    misreading it.
    """
    # TODO should we switch to boto3 and make it sane?
    result = run_aws(["configure", "list"])
    for line in result.stdout.splitlines():
        name, sep, rest = line.partition(":")
        if name.strip() != "region":
            continue
        if not sep:
            raise CloudvmError(
                f"unexpected `aws configure list` row (no ':' separator): {line!r}"
            )
        value = rest.partition(":")[0].strip()
        if value == "<not set>":
            return None
        if not _REGION_NAME_RE.match(value):
            raise CloudvmError(
                f"unexpected region value from `aws configure list`: {value!r}"
            )
        return value
    raise CloudvmError("could not find 'region' row in `aws configure list` output")


def _effective_region() -> str:
    """Return the AWS region in use."""
    region = _configured_region()
    if region is None:
        raise CloudvmError(
            "no AWS region configured (set AWS_REGION, AWS_DEFAULT_REGION, or `aws configure`)"
        )
    return region


def format_table(headers: list[str], rows: list[list[str]],
                 col_wrappers: list[Callable[[str], str] | None] | None = None) -> str:
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

    return "\n".join([
        sep.join(h.ljust(w) for h, w in zip(headers, widths)),
        sep.join("-" * w for w in widths),
        *(fmt_data_row(r) for r in rows),
    ])


def _list_in_region(name_pattern: str, region: str) -> list[list[str]]:
    result = run_aws([
        "ec2", "describe-instances",
        "--region", region,
        "--filters",
        f"Name=tag:Name,Values={name_pattern}",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--output", "json",
    ])
    data = json.loads(result.stdout)
    rows: list[list[str]] = []
    for r in data.get("Reservations", []):
        for inst in r.get("Instances", []):
            name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "")
            state = inst["State"]["Name"]
            ip = inst.get("PublicIpAddress") or "-"
            rows.append([region, name, state, ip])
    return rows


def cmd_list(args: argparse.Namespace) -> int:
    ensure_sso()
    name_pattern = args.name  # AWS filter natively supports * and ? wildcards
    rows: list[list[str]] = []

    if args.region:
        region_patterns = _split_csv(args.region)
        all_regions = list_aws_regions()
        regions = match_globs(all_regions, region_patterns)
        if not regions:
            raise CloudvmError(
                f"no AWS regions match {region_patterns!r} (available: {len(all_regions)} regions)"
            )
        for region in regions:
            rows.extend(_list_in_region(name_pattern, region))
    else:
        rows.extend(_list_in_region(name_pattern, _effective_region()))

    if not rows:
        print("(no instances match)")
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    print(format_table(["region", "name", "status", "public IP"], rows,
                       col_wrappers=[None, None, _hl, None]))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudvm",
        description="Manage cloud development instances. Currently AWS EC2 is supported.",
    )
    parser.add_argument("--version", action="version", version=_version_string())
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="ensure SSO + start instance + print public IP")
    up.add_argument("--name", "-n", required=True, help="EC2 Name tag of the instance")
    up.add_argument("--region", "-r", default=None,
                    help="AWS region; omit to let aws-cli use its own default")
    up.add_argument("--update-ssh", action="store_true",
                    help="update the matching Host block's Hostname in ~/.ssh/config")
    up.add_argument("--ssh-alias",
                    help="ssh_config Host alias to update (defaults to --name)")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="trigger stop on a running instance (does not wait for fully stopped)")
    down.add_argument("--name", "-n", required=True, help="EC2 Name tag of the instance")
    down.add_argument("--region", "-r", default=None,
                     help="AWS region; omit to let aws-cli use its own default")
    down.set_defaults(func=cmd_down)

    lst = sub.add_parser("list", help="list instances across regions matching name and region globs")
    lst.add_argument("--region", "-r", action="append", default=None,
                     help="region glob(s); repeatable and/or comma-separated (e.g. 'eu-central-*,us-*'). "
                          "Defaults to AWS_REGION / AWS_DEFAULT_REGION.")
    lst.add_argument("--name", "-n", default="*",
                     help="Name-tag glob; AWS-native wildcards '*' and '?' (default: '*')")
    lst.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CloudvmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
