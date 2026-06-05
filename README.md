# cloudvm

A command-line tool for managing cloud development instances. AWS EC2 is currently supported.

`cloudvm` streamlines the daily routine of working with a development EC2
instance: it refreshes the AWS SSO token only when expired, starts the
instance, waits for its public IP to be assigned, and reports it. A `list`
subcommand surveys instances across regions to see what is still running.

## Install

```bash
pipx install cloudvm
```

Or, if `pipx` is not available:

```bash
pip install cloudvm
```

Requires Python 3.9 or newer and a configured `aws` CLI (SSO or static
credentials).

## Usage

```bash
# Start an instance by its Name tag
cloudvm up --name my-dev-box

# Stop an instance — returns once the stop has been triggered
cloudvm down --name my-dev-box

# List instances across regions
cloudvm list --region 'eu-central-*,us-*' --name 'my-*'
```

All subcommands accept `--region` / `-r` and `--name` / `-n`, and honor the
usual AWS environment variables (`AWS_REGION`, `AWS_PROFILE`, ...).

Pass `--update-ssh` to `up` to point the matching `~/.ssh/config` host
alias at the new IP.

## Shell completion

To enable tab-completion, add this to your `~/.bashrc` (or `~/.zshrc`,
after `compinit`):

```bash
eval "$(cloudvm --print-completion bash)"   # or: zsh / tcsh / fish
```

Then open a new shell, or `source` the rc file. `cloudvm <TAB>` will now
complete subcommands and flags.

## License

Apache License 2.0
