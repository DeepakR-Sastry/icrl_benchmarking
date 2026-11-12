"""Upload Liu's runs (any of his ICRL-loop methods) to wandb from their
Slurm logs, to chart them next to ours. His code doesn't log to wandb on
this path; it prints one table per round (his SB3 logger's text output) to
stdout.

    python scripts/upload_liu_logs.py logs/liu_me_2301234_*.out \
        --project icrl-hc-icrl --config liu_maximum_entropy

--config names OUR config for the same setup (configs/hc/<name>.yaml); its
method is recorded. Group liu__<name>, run liu__<name>__seed<k> (k from the
file name ..._<k>.out, the Slurm array index = his -s seed), as our
run_icrl.py names ours ours__<name> (docs/decisions.md 2026-11-12). Fields:
code liu, config_file, method, git_commit (his commit), source_log,
his_config (the "config: ..." line his job script prints first). One point
per round, step = run_iter, as ours (step = round).

Limits of his text output: values are rounded to 3 significant figures, and
names longer than the column are cut ("unweighted_nomina..."): those are
skipped. Non-numbers (backward/data_shape "[10000, 24]") are skipped too.
"""
import argparse
import re
from pathlib import Path

import yaml

HIS_COMMIT = "0d79d2c"  # ~/refs/liu_icrl, the clone his runs use (Vulcan too)

# His name -> ours, where both mean the same thing (docs/methods.md §4).
# Everything else keeps his name: backward/* already equals ours (vicrl.py
# copies it), forward/* and rollout/* are Malik's SB3 names, which our runs
# use on wandb too (scripts/run_icrl.py MALIK_ICRL_NAMES).
OUR_NAMES = {
    "run_iter": "iteration",
    "true/mean_reward": "true/reward",
    "true/std_reward": "true/reward_std",
    "true/mean_nc_reward": "true/feasible_reward",
    "true/std_nc_reward": "true/feasible_reward_std",
}

# A table line: "| name | value |" (the value may be empty, for a heading).
TABLE_LINE = re.compile(r"^\|(\s*)(\S+)\s*\|\s*(.*?)\s*\|$")


def parse_tables(text):
    """Each table in his log -> one dict {full name: number}, in order.
    Headings ("| backward/ |  |") prefix the indented names below them;
    unindented names (run_iter, timesteps) stand alone. Only tables with
    run_iter (one per round) are kept."""
    rows, row, heading = [], None, ""
    for line in text.splitlines():
        if line.startswith("-----"):  # a table's top or bottom border
            if row:
                rows.append(row)
            row, heading = {}, ""
            continue
        match = TABLE_LINE.match(line)
        if row is None or not match:
            continue
        indent, name, value = match.groups()
        if name.endswith("/") and value == "":
            heading = name
            continue
        # Names under a heading are indented ("|    cn_loss"); top-level ones
        # have one space ("| run_iter").
        full = heading + name if len(indent) > 1 else name
        if full.endswith("..."):
            continue
        try:
            row[full] = float(value)
        except ValueError:
            continue
    return [r for r in rows if "run_iter" in r]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", required=True, help="our config for this setup, configs/hc/<name>.yaml")
    parser.add_argument("--entity", default="irl-chip")
    args = parser.parse_args()

    config_file = Path("configs/hc") / f"{args.config}.yaml"
    method = yaml.safe_load(config_file.read_text())["method"]["name"]  # refuses a missing file
    group = f"liu__{args.config}"

    import wandb

    for path in map(Path, args.logs):
        seed = int(path.stem.rsplit("_", 1)[1])
        text = path.read_text(errors="replace")
        rows = parse_tables(text)
        his_config = re.search(r"^config: (\S+)", text, re.MULTILINE)
        run = wandb.init(entity=args.entity, project=args.project,
                         group=group, name=f"{group}__seed{seed}",
                         config={"seed": seed, "source_log": path.name, "code": "liu",
                                 "config_file": str(config_file), "method": method,
                                 "git_commit": HIS_COMMIT,
                                 "his_config": his_config.group(1) if his_config else "unknown"})
        for row in rows:
            renamed = {OUR_NAMES.get(k, k): v for k, v in row.items()}
            wandb.log(renamed, step=int(row["run_iter"]))
        run.finish()
        print(f"{path.name}: seed {seed}, {len(rows)} rounds")


if __name__ == "__main__":
    main()
