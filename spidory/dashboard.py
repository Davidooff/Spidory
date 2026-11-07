"""Write the comparison dashboard's tables from the model design and finished results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import ModelConfig
from .flops import tflops_by_k
from .model import Spidory

MODELS = {"Plain": False, "Constant": True, "Variable": True}
RESULT_FIELDS = ["seed", "model", "setting", "mean_loss", "story", "maths", "code", "qa",
                 "maths_accuracy", "tflops_per_token", "mean_runs"]


def cost_rows() -> list[dict]:
    rows = []
    for name, experts in MODELS.items():
        cfg = ModelConfig(use_experts=experts)
        params = Spidory(cfg).num_params()
        for k, tflops in tflops_by_k(cfg).items():
            rows.append({"model": name, "runs": k, "params": params,
                         "tflops_per_token": f"{tflops:.6e}", "mflops_per_token": round(tflops * 1e6, 3)})
    return rows


def result_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    report, rows = json.loads(path.read_text()), []
    for entry in report["seeds"]:
        for name, run in zip(MODELS, entry["runs"]):
            grids = [(f"k={k}", v, int(k)) for k, v in run["by_k"].items()]
            grids += [(f"mean={m}", v, v["mean_runs"]) for m, v in run["by_mean"].items()]
            if "learned" in run:
                grids.append(("learned", run["learned"], run["learned"]["mean_runs"]))
            for setting, value, mean_runs in grids:
                rows.append({"seed": entry["seed"], "model": name, "setting": setting,
                             "mean_loss": value["mean_loss"], **value["loss"],
                             "maths_accuracy": value["maths_accuracy"],
                             "tflops_per_token": f"{value['tflops_per_token']:.6e}",
                             "mean_runs": mean_runs})
    return rows


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as sink:
        writer = csv.DictWriter(sink, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = Path("runs/dashboard")
    out.mkdir(parents=True, exist_ok=True)
    cost = cost_rows()
    write(out / "cost.csv", cost, list(cost[0]))
    write(out / "results.csv", result_rows(Path("runs/remote_big/results.json")), RESULT_FIELDS)


if __name__ == "__main__":
    main()
