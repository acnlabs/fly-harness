"""Generate in-repo connectome loader fixtures (run once, commit output)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

# FlyWire-style integer ids for a toy 8-neuron touch reflex subgraph.
NEURON_IDS = np.array(
    [581001, 581002, 581101, 581102, 581201, 581301, 581302, 581303],
    dtype=np.int64,
)
SUBSET_IDS = NEURON_IDS[:6]  # drop two motor neurons to exercise subset loading

PRE = np.array(
    [581001, 581001, 581002, 581002, 581101, 581102, 581201, 581201],
    dtype=np.int64,
)
POST = np.array(
    [581101, 581301, 581102, 581302, 581301, 581302, 581301, 581302],
    dtype=np.int64,
)
WEIGHT = np.array([0.9, 0.85, 0.9, 0.85, 0.7, 0.7, 0.25, 0.25], dtype=np.float64)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    np.savez(
        FIXTURES / "mini_circuit.npz",
        pre=PRE,
        post=POST,
        weight=WEIGHT,
        neuron_ids=SUBSET_IDS,
    )

    csv_path = FIXTURES / "mini_circuit.csv"
    header = "pre_id,post_id,weight\n"
    rows = "\n".join(f"{p},{q},{w}" for p, q, w in zip(PRE, POST, WEIGHT, strict=True))
    csv_path.write_text(header + rows + "\n", encoding="utf-8")

    ids_path = FIXTURES / "mini_circuit_neuron_ids.txt"
    ids_path.write_text(
        "\n".join(str(nid) for nid in SUBSET_IDS) + "\n",
        encoding="utf-8",
    )

    print(f"wrote fixtures under {FIXTURES}")


if __name__ == "__main__":
    main()
