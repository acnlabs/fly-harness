"""Load sparse connectome edge lists into BrainState without dense matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from fly_harness.brain_state import BrainState

ConnectomeFormat = Literal["auto", "npz", "csv"]


@dataclass(frozen=True)
class ConnectomeLoadResult:
    """Sparse connectome mapped to local BrainState indices."""

    state: BrainState
    neuron_ids: np.ndarray
    id_to_index: dict[int, int]

    @property
    def n_neurons(self) -> int:
        return int(self.neuron_ids.shape[0])


def load_connectome(
    path: str | Path,
    *,
    neuron_ids: Sequence[int] | None = None,
    neuron_ids_file: str | Path | None = None,
    format: ConnectomeFormat = "auto",
    initial_potentials: float | np.ndarray = 0.0,
    neuron_labels: tuple[str, ...] | None = None,
) -> ConnectomeLoadResult:
    """Load a sparse connectome from a local NPZ or CSV edge list.

    Edge files store presynaptic id, postsynaptic id, and weight. Neuron ids may
    be arbitrary integers (e.g. FlyWire root ids); they are remapped to contiguous
    local indices ``0 .. n-1`` for the requested circuit subset.

    Parameters
    ----------
    path:
        ``.npz`` with arrays ``pre``, ``post``, ``weight`` (and optional
        ``neuron_ids``), or a CSV edge list with columns ``pre_id,post_id,weight``.
    neuron_ids:
        Explicit circuit subset. Only synapses whose pre and post are both in this
        set are kept. When omitted, ids are taken from ``neuron_ids_file``, an
        optional ``neuron_ids`` array inside the NPZ, or the unique ids appearing
        in the edge list (bounded by the file contents).
    neuron_ids_file:
        Text file with one integer neuron id per line, or ``.npy`` / ``.npz``
        containing a 1-D ``neuron_ids`` array.
    format:
        ``"auto"`` infers from the file extension; otherwise force ``"npz"`` or
        ``"csv"``.
    initial_potentials:
        Scalar or length-``n`` vector of starting membrane potentials.
    neuron_labels:
        Optional human-readable labels aligned with local indices.

    Returns
    -------
    ConnectomeLoadResult
        ``state`` is a ``BrainState`` with CSR weights of shape ``(n, n)`` where
        ``n`` is the subset size — never a full-brain dense matrix.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"connectome file not found: {path}")

    resolved_format = _resolve_format(path, format)
    pre, post, weight, file_neuron_ids = _read_edges(path, resolved_format)

    subset_ids = _resolve_neuron_subset(
        pre,
        post,
        neuron_ids=neuron_ids,
        neuron_ids_file=neuron_ids_file,
        file_neuron_ids=file_neuron_ids,
    )
    id_to_index = {int(nid): idx for idx, nid in enumerate(subset_ids)}

    local_pre, local_post, local_weight = _filter_and_remap(
        pre, post, weight, id_to_index
    )

    n = len(subset_ids)
    weights = sparse.coo_matrix(
        (local_weight, (local_pre, local_post)),
        shape=(n, n),
    ).tocsr()

    potentials = _initial_potentials(initial_potentials, n)
    if neuron_labels is not None and len(neuron_labels) != n:
        raise ValueError(
            f"neuron_labels length {len(neuron_labels)} must match subset size {n}"
        )

    state = BrainState(
        potentials=potentials,
        weights=weights,
        timestamp=0.0,
        neuron_labels=neuron_labels,
    )
    return ConnectomeLoadResult(
        state=state,
        neuron_ids=np.asarray(subset_ids, dtype=np.int64),
        id_to_index=id_to_index,
    )


def _resolve_format(path: Path, format: ConnectomeFormat) -> Literal["npz", "csv"]:
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return "npz"
        if suffix == ".csv":
            return "csv"
        raise ValueError(
            f"cannot infer connectome format from '{path.name}'; "
            "use format='npz' or format='csv'"
        )
    if format not in ("npz", "csv"):
        raise ValueError(f"unsupported format: {format}")
    return format


def _read_edges(
    path: Path,
    format: Literal["npz", "csv"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if format == "npz":
        return _read_npz_edges(path)
    pre, post, weight = _read_csv_edges(path)
    return pre, post, weight, None


def _read_npz_edges(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        missing = {"pre", "post", "weight"} - set(archive.files)
        if missing:
            raise ValueError(
                f"NPZ connectome must contain pre, post, weight; missing {sorted(missing)}"
            )
        pre = np.asarray(archive["pre"]).reshape(-1)
        post = np.asarray(archive["post"]).reshape(-1)
        weight = np.asarray(archive["weight"], dtype=np.float64).reshape(-1)
        file_neuron_ids = None
        if "neuron_ids" in archive.files:
            file_neuron_ids = np.asarray(archive["neuron_ids"]).reshape(-1)

    if not (pre.shape == post.shape == weight.shape):
        raise ValueError("pre, post, and weight must have the same length")
    if pre.size == 0:
        raise ValueError("connectome edge list is empty")
    return pre, post, weight, file_neuron_ids


def _read_csv_edges(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        table = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to parse CSV connectome: {path}") from exc

    if table.size == 0:
        raise ValueError("connectome edge list is empty")

    colnames = {name.lower() for name in table.dtype.names or ()}
    pre_key = _pick_column(colnames, ("pre_id", "pre", "presynaptic_id"))
    post_key = _pick_column(colnames, ("post_id", "post", "postsynaptic_id"))
    weight_key = _pick_column(colnames, ("weight", "w", "synapse_weight"))

    pre = np.asarray(table[pre_key], dtype=np.int64).reshape(-1)
    post = np.asarray(table[post_key], dtype=np.int64).reshape(-1)
    weight = np.asarray(table[weight_key], dtype=np.float64).reshape(-1)
    if not (pre.shape == post.shape == weight.shape):
        raise ValueError("pre, post, and weight must have the same length")
    return pre, post, weight


def _pick_column(available: set[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in available:
            return name
    raise ValueError(
        f"CSV must include one of {candidates}; found columns {sorted(available)}"
    )


def _load_neuron_ids_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        ids = np.load(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "neuron_ids" not in archive.files:
                raise ValueError(f"NPZ neuron id file must contain neuron_ids: {path}")
            ids = archive["neuron_ids"]
    else:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            raise ValueError(f"neuron id file is empty: {path}")
        ids = np.asarray([int(value) for value in lines], dtype=np.int64)
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def _resolve_neuron_subset(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    neuron_ids: Sequence[int] | None,
    neuron_ids_file: str | Path | None,
    file_neuron_ids: np.ndarray | None,
) -> np.ndarray:
    if neuron_ids is not None:
        subset = np.asarray(neuron_ids, dtype=np.int64).reshape(-1)
    elif neuron_ids_file is not None:
        subset = _load_neuron_ids_file(Path(neuron_ids_file))
    elif file_neuron_ids is not None:
        subset = np.asarray(file_neuron_ids, dtype=np.int64).reshape(-1)
    else:
        subset = np.unique(np.concatenate([pre, post])).astype(np.int64, copy=False)

    if subset.size == 0:
        raise ValueError("neuron subset is empty")
    return subset


def _filter_and_remap(
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    id_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_i = np.asarray(pre, dtype=np.int64)
    post_i = np.asarray(post, dtype=np.int64)

    keep = np.array(
        [
            (int(p) in id_to_index and int(q) in id_to_index)
            for p, q in zip(pre_i, post_i, strict=True)
        ],
        dtype=bool,
    )
    if not np.any(keep):
        raise ValueError("no synapses remain after applying neuron subset filter")

    local_pre = np.fromiter(
        (id_to_index[int(p)] for p in pre_i[keep]),
        dtype=np.int64,
        count=int(np.count_nonzero(keep)),
    )
    local_post = np.fromiter(
        (id_to_index[int(q)] for q in post_i[keep]),
        dtype=np.int64,
        count=int(np.count_nonzero(keep)),
    )
    local_weight = weight[keep].astype(np.float64, copy=False)
    return local_pre, local_post, local_weight


def _initial_potentials(value: float | np.ndarray, n: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        potentials = np.asarray(value, dtype=np.float64).reshape(-1)
        if potentials.shape != (n,):
            raise ValueError(f"initial_potentials length must be {n}")
        return potentials
    return np.full(n, float(value), dtype=np.float64)
