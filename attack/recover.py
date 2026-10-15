"""Unscored, label-free Hard/Soft Count recovery from sanitized captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "pisa_core"))
from channels import make_posterior_count_channel  # noqa: E402
from multistart_count_solver import solve_multistart_prefix  # noqa: E402
from poi_selection import select_global_poi_grid  # noqa: E402


def challenge(seed: bytes) -> np.ndarray:
    """ML-DSA-44 challenge polynomial from its public response seed."""
    buf = hashlib.shake_256(seed).digest(136 * 4)
    signs = int.from_bytes(buf[:8], "little")
    pos = 8
    c = np.zeros(256, dtype=np.int64)
    for i in range(256 - 39, 256):
        while True:
            if pos >= len(buf):
                raise RuntimeError("challenge sampler exhausted SHAKE buffer")
            b = buf[pos]
            pos += 1
            if b <= i:
                break
        c[i] = c[b]
        c[b] = 1 - 2 * (signs & 1)
        signs >>= 1
    return c


def fit_gmm_posteriors(feature: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Original 40-step, per-slot two-component GMM; high-mean class last."""
    x_train = np.asarray(feature[train], dtype=float)[:, :, None]
    all_values = np.asarray(feature, dtype=float)
    mean = np.stack([
        np.quantile(x_train[:, :, 0], 0.25, axis=0),
        np.quantile(x_train[:, :, 0], 0.75, axis=0),
    ], axis=1)
    variance = np.repeat(
        np.maximum(np.var(x_train[:, :, 0], axis=0), 1e-8)[:, None], 2, axis=1
    )
    weight = np.full((256, 2), 0.5, dtype=float)
    for _ in range(40):
        log_prob = (
            np.log(np.maximum(weight[None, :, :], 1e-12))
            - 0.5 * (
                np.log(2.0 * np.pi * variance[None, :, :])
                + (x_train - mean[None, :, :]) ** 2 / variance[None, :, :]
            )
        )
        log_prob -= log_prob.max(axis=2, keepdims=True)
        resp = np.exp(log_prob)
        resp /= np.maximum(resp.sum(axis=2, keepdims=True), 1e-12)
        count = np.maximum(resp.sum(axis=0), 1e-8)
        mean = (resp * x_train).sum(axis=0) / count
        variance = (resp * (x_train - mean[None, :, :]) ** 2).sum(axis=0) / count
        variance = np.maximum(variance, 1e-8)
        weight = count / len(train)
    x = all_values[:, :, None]
    log_prob = (
        np.log(np.maximum(weight[None, :, :], 1e-12))
        - 0.5 * (
            np.log(2.0 * np.pi * variance[None, :, :])
            + (x - mean[None, :, :]) ** 2 / variance[None, :, :]
        )
    )
    log_prob -= log_prob.max(axis=2, keepdims=True)
    posterior = np.exp(log_prob)
    posterior /= np.maximum(posterior.sum(axis=2, keepdims=True), 1e-12)
    order = np.argsort(mean, axis=1)
    return np.take_along_axis(posterior, order[None, :, :], axis=2)


def load_capture(path: Path, scheme: str, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        expected = ({"traces", "responses", "trigger_counts", "valid",
                     "decimate", "adc_source", "scheme"} if scheme == "mldsa"
                    else {"traces", "u_centered", "trigger_counts", "valid",
                          "decimate", "adc_source", "scheme"})
        if set(data.files) != expected:
            raise ValueError(f"unexpected NPZ member names: {sorted(set(data.files) ^ expected)}")
        if str(np.asarray(data["scheme"]).reshape(())) != scheme:
            raise ValueError("capture scheme mismatch")
        valid = np.asarray(data["valid"], dtype=bool)
        if not bool(valid.all()):
            raise ValueError("release capture must contain only validated rows")
        if len(valid) < maximum:
            raise ValueError(f"capture has {len(valid)} rows, need {maximum}")
        traces = np.asarray(data["traces"][:maximum], dtype=np.float32)
        triggers = np.asarray(data["trigger_counts"][:maximum], dtype=np.float64)
        decimate = int(np.asarray(data["decimate"]).reshape(()))
        if scheme == "mldsa":
            responses = np.asarray(data["responses"][:maximum], dtype=np.uint8)
            if responses.shape != (maximum, 32):
                raise ValueError("expected one 32-byte public challenge seed per row")
            vectors = np.asarray([challenge(bytes(row)) for row in responses], dtype=np.float32)
        else:
            u = np.asarray(data["u_centered"][:maximum], dtype=np.float32)
            if u.shape != (maximum, 2, 256):
                raise ValueError("expected public centered u vectors of shape (M,2,256)")
            vectors = u[:, 0] / 3.0
    if traces.ndim != 2 or traces.shape[1] != 24400:
        raise ValueError("unexpected waveform shape")
    return traces, triggers, vectors, decimate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["mldsa", "mlkem"], required=True)
    parser.add_argument("--key", type=int, choices=range(8), required=True)
    parser.add_argument("--capture-path", type=Path, required=True,
                        help="external capture NPZ produced by capture/capture.py")
    parser.add_argument("--prefixes", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=["hard_count", "posterior_count"],
                        default=["hard_count", "posterior_count"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1009, 1013, 1021, 1031])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefixes = sorted(set(args.prefixes))
    if prefixes[0] < 500:
        parser.error("all prefixes must include the first 500 calibration traces")
    capture_path = args.capture_path.resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    if capture_path.is_relative_to(ROOT):
        parser.error("keep waveform captures outside this source package")
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("write candidate output outside this source package")
    if args.output.exists():
        raise FileExistsError(args.output)
    traces, triggers, vectors, decimate = load_capture(
        capture_path, args.scheme, max(prefixes)
    )
    b, a = butter(2, 0.05, btype="highpass")
    filtered = np.abs(filtfilt(b, a, traces.astype(np.float64), axis=1)).astype(np.float32)
    if args.scheme == "mldsa":
        cadence_candidates = range(19, 24)
        offset_candidates = range(0, 85)
        eta = 2
    else:
        center = max(1, int(round(np.median(triggers[:500] / decimate / 256))))
        cadence_candidates = range(center - 1, center + 2)
        offset_candidates = range(0, max(49, center * 4 + 1))
        eta = 3
    selector, _ = select_global_poi_grid(
        filtered[:500], cadence_candidates=cadence_candidates,
        offset_candidates=offset_candidates, n_coefficients=256,
        n_stability_folds=5, trim_fraction=0.1,
    )
    posterior = np.asarray(
        fit_gmm_posteriors(filtered[:, selector.indices], np.arange(500))[:, :, 1],
        dtype=np.float32,
    )
    hard_counts = (posterior >= 0.5).sum(axis=1).astype(np.int64)
    pools = {method: [] for method in args.methods}
    rows = []
    for prefix in prefixes:
        log_pmf = np.asarray(
            make_posterior_count_channel(posterior[:prefix]).log_count_posterior,
            dtype=np.float32,
        )
        for method in args.methods:
            run = solve_multistart_prefix(
                vectors[:prefix], hard_counts[:prefix], log_pmf, method,
                seeds=args.seeds, retained_candidates=pools[method],
                eta=eta, epochs=args.epochs, device=args.device,
            )
            pools[method] = run["candidate_pool"]
            rows.append({"prefix": prefix, "method": method,
                         "candidate": run["selected_candidate"],
                         "public_score": run["selected_public_score"]})
            print(f"M={prefix} {method}: public score {run['selected_public_score']:.6g}", flush=True)
    payload = {"scheme": args.scheme, "key_index": args.key,
               "capture_file": capture_path.name,
               "capture_sha256": sha256(capture_path),
               "calibration_traces": 500, "calibration_included_in_M": True,
               "poi_cadence": selector.cadence, "poi_offset": selector.offset,
               "epochs_per_start": args.epochs, "seeds": args.seeds,
               "no_reference_secret_or_labels_loaded": True, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
