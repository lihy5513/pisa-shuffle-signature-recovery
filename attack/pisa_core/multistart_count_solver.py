"""Public-score checkpointing and prefix retention for paired count recovery.

No evaluation secret, labels, or shuffle indices are accepted by this module.
Restarts are independent Adam trajectories, vectorized on the same device.
Retained integer candidates are rescored using only the CURRENT trace prefix.
"""

from __future__ import annotations

import time
import numpy as np
import torch


def products(c_fft: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    padded = torch.cat((candidates, torch.zeros_like(candidates)), dim=-1)
    spectrum = torch.fft.fft(padded, dim=-1)
    return torch.fft.ifft(c_fft[None] * spectrum[:, None], dim=-1).real[..., :256]


def public_scores(c_fft, candidates, observed, log_pmf, method):
    """Higher is better; integer rounding removes FFT roundoff at zero."""
    counts = (products(c_fft, candidates).round() < 0).sum(dim=-1)
    if method == "hard_count":
        scores = -((counts.double() - observed.double()[None]) ** 2).mean(dim=1)
    elif method == "posterior_count":
        scores = log_pmf.double()[None].expand(len(candidates), -1, -1).gather(
            2, counts[..., None]
        ).squeeze(-1).mean(dim=1)
    else:
        raise ValueError(method)
    return scores, counts


def solve_multistart_prefix(
    c_vectors, observed_counts, log_count_posterior, method,
    seeds=(1001, 1009, 1013, 1021, 1031), retained_candidates=None,
    eta=2, epochs=3000, device="cuda", lr=0.08,
):
    """Five fresh starts plus rescoring of earlier public-score candidates.

    Same checkpoint frequency, annealing, and per-start budget for both methods.
    This is a sequential-prefix protocol, NOT independent-prefix evaluation.
    Equal scores keep the earlier incumbent deterministically.
    """
    if epochs < 1 or not seeds:
        raise ValueError("positive epochs and at least one seed required")
    dev = torch.device(device)
    c = np.asarray(c_vectors, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != 256 or not np.all(c == np.rint(c)):
        raise ValueError("integer public vectors of shape (M,256) required")
    observed = torch.as_tensor(observed_counts, dtype=torch.float32, device=dev)
    pmf = torch.as_tensor(log_count_posterior, dtype=torch.float32, device=dev)
    if observed.shape != (len(c),) or pmf.shape != (len(c), 257):
        raise ValueError("count inputs do not match current public prefix")
    c_fft = torch.fft.fft(torch.as_tensor(np.concatenate((c, -c), axis=1), device=dev))
    initial = []
    for seed in seeds:
        generator = torch.Generator(device=dev).manual_seed(int(seed))
        initial.append(torch.empty(256, device=dev).uniform_(-eta, eta, generator=generator))
    z = torch.nn.Parameter(torch.stack(initial))
    optimizer = torch.optim.Adam([z], lr=lr)
    best_candidates = z.detach().round().clone()
    with torch.no_grad():
        best_scores, _ = public_scores(c_fft, best_candidates, observed, pmf, method)
    best_scores = best_scores.clone()
    temperature = 25.0
    checkpoints = max(1, epochs // 30)
    decay_step = max(1, epochs // 12)
    started = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = torch.sigmoid(-(products(c_fft, z) + 0.5) / temperature).sum(dim=-1)
        if method == "hard_count":
            losses = ((predicted - observed[None]) ** 2).mean(dim=1)
        else:
            lower = predicted.clamp(0, 256).floor().long()
            upper = (lower + 1).clamp(max=256)
            fraction = predicted.clamp(0, 256) - lower
            rows = pmf[None].expand(len(seeds), -1, -1)
            low = rows.gather(2, lower[..., None]).squeeze(-1)
            high = rows.gather(2, upper[..., None]).squeeze(-1)
            losses = -(low + fraction * (high - low)).mean(dim=1)
        # Sum, not mean: each restart has the single-trajectory gradient scale.
        losses.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(-eta, eta)
            if (epoch + 1) % checkpoints == 0 or epoch == epochs - 1:
                candidates = z.round()
                scores, _ = public_scores(c_fft, candidates, observed, pmf, method)
                improved = scores > best_scores
                best_candidates[improved] = candidates[improved]
                best_scores[improved] = scores[improved]
        if (epoch + 1) % decay_step == 0:
            temperature = max(0.25, temperature * 0.72)

    # Earlier candidates have priority only in an EXACT public-score tie.
    pool = []
    seen = set()
    earlier = [] if retained_candidates is None else list(retained_candidates)
    for candidate in earlier + list(best_candidates.detach().cpu().numpy()):
        candidate = np.asarray(candidate, dtype=np.int32)
        if candidate.shape != (256,) or np.any(np.abs(candidate) > eta):
            raise ValueError("invalid retained candidate")
        key = candidate.tobytes()
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
    tensor_pool = torch.as_tensor(np.stack(pool), device=dev, dtype=torch.float32)
    with torch.no_grad():
        scores = []
        for begin in range(0, len(pool), 8):
            score, _ = public_scores(c_fft, tensor_pool[begin:begin + 8], observed, pmf, method)
            scores.extend(score.cpu().tolist())
    selected = int(np.argmax(scores))
    return {
        "selected_candidate": pool[selected].tolist(),
        "selected_public_score": float(scores[selected]),
        "selected_pool_index": selected,
        "retained_candidate_count": len(pool),
        "candidate_pool": [item.tolist() for item in pool],
        "candidate_pool_scores": scores,
        "restart_candidates": best_candidates.cpu().numpy().astype(np.int32).tolist(),
        "restart_scores": best_scores.cpu().tolist(),
        "final_surrogate_losses": losses.detach().cpu().tolist(),
        "seeds": list(seeds), "epochs_per_restart": epochs,
        "temperature_final": temperature,
        "solver_time_seconds": time.perf_counter() - started,
    }
