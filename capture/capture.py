"""Key-external ChipWhisperer controller for the two PISA device experiments.

The capture is a fresh acquisition, not an exporter of the archived traces.
The test key is supplied explicitly and is never serialized in the output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


SAMPLES = 24400
N = 256
ARTIFACT = Path(__file__).resolve().parents[1]


def upload_key(target, key: bytes) -> None:
    if len(key) % 32:
        raise ValueError("key length must be a multiple of 32 bytes")
    for index in range(len(key) // 32):
        block = key[index * 32:(index + 1) * 32]
        target.simpleserial_write("k", bytes([index]) + block)
        echoed = target.simpleserial_read("k", 32, timeout=3000)
        if echoed is None or bytes(echoed) != block:
            raise RuntimeError(f"key upload failed at block {index}")


def pack_10bit(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(N)
    out = bytearray(320)
    for j in range(N // 4):
        t0, t1, t2, t3 = (int(x) for x in values[4 * j:4 * j + 4])
        out[5 * j] = t0 & 255
        out[5 * j + 1] = (t0 >> 8) | ((t1 & 63) << 2)
        out[5 * j + 2] = (t1 >> 6) | ((t2 & 15) << 4)
        out[5 * j + 3] = (t2 >> 4) | ((t3 & 3) << 6)
        out[5 * j + 4] = t3 >> 2
    return bytes(out)


def make_mlkem_input(rng: np.random.Generator) -> tuple[bytes, np.ndarray]:
    """Selected component 0: 16 random signed +/-3 coefficients; v=0."""
    codes = np.zeros((2, N), dtype=np.uint16)
    positions = np.sort(rng.choice(N, size=16, replace=False))
    codes[0, positions] = rng.choice(np.asarray([1, 1023], dtype=np.uint16), size=16)
    centered = np.zeros((2, N), dtype=np.int16)
    centered[0, positions] = np.where(codes[0, positions] == 1, 3, -3)
    ciphertext = pack_10bit(codes[0]) + pack_10bit(codes[1]) + bytes(128)
    assert len(ciphertext) == 768
    return ciphertext, centered


def send_ciphertext(target, ciphertext: bytes, delay: float) -> None:
    for index in range(24):
        block = ciphertext[index * 32:(index + 1) * 32]
        target.simpleserial_write("p", bytes([index]) + block)
        if delay:
            time.sleep(delay)


def configure_scope(cw, samples: int, timeout: float):
    scope = cw.scope()
    target = cw.target(scope)
    scope.default_setup()
    scope.trigger.triggers = "tio4"
    scope.clock.adc_src = "clkgen_x1"
    time.sleep(1.0)
    scope.adc.decimate = 1
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.timeout = timeout
    return scope, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["mldsa", "mlkem"], required=True)
    parser.add_argument("--key-file", type=Path, required=True,
                        help="experiment-only test key; packaged keys are under test_keys/")
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=int, required=True,
                        help="number of valid responses to collect")
    parser.add_argument("--max-extra-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--chunk-delay", type=float, default=0.03)
    args = parser.parse_args()
    if args.traces < 1 or args.samples < 1 or args.max_extra_attempts < 0:
        parser.error("traces/samples must be positive; extra attempts non-negative")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must be an .npz file")
    key_path = args.key_file.resolve()
    if key_path.is_relative_to(ARTIFACT) and not key_path.is_relative_to(ARTIFACT / "test_keys"):
        parser.error("keys inside the package must come from its test_keys/ directory")
    if args.output.resolve().is_relative_to(ARTIFACT):
        parser.error("write new captures outside the publication artifact for review")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.firmware.is_file():
        raise FileNotFoundError(args.firmware)
    key = key_path.read_bytes()
    expected = 2560 if args.scheme == "mldsa" else 1632
    if len(key) != expected:
        raise ValueError(f"{args.scheme} test key must be {expected} bytes")

    import chipwhisperer as cw

    rng = np.random.default_rng(args.seed)
    scope, target = configure_scope(cw, args.samples, args.timeout)
    rows: list[dict[str, np.ndarray | int | bool]] = []
    try:
        cw.program_target(scope, cw.programmers.STM32FProgrammer, str(args.firmware))
        time.sleep(1.0)
        try:
            target.ser.flush()
        except Exception:
            pass
        upload_key(target, key)
        valid_count = 0
        for attempt in range(args.traces + args.max_extra_attempts):
            if args.scheme == "mldsa":
                message = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
                scope.arm()
                target.simpleserial_write("p", message)
                timed_out = bool(scope.capture())
                response = target.simpleserial_read("r", 32, timeout=60000)
            else:
                ciphertext, u_centered = make_mlkem_input(rng)
                scope.arm()
                send_ciphertext(target, ciphertext, args.chunk_delay)
                target.simpleserial_write("p", bytes([255]) + bytes(32))
                timed_out = bool(scope.capture())
                response = target.simpleserial_read("r", 32, timeout=30000)
            trace = np.asarray(scope.get_last_trace(), dtype=np.float32).copy()
            response_bytes = b"" if response is None else bytes(response)
            triggers = int(scope.adc.trig_count)
            valid = (not timed_out and len(trace) == args.samples
                     and len(response_bytes) == 32 and triggers > 0)
            if args.scheme == "mlkem":
                valid = valid and triggers <= args.samples
            if valid:
                row = {"traces": trace, "trigger_counts": triggers, "valid": True}
                if args.scheme == "mldsa":
                    row["responses"] = np.frombuffer(response_bytes, dtype=np.uint8).copy()
                else:
                    row["u_centered"] = u_centered
                rows.append(row)
                valid_count += 1
            if (attempt + 1) % 25 == 0 or valid_count == args.traces:
                print(f"attempts={attempt + 1} valid={valid_count}/{args.traces}", flush=True)
            if valid_count == args.traces:
                break
        if valid_count != args.traces:
            raise RuntimeError(f"only {valid_count}/{args.traces} valid responses; no output written")
        arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
        arrays.update(decimate=np.asarray(1), adc_source=np.asarray("clkgen_x1"),
                      scheme=np.asarray(args.scheme))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(args.output)
        print(f"saved {args.output}")
    finally:
        scope.dis()


if __name__ == "__main__":
    main()
