# PISA device-attack reproduction package

This source package contains the ChipWhisperer target firmware, acquisition
controller, and label-free Hard/Posterior Count recovery code for the
ML-DSA-44 and ML-KEM-512 experiments. It also includes the experiment-only
test key pairs used for the reported device runs.

The test keys were generated for these measurements and are not production or
user keys. They are published solely so that the device experiments can be
repeated with the same targets. Do not use them for any security-sensitive
purpose.

## What is and is not included

- `firmware/`: source-only ChipWhisperer targets; no prebuilt firmware images.
- `capture/capture.py`: acquisition controller. It reads a selected test key
  to load the target, but never writes the key into a capture file.
- `test_keys/`: the eight experiment-only key pairs used per scheme. The
  generation manifest is replaced with a portable manifest containing only
  relative names and file hashes.
- `attack/`: the unscored recovery implementation. It does not load a
  reference secret, coordinate labels, or true permutations.
- `results/summary.json`: aggregate evaluation results only; the separate
  truth-scoring code and per-key truth vectors are not included.

Waveform captures are deliberately not included. Consequently, this package
supports rerunning the acquisition and attack on the same experimental keys,
but does not provide the original sample-for-sample traces needed to reproduce
the exact reported run. New captures are large and depend on the board,
oscilloscope configuration, and measurement session.

The reported scope is one target secret polynomial/component per key, not a
claim of complete private-key recovery. The first 500 unlabeled calibration
traces are included in every reported prefix `M`.

## Build and capture

Install the Python packages in `requirements.txt`. To acquire traces, install
the ChipWhisperer SDK separately, connect a CW308 STM32F3 target and
ChipWhisperer Lite, and build the relevant firmware with that SDK's
`firmware/mcu` directory as `FIRMWAREPATH`. The ML-DSA target uses
`CAPTURE_ACCEPTED_CS1=1`; the ML-KEM target uses its default
`KYBER_CAPTURE_USHUF` configuration. See each firmware directory's makefile.

Example commands (write captures outside this repository):

```powershell
python capture/capture.py --scheme mldsa --key-file test_keys/mldsa/key00.sk.bin --firmware C:/lab/mldsa.hex --output C:/lab/captures/mldsa-key00.npz --traces 2000
python capture/capture.py --scheme mlkem --key-file test_keys/mlkem/key00.sk.bin --firmware C:/lab/mlkem.hex --output C:/lab/captures/mlkem-key00.npz --traces 2400
```

The acquisition controller uses fresh random messages for ML-DSA and fresh
chosen inputs for ML-KEM. Device randomness in these experimental firmware
builds is not suitable for production use.

## Run recovery

Use an externally acquired `.npz` file with the schema emitted by
`capture/capture.py`:

```powershell
python attack/recover.py --scheme mldsa --key 0 --capture-path C:/lab/captures/mldsa-key00.npz --prefixes 500 800 1000 1200 1400 1600 --device cpu --output C:/lab/mldsa-key00-candidates.json
python attack/recover.py --scheme mlkem --key 0 --capture-path C:/lab/captures/mlkem-key00.npz --prefixes 500 800 1000 1200 1400 1600 1800 2000 2200 2400 --device cpu --output C:/lab/mlkem-key00-candidates.json
```

The defaults use five solver starts and 3,000 epochs per prefix. CPU runs can
take a long time. `--epochs 1 --prefixes 500` is only a smoke test, not a
reproduction of the reported recovery performance. Candidate outputs are not
scored by the attack code.

Run `python verify_release.py` to verify the file and test-key hashes and to
confirm that no waveform capture has entered the package.

## Licensing

This package is released under GPL-3.0-or-later; see `LICENSE`. Third-party
components retain their source notices and license terms, documented in
`THIRD_PARTY_NOTICES.md` and in the source directories.
