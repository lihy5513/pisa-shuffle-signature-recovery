# Third-party notices

- The ML-DSA firmware includes code based on the public-domain
  CRYSTALS-Dilithium implementation; its upstream notice is retained in
  `firmware/mldsa/LICENSE`.
- The ChipWhisperer SimpleSerial target code carries NewAE Technology's
  GPL-3.0-or-later notice in its source headers. The license text is included
  at `firmware/COPYING.GPL-3.0.txt`.
- The ML-KEM optimized assembly includes the Plantard arithmetic macros by
  Junhao Huang, copied from the `mupq/pqm4` ML-KEM-768 `m4fspeed/macros.i`
  source. The `pqm4` README states that general repository code is
  dual-licensed under Apache-2.0 and CC0; this package relies on the CC0
  option. The author, date, and paper notices are preserved in the file. The upstream source is at
  <https://github.com/mupq/pqm4/blob/5e5cc76d0b072ee7077e58afcbc619a1a67a53b4/crypto_kem/ml-kem-768/m4fspeed/macros.i>.
- `firmware/mlkem/matacci.i` is required by the experiment's ML-KEM assembly
  target. Its contents and local attribution are preserved from the source
  tree.

Other upstream notices in individual source files remain unchanged. This
package does not claim ownership of third-party components.
