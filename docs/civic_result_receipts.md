# CIViC aggregate result receipts

`civic_result_receipts.json` contains privacy-safe aggregate calculations for
the mixed-condition study. It provides per-run values, recomputable means and
population standard deviations, and a cryptographic commitment to the withheld
member mapping.

The package deliberately excludes raw media, identifiers, raw features,
checkpoints, execution records, and pair-level bootstrap inputs. It therefore
does not reproduce the private raster-pair intervals or a complete training
run. Use the receipt to verify the mixed-condition aggregate calculations, not
to infer population uncertainty or broader audio-to-video performance.

To validate a supplied receipt, use the Python function
`a2v.reporting.civic_result_receipts.validate_receipt`.
