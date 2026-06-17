"""GECToR-style sequence tagger for code identifier typo fixing.

Architecture
------------
Adapted from "GECToR – Grammatical Error Correction: Tag, Not Rewrite"
(Omelianchuk et al., 2020) for Python source-code identifier correction.

Instead of natural-language grammatical tags, we use a compact vocabulary of
token-level edit operations derived from the training data:

  ``$KEEP``              — leave this token unchanged
  ``$DELETE``            — remove this token (rare; kept for completeness)
  ``$REPLACE_<token>``   — replace this token with ``<token>``

The model is a pre-trained Transformer encoder (default: ``roberta-base``)
with two linear classification heads stacked on top:

  1. **Error-detection head** — binary: is this token erroneous?
  2. **Tag head** — multi-class: which edit tag to apply?

Both heads are trained jointly with cross-entropy loss.  At inference time
the tag head output is used; the detection head provides an optional
confidence gate.

Iterative correction (up to ``max_iter`` passes) is applied: the model is
re-run on its own output until no more ``$REPLACE`` tags are predicted or
the iteration limit is reached.

Modules
-------
vocab           — tag vocabulary (build from dataset, save/load)
tokenize_code   — Python-tokenize → subword alignment helpers
dataset         — PyTorch Dataset wrapping the JSONL files
model           — GECToRModel nn.Module
train           — training-loop CLI  (``python -m src.gector.train``)
predict         — iterative inference engine
fixer           — GECToRFixer(NameFixer) for harness integration
"""

from .fixer import GECToRFixer

__all__ = ["GECToRFixer"]
