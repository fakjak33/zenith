"""No-code composition (blend, conditions, regime rules) and their templates."""

from .compose import (OPS, REGIME_USES, SLOT_LABEL, SLOTS, BlendStrategy, ConditionStrategy,  # noqa: F401
                      apply_regime, conditions_state, needs_of, random_mask_test, regime_labels, sentence)
from .templates import TEMPLATES  # noqa: F401
