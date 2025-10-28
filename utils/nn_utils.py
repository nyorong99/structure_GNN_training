# -*- coding: utf-8 -*-
def count_parameters(model) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
