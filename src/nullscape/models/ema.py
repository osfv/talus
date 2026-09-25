from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model weights (buffers are copied)."""

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup_steps: int = 1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # ramp decay up early so the average isn't dominated by the random init
        d = min(self.decay, (1 + self.num_updates) / (self.warmup_steps + self.num_updates))
        torch._foreach_lerp_(list(self.shadow.parameters()), [p.detach() for p in model.parameters()], 1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow.state_dict(), "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state["shadow"])
        self.num_updates = state["num_updates"]
        self.decay = state.get("decay", self.decay)
