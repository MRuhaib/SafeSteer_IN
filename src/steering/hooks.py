"""
SafeSteer-IN  ·  PyTorch Forward-Hook Utilities
=================================================
Provides an architecture-agnostic way to capture and inject into the
residual stream of any HuggingFace CausalLM.

Two main abstractions:
    • ActivationCollector  — record residual-stream activations
    • SteeringHook         — add a scaled steering vector at inference time
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helper: resolve model layer list
# ─────────────────────────────────────────────────────────────────────────────


def get_layers(model: nn.Module, accessor: str = "model.layers") -> nn.ModuleList:
    """
    Walk the attribute path ``accessor`` (e.g. "model.layers") on ``model``
    and return the resulting ``nn.ModuleList``.

    Supports:
        LLaMA / Mistral  →  model.model.layers
        GPT-NeoX          →  model.gpt_neox.layers
        Falcon             →  model.transformer.h
    """
    obj = model
    for attr in accessor.split("."):
        obj = getattr(obj, attr)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ActivationCollector — capture residual-stream activations
# ─────────────────────────────────────────────────────────────────────────────


class ActivationCollector:
    """
    Registers forward hooks on the specified layers and stores the
    hidden-state output (first element of the layer's output tuple).

    Usage::

        collector = ActivationCollector(model, target_layers=[12, 16, 20])
        with torch.no_grad():
            model(**inputs)
        acts = collector.get()          # {12: Tensor, 16: Tensor, 20: Tensor}
        collector.clear()               # reset for next input
        collector.remove_hooks()        # clean up when done
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[int],
        layer_accessor: str = "model.layers",
        reduction: str = "mean",  # "mean" | "last" | "none"
    ):
        self.target_layers = target_layers
        self.reduction = reduction
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        layers = get_layers(model, layer_accessor)
        for idx in target_layers:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    # ── internal ─────────────────────────────────────────────────────────
    def _make_hook(self, layer_idx: int) -> Callable:
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.reduction == "mean":
                # Mean over sequence-length dimension → (batch, hidden_dim)
                act = hidden.mean(dim=1).detach().cpu()
            elif self.reduction == "last":
                # Last token position → (batch, hidden_dim)
                act = hidden[:, -1, :].detach().cpu()
            else:
                act = hidden.detach().cpu()
            self._activations[layer_idx] = act

        return hook_fn

    # ── public API ───────────────────────────────────────────────────────
    def get(self) -> Dict[int, torch.Tensor]:
        return dict(self._activations)

    def clear(self):
        self._activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SteeringHook — inject a steering vector into the residual stream
# ─────────────────────────────────────────────────────────────────────────────


class SteeringHook:
    """
    Adds ``alpha * vector`` to the residual stream at a given layer during
    the forward pass.  The vector is *subtracted* from the unsafe direction,
    i.e., it steers the model **away** from the unsafe manifold.

    Usage::

        hook = SteeringHook(model, layer_idx=16, vector=sv, alpha=15.0)
        output_ids = model.generate(input_ids, max_new_tokens=200)
        hook.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float = 15.0,
        layer_accessor: str = "model.layers",
    ):
        self.layer_idx = layer_idx
        self.alpha = alpha
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None

        # Ensure vector is on the right device and shape
        layers = get_layers(model, layer_accessor)
        layer = layers[layer_idx]

        # Try to detect device from layer parameters
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.vector = vector.to(device).to(torch.float16)
        if self.vector.dim() == 1:
            self.vector = self.vector.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
        elif self.vector.dim() == 2:
            self.vector = self.vector.unsqueeze(1)  # (1, 1, hidden)

        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Steer: subtract the unsafe direction (equivalently, add the negated vector)
        # Convention: steering_vec = mean_unsafe − mean_safe
        # To make model safer, we subtract alpha * steering_vec
        hidden = hidden - self.alpha * self.vector.to(hidden.device).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        self.remove()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convenience: multi-layer steering
# ─────────────────────────────────────────────────────────────────────────────


class MultiLayerSteeringHook:
    """
    Apply steering vectors at multiple layers simultaneously.
    ``vectors`` is a dict  {layer_idx: (vector, alpha)}.
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: Dict[int, Tuple[torch.Tensor, float]],
        layer_accessor: str = "model.layers",
    ):
        self._hooks: List[SteeringHook] = []
        for layer_idx, (vec, alpha) in vectors.items():
            self._hooks.append(
                SteeringHook(model, layer_idx, vec, alpha, layer_accessor)
            )

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove()
