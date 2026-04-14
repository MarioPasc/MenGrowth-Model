"""Capture WindowAttention weights from MONAI's SwinUNETR.

The MONAI ``WindowAttention.forward`` returns the projected token output
``self.proj(attn @ v)`` and discards ``attn`` immediately. A standard
``register_forward_hook`` cannot expose the post-softmax attention matrix,
so this module monkey-patches the bound ``forward`` method of every
``WindowAttention`` instance it discovers in the model.

The patched forward is a verbatim port of MONAI 1.5's implementation
(``swin_unetr.py`` lines 509-532). The only behaviour change is a single
line that calls a user-supplied callback with the post-softmax tensor
*before* dropout. The patch is removed on context exit.

Two consumption modes are provided:

- **store** (default): retain the raw attention tensor on CPU. Useful for
  unit tests and small inputs but quickly exhausts memory at stage 1.
- **callback**: the user passes a ``process_fn(key, attn, mask)`` that is
  invoked synchronously inside the patched forward; only the summary that
  the callback writes survives the hook.

Stage indexing follows the convention in ``engine/tsi.py``:
``layers1 -> stage 1, layers2 -> stage 2, layers3 -> stage 3, layers4 ->
stage 4``. Stage 0 (patch embedding) has no WindowAttention modules.
"""

from __future__ import annotations

import logging
import re
import types
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Regex to parse e.g. "swinViT.layers3.0.blocks.1.attn" -> stage=3, block=1.
_NAME_RE = re.compile(r"layers(?P<stage>\d+)\.0\.blocks\.(?P<block>\d+)\.attn$")


@dataclass
class CapturedAttention:
    """Metadata + raw attention for one WindowAttention call.

    Attributes
    ----------
    stage : int
        Encoder stage (1-4).
    block : int
        Block index within the stage (0=non-shifted, 1=shifted).
    num_heads : int
    declared_window_size : tuple[int, ...]
        The module's ``self.window_size`` (the *configured* window). MONAI's
        ``forward_part1`` clamps this to the feature-map size at runtime,
        so the actual window may be smaller than declared.
    n_tokens : int
        Actual ``N`` observed in the post-softmax attention tensor; this
        equals the product of the clamped window dimensions.
    attn_weights : torch.Tensor
        Post-softmax attention tensor of shape
        ``[n_windows * batch, num_heads, n_tokens, n_tokens]``.
    """

    stage: int
    block: int
    num_heads: int
    declared_window_size: tuple[int, ...]
    n_tokens: int
    attn_weights: torch.Tensor


# Public type aliases.
ProcessFn = Callable[[str, torch.Tensor, "torch.Tensor | None"], None]
"""Callback signature: ``process_fn(key, attn_post_softmax, mask)``.

``key`` is the canonical ``"stage_{s}_block_{b}"`` identifier; ``attn`` is
the post-softmax attention tensor; ``mask`` is the attention mask passed to
the forward call (``None`` for non-shifted blocks).
"""


def _make_patched_forward(module: nn.Module, key: str, owner: "AttentionCapture") -> Callable:
    """Build a bound replacement for ``WindowAttention.forward`` that captures attn.

    The body mirrors MONAI's implementation exactly so the patched model
    is numerically identical to the original. The only addition is a
    branch that either stores the raw ``attn`` tensor or invokes the
    user-supplied callback.
    """

    def patched_forward(self: nn.Module, x: torch.Tensor, mask: "torch.Tensor | None"):
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # ---- CAPTURE ----
        owner._on_attention(key, self, attn, mask)
        # -----------------

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    return patched_forward


class AttentionCapture(AbstractContextManager):
    """Context manager that captures WindowAttention weights from a model.

    Parameters
    ----------
    model : nn.Module
        Any module containing ``WindowAttention`` instances. Both raw
        ``SwinUNETR`` and LoRA-wrapped ``LoRAOriginalDecoderModel`` work
        because the LoRA wrapper does not change attribute paths.
    mode : str
        ``"store"`` retains raw attention tensors on CPU (memory heavy).
        ``"callback"`` only forwards them to ``process_fn``.
    process_fn : ProcessFn | None
        Required when ``mode="callback"``. Called synchronously inside the
        patched forward, so it must be cheap and side-effect-free relative
        to the model's own state.
    target_stages : Iterable[int] | None
        If given, only patch modules whose stage index is in this set.
        Useful to skip stage 1 (large window count) for memory.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "store",
        process_fn: ProcessFn | None = None,
        target_stages: "set[int] | None" = None,
    ) -> None:
        if mode not in ("store", "callback"):
            raise ValueError(f"mode must be 'store' or 'callback', got {mode!r}")
        if mode == "callback" and process_fn is None:
            raise ValueError("mode='callback' requires a process_fn")
        self.model = model
        self.mode = mode
        self.process_fn = process_fn
        self.target_stages = set(target_stages) if target_stages is not None else None

        self._captured: dict[str, CapturedAttention] = {}
        self._originals: dict[str, Callable] = {}
        self._modules: dict[str, nn.Module] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "AttentionCapture":
        for name, module in self.model.named_modules():
            if type(module).__name__ != "WindowAttention":
                continue
            match = _NAME_RE.search(name)
            if match is None:
                logger.debug("Skipping WindowAttention with unparseable path: %s", name)
                continue
            stage = int(match.group("stage"))
            block = int(match.group("block"))
            if self.target_stages is not None and stage not in self.target_stages:
                continue
            key = f"stage_{stage}_block_{block}"
            self._modules[key] = module
            self._originals[key] = module.forward  # bound method reference
            module.forward = types.MethodType(
                _make_patched_forward(module, key, self), module
            )
        if not self._modules:
            logger.warning(
                "AttentionCapture installed but found no WindowAttention modules "
                "(target_stages=%s)", self.target_stages,
            )
        else:
            logger.info(
                "AttentionCapture patched %d WindowAttention modules: %s",
                len(self._modules), sorted(self._modules.keys()),
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore originals first (safe to call even if exc raised).
        for key, original in self._originals.items():
            module = self._modules[key]
            module.forward = original  # type: ignore[assignment]
        self._originals.clear()
        self._modules.clear()
        # Keep self._captured so the caller can still read it after exit.
        return None

    # ------------------------------------------------------------------
    # Internal callback dispatch
    # ------------------------------------------------------------------
    def _on_attention(
        self,
        key: str,
        module: nn.Module,
        attn: torch.Tensor,
        mask: "torch.Tensor | None",
    ) -> None:
        """Dispatch the post-softmax attention tensor."""
        if self.mode == "store":
            stage = int(key.split("_")[1])
            block = int(key.split("_")[3])
            self._captured[key] = CapturedAttention(
                stage=stage,
                block=block,
                num_heads=int(module.num_heads),
                declared_window_size=tuple(int(w) for w in module.window_size),
                n_tokens=int(attn.shape[-1]),
                attn_weights=attn.detach().cpu(),
            )
        else:
            assert self.process_fn is not None
            self.process_fn(key, attn.detach(), mask)

    # ------------------------------------------------------------------
    # Public access to stored data
    # ------------------------------------------------------------------
    def get_attention_maps(self) -> dict[str, CapturedAttention]:
        """Return the captured attention tensors keyed by stage/block.

        Only meaningful when ``mode='store'``.
        """
        if self.mode != "store":
            raise RuntimeError("get_attention_maps() is only valid for mode='store'")
        return dict(self._captured)

    def keys(self) -> list[str]:
        """Discovered hook keys (``stage_s_block_b``)."""
        return sorted(self._modules.keys())


def discover_window_attention_modules(
    model: nn.Module,
) -> list[tuple[str, int, int, nn.Module]]:
    """Enumerate ``WindowAttention`` modules in ``model``.

    Returns a list of ``(name, stage, block, module)`` tuples. Useful for
    diagnostics and for tests that verify the hook discovery logic without
    actually installing the patch.
    """
    out: list[tuple[str, int, int, nn.Module]] = []
    for name, module in model.named_modules():
        if type(module).__name__ != "WindowAttention":
            continue
        match = _NAME_RE.search(name)
        if match is None:
            continue
        out.append((name, int(match.group("stage")), int(match.group("block")), module))
    return out
