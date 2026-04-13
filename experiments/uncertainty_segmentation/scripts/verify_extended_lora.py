"""Local verification script for extended LoRA target_module_types.

Builds ``LoRASwinViT`` wrappers on top of the actual BrainSegFounder
checkpoint for several ``(rank, stages, module_types)`` configurations and
checks that:

1. ``_find_lora_targets`` discovers exactly the expected number of targets.
2. Every ``(stage, block, type)`` triple appears exactly once.
3. PEFT actually injects one ``lora_A.default`` and one ``lora_B.default``
   per target (guards against silent endswith-mismatches).
4. The trainable parameter count equals ``sum(r*(in+out))`` over all
   targeted Linear modules (exact, not tolerance-based).
5. On GPU, a fp16-autocast forward pass produces the expected output
   shape, a backward pass populates gradients on LoRA params only, and
   peak memory is reported. OOM is caught so downstream configs still run.

Usage::

    ~/.conda/envs/growth/bin/python \\
        experiments/uncertainty_segmentation/scripts/verify_extended_lora.py \\
        --checkpoint-path <PATH_TO_finetuned_model_fold_0.pt> \\
        --device cuda:0

A markdown report is written next to this script.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from growth.models.encoder.lora_adapter import (
    MODULE_TYPE_SUFFIX,
    LoRASwinViT,
    _find_lora_targets,
)
from growth.models.encoder.swin_loader import (
    BRAINSEGFOUNDER_DEPTHS,
    load_full_swinunetr,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s  %(message)s"
)
logger = logging.getLogger("verify_extended_lora")

REPORT_PATH = Path(__file__).with_name("verify_extended_lora_report.md")


@dataclass
class ConfigSpec:
    """One row of the verification matrix."""

    label: str
    rank: int
    stages: List[int]
    module_types: List[str]

    def expected_target_count(self) -> int:
        depth = BRAINSEGFOUNDER_DEPTHS[0]
        return len(self.stages) * depth * len(self.module_types)


@dataclass
class ConfigResult:
    """Outcome for one config — one row of the markdown report."""

    spec: ConfigSpec
    discovery_count: int = 0
    expected_count: int = 0
    combinatorial_pass: bool = False
    peft_wired_pairs: int = 0
    trainable_params: int = 0
    expected_trainable_params: int = 0
    forward_shape: str = ""
    backward_ok: bool = False
    peak_memory_mb: float = float("nan")
    spatial_used: int = 0
    status: str = "PENDING"
    messages: List[str] = field(default_factory=list)


def _reclaim_gpu(device: torch.device) -> None:
    """Force Python GC and CUDA cache release between configs."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


VERIFICATION_MATRIX: List[ConfigSpec] = [
    ConfigSpec("stages34_qkv_r8", 8, [3, 4], ["qkv"]),
    ConfigSpec("stages123_qkv_r8", 8, [1, 2, 3], ["qkv"]),
    ConfigSpec("stages123_extended_r2", 2, [1, 2, 3], ["qkv", "proj", "fc1", "fc2"]),
    ConfigSpec("stages123_extended_r8", 8, [1, 2, 3], ["qkv", "proj", "fc1", "fc2"]),
    ConfigSpec("stages123_extended_r32", 32, [1, 2, 3], ["qkv", "proj", "fc1", "fc2"]),
    # stages 1-4 variants: include stage 4 so its weights can adapt to the
    # modulated features arriving from stage 3 LoRA (closes a within-network
    # distribution shift). Cost is small at stage 4 (4^3 tokens) but
    # parameter count scales because layers4 has dim=768.
    ConfigSpec("stages1234_extended_r2", 2, [1, 2, 3, 4], ["qkv", "proj", "fc1", "fc2"]),
    ConfigSpec("stages1234_extended_r8", 8, [1, 2, 3, 4], ["qkv", "proj", "fc1", "fc2"]),
    ConfigSpec("stages1234_extended_r32", 32, [1, 2, 3, 4], ["qkv", "proj", "fc1", "fc2"]),
]


def _enumerate_reference_linears(reference_model: torch.nn.Module) -> Dict[str, torch.nn.Linear]:
    """Map relative module name → Linear, from a pristine (non-PEFT) model."""
    out: Dict[str, torch.nn.Linear] = {}
    for name, mod in reference_model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            rel = name[len("swinViT."):] if name.startswith("swinViT.") else name
            out[rel] = mod
    return out


def _run_one(
    spec: ConfigSpec,
    checkpoint_path: Path,
    device: torch.device,
    spatial_candidates: List[int],
) -> ConfigResult:
    """Execute all checks for one spec and return a populated result.

    ``spatial_candidates`` is tried in order for the GPU forward/backward
    phase. If the first spatial value OOMs, fall back to the next, and so
    on. Correctness checks (discovery, wiring, param-count) are
    spatial-independent.
    """
    res = ConfigResult(spec=spec)
    res.expected_count = spec.expected_target_count()

    t0 = time.time()
    logger.info("=" * 72)
    logger.info(
        "Config %s: rank=%d stages=%s types=%s (expected targets=%d)",
        spec.label, spec.rank, spec.stages, spec.module_types, res.expected_count,
    )

    # --- Correctness phase on CPU --------------------------------------
    reference = load_full_swinunetr(
        checkpoint_path, freeze_encoder=True, freeze_decoder=False,
        out_channels=3, device="cpu",
    )
    linears = _enumerate_reference_linears(reference)

    # A. discovery count
    discovered = _find_lora_targets(
        reference, stages=spec.stages, module_types=spec.module_types,
    )
    res.discovery_count = len(discovered)
    if res.discovery_count != res.expected_count:
        res.status = "FAIL"
        res.messages.append(
            f"Discovery count mismatch: expected {res.expected_count}, "
            f"got {res.discovery_count}."
        )
        return res

    # B. combinatorial presence
    expected_names = {
        f"layers{stage}.0.blocks.{block}{MODULE_TYPE_SUFFIX[t]}"
        for stage in spec.stages
        for block in range(BRAINSEGFOUNDER_DEPTHS[0])
        for t in spec.module_types
    }
    discovered_set = set(discovered)
    if discovered_set != expected_names:
        res.status = "FAIL"
        missing = expected_names - discovered_set
        extra = discovered_set - expected_names
        res.messages.append(
            f"Combinatorial mismatch: missing={sorted(missing)[:5]} "
            f"extra={sorted(extra)[:5]}"
        )
        return res
    res.combinatorial_pass = True

    # Compute expected trainable params using reference Linear shapes.
    res.expected_trainable_params = sum(
        spec.rank * (linears[name].in_features + linears[name].out_features)
        for name in expected_names
    )

    # Build a fresh model and wrap it — PEFT renames wrapped Linears.
    base = load_full_swinunetr(
        checkpoint_path, freeze_encoder=True, freeze_decoder=False,
        out_channels=3, device="cpu",
    )
    lora = LoRASwinViT(
        base,
        rank=spec.rank,
        alpha=2 * spec.rank,  # keep alpha/r = 2.0
        target_stages=spec.stages,
        target_module_types=spec.module_types,
    )

    # C. PEFT wiring count
    res.peft_wired_pairs = sum(
        1 for n, _ in lora.model.named_modules() if n.endswith(".lora_A.default")
    )
    lora_b_count = sum(
        1 for n, _ in lora.model.named_modules() if n.endswith(".lora_B.default")
    )
    if res.peft_wired_pairs != res.expected_count or lora_b_count != res.expected_count:
        res.status = "FAIL"
        res.messages.append(
            f"PEFT wiring mismatch: lora_A={res.peft_wired_pairs}, "
            f"lora_B={lora_b_count}, expected {res.expected_count}."
        )
        return res

    # D. trainable parameter count
    res.trainable_params = lora.get_trainable_params()
    if res.trainable_params != res.expected_trainable_params:
        res.status = "FAIL"
        res.messages.append(
            f"Trainable-param mismatch: expected {res.expected_trainable_params}, "
            f"got {res.trainable_params}."
        )
        return res

    # --- GPU forward/backward phase (with OOM guard + spatial fallback) ---
    if device.type != "cuda":
        res.forward_shape = "skipped (CPU)"
        res.status = "PASS"
        return res

    lora.to(device)
    last_oom_msg: str | None = None
    succeeded_spatial: int | None = None
    for spatial in spatial_candidates:
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats(device)
            lora.train()
            lora.zero_grad(set_to_none=True)
            x = torch.randn(1, 4, spatial, spatial, spatial, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = lora(x)
            res.forward_shape = str(tuple(out.shape))
            if tuple(out.shape) != (1, 3, spatial, spatial, spatial):
                res.status = "FAIL"
                res.messages.append(
                    f"Unexpected output shape: got {res.forward_shape}."
                )
                break
            loss = out.mean()
            loss.backward()

            # Gradients must exist on LoRA params and be None everywhere else.
            any_lora_grad = False
            leaked_grad = False
            for name, param in lora.model.named_parameters():
                if "lora_" in name:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        any_lora_grad = True
                else:
                    if (
                        param.requires_grad
                        and param.grad is not None
                        and param.grad.abs().sum() > 0
                    ):
                        leaked_grad = True
            if not any_lora_grad or leaked_grad:
                res.status = "FAIL"
                res.messages.append(
                    f"Gradient routing incorrect: any_lora_grad={any_lora_grad}, "
                    f"leaked_grad={leaked_grad}"
                )
                break
            res.backward_ok = True
            res.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            succeeded_spatial = spatial
            break
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            last_oom_msg = f"OOM at spatial={spatial}: {exc}"
            logger.warning("[%s] %s", spec.label, last_oom_msg)
            # Drop any partially-allocated state before the next attempt.
            try:
                del x, out, loss  # noqa: F821 (may not exist on very early OOM)
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
            continue

    del lora, base, reference, linears
    _reclaim_gpu(device)

    if res.status == "FAIL":
        return res
    if succeeded_spatial is None:
        res.status = "OOM"
        if last_oom_msg is not None:
            res.messages.append(last_oom_msg)
        res.messages.append(
            "Correctness checks passed; skipped functional forward/backward due"
            " to 8 GB VRAM constraint. This config is designed for Picasso A100."
        )
        return res

    res.spatial_used = succeeded_spatial
    res.status = "PASS"
    if succeeded_spatial != spatial_candidates[0]:
        res.messages.append(
            f"Functional check ran at spatial={succeeded_spatial}^3 (fallback"
            f" from {spatial_candidates[0]}^3 due to local VRAM)."
        )
    logger.info(
        "%s: %s in %.1fs (peak=%.1f MB at spatial=%d, targets=%d, params=%d)",
        spec.label, res.status, time.time() - t0, res.peak_memory_mb,
        succeeded_spatial, res.discovery_count, res.trainable_params,
    )
    return res


def _write_report(results: List[ConfigResult], device: torch.device) -> None:
    """Emit a compact markdown report next to the script."""
    lines: List[str] = [
        "# Extended-LoRA verification report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Device: `{device}`  ",
        "GPU phase: fp16 autocast, batch=1. Spatial extent is tried from the",
        "largest candidate down; first one that fits wins.",
        "",
        "| Label | r | stages | types | targets | wired | params | spatial | fwd shape | grads | peak MB | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            "| {label} | {rank} | {stages} | {types} | {tgt}/{exp_tgt} | {wired} | "
            "{params} | {spatial} | {fwd} | {grads} | {mem} | **{status}** |".format(
                label=r.spec.label,
                rank=r.spec.rank,
                stages=",".join(str(s) for s in r.spec.stages),
                types=",".join(r.spec.module_types),
                tgt=r.discovery_count,
                exp_tgt=r.expected_count,
                wired=r.peft_wired_pairs,
                params=f"{r.trainable_params:,}",
                spatial=(
                    f"{r.spatial_used}^3" if r.spatial_used else "-"
                ),
                fwd=r.forward_shape or "-",
                grads="ok" if r.backward_ok else "-",
                mem=(
                    f"{r.peak_memory_mb:.0f}"
                    if r.peak_memory_mb == r.peak_memory_mb  # not NaN
                    else "-"
                ),
                status=r.status,
            )
        )
    # Messages block.
    notes = [r for r in results if r.messages]
    if notes:
        lines.append("")
        lines.append("## Notes")
        for r in notes:
            lines.append(f"- **{r.spec.label}** ({r.status}): " + "; ".join(r.messages))

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", REPORT_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify extended LoRA targets locally.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to BrainSegFounder checkpoint (.pt).",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="PyTorch device (cpu or cuda:N).",
    )
    parser.add_argument(
        "--spatial",
        type=int,
        nargs="+",
        default=[128, 96, 64],
        help=(
            "Input spatial extents (cubed) to try for the GPU forward/backward"
            " check. Tried in order; on OOM, fall back to the next. Default"
            " tries 128^3 (training ROI), then 96^3 and 64^3."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint_path)
        return 2

    device = torch.device(args.device)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s (%.1f GB)", props.name, props.total_memory / (1024 ** 3),
        )

    results: List[ConfigResult] = []
    for spec in VERIFICATION_MATRIX:
        _reclaim_gpu(device)
        results.append(
            _run_one(spec, args.checkpoint_path, device, spatial_candidates=args.spatial)
        )

    _write_report(results, device)

    # Non-zero exit if any correctness (non-OOM) failure.
    fatal = [r for r in results if r.status == "FAIL"]
    if fatal:
        logger.error("%d configuration(s) failed correctness checks.", len(fatal))
        return 1
    logger.info("All correctness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
