# src/growth/utils/model_card.py
"""Model card generation for LoRA adapters.

Creates HuggingFace-compatible model cards with training metadata.
PEFT's save_pretrained() will merge its own metadata into these cards.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import ModelCard, ModelCardData

logger = logging.getLogger(__name__)

TEMPLATE = """---
{{ card_data }}
---

# {{ model_name }}

{{ description }}

## Model Details

- **Base Model**: {{ base_model }}
- **Adaptation**: LoRA (Low-Rank Adaptation)
- **Task**: {{ task }}

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {{ lora_rank }} |
| Alpha | {{ lora_alpha }} |
| Dropout | {{ lora_dropout }} |
| Target Modules | {{ target_modules }} |
| Trainable Parameters | {{ trainable_params }} |

## Training Details

### Dataset

- **Name**: {{ dataset_name }}
- **Train Samples**: {{ train_samples }}
- **Validation Samples**: {{ val_samples }}

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {{ epochs }} |
| Batch Size | {{ batch_size }} |
| Learning Rate (LoRA) | {{ lr_encoder }} |
| Learning Rate (Decoder) | {{ lr_decoder }} |
| Weight Decay | {{ weight_decay }} |
| Early Stopping Patience | {{ patience }} |

### Training Metrics

| Metric | Value |
|--------|-------|
| Best Validation Dice | {{ best_val_dice }} |
| Final Training Loss | {{ final_train_loss }} |
| Training Time | {{ training_time }} |

## Hardware

- **Device**: {{ device }}
- **Training Date**: {{ training_date }}

## Reproducibility

- **Random Seed**: {{ seed }}
{% if git_commit %}- **Git Commit**: {{ git_commit }}{% endif %}

## Usage

```python
from peft import PeftModel
from growth.models.encoder.swin_loader import load_swin_encoder

# Load base encoder
base_encoder = load_swin_encoder("path/to/checkpoint.pt")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_encoder, "path/to/adapter")
```

### Framework Versions
"""


@dataclass
class LoRAModelCardConfig:
    """Configuration for LoRA model card generation."""

    # Model info
    model_name: str = "LoRA-adapted SwinViT Encoder"
    description: str = "SwinViT encoder adapted for meningioma segmentation using LoRA."
    base_model: str = "BrainSegFounder"
    task: str = "3D Medical Image Segmentation"

    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: str = "layers3, layers4 (Q, K, V)"
    trainable_params: str = "~37K"

    # Dataset
    dataset_name: str = "BraTS-MEN 2024"
    train_samples: int = 0
    val_samples: int = 0

    # Training config
    epochs: int = 0
    batch_size: int = 4
    lr_encoder: float = 1e-4
    lr_decoder: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 10

    # Metrics
    best_val_dice: str = "N/A"
    final_train_loss: str = "N/A"
    training_time: str = "N/A"

    # Hardware/reproducibility
    device: str = "N/A"
    seed: int = 42
    git_commit: Optional[str] = None
    training_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    # HuggingFace metadata
    license: str = "mit"
    tags: List[str] = field(
        default_factory=lambda: ["medical-imaging", "meningioma", "swin-transformer", "lora"]
    )


def create_lora_model_card(config: LoRAModelCardConfig) -> ModelCard:
    """Create a model card for a LoRA adapter.

    Args:
        config: Model card configuration.

    Returns:
        ModelCard instance ready to save.
    """
    card_data = ModelCardData(
        license=config.license,
        tags=config.tags,
        datasets=[config.dataset_name.lower().replace(" ", "-")],
        base_model=config.base_model,
    )

    card = ModelCard.from_template(
        card_data,
        template_str=TEMPLATE,
        model_name=config.model_name,
        description=config.description,
        base_model=config.base_model,
        task=config.task,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        trainable_params=config.trainable_params,
        dataset_name=config.dataset_name,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr_encoder=config.lr_encoder,
        lr_decoder=config.lr_decoder,
        weight_decay=config.weight_decay,
        patience=config.patience,
        best_val_dice=config.best_val_dice,
        final_train_loss=config.final_train_loss,
        training_time=config.training_time,
        device=config.device,
        seed=config.seed,
        git_commit=config.git_commit,
        training_date=config.training_date,
    )

    return card


def save_lora_model_card(
    output_dir: Union[str, Path],
    config: LoRAModelCardConfig,
) -> Path:
    """Save a model card to a directory.

    Call this BEFORE peft's save_pretrained() so PEFT can merge its metadata.

    Args:
        output_dir: Directory to save README.md.
        config: Model card configuration.

    Returns:
        Path to saved README.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    card = create_lora_model_card(config)
    readme_path = output_dir / "README.md"
    card.save(readme_path)

    logger.info(f"Model card saved to {readme_path}")
    return readme_path


def model_card_from_training(
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_decoder: float,
    best_val_dice: float,
    final_train_loss: float,
    training_time_seconds: float,
    device: str,
    seed: int,
    trainable_params: Optional[int] = None,
    base_model_path: Optional[str] = None,
    condition_name: Optional[str] = None,
    git_commit: Optional[str] = None,
    **kwargs: Any,
) -> LoRAModelCardConfig:
    """Create model card config from training results.

    Convenience function to build config from typical training outputs.

    Args:
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        epochs: Number of epochs trained.
        batch_size: Training batch size.
        lr_encoder: Encoder (LoRA) learning rate.
        lr_decoder: Decoder learning rate.
        best_val_dice: Best validation Dice score.
        final_train_loss: Final training loss.
        training_time_seconds: Total training time in seconds.
        device: Training device (e.g., "cuda:0").
        seed: Random seed.
        trainable_params: Number of trainable parameters.
        base_model_path: Path to base model checkpoint.
        condition_name: Experiment condition name (e.g., "lora_r8").
        git_commit: Git commit hash for reproducibility.
        **kwargs: Additional fields to override.

    Returns:
        LoRAModelCardConfig instance.
    """
    # Format training time
    hours = int(training_time_seconds // 3600)
    minutes = int((training_time_seconds % 3600) // 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m"
    else:
        time_str = f"{minutes}m {int(training_time_seconds % 60)}s"

    # Format trainable params
    if trainable_params is not None:
        if trainable_params >= 1_000_000:
            params_str = f"~{trainable_params / 1_000_000:.1f}M"
        else:
            params_str = f"~{trainable_params / 1_000:.0f}K"
    else:
        params_str = "N/A"

    # Model name
    model_name = "LoRA-adapted SwinViT Encoder"
    if condition_name:
        model_name = f"{model_name} ({condition_name})"

    return LoRAModelCardConfig(
        model_name=model_name,
        base_model=base_model_path or "BrainSegFounder",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        trainable_params=params_str,
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=epochs,
        batch_size=batch_size,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        best_val_dice=f"{best_val_dice:.4f}",
        final_train_loss=f"{final_train_loss:.4f}",
        training_time=time_str,
        device=device,
        seed=seed,
        git_commit=git_commit,
        **kwargs,
    )
