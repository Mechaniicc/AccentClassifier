"""
Accent Classifier Model Training Script (Version 3)
This script trains a Wav2Vec2 model to classify accents in speech audio using the validated.tsv dataset.
Includes more accent categories and proper CUDA usage.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings
import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration Settings
# ============================================================================

# Data paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TSV_FILE = os.path.join(DATA_DIR, "validated.tsv")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")

# Model output path
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Model", "fine_tuned_model_v3")
)

# Improved accent mapping from Common Voice accent descriptions to standardized labels
ACCENT_MAPPING = {
    # American variants
    "united states english": "American",
    "southern united states english": "American",
    "lightly southern": "American",
    # British variants
    "england english": "British",
    "british english / received pronunciation (rp)": "British",
    # Individual accent categories
    "scottish english": "Scottish",
    "canadian english": "Canadian",
    "australian english": "Australian",
    "irish english": "Irish",
    "new zealand english": "New Zealand",
    "south african english": "South African",
    # Indian subcontinent
    "india and south asia (india, pakistan, sri lanka)": "Indian",
    "indian english": "Indian",
    # Other specific accents
    "nigerian accent": "Nigerian",
    "malaysian english": "Malaysian",
    "singapore english": "Singaporean",
    "filipino": "Filipino",
    # Mixed accents - take the first/primary accent
    "united states english,filipino": "Filipino",
    "united states english,southern united states english,lightly southern": "American",
    "australian english,canadian english": "Australian",  # Pick first one
    # Non-native/other
    "l2": "Other",
    "non-native": "Other",
    "lithuanian,non-native": "Other",
    "russian": "Other",
}

# Training parameters - Optimized for better performance
EPOCHS = 10  # Increased epochs for better convergence
LEARNING_RATE = 2e-5  # Slightly lower for more stable training
TRAIN_BATCH_SIZE = 8  # Increased batch size for better gradient estimates
EVAL_BATCH_SIZE = 8
TEST_SIZE = 0.15  # Reduced to have more training data
MAX_SAMPLES_PER_ACCENT = 1000  # Increased for better representation
MIN_SAMPLES_PER_ACCENT = 10  # Increased minimum for more robust training
WARMUP_STEPS = 0.1  # 10% of training for warmup
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping for stability
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs

# ============================================================================
# Data Augmentation Functions
# ============================================================================


class AudioAugmentation:
    """Audio data augmentation techniques for better model robustness"""

    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def add_noise(self, audio, noise_factor=0.01):
        """Add random noise to audio"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise

    def time_shift(self, audio, shift_max=0.1):
        """Randomly shift audio in time"""
        shift_amount = int(random.uniform(-shift_max, shift_max) * len(audio))
        if shift_amount > 0:
            return torch.cat([torch.zeros(shift_amount), audio[:-shift_amount]])
        elif shift_amount < 0:
            return torch.cat([audio[-shift_amount:], torch.zeros(-shift_amount)])
        return audio

    def speed_change(self, audio, speed_factor_range=(0.9, 1.1)):
        """Change audio speed (and pitch)"""
        speed_factor = random.uniform(*speed_factor_range)
        if speed_factor == 1.0:
            return audio

        # Simple speed change by resampling
        indices = torch.arange(0, len(audio), speed_factor)
        indices = indices[indices < len(audio)].long()
        return audio[indices]

    def pitch_shift(self, audio, n_steps_range=(-2, 2)):
        """Shift pitch without changing speed"""
        n_steps = random.uniform(*n_steps_range)
        if abs(n_steps) < 0.1:
            return audio

        # Simple pitch shift approximation
        shift_factor = 2 ** (n_steps / 12)
        indices = torch.arange(0, len(audio), shift_factor)
        indices = indices[indices < len(audio)].long()
        return audio[indices]

    def apply_random_augmentation(self, audio, augment_prob=0.3):
        """Apply random augmentation with given probability"""
        if random.random() > augment_prob:
            return audio

        # Choose random augmentation
        augmentations = [
            lambda x: self.add_noise(x, random.uniform(0.005, 0.02)),
            lambda x: self.time_shift(x, random.uniform(0.05, 0.15)),
            lambda x: self.speed_change(x),
            lambda x: self.pitch_shift(x),
        ]

        aug_func = random.choice(augmentations)
        return aug_func(audio)


# ============================================================================
# Data Loading and Preparation
# ============================================================================


def load_and_prepare_data():
    """Load and prepare the dataset from validated.tsv"""
    print("Loading and preparing dataset...")
    print(f"Looking for TSV at: {TSV_FILE}")

    # Load the TSV file
    df = pd.read_csv(TSV_FILE, sep="\t")
    print(f"Loaded {len(df)} rows from TSV file")
    print(f"Columns: {df.columns.tolist()}")

    # Filter for rows that have both accent and path information
    initial_count = len(df)
    df = df[df["accents"].notnull() & df["path"].notnull()]
    print(
        f"After filtering for accent and path: {len(df)} rows ({initial_count - len(df)} removed)"
    )

    # Clean and normalize accent names
    df["accents_clean"] = df["accents"].str.lower().str.strip()

    # Show original accent distribution
    print("\nOriginal accent distribution:")
    original_counts = df["accents"].value_counts()
    for accent, count in original_counts.items():
        print(f"  {accent}: {count}")

    # Map accents to standardized labels
    df["mapped_accent"] = df["accents_clean"].map(ACCENT_MAPPING)

    # Handle unmapped accents
    unmapped_mask = df["mapped_accent"].isnull()
    if unmapped_mask.sum() > 0:
        print(f"\nUnmapped accents found: {unmapped_mask.sum()}")
        unmapped_accents = df[unmapped_mask]["accents_clean"].value_counts()
        print("Top unmapped accents:")
        print(unmapped_accents.head(10))

        # Map unmapped accents to "Other"
        df.loc[unmapped_mask, "mapped_accent"] = "Other"

    # Remove rows where mapped_accent is still null
    df = df[df["mapped_accent"].notnull()]

    # Create full audio file paths
    df["audio_path"] = df["path"].apply(lambda x: os.path.join(CLIPS_DIR, x))

    # Check which audio files exist
    print("Checking audio file existence...")
    exists_mask = df["audio_path"].apply(os.path.exists)
    missing_count = (~exists_mask).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} audio files not found and will be excluded")

    df = df[exists_mask]
    print(f"Final dataset size: {len(df)} samples")

    # Show mapped accent distribution
    accent_counts = df["mapped_accent"].value_counts()
    print("\nMapped accent distribution:")
    for accent, count in accent_counts.items():
        print(f"  {accent}: {count}")

    # Filter accents with minimum samples (lowered threshold)
    valid_accents = accent_counts[accent_counts >= MIN_SAMPLES_PER_ACCENT].index
    df = df[df["mapped_accent"].isin(valid_accents)]
    print(
        f"\nAfter filtering accents with <{MIN_SAMPLES_PER_ACCENT} samples: {len(df)} samples"
    )

    # Show which accents made it through
    surviving_counts = df["mapped_accent"].value_counts()
    print("Accents included in training:")
    for accent, count in surviving_counts.items():
        print(f"  {accent}: {count}")

    # Balance dataset by limiting samples per accent (but allow smaller accents)
    balanced_dfs = []
    for accent in df["mapped_accent"].unique():
        accent_df = df[df["mapped_accent"] == accent]
        if len(accent_df) > MAX_SAMPLES_PER_ACCENT:
            accent_df = accent_df.sample(n=MAX_SAMPLES_PER_ACCENT, random_state=42)
        balanced_dfs.append(accent_df)

    df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"After balancing: {len(df)} samples")

    # Final accent distribution
    final_counts = df["mapped_accent"].value_counts()
    print("\nFinal accent distribution:")
    for accent, count in final_counts.items():
        print(f"  {accent}: {count}")

    return df


# ============================================================================
# Dataset Creation and Audio Processing
# ============================================================================


def create_dataset(df):
    """Create and preprocess the dataset"""
    print("\nCreating dataset...")

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["mapped_accent"])
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    print(f"Label encoding: {id2label}")

    # Create train/test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=42
    )

    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[["audio_path", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["audio_path", "label"]])

    # Rename column and cast to Audio
    train_dataset = train_dataset.rename_column("audio_path", "audio")
    test_dataset = test_dataset.rename_column("audio_path", "audio")

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset, test_dataset, id2label, label2id


def preprocess_function(batch, processor, augmenter=None, is_training=True):
    """Enhanced preprocess audio data with optional augmentation"""
    audio = batch["audio"]
    audio_array = audio["array"]

    # Convert to tensor for augmentation
    if augmenter and is_training:
        audio_tensor = torch.FloatTensor(audio_array)
        audio_tensor = augmenter.apply_random_augmentation(audio_tensor)
        audio_array = audio_tensor.numpy()

    # Normalize audio
    if len(audio_array) > 0:
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

    # Process with processor
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,  # Increased to 10 seconds for more context
        truncation=True,
    )

    batch["input_values"] = inputs.input_values[0]
    if "attention_mask" in inputs:
        batch["attention_mask"] = inputs.attention_mask[0]
    return batch


# ============================================================================
# Data Collator
# ============================================================================


@dataclass
class DataCollatorForAudioClassification:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    sampling_rate: int = 16000

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Extract input values and labels
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["label"] for feature in features]

        # Convert to lists if tensors
        input_values = [
            iv.tolist() if isinstance(iv, torch.Tensor) else iv for iv in input_values
        ]

        # Pad input values with consistent length
        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Add labels
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


# ============================================================================
# Training Functions
# ============================================================================


def compute_metrics(eval_pred):
    """Enhanced compute accuracy and additional metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Basic accuracy
    accuracy = (predictions == labels).mean()

    # Additional metrics using sklearn
    from sklearn.metrics import precision_recall_fscore_support

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
    )

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
    }

    # Add per-class F1 scores
    unique_labels = np.unique(labels)
    for i, label_id in enumerate(unique_labels):
        if i < len(f1_per_class):
            metrics[f"f1_class_{label_id}"] = f1_per_class[i]

    return metrics


def train_model(train_dataset, test_dataset, id2label, label2id):
    """Enhanced train the accent classification model with optimizations"""
    print("\nInitializing enhanced model...")

    # Check and set device explicitly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Initialize augmenter
    augmenter = AudioAugmentation()

    # Enhanced preprocessing with augmentation
    print("Preprocessing datasets with augmentation...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, processor, augmenter, is_training=True),
        remove_columns=["audio"],
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, processor, augmenter=None, is_training=False),
        remove_columns=["audio"],
    )

    # Initialize model with improved architecture
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        classifier_proj_size=512,  # Increased from 256 for better representation
        mask_time_prob=0.05,  # Reduce masking for classification task
        mask_feature_prob=0.0,
        mask_time_length=10,
        mask_feature_length=10,
    )

    # Add dropout for regularization
    model.classifier.dropout = nn.Dropout(0.1)

    # Move model to GPU explicitly
    model.to(device)
    print(f"Model moved to: {next(model.parameters()).device}")

    # Create enhanced data collator
    data_collator = DataCollatorForAudioClassification(
        processor=processor,
        padding=True,
        max_length=16000 * 10,  # Match preprocessing
    )

    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir="./Modeling/checkpoints_v3_enhanced",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        logging_steps=10,
        save_total_limit=3,  # Keep more checkpoints
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",  # Use F1 instead of accuracy
        greater_is_better=True,
        warmup_ratio=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        # Enhanced learning rate scheduling
        lr_scheduler_type="cosine",  # Cosine annealing
        save_safetensors=True,
        # Additional optimizations
        dataloader_num_workers=0,  # Disable multiprocessing to avoid length issues
        max_grad_norm=GRADIENT_CLIP_VALUE,  # Correct parameter name for gradient clipping
    )

    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    # Train the model
    print(f"\nStarting enhanced training on {device}...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(test_dataset)}")
    print(f"Number of classes: {len(label2id)}")

    trainer.train()

    # Evaluate final model
    print("\nEvaluating final model...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return model, processor, trainer


def save_model(model, processor, id2label, label2id):
    """Save the enhanced trained model and components"""
    print(f"\nSaving enhanced model to {MODEL_DIR}...")

    # Create directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the base model and processor
    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    base_model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)

    # Extract and save the classifier with correct dimensions
    projector_dim = model.projector.out_features  # Should be 512 (enhanced)
    classifier_input_dim = model.classifier.in_features  # Should be 512

    print(f"Enhanced projector output dim: {projector_dim}")
    print(f"Enhanced classifier input dim: {classifier_input_dim}")

    # Create classifier with the correct input dimension
    classifier = nn.Linear(classifier_input_dim, len(label2id))

    with torch.no_grad():
        classifier.weight.copy_(model.classifier.weight.cpu())
        classifier.bias.copy_(model.classifier.bias.cpu())

    # Also save the projector (enhanced size)
    projector = nn.Linear(768, projector_dim)  # 768 is Wav2Vec2 hidden size
    with torch.no_grad():
        projector.weight.copy_(model.projector.weight.cpu())
        projector.bias.copy_(model.projector.bias.cpu())

    # Save both components
    torch.save(projector, os.path.join(MODEL_DIR, "projector.pt"))
    torch.save(classifier, os.path.join(MODEL_DIR, "classifier.pt"))

    # Save label mappings
    with open(os.path.join(MODEL_DIR, "label_mappings.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    # Save enhanced training info
    training_info = {
        "model_type": "Enhanced_Wav2Vec2ForSequenceClassification",
        "base_model": "facebook/wav2vec2-base-960h",
        "num_labels": len(label2id),
        "training_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "accents": list(label2id.keys()),
        "min_samples_per_accent": MIN_SAMPLES_PER_ACCENT,
        "max_samples_per_accent": MAX_SAMPLES_PER_ACCENT,
        "projector_dim": projector_dim,
        "classifier_input_dim": classifier_input_dim,
        "data_augmentation": True,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_value": GRADIENT_CLIP_VALUE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_scheduler": "cosine",
        "max_audio_length": 10,  # seconds
    }

    with open(os.path.join(MODEL_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    print("Enhanced model saved successfully!")
    print(f"Files saved in: {MODEL_DIR}")
    print("Model improvements:")
    print(f"  - Increased projector size: {projector_dim}")
    print("  - Data augmentation: Enabled")
    print("  - Enhanced metrics: F1, Precision, Recall")
    print(f"  - Early stopping: {EARLY_STOPPING_PATIENCE} epochs patience")
    print("  - Cosine learning rate scheduling")
    print(f"  - Gradient clipping: {GRADIENT_CLIP_VALUE}")
    print("  - Longer audio context: 10 seconds")


# ============================================================================
# Main Training Pipeline
# ============================================================================


def main():
    """Enhanced main training pipeline with optimizations"""
    print("=" * 80)
    print("Enhanced Accent Classifier Training Script v3")
    print("🚀 WITH PERFORMANCE OPTIMIZATIONS 🚀")
    print("=" * 80)

    # Print CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Print optimization features
    print("\n🔧 ENHANCED FEATURES:")
    print(f"  📈 Increased epochs: {EPOCHS}")
    print("  🧠 Enhanced projector size: 512 (vs 256)")
    print("  🔊 Data augmentation: Enabled")
    print("  📏 Longer audio context: 10 seconds")
    print("  📊 Enhanced metrics: F1, Precision, Recall")
    print(f"  ⏰ Early stopping: {EARLY_STOPPING_PATIENCE} epochs patience")
    print("  📉 Cosine learning rate scheduling")
    print(f"  ✂️ Gradient clipping: {GRADIENT_CLIP_VALUE}")
    print(f"  📚 Larger batch size: {TRAIN_BATCH_SIZE}")
    print(f"  🎯 More training data per accent: up to {MAX_SAMPLES_PER_ACCENT}")
    print("=" * 80)

    try:
        # Load and prepare data
        df = load_and_prepare_data()

        if len(df) == 0:
            raise ValueError("No valid data found. Please check your dataset.")

        # Create datasets
        train_dataset, test_dataset, id2label, label2id = create_dataset(df)

        # Train enhanced model
        model, processor, trainer = train_model(
            train_dataset, test_dataset, id2label, label2id
        )

        # Save model
        save_model(model, processor, id2label, label2id)

        print("\n" + "=" * 80)
        print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"Model saved to: {MODEL_DIR}")
        print(f"Trained on {len(label2id)} accent categories: {list(label2id.keys())}")
        print("✨ Improvements implemented:")
        print("  - Better model architecture")
        print("  - Data augmentation for robustness")
        print("  - Enhanced evaluation metrics")
        print("  - Early stopping to prevent overfitting")
        print("  - Optimized learning rate scheduling")
        print("You can now use this enhanced model in the accent classifier app!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
