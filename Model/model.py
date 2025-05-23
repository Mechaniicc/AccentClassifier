import torch
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import numpy as np
import librosa
from typing import Dict, List, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class AccentDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the AccentDetector using the ylacombe/accent-classifier model.

        Args:
            model_path (str, optional): Not used in this implementation as we use the pre-trained HF model.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model identifier for the accent classifier
        self.model_name = "ylacombe/accent-classifier"

        try:
            # Initialize the audio classification pipeline
            print(f"Loading accent classifier model: {self.model_name}")
            self.classifier = pipeline(
                "audio-classification",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True,
            )

            # Load model and processor separately for more control if needed
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            # Get label mappings from the model config
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            self.labels = list(self.label2id.keys())

            print("Model loaded successfully!")
            print(f"Available accent labels: {self.labels}")

        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to default labels if model loading fails
            self.labels = [
                "american",
                "british",
                "australian",
                "indian",
                "canadian",
                "south_african",
                "irish",
                "scottish",
            ]
            self.label2id = {label: i for i, label in enumerate(self.labels)}
            self.id2label = {i: label for i, label in enumerate(self.labels)}
            print("Using fallback labels")

    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for accent detection.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Preprocessed audio array.
        """
        try:
            # Load audio using librosa (more robust than torchaudio for various formats)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Ensure audio is not too short (minimum 1 second)
            min_length = 16000  # 1 second at 16kHz
            if len(audio) < min_length:
                # Pad audio if too short
                audio = np.pad(audio, (0, min_length - len(audio)), mode="constant")

            # Limit to maximum 30 seconds to avoid memory issues
            max_length = 16000 * 30  # 30 seconds
            if len(audio) > max_length:
                audio = audio[:max_length]

            # Normalize audio
            audio = audio / np.max(np.abs(audio) + 1e-8)

            return audio

        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            # Return silence as fallback
            return np.zeros(16000, dtype=np.float32)

    def detect_accent(
        self, audio_path: str
    ) -> Dict[str, Union[str, float, Dict[str, float], bool]]:
        """
        Detect the accent of an audio file using the ylacombe/accent-classifier model.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Dict: Dictionary containing:
                - accent: Predicted accent name
                - confidence: Confidence score (0-100)
                - scores: Dictionary of all accent scores
                - is_english: Boolean indicating if it's an English accent
        """
        try:
            # Preprocess audio
            audio_array = self._preprocess_audio(audio_path)

            # Get predictions using the pipeline
            predictions = self.classifier(audio_array)

            # Process results
            accent_scores = {}
            max_score = 0
            predicted_accent = "unknown"

            for pred in predictions:
                label = pred["label"].lower()
                score = pred["score"] * 100  # Convert to percentage
                accent_scores[label] = score

                if score > max_score:
                    max_score = score
                    predicted_accent = label

            # Clean up accent name for display
            display_accent = self._format_accent_name(predicted_accent)

            # Determine if it's an English accent
            is_english = self._is_english_accent(predicted_accent, max_score)

            return {
                "accent": display_accent,
                "confidence": round(max_score, 2),
                "scores": {
                    self._format_accent_name(k): round(v, 2)
                    for k, v in accent_scores.items()
                },
                "is_english": is_english,
            }

        except Exception as e:
            print(f"Error during accent detection: {e}")
            return {
                "accent": "Unknown",
                "confidence": 0.0,
                "scores": {"Unknown": 0.0},
                "is_english": False,
            }

    def _format_accent_name(self, accent: str) -> str:
        """
        Format accent name for better display.

        Args:
            accent (str): Raw accent name from model.

        Returns:
            str: Formatted accent name.
        """
        # Handle common accent name formats
        accent_mapping = {
            "american": "American",
            "british": "British",
            "australian": "Australian",
            "indian": "Indian",
            "canadian": "Canadian",
            "south_african": "South African",
            "irish": "Irish",
            "scottish": "Scottish",
            "welsh": "Welsh",
            "new_zealand": "New Zealand",
            "nigerian": "Nigerian",
            "jamaican": "Jamaican",
            "filipino": "Filipino",
            "malaysian": "Malaysian",
            "singaporean": "Singaporean",
        }

        accent_lower = accent.lower().replace("-", "_").replace(" ", "_")
        return accent_mapping.get(accent_lower, accent.title())

    def _is_english_accent(self, accent: str, confidence: float) -> bool:
        """
        Determine if the detected accent is considered an English accent.

        Args:
            accent (str): Detected accent name.
            confidence (float): Confidence score.

        Returns:
            bool: True if it's considered an English accent.
        """
        # Define which accents are considered English accents
        english_accents = {
            "american",
            "british",
            "australian",
            "canadian",
            "irish",
            "scottish",
            "welsh",
            "new_zealand",
            "south_african",
            "indian",  # Indian English is also English
        }

        accent_lower = accent.lower().replace("-", "_").replace(" ", "_")

        # Consider it English if it's in our English accents list and confidence > 30%
        return accent_lower in english_accents and confidence > 30.0

    def get_supported_accents(self) -> List[str]:
        """
        Get list of supported accent labels.

        Returns:
            List[str]: List of supported accent names.
        """
        return [self._format_accent_name(label) for label in self.labels]

    def batch_detect_accents(self, audio_paths: List[str]) -> List[Dict]:
        """
        Detect accents for multiple audio files.

        Args:
            audio_paths (List[str]): List of audio file paths.

        Returns:
            List[Dict]: List of detection results for each file.
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.detect_accent(audio_path)
                result["file_path"] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append(
                    {
                        "file_path": audio_path,
                        "accent": "Error",
                        "confidence": 0.0,
                        "scores": {},
                        "is_english": False,
                    }
                )
        return results
