#!/usr/bin/env python3
"""
Test script for the updated AccentDetector using ylacombe/accent-classifier
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "Model"))

from Model.model import AccentDetector


def test_accent_detector():
    """Test the AccentDetector functionality"""
    print("=" * 60)
    print("Testing AccentDetector with ylacombe/accent-classifier")
    print("=" * 60)

    try:
        # Initialize the detector
        print("\n1. Initializing AccentDetector...")
        detector = AccentDetector()

        # Test getting supported accents
        print("\n2. Getting supported accents...")
        supported_accents = detector.get_supported_accents()
        print(f"Supported accents: {supported_accents}")

        # Test with a sample audio file (if available)
        print("\n3. Looking for sample audio files...")

        # Common audio file extensions
        audio_extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
        sample_files = []

        # Look for audio files in common directories
        search_dirs = [".", "./audio", "./samples", "./test_audio"]

        for directory in search_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        sample_files.append(os.path.join(directory, file))

        if sample_files:
            print(f"Found audio files: {sample_files[:3]}")  # Show first 3

            # Test with the first audio file
            test_file = sample_files[0]
            print(f"\n4. Testing accent detection with: {test_file}")

            result = detector.detect_accent(test_file)

            print("\nResults:")
            print(f"  Detected Accent: {result['accent']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Is English: {result['is_english']}")
            print("  All Scores:")
            for accent, score in result["scores"].items():
                print(f"    {accent}: {score:.2f}%")

            # Test batch processing if multiple files
            if len(sample_files) > 1:
                print(
                    f"\n5. Testing batch processing with {min(3, len(sample_files))} files..."
                )
                batch_results = detector.batch_detect_accents(sample_files[:3])

                for i, result in enumerate(batch_results):
                    print(
                        f"  File {i + 1}: {result['accent']} ({result['confidence']:.1f}%)"
                    )
        else:
            print("No audio files found for testing.")
            print(
                "To test with your own audio, place .wav, .mp3, or other audio files in:"
            )
            print("  - Current directory")
            print("  - ./audio/ folder")
            print("  - ./samples/ folder")
            print("  - ./test_audio/ folder")

        print("\n" + "=" * 60)
        print("AccentDetector test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_accent_detector()
