import streamlit as st
from VideoProcessor.VP import VideoProcessor
from Model.model import AccentDetector
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="English Accent Classifier",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize model
@st.cache_resource
def load_model():
    return AccentDetector()


def main():
    st.title("🎤 English Accent Classifier")
    st.subheader("Detect English accents from YouTube videos or direct video links")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a Wav2Vec2-based accent classifier to detect various English accents from audio.
        
        **Features:**
        - Supports YouTube URLs and direct video links
        - Detects multiple accent types
        - Provides confidence scores
        - Real-time audio processing
        - Visualization
        
        **Supported Accents:**
        - American 🇺🇸
        - British 🇬🇧
        - Australian 🇦🇺
        - Indian 🇮🇳
        - Canadian 🇨🇦
        - Irish 🇮🇪
        - Scottish 🏴󠁧󠁢󠁳󠁣󠁴󠁿
        - South African 🇿🇦
        And more!
        """)

        st.divider()
        st.markdown("### Model Information")
        st.markdown("""
        Using a fine-tuned model for accent detection.
        
        The model analyzes speech patterns, pronunciation, and acoustic features to identify accents.
        
        Note: The model is still a work in progress and may not be 100% accurate.
        
        For best results:
        - Use clear audio
        - Minimize background noise
        - Provide 5-30 seconds of speech
        """)

    # Load model
    with st.spinner("Loading accent detection model..."):
        model = load_model()

    # URL Input
    st.markdown("### Enter Video URL")
    url = st.text_input(
        "YouTube URL or direct video link:",
        placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
    )

    if st.button("🎯 Detect Accent", type="primary") and url:
        try:
            # Process video
            with st.spinner("Downloading video and extracting audio..."):
                processor = VideoProcessor()
                audio_path = processor.process_video(url)
                st.success("✅ Audio extracted successfully!")

            # Display audio player
            st.markdown("### 🎵 Extracted Audio")
            st.audio(audio_path)

            with st.spinner("🔍 Analyzing accent..."):
                # Detect accent
                result = model.detect_accent(audio_path)

                # Display results
                st.markdown("### 🎯 Results")
                col1, col2 = st.columns(2)

                with col1:
                    # Main result display
                    st.markdown(f"### Detected Accent: **{result['accent']}**")
                    st.markdown(f"**Confidence:** {result['confidence']:.2f}%")

                    # Confidence indicator
                    if result["confidence"] >= 70:
                        st.success("🎯 High confidence prediction")
                    elif result["confidence"] >= 40:
                        st.warning("⚠️ Medium confidence prediction")
                    else:
                        st.error("❌ Low confidence prediction")

                    if result["is_english"]:
                        st.info("✓ Confirmed English accent")
                    else:
                        st.warning("⚠️ May not be a native English accent")

                    # Accent characteristics
                    st.markdown("### 📝 Accent Characteristics")
                    accent_descriptions = {
                        "American": "General American accent (GenAm) characterized by rhotic pronunciation and T-flapping.",
                        "British": "Received Pronunciation (RP) features with non-rhotic pronunciation and distinct vowel sounds.",
                        "Australian": "Non-rhotic with distinctive vowel shifts and rising intonation patterns.",
                        "Indian": "Syllable-timed rhythm with retroflex consonants and unique stress patterns.",
                        "Canadian": "Similar to GenAm but with Canadian raising and British influences.",
                        "South African": "Influenced by Afrikaans, with distinctive vowel sounds and intonation.",
                        "Irish": "Melodic intonation with Gaelic influences and specific consonant variations.",
                        "Scottish": "Strong rhotic pronunciation with distinct vowel system and Scots influences.",
                        "Welsh": "Distinctive rising and falling intonation with Welsh language influences.",
                        "New Zealand": "Similar to Australian but with higher vowels and specific vowel shifts.",
                        "Jamaican": "Influenced by Patois, with distinct rhythm and pronunciation patterns.",
                        "Singaporean": "Influenced by Chinese and Malay, with unique stress and tone patterns.",
                        "Malaysian": "Mix of British English with local language influences.",
                        "Filipino": "Influenced by Spanish and Tagalog, with specific stress patterns.",
                        "Other": "Accent characteristics not matching the trained categories or mixed accents.",
                    }
                    st.markdown(
                        accent_descriptions.get(
                            result["accent"],
                            "Unique accent pattern with mixed characteristics.",
                        )
                    )

                with col2:
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Sort scores for better visualization
                    scores = result["scores"]
                    sorted_scores = dict(
                        sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    )

                    # Take top 12 accents for cleaner visualization
                    top_accents = dict(list(sorted_scores.items())[:12])
                    accents = list(top_accents.keys())
                    values = list(top_accents.values())

                    # Color scheme
                    colors = [
                        "#3498db"
                        if accent == result["accent"]
                        else "#2ecc71"
                        if value > 30
                        else "#e74c3c"
                        if value < 10
                        else "#f1c40f"
                        for accent, value in zip(accents, values)
                    ]

                    y_pos = np.arange(len(accents))
                    ax.barh(y_pos, values, color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(accents)
                    ax.invert_yaxis()
                    ax.set_xlabel("Confidence (%)")
                    ax.set_title("Top Accent Predictions")
                    ax.set_xlim(0, max(values) + 5)

                    # Add value labels
                    for i, v in enumerate(values):
                        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

                    # Add grid
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()

                    # Display plot
                    st.pyplot(fig)

                # Additional metrics
                st.markdown("### 📊 Analysis Summary")
                col3, col4, col5 = st.columns(3)

                with col3:
                    st.metric(
                        "Primary Accent",
                        result["accent"],
                        f"{result['confidence']:.1f}%",
                    )

                with col4:
                    # Get second highest prediction
                    second_accent = list(sorted_scores.items())[1]
                    st.metric(
                        "Secondary Accent", second_accent[0], f"{second_accent[1]:.1f}%"
                    )

                with col5:
                    # Calculate accent clarity (difference between top two predictions)
                    clarity = result["confidence"] - second_accent[1]
                    st.metric(
                        "Accent Clarity",
                        f"{clarity:.1f}%",
                        "Distinct" if clarity > 20 else "Mixed",
                    )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please check if the URL is valid and accessible.")

            # Debug information in expandable section
            with st.expander("🔧 Debug Information"):
                st.text(f"URL provided: {url}")
                import traceback

                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
