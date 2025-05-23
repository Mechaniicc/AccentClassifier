# English Accent Classifier

A streamlined application that detects and classifies English accents from audio using `ylacombe/accent-classifier` model from Hugging Face. The app can process audio from YouTube videos or direct MP4 links.

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
AccentClassifier/
├── Model/
│   └── model.py              # Accent detection model implementation
├── VideoProcessor/
│   └── VP.py                 # Video processing and audio extraction
├── Audio/                    # Temporary audio storage (auto-cleaned)
├── data/                     # Data for training (work in progress)
├── Modeling/
│   └── train.py              # Training script (work in progress)
└── app.py                    # Streamlit web interface
```

## 🎯 Features

### Accent Detection
- Powered by `ylacombe/accent-classifier` from Hugging Face
- Supports multiple English accents:
  - American 🇺🇸
  - British 🇬🇧
  - Australian 🇦🇺
  - Indian 🇮🇳
  - Canadian 🇨🇦
  - Irish 🇮🇪
  - Scottish 🏴󠁧󠁢󠁳󠁣󠁴󠁿
  - South African 🇿🇦
  - And more!

### Video Processing
- YouTube video support
- Direct MP4 link support
- Automatic audio extraction
- Smart file cleanup
- Progress tracking

### User Interface
- Clean, modern Streamlit interface
- Real-time processing feedback
- Interactive visualizations
- Detailed accent analysis
- Confidence scores

### Training Capability (Work in Progress)
The project includes a training script (`Modeling/train.py`) that allows you to:
- Train on your own accent dataset
- Fine-tune the model parameters
- Customize accent categories
- Use data augmentation
- Monitor training metrics

Note: The training functionality is currently under development and may require additional setup and dependencies.

## 🔧 Components

### Model (`model.py`)
- Uses Wav2Vec2-based accent classifier
- GPU acceleration when available
- Confidence scoring
- English accent verification
- Batch processing support

### Video Processor (`VP.py`)
- Automatic file cleanup
- Multiple format support
- Progress tracking
- Resource management
- Error handling

### Web Interface (`app.py`)
- Real-time accent detection
- Beautiful visualizations
- Accent characteristics
- Confidence metrics
- Debug information

## 📊 Output Metrics

The classifier provides:
- Primary accent detection
- Confidence scores
- Secondary accent detection
- Accent clarity score
- Detailed accent characteristics
- Visualization of top predictions

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- yt-dlp
- librosa
- moviepy
- numpy
- matplotlib

## 🎵 Supported Formats

Input video sources:
- YouTube URLs
- Direct MP4 links


## 🔍 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Enter a video URL:
   - YouTube link (e.g., https://www.youtube.com/watch?v=...)
   - Direct MP4 link (e.g., https://example.com/video.mp4)

3. Click "Detect Accent" and wait for:
   - Video download
   - Audio extraction
   - Accent analysis
   - Results visualization

## 🎯 Best Practices

For optimal results:
- Use clear speech audio
- Minimize background noise
- Provide 5-30 seconds of speech
- Ensure good audio quality
- Use native speaker samples

## 🐛 Troubleshooting

### Common Issues

**Video Download Fails**
- Check URL validity
- Verify internet connection
- Ensure video is accessible
- Make sure URL is YouTube or direct MP4

**Audio Processing Issues**
- Check disk space
- Verify file permissions
- Check supported formats

**Model Performance**
- Ensure GPU availability (if applicable)
- Check memory usage
- Verify model downloads

## 📈 Performance Notes

- GPU acceleration supported
- Automatic CPU fallback
- ~4GB RAM required
- ~2GB disk space needed
- Processing time: 5-30 seconds

## 🔒 Privacy & Data

- No audio data is stored permanently
- Automatic file cleanup
- Local processing only
- No data sent to external servers

## 📝 License

This project uses the following components:
- `ylacombe/accent-classifier`: [CC-BY-NC-4.0 License](https://huggingface.co/ylacombe/accent-classifier)
- Other dependencies: See respective licenses

---

**Note**: This project is for educational and research purposes. Accent detection should not be used for discrimination or bias. 