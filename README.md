# StepMania Trance ML Project

A machine learning system for predicting stepchart measures from musical features extracted from trance music audio files.

## 🎵 Project Overview

This project analyzes StepMania stepcharts and their corresponding audio files to build ML models that can predict step patterns based on musical features. The system focuses on trance music with 16th note granularity for precise rhythmic analysis.

## 🏗️ Architecture

### Core Components

- **`musical_features.py`**: Extracts musical features from .ogg audio files
- **`process.py`**: Parses StepMania .sm files into JSON format
- **`clean.py`**: Cleans and standardizes stepchart data
- **`profile.py`**: Analyzes stepchart character usage patterns
- **`distro.py`**: Analyzes difficulty distribution with ban list support

### Feature Extraction

The musical feature extraction system provides:

- **16th note granularity**: 16 positions per 4-beat measure
- **Dual offset handling**: Song offset + pre-offset for waveform swell
- **Comprehensive features**: Energy, onset detection, frequency bands, spectral analysis
- **Efficient storage**: Array-based JSON structure for ML training

## 🚀 Quick Start

### Prerequisites

```bash
pip install librosa scipy numpy
```

### Extract Musical Features

```python
from src.feature_processing.musical_features import MusicalFeatureExtractor

# Create extractor with song metadata
extractor = MusicalFeatureExtractor(
    bpm=140.0,
    song_offset=-0.068,  # From .sm file
    pre_offset_ms=8.0    # Waveform swell capture
)

# Extract features for multiple measures
features = extractor.extract_song_features("song.ogg", num_measures=10)
extractor.save_features(features, "output/features.json")
```

### Process Stepcharts

```python
from src.feature_processing.process import process_sm_files

# Process all .sm files in a directory
process_sm_files("stepfile_assets/", "output/processed/")
```

## 📊 Feature Structure

Each measure contains 24 features with 16-position arrays:

```json
{
  "measure_index": 0,
  "measure_start_time": 0.0,
  "rms_energy": [0.309, 0.168, 0.104, ...],      // 16 values
  "bass_energy": [0.286, 0.081, 0.044, ...],     // 16 values
  "onset_strength": [7.656, 4.489, 8.528, ...],  // 16 values
  "beat_correlation": [1.0, 0.3, 0.8, ...],      // 16 values
  // ... 20 more feature arrays
  "measure_total_energy": 2634.94,
  "measure_rhythm_complexity": 4.13
}
```

## 🎯 Musical Features

### Energy Features
- **RMS Energy**: Root mean square energy at each position
- **Peak Energy**: Maximum amplitude at each position
- **Energy Ratio**: Peak to RMS energy ratio

### Onset Detection
- **Onset Strength**: Rhythmic change detection
- **Onset Peak**: Clear onset peak detection

### Frequency Band Analysis
- **Bass Energy**: 60-250 Hz (bass lines)
- **Kick Energy**: 60-80 Hz (kick drums)
- **Hi-hat Energy**: 8-12 kHz (hi-hats)
- **Snare Energy**: 200-400 Hz (snares)

### Spectral Features
- **Spectral Centroid**: Brightness of sound
- **Spectral Rolloff**: Frequency distribution
- **Spectral Bandwidth**: Frequency spread
- **Spectral Contrast**: Harmonic content

### Timing Features
- **Timing Offset**: Deviation from beat center
- **Beat Correlation**: 4/4 time signature pattern
- **Zero-crossing Rate**: Percussive content

## 📁 Project Structure

```
sm-trance-ml/
├── src/feature_processing/
│   ├── musical_features.py    # Audio feature extraction
│   ├── process.py            # .sm file processing
│   ├── clean.py              # Stepchart cleaning
│   ├── profile.py            # Character profiling
│   └── distro.py             # Difficulty analysis
├── stepfile_assets/          # StepMania files
│   ├── TranceMania/
│   ├── TranceMania2/
│   └── TranceMania3/
├── output/                   # Generated data
└── README.md
```

## 🧪 Testing

Test the musical feature extraction:

```bash
python src/feature_processing/musical_features.py
```

This will:
1. Parse BPM and offset from a .sm file
2. Extract features from the corresponding .ogg file
3. Save results to `output/child_musical_features.json`

## 🔧 Configuration

### Offset Handling

The system handles both positive and negative song offsets:

- **Negative offset** (e.g., -0.068s): Adds silence to beginning
- **Positive offset** (e.g., +0.009s): Trims audio from beginning
- **Pre-offset**: Captures waveform swell (default: 8ms)

### BPM and Timing

- **Beat duration**: `60.0 / bpm` seconds per beat
- **16th note duration**: `beat_duration / 4` seconds
- **Measure duration**: `beat_duration * 4` seconds (4 beats)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🎵 Dataset

The project includes trance music stepcharts from:
- TranceMania
- TranceMania2  
- TranceMania3

Each song includes:
- `.sm` stepchart file
- `.ogg` audio file
- Background images (excluded from repo)

## 🔮 Future Work

- [ ] Batch processing for all songs
- [ ] ML model training pipeline
- [ ] Feature visualization tools
- [ ] Real-time step prediction
- [ ] Cross-validation with different genres 