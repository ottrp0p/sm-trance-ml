"""
Musical Feature Extraction for StepMania Stepchart Prediction

This module extracts musical features from audio files that correlate with stepchart patterns.
Features are extracted at 16th note granularity (16 positions per 4-beat measure).
"""

import librosa
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicalFeatureExtractor:
    """
    Extracts musical features from audio files for stepchart prediction.
    
    Features are extracted at 16th note granularity with support for:
    - Song offset (from stepfile metadata)
    - Pre-offset (to capture waveform swell)
    - Beat-aligned energy analysis
    - Spectral analysis
    - Onset detection
    - Frequency band analysis
    """
    
    def __init__(self, 
                 bpm: float, 
                 song_offset: float = 0.0,
                 pre_offset_ms: float = 5.0,
                 sample_rate: int = 44100,
                 hop_length: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            bpm: Beats per minute from stepfile
            song_offset: Song offset in seconds from stepfile
            pre_offset_ms: Pre-offset in milliseconds to capture waveform swell
            sample_rate: Audio sample rate
            hop_length: Hop length for librosa analysis
        """
        self.bpm = bpm
        self.song_offset = song_offset
        self.pre_offset = pre_offset_ms / 1000.0  # Convert to seconds
        self.sr = sample_rate
        self.hop_length = hop_length
        
        # Calculate timing constants
        self.beat_duration = 60.0 / bpm  # seconds per beat
        self.sixteenth_duration = self.beat_duration / 4  # seconds per 16th note
        self.measure_duration = self.beat_duration * 4  # seconds per measure (4 beats)
        
        # Beat template for 4/4 time (strong on 1,3; medium on 2,4; weak on off-beats)
        self.beat_template = np.array([
            1.0, 0.3, 0.8, 0.3,  # Beat 1: strong, weak, medium, weak
            0.6, 0.3, 0.8, 0.3,  # Beat 2: medium, weak, medium, weak  
            0.6, 0.3, 0.8, 0.3,  # Beat 3: medium, weak, medium, weak
            0.6, 0.3, 0.8, 0.3   # Beat 4: medium, weak, medium, weak
        ])
        
        logger.info(f"Initialized extractor: BPM={bpm}, offset={song_offset}s, pre_offset={pre_offset_ms}ms")
    
    def extract_measure_features(self, 
                               audio_file: Union[str, Path], 
                               measure_start_time: float,
                               measure_index: int = 0) -> Dict[str, Union[float, List[float]]]:
        """
        Extract all musical features for one 4-beat measure.
        
        Args:
            audio_file: Path to audio file
            measure_start_time: Start time of the measure in seconds
            measure_index: Index of the measure (for logging)
            
        Returns:
            Dictionary of features for the measure with arrays for positional features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # Apply song offset by shifting the audio
            # If song_offset is negative, we need to add silence at the beginning
            # If song_offset is positive, we need to trim from the beginning
            if self.song_offset < 0:
                # Add silence at the beginning
                silence_samples = int(abs(self.song_offset) * sr)
                silence = np.zeros(silence_samples)
                y = np.concatenate([silence, y])
                logger.debug(f"Added {abs(self.song_offset):.3f}s of silence to beginning")
            elif self.song_offset > 0:
                # Trim from the beginning
                trim_samples = int(self.song_offset * sr)
                y = y[trim_samples:]
                logger.debug(f"Trimmed {self.song_offset:.3f}s from beginning")
            
            # Now measure_start_time corresponds directly to the audio
            actual_start_time = measure_start_time
            
            features = {
                'measure_index': measure_index,
                'measure_start_time': measure_start_time
            }
            
            # Initialize arrays for positional features
            position_features = {
                'rms_energy': [],
                'peak_energy': [],
                'energy_ratio': [],
                'onset_strength': [],
                'onset_peak': [],
                'bass_energy': [],
                'kick_energy': [],
                'hihat_energy': [],
                'snare_energy': [],
                'spectral_centroid': [],
                'spectral_rolloff': [],
                'spectral_bandwidth': [],
                'spectral_contrast': [],
                'timing_offset': [],
                'beat_correlation': [],
                'zero_crossing_rate': []
            }
            
            # Extract features for each 16th note position
            for i in range(16):
                pos_features = self._extract_position_features(y, actual_start_time, i)
                
                # Add each feature to the appropriate array
                for feature_name in position_features.keys():
                    feature_key = f'pos_{i:02d}_{feature_name}'
                    position_features[feature_name].append(pos_features.get(feature_key, 0.0))
            
            # Add the arrays to features
            features.update(position_features)
            
            # Add measure-level features
            measure_features = self._extract_measure_level_features(y, actual_start_time)
            features.update(measure_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for measure {measure_index}: {e}")
            return {}
    
    def _extract_position_features(self, 
                                 audio: np.ndarray, 
                                 measure_start: float, 
                                 position: int) -> Dict[str, float]:
        """
        Extract features for a specific 16th note position.
        
        Args:
            audio: Audio data
            measure_start: Start time of the measure
            position: 16th note position (0-15)
            
        Returns:
            Dictionary of features for this position
        """
        # Calculate the beat position
        beat_time = measure_start + (position * self.sixteenth_duration)
        
        # Apply pre-offset to capture waveform swell
        window_start = beat_time - self.pre_offset
        window_end = beat_time + self.sixteenth_duration
        
        # Convert to sample indices
        start_sample = max(0, int(window_start * self.sr))
        end_sample = min(len(audio), int(window_end * self.sr))
        segment = audio[start_sample:end_sample]
        
        features = {}
        prefix = f'pos_{position:02d}'
        
        # Basic energy features
        features[f'{prefix}_rms_energy'] = self._extract_rms_energy(segment)
        features[f'{prefix}_peak_energy'] = self._extract_peak_energy(segment)
        features[f'{prefix}_energy_ratio'] = self._extract_energy_ratio(segment)
        
        # Onset detection
        features[f'{prefix}_onset_strength'] = self._detect_onset_strength(segment)
        features[f'{prefix}_onset_peak'] = self._detect_onset_peak(segment)
        
        # Frequency band analysis
        features[f'{prefix}_bass_energy'] = self._extract_bass_energy(segment)
        features[f'{prefix}_kick_energy'] = self._extract_kick_energy(segment)
        features[f'{prefix}_hihat_energy'] = self._extract_hihat_energy(segment)
        features[f'{prefix}_snare_energy'] = self._extract_snare_energy(segment)
        
        # Spectral features
        features[f'{prefix}_spectral_centroid'] = self._extract_spectral_centroid(segment)
        features[f'{prefix}_spectral_rolloff'] = self._extract_spectral_rolloff(segment)
        features[f'{prefix}_spectral_bandwidth'] = self._extract_spectral_bandwidth(segment)
        features[f'{prefix}_spectral_contrast'] = self._extract_spectral_contrast(segment)
        
        # Timing features
        features[f'{prefix}_timing_offset'] = self._calculate_timing_offset(segment, beat_time)
        features[f'{prefix}_beat_correlation'] = self._calculate_beat_correlation(position)
        
        # Zero-crossing rate (useful for percussive detection)
        features[f'{prefix}_zero_crossing_rate'] = self._extract_zero_crossing_rate(segment)
        
        return features
    
    def _extract_measure_level_features(self, 
                                      audio: np.ndarray, 
                                      measure_start: float) -> Dict[str, float]:
        """
        Extract features that apply to the entire measure.
        
        Args:
            audio: Audio data
            measure_start: Start time of the measure
            
        Returns:
            Dictionary of measure-level features
        """
        # Extract the entire measure
        measure_end = measure_start + self.measure_duration
        start_sample = max(0, int(measure_start * self.sr))
        end_sample = min(len(audio), int(measure_end * self.sr))
        measure_segment = audio[start_sample:end_sample]
        
        features = {}
        
        # Overall energy statistics
        features['measure_total_energy'] = np.sum(measure_segment**2)
        features['measure_energy_variance'] = np.var(measure_segment**2)
        features['measure_energy_std'] = np.std(measure_segment**2)
        
        # Spectral statistics
        if len(measure_segment) > 0:
            features['measure_spectral_centroid_mean'] = librosa.feature.spectral_centroid(
                y=measure_segment, sr=self.sr)[0].mean()
            features['measure_spectral_rolloff_mean'] = librosa.feature.spectral_rolloff(
                y=measure_segment, sr=self.sr)[0].mean()
        else:
            features['measure_spectral_centroid_mean'] = 0.0
            features['measure_spectral_rolloff_mean'] = 0.0
        
        # Rhythm complexity
        features['measure_rhythm_complexity'] = self._calculate_rhythm_complexity(measure_segment)
        
        return features
    
    # Energy extraction methods
    def _extract_rms_energy(self, segment: np.ndarray) -> float:
        """Extract RMS energy from audio segment."""
        if len(segment) == 0:
            return 0.0
        return np.sqrt(np.mean(segment**2))
    
    def _extract_peak_energy(self, segment: np.ndarray) -> float:
        """Extract peak energy from audio segment."""
        if len(segment) == 0:
            return 0.0
        return np.max(np.abs(segment))
    
    def _extract_energy_ratio(self, segment: np.ndarray) -> float:
        """Calculate ratio of peak to RMS energy."""
        rms = self._extract_rms_energy(segment)
        peak = self._extract_peak_energy(segment)
        return peak / (rms + 1e-8)
    
    # Onset detection methods
    def _detect_onset_strength(self, segment: np.ndarray) -> float:
        """Detect onset strength in segment."""
        if len(segment) == 0:
            return 0.0
        try:
            onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr, hop_length=self.hop_length)
            return np.max(onset_env) if len(onset_env) > 0 else 0.0
        except:
            return 0.0
    
    def _detect_onset_peak(self, segment: np.ndarray) -> float:
        """Detect if there's a clear onset peak."""
        if len(segment) == 0:
            return 0.0
        # Simple peak detection
        diff = np.diff(np.abs(segment))
        peaks = np.where(diff > np.std(diff))[0]
        return len(peaks) / len(segment) if len(segment) > 0 else 0.0
    
    # Frequency band extraction methods
    def _extract_bass_energy(self, segment: np.ndarray) -> float:
        """Extract bass frequency energy (60-250 Hz)."""
        if len(segment) == 0:
            return 0.0
        try:
            sos = signal.butter(4, [60, 250], 'bandpass', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, segment)
            return np.sqrt(np.mean(filtered**2))
        except:
            return 0.0
    
    def _extract_kick_energy(self, segment: np.ndarray) -> float:
        """Extract kick drum energy (60-80 Hz)."""
        if len(segment) == 0:
            return 0.0
        try:
            sos = signal.butter(4, [60, 80], 'bandpass', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, segment)
            return np.sqrt(np.mean(filtered**2))
        except:
            return 0.0
    
    def _extract_hihat_energy(self, segment: np.ndarray) -> float:
        """Extract hi-hat energy (8-12 kHz)."""
        if len(segment) == 0:
            return 0.0
        try:
            sos = signal.butter(4, [8000, 12000], 'bandpass', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, segment)
            return np.sqrt(np.mean(filtered**2))
        except:
            return 0.0
    
    def _extract_snare_energy(self, segment: np.ndarray) -> float:
        """Extract snare energy (200-400 Hz)."""
        if len(segment) == 0:
            return 0.0
        try:
            sos = signal.butter(4, [200, 400], 'bandpass', fs=self.sr, output='sos')
            filtered = signal.sosfilt(sos, segment)
            return np.sqrt(np.mean(filtered**2))
        except:
            return 0.0
    
    # Spectral feature extraction methods
    def _extract_spectral_centroid(self, segment: np.ndarray) -> float:
        """Extract spectral centroid."""
        if len(segment) == 0:
            return 0.0
        try:
            return librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0, 0]
        except:
            return 0.0
    
    def _extract_spectral_rolloff(self, segment: np.ndarray) -> float:
        """Extract spectral rolloff."""
        if len(segment) == 0:
            return 0.0
        try:
            return librosa.feature.spectral_rolloff(y=segment, sr=self.sr)[0, 0]
        except:
            return 0.0
    
    def _extract_spectral_bandwidth(self, segment: np.ndarray) -> float:
        """Extract spectral bandwidth."""
        if len(segment) == 0:
            return 0.0
        try:
            return librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)[0, 0]
        except:
            return 0.0
    
    def _extract_spectral_contrast(self, segment: np.ndarray) -> float:
        """Extract spectral contrast."""
        if len(segment) == 0:
            return 0.0
        try:
            return librosa.feature.spectral_contrast(y=segment, sr=self.sr)[0, 0]
        except:
            return 0.0
    
    def _extract_zero_crossing_rate(self, segment: np.ndarray) -> float:
        """Extract zero-crossing rate."""
        if len(segment) == 0:
            return 0.0
        try:
            return librosa.feature.zero_crossing_rate(segment)[0, 0]
        except:
            return 0.0
    
    # Timing and rhythm methods
    def _calculate_timing_offset(self, segment: np.ndarray, beat_time: float) -> float:
        """Calculate timing offset from beat center."""
        if len(segment) == 0:
            return 0.0
        
        # Find the peak within the segment
        peak_idx = np.argmax(np.abs(segment))
        segment_duration = len(segment) / self.sr
        peak_time_in_segment = peak_idx / self.sr
        
        # Calculate offset from beat center
        beat_center = self.pre_offset + (self.sixteenth_duration / 2)
        timing_offset = peak_time_in_segment - beat_center
        
        return timing_offset
    
    def _calculate_beat_correlation(self, position: int) -> float:
        """Calculate correlation with beat template."""
        return self.beat_template[position]
    
    def _calculate_rhythm_complexity(self, segment: np.ndarray) -> float:
        """Calculate rhythm complexity of the measure."""
        if len(segment) == 0:
            return 0.0
        
        # Use spectral flux as a measure of rhythm complexity
        try:
            flux = librosa.onset.onset_strength(y=segment, sr=self.sr, hop_length=self.hop_length)
            return np.std(flux) if len(flux) > 0 else 0.0
        except:
            return 0.0
    
    def calculate_total_measures(self, audio_file: Union[str, Path]) -> int:
        """
        Calculate the total number of measures in an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Total number of measures
        """
        try:
            # Load audio to get duration
            y, sr = librosa.load(audio_file, sr=self.sr)
            duration = len(y) / sr
            
            # Apply song offset adjustment
            if self.song_offset < 0:
                # For negative offset, we add silence, so duration increases
                duration += abs(self.song_offset)
            elif self.song_offset > 0:
                # For positive offset, we trim, so duration decreases
                duration = max(0, duration - self.song_offset)
            
            # Calculate total measures
            total_measures = int(duration / self.measure_duration)
            
            logger.info(f"Audio duration: {duration:.2f}s, Total measures: {total_measures}")
            return total_measures
            
        except Exception as e:
            logger.error(f"Error calculating total measures for {audio_file}: {e}")
            return 0
    
    def extract_song_features(self, 
                            audio_file: Union[str, Path], 
                            num_measures: Optional[int] = None) -> List[Dict[str, Union[float, List[float]]]]:
        """
        Extract features for multiple measures of a song.
        
        Args:
            audio_file: Path to audio file
            num_measures: Number of measures to extract. If None, extracts all measures.
            
        Returns:
            List of feature dictionaries, one per measure
        """
        # If num_measures is None, calculate total measures in the song
        if num_measures is None:
            num_measures = self.calculate_total_measures(audio_file)
            if num_measures == 0:
                logger.error(f"Could not determine number of measures for {audio_file}")
                return []
        
        features_list = []
        
        for measure_idx in range(num_measures):
            measure_start_time = measure_idx * self.measure_duration
            measure_features = self.extract_measure_features(
                audio_file, measure_start_time, measure_idx
            )
            
            if measure_features:  # Only add if extraction was successful
                features_list.append(measure_features)
            else:
                # If a measure fails, log it but continue with the next
                logger.warning(f"Failed to extract features for measure {measure_idx} from {audio_file}")
        
        logger.info(f"Extracted features for {len(features_list)} measures from {audio_file}")
        return features_list
    
    def save_features(self, 
                     features: List[Dict[str, Union[float, List[float]]]], 
                     output_file: Union[str, Path]) -> None:
        """
        Save extracted features to JSON file.
        
        Args:
            features: List of feature dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert all values in the features
        serializable_features = []
        for measure_features in features:
            converted_measure = {}
            for key, value in measure_features.items():
                converted_measure[key] = convert_numpy_types(value)
            serializable_features.append(converted_measure)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_features, f, indent=2)
        
        logger.info(f"Saved features to {output_file}")


def create_feature_extractor_from_stepfile(stepfile_path: Union[str, Path]) -> MusicalFeatureExtractor:
    """
    Create a MusicalFeatureExtractor from stepfile metadata.
    
    Args:
        stepfile_path: Path to .sm stepfile
        
    Returns:
        MusicalFeatureExtractor instance
    """
    # This would need to be implemented to parse .sm files
    # For now, return a default extractor
    logger.warning("create_feature_extractor_from_stepfile not yet implemented")
    return MusicalFeatureExtractor(bpm=140.0, song_offset=0.0)


def parse_sm_metadata(sm_file_path: Union[str, Path]) -> Tuple[float, float]:
    """
    Parse BPM and offset from a .sm stepfile.
    
    Args:
        sm_file_path: Path to .sm file
        
    Returns:
        Tuple of (bpm, offset)
    """
    with open(sm_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse BPM
    bpm_match = re.search(r'#BPMS:0\.000=(\d+\.?\d*);', content)
    if bpm_match:
        bpm = float(bpm_match.group(1))
    else:
        bpm = 140.0  # Default
        logger.warning(f"Could not parse BPM from {sm_file_path}, using default {bpm}")
    
    # Parse offset
    offset_match = re.search(r'#OFFSET:(-?\d+\.?\d*);', content)
    if offset_match:
        offset = float(offset_match.group(1))
    else:
        offset = 0.0  # Default
        logger.warning(f"Could not parse offset from {sm_file_path}, using default {offset}")
    
    return bpm, offset


def find_ogg_sm_pairs(stepfile_assets_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Find all .ogg files and their corresponding .sm files in stepfile_assets.
    
    Args:
        stepfile_assets_dir: Directory containing stepfile assets
        
    Returns:
        List of tuples (ogg_file_path, sm_file_path)
    """
    stepfile_assets_path = Path(stepfile_assets_dir)
    ogg_sm_pairs = []
    
    # Find all .ogg files recursively
    for ogg_file in stepfile_assets_path.rglob("*.ogg"):
        # Look for .sm files in the same directory
        sm_files = list(ogg_file.parent.glob("*.sm"))
        
        if sm_files:
            # Use the first .sm file found in the directory
            sm_file = sm_files[0]
            ogg_sm_pairs.append((ogg_file, sm_file))
            logger.info(f"Found pair: {ogg_file.name} <-> {sm_file.name}")
        else:
            logger.warning(f"No .sm file found in directory for {ogg_file}")
    
    return ogg_sm_pairs


def process_all_songs(stepfile_assets_dir: str = "stepfile_assets", 
                     output_dir: str = "feature_data/musical_features",
                     num_measures: Optional[int] = None,
                     pre_offset_ms: float = 8.0) -> None:
    """
    Process all .ogg files in stepfile_assets and extract musical features.
    
    Args:
        stepfile_assets_dir: Directory containing stepfile assets
        output_dir: Directory to save musical features
        num_measures: Number of measures to extract per song. If None, extracts all measures.
        pre_offset_ms: Pre-offset in milliseconds
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all ogg-sm pairs
    ogg_sm_pairs = find_ogg_sm_pairs(stepfile_assets_dir)
    
    if not ogg_sm_pairs:
        logger.error(f"No .ogg/.sm pairs found in {stepfile_assets_dir}")
        return
    
    logger.info(f"Found {len(ogg_sm_pairs)} .ogg/.sm pairs to process")
    
    successful = 0
    failed = 0
    
    for ogg_file, sm_file in ogg_sm_pairs:
        # Create output filename based on .sm filename
        output_filename = f"{sm_file.stem}.json"
        output_file = output_path / output_filename
        
        # Check if file already exists
        if output_file.exists():
            logger.info(f"⏭️  Skipping {sm_file.name} - already processed ({output_filename} exists)")
            successful += 1
            continue
            
        try:
            # Parse metadata from .sm file
            bpm, offset = parse_sm_metadata(sm_file)
            logger.info(f"Processing {ogg_file.name}: BPM={bpm}, Offset={offset}s")
            
            # Create extractor
            extractor = MusicalFeatureExtractor(
                bpm=bpm,
                song_offset=offset,
                pre_offset_ms=pre_offset_ms
            )
            
            # Extract features
            features = extractor.extract_song_features(ogg_file, num_measures=num_measures)
            
            if features:
                # Save features
                extractor.save_features(features, output_file)
                
                logger.info(f"✓ Successfully processed {ogg_file.name} -> {output_filename} ({len(features)} measures)")
                successful += 1
            else:
                logger.warning(f"✗ No features extracted from {ogg_file.name}")
                failed += 1
                
        except Exception as e:
            logger.error(f"✗ Error processing {ogg_file.name}: {e}")
            failed += 1
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"✓ Successful: {successful}")
    logger.info(f"✗ Failed: {failed}")
    logger.info(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    import re
    
    print("="*60)
    print("MUSICAL FEATURE EXTRACTION - BATCH PROCESSING")
    print("="*60)
    
    # Process all songs in stepfile_assets
    process_all_songs(
        stepfile_assets_dir="stepfile_assets",
        output_dir="feature_data/musical_features",
        num_measures=None,  # Extract ALL measures from each song
        pre_offset_ms=8.0  # 8ms pre-offset to capture waveform swell
    )
    
    print("\nMusicalFeatureExtractor batch processing complete!") 