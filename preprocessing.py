import numpy as np
import librosa
import soundfile as sf


class AudioPreprocessor:
    def __init__(self, target_sr=22050):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            target_sr (int): Target sample rate for audio files
        """
        self.target_sr = target_sr
        
    def load_and_preprocess(self, file_path, normalize=True, trim_silence=True, 
                           noise_reduce=True, mono=True):
        """
        Load and preprocess an audio file.
        
        Args:
            file_path (str): Path to the audio file
            normalize (bool): Whether to normalize the audio
            trim_silence (bool): Whether to trim silence
            noise_reduce (bool): Whether to reduce noise
            mono (bool): Whether to convert to mono
            
        Returns:
            tuple: (preprocessed_audio, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=mono)
            
            # Apply preprocessing steps
            if normalize:
                audio = self._normalize(audio)
                
            if trim_silence:
                audio = self._trim_silence(audio)
                
            if noise_reduce:
                audio = self._reduce_noise(audio)
                
            return audio, sr
            
        except Exception as e:
            print(f"Error preprocessing {file_path}: {e}")
            return None, None
    
    def _normalize(self, audio):
        """
        Normalize audio to have zero mean and unit variance.
        
        Args:
            audio (numpy.ndarray): Audio signal
            
        Returns:
            numpy.ndarray: Normalized audio signal
        """
        return librosa.util.normalize(audio)
    
    def _trim_silence(self, audio, threshold_db=20):
        """
        Trim silence from the beginning and end of an audio signal.
        
        Args:
            audio (numpy.ndarray): Audio signal
            threshold_db (float): Threshold in decibels below reference to consider as silence
            
        Returns:
            numpy.ndarray: Trimmed audio signal
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=threshold_db)
        return trimmed
    
    def _reduce_noise(self, audio, frame_length=2048, hop_length=512):
        """
        Simple noise reduction using spectral gating.
        
        Args:
            audio (numpy.ndarray): Audio signal
            frame_length (int): Frame length for STFT
            hop_length (int): Hop length for STFT
            
        Returns:
            numpy.ndarray: Noise-reduced audio signal
        """
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude, phase = librosa.magphase(stft)
        
        # Estimate noise floor (simple approach)
        noise_floor = np.mean(magnitude, axis=1, keepdims=True) * 0.1
        
        # Apply spectral gating
        magnitude = np.maximum(magnitude - noise_floor, 0)
        
        # Reconstruct signal
        stft_denoised = magnitude * phase
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)
        
        return audio_denoised
    
    def save_audio(self, audio, sr, output_path):
        """
        Save audio to a file.
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            output_path (str): Path to save the audio file
        """
        sf.write(output_path, audio, sr)
        
    def segment_audio(self, audio, sr, segment_length=5):
        """
        Segment audio into fixed-length segments.
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            segment_length (int): Length of each segment in seconds
            
        Returns:
            list: List of audio segments
        """
        segment_samples = segment_length * sr
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            
            # Pad if necessary
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
                
            segments.append(segment)
            
        return segments



