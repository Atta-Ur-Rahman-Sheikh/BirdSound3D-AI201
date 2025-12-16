import numpy as np
import librosa
import pandas as pd


class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13, n_mels=128, n_chroma=12):
        """
        Initialize the FeatureExtractor.
        
        Args:
            sr (int): Sample rate
            n_mfcc (int): Number of MFCCs to extract
            n_mels (int): Number of Mel bands
            n_chroma (int): Number of chroma bins
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        
    def extract_features(self, audio, feature_types=None):
        """
        Extract audio features.
        
        Args:
            audio (numpy.ndarray): Audio signal
            feature_types (list): List of feature types to extract.
                                 Options: 'mfcc', 'mel_spectrogram', 'chroma', 'spectral_contrast'
                                 If None, extracts all features
                                 
        Returns:
            dict: Dictionary of extracted features
        """
        if feature_types is None:
            feature_types = ['mfcc', 'mel_spectrogram', 'chroma', 'spectral_contrast']
            
        features = {}
        
        if 'mfcc' in feature_types:
            features['mfcc'] = self.extract_mfcc(audio)
            
        if 'mel_spectrogram' in feature_types:
            features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
            
        if 'chroma' in feature_types:
            features['chroma'] = self.extract_chroma(audio)
            
        if 'spectral_contrast' in feature_types:
            features['spectral_contrast'] = self.extract_spectral_contrast(audio)
            
        return features
    
    def extract_mfcc(self, audio, n_mfcc=None):
        """
        Extract Mel-frequency cepstral coefficients.
        
        Args:
            audio (numpy.ndarray): Audio signal
            n_mfcc (int): Number of MFCCs to extract. If None, uses self.n_mfcc
            
        Returns:
            numpy.ndarray: MFCCs
        """
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
            
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        return mfccs
    
    def extract_mel_spectrogram(self, audio, n_mels=None):
        """
        Extract Mel spectrogram.
        
        Args:
            audio (numpy.ndarray): Audio signal
            n_mels (int): Number of Mel bands. If None, uses self.n_mels
            
        Returns:
            numpy.ndarray: Mel spectrogram
        """
        if n_mels is None:
            n_mels = self.n_mels
            
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=n_mels)
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma(self, audio, n_chroma=None):
        """
        Extract chromagram.
        
        Args:
            audio (numpy.ndarray): Audio signal
            n_chroma (int): Number of chroma bins. If None, uses self.n_chroma
            
        Returns:
            numpy.ndarray: Chromagram
        """
        if n_chroma is None:
            n_chroma = self.n_chroma
            
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, n_chroma=n_chroma)
        return chroma
    
    def extract_spectral_contrast(self, audio, n_bands=6):
        """
        Extract spectral contrast.
        
        Args:
            audio (numpy.ndarray): Audio signal
            n_bands (int): Number of bands for spectral contrast
            
        Returns:
            numpy.ndarray: Spectral contrast
        """
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, n_bands=n_bands)
        return contrast
    
    def extract_time_domain_features(self, audio):
        """
        Extract time-domain features.
        
        Args:
            audio (numpy.ndarray): Audio signal
            
        Returns:
            dict: Dictionary of time-domain features
        """
        features = {}
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        
        # RMS energy
        features['rms'] = librosa.feature.rms(y=audio)[0]
        
        # Basic statistics
        features['mean'] = np.mean(audio)
        features['std'] = np.std(audio)
        features['max'] = np.max(audio)
        features['min'] = np.min(audio)
        
        return features
    
    def extract_all_features_as_vector(self, audio):
        """
        Extract all features and flatten them into a single vector.
        
        Args:
            audio (numpy.ndarray): Audio signal
            
        Returns:
            numpy.ndarray: Flattened feature vector
        """
        features = self.extract_features(audio)
        
        # Extract summary statistics from each feature
        feature_vector = []
        
        # Process MFCCs
        mfccs = features['mfcc']
        feature_vector.extend(np.mean(mfccs, axis=1))
        feature_vector.extend(np.std(mfccs, axis=1))
        
        # Process Mel spectrogram
        mel_spec = features['mel_spectrogram']
        feature_vector.extend([np.mean(mel_spec), np.std(mel_spec), np.max(mel_spec), np.min(mel_spec)])
        
        # Process chroma
        chroma = features['chroma']
        feature_vector.extend(np.mean(chroma, axis=1))
        
        # Process spectral contrast
        contrast = features['spectral_contrast']
        feature_vector.extend(np.mean(contrast, axis=1))
        
        # Add time-domain features
        time_features = self.extract_time_domain_features(audio)
        feature_vector.extend([
            np.mean(time_features['zero_crossing_rate']),
            np.mean(time_features['rms']),
            time_features['mean'],
            time_features['std']
        ])
        
        return np.array(feature_vector)
    
    def get_feature_names(self):
        """
        Get names of features in the feature vector.
        
        Returns:
            list: List of feature names
        """
        feature_names = []
        
        # MFCC names
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_mean_{i}')
        for i in range(self.n_mfcc):
            feature_names.append(f'mfcc_std_{i}')
        
        # Mel spectrogram statistics
        feature_names.extend(['mel_mean', 'mel_std', 'mel_max', 'mel_min'])
        
        # Chroma names
        for i in range(self.n_chroma):
            feature_names.append(f'chroma_mean_{i}')
        
        # Spectral contrast names
        for i in range(6 + 1):  # n_bands + 1
            feature_names.append(f'contrast_mean_{i}')
        
        # Time domain features
        feature_names.extend(['zcr_mean', 'rms_mean', 'audio_mean', 'audio_std'])
        
        return feature_names



