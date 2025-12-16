import os
import pandas as pd
import soundfile as sf
from pathlib import Path


class DataCollector:
    def __init__(self, data_dir="data"):
        """
        Initialize the DataCollector.
        
        Args:
            data_dir (str): Directory to store/read audio files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_local_data(self, audio_dir=None):
        """
        Load audio files from a local directory organized by species.
        
        Args:
            audio_dir (str): Directory containing audio files organized in subdirectories by species
                             If None, uses self.data_dir
        
        Returns:
            dict: Dictionary mapping file paths to species names
        """
        if audio_dir is None:
            audio_dir = self.data_dir
            
        file_species_map = {}
        
        for species_dir in os.listdir(audio_dir):
            species_path = os.path.join(audio_dir, species_dir)
            
            if os.path.isdir(species_path):
                for audio_file in os.listdir(species_path):
                    if audio_file.endswith(('.wav', '.mp3', '.ogg')):
                        file_path = os.path.join(species_path, audio_file)
                        file_species_map[file_path] = species_dir
        
        return file_species_map
    
    
    def get_audio_metadata(self, file_path):
        """
        Get metadata for an audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Audio metadata
        """
        try:
            data, sample_rate = sf.read(file_path)
            return {
                'path': file_path,
                'sample_rate': sample_rate,
                'duration': len(data) / sample_rate,
                'channels': 1 if len(data.shape) == 1 else data.shape[1]
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
