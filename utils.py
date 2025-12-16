import os
import numpy as np
import soundfile as sf
import librosa
import pickle
import json
from pathlib import Path


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_features(features, file_path):
    """
    Save features to a file using pickle.
    
    Args:
        features (dict): Features dictionary
        file_path (str): Path to save the features
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)


def load_features(file_path):
    """
    Load features from a file.
    
    Args:
        file_path (str): Path to the features file
        
    Returns:
        dict: Loaded features
    """
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    
    return features


def save_metadata(metadata, file_path):
    """
    Save metadata to a JSON file.
    
    Args:
        metadata (dict): Metadata dictionary
        file_path (str): Path to save the metadata
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(file_path):
    """
    Load metadata from a JSON file.
    
    Args:
        file_path (str): Path to the metadata file
        
    Returns:
        dict: Loaded metadata
    """
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def get_audio_duration(file_path):
    """
    Get the duration of an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return len(audio) / sr
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return None


def convert_audio_format(input_path, output_path, target_sr=22050):
    """
    Convert audio file to a different format.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to output audio file
        target_sr (int): Target sample rate
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=target_sr)
        
        # Create output directory if it doesn't exist
        ensure_dir(os.path.dirname(output_path))
        
        # Save in new format
        sf.write(output_path, audio, target_sr)
        
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def batch_process_audio(input_files, output_dir, process_func, **kwargs):
    """
    Process multiple audio files with a given function.
    
    Args:
        input_files (list): List of input file paths
        output_dir (str): Directory to save processed files
        process_func (callable): Processing function
        **kwargs: Additional arguments for process_func
        
    Returns:
        list: List of output file paths
    """
    ensure_dir(output_dir)
    output_files = []
    
    for input_file in input_files:
        try:
            # Generate output file path
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
            
            # Process file
            result = process_func(input_file, output_file, **kwargs)
            
            if result:
                output_files.append(output_file)
                
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    return output_files


def get_file_extension(file_path):
    """
    Get the extension of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension
    """
    return os.path.splitext(file_path)[1].lower()


def is_audio_file(file_path):
    """
    Check if a file is an audio file based on extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if it's an audio file, False otherwise
    """
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
    return get_file_extension(file_path) in audio_extensions



