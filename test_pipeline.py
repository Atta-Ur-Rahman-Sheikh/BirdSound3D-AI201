import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Import custom modules
from preprocessing import AudioPreprocessor
from feature_extraction import FeatureExtractor
from visualization import AudioVisualizer
from classification import BirdSoundClassifier
import utils


def test_pipeline(audio_file=None, output_dir="test_output"):
    """
    Test the entire pipeline with a single audio file.
    
    Args:
        audio_file (str): Path to an audio file. If None, uses a test tone.
        output_dir (str): Directory to save output files
    """
    print("Testing BirdSound3D Pipeline")
    print("-" * 50)
    
    # Create output directory
    utils.ensure_dir(output_dir)
    
    # Generate test audio if not provided
    if audio_file is None or not os.path.exists(audio_file):
        print("No audio file provided. Generating test tone...")
        sr = 22050
        duration = 3  # seconds
        
        # Generate a chirp (frequency sweep)
        t = np.linspace(0, duration, int(sr * duration))
        start_freq = 2000
        end_freq = 8000
        
        # Frequency increases exponentially
        freq = start_freq * np.exp(np.log(end_freq / start_freq) * t / duration)
        
        # Generate sine wave with varying frequency
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        audio = audio + 0.01 * np.random.randn(len(audio))
        
        # Save test audio
        test_audio_path = os.path.join(output_dir, "test_chirp.wav")
        wavfile.write(test_audio_path, sr, audio.astype(np.float32))
        
        audio_file = test_audio_path
        print(f"Test audio generated and saved to {test_audio_path}")
    
    # Initialize components
    print("\nInitializing components and processing audio...")
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    visualizer = AudioVisualizer()
    
    # Preprocess audio
    audio, sr = preprocessor.load_and_preprocess(
        audio_file, 
        normalize=True, 
        trim_silence=True, 
        noise_reduce=True
    )
    
    if audio is None:
        print("Error: Failed to preprocess audio.")
        return
    
    # Extract features
    features = feature_extractor.extract_features(audio)
    feature_vector = feature_extractor.extract_all_features_as_vector(audio)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 3D Spectrogram
    visualizer.create_3d_spectrogram_matplotlib(
        audio, sr, title="3D Spectrogram", 
        save_path=os.path.join(output_dir, "3d_spectrogram.png")
    )
    
    # 3D MFCC Visualization
    fig_mfcc = visualizer.create_3d_mfcc_visualization(
        features['mfcc'], sr, title="3D MFCC Visualization"
    )
    fig_mfcc.write_html(os.path.join(output_dir, "3d_mfcc.html"))
    
    print(f"Visualizations saved to {output_dir}")
    print("\nPipeline test completed successfully!")


if __name__ == "__main__":
    import sys
    
    # Simple argument parsing
    audio_file = None
    output_dir = "test_output"
    
    # Check for audio file argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    # Run test pipeline
    test_pipeline(audio_file, output_dir)



