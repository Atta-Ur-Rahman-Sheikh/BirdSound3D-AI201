import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import librosa


class AudioVisualizer:
    def __init__(self):
        """
        Initialize the AudioVisualizer.
        """
        pass
    
    def create_3d_spectrogram_matplotlib(self, audio, sr, title="3D Spectrogram", 
                                        n_fft=2048, hop_length=512, save_path=None):
        """
        Create a 3D spectrogram visualization using Matplotlib.
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            title (str): Plot title
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            save_path (str): Path to save the figure (if None, displays the figure)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Compute spectrogram
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        
        # Convert to dB scale
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        
        # Create time and frequency arrays
        times = librosa.times_like(spec, sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Create mesh grid for 3D plot
        time_grid, freq_grid = np.meshgrid(times, freqs)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(time_grid, freq_grid, spec_db, cmap=cm.viridis, 
                              linewidth=0, antialiased=False)
        
        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel('Amplitude (dB)')
        ax.set_title(title)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_3d_spectrogram_plotly(self, audio, sr, title="3D Spectrogram", 
                                    n_fft=2048, hop_length=512):
        """
        Create an interactive 3D spectrogram visualization using Plotly.
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            title (str): Plot title
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Compute spectrogram
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        
        # Convert to dB scale
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        
        # Create time and frequency arrays
        times = librosa.times_like(spec, sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Create mesh grid for 3D plot
        time_grid, freq_grid = np.meshgrid(times, freqs)
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Surface(z=spec_db, x=time_grid, y=freq_grid, 
                                       colorscale='Viridis')])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                zaxis_title='Amplitude (dB)',
                aspectratio=dict(x=1, y=1, z=0.7),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=0.8)
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def create_3d_mfcc_visualization(self, mfcc, sr, hop_length=512, title="3D MFCC Visualization"):
        """
        Create a 3D visualization of MFCCs using Plotly.
        
        Args:
            mfcc (numpy.ndarray): MFCC features
            sr (int): Sample rate
            hop_length (int): Hop length used for MFCC extraction
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Create time array
        times = librosa.times_like(mfcc, sr=sr, hop_length=hop_length)
        
        # Create coefficient indices
        coeffs = np.arange(mfcc.shape[0])
        
        # Create mesh grid for 3D plot
        time_grid, coeff_grid = np.meshgrid(times, coeffs)
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Surface(z=mfcc, x=time_grid, y=coeff_grid, 
                                       colorscale='Viridis')])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='MFCC Coefficient',
                zaxis_title='Amplitude',
                aspectratio=dict(x=1, y=1, z=0.7),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=0.8)
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def create_3d_chromagram(self, chroma, sr, hop_length=512, title="3D Chromagram"):
        """
        Create a 3D visualization of a chromagram using Plotly.
        
        Args:
            chroma (numpy.ndarray): Chroma features
            sr (int): Sample rate
            hop_length (int): Hop length used for chroma extraction
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Create time array
        times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)
        
        # Create pitch class names
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_indices = np.arange(chroma.shape[0])
        
        # Create mesh grid for 3D plot
        time_grid, pitch_grid = np.meshgrid(times, pitch_indices)
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Surface(z=chroma, x=time_grid, y=pitch_grid, 
                                       colorscale='Viridis')])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='Pitch Class',
                zaxis_title='Magnitude',
                aspectratio=dict(x=1, y=1, z=0.7),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=0.8)
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Update y-axis ticks to show pitch class names
        fig.update_layout(
            scene=dict(
                yaxis=dict(
                    tickvals=pitch_indices,
                    ticktext=pitch_classes
                )
            )
        )
        
        return fig
    
    def create_feature_comparison_plot(self, features_dict, title="Feature Comparison"):
        """
        Create a subplot with different feature visualizations.
        
        Args:
            features_dict (dict): Dictionary of extracted features
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot MFCC
        if 'mfcc' in features_dict:
            librosa.display.specshow(features_dict['mfcc'], ax=axes[0, 0], x_axis='time')
            axes[0, 0].set_title('MFCC')
            axes[0, 0].set_ylabel('Coefficient')
        
        # Plot Mel spectrogram
        if 'mel_spectrogram' in features_dict:
            librosa.display.specshow(features_dict['mel_spectrogram'], ax=axes[0, 1], 
                                   x_axis='time', y_axis='mel')
            axes[0, 1].set_title('Mel Spectrogram')
        
        # Plot Chroma
        if 'chroma' in features_dict:
            librosa.display.specshow(features_dict['chroma'], ax=axes[1, 0], 
                                   x_axis='time', y_axis='chroma')
            axes[1, 0].set_title('Chromagram')
        
        # Plot Spectral Contrast
        if 'spectral_contrast' in features_dict:
            librosa.display.specshow(features_dict['spectral_contrast'], ax=axes[1, 1], 
                                   x_axis='time')
            axes[1, 1].set_title('Spectral Contrast')
        
        plt.tight_layout()
        return fig



