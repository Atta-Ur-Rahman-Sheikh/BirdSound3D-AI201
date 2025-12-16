import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import librosa
import plotly.graph_objects as go
from pathlib import Path
import io
import matplotlib.pyplot as plt

# Import custom modules
from data_collection import DataCollector
from preprocessing import AudioPreprocessor
from feature_extraction import FeatureExtractor
from visualization import AudioVisualizer
from classification import BirdSoundClassifier
import utils

# Import demo data generator
from demo_data import create_sample_data

# Set page configuration
st.set_page_config(
    page_title="BirdSound3D",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #0D47A1;
            margin-top: 2rem;
        }
        .card {
            border-radius: 5px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
            border-left: 5px solid #1E88E5;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .feature-title {
            color: #0D47A1;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .feature-desc {
            margin-top: 0.5rem;
            font-size: 1rem;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            font-size: 0.8rem;
            color: #666;
        }
        .plot-container {
            border-radius: 5px;
            padding: 1rem;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .bird-info {
            padding: 1rem;
            background-color: #e8f5e9;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .bird-name {
            font-weight: bold;
            color: #2E7D32;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create a waveform image
def create_waveform_image(audio, sr):
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color='#1E88E5')
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    return buf

# Function to get bird information
def get_bird_info():
    bird_info = {}
    info_file = os.path.join("data", "info", "bird_info.txt")
    
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            for line in f:
                if "|" in line:
                    species, info = line.strip().split("|", 1)
                    bird_info[species] = info
    
    return bird_info


def main():
    # Apply custom CSS
    local_css()
    
    # Create directories if they don't exist
    data_dir = "data"
    models_dir = "models"
    utils.ensure_dir(data_dir)
    utils.ensure_dir(models_dir)
    
    # Initialize components
    data_collector = DataCollector(data_dir=data_dir)
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()
    visualizer = AudioVisualizer()
    
    # Check if model exists
    model_path = os.path.join(models_dir, "bird_classifier.pkl")
    if os.path.exists(model_path):
        classifier = BirdSoundClassifier(model_path=model_path)
        model_loaded = True
    else:
        classifier = BirdSoundClassifier()
        model_loaded = False
    
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center;'>🐦 BirdSound3D</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Demo data generation
    with st.sidebar.expander("Demo Data", expanded=True):
        st.markdown("Generate sample bird calls for demonstration")
        if st.button("Generate Demo Data", key="generate_demo"):
            with st.spinner("Generating demo data..."):
                try:
                    data_dir = create_sample_data()
                    st.success(f"Demo data generated successfully!")
                except Exception as e:
                    st.error(f"Error generating demo data: {e}")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["🏠 Home", "🔍 Analyze Bird Calls", "📁 Data Management", "🧠 Train Model"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class='footer'>
        BirdSound3D - Bird Call Analysis Tool<br>
        Created with Streamlit & Python
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Analyze Bird Calls":
        show_analysis_page(preprocessor, feature_extractor, visualizer, classifier, model_loaded)
    elif page == "📁 Data Management":
        show_data_collection_page(data_collector)
    elif page == "🧠 Train Model":
        show_model_training_page(data_collector, preprocessor, feature_extractor, classifier, models_dir)


def show_home_page():
    # Header
    st.markdown("<h1 class='main-header'>Welcome to BirdSound3D</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='card'>
        <p>BirdSound3D is an interactive tool for analyzing and visualizing bird calls in three dimensions. 
        This application allows ornithologists, researchers, and bird enthusiasts to explore the acoustic 
        patterns of different bird species through advanced audio processing techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("<h2 class='sub-header'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <p class='feature-title'>🔊 3D Audio Visualization</p>
            <p class='feature-desc'>Transform bird calls into interactive 3D spectrograms, allowing you to explore frequency patterns over time from multiple angles.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <p class='feature-title'>🧠 Machine Learning Classification</p>
            <p class='feature-desc'>Train models to automatically identify bird species from their calls using advanced audio feature extraction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <p class='feature-title'>📊 Feature Extraction</p>
            <p class='feature-desc'>Extract MFCCs, spectrograms, and chromagrams to analyze the acoustic signatures of different bird species.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <p class='feature-title'>🔍 Audio Preprocessing</p>
            <p class='feature-desc'>Clean and enhance bird call recordings with normalization, noise reduction, and silence trimming.</p>
        </div>
        """, unsafe_allow_html=True)


def show_analysis_page(preprocessor, feature_extractor, visualizer, classifier, model_loaded):
    # Header
    st.markdown("<h1 class='main-header'>Analyze Bird Calls</h1>", unsafe_allow_html=True)
    
    # Two options: upload file or use sample
    st.markdown("<h2 class='sub-header'>Select Audio Source</h2>", unsafe_allow_html=True)
    
    source_option = st.radio(
        "Choose an audio source",
        ["Upload your own audio file", "Use a sample from the demo data"]
    )
    
    audio_path = None
    
    if source_option == "Upload your own audio file":
        # File uploader
        uploaded_file = st.file_uploader("Upload a bird call recording", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            # Display audio player in a card
            st.markdown("""
            <div class='card'>
                <p class='feature-title'>Uploaded Audio</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(uploaded_file)
    else:
        # Sample selection
        data_dir = "data"
        species_dirs = []
        
        if os.path.exists(data_dir):
            species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "info"]
        
        if species_dirs:
            # Select species
            selected_species = st.selectbox("Select bird species", species_dirs)
            
            # Get audio files for the selected species
            species_path = os.path.join(data_dir, selected_species)
            audio_files = [f for f in os.listdir(species_path) if utils.is_audio_file(f)]
            
            if audio_files:
                # Select audio file
                selected_file = st.selectbox("Select audio sample", audio_files)
                audio_path = os.path.join(species_path, selected_file)
                
                # Display audio player in a card
                st.markdown("""
                <div class='card'>
                    <p class='feature-title'>Selected Sample</p>
                </div>
                """, unsafe_allow_html=True)
                st.audio(audio_path)
                
                # Get bird information
                bird_info = get_bird_info()
                if selected_species in bird_info:
                    st.markdown(f"""
                    <div class='bird-info'>
                        <p class='bird-name'>{selected_species.replace('_', ' ')}</p>
                        <p>{bird_info[selected_species]}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No sample data found. Click 'Generate Demo Data' in the sidebar first.")
    
    # Process button
    if audio_path and st.button("Analyze Audio", key="analyze_btn"):
        with st.spinner("Processing audio file..."):
            # Preprocess audio
            audio, sr = preprocessor.load_and_preprocess(
                audio_path, 
                normalize=True, 
                trim_silence=True, 
                noise_reduce=True
            )
            
            if audio is not None:
                # Extract features
                features = feature_extractor.extract_features(audio)
                feature_vector = feature_extractor.extract_all_features_as_vector(audio)
                
                # Display visualizations
                st.markdown("<h2 class='sub-header'>3D Visualizations</h2>", unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["3D Spectrogram", "3D MFCC", "3D Chromagram"])
                
                with viz_tab1:
                    st.markdown("""
                    <div class='info-box'>
                        <p><strong>3D Spectrogram</strong> shows how the frequency content of the bird call changes over time. 
                        The x-axis represents time, the y-axis represents frequency, and the z-axis (height/color) represents amplitude.</p>
                        <p>Unique patterns in the spectrogram can help identify different bird species.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.container():
                        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
                        fig = visualizer.create_3d_spectrogram_plotly(audio, sr)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with viz_tab2:
                    st.markdown("""
                    <div class='info-box'>
                        <p><strong>Mel-frequency cepstral coefficients (MFCCs)</strong> capture the timbral characteristics of the bird call.
                        They represent the short-term power spectrum of the sound based on a linear cosine transform of the log power spectrum
                        on a nonlinear mel scale of frequency.</p>
                        <p>MFCCs are particularly useful for identifying the unique "voice" characteristics of different bird species.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.container():
                        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
                        mfcc_fig = visualizer.create_3d_mfcc_visualization(features['mfcc'], sr)
                        st.plotly_chart(mfcc_fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with viz_tab3:
                    st.markdown("""
                    <div class='info-box'>
                        <p><strong>Chromagram</strong> represents the distribution of energy across the 12 pitch classes of Western music.
                        It helps visualize the tonal content of the bird call, showing which pitches are most prominent.</p>
                        <p>Some bird species have distinctive pitch patterns that can be identified in the chromagram.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.container():
                        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
                        chroma_fig = visualizer.create_3d_chromagram(features['chroma'], sr)
                        st.plotly_chart(chroma_fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Classification (if model is loaded) - Moved after visualizations
                if model_loaded:
                    st.markdown("<h2 class='sub-header'>Classification Result</h2>", unsafe_allow_html=True)
                    
                    try:
                        # Predict species
                        prediction = classifier.predict(feature_vector)[0]
                        
                        # Create columns for prediction and confidence
                        pred_col1, pred_col2 = st.columns(2)
                        
                        with pred_col1:
                            st.markdown(f"""
                            <div class='card'>
                                <p class='feature-title'>Predicted Species</p>
                                <p style='font-size: 1.5rem; color: #1E88E5;'>{prediction.replace('_', ' ')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Get prediction probabilities if available
                        try:
                            proba = classifier.predict_proba(feature_vector)[0]
                            
                            # Get class names
                            class_names = classifier.model.classes_
                            
                            # Create DataFrame for probabilities
                            proba_df = pd.DataFrame({
                                'Species': [name.replace('_', ' ') for name in class_names],
                                'Probability': proba
                            }).sort_values('Probability', ascending=False)
                            
                            with pred_col2:
                                st.markdown("""
                                <div class='card'>
                                    <p class='feature-title'>Confidence Levels</p>
                                </div>
                                """, unsafe_allow_html=True)
                                # Display top 5 probabilities
                                st.bar_chart(proba_df.set_index('Species').head(5))
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"Classification error: {e}")
                else:
                    st.info("No classification model loaded. Train a model in the 'Train Model' page.")
            else:
                st.error("Error processing the audio file.")
        
        # Clean up temporary file if it was uploaded
        if source_option == "Upload your own audio file" and audio_path:
            try:
                os.unlink(audio_path)
            except:
                pass


def show_data_collection_page(data_collector):
    # Header
    st.markdown("<h1 class='main-header'>Data Management</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='card'>
        <p>Manage your bird call recordings and organize them by species. You can upload your own recordings 
        or use the demo data generated by the system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display existing data
    st.markdown("<h2 class='sub-header'>Available Data</h2>", unsafe_allow_html=True)
    
    data_dir = data_collector.data_dir
    if os.path.exists(data_dir):
        species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "info"]
        
        if species_dirs:
            # Create a DataFrame to display the data
            data_stats = []
            for species in species_dirs:
                species_path = os.path.join(data_dir, species)
                audio_files = [f for f in os.listdir(species_path) if utils.is_audio_file(f)]
                data_stats.append({
                    "Species": species.replace("_", " "),
                    "Files": len(audio_files)
                })
            
            stats_df = pd.DataFrame(data_stats)
            
            # Display as a bar chart
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.bar_chart(stats_df.set_index("Species"))
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display as a table
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.dataframe(stats_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display samples for each species
            st.markdown("<h2 class='sub-header'>Browse Samples</h2>", unsafe_allow_html=True)
            
            # Get bird information
            bird_info = get_bird_info()
            
            selected_species = st.selectbox("Select species to browse", species_dirs)
            species_path = os.path.join(data_dir, selected_species)
            audio_files = [f for f in os.listdir(species_path) if utils.is_audio_file(f)]
            
            if audio_files:
                # Display bird information if available
                if selected_species in bird_info:
                    st.markdown(f"""
                    <div class='bird-info'>
                        <p class='bird-name'>{selected_species.replace('_', ' ')}</p>
                        <p>{bird_info[selected_species]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create columns for the audio files
                cols = st.columns(3)
                
                for i, audio_file in enumerate(audio_files[:9]):  # Show up to 9 samples
                    file_path = os.path.join(species_path, audio_file)
                    with cols[i % 3]:
                        st.markdown(f"<p><strong>Sample {i+1}</strong></p>", unsafe_allow_html=True)
                        st.audio(file_path)
        else:
            st.info("No data found. Upload files below or generate demo data from the sidebar.")
    
    # Upload files section
    st.markdown("<h2 class='sub-header'>Upload Your Own Recordings</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>Upload your own bird call recordings and organize them by species. Each species will be stored in its own folder.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Species input
    species_name = st.text_input("Species Name (used for folder name)", help="Example: Common_Blackbird")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload audio files", 
        type=["wav", "mp3", "ogg"], 
        accept_multiple_files=True
    )
    
    if species_name and uploaded_files:
        if st.button("Save Files", key="save_files_btn"):
            with st.spinner("Saving files..."):
                species_dir = os.path.join(data_dir, species_name.replace(" ", "_"))
                utils.ensure_dir(species_dir)
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(species_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                
                st.success(f"Saved {len(uploaded_files)} files to {species_dir}")
    


def show_model_training_page(data_collector, preprocessor, feature_extractor, classifier, models_dir):
    # Header
    st.markdown("<h1 class='main-header'>Train Classification Model</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='card'>
        <p>Train a machine learning model to automatically identify bird species from their calls.
        The model will learn from the audio features extracted from your bird call recordings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for available data
    data_dir = data_collector.data_dir
    species_dirs = []
    
    if os.path.exists(data_dir):
        species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "info"]
    
    if not species_dirs:
        st.warning("No training data available. Please generate demo data first or upload your own recordings.")
        return
    
    # Display available data
    st.markdown("<h2 class='sub-header'>Training Data</h2>", unsafe_allow_html=True)
    
    data_stats = {}
    for species in species_dirs:
        species_path = os.path.join(data_dir, species)
        audio_files = [f for f in os.listdir(species_path) if utils.is_audio_file(f)]
        data_stats[species] = len(audio_files)
    
    # Create a DataFrame to display the data
    stats_df = pd.DataFrame({
        'Species': [species.replace('_', ' ') for species in data_stats.keys()],
        'Files': list(data_stats.values())
    }).sort_values('Files', ascending=False)
    
    # Display as a bar chart
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.bar_chart(stats_df.set_index("Species"))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Training options
    st.markdown("<h2 class='sub-header'>Training Options</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>Select the model type and training parameters. For most cases, the default settings will work well.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Random Forest model only
    model_type = "random_forest"
    st.info("Using Random Forest classifier for bird species identification")
    
    # Train button
    if st.button("Train Model", key="train_model_btn"):
        with st.spinner("Processing audio files and extracting features..."):
            # Load and process all audio files
            features_list = []
            labels = []
            
            # Create a progress bar
            progress_text = "Processing audio files..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Count total files
            total_files = sum(data_stats.values())
            processed_files = 0
            
            for species in species_dirs:
                species_path = os.path.join(data_dir, species)
                audio_files = [os.path.join(species_path, f) for f in os.listdir(species_path) 
                              if utils.is_audio_file(os.path.join(species_path, f))]
                
                status_text.text(f"{progress_text} Processing {species.replace('_', ' ')}...")
                
                for audio_file in audio_files:
                    try:
                        # Preprocess audio
                        audio, sr = preprocessor.load_and_preprocess(audio_file)
                        
                        if audio is not None:
                            # Extract feature vector
                            feature_vector = feature_extractor.extract_all_features_as_vector(audio)
                            
                            # Add to dataset
                            features_list.append(feature_vector)
                            labels.append(species)
                    except Exception as e:
                        st.error(f"Error processing {audio_file}: {e}")
                    
                    # Update progress
                    processed_files += 1
                    progress_bar.progress(processed_files / total_files)
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels)
            
            if len(X) > 0:
                status_text.text(f"Extracted features from {len(X)} audio files across {len(set(labels))} species")
                
                # Train model
                with st.spinner("Training classification model..."):
                    # Initialize classifier with selected model type
                    classifier = BirdSoundClassifier(model_type=model_type)
                    
                    # Train model with fixed parameters
                    results = classifier.train(
                        X, y, 
                        test_size=0.2,  # Fixed test size
                        optimize_hyperparams=False  # No hyperparameter optimization for simplicity
                    )
                    
                    # Display results
                    st.markdown("<h2 class='sub-header'>Training Results</h2>", unsafe_allow_html=True)
                    
                    # Create a card for the accuracy
                    st.markdown(f"""
                    <div class='card'>
                        <p class='feature-title'>Model Accuracy</p>
                        <p style='font-size: 2rem; color: #1E88E5;'>{results['accuracy']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display classification report in a nicer format
                    report = results['classification_report']
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Rename the index for better display
                    report_df.index = [idx.replace('_', ' ') for idx in report_df.index]
                    
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("<p><strong>Classification Report</strong></p>", unsafe_allow_html=True)
                    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Save model
                    model_path = os.path.join(models_dir, "bird_classifier.pkl")
                    classifier.save_model(model_path)
                    
                    st.success("Model trained successfully! You can now use it to classify bird calls.")
                    
                    # Display feature importance if available
                    if hasattr(classifier.model, 'feature_importances_'):
                        st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class='info-box'>
                            <p>These are the most important audio features used by the model to distinguish between bird species.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        feature_names = feature_extractor.get_feature_names()
                        importance_df = classifier.get_feature_importance(feature_names)
                        
                        # Display top 10 features as a bar chart
                        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
                        st.bar_chart(importance_df.set_index('Feature').head(10))
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("No features extracted. Check audio files and try again.")
    


if __name__ == "__main__":
    main()
