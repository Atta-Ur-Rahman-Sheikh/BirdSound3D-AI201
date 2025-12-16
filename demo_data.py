import os
import numpy as np
from scipy.io import wavfile

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def generate_sine_wave(freq, duration, sample_rate=22050):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    return data

def generate_chirp(start_freq, end_freq, duration, sample_rate=22050):
    """Generate a frequency sweep (chirp)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    freq = np.linspace(start_freq, end_freq, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    data = 0.5 * np.sin(phase)
    return data

def generate_bird_call(species_type, duration=3.0, sr=22050):
    """Generate a synthetic bird call based on species type."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    if species_type == "blackbird":
        # Blackbirds have melodious flute-like calls
        # Create a series of pure tones with slight frequency modulation
        n_notes = np.random.randint(4, 8)
        note_duration = duration / n_notes
        
        for i in range(n_notes):
            start_idx = int(i * note_duration * sr)
            note_len = int(note_duration * 0.8 * sr)  # Leave small gaps between notes
            end_idx = min(start_idx + note_len, len(t))
            
            # Create a note with slight frequency modulation
            base_freq = np.random.choice([1800, 2000, 2200, 2400])
            mod_depth = np.random.uniform(50, 150)
            mod_freq = np.random.uniform(3, 8)
            
            note_t = t[start_idx:end_idx] - t[start_idx]
            freq = base_freq + mod_depth * np.sin(2 * np.pi * mod_freq * note_t)
            phase = 2 * np.pi * np.cumsum(freq) / sr
            note = 0.7 * np.sin(phase)
            
            # Apply envelope
            env = np.ones_like(note)
            attack = int(0.1 * len(note))
            release = int(0.2 * len(note))
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
            note = note * env
            
            audio[start_idx:end_idx] += note
            
    elif species_type == "robin":
        # Robins have thin, high-pitched calls with repeated patterns
        # Create several short chirps
        n_chirps = np.random.randint(8, 15)
        
        for i in range(n_chirps):
            start_time = np.random.uniform(0, duration * 0.8)
            start_idx = int(start_time * sr)
            chirp_len = int(np.random.uniform(0.1, 0.25) * sr)
            end_idx = min(start_idx + chirp_len, len(t))
            
            # Create a chirp
            start_freq = np.random.uniform(3500, 4500)
            end_freq = np.random.uniform(4500, 5500)
            
            chirp_t = np.linspace(0, 1, end_idx - start_idx)
            freq = start_freq + (end_freq - start_freq) * chirp_t
            phase = 2 * np.pi * np.cumsum(freq) / sr
            chirp = 0.6 * np.sin(phase)
            
            # Apply envelope
            env = np.ones_like(chirp)
            attack = int(0.2 * len(chirp))
            release = int(0.3 * len(chirp))
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
            chirp = chirp * env
            
            audio[start_idx:end_idx] += chirp
            
    elif species_type == "nightingale":
        # Nightingales have complex, varied songs with trills and whistles
        # Create a mix of trills, whistles and clicks
        segments = np.random.randint(6, 12)
        
        for i in range(segments):
            segment_type = np.random.choice(['trill', 'whistle', 'click'])
            start_time = np.random.uniform(0, duration * 0.8)
            start_idx = int(start_time * sr)
            
            if segment_type == 'trill':
                # Fast repeated notes
                trill_duration = np.random.uniform(0.3, 0.8)
                trill_len = int(trill_duration * sr)
                end_idx = min(start_idx + trill_len, len(t))
                
                trill_freq = np.random.uniform(2500, 4000)
                trill_rate = np.random.uniform(15, 25)  # Notes per second
                
                trill_t = t[start_idx:end_idx] - t[start_idx]
                trill = 0.5 * np.sin(2 * np.pi * trill_freq * trill_t) * np.sin(2 * np.pi * trill_rate * trill_t)
                
                # Apply envelope
                env = np.ones_like(trill)
                attack = int(0.1 * len(trill))
                release = int(0.2 * len(trill))
                env[:attack] = np.linspace(0, 1, attack)
                env[-release:] = np.linspace(1, 0, release)
                trill = trill * env
                
                audio[start_idx:end_idx] += trill
                
            elif segment_type == 'whistle':
                # Pure tone whistle with frequency modulation
                whistle_duration = np.random.uniform(0.2, 0.5)
                whistle_len = int(whistle_duration * sr)
                end_idx = min(start_idx + whistle_len, len(t))
                
                start_freq = np.random.uniform(3000, 4500)
                end_freq = np.random.uniform(4500, 6000)
                
                whistle_t = np.linspace(0, 1, end_idx - start_idx)
                freq = start_freq + (end_freq - start_freq) * whistle_t
                phase = 2 * np.pi * np.cumsum(freq) / sr
                whistle = 0.6 * np.sin(phase)
                
                # Apply envelope
                env = np.ones_like(whistle)
                attack = int(0.1 * len(whistle))
                release = int(0.2 * len(whistle))
                env[:attack] = np.linspace(0, 1, attack)
                env[-release:] = np.linspace(1, 0, release)
                whistle = whistle * env
                
                audio[start_idx:end_idx] += whistle
                
            else:  # click
                # Short click
                click_duration = 0.05
                click_len = int(click_duration * sr)
                end_idx = min(start_idx + click_len, len(t))
                
                click = np.random.randn(end_idx - start_idx) * 0.3
                
                # Apply quick fade out
                env = np.ones_like(click)
                env = np.linspace(1, 0, len(env))
                click = click * env
                
                audio[start_idx:end_idx] += click
                
    elif species_type == "warbler":
        # Warblers have rapid, complex trills
        # Create a series of rapid frequency modulations
        n_phrases = np.random.randint(3, 6)
        
        for i in range(n_phrases):
            start_time = i * (duration / n_phrases)
            start_idx = int(start_time * sr)
            phrase_len = int((duration / n_phrases) * 0.9 * sr)
            end_idx = min(start_idx + phrase_len, len(t))
            
            # Create a warbling phrase
            base_freq = np.random.uniform(3000, 5000)
            mod_depth = np.random.uniform(500, 1000)
            mod_freq = np.random.uniform(8, 15)
            
            phrase_t = t[start_idx:end_idx] - t[start_idx]
            freq = base_freq + mod_depth * np.sin(2 * np.pi * mod_freq * phrase_t)
            phase = 2 * np.pi * np.cumsum(freq) / sr
            phrase = 0.6 * np.sin(phase)
            
            # Apply envelope
            env = np.ones_like(phrase)
            attack = int(0.1 * len(phrase))
            release = int(0.2 * len(phrase))
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
            phrase = phrase * env
            
            audio[start_idx:end_idx] += phrase
            
    elif species_type == "owl":
        # Owls have deep, resonant hoots
        # Create a series of low-frequency hoots
        n_hoots = np.random.randint(3, 6)
        
        for i in range(n_hoots):
            start_time = i * (duration / n_hoots)
            start_idx = int(start_time * sr)
            hoot_len = int((duration / n_hoots) * 0.7 * sr)
            end_idx = min(start_idx + hoot_len, len(t))
            
            # Create a hoot with harmonics
            base_freq = np.random.uniform(200, 400)
            hoot_t = t[start_idx:end_idx] - t[start_idx]
            
            # Fundamental frequency
            hoot = 0.7 * np.sin(2 * np.pi * base_freq * hoot_t)
            # Add harmonics
            hoot += 0.3 * np.sin(2 * np.pi * base_freq * 2 * hoot_t)
            hoot += 0.2 * np.sin(2 * np.pi * base_freq * 3 * hoot_t)
            
            # Apply bell-shaped envelope
            env = np.sin(np.pi * np.linspace(0, 1, end_idx - start_idx))
            hoot = hoot * env
            
            audio[start_idx:end_idx] += hoot
    
    # Add some background noise
    audio += 0.01 * np.random.randn(len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def create_sample_data():
    """Create sample data for bird species."""
    print("Creating sample data for demo...")
    
    # Sample rate
    sr = 22050
    
    # Create base data directory
    data_dir = "data"
    ensure_dir(data_dir)
    
    # Create species directories with their corresponding generation function
    species_dirs = {
        "Common_Blackbird": "blackbird",
        "European_Robin": "robin",
        "Nightingale": "nightingale",
        "Garden_Warbler": "warbler",
        "Tawny_Owl": "owl"
    }
    
    # Bird information for display in the app
    bird_info = {
        "Common_Blackbird": "The Common Blackbird has a melodious, flute-like song with a series of rich phrases.",
        "European_Robin": "The European Robin has a thin, high-pitched song with repeated patterns and clear notes.",
        "Nightingale": "The Nightingale is known for its complex, varied song with trills, whistles, and clicks.",
        "Garden_Warbler": "The Garden Warbler has a rapid, complex trill with continuous melodic phrases.",
        "Tawny_Owl": "The Tawny Owl produces deep, resonant hoots that carry over long distances."
    }
    
    # Save bird information
    info_dir = os.path.join(data_dir, "info")
    ensure_dir(info_dir)
    with open(os.path.join(info_dir, "bird_info.txt"), "w") as f:
        for species, info in bird_info.items():
            f.write(f"{species}|{info}\n")
    
    for species, bird_type in species_dirs.items():
        species_dir = os.path.join(data_dir, species)
        ensure_dir(species_dir)
        print(f"Created directory: {species_dir}")
        
        # Generate 10 samples for each species
        for i in range(10):
            # Generate a bird call based on species type
            duration = np.random.uniform(2.5, 4.0)  # Random duration between 2.5-4 seconds
            audio = generate_bird_call(bird_type, duration, sr)
            
            # Save as WAV file
            file_path = os.path.join(species_dir, f"{species}_{i+1}.wav")
            wavfile.write(file_path, sr, audio.astype(np.float32))
            print(f"Created: {file_path}")
    
    print("Sample data creation completed!")
    return data_dir

if __name__ == "__main__":
    create_sample_data()



