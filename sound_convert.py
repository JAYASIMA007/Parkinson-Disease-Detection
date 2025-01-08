import os
import gc
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'data' folder and subfolders are created
def create_data_folders(categories, base_dir="data"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            
# Initialize categories and data directories
CATEGORIES = [class_name for class_name in os.listdir("data")]
DATADIR = "data"
DATADIR1 = "dataset"

# Create 'data' folder and subfolders
create_data_folders(CATEGORIES, DATADIR1)

def read_audio(conf, pathname, trim_long_data):
    print(pathname)
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0+conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels

def rename_file(img_name):
    img_name = img_name[:-4]  # Remove '.wav' extension
    img_name += ".jpg"        # Add '.jpg' extension
    return img_name

# Configuration for audio processing
class conf:
    sampling_rate = 44100
    duration = 3
    hop_length = 700 * duration
    fmin = 1
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration

# Process the audio files and save spectrogram images
lp = 0
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    path1 = os.path.join(DATADIR1, category)  # Save images in corresponding 'data' subfolder
    
    for i, fn in enumerate(os.listdir(path)):
        print(f"Processing {i}: {fn}")
        pathi = os.path.join(path, fn)
        filename = rename_file(pathi)
        
        # Convert the audio to mel-spectrogram
        x = read_as_melspectrogram(conf, pathi, trim_long_data=False, debug_display=False)
        
        lp += 1
        filename = str(lp)  # Assign sequential numbers to filenames
        
        plt.imshow(x, interpolation='nearest')
        for ju in range(2):
            plt.savefig(os.path.join(path1, f"{ju}_{filename}.jpg"))
        
        # Close plot and clean up memory
        plt.close()
        del x
        gc.collect()
