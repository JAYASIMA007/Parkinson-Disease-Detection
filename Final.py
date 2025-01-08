import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import gc
import wavio
from transformers import pipeline

# Define settings for audio processing
class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 3
    hop_length = 700 * duration  # to make time steps 128
    fmin = 1
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration

# Load the pre-trained model for sound classification
model = tf.keras.models.load_model("SOUND.model")
DATADIR = "data"
CATEGORIES = os.listdir(DATADIR)

# Initialize Hugging Face GPT-2 text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Tkinter setup
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window initially

import os

def rename_file(img_name):
    # Extract the file name without extension
    img_name = os.path.basename(img_name)  # This will get the file name
    img_name = os.path.splitext(img_name)[0]  # Remove the extension
    img_name += ".jpg"  # Add the .jpg extension
    return img_name

def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)  # trim silence
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0 + conf.samples]
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
    return spectrogram.astype(np.float32)

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    return audio_to_melspectrogram(conf, x)

import gc
lp = 0  # Initialize lp here

def save_image_from_sound(img_path):
    global lp  # Declare lp as global to modify the global variable
    filename = rename_file(img_path)
    x = read_as_melspectrogram(conf, img_path, trim_long_data=False, debug_display=True)
    
    lp += 1  # Increment lp
    filename = str(lp)
    
    plt.imshow(x, interpolation='nearest')
    plt.savefig("1.png")
    
    plt.close()
    del x
    gc.collect()

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread("1.png", 1)
    img_array = cv2.Canny(img_array, threshold1=10, threshold2=10)
    img_array = cv2.medianBlur(img_array, 1)
    img_array = cv2.equalizeHist(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(new_array, axis=0)

def analyze_sound(file_path):
    save_image_from_sound(file_path)
    prediction = model.predict(prepare("1.png"))
    prediction = list(prediction[0])
    result = CATEGORIES[prediction.index(max(prediction))]
    
    # Generate feedback using GPT-2
    feedback_prompt = f"Provide detailed feedback and suggestions for the sound classification result: {result}"
    generated_feedback = generator(feedback_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Show result with generated feedback
    messagebox.showinfo("RESULT", f"Classification Result: {result}\n\nGenerated Feedback:\n{generated_feedback}")

def record_audio():
    fs = 44100  # Sample rate
    duration = 3  # seconds
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    wavio.write("real_time_recording.wav", recording, fs, sampwidth=2)
    analyze_sound("real_time_recording.wav")

# Function to handle user choice
def ask_user_choice():
    while True:
        response = messagebox.askyesno("Audio Input", "Do you want to record real-time audio?")
        if response:
            record_audio()  # Record audio for 3 seconds and analyze
        else:
            file_path = filedialog.askopenfilename(title="Select Audio File")
            if file_path:
                analyze_sound(file_path)  # Analyze selected audio file
            else:
                break  # Break the loop if no file is selected
        
        # Ask user if they want to continue or exit
        continue_response = messagebox.askyesno("Continue", "Do you want to continue?")
        if not continue_response:
            break  # Exit the loop if the user doesn't want to continue

# Start asking the user
ask_user_choice()

root.mainloop()
