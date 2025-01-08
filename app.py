from flask import Flask, render_template, request, jsonify
import os
import librosa
import librosa.display
import tensorflow as tf
import numpy as np
import wavio
from transformers import pipeline
import matplotlib.pyplot as plt
import cv2
import gc

app = Flask(__name__)

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

# Initialize the global lp variable
lp = 0  # Image index variable

# Function to rename files
def rename_file(img_name):
    img_name = os.path.basename(img_name)  # This will get the file name
    img_name = os.path.splitext(img_name)[0]  # Remove the extension
    img_name += ".jpg"  # Add the .jpg extension
    return img_name

# Audio preprocessing functions
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if len(y) > 0:
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

def save_image_from_sound(img_path):
    global lp  # Use the global variable lp
    filename = rename_file(img_path)
    x = read_as_melspectrogram(conf, img_path, trim_long_data=False, debug_display=True)
    
    lp += 1
    filename = str(lp)
    
    plt.imshow(x, interpolation='nearest')
    plt.savefig("static/1.png")
    plt.close()
    del x
    gc.collect()

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread("static/1.png", 1)
    img_array = cv2.Canny(img_array, threshold1=10, threshold2=10)
    img_array = cv2.medianBlur(img_array, 1)
    img_array = cv2.equalizeHist(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(new_array, axis=0)

def analyze_sound(file_path):
    save_image_from_sound(file_path)
    prediction = model.predict(prepare("static/1.png"))
    prediction = list(prediction[0])
    result = CATEGORIES[prediction.index(max(prediction))]
    
    # Generate feedback using GPT-2
    feedback_prompt = f"Provide detailed feedback and suggestions for the sound classification result: {result}"
    generated_feedback = generator(feedback_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return result, generated_feedback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['audioFile']
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        
        result, feedback = analyze_sound(file_path)
        
        return jsonify({
            'result': result,
            'feedback': feedback
        })
    else:
        return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
