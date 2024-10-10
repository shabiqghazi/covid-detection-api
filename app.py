from flask import Flask, request, redirect, url_for, jsonify
from flask_restful import Api, Resource, reqparse, inputs
import librosa
import numpy as np
import os
import requests
import uuid
from werkzeug.utils import secure_filename
import json
import wave
import subprocess
# from audio_denoiser.AudioDenoiser import AudioDenoiser
import pickle

app = Flask(__name__)

# denoiser = AudioDenoiser()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder unggahan ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hanya izinkan file dengan ekstensi tertentu
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'pcm'}

parser = reqparse.RequestParser()
parser.add_argument('file', type=inputs.regex('^[0-9]+$'))

# def preprocess(filename):
#     in_audio_file = filename
#     out_audio_file = filename
#     denoiser.process_audio_file(in_audio_file, out_audio_file)
#     return out_audio_file


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_uuid():
    # Menghasilkan UUID4 yang unik
    unique_id = str(uuid.uuid4())
    return unique_id

def check(filename):
    try:
        with open(filename, 'rb') as file:
            filename = filename.split('\\')[1]
            files = {
                # 'filename': filename.split('\\')[len(filename) - 1], 
                'file': (filename,file)
            }
            print(filename)
            # Mengirim POST request dengan file dan header kustom
            # response = requests.post('http://192.168.1.76:8080/run', files=files)

            result = subprocess.run(['./main.exe', 'uploads/' + filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print(result.stdout)
            return result.stdout
            # print("Status Code:", response.status_code)
            # print("Response Text:", response.text)
    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Proses gagal dengan kode keluar: {e.returncode}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
    
    # return response.text

def pcm_to_wav(pcm_file_path, wav_file_path, channels, sample_rate, sample_width):
    # Membuka file PCM
    with open(pcm_file_path, 'rb') as pcmfile:
        pcm_data = pcmfile.read()

    # Membuat file WAV dengan parameter yang ditentukan
    with wave.open(wav_file_path, 'wb') as wavfile:
        wavfile.setnchannels(channels)
        wavfile.setsampwidth(sample_width)
        wavfile.setframerate(sample_rate)
        wavfile.writeframes(pcm_data)

    print(f"File WAV berhasil dibuat: {wav_file_path}")
def predict(input):
    # Memuat model dari file
    with open('knn_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Sekarang Anda bisa menggunakan loaded_model untuk prediksi
    predictions = loaded_model.predict(input)
    return predictions
@app.route('/', methods=['GET'])
def home():
    return "Hello World!"

@app.route('/get_signal', methods=['POST'])
def get_signal():
    if 'file' not in request.files:
        print('No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        print('No file selected for uploading')
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = generate_uuid()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath + '.wav')

        # pcm_to_wav(filepath, filepath + '.wav', 1, 48000, 2)
        # filepath = preprocess(filepath + '.wav')

        signal, sr = librosa.load(filepath + '.wav', sr=None)
        signal = signal / np.max(np.abs(signal))

        y = np.array(signal, dtype=np.float32)
        y_list = y.astype(float).tolist()

        start = 0

        for i in range(len(y_list)):
            if abs(y_list[i]) > 0.01 and abs(y_list[i+2]) > 0.01 and abs(y_list[i+4]) > 0.01:
                start = i
                break

        y_list = y_list[start:start+48000]
        series_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + '_cough.txt')
        np.savetxt(series_path, y_list, fmt='%f')
        
        response = check(series_path)

        if os.path.exists(filepath):
            os.remove(filename)
            print(f"File '{filepath}' berhasil dihapus.")
        
        response = response.split(',')
        status = predict([[float(response[1]), float(response[2]), float(response[3])]])
        status = "Positif" if int(status[0]) == 1 else "Negatif"
        if response[4] == "Tidak Konklusif" :
            status = "Tidak Konklusif"
        elif response[4] != status :
            status = "Tidak Konklusif"
        # return data
        data = {
            'filename': filename,
            'dimension': response[1],
            'size': response[2],
            'dispersi': response[3],
            'status': status,
        }
        return data

    return 'Invalid file type! Only .mp3, .wav, .ogg are allowed.'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)