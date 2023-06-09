from fastapi import FastAPI, File, UploadFile
import joblib
import librosa
import numpy as np
import uvicorn
import uuid
from google.cloud import storage
import os, io
import requests
from tensorflow.keras.models import load_model



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"

# Load the H5 model
model = load_model('model.h5')

# model_file = open('model.pkl','rb')
# model = joblib.load(model_file)

app = FastAPI()
storage_client = storage.Client()

@app.post('/predict')
async def predict(audio: UploadFile = File(...)):
    try:
        # Upload the audio file to Google Cloud Storage
        bucket_name = 'audio-bucket-98'
        filename = f'{uuid.uuid4()}.wav'  # Generate a unique filename
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_file(audio.file)

        # Get the URL of the uploaded audio file
        audio_url = "https://storage.googleapis.com/" + bucket_name + "/" + filename
        
        # Download the audio file from the URL
        response = requests.get(audio_url)
        audio_data = response.content

        # Load the audio data using soundfile
        with io.BytesIO(audio_data) as audio_file:
            signal, sr = librosa.load(audio_file, sr=22050)

        # Extract the MFCC from the audio
        mfcc = librosa.feature.mfcc(y=signal,
                                    sr=sr,
                                    n_mfcc=13,
                                    n_fft=2048,
                                    hop_length=512)

        # Normalize the MFCC features
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc = mfcc.T

        num_mfcc = 13

        # Check the length of mfcc
        mfcc_length = len(mfcc)
        if mfcc_length < 308:
            # Calculate the number of additional arrays needed
            num_additional_arrays = 308 - mfcc_length

            # Create an empty array with the shape (num_additional_arrays, num_mfcc)
            empty_arrays = np.zeros((num_additional_arrays, num_mfcc))

            # Concatenate the empty arrays to mfcc
            input_data = np.concatenate((mfcc, empty_arrays), axis=0)
        else:
            input_data = np.array(mfcc)

        input_data = np.reshape(input_data, (308, 13, 1))
        input_data = np.expand_dims(input_data, axis=0)
        predictions = model.predict(input_data)
        predicted_label = np.argmax(predictions)
        emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        predicted_emotion = emotion_labels[predicted_label]

        response = {
            'predicted_emotion': predicted_emotion,
            'filename': filename
        }

        return response
    except Exception as e:
        return {'error': 'Error processing audio file: ' + str(e)}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=3000)