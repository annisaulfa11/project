from fastapi import FastAPI, File, UploadFile
import joblib
import librosa
import numpy as np
import uvicorn
# from google.cloud import storage

model_file = open('model.pkl','rb')
model = joblib.load(model_file)

app = FastAPI()
# storage_client = storage.Client()

@app.post('/predict')
async def predict(audio: UploadFile = File(...)):
    try:
        # Upload the audio file to Google Cloud Storage
        # bucket_name = 'audio-bucket-99'
        # filename = f'{audio.filename}-{uuid.uuid4()}.wav'  # Generate a unique filename
        # bucket = storage_client.bucket(bucket_name)
        # blob = bucket.blob(filename)
        # blob.upload_from_file(audio.file)

        signal, sr = librosa.load(audio.file, sr=22050)
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
            # 'filename': filename
        }

        return response
    except Exception as e:
        return {'error': 'Error processing audio file: ' + str(e)}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)