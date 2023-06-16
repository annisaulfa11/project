import os
import tempfile
import librosa
import numpy as np
import logging
from keras.models import load_model
from google.cloud import storage
from google.cloud import pubsub
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"

def process_uploaded_audio(event, context):
    """Cloud Function to process uploaded audio files in Cloud Storage."""
    # Get the file details from the event
    file_data = event

    # Extract the bucket and file name
    bucket_name = file_data['bucket']
    file_name = file_data['name']

    # Create a temporary directory to store the audio file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file_name)

    try:
        # Download the audio file from Cloud Storage to the temporary directory
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(temp_file_path)

        # Process the audio file using your machine learning model
        predictions = process_audio(temp_file_path)

        # Log the predicted_emotion
        logging.info("Predicted emotion: %s", predictions)

        # Publish the predicted emotion to the Pub/Sub topic
        project_id = "capstone-project-387214"
        topic_name = "prediction_result"

        publisher = pubsub.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)

        message = {
            "emotion": predictions
        }
        message_data = json.dumps(message).encode("utf-8")

        future = publisher.publish(topic_path, data=message_data)
        future.result()

        print(f"Message published to Pub/Sub : {topic_name}")
        print(f"Message published to Pub/Sub : {message}")

        print(f"Processing complete for file '{file_name}'. Predictions is '{predictions}'.")
    except Exception as e:
        print(f"Error processing file '{file_name}': {str(e)}")
    finally:
        # Clean up the temporary directory and file
        if os.path.exists(temp_dir):
            os.remove(temp_file_path)
            os.rmdir(temp_dir)

def process_audio(file_path):
    # Load the H5 model
    model = load_model('model.h5')

    # Check file extension
    audio_extensions = ['wav', 'mp3', 'flac']
    file_extension = file_path.split('.')[-1]
    if file_extension not in audio_extensions:
        print('File format is: {}'.format(file_extension))
        return print("File format not supported")
    
    # Load audio file
    signal, sr = librosa.load(file_path, sr=44100)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sr,
                                n_mfcc=13,
                                n_fft=2048)

    # Normalize and transpose MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc.T
    
    # Pad or truncate MFCC
    num_mfcc = 13
    target_length = 615

    # Check the length of MFCC
    mfcc_length = len(mfcc)

    if mfcc_length < target_length:
        # Calculate the number of additional arrays needed
        num_additional_arrays = target_length - mfcc_length

        # Create an empty array with the shape (num_additional_arrays, num_mfcc)
        empty_arrays = np.zeros((num_additional_arrays, num_mfcc))

        # Concatenate the empty arrays to MFCC
        input_data = np.concatenate((mfcc, empty_arrays), axis=0)
    elif mfcc_length > target_length:
        # Truncate the mfcc array to the target length
        input_data = mfcc[:target_length]
    else:
        input_data = np.array(mfcc)
    
    # Reshape MFCC to fit model's input size
    input_data = np.reshape(input_data, (615, 13, 1))
    input_data = np.expand_dims(input_data, axis=0)

    # Make prediction
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)
    emotion_labels = ['neutral','happy','sad','angry','fearful','disgust']
    predicted_emotion = emotion_labels[predicted_label]

    return predicted_emotion
