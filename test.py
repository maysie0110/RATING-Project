from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from keras_preprocessing import image

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import os

from audio_classification import get_cnn_model
from transformer_train import get_compiled_model
from early_fusion import get_early_fusion_model
from late_fusion import Accuracy, predictLabelForGivenThreshold
## global variables

train_dir = os.getcwd() + "/train_data/"
val_dir = os.getcwd() + "/val_data/"
test_dir = os.getcwd() + "/test_data/"

# Hyperparameters
MAX_SEQ_LENGTH = 128
FRAME_GAP = 11
NUM_FEATURES = 1024
IMG_SIZE = 224

EPOCHS = 10
BATCH_SIZE = 64

train_image_data, train_labels = np.load("seq_length_128/extracted_data/train_data.npy"), np.load("seq_length_128/extracted_data/train_labels.npy")
val_image_data, val_labels = np.load("seq_length_128/extracted_data/val_data.npy"), np.load("seq_length_128/extracted_data/val_labels.npy")
test_image_data, test_labels = np.load("seq_length_128/extracted_data/test_data.npy"), np.load("seq_length_128/extracted_data/test_labels.npy")

train_spectrograms = glob.glob('audio_classification/extracted_train_spectrogram/*')
val_spectrograms = glob.glob('audio_classification/extracted_val_spectrogram/*')
test_spectrograms = glob.glob('audio_classification/extracted_test_spectrogram/*')

train_audio_data = []
val_audio_data = []
test_audio_data = []

for f in train_spectrograms:
    img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
    img = image.img_to_array(img)
    train_audio_data.append(img)
    
train_audio_data = np.array(train_audio_data)

for f in val_spectrograms:
    img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
    img = image.img_to_array(img)
    val_audio_data.append(img)
    
val_audio_data = np.array(val_audio_data)

for f in test_spectrograms:
    img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
    img = image.img_to_array(img)
    test_audio_data.append(img)
    
test_audio_data = np.array(test_audio_data)

# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    count = 0
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            if count % 5 == 0:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

            count=count+1
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def main():
    print("Transformer-based Model")
    # filepath = os.getcwd() + "/tmp_3_4/video_classifier"
    filepath = os.getcwd() + "/seq_length_128/video_chkpt_2/video_classifier"
    # filepath = os.getcwd() + "/seq_length_128/video_chkpt/video_classifier"
    transformer = get_compiled_model()
    transformer.load_weights(filepath)

    # evaluate the transformer model
    loss, accuracy, f1_score, precision, recall = transformer.evaluate(test_image_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    ############################################################################################
    print("CNN Model")
    filepath = os.getcwd() + "/audio_classification/audio_chkpt_sgd/audio_classifier"
    cnn = get_cnn_model()
    cnn.load_weights(filepath)

    # evaluate the cnn model
    loss, accuracy, f1_score, precision, recall = cnn.evaluate(test_audio_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    #############################################################################################
    print("Early Fusion")
    # filepath = os.getcwd() + "/early_fusion_chkpt/classifier"
    filepath = os.getcwd() + "/early_fusion_temp_2/classifier"
    early_fusion = get_early_fusion_model()
    early_fusion.load_weights(filepath)

    # evaluate the early fusion multimodal
    loss, accuracy, f1_score, precision, recall = early_fusion.evaluate([test_image_data, test_audio_data], test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    label_names = ['Mature', 'Slapstick', 'Gory', 'Sarcasm']
    y_pred = early_fusion.predict([test_image_data, test_audio_data])
    y_pred = y_pred.round()

    # y_pred = (y_pred > 0.5).astype(int)
    print(classification_report(test_labels, y_pred,target_names=label_names))
    # print(accuracy_score(test_labels, y_pred))
    # print(Accuracy(test_labels, y_pred))

if __name__ == '__main__':
    main()