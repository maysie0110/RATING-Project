import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from utilities import f1_m, recall_m, precision_m
from keras_preprocessing import image
from tensorflow import keras

from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os 
import glob
from audio_classification import get_cnn_model
from transformer_train import get_compiled_model

# Hyperparameters
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32

MAX_SEQ_LENGTH = 128
FRAME_GAP = 11
NUM_FEATURES = 1024



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


def get_late_fusion(transformer, cnn):
    ## Extract the probabilities from each classifier for the late fusion
    res1 = transformer.predict(test_image_data)
    # print(res1)
    res2 = cnn.predict(test_audio_data)
    # print(res2)
    all_res = np.array([res1,res2])
    all_res = all_res.sum(0)
    return all_res

def predictLabelForGivenThreshold(results, threshold):
    # y_pred=[]
    # for sample in results:
    #     y_pred.append([1 if i>=threshold else 0 for i in sample ] )
    # return np.array(y_pred)

    predictions = []
    for key,values in enumerate(list(results)):
        temp = []
        for v in values:
            v = (v >= threshold).astype(int)
            temp.append(v)
        predictions.append(temp) 
    predictions = np.array(predictions)

    return predictions

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

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
    print("Late Fusion")
    results = get_late_fusion(transformer,cnn)
    label_names = ['Mature', 'Slapstick', 'Gory', 'Sarcasm']

    y_pred  = predictLabelForGivenThreshold(results,0.2)
    print(classification_report(test_labels, y_pred,target_names=label_names))
    print(Accuracy(test_labels, y_pred))

    y_pred  = predictLabelForGivenThreshold(results,0.5)
    print(classification_report(test_labels, y_pred,target_names=label_names))
    print(Accuracy(test_labels, y_pred))

    y_pred  = predictLabelForGivenThreshold(results,0.7)
    print(classification_report(test_labels, y_pred,target_names=label_names))
    print(Accuracy(test_labels, y_pred))

if __name__ == '__main__':
    main()