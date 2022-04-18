import os
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# from utilities import f1_m, recall_m, precision_m
from tensorflow.keras import backend as K

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# Hyperparameters
IMG_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 32

train_data, train_labels = np.load("extracted_audio_data/train_data.npy", allow_pickle=True), np.load("extracted_data/train_labels.npy")
val_data, val_labels = np.load("extracted_audio_data/val_data.npy", allow_pickle=True), np.load("extracted_data/val_labels.npy")
test_data, test_labels = np.load("extracted_audio_data/test_data.npy", allow_pickle=True), np.load("extracted_data/test_labels.npy")

# def get_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     flatten = model.predict(x)
#     return list(flatten[0])


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_cnn_model():
    classes = 4

    # Create a VGG19 model, and removing the last layer that is classifying 1000 images. 
    # # This will be replaced with images classes we have. 
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    # freeze all layers in the the base model
    base_model.trainable = False

    # Model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    x = layers.Flatten()(base_model.output) #Output obtained on vgg16 is now flattened. 
    outputs = layers.Dense(classes, activation="sigmoid")(x)

    #Creating model object 
    model = keras.Model(inputs=base_model.input, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # compile the model
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )
    model.summary()
    return model


def run_experiment():
    log_dir = "logs/fit/temp" 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = os.getcwd() + "/temp/audio_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    with tf.device('/device:CPU:0'):
        model = get_cnn_model()
        history = model.fit(
            train_data,
            train_labels,
            validation_data=(val_data,val_labels),
            epochs=EPOCHS,
            callbacks=[checkpoint, tensorboard_callback],
        )

    model.load_weights(filepath)
    # _, accuracy = model.evaluate(test_data, test_labels)
    # evaluate the model
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    return model

def main():
   trained_model = run_experiment()


if __name__ == '__main__':
    main()