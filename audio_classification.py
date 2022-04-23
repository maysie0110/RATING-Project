import os
import glob
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from keras_preprocessing import image
import numpy as np

# from utilities import f1_m, recall_m, precision_m
from tensorflow.keras import backend as K

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# Hyperparameters
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32

train_spectrograms = glob.glob('extracted_train_spectrogram/*')
val_spectrograms = glob.glob('extracted_val_spectrogram/*')
test_spectrograms = glob.glob('extracted_test_spectrogram/*')

train_labels = np.load("extracted_data/train_labels.npy")
val_labels = np.load("extracted_data/val_labels.npy")
test_labels = np.load("extracted_data/test_labels.npy")

# def get_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     flatten = model.predict(x)
#     return list(flatten[0])

def get_data():
    train_data = []
    val_data = []
    test_data = []

    for f in train_spectrograms:
        img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
        img = image.img_to_array(img)
        train_data.append(img)
        
    train_data = np.array(train_data)
    print(train_data.shape)

    for f in val_spectrograms:
        img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
        img = image.img_to_array(img)
        val_data.append(img)
        
    val_data = np.array(val_data)
    print(val_data.shape)

    for f in test_spectrograms:
        img = image.load_img(f, target_size= (IMG_SIZE,IMG_SIZE))
        img = image.img_to_array(img)
        test_data.append(img)
        
    test_data = np.array(test_data)
    print(test_data.shape)
    return train_data, val_data, test_data

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
    # classes = 4

    # # Create a VGG19 model, and removing the last layer that is classifying 1000 images. 
    # # # This will be replaced with images classes we have. 
    # base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    # # freeze all layers in the the base model
    # base_model.trainable = False

    # # Model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    # x = layers.Flatten()(base_model.output) #Output obtained on vgg16 is now flattened. 
    # outputs = layers.Dense(classes, activation="sigmoid")(x)

    # #Creating model object 
    # model = keras.Model(inputs=base_model.input, outputs=outputs)

    # optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # # compile the model
    # model.compile(
    #     optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    # )
    # model.summary()
    # return model

    # model = keras.Sequential()
    # model.add(layers.Conv2D(input_shape=(IMG_SIZE,IMG_SIZE,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=4096,activation="relu"))
    # model.add(layers.Dense(units=4096,activation="relu"))
    # model.add(layers.Dense(units=4, activation='sigmoid'))



    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='sigmoid'))

    # optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # optimizer = keras.optimizers.SGD(learning_rate=10.0 ** (-1), decay=1e-5)
    optimizer = keras.optimizers.SGD(learning_rate=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.SGD(learning_rate=0.0000001, decay=1e-6, momentum=0.9)

    # compile the model
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )

    model.summary()
    return model


def run_experiment(train_data,val_data,test_data):
    log_dir = "logs/fit/audio_chkpt" 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = os.getcwd() + "/audio_chkpt/audio_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True,
        monitor='val_f1_m',
        mode='max',
        save_best_only=True,
        verbose = True
    )

    # with tf.device('/device:GPU:0'):
    model = get_cnn_model()
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data,val_labels),
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )

    model.load_weights(filepath)

    # evaluate the model
    # _, accuracy = model.evaluate(test_data, test_labels)
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")


    return model

def main():
    train_data, val_data, test_data = get_data()
    trained_model = run_experiment(train_data,val_data,test_data)


if __name__ == '__main__':
    main()