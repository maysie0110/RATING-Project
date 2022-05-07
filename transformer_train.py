from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os


# Hyperparameters
MAX_SEQ_LENGTH = 128
FRAME_GAP = 11
NUM_FEATURES = 1024
IMG_SIZE = 224

EPOCHS = 10

train_data, train_labels = np.load("seq_length_128/extracted_data/train_data.npy"), np.load("seq_length_128/extracted_data/train_labels.npy")
val_data, val_labels = np.load("seq_length_128/extracted_data/val_data.npy"), np.load("seq_length_128/extracted_data/val_labels.npy")
test_data, test_labels = np.load("seq_length_128/extracted_data/test_data.npy"), np.load("seq_length_128/extracted_data/test_labels.npy")


# Embedding Layer
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim
        })
        return config


# Subclassed layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, resize=(IMG_SIZE, IMG_SIZE)):
    count = 0
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            if count % FRAME_GAP == 0:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

            count=count+1
            
            if len(frames) == MAX_SEQ_LENGTH:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

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

def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = 4

    inputs = keras.Input(shape=(None, None), name="input")
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)

    x = layers.Dense(units=embed_dim, activation='gelu')(x)
    x = layers.LayerNormalization()(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)



    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # compile the model
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )
    
    model.summary()
    return model


def run_experiment():
    log_dir = "logs/fit/video_chkpt_128_3" 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = os.getcwd() + "/seq_length_128/video_chkpt_3/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, 
        monitor='val_f1_m',
        mode='max',
        save_best_only=True,
        verbose = 1
    )

    with tf.device('/device:CPU:0'):
        model = get_compiled_model()
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

def prepare_single_video(frames):
    feature_extractor = build_feature_extractor()
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate((frames, padding))

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features


def main():
    trained_model = run_experiment()

    # predict = []

    # for path in test_df['path']:
    #     frames = load_video(os.path.join("test", path))
    #     frame_features = prepare_single_video(frames)
    #     y_pred = trained_model.predict(frame_features)[0]
        
    #     # round probabilities to class labels
    #     y_pred = y_pred.round()
    #     predict.append(y_pred)

    # recall = recall_score(y_true=test_labels, y_pred=predict, average='weighted')
    # print("Recall", recall)
    # precision = precision_score(y_true=test_labels, y_pred=predict, average='weighted')
    # print("Precision", precision)
    # f1 = f1_score(y_true=test_labels, y_pred=predict, average='weighted')
    # print("F1 score", f1)




if __name__ == '__main__':
    main()