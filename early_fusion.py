import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from utilities import f1_m, recall_m, precision_m

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from keras_preprocessing import image
from tensorflow import keras
import numpy as np
import os 
import glob


# Hyperparameters
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32


MAX_SEQ_LENGTH = 128
FRAME_GAP = 11
NUM_FEATURES = 1024



train_image_data, train_labels = np.load("extracted_data/train_data.npy"), np.load("extracted_data/train_labels.npy")
val_image_data, val_labels = np.load("extracted_data/val_data.npy"), np.load("extracted_data/val_labels.npy")
test_image_data, test_labels = np.load("extracted_data/test_data.npy"), np.load("extracted_data/test_labels.npy")

train_spectrograms = glob.glob('extracted_train_spectrogram/*')
val_spectrograms = glob.glob('extracted_val_spectrogram/*')
test_spectrograms = glob.glob('extracted_test_spectrogram/*')

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

def get_early_fusion_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = 4

    # Create Transformer-based model
    inputs_rgb = keras.Input(shape=(None, None), name="input_image")
    x1 = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs_rgb)
    x1 = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x1)
    x1 = layers.Dense(units=embed_dim, activation='gelu')(x1)
    x1 = layers.LayerNormalization()(x1)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x1 = layers.Dropout(0.5)(x1)

    #Create CNN model
    inputs_spec = keras.Input(shape=(IMG_SIZE,IMG_SIZE,3), name="input_spectrogram")
    # x2 = keras.Sequential()(inputs_spec)
    x2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3))(inputs_spec)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Dropout(0.25)(x2)
    x2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x2)
    x2 = layers.Conv2D(128, (3, 3), activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(512, activation='relu')(x2)
    x2 = layers.Dropout(0.5)(x2)

    # EARLY FUSION
    x = layers.concatenate([x1, x2])
    # x = keras.Sequential()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)

    
    model = keras.Model(inputs=[inputs_rgb, inputs_spec], outputs=outputs) # Inputs go into two different layers

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # optimizer = keras.optimizers.SGD(learning_rate=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
    # compile the model
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )
    
    model.summary()
    return model

def get_audio_data():
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

def run_experiment(train_audio_data, val_audio_data, test_audio_data):
    log_dir = "logs/fit/early_fusion_temp" 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = os.getcwd() + "/early_fusion_temp/classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, monitor='val_f1_m',
        mode='max',
        save_best_only=True,
        verbose = True
    )

    # with tf.device('/device:CPU:0'):
        # model = get_early_fusion_model(transformer,cnn)
    model = get_early_fusion_model()
    history = model.fit(
        [train_image_data, train_audio_data],
        train_labels,
        validation_data=([val_image_data, val_audio_data],val_labels),
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )

    model.load_weights(filepath)
    # _, accuracy = model.evaluate(test_data, test_labels)
    # evaluate the model
    loss, accuracy, f1_score, precision, recall = model.evaluate([test_image_data, test_audio_data], test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    return model

def main():
    train_audio_data, val_audio_data, test_audio_data = get_audio_data()
    trained_model = run_experiment(train_audio_data, val_audio_data, test_audio_data)


if __name__ == '__main__':
    main()