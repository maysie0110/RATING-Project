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

def get_cnn_model():

    #audio_temp/
    model_pretrained = VGG19(include_top=True, input_shape=(IMG_SIZE, IMG_SIZE, 3))#, weights="imagenet")
    x = layers.Dense(4096, activation='relu', name='predictions1', dtype='float32')(model_pretrained.layers[-2].output)
    output = layers.Dense(4, activation='sigmoid', name='predictions', dtype='float32')(x)
    model = Model(model_pretrained.input, output)

    # compile the model
    optimizer = keras.optimizers.SGD(learning_rate=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )

    model.summary()
    return model

def get_transformer_model():
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

    # x = layers.Dense(units=embed_dim, activation='gelu')(x)
    # x = layers.LayerNormalization()(x)

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
    # filepath = os.getcwd() + "/video_chkpt/video_classifier"
    filepath = os.getcwd() + "/seq_length_128/video_chkpt_128/video_classifier"
    transformer = get_transformer_model()
    transformer.load_weights(filepath)

    # evaluate the transformer model
    loss, accuracy, f1_score, precision, recall = transformer.evaluate(test_image_data, test_labels, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 score: {round(f1_score, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")

    #############################################################################################
    print("CNN Model")
    # filepath = os.getcwd() + "/temp/audio_classifier"
    # filepath = os.getcwd() + "/audio_chkpt/audio_classifier"
    # filepath = os.getcwd() + "/audio_temp/audio_classifier"
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

    y_pred  = predictLabelForGivenThreshold(results,0.5)
    print(classification_report(test_labels, y_pred,target_names=label_names))
    print(Accuracy(test_labels, y_pred))

if __name__ == '__main__':
    main()