import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from utilities import f1_m, recall_m, precision_m

def get_cnn_model():
    classes = 4

    # Create a VGG19 model, and removing the last layer that is classifying 1000 images. 
    # # This will be replaced with images classes we have. 
    base_model = VGG19(weights='imagenet', include_top=False)
    # freeze all layers in the the base model
    base_model.trainable = False

    # Model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    x = Flatten()(base_model.output) #Output obtained on vgg16 is now flattened. 
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

def get_late_fusion_model(model_1,model_2):
    classes = 4
    x1 = model_1.output
    x2 = model_2.output
    
    # LATE FUSION
    x = concatenate([x1, x2])
    x = Sequential()(x)
    # x = Dense(x.shape[1], activation='relu')(x) #12
    # x = Dropout(DROPOUT_PROB)(x)
    # x = Dense(ceil(x.shape[1]/2), activation='relu')(x) #8
    # x = Dropout(DROPOUT_PROB)(x)
    # predictions = Dense(classes, activation='softmax')(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)

    model = keras.Model(inputs=[model_1.input, model_2.input], outputs=outputs) # Inputs go into two different layers

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # compile the model
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m]
    )
    
    model.summary()
    return model


def run_experiment():
    log_dir = "logs/fit/fusion_temp" 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = os.getcwd() + "/fusion_temp/classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
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

def main():
    print("Late fusion Transformer-CNN")

    filepath = os.getcwd() + "/temp/audio_classifier"
    cnn = get_cnn_model()
    cnn.load_weights(filepath)


    filepath = os.getcwd() + "/tmp_3_4/video_classifier"
    transformer = get_transformer_model()
    model.load_weights(filepath)

    model = get_late_fusion_model(cnn, transformer)

if __name__ == '__main__':
    main()