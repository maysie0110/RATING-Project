from importlib.resources import path
from traceback import FrameSummary
from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import argparse
import sys


# global variables

train_dir = os.getcwd() + "/train_data/"
val_dir = os.getcwd() + "/val_data/"
test_dir = os.getcwd() + "/test_data/"

# Hyperparameters
MAX_SEQ_LENGTH = 128
FRAME_GAP = 11
NUM_FEATURES = 1024
IMG_SIZE = 224

EPOCHS = 10


# Create a dataframe which contains multiclass classification content annotations for each video scene used in the training set.
train_df = pd.read_csv('train-updated.csv', dtype={'combination': object}).iloc[:,1:]
train_df["path"] = train_dir + train_df["Video ID"]+ ".0" + train_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the validation set.
val_df = pd.read_csv('val.csv', dtype={'combination': object}).iloc[:,1:]
val_df["path"] = val_dir + val_df["Video ID"]+ ".0" + val_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the test set.
test_df = pd.read_csv('test-updated.csv', dtype={'combination': object}).iloc[:,1:]
test_df["path"] = test_dir + test_df["Video ID"]+ ".0" + test_df["Scene_ID"].astype(str) + ".mp4"


#path to save extracted frames
extracted_train_path = os.getcwd() + "/extracted_train_frames/"
extracted_val_path = os.getcwd() + "/extracted_val_frames/"
extracted_test_path = os.getcwd() + "/extracted_test_frames/"
os.mkdir(extracted_train_path)
os.mkdir(extracted_val_path)
os.mkdir(extracted_test_path)

#directory to save extracted data
extracted_data_dir = os.getcwd() + "/extracted_data/"
os.mkdir(extracted_data_dir)

# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(save_dir, path, resize=(IMG_SIZE, IMG_SIZE)):
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
                
                save_path = save_dir + str(count) + '.jpg'
                print(save_path)
                #Save extracted frames to file
                cv2.imwrite(save_path, frame)

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


def prepare_all_videos(df, save_dir):
    feature_extractor = build_feature_extractor()
    num_samples = len(df)
    video_paths = df["path"].values.tolist()
    video_ids = df["Video ID"].values.tolist()
    scene_ids = df["Scene_ID"].values.tolist()

    labels = []
    for x in df["combination"].values:
        lst = list(map(int, x))
        arr = np.asarray(lst)
        labels.append(arr)
    labels = np.reshape(labels,(len(labels),4))

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        print(path)
        video_id = video_ids[idx]
        scene_id = scene_ids[idx]
        video_dir = save_dir + str(video_id) + '.' + str(scene_id) + '/'
        

        os.mkdir(video_dir)
        print(video_dir)
        
        # Gather all its frames and add a batch dimension.
        frames = load_video(video_dir, path)

        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate((frames, padding))

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )

                else:
                    temp_frame_features[i, j, :] = 0.0

        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels

def load_frames(dir):
    frames = []

    for file in os.scandir(dir):
        if file.is_file():
            print(file.path)
            frame = cv2.imread(file.path)
            frames.append(frame)
            
    return np.array(frames)


def extract_all_frames(df, save_dir):
    feature_extractor = build_feature_extractor()
    num_samples = len(df)
    video_paths = df["path"].values.tolist()
    video_ids = df["Video ID"].values.tolist()
    scene_ids = df["Scene_ID"].values.tolist()

    labels = []
    for x in df["combination"].values:
        lst = list(map(int, x))
        arr = np.asarray(lst)
        labels.append(arr)
    labels = np.reshape(labels,(len(labels),4))

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        print(path)
        video_id = video_ids[idx]
        scene_id = scene_ids[idx]
        video_dir = save_dir + str(video_id) + '.' + str(scene_id) + '/'
        print(video_dir)

        # Gather all its frames and add a batch dimension.
        frames = load_frames(video_dir)

        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(frames)
            padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
            frames = np.concatenate((frames, padding))

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )

                else:
                    temp_frame_features[i, j, :] = 0.0

        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--extract_frames", help="If you want to extract frames from video, set to True", action=argparse.BooleanOptionalAction)
        args = parser.parse_args()

        extract_frames = args.extract_frames
        print(extract_frames)
        
        if extract_frames: #extract frames + features
            print("Extract frames and features")
            train_data, train_labels = prepare_all_videos(train_df, extracted_train_path)
            val_data, val_labels = prepare_all_videos(val_df, extracted_val_path)
            test_data, test_labels = prepare_all_videos(test_df, extracted_test_path)
        else: #frames are already extracted and stores, extract features
            print("Extract features from extracted frames")
            train_data, train_labels = extract_all_frames(train_df, extracted_train_path)
            val_data, val_labels = extract_all_frames(val_df, extracted_val_path)
            test_data, test_labels = extract_all_frames(test_df, extracted_test_path)


        print(f"Frame features in train set: {train_data.shape}")
        print(f"Train labels: {train_labels}")


        
        # Save extracted data to file
        with open(extracted_data_dir + 'train_data.npy', 'wb') as f:
            np.save(f, train_data)

        with open(extracted_data_dir + 'train_labels.npy', 'wb') as f:
            np.save(f, train_labels)

        with open(extracted_data_dir + 'val_data.npy', 'wb') as f:
            np.save(f, val_data)

        with open(extracted_data_dir + 'val_labels.npy', 'wb') as f:
            np.save(f, val_labels)

        with open(extracted_data_dir + 'test_data.npy', 'wb') as f:
            np.save(f, test_data)

        with open(extracted_data_dir + 'test_labels.npy', 'wb') as f:
            np.save(f, test_labels)
            
    except:
        e = sys.exc_info()[0]
        print(e)

    # train_data, train_labels = prepare_all_videos(train_df, extracted_train_path)
    # val_data, val_labels = prepare_all_videos(val_df, extracted_val_path)
    # test_data, test_labels = prepare_all_videos(test_df, extracted_test_path)


    # # train_data, train_labels = extract_all_frames(train_df, extracted_train_path)
    # # val_data, val_labels = extract_all_frames(val_df, extracted_val_path)
    # # test_data, test_labels = extract_all_frames(test_df, extracted_test_path)

    # print(f"Frame features in train set: {train_data.shape}")
    # print(f"Train labels: {train_labels}")


    # # Save extracted data to file
    # with open('extracted_data/train_data.npy', 'wb') as f:
    #     np.save(f, train_data)

    # with open('extracted_data/train_labels.npy', 'wb') as f:
    #     np.save(f, train_labels)

    # with open('extracted_data/val_data.npy', 'wb') as f:
    #     np.save(f, val_data)

    # with open('extracted_data/val_labels.npy', 'wb') as f:
    #     np.save(f, val_labels)

    # with open('extracted_data/test_data.npy', 'wb') as f:
    #     np.save(f, test_data)

    # with open('extracted_data/test_labels.npy', 'wb') as f:
    #     np.save(f, test_labels)

if __name__ == '__main__':
    main()