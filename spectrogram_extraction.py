from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft # fourier transform
from scipy import signal

import librosa.display
import pylab
import librosa    
import glob
import os
import pandas as pd

# global variables
train_dir = os.getcwd() + "/train_data/"
val_dir = os.getcwd() + "/val_data/"
test_dir = os.getcwd() + "/test_data/"

train_audio_dir = os.getcwd() + "/extracted_train_audio/"
val_audio_dir = os.getcwd() + "/extracted_val_audio/"
test_audio_dir = os.getcwd() + "/extracted_test_audio/"

#directory to save extracted data
extracted_root_dir = os.getcwd() + "/audio_classification"
# os.mkdir(extracted_root_dir)

#path to save extracted audio spectrogram
extracted_train_path = extracted_root_dir + "/extracted_train_spectrogram/"
extracted_val_path = extracted_root_dir + "/extracted_val_spectrogram/"
extracted_test_path = extracted_root_dir + "/extracted_test_spectrogram/"
# os.mkdir(extracted_train_path)
# os.mkdir(extracted_val_path)
# os.mkdir(extracted_test_path)

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the training set.
train_df = pd.read_csv('train-updated.csv', dtype={'combination': object}).iloc[:,1:]
train_df["path"] = train_dir + train_df["Video ID"]+ ".0" + train_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the validation set.
val_df = pd.read_csv('val.csv', dtype={'combination': object}).iloc[:,1:]
val_df["path"] = val_dir + val_df["Video ID"]+ ".0" + val_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the test set.
test_df = pd.read_csv('test-updated.csv', dtype={'combination': object}).iloc[:,1:]
test_df["path"] = test_dir + test_df["Video ID"]+ ".0" + test_df["Scene_ID"].astype(str) + ".mp4"


def extract_mel_spectrograms(audio_dir):
    # Load the audio wav file into numpy array
    audio_data, samplerate = librosa.load(audio_dir)
    # print(audio_data.shape)
    # print(samplerate)

    #Compute a mel-scaled spectrogram
    mel_feat = librosa.feature.melspectrogram(y=audio_data, sr=samplerate)
    power = librosa.power_to_db(mel_feat,ref=np.max)
    power = power.reshape(-1,1)
    return power


def create_spectrogram(filename, audio_dir):
    plt.interactive(False)
    clip, sample_rate = librosa.load(audio_dir, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')


def prepare_all_spectrograms(df, audio_root_dir):
    video_paths = df["path"].values.tolist()
    video_ids = df["Video ID"].values.tolist()
    scene_ids = df["Scene_ID"].values.tolist()

    audio_features = []
    # For each video.
    for idx, path in enumerate(video_paths):
        print(path)
        video_id = video_ids[idx]
        scene_id = scene_ids[idx]

        audio_dir = audio_root_dir + str(video_id) + '.0' + str(scene_id) + '.wav'
        print(audio_dir)

        mel = extract_mel_spectrograms(audio_dir)
        audio_features.append(mel)

    audio_features = np.array(audio_features)
    print(audio_features.shape)
    return audio_features

def extract_all_spectrograms(df, audio_root_dir, save_dir):
    video_paths = df["path"].values.tolist()
    video_ids = df["Video ID"].values.tolist()
    scene_ids = df["Scene_ID"].values.tolist()

    # For each video.
    for idx, path in enumerate(video_paths):
        print(path)
        video_id = video_ids[idx]
        scene_id = scene_ids[idx]

        audio_dir = audio_root_dir + str(video_id) + '.0' + str(scene_id) + '.wav'
        print(audio_dir)

        save_filename = save_dir + str(video_id) + '.0' + str(scene_id) + '.jpg'
        print(save_filename)
        create_spectrogram(save_filename, audio_dir)
      

def main():
    extract_all_spectrograms(train_df,train_audio_dir,extracted_train_path)
    extract_all_spectrograms(val_df,val_audio_dir,extracted_val_path)
    extract_all_spectrograms(test_df,test_audio_dir,extracted_test_path)

if __name__ == '__main__':
    main()