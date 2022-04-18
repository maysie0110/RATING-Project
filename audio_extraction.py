# import moviepy.editor as mp
import os
import pandas as pd
import subprocess

# global variables

train_dir = os.getcwd() + "/train_data/"
val_dir = os.getcwd() + "/val_data/"
test_dir = os.getcwd() + "/test_data/"

#path to save extracted audio
extracted_train_path = os.getcwd() + "/extracted_train_audio/"
extracted_val_path = os.getcwd() + "/extracted_val_audio/"
extracted_test_path = os.getcwd() + "/extracted_test_audio/"
os.mkdir(extracted_train_path)
os.mkdir(extracted_val_path)
os.mkdir(extracted_test_path)

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the training set.
train_df = pd.read_csv('train-updated.csv', dtype={'combination': object}).iloc[:,1:]
train_df["path"] = train_dir + train_df["Video ID"]+ ".0" + train_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the validation set.
val_df = pd.read_csv('val.csv', dtype={'combination': object}).iloc[:,1:]
val_df["path"] = val_dir + val_df["Video ID"]+ ".0" + val_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the test set.
test_df = pd.read_csv('test-updated.csv', dtype={'combination': object}).iloc[:,1:]
test_df["path"] = test_dir + test_df["Video ID"]+ ".0" + test_df["Scene_ID"].astype(str) + ".mp4"

def prepare_all_audios(df, save_dir):
    video_paths = df["path"].values.tolist()
    video_ids = df["Video ID"].values.tolist()
    scene_ids = df["Scene_ID"].values.tolist()

    # For each video.
    for idx, path in enumerate(video_paths):
        print(path)
        video_id = video_ids[idx]
        scene_id = scene_ids[idx]

        # audio_dir = save_dir + str(video_id) + '.' + str(scene_id) + '/'
        audio_dir = save_dir + str(video_id) + '.0' + str(scene_id) + '.wav'
        print(audio_dir)

        # clip = mp.VideoFileClip(path)
        # clip.audio.write_audiofile(audio_dir)

        subprocess.call(["ffmpeg", "-y", "-i", path, audio_dir], 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

def main():
    # my_clip = mp.VideoFileClip("tt2872718.00.mp4")
    # print(my_clip.size)
    # my_clip.audio.write_audiofile(f"my_result.wav")

    # subprocess.call(["ffmpeg", "-y", "-i", "tt2872718.00.mp4", f"my_result.wav"], 
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.STDOUT)

    prepare_all_audios(train_df, extracted_train_path)
    prepare_all_audios(val_df, extracted_val_path)
    prepare_all_audios(test_df, extracted_test_path)

if __name__ == '__main__':
    main()
