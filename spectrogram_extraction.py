from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft # fourier transform
from scipy import signal

import librosa.display
import pylab
import librosa    
import glob


# global variables
train_dir = os.getcwd() + "/train_data/"
val_dir = os.getcwd() + "/val_data/"
test_dir = os.getcwd() + "/test_data/"

train_audio_dir = os.getcwd() + "/extracted_train_audio/"
val_audio_dir = os.getcwd() + "/extracted_val_audio/"
test_audio_dir = os.getcwd() + "/extracted_test_audio/"

# #path to save extracted audio
# extracted_train_path = os.getcwd() + "/extracted_train_spectrogram/"
# extracted_val_path = os.getcwd() + "/extracted_val_spectrogram/"
# extracted_test_path = os.getcwd() + "/extracted_test_spectrogram/"
# os.mkdir(extracted_train_path)
# os.mkdir(extracted_val_path)
# os.mkdir(extracted_test_path)


#directory to save extracted data
extracted_data_dir = os.getcwd() + "/extracted_audio_data/"
os.mkdir(extracted_data_dir)

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the training set.
train_df = pd.read_csv('train-updated.csv', dtype={'combination': object}).iloc[:,1:]
train_df["path"] = train_dir + train_df["Video ID"]+ ".0" + train_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the validation set.
val_df = pd.read_csv('val.csv', dtype={'combination': object}).iloc[:,1:]
val_df["path"] = val_dir + val_df["Video ID"]+ ".0" + val_df["Scene_ID"].astype(str) + ".mp4"

# Create a dataframe which contains multiclass classification content annotations for each video scene used in the test set.
test_df = pd.read_csv('test-updated.csv', dtype={'combination': object}).iloc[:,1:]
test_df["path"] = test_dir + test_df["Video ID"]+ ".0" + test_df["Scene_ID"].astype(str) + ".mp4"


def extract_spectrogram(audio_name):
    # AudioName = "vignesh.wav" # Audio File
    # fs, Audiodata = wavfile.read(AudioName)

    AudioName = "vignesh.wav" # Audio File
    fs, Audiodata = wavfile.read(audio_name)

    # Plot the audio signal in time
    plt.plot(Audiodata)
    plt.title('Audio signal in time',size=16)

    # spectrum
    n = len(Audiodata) 
    AudioFreq = fft(Audiodata)
    AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
    MagFreq = np.abs(AudioFreq) # Magnitude
    MagFreq = MagFreq / float(n)
    # power spectrum
    MagFreq = MagFreq**2
    if n % 2 > 0: # ffte odd 
        MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
    else:# fft even
        MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 

    plt.figure()
    freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n);
    plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
    plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');


    #Spectrogram
    N = 512 #Number of point in the fft
    f, t, Sxx = signal.spectrogram(Audiodata, fs,window = signal.blackman(N),nfft=N)
    plt.figure()
    plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
    #plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seg]')
    plt.title('Spectrogram with scipy.signal',size=16);

    plt.show()

def extract_mel_spectrograms(audio_dir):
    # Load the audio as a waveform 'data'
    # Store the sampling rate
    audio_data, samplerate = librosa.load(audio_dir)

    #Compute a mel-scaled spectrogram
    mel_feat = librosa.feature.melspectrogram(y=audio_data)
    power = librosa.power_to_db(mel_feat,ref=np.max)
    return power


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

        audio_dir = audio_root_dir + str(video_id) + '.' + str(scene_id) + '.wav'
        print(audio_dir)

        mel = extract_mel_spectrograms(audio_dir)

        audio_features.append(mel)
    
    return audio_features


def main():
    train_audio_data = prepare_all_spectrograms(train_df, train_audio_dir)
    val_audio_data = prepare_all_spectrograms(val_df, val_audio_dir)
    test_audio_data = prepare_all_spectrograms(test_df, test_audio_dir)

     # Save extracted data to file
    with open(extracted_data_dir + 'train_data.npy', 'wb') as f:
        np.save(f, train_audio_data)

    with open(extracted_data_dir + 'val_data.npy', 'wb') as f:
        np.save(f, val_audio_data)

    with open(extracted_data_dir + 'test_data.npy', 'wb') as f:
        np.save(f, test_audio_data)

if __name__ == '__main__':
    main()