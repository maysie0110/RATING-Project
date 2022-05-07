# RATING-Project
conda environment tf.2.6

The following dependencies need to be satisfied to run the scripts:
1. !apt update && apt install ffmpeg libsm6 libxext6  -y
2. !pip install opencv-python
3. !pip install imageio
4. !pip install -q git+https://github.com/tensorflow/docs
5. !pip install -U scikit-learn

## Transformer-based Model for video classification using visual information
### Feature Extraction
1. Extracted frames:
- All extracted frames used for training are stored in the "seq_length_128/extracted_train_frames" directory
- All extracted frames used for validation are stored in the "seq_length_128/extracted_val_frames" directory
- All extracted frames used for testing are stored in the "seq_length_128/extracted_test_frames" directory

2. Extracted features data:
- All extracted features data are stored in the "seq_length_128/extracted_data" directory. 
- The feature data and labels are generated using the extracted frames and are stored in ".npy" files
- If you want to extract the features again, run the script "image_feature_extraction.py". Note that the feature extraction is a long process and takes about 2 hours.
    - If the frames are already extracted from the video, run "python image_feature_extraction.py".
    - If the extracted frames are not present, we need to extract the frames first, then extract the features from each frame. Run "python image_feature_extraction.py --extract_frames"

### Training
- Run "python transformer_train.py" using the extracted features data and label. 
- Weights for trained model is saved in "seq_length_128/video_chkpt_128/video_classifier"

## CNN model for video classification using audio information
### Audio Extraction
pip install ffmpeg moviepy
pip install librosa
1. Extract audio:
- Extract audio from video directly using ffmpeg command with the help of subprocess module
- Run 'python audio_extraction.py'
- All extracted audio files used for training are stored in the "audio_classification/extracted_train_audio" directory
- All extracted audio files used for validation are stored in the "audio_classification/extracted_val_audio" directory
- All extracted audio files used for testing are stored in the "audio_classification/extracted_test_audio" directory

2. Extract mel spectrogram:
- Compute mel spectrogram from audio files using librosa
- Run 'python spectrogram_extractrion.py'
- All extracted audio spectrograms used for training are stored in the "audio_classification/extracted_train_spectrogram" directory
- All extracted audio spectrograms used for validation are stored in the "audio_classification/extracted_val_spectrogram" directory
- All extracted audio spectrograms used for testing are stored in the "audio_classification/extracted_test_spectrogram" directory

3. Train CNN model for audio classification
- Run 'python audio_classification.py'
- Weights for the trained model is saved in "audio_classification/audio_chkpt_sgd/audio_classifier"

## Multimodal Fusion
1. Early Fusion
- To train, 'python early_fusion.py'
- Weights for the trained model is saved in ""

2. Late Fusion
- Run 'python late_fusion.py'

## Test
- Run 'python test.py'