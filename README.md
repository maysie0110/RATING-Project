# RATING-Project

The following dependencies need to be satisfied to run the scripts:
1. !apt update && apt install ffmpeg libsm6 libxext6  -y
2. !pip install opencv-python
3. !pip install imageio
4. !pip install -q git+https://github.com/tensorflow/docs
5. !pip install -U scikit-learn

## Feature Extraction
1. Extracted frames:
- All extracted frames used for training are stored in the "extracted_train_frames" directory
- All extracted frames used for validation are stored in the "extracted_val_frames" directory
- All extracted frames used for testing are stored in the "extracted_test_frames" directory

2. Extracted features data:
- All extracted features data are stored in the "extracted_data" directory. 
- The feature data and labels are generated using the extracted frames and are stored in ".npy" files
- If you want to extract the features again, run the script "feature_extraction.py". Note that the feature extraction is a long process and takes about 2 hours.
-- If the frames are already extracted from the video, run "python feature_extraction.py".
-- If the extracted frames are not present, we need to extract the frames first, then extract the features from each frame. Run "python feature_extraction.py --extract_frames"

## Training
- Run "train.py" using the extracted features data and label. 