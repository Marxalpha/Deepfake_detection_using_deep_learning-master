# DeepFake Detection
##  Follow the instructions below to run the code.
## Folder structure
```
root/
│── Dataset/
│   ├── FF++/
│   │   ├── Real/      # Contains 404 real videos
│   │   │   ├── real_001.mp4
│   │   │   ├── real_002.mp4
│   │   │   ├── ...
│   │   ├── Fake/      # Contains 404 fake videos
│   │   │   ├── fake_001.mp4
│   │   │   ├── fake_002.mp4
│   │   │   ├── ...
│── content/
│── Helpers/
│── Labels/ 
│── preprocessing.ipynb  
│── model_and_train.ipynb
│── metadata_generator.py
│── predict.ipynb  
│── README.md  # Project documentation

```

# Dataset link
find the dataset here (only a part of it is here) (full dataset is very large)
https://www.kaggle.com/datasets/hungle3401/faceforensics

# Requirements
install the requirements, only face_recognition package needs different instructions to be installed which is given below 
```
pip install torch torchvision numpy opencv-python matplotlib tqdm pandas scikit-learn seaborn mediapipe
```

# Running the preprocessing.ipynb


to install face-recognition, u have to execute the following commands in the powershell as admin
```bash
winget install -e --id Kitware.CMake
pip install cmake
pip install dlib
pip install face-recognition
```
follow the folder structure and keep all the files in place


# Running model_and_train.ipynb
install necessary packages 
```bash
pip install torch torchvision numpy opencv-python matplotlib
```
## Run metadata_generator.py
give correct path to dataset to gererate the global_metadata.csv and save it in labels folder
# Running Predict.ipynb
modify the path to the test video and run each cell

### Keep running all cells one after the other, thats it!


