# Kaggle [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)


## Setup: 

pip install -m requirements.txt

## Results:

Losses: Focal + Dice + BCE

Sampling: Random
1. Resize to 256: "dice": 0.732, "precision": 0.824, "recall": 0.659

Sampling: Balanced
1. **Resize to 256: "dice": 0.784, "precision": 0.834, "recall": 0.740**
2. Random ship-centered 256 crop: "dice": 0.666, "precision": 0.616, "recall": 0.724
3. Random resize from 768 to 384, ship-centered crop: "dice": 0.704, "precision": 0.690, "recall": 0.718
4. Random resize from 768 to 256, ship-centered crop: "dice": 0.528, "precision": 0.399, "recall": 0.780 **TODO: Retrain and confirm**
5. Random resize from 768 to 384, random crop: "dice": 0.646, "precision": 0.578, "recall": 0.731

## To Train

1. Download training images and CSV annotations from [Kaggle](https://www.kaggle.com/c/airbus-ship-detection)
2. Prepare annotations for training (convert CSV annotations to JSON and perform split) using src/data/prepare_for_training.py
3. Train via src/train/train.py entry point.

## To Eval
Run src/inference/predict_test.py to inference predictions and calculate metrics

## Analyse training distribution via [Jupyter Notebook](https://github.com/heorhiiberezovskyi/AirbusShipDetection/blob/main/AirbusShipDetectionDataAnalysis.ipynb)

## Logs and checkpoints can be found on [Google Drive](https://drive.google.com/drive/folders/1JHFlDxpcyJFq9DXmUOe1uX-jaAc4uOmC?usp=sharing)
