# Facial Emotion Detection using Deep Learning

# Overview

This project implements a real-time facial emotion detection system using Deep Learning (DL) and OpenCV. The model detects human facial expressions from a webcam feed and classifies them into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

#Features
Real-time emotion detection using a webcam.
Uses Haar cascade classifier for face detection.
Deep learning model trained on facial emotion datasets.
Optimized preprocessing and image normalization for better accuracy.
Press 'q' to exit the detection window.

#File Structure 
📂 emotion-detector
│── archive (images dataset)
│   ├── train
│   ├── test
│── facialemotionmodel.json  # Model architecture
│── facialemotionmodel.weights.h5  # Model weights
│── emotiondetector.h5
│── emotiondetector.json
│── emotiondetector.weights.h5
│── realtimedetection.py  # Main script
│── train.ipynb  # Training notebook
│── requirements.txt
│── README.md  # Project documentation

# Model and Algorithm Details
The model is built using Keras and TensorFlow.
Uses CNN (Convolutional Neural Networks) to classify emotions.
Input images are converted to grayscale (48x48) before being fed into the model.
Trained on datasets like FER-2013.

# Example Output
When a face is detected, a bounding box is drawn around it, and the predicted emotion is displayed:
Known Issues & Improvements
Accuracy can be improved with a larger dataset.
Additional filters can be applied to improve face detection.
GPU acceleration can enhance performance.

#Contributing

Feel free to fork this repository and submit a pull request with any improvement
