# Chest X-Ray Classification Android App
This Android application classifies chest X-ray images into COVID-19, Pneumonia, or Normal categories using a TensorFlow Lite model.

## Model
The classification is done using a Convolutional Neural Network (CNN) model built with TensorFlow. This model was trained on a dataset of chest X-ray images, then converted to TensorFlow Lite to deploy in Android App.

## Dataset
The training dataset used to build the model can be found here: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

## Android Specifications
Gradle Version: 8.7

Minimum Android SDK: 27

Target Android SDK: 34

## Usage
**Image Upload:** Users can upload a chest X-ray image by either:
- Selecting an image from their device gallery, or
- Capturing a new image using the device camera (requires camera access permission).
**Prediction:** Once the image is uploaded, the app will process it and display the prediction result: COVID-19, Pneumonia, or Normal.
