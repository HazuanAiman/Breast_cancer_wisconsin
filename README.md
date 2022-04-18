# Breast Cancer Prediction using Feedforward Neural Network (FNN)
## Problem Statement
The aim of this project is to make a model that can predict breast cancer based on the features and data available. The output will predict whether the tumour is malignant or benign. The dataset used can be obtained [here](https://github.com/HazuanAiman/Breast_cancer_wisconsin/blob/main/dataset/breast%20cancer%20wisconsin.csv).
<br>
<br>
Credit: [source](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Methodology
#### IDE and Library
This project is made using Spyder as the main IDE. The main libraries used in this project are Pandas, Tensorflow Keras and Scikit-learn.

#### Model Pipeline
In this project, a feedforward neural network is used because the output result is a binary classification problem. The activation function used is sigmoid. Figure below shows the structure of the model created.
<p align="center">
  <img src="https://github.com/HazuanAiman/Breast_cancer_wisconsin/blob/main/images/breast%20cancer%20model%20pipeline.PNG">
<p>

The model is trained with a batch size of 32 and 100 epochs. Early stopping is applied in the training and it triggers at epochs 39/100. This is to prevent the model from overfitting. The training accurary achieved is 99% and the validation accuracy is 96%. Figures below show the graph of the training process.
<p align="center">
  <img src="https://github.com/HazuanAiman/Breast_cancer_wisconsin/blob/main/images/breast%20cancer%20epoch%20acc.PNG">
<p>
<p align="center">
  <img src="https://github.com/HazuanAiman/Breast_cancer_wisconsin/blob/main/images/breast%20cancer%20epoch%20loss.PNG">
<p>
  
## Results
The model is trained using the train dataset and evaluated using the test dataset. The test result are as show below:
 <p align="center">
  <img src="https://github.com/HazuanAiman/Breast_cancer_wisconsin/blob/main/images/breast%20cancer%20result.PNG">
<p>
