# CNN for Classification and Visualisation of Potential COVID CT-Scans

This repository builds and trains a Convolutional Neural Network in order to accurately classify CT scans of lungs that have or do not have COVID. It also contains a module that allows GRAD-CAM heatmaps to be imposed onto predicted images which shows what the model focuses on in the images in order to make its predictions.

The modules require the following installed packages:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install matplotlib
```

The model is written in the 'network' module and is a standard convolutional neural network. It has three convolutional layers that feed into two fully connected layers, all with biases.

The model is trained on a dataset of CT scans obtained from Kaggle with an example below.

![Diagram](readme-diagrams/COVID-19_0015.png)