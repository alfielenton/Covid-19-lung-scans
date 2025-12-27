# CNN for Classification and Visualisation of Potential COVID CT-Scans

This repository builds and trains a Convolutional Neural Network in order to accurately classify CT scans of lungs that have or do not have COVID. It also contains a module that allows GRAD-CAM heatmaps to be imposed onto predicted images which shows what the model focuses on in the images in order to make its predictions.

The modules require the following installed packages:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install json
pip install matplotlib
```

In terminal, you can run the training.py module. 