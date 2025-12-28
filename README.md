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

The dataset used has a high imbalance towards COVID positive samples, so in training a weighted sampler is used.
In both the visualising and training module there is a line that looks like `general_transforms = transforms_with_crop` or `general_transforms = transforms_without_crop`.

This line was later included in training because it was found that in early methods of training, the general transformations would only resize the images without cropping. It can be seen that without cropping, the image tends to focus on parts of the image around the lung. Cropping the image means that the model only focuses on the lung itself. You can test this by changing this line to each option of general transformation.

As the dataset is quite small, data augmentation is also used to increase the variety on which the model is trained, allowing it to generalise better.

After training happens, the module produces plots that show you the epoch loss and the training loss. It also goes on to testing where different metrics are calculated and printed.

Use this line to clone this repository:

```
git clone https://github.com/alfielenton/Covid-19-lung-scans.git
```