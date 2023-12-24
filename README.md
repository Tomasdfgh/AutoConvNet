# AutoConvNet

## Introduction

Welcome to AutoConvNet, an innovative and user-friendly tool designed to streamline the Convolutional Neural Network (CNN) model training process. Whether you're a student delving into the worlds of CNNs or a researcher seeking a swift and code-free solution, this software empoweres you to effortlessly train, evaluate, and deploy image detection models in just minutes. To get started, access the executable file (.exe) through the shared link provided, or alternatively, download all the provided codes and run the `AutoConvNetUI.py` file.

#### Executable Link:
AutoConvNet executable file: [link here]

#### Dataset Links:
In order to access the different datasets available, click the links below:

Face Dataset (.jpg): [link here]

Dog Dataset (.jpg): [link here]

mnist (.csv): [link here]

cifar10 (.bin): [link here]

#### Framework and Libraries Used:

-   tkinter: GUI toolkit for the graphical interface.
-   matplotlib: Plotting library for visualizations.
-   torch and torchvision: PyTorch framework for deep learning and computer vision tasks.
-   PIL (Python Imaging Library) and ImageTk: Image processing libraries for handling images.
-   seaborn: Statistical data visualization library based on Matplotlib.
-   queue, threading, time: Standard Python libraries for threading and timing operations.
-   warnings: Python module for issuing warnings.
-   subprocess: Module to spawn new processes, useful for executing external commands.
-   sys: Module providing access to some variables used or maintained by the Python interpreter.
-   csv: Module for reading and writing CSV files.
-   functools: Module for higher-order functions and operations on callable objects.

## Key Features:
- **Code-Free Operation:** Say goodbye to complex coding! AutoConvNet allows you to configure CNN architectures with a simple point-and-click interface, eliminating the need for manual coding during the model training process.
- **Real-Time Training Updates:** Witness the model's training process unfold with the software's live update feature. Track key metrics, assess convergence, and make informed decisions throughout the training journey.
- **Versatile Model Performance Metrics:** Evaluate your model with precision using a variety of performance metrics. Dive into accuracy, losses, errors, True and False positives and negatives, as well as Micro and Macro F1, Precision, and Recall.
- **Dynamic Feature Maps:** Explore the inner workings of your model with dynamic feature maps. Visualize extracted patterns and features from images, gaining insights into how your model interprets input data.
- **Download Model State and Trace:** Capture the state of your model by downloading its weights as a .pth file. Additionally, download the trace of th emodel as a .pt file. These files provide flexibility-continue training, put the model into production, or explore the inner workings of the trained model.

## How to Use:
1. **Model Configuration (Step One):** Seamlessly input your data, supporting various formats such as .jpg files, .csv files, and binary files (.bin). Customize your model architecture with the option to use pre-trained models. Specify the width and height to resize the images, or choose from popular CNN architectures. If you don't have a dataset available, you can explore the provided datasets above.
2. **Fill in Hyperparameters and Data Split (Step Two):** Set batch sizes, learning rates, and other essential hyperparameters. Furthermore, determine the training, validation, and testing split to configure your dataset. Once this step is done, the result will appear in the architecture bar.
3. **Enter Convolutional Layers (Step Three):** After completing Step 2, proceed to Step 3 to tailor your convolutional layers. Here, you have the flexibility to define the number of input and output channels, set up padding, and incorporate a pooling layer if desired immediately following the convolutional layer. This is a repeatable step, allowing you to add multiple convolutional and pooling layers to your model.
4. **Enter Fully Connected Layers (Step Four):** Following the completion of Step 2, you can proceed to Step 4 to fine-tune your fully connected layers. In this step, specify the number of input and output neurons, define the dropout rate, and choose the activation function. It's worth noting that Step 4 can be completed independently of Step 3, resulting in a model without convolutional layers. In this scenario, the image will be flattened into a single vector array and processed by the fully connected layers.
5. **Training:**

   #### Commencing Training

   Once you have finished configuring your model, you may begin training by pressing the training button. The software will warn you if something is configured incorrectly. You have the option to train a classification or regression model. The requirements to train a regression model are as follows:
   - Your data needs to be labeled with integers.
   - The criterion must be L1, Smooth L1, or MSE.
   - The activation layer of the last fully connected layer must be None.

   #### Inside Training Page: Main Features

   Once on the training page, observe your model undergoing live training with real-time updates on its performance and properties. In this page, AutoConvNet offers a variety of features for you to use. The main ones are will be listed here:
   - **Downloading the Training Code:** you can download the code employed for training the model. Running this code encompasses the entire process executed through the software, including steps such as loading and resizing your data, defining hyperparameters, configuring the model architecture, and initiating the training. If you wish to augment the training process with functionalities not available in the software, you have the freedom to do so by modifying this code file. The primary purpose of this feature is to provide flexibility. If there's anything the software doesn't cover or if you prefer a different approach, you can achieve it by obtaining the code responsible for constructing your model.
   - **Downloading the State of the Model:** You have the option to download the model's weights if you wish to resume the training process or adjust certain hyperparameters while continuing with the same model. AutoConvNet facilitates this by allowing you to download the model's state as a .pth file, which can then be uploaded in Step One to resume training. It's crucial to emphasize that for successful continuation of training with a pre-existing model, the architecture of the new model must precisely match. While hyperparameters can be modified, the structure, number, and properties of convolutional layers, pooling layers, and fully connected layers must remain unchanged. AutoConvNet will provide detailed feedback if any mismatches are detected.
   - **Downloading the Torch Trace of the model:** Upon completing the training process and achieving satisfactory results, you can download the trace of the model. This trace can be utilized for deploying the model in a software production environment or for testing purposes. Additionally, I've developed another software, DeepCapture Studio, which complements AutoConvNet. With DeepCapture Studio, you can upload the model's trace trained using AutoConvNet. The software captures images from your webcam and feeds them into the model, providing the output result. For comprehensive documentation on utilizing the model, refer to the link provided below.

      DeepCapture Studio GitHub Repository Link: [Link Here]
   

## Target Audience:

- **Students:** Perfect for those learning intricacies of CNN model training, offering a code-free entry into the world of deep learning.
- **Researchers:** Accelerate your research by training CNN models in minutes and gain insights into model behavior through dynamic feature maps.

AutoConvNet simplifies the complexities of CNN model training, putting the power of deep learning at your fingertips. Go nuts!

