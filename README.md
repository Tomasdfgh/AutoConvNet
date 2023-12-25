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

## Table of Content
- [Key Features](#Key-Features)
- [How to Use](#How-to-Use)
  -  [Training Live Update Feature Demo](#Training-Live-Update-Feature-Demo)
- [Target Audience](#Target-Audience)

## Key Features
- **Code-Free Operation:** Say goodbye to complex coding! AutoConvNet allows you to configure CNN architectures with a simple point-and-click interface, eliminating the need for manual coding during the model training process.
- **Real-Time Training Updates:** Witness the model's training process unfold with the software's live update feature. Track key metrics, assess convergence, and make informed decisions throughout the training journey.
- **Versatile Model Performance Metrics:** Evaluate your model with precision using a variety of performance metrics. Dive into accuracy, losses, errors, True and False positives and negatives, as well as Micro and Macro F1, Precision, and Recall.
- **Dynamic Feature Maps:** Explore the inner workings of your model with dynamic feature maps. Visualize extracted patterns and features from images, gaining insights into how your model interprets input data.
- **Download Model State and Trace:** Capture the state of your model by downloading its weights as a .pth file. Additionally, download the trace of th emodel as a .pt file. These files provide flexibility-continue training, put the model into production, or explore the inner workings of the trained model.

## How to Use
1. **Model Configuration (Step One):** Seamlessly input your data, supporting various formats such as .jpg files, .csv files, and binary files (.bin). Customize your model architecture with the option to use pre-trained models. Specify the width and height to resize the images, or choose from popular CNN architectures. If you don't have a dataset available, you can explore the provided datasets above.
![Step1_github](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/2ac8f39b-ed4a-46df-b600-105ecf4e7706)

2. **Fill in Hyperparameters and Data Split (Step Two):** Set batch sizes, learning rates, and other essential hyperparameters. Furthermore, determine the training, validation, and testing split to configure your dataset. Once this step is done, the result will appear in the architecture bar.
![image](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/0a7f8f00-e7ba-4405-a014-75cf9561db2e)

3. **Enter Convolutional Layers (Step Three):** After completing Step 2, proceed to Step 3 to tailor your convolutional layers. Here, you have the flexibility to define the number of input and output channels, set up padding, and incorporate a pooling layer if desired immediately following the convolutional layer. This is a repeatable step, allowing you to add multiple convolutional and pooling layers to your model.
![image](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/5db7c573-f5bc-4672-94be-f344c677ccb6)
In this example, I've configured two convolutional layers, and you can review their properties in the CNN Architecture tab. Currently, I'm focusing on the second convolutional layer, indicated by "2/2" in step 3, signifying the second layer out of two. The second convolutional layer is also highlighted. To make adjustments to the layers, you can navigate between them using the back and forth arrow buttons. To add a new layer, simply move to layer 3 out of 2, indicating the creation of a new, configurable layer.

4. **Enter Fully Connected Layers (Step Four):** Following the completion of Step 2, you can proceed to Step 4 to fine-tune your fully connected layers. In this step, specify the number of input and output neurons, define the dropout rate, and choose the activation function. It's worth noting that Step 4 can be completed independently of Step 3, resulting in a model without convolutional layers. In this scenario, the image will be flattened into a single vector array and processed by the fully connected layers.
![image](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/2b1fb819-ed17-4f19-87b7-03052e03ea3c)
The same method can be used in step 3 to change or add new Fully Connected Layers

   **Final Note on Step 3 and 4:** Every time a new convolutional or fully connected layer is added, AutoConvNet dynamically updates the model and performs a test by using a random image from the dataset for a forward pass. This process allows AutoConvNet to assess the viability of the model. Additionally, it captures the image input and output after each convolutional layer, pooling layer, and fully connected layer. If a new layer is added that doesn't align with the existing architecture, the forward pass will fail, and AutoConvNet will display [???] for every layer in the response.

6. **Training:**

   #### Commencing Training

   Once you have finished configuring your model, you may begin training by pressing the training button. The software will warn you if something is configured incorrectly. You have the option to train a classification or regression model. The requirements to train a regression model are as follows:
   - Your data needs to be labeled with integers
   - The criterion must be L1, Smooth L1, or MSE
   - The activation layer of the last fully connected layer must be None
   - The last output neuron must take a value of 1

   #### Inside Training Page: Main Features

   Once on the training page, observe your model undergoing live training with real-time updates on its performance and properties. In this page, AutoConvNet offers a variety of features for you to use. The main ones are will be listed here:
   - **Downloading the Training Code:** you can download the code employed for training the model. Running this code encompasses the entire process executed through the software, including steps such as loading and resizing your data, defining hyperparameters, configuring the model architecture, and initiating the training. If you wish to augment the training process with functionalities not available in the software, you have the freedom to do so by modifying this code file. The primary purpose of this feature is to provide flexibility. If there's anything the software doesn't cover or if you prefer a different approach, you can achieve it by obtaining the code responsible for constructing your model.
   - **Downloading the State of the Model:** You have the option to download the model's weights if you wish to resume the training process or adjust certain hyperparameters while continuing with the same model. AutoConvNet facilitates this by allowing you to download the model's state as a .pth file, which can then be uploaded in Step One to resume training. It's crucial to emphasize that for successful continuation of training with a pre-existing model, the architecture of the new model must precisely match. While hyperparameters can be modified, the structure, number, and properties of convolutional layers, pooling layers, and fully connected layers must remain unchanged. AutoConvNet will provide detailed feedback if any mismatches are detected.
   - **Downloading the Torch Trace of the model:** Upon completing the training process and achieving satisfactory results, you can download the trace of the model. This trace can be utilized for deploying the model in a software production environment or for testing purposes. Additionally, I've developed another software, DeepCapture Studio, which complements AutoConvNet. With DeepCapture Studio, you can upload the model's trace trained using AutoConvNet. The software captures images from your webcam and feeds them into the model, providing the output result. For comprehensive documentation on utilizing the model, refer to the link provided below.

      DeepCapture Studio GitHub Repository Link: [Link Here]

   #### Inside Training Page: Other Features

   Besides from the main features that AutoConvNet has in the training page, there are other smaller features that allow you understand your model better. Those features will be listed here:
      - **View Model Performance:** Monitor your model's performance in the **Model Performance** section, which consists of three pages, each displaying three graphs to illustrate your model's progress. Access any performance page during or after training. For graph customization or saving, click the "View Graph" button.
  
        
      - **Performance Report:** The **Performance Report** section mirrors the Model Performance in numerical values, showcasing the model's progress for each epoch. Upon completion, it displays testing accuracy. The training status appears at the bottom left, and once training concludes, it confirms the status. Download model stats from the top right.
      - **Feature Maps:** Explore feature maps, with six displayed at a time. For convolutional layers with more than six feature maps, use **Change Channels** to view others. To inspect feature maps of different convolutional layers, switch the convolutional layer via **Change Conv Layer**.
      - **Reconfigure Model and Retrain:** If you are unhappy with your current model, you have the option to return to the model configuration page by cliking on the **Train Another Model** Button. If you want to retrain the model you have right now, you can click on the **Retrain Model** Button.

![image](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/687b8f4c-ec9e-4edc-a67c-09e371962b5b)

  #### Training Live Update Feature Demo
  As a demonstration, check out the GIF below to see the live update feature being in-play when the model is still being trained. While being trained, you cannot download the trace or the status of the model.
  ![ezgif com-video-to-gif-converter](https://github.com/Tomasdfgh/AutoConvNet/assets/86145397/a1c804a3-e820-4af4-ad62-bbbbaf9ad13d)
   

## Target Audience

- **Students:** Perfect for those learning intricacies of CNN model training, offering a code-free entry into the world of deep learning.
- **Researchers:** Accelerate your research by training CNN models in minutes and gain insights into model behavior through dynamic feature maps.

AutoConvNet simplifies the complexities of CNN model training, putting the power of deep learning at your fingertips. Go nuts!

## Code Base Overview
In the upcoming sections, I'll present a comprehensive overview of the code constituting AutoConvNet. As highlighted in the list of libraries and frameworks used, the AutoConvNet UI is developed using TKinter, with a specific emphasis on the ttk library for enhanced GUI elements. The backend functionalities, responsible for tasks such as converting data into tensor arrays, configuring the model, and conducting training, heavily leverage torch and torchvision.

#### Dynamic Addition of Convolutional or Fully Connected Layers
The very first question I had when making AutoConvNet is how could I build it to provide users the ability to add on as many conv layers or fully connected layers as they wish. The method to approach that is using the nn.ModuleList(). Take a look below for the CNN class constructor:
```python
class ReplicaConvolutionalNetwork(nn.Module):

    def __init__(self):
        super(ReplicaConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()  # Store the fully connected layers
        self.pool_layers = []  # Store the pooling layers
        self.padding_list = []
        self.acts = []
        self.acts2 = []
        self.feature_maps = {}
        self.fc_dropout = nn.ModuleList()
        self.conv_dropout = nn.ModuleList()
```
In the Constructor, I have defined the conv_layers and fc_layers as nn.ModuleList() in order to add on any convolutional or dense layers.
```python
    def add_conv_layer(self, in_channels, out_channels, kernel_size_L, kernel_size_W, conv_stride_L, conv_stride_W, pooling_size_L, pooling_size_W, padding_size_L, padding_size_W, pool_stride_L, pool_stride_W, dropoutrate, acti_func, padding_left, padding_top, add_pooling=True, add_padding=True):
        new_conv_layer = nn.Conv2d(in_channels, out_channels, (kernel_size_L, kernel_size_W), stride = (conv_stride_L, conv_stride_W))
        dropoutLayer = nn.Dropout(dropoutrate)
        self.conv_layers.append(new_conv_layer)
        self.conv_dropout.append(dropoutLayer)
        self.acts.append(acti_func)
        if add_pooling:
            pool_layer = nn.MaxPool2d(kernel_size = (pooling_size_L,pooling_size_W), stride = (pool_stride_L,pool_stride_W))
            self.pool_layers.append(pool_layer)
        else:
            self.pool_layers.append(None)

        if add_padding:
            padding_layer = nn.ZeroPad2d((padding_left, padding_size_L, padding_top, padding_size_W)) # (left, right, top, bottom)
            self.padding_list.append(padding_layer)
        else:
            self.padding_list.append(None)
```
In the add_conv_layer method, the convolutional layer is appended onto the conv_layers attribute. in the UI, everytime you submit a new conv layer in step 3, the add_conv_layer method is called in the backEnd in order to add that layer to the Model. Within the same method, you can also choose to add any padding or pooling layers right after the convolutional layer as well.
```python
    def add_fc_layer(self, in_features, out_features,dropoutrate, act_func=1):
        self.acts2.append(act_func)
        new_fc_layer = nn.Linear(in_features, out_features)
        dropoutLayer = nn.Dropout(dropoutrate)
        self.fc_layers.append(new_fc_layer)
        self.fc_dropout.append(dropoutLayer)
```
The same concept applies to add_fc_layer method. The fully connected layer is appened onto the fc_layers attribute. In the UI, everytime you submit a new Dense layer in step 4, the add_fc_layer is called in the backEnd in order to add that layer to the Model.
```python
    def forward(self, x):
        try:
            #Layers key mapping: 0: Before all, 1: after activation, 2: after padding, 3: after pooling
            convShapes = {}
            count = 0
            for conv_layer, pool_layer, padding, act, dropout in zip(self.conv_layers, self.pool_layers, self.padding_list, self.acts, self.conv_dropout):
                convlayers = {}
                convlayers[0] = [x[0].shape[0],x[0].shape[1],x[0].shape[2]]
                x = conv_layer(x)
                # 0: None, 1: relu, 2: sigmoids, 3: tanh, 4: lrelu, 5: prelu, 6: elu
                if act == 0:
                    x = dropout(x)
                elif act == 1:
                    x = dropout(F.relu(x))
                elif act == 2:
                    x = dropout(torch.sigmoid(x))
                elif act == 3:
                    x = dropout(torch.tanh(x))
                elif act == 4:
                    x = dropout(F.leaky_relu(x, negative_slope=0.2))
                elif act == 5:
                    x = dropout(F.prelu(x))
                elif act == 6:
                    x = dropout(F.elu(x))
                elif act == 7:
                    x = dropout(F.softmax(x, dim = 1))
                convlayers[1] = [x[0].shape[0],x[0].shape[1],x[0].shape[2]]
                if padding:
                    x = padding(x)
                convlayers[2] = [x[0].shape[0],x[0].shape[1],x[0].shape[2]]
                if pool_layer:
                    x = pool_layer(x)
                convlayers[3] = [x[0].shape[0],x[0].shape[1],x[0].shape[2]]
                count += 1
                convShapes[count] = convlayers
        except:
            return x, None, None

        try:
            #Layers key mapping: 0: before all, 1: after all
            fullShapes = {}
            count2 = 0
            x = x.view(x.size(0), -1)
            for layer, act, dropout in zip(self.fc_layers, self.acts2, self.fc_dropout):
                fullLayer = {}
                count2 += 1
                fullLayer[0] = x[0].shape[0]
                if act == 0:
                    x = dropout(layer(x))
                elif act == 1:
                    x = dropout(F.relu(layer(x)))
                elif act == 2:
                    x = dropout(torch.sigmoid(layer(x)))
                elif act == 3:
                    x = dropout(torch.tanh(layer(x)))
                elif act == 4:
                    x = dropout(F.leaky_relu(layer(x), negative_slope=0.2))
                elif act == 5:
                    x = dropout(F.prelu(layer(x))) 
                elif act == 6:
                    x = dropout(F.elu(layer(x)))
                elif act == 7:
                    x = dropout(F.softmax(layer(x), dim = 1))
                fullLayer[1] = x[0].shape[0]
                fullShapes[count2] = fullLayer
        except:
            return x, None, None
        return x, convShapes, fullShapes
```
In essence, the dynamic addition of layers and the informative forward pass contribute to AutoConvNet's adaptability and robustness in handling various model configurations. If an incorrect architecture is detected, AutoConvNet promptly notifies the user.

Note: The ReplicaConvolutionalNetwork class is specifically designed for testing the model's viability and image dimensions after each layer. During training, a separate class is utilized, ensuring compatibility with TorchScript Trace.

#### Threading Implementation for Parallel Execution
In AutoConvNet, threading is a pivotal component that guarantees seamless and non-blocking execution, particularly when tackling tasks that may potentially disrupt the main GUI thread. The implementation leverages Python's threading module to introduce parallelism, significantly enhancing the responsiveness and overall performance of the application.

Threading is systematically integrated into nearly every button within the program, ensuring a responsive user interface by preventing lag during the processing of button functionalities. To facilitate communication with the main GUI thread, a queue is employed to seamlessly pass information back and forth. Upon completing its functionalities, the thread signals the main thread by placing a message in the queue, indicating the completion of its tasks. Examples of threaded operations triggered by user actions, such as the submission of step 1, can be observed below.
```python
        if self.dataType == ".bin":
            self.loading_thread = threading.Thread(target = lambda: self.backEnd.load_data_binary(self.loading_queue, datalink, height, width))
            self.loading_thread.daemon = True
            self.loading_thread.start()
            self.notification_textOne.set("Loading Binary Data")
            self.notificationOne.place(x = 30, y = 367)
            self.get_training_queue()
```
The following code snippet orchestrates the submission of .bin data type in Step 1 of AutoConvNet. A dedicated loading thread is instantiated, equipped with a queue parameter that facilitates communication with the main thread, conveying progress updates.
```python
        if self.dataType == ".bin":
            if data == "Step1":
                self.notification_textOne.set("Binary file matches compatibility")
                self.notificationOne.place(x = 30, y = 367)
            
            if data == "Fail1":
                self.notification_textOne.set("Failed to load File: Incorrect File Type")
                self.notificationOne.place(x = 30, y = 367)
                self.step1 = False
                return None
            
            if data == "Fail2":
                self.notification_textOne.set("Failed to load File: Inconsistence data dimensions")
                self.notificationOne.place(x = 30, y = 367)
                self.step1 = False
                return None
            
            if data == "ImNotSquare":
                self.notification_textOne.set("Failed to load File: Images not square")
                self.notificationOne.place(x = 30, y = 367)
                self.step1 = False
                return None
            
            if type(data) == float:
                self.notification_textOne.set("Loaded " + str(round(data * 100,3)) + "% of data")
                self.notificationOne.place(x = 30, y = 367)
```
This threading approach is not exclusive to data loading; it is also implemented during the training process. 
```python
        self.training_thread = threading.Thread(target=lambda: self.backEnd.trainingLoop(self.queue, self.stop_training_event))
        self.training_thread.daemon = True
        self.training_thread.start()
```
The training thread communicates vital information, such as performance metrics and feature maps, back to the main GUI thread, ensuring a synchronized and informative training experience for the user.
```python
    def get_queue(self):
        if self.queue.qsize() >= 1:
            data_get = self.queue.get_nowait()
            if data_get == True:
                self.doneTraining = True
                self.notification_textinTrain.set("Training is complete!")
                if not self.backEnd.doingRegression:
                    self.notification_inTrain.place(x = 965, y = 420)
                else:
                    self.notification_inTrain.place(x = 965, y = 630)
                self.infrastructureVisualization(Train = True, report = self.dataTrain[11], theme_ = self.themeBool)
                return None
            if data_get[20][0]:
                self.doneTraining = True
                self.ES = True
                self.notification_textinTrain.set("Training is complete!")
                if not self.backEnd.doingRegression:
                    self.notification_inTrain.place(x = 965, y = 420)
                else:
                    self.notification_inTrain.place(x = 965, y = 630)
                self.infrastructureVisualization(Train = True, report = self.dataTrain[11], theme_ = self.themeBool)
            if len(data_get[21]) == 1:
                self.dataTrain = data_get
                self.update_gui(self.dataTrain)
            if len(data_get[21]) == 3:
                if data_get[21][2]:
                    self.notification_textinTrain.set("Training Epoch " + str(data_get[21][0]) + ": " + str(round(data_get[21][1])) + "% Done")
                    if not self.backEnd.doingRegression:
                        self.notification_inTrain.place(x = 965, y = 420)
                    else:
                        self.notification_inTrain.place(x = 965, y = 630)
                if not data_get[21][2]:
                    self.notification_textinTrain.set("Validating Epoch " + str(data_get[21][0]) + ": " + str(round(data_get[21][1])) + "% Done")
                    if not self.backEnd.doingRegression:
                        self.notification_inTrain.place(x = 965, y = 420)
                    else:
                        self.notification_inTrain.place(x = 965, y = 630)            
        self.master.after(10, self.get_queue)
```

#### Code Base Overview Conclusion

In this detailed overview, we've delved into the core components of AutoConvNet, shedding light on its architecture, threading implementation, and core functionalities. The use of threading is a crucial aspect of the program, ensuring smooth and non-blocking execution during various tasks. Each step of the training process and data loading is orchestrated through dedicated threads, enhancing responsiveness.

The code snippets provided offer a glimpse into the inner workings of the application, showcasing the utilization of Python's threading module, Tkinter for the GUI, and PyTorch for the backend neural network operations. For a deeper understanding and exploration, feel free to examine the full codebase, where comprehensive comments and documentation are available.

For more information and insights, dive into the code repository and explore the intricacies of AutoConvNet firsthand. The repository serves as a valuable resource for developers, students, and researchers keen on understanding the implementation details and potentially contributing to the project's evolution.
