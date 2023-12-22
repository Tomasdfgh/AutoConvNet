import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import functools
from tkinter import filedialog
import math
from torchsummary import summary
import subprocess
from io import StringIO
import sys
import csv
import torch.nn.functional as F
import pickle
import csv

import numpy as np


class ConvertData(Dataset):
    def __init__(self,array,transform = None):
        self.array = array
        self.transform = transform

    def __getitem__(self,index):
        image, label = self.array[index]
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert RGBA to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.array)

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

    def add_fc_layer(self, in_features, out_features,dropoutrate, act_func=1):
        self.acts2.append(act_func)
        new_fc_layer = nn.Linear(in_features, out_features)
        dropoutLayer = nn.Dropout(dropoutrate)
        self.fc_layers.append(new_fc_layer)
        self.fc_dropout.append(dropoutLayer)

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

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers: List[nn.Module] = nn.ModuleList()
        self.fc_layers: List[nn.Module] = nn.ModuleList()
        self.pool_layers: List[nn.Module] = nn.ModuleList()
        self.padding_list: List[nn.Module] = nn.ModuleList()
        self.acts = []
        self.acts2 = []
        self.fc_dropout: List[nn.Module] = nn.ModuleList()
        self.conv_dropout: List[nn.Module] = nn.ModuleList()

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

    def add_fc_layer(self, in_features, out_features,dropoutrate, act_func=1):
        self.acts2.append(act_func)
        new_fc_layer = nn.Linear(in_features, out_features)
        dropoutLayer = nn.Dropout(dropoutrate)
        self.fc_layers.append(new_fc_layer)
        self.fc_dropout.append(dropoutLayer)

    def forward(self, x):
        for conv_layer, pool_layer, padding, act, dropout in zip(self.conv_layers, self.pool_layers, self.padding_list, self.acts, self.conv_dropout):
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
            if padding:
                x = padding(x)
            if pool_layer:
                x = pool_layer(x)

        #Layers key mapping: 0: before all, 1: after all
        x = x.view(x.size(0), -1)
        for layer, act, dropout in zip(self.fc_layers, self.acts2, self.fc_dropout):
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
        return x

class BackEnd:

    def __init__(self):
        self.height = 0
        self.width = 0
        self.datalink = ""
        self.folders = []
        self.dataset = []
        self.dataset_regression = []
        self.labelMap = {}
        self.dataset_nonIm = []
        self.dataPerLabel = {}
        self.mean = 0
        self.dataRatio = 0
        self.gini_ = 0
        self.train_loader = 0
        self.validation_loader = 0
        self.test_loader = 0
        self.datalist = []
        self.batchSize = 0
        self.learningRate = 0
        self.epochs = 0
        self.convLayers = {}
        self.convLayerSteps = 1
        self.fullLayers = {}
        self.fullLayerSteps = 1
        self.fullShapes = {}
        self.convShapes = {}
        self.model = None
        self.criterion = None
        self.feature_maps = {}
        self.feature_maps_keys = {}
        self.dropout = None
        self.weightdecay = 0
        self.addNormalize = None
        self.earlyStopping = 0
        self.optimizer = None
        self.criterion = None
        self.critIdentifier = None
        self.report = None
        self.preTrainedModel = None
        self.imDim = []
        self.device = None
        self.regression_possibility = True
        self.doingRegression = False
        self.criterion_dict = {
            "Cross Entropy": nn.CrossEntropyLoss(), "L1" : nn.L1Loss(), "MSE": nn.MSELoss(),
            "BCE": nn.BCELoss(), "BCE w/ Logits": nn.BCEWithLogitsLoss(), "Smooth L1": nn.SmoothL1Loss(),
            "KLDiv": nn.KLDivLoss(), "Poisson NLL": nn.PoissonNLLLoss(), "Triplet Margin":nn.TripletMarginLoss(), 
            "Multi Label Marg": nn.MultiLabelMarginLoss(),  "Hinge Embed": nn.HingeEmbeddingLoss(),
            "Multi Margin": nn.MultiMarginLoss()
        }

    def check_gpu(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_quality(self):
        if len(self.dataPerLabel) > 0:
            # Average Data per Label, Ratio of smallest label / largest label, gini
            self.mean = round(sum(self.dataPerLabel.values())/len(self.dataPerLabel),2)
            self.dataRatio = round(min(self.dataPerLabel.values())/max(self.dataPerLabel.values()),2)
            s = ""
            l = []
            for i in self.dataPerLabel:
                s += str(self.dataPerLabel[i]) + ","
                l.append(self.dataPerLabel[i])
            l = sorted(l)
            
            #To Calculate the Gini Score, Get the lorenz curve first.
            #Key for list in each label: income, %pop, %income, %cummalitve income
            lorenz = {0: [0,0,0,0]}
            for i in range(1,len(l) + 1):
                lorenz[i] = [l[i-1],i/len(l),l[i-1]/sum(l)]

            lorenz[1].append(lorenz[1][2])
            for i in range(2, len(l) + 1):
                lorenz[i].append(lorenz[i-1][3] + lorenz[i][2])

            #Different Method of Calculating Gini Score
            #x: lorenz[i][1]
            #y: lorenz[i][3]
            B = 0
            for i in range(1,len(lorenz)):
                base_length = lorenz[i][1] - lorenz[i-1][1]
                square = base_length * lorenz[i-1][3]
                triangle = (1/2) * base_length * (lorenz[i][3] - lorenz[i-1][3])
                B += square + triangle
            self.gini_ = round(1 - 2*B,4)
        else: 
            self.mean = 0
            self.dataRatio = 0
            self.gini_ = 0

    def download_CSV(self):
        #               1                   2                   5              6                    3                       4                  9                       10                      11                  12                       13              14                  18              15                 16          17         7
        csv_ = [["","Training Loss", "Training Accuracies","Training MSE", "Training MAE", "Validation Accuracies", "Validation Loss", "Micro True Positive", "Micro True Negative", "Micro False Postive", "Micro False Negative", "Training Time", "Micro Precision","Macro Precision", "Micro Recall", "Macro Recall", "Micro F1", "Macro F1"]]
        for i in range(1, len(self.report)):
            temp_ = ["Epoch " + str(i), self.report[i][1], self.report[i][2], self.report[i][5], self.report[i][6], self.report[i][3], self.report[i][4], self.report[i][9], self.report[i][10], self.report[i][11], self.report[i][12], self.report[i][13], self.report[i][14], self.report[i][18], self.report[i][15], self.report[i][16], self.report[i][17], self.report[i][7]]
            csv_.append(temp_)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for row in csv_:
                    csv_writer.writerow(row)

    def downloadModel(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch Model Files", "*.pt")], title="Save Model As")
        ex_input = torch.rand(1,self.imDim[0],self.height, self.width)
        trace_net = torch.jit.trace(self.model, ex_input)
        if file_path:
            trace_net.save(file_path)

    def downloadModelState(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model Files", "*.pth")], title="Save Model As")
        if file_path:
            torch.save(self.model.state_dict(), file_path)

    def checkPreTrained(self, link):
        try:
            file_extension = os.path.splitext(link)[1]
            if file_extension != '.pth':
                return False
            self.preTrainedModel = torch.load(link)
        except:
            self.preTrainedModel = None
            return False
        return True
        

    def get_folders(self, directory):
        self.folders = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                self.folders.append(item_path)
        return self.folders

    def get_data(self, link):
        self.datalink = link
        try:
            self.folders = self.get_folders(link)
        except:
            return None
        return self.folders

    def load_data(self, queue, height,width):
        self.dataset = []
        self.dataset_regression = []
        self.labelMap = {}
        self.height = height
        self.width = width
        self.dataPerLabel = {}
        label = 0

        #Check to see if dataset is compatible for regression training
        for folder in self.folders:
            folder_name = os.path.basename(folder)
            try:
                folder_name = int(folder_name)
                self.regression_possibility = True
            except:
                self.regression_possibility = False
                break

        #Adding the data into the dataset variable
        for folder in self.folders:
            self.labelMap[label] = folder
            self.dataPerLabel[label] = 0
            folder_name = os.path.basename(folder)
            if self.regression_possibility:
                folder_name = int(folder_name)
            for filename in os.listdir(folder):
                try:
                    filepath = os.path.join(folder, filename)
                    if filepath[-3:] == 'tif':
                        continue
                    image = Image.open(filepath)
                    image = image.resize((self.width, self.height))
                    if self.regression_possibility:
                        self.dataset_regression.append((image, folder_name))
                    self.dataset.append((image,label))
                    self.dataPerLabel[label] += 1
                except (IOError, SyntaxError) as e:
                    continue
            label += 1
            if queue is not None:
                queue.put(int(label))
        if queue is not None:
            queue.put("Finished")
        return None

    def load_data_binary(self, queue, file, height, width):
        self.datalink = file
        self.height = height
        self.width = width
        self.dataset = []
        self.dataPerLabel = {}
        self.dataset_regression = []
        data_temp_ = {}

        #Checking File's Existence
        for i, item in enumerate(os.listdir(file)):
            item_path = os.path.join(file, item)
            try:
                with open(item_path, 'rb') as fo:
                    data_temp_[i] = pickle.load(fo, encoding='bytes')
            except:
                if queue is not None:
                    queue.put("Fail1")
                return False

        if queue is not None:
            queue.put("Step1")

        
        #Getting Total Data Points and Checking if array dimension are viable
        data_len = []
        w_now = 0 #second
        h_now = 0 #first
        total_num = 0

        for i in data_temp_:
            try:
                if any(not isinstance(element, int) for element in data_temp_[i][b'labels']):
                    self.regression_possibility = False
                else:
                    self.regression_possibility = True
                total_num += len(data_temp_[i][b'labels'])
                data_len.append(data_temp_[i][b'data'].shape[1])
                if data_len[len(data_len) - 1] != data_len[len(data_len) - 2]:
                    if queue is not None:
                        queue.put("Fail2")
                    return False
            except:
                if queue is not None:
                    queue.put("Fail2")
                return False

        #Getting image dimensions
        h_now = (data_len[0] / 3) ** (1/2)
        w_now = h_now
        if h_now != round(h_now):
            if queue is not None:
                queue.put("ImNotSquare")
            return False
  
        #Converting array to Image
        current_ = 0
        current_5per = 0
        for i in data_temp_:
            images = data_temp_[i][b'data'].reshape(-1,3,int(h_now),int(w_now)).transpose(0,2,3,1)
            for z, im in enumerate(images):
                try:
                    image = Image.fromarray(im)
                    image = image.resize((self.width, self.height))
                    self.dataset.append((image, data_temp_[i][b'labels'][z]))
                    if self.regression_possibility:
                        self.dataset_regression.append((image, data_temp_[i][b'labels'][z]))
                    if data_temp_[i][b'labels'][z] not in self.dataPerLabel:
                        self.dataPerLabel[data_temp_[i][b'labels'][z]] = 0
                    self.dataPerLabel[data_temp_[i][b'labels'][z]] += 1
                    current_ += 1
                    current_5per += 1
                    if current_5per/total_num >= 0.05:
                        current_5per = 0
                        if queue is not None:
                           queue.put(round(current_/total_num,3))
                except:
                    continue
        if queue is not None:
            queue.put("Finished")

        return None
        
    def load_data_csv(self, queue, link, height, width):
        self.datalink = link
        self.height = height
        self.width = width
        self.dataset = []
        self.dataPerLabel = {}
        self.dataset_regression = []
        current_ = 0
        current_5per = 0

        try:
            row_count = 0
            temp_labels = []
            with open(link, newline = '') as csvfile:
                csv_reader = csv.reader(csvfile)
                for i in csv_reader:
                    row_count += 1
                    try:
                        temp_labels.append(int(i[0]))
                    except:
                        temp_labels.append(i[0])
            if any(not isinstance(element, int) for element in temp_labels):
                self.regression_possibility = False
            else:
                self.regression_possibility = True



            with open(link, newline = '') as csvfile:

                csv_reader = csv.reader(csvfile)

                for c,i in enumerate(csv_reader):
                    if c == 0:
                        continue
                    try:
                        im = [int(val) for val in i[1:]]
                        image = Image.new('L', (int(len(i[1:]) ** (1/2)), int(len(i[1:]) ** (1/2))), 'white')
                        image.putdata(im)
                        image = image.resize((self.width, self.height))
                        if self.regression_possibility:
                            self.dataset.append((image, int(i[0])))
                            self.dataset_regression.append((image, int(i[0])))
                        else:
                            self.dataset.append((image, i[0]))
                    except Exception as e:
                        pass

                    if i[0] not in self.dataPerLabel:
                        self.dataPerLabel[i[0]] = 0
                    self.dataPerLabel[i[0]] += 1


                    current_ += 1
                    current_5per += 1
                    if current_5per / row_count >= 0.05:
                        current_5per = 0
                        if queue is not None:
                            queue.put(current_/ row_count)
            
            if queue is not None:
                test = r"C:\Users\tomng\Desktop\test_0.PNG"
                print(self.dataset[0][0])
                self.dataset[0][0].show()
                self.dataset[0][0].save(test)
                queue.put("Finished")

        except Exception as e:
            print(f"Failed with error: {e}")
            if queue is not None:
                queue.put("Failed")
            return None




    def split_data(self, batch_size, Norm, trainingSplit, validationSplit, testingSplit, learningRate, epochs, weightdecay, earlyStopping, optim, criterion):
        if Norm == True and self.imDim[0] == 3:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        else:
            print("Not being normalized")
            transform = transforms.Compose([transforms.ToTensor()])
        ImageSet = ConvertData(self.dataset, transform = transform)
        train_set, validation_set, test_set = torch.utils.data.random_split(ImageSet, [int((trainingSplit/100) * len(self.dataset)),int((validationSplit/100) * len(self.dataset)),len(self.dataset) - int((trainingSplit/100) * len(self.dataset)) - int((validationSplit/100) * len(self.dataset))])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

        if self.regression_possibility:
            ImageSetRegression = ConvertData(self.dataset_regression, transform = transform)
            train_set_regression, validation_set_regression, test_set_regression = torch.utils.data.random_split(ImageSetRegression, [int((trainingSplit/100) * len(self.dataset_regression)),int((validationSplit/100) * len(self.dataset_regression)),len(self.dataset_regression) - int((trainingSplit/100) * len(self.dataset_regression)) - int((validationSplit/100) * len(self.dataset_regression))])
            train_loader_regression = torch.utils.data.DataLoader(train_set_regression, batch_size = batch_size, shuffle = True)
            validation_loader_regression = torch.utils.data.DataLoader(validation_set_regression, batch_size = batch_size, shuffle = True)
            test_loader_regression = torch.utils.data.DataLoader(test_set_regression, batch_size = batch_size, shuffle = True)

            self.train_loader_regression = train_loader_regression
            self.validation_loader_regression = validation_loader_regression
            self.test_loader_regression = test_loader_regression

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.trainingSplit = trainingSplit
        self.validationSplit = validationSplit
        self.testingSplit = testingSplit
        self.batchSize = batch_size
        self.learningRate = learningRate
        self.epochs = epochs
        self.weightdecay = weightdecay
        self.addNormalize = Norm
        self.datalist = [int((trainingSplit/100) * len(self.dataset)),int((validationSplit/100) * len(self.dataset)),len(self.dataset) - int((trainingSplit/100) * len(self.dataset)) - int((validationSplit/100) * len(self.dataset))]
        self.earlyStopping = earlyStopping
        self.optimizer = optim
        self.criterion = self.criterion_dict[criterion]
        self.critIdentifier  = criterion
        return None

    def singleForwardPass(self, queue = None, loadPreTrained = None):
        testModel = ReplicaConvolutionalNetwork()
        criterion = nn.CrossEntropyLoss()
        acti = {"None": 0, "ReLU": 1, "Sigmoid":2, "Tanh": 3, "lrelu" : 4, "PReLU": 5, "ELU" : 6, "Softmax": 7}
        for k,v in self.convLayers.items():
            testModel.add_conv_layer(in_channels = v[0],out_channels = v[1],kernel_size_L = v[2], kernel_size_W = v[3], conv_stride_L = v[4], conv_stride_W = v[5], add_pooling = bool(v[6]), add_padding = bool(v[7]), pooling_size_L = v[8], pooling_size_W = v[9], padding_size_L = v[10], padding_size_W = v[11], pool_stride_L = v[12], pool_stride_W = v[13], dropoutrate = v[14], acti_func = acti[v[15]], padding_left = v[-2], padding_top = v[-1])
        if self.fullLayers != {}: 
            for k,v in self.fullLayers.items():
                testModel.add_fc_layer(v[0], v[1], dropoutrate = v[2],act_func = acti[v[3]])
        optimizer_dict = {
        "SGD": torch.optim.SGD(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adam": torch.optim.Adam(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "RMSprop": torch.optim.RMSprop(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adagrad": torch.optim.Adagrad(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adadelta": torch.optim.Adadelta(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "AdamW": torch.optim.AdamW(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "SparseAdam": torch.optim.SparseAdam(testModel.parameters(), lr=self.learningRate),
        "ASGD": torch.optim.ASGD(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Rprop": torch.optim.Rprop(testModel.parameters(), lr=self.learningRate),
        "RAdam": torch.optim.RAdam(testModel.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        }
        optimizer = optimizer_dict[self.optimizer]
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        extracted = images[0,0,:,:]
        extracted = extracted.numpy()
        np.set_printoptions(threshold=np.inf)
        print(extracted)
        print(extracted.shape)
        output, self.convShapes, self.fullShapes = testModel(images)

        if loadPreTrained:
            if self.preTrainedModel is not None:
                try:
                    testModel.load_state_dict(self.preTrainedModel)
                    if queue is not None:
                        queue.put(True)
                    return None
                except:
                    if queue is not None:
                        queue.put(False)
                    return None
            else:
                if queue is not None:
                    queue.put(True)
                return None
        del testModel
        del criterion
        del optimizer
        if queue is not None:
            queue.put(True)
        return None

    def copyCNNArchitecture(self):
        try:
            testModel = ReplicaConvolutionalNetwork()
            criterion = self.criterion
            acti = {"None": 0, "ReLU": 1, "Sigmoid":2, "Tanh": 3, "lrelu" : 4, "PReLU": 5, "ELU" : 6, "Softmax": 7}
            for k,v in self.convLayers.items():
                testModel.add_conv_layer(in_channels = v[0],out_channels = v[1],kernel_size_L = v[2], kernel_size_W = v[3], conv_stride_L = v[4], conv_stride_W = v[5], add_pooling = bool(v[6]), add_padding = bool(v[7]), pooling_size_L = v[8], pooling_size_W = v[9], padding_size_L = v[10], padding_size_W = v[11], pool_stride_L = v[12], pool_stride_W = v[13], dropoutrate = v[14], acti_func = acti[v[15]], padding_left = v[-2], padding_top = v[-1])
            if self.fullLayers != {}: 
                for k,v in self.fullLayers.items():
                    testModel.add_fc_layer(v[0], v[1], dropoutrate = v[2],act_func = acti[v[3]])
            original_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            summary(testModel, input_size = (self.convLayers[1][0],self.height, self.width))
            print_output = buffer.getvalue()
            sys.stdout = original_stdout
            if sys.platform.startswith('win'):
                subprocess.run(['clip'], input=print_output, text=True, check=True)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['pbcopy'], input=print_output, text=True, check=True)
            del testModel
            del criterion
            return True
        except:
            return False

    def trainingLoop(self,queue = None, flag = None):
        #Setting Up Model
        self.model = ConvolutionalNetwork().to(self.device)
        acti = {"None": 0, "ReLU": 1, "Sigmoid":2, "Tanh": 3, "lrelu" : 4, "PReLU": 5, "ELU" : 6, "Softmax": 7}
        for k,v in self.convLayers.items():
            self.model.add_conv_layer(in_channels = v[0],out_channels = v[1],kernel_size_L = v[2], kernel_size_W = v[3], conv_stride_L = v[4], conv_stride_W = v[5], add_pooling = bool(v[6]), add_padding = bool(v[7]), pooling_size_L = v[8], pooling_size_W = v[9], padding_size_L = v[10], padding_size_W = v[11], pool_stride_L = v[12], pool_stride_W = v[13], dropoutrate = v[14], acti_func = acti[v[15]], padding_left = v[-2], padding_top = v[-1])
        if self.fullLayers is not None: 
            for k,v in self.fullLayers.items():
                self.model.add_fc_layer(v[0], v[1], dropoutrate = v[2], act_func = acti[v[3]])
        optimizer_dict = {
        "SGD": torch.optim.SGD(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adam": torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "RMSprop": torch.optim.RMSprop(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adagrad": torch.optim.Adagrad(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Adadelta": torch.optim.Adadelta(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "AdamW": torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "SparseAdam": torch.optim.SparseAdam(self.model.parameters(), lr=self.learningRate),
        "ASGD": torch.optim.ASGD(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        "Rprop": torch.optim.Rprop(self.model.parameters(), lr=self.learningRate),
        "RAdam": torch.optim.RAdam(self.model.parameters(), lr=self.learningRate, weight_decay = self.weightdecay),
        }
        optimizer = optimizer_dict[self.optimizer]

        try:
            self.model.load_state_dict(self.preTrainedModel)
        except:
            pass

        def hook_fn(count, module, input, output):
            self.feature_maps_keys[count] = module
            self.feature_maps[module] = output

        count = 1
        for layer in self.model.conv_layers:
            layer.register_forward_hook(functools.partial(hook_fn, count))
            count += 1

        if self.doingRegression:
            trainloader = self.train_loader_regression
        if not self.doingRegression:
            trainloader = self.train_loader

        total_steps = len(trainloader)
        increasing = 0

        #plotting
        training_accuracy = []
        total_training_data = []
        valid_accuracy = []
        total_valid_data = []
        training_loss = []
        valid_loss = []
        testing_acc = []
        epochs = []
        mse_plotting = []
        mae_plotting = []
        true_pos = []
        true_neg = []
        false_pos = []
        false_neg = []
        time_taken = []
        precision = []
        recall = []
        f1_score = []
        report = {}
        ma_precision = []
        ma_recall = []
        ma_f1 = []

        #Report Key: 1: Training Loss, 2: Training Accuracy, 3: Validation Accuracy, 4: Validation Loss
        for epoch in range(1,self.epochs+1):

            start_time = time.time()
            innerReport = {}

            ES = False
            if increasing >= self.earlyStopping:
                ES = True 
                break

            if flag is not None:
                if flag.is_set():
                    return None

            training_acc = 0
            num_samples = 0
            y_pred = []
            y_true = []
            i_5per = 0
            for i, (images, labels) in enumerate(trainloader):
                if flag is not None:
                    if flag.is_set():
                        return None
                images, labels = images.to(self.device), labels.to(self.device)
                i_5per += 1
                if i_5per/(math.floor(self.datalist[0]/self.batchSize)) >= 0.01:
                    i_5per = 0
                    try:
                        epoch_data = (
                            False, False, False, False, False, False,          
                            False, False, False, False, False, False, False, False,             
                            False, False, False,False,False,False,list([False]), list([epoch,round(100 * i/(math.floor(self.datalist[0]/self.batchSize)),2),True]) #True means Training, False means Validation                   
                        )
                        if queue is not None:
                            queue.put(epoch_data)
                    except:
                        pass
                outputs = self.model(images)
                predicted = torch.argmax(outputs, dim = 1)
                if not self.doingRegression:
                    if self.critIdentifier != "Multi Margin":
                        labels = F.one_hot(labels, len(self.dataPerLabel)).float()

                if self.critIdentifier == "Multi Label Marg":
                    labels = labels.long()
                if self.doingRegression and self.critIdentifier == "MSE":
                    labels = labels.float()
                if self.doingRegression:
                    outputs = outputs.squeeze(dim = 1)
                loss = self.criterion(outputs, labels)

                if not self.doingRegression:
                    if self.critIdentifier != "Multi Margin":
                        labels = torch.argmax(labels, dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_acc += (predicted == labels).sum().item()
                num_samples += labels.size(0)
                if (i + 1) % len(trainloader) == 0:
                    e_sq = 0
                    for z in range(len(predicted)):
                          e_sq += (predicted[z] - labels[z]) ** 2
                    mse = e_sq/len(predicted)
                    mse_plotting.append(mse.item())


                    e_abs = 0
                    for c in range(len(predicted)):
                        e_abs += abs(predicted[c] - labels[c])
                    mae = e_abs/len(predicted)
                    mae_plotting.append(mae.item())

                    training_accuracy.append(100 * (training_acc/num_samples))
                    training_loss.append(loss.item())
                    epochs.append(epoch)
                    innerReport[1] = loss.item()
                    innerReport[2] = 100 * (training_acc / num_samples)
                    innerReport[5] = mse.item()
                    innerReport[6] = mae.item()

            i_5per = 0
            valid_acc = 0
            num_samples = 0
            for i, (images,labels) in enumerate(self.validation_loader):
                if flag is not None:
                    if flag.is_set():
                        return None
                images, labels = images.to(self.device), labels.to(self.device)
                i_5per += 1
                if i_5per/math.floor(self.datalist[1]/self.batchSize) >= 0.01:
                    i_5per = 0
                    try:
                        epoch_data = (
                            False, False, False, False, False, False,          
                            False, False, False, False, False, False, False, False,             
                            False, False, False,False,False,False,list([False]), list([epoch,round(100 * i/math.floor(self.datalist[1]/self.batchSize),2),False]) #True means Training, False means Validation                   
                        )
                        if queue is not None:
                            queue.put(epoch_data)
                    except:
                        pass
                outputs = self.model(images)
                predicted = torch.argmax(outputs, dim = 1)

                if not self.doingRegression:
                    if self.critIdentifier != "Multi Margin":
                        labels = F.one_hot(labels, len(self.dataPerLabel)).float()

                if self.critIdentifier == "Multi Label Marg":
                    labels = labels.long()
                if self.doingRegression and self.critIdentifier == "MSE":
                    labels = labels.float()
                if self.doingRegression:
                    outputs = outputs.squeeze(dim = 1)
                loss = self.criterion(outputs, labels)

                if not self.doingRegression:
                    if self.critIdentifier != "Multi Margin":
                        labels = torch.argmax(labels, dim=1)

                loss.backward()
                valid_acc += (predicted == labels).sum().item()
                num_samples += labels.size(0)
                y_pred.extend(predicted.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

                if (i + 1) % len(self.validation_loader) == 0:
                    valid_accuracy.append(100 * (valid_acc/num_samples))
                    try:
                        if loss.item() < valid_loss[-1]:
                            increasing = 0
                        else:
                            increasing += 1
                    except:
                        pass
                    valid_loss.append(loss.item())
                    innerReport[3] = 100 * (valid_acc / num_samples)
                    innerReport[4] = loss.item()
            
            if flag is not None:
                if flag.is_set():
                    return None

            con_mat = torch.tensor(confusion_matrix(y_true, y_pred), dtype=torch.float32)
            #----Getting info from CM----#
            true_pos_ = {}
            true_neg_ = {}
            false_pos_ = {}
            false_neg_ = {}

            #tag
            for i in range(len(con_mat)):
                true_pos_[i] = con_mat[i][i]
                true_neg_[i] = torch.sum(con_mat).item() - torch.sum(con_mat[i, :]).item() - torch.sum(con_mat[:, i]).item() + con_mat[i, i]
                false_pos_[i] = torch.sum(con_mat[:, i]).item() - con_mat[i, i]
                false_neg_[i] = torch.sum(con_mat[i, :]).item() - true_pos_[i]

            #----Calculating Micro------#
            tp_micro = sum(true_pos_.values())
            tn_micro = sum(true_neg_.values())
            fp_micro = sum(false_pos_.values())
            fn_micro = sum(false_neg_.values())

            #Calculating Recall
            if tp_micro == 0 and fn_micro == 0:
                micro_recall = 0
            else:
                micro_recall = tp_micro / tp_micro + fn_micro

            #Calculating Precision
            if tp_micro == 0 and fp_micro == 0:
                micro_precision = 0
            else:
                micro_precision = tp_micro / tp_micro + fn_micro

            #Calculating F1
            if micro_precision == 0 and micro_recall == 0:
                micro_f1 = 0
            else:
                micro_f1 = 2 * (micro_precision  * micro_recall)/ (micro_precision + micro_recall)

            #----Calculating Macro-----#
            macro_recall = {}
            macro_precision = {}
            macro_f1 = {}
            for i in range(len(con_mat)):
                #Calculate Recall
                if true_pos_[i] == 0 and false_neg_[i] == 0:
                    macro_recall[i] = 0
                else:
                    macro_recall[i] = true_pos_[i] / (true_pos_[i] + false_neg_[i])

                #Calculate Precision
                if true_pos_[i] == 0 and false_pos_[i] == 0:
                    macro_precision[i] = 0
                else:
                    macro_precision[i] = true_pos_[i] / (true_pos_[i] + false_neg_[i])

                #Calculating F1
                if macro_precision[i] == 0 and macro_recall[i] == 0:
                    macro_f1[i] = 0
                else:
                    macro_f1[i] = 2 * (macro_precision[i] * macro_recall[i]) / (macro_precision[i] + macro_recall[i])
            m_recall = sum(macro_recall.values())/len(macro_recall)
            m_precision = sum(macro_precision.values())/len(macro_precision)
            m_f1 = sum(macro_f1.values())/len(macro_f1)

            #------------#
            true_pos.append(tp_micro)
            true_neg.append(tn_micro)
            false_pos.append(fp_micro)
            false_neg.append(fn_micro)

            ma_precision.append(m_precision)
            ma_recall.append(m_recall)
            ma_f1.append(m_f1)

            time_taken.append(time.time() - start_time)
            precision.append(micro_precision)
            recall.append(micro_recall)
            f1_score.append(micro_f1)


            #----Changing the structure of Conmat to show the values in each square as a percentage instead of a value------#
            for i in range(len(con_mat)):
                for z in range(len(con_mat[i])):
                    con_mat[i][z] = round((100 * con_mat[i][z]/torch.sum(con_mat)).item())


            innerReport[9] = tp_micro
            innerReport[10] = tn_micro
            innerReport[11] = fp_micro
            innerReport[12] = fn_micro
            innerReport[13] = time.time() - start_time
            innerReport[14] = micro_precision
            innerReport[15] = micro_recall
            innerReport[16] = m_recall
            innerReport[17] = micro_f1 
            innerReport[18] = m_precision
            innerReport[7] = m_f1
            report[epoch] = innerReport
            self.report = report
            epoch_data = (
                list(epochs),                  # 0
                list(training_accuracy),      # 1
                list(valid_accuracy),          # 2
                list(training_loss),          # 3
                list(valid_loss),              # 4
                list(mae_plotting),              # 5
                list(mse_plotting),              # 6
                list(true_pos),                  # 7
                list(true_neg),                  # 8
                list(false_pos),              # 9
                list(false_neg),              # 10
                dict(report),                  # 11
                list(time_taken),              # 12
                list(precision),              # 13
                list(recall),                  # 14
                list(ma_precision),              # 15
                list(f1_score),                  # 16
                list(ma_recall),              # 17
                dict(self.feature_maps),      # 18
                dict(self.feature_maps_keys), # 19
                list([bool(False)]),          # 20
                list([None]),                  # 21
                list(ma_f1),                  # 22
                list(con_mat)
            )
            if queue is not None:
                queue.put(epoch_data)
        with torch.no_grad():
            age_loss = 0
            correct = 0
            total_samples = 0
            for images,labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                predicted = torch.argmax(F.softmax(outputs, dim = 0), dim = 1)
                total_samples += labels.size(0)
                for i in range(len(predicted)):
                    label = labels[i]
                    pred = predicted[i]
                    if pred == label:
                        correct += 1
                    age_loss += abs(pred - label)
            acc = 100 * correct / total_samples
            testing_acc.append(acc)
            report[max(report) + 1] = acc
            epoch_data = (
                list(epochs),                  # 0
                list(training_accuracy),      # 1
                list(valid_accuracy),          # 2
                list(training_loss),          # 3
                list(valid_loss),              # 4
                list(mae_plotting),              # 5
                list(mse_plotting),              # 6
                list(true_pos),                  # 7
                list(true_neg),                  # 8
                list(false_pos),              # 9
                list(false_neg),              # 10
                dict(report),                  # 11
                list(time_taken),              # 12
                list(precision),              # 13
                list(recall),                  # 14
                list(ma_precision),              # 15
                list(f1_score),                  # 16
                list(ma_recall),              # 17
                dict(self.feature_maps),      # 18
                dict(self.feature_maps_keys), # 19
                list([bool(False)]),          # 20
                list([None]),                  # 21
                list(ma_f1),                  # 22
                list(con_mat)
            )
            if queue is not None:
                queue.put(epoch_data)
        if queue is not None:
            queue.put(True)    
        return epochs, training_accuracy, valid_accuracy, training_loss, valid_loss, mse_plotting, mae_plotting, testing_acc
