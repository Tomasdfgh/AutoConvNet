import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import AutoConvNet
import imageStorage
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import queue
import threading
import time
import warnings
import torchvision.models as models
from tkinter import PhotoImage
from PIL import Image, ImageTk
import base64
from io import BytesIO
import seaborn as sns

class UI():
    
    def __init__(self, master):
        self.master = master
        self.imageStorage = imageStorage.Images()
        self.master.title("AutoConvNet")
        self.theme = {True: '#343541', False: "#F0F0F0"}        #Background color: True means Dark theme and False means grey
        self.theme_bold = {True: "white", False: "black"}       #Meant for bolded Words
        self.theme_notbold = {True: "#D8DEE9", False: "black"}  #Meant for non Bolded Words
        self.theme_fields = {True:'#444654', False: '#FFFFFF'}
        self.master.configure(bg = self.theme[True])
        self.icon = self.imageStorage.call(17, True)
        self.master.call('wm', 'iconphoto', self.master._w, self.icon)
        self.themeButton_tag = True
        self.deleteAllConv_tag = True
        self.deleteConv_tag = True
        self.deleteFull_tag = True
        self.instructionButton_tag = True
        self.deleteAllFull_tag = True
        self.copyButton_tag = True
        self.downloadState_tag = True
        self.downloadCode_tag = True
        self.downloadStats_tag = True
        self.downloadModel_tag = True
        self.popOne_tag = True
        self.popTwo_tag = True
        self.popThree_tag = True
        self.popFour_tag = True
        self.popCM_tag = True
        self.threadcheck_fit = None

                                                    #---------ML Details---------#
        self.backEnd = AutoConvNet.BackEnd()

        self.optimizers = ""
        self.addPoolingBool = True
        self.addPaddingBool = True
        self.addNormalize = False
        self.actiFuncConv = ""
        self.actiFuncFull = ""
        self.queue = queue.Queue()
        self.loading_queue = queue.Queue()
        self.stepTwo_queue = queue.Queue()
        self.deleteAllFull_queue = queue.Queue()
        self.deleteAllConv_queue = queue.Queue()
        self.deleteFull_queue = queue.Queue()
        self.deleteConv_queue = queue.Queue()
        self.check_Train_queue = queue.Queue()
        self.statsPage = 1
        self.dataTrain = {}
        self.featureMapConv = 1
        self.featureMapCurrent = 0
        self.page_figures = []
        self.doneTraining = False
        self.stop_training_event = threading.Event()
        self.threadTrainDifferent = None
        self.training_thread = None
        self.ES = False
        self.dataType = None
        self.preArch = "None"
        self.criterion = None
        self.inPageOne = True
        self.current_position = (0,1)
        self.themeBool = False
                                                    #---------Graphing---------#
        self.fig = plt.Figure(figsize = (3,2), dpi = 100)
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.fig.set_facecolor(self.theme[self.themeBool])
        self.chart = FigureCanvasTkAgg(self.fig, master=self.master)
        self.chart_widget = self.chart.get_tk_widget()

        self.fig2 = plt.Figure(figsize=(3, 2), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.tick_params(axis='both', which='major', labelsize=6)
        self.fig2.set_facecolor(self.theme[self.themeBool])
        self.chart2 = FigureCanvasTkAgg(self.fig2, master=self.master)
        self.chart2_widget = self.chart2.get_tk_widget()

        self.fig3 = plt.Figure(figsize=(3, 2), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.tick_params(axis='both', which='major', labelsize=6)
        self.fig3.set_facecolor(self.theme[self.themeBool])
        self.chart3 = FigureCanvasTkAgg(self.fig3, master=self.master)
        self.chart3_widget = self.chart3.get_tk_widget()

        self.fig4, self.ax4 = plt.subplots(1, 6, figsize=(9, 2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1], 'height_ratios': [1]})
        self.fig4.set_facecolor(self.theme[self.themeBool])
        self.chart4 = FigureCanvasTkAgg(self.fig4, master = self.master)
        self.chart4_widget = self.chart4.get_tk_widget()

        self.fig_heatmap = plt.Figure(figsize=(3, 2), dpi=100)
        self.ax_heatmap = self.fig_heatmap.add_subplot(111)
        self.ax_heatmap.tick_params(axis='both', which='major', labelsize=6)
        self.fig_heatmap.set_facecolor(self.theme[self.themeBool])
        self.chart_heatmap = FigureCanvasTkAgg(self.fig_heatmap, master=self.master)
        self.chart_heatmap_widget = self.chart_heatmap.get_tk_widget()

                                                    #---------Logo and Others---------#
        self.uiCanvas = tk.Canvas(self.master, width = 1400, height = 750, bg = self.theme[self.themeBool],borderwidth=0, highlightthickness=0, highlightbackground=self.theme[self.themeBool])
        self.uiBox = self.uiCanvas.create_rectangle(4,4,1396,746, fill = self.theme[self.themeBool], outline = "black", width = 3)
        self.uiCanvas.pack()
        self.logo = tk.Label(self.master, text = "AutoConvNet", font = ("Cascadia Mono SemiBold", 30, "bold"))
        self.logo.place(x = 20, y = 20)
        self.madeBy = tk.Label(self.master, text = "Thomas Nguyen, UofT EngSci MI 2T4")
        self.madeBy.place(x = 10, y = 720)
        self.themeLbl = tk.Label(self.master, text = "Theme:")
        self.themeLbl.place(x = 1220, y = 33)
                                #---------Training Page Utlities---------#    
        self.modelPerf = tk.Label(self.master, text = "Model Performance", font = ("Mulish", 12, "bold"))
        self.featureMaps = tk.Label(self.master, text = "Feature Maps", font = ("Mulish", 12, "bold"))
        self.statsPageVar = tk.StringVar(self.master)
        self.statsPageLabel = tk.Label(self.master, textvariable = self.statsPageVar)
        self.trainForwardButton = ttk.Button(self.master, text = "\u2192", command = self.statsNext, style = "CustomArrow.TButton", takefocus=0)
        self.trainBackwardButton = ttk.Button(self.master, text = "\u2190", command = self.statsBack, style = "CustomArrow.TButton", takefocus=0)
        self.statsReportVar = tk.StringVar(self.master)
        self.statsReport = tk.Label(self.master,  textvariable = self.statsReportVar)
        self.featureNumVar = tk.StringVar(self.master)
        self.featureNumVar.set("")
        self.featureNumReport = tk.Label(self.master, textvariable = self.featureNumVar)
        self.featureNumForwardButton = ttk.Button(self.master, text = "\u2192", command = self.featureNumNext, style = "CustomArrow.TButton", takefocus=0)
        self.featureNumBackwardButton = ttk.Button(self.master, text = "\u2190", command = self.featureNumBack, style = "CustomArrow.TButton", takefocus=0)
        self.featureConvLayerVar = tk.StringVar(self.master)
        self.featureConvLayerReport = tk.Label(self.master, textvariable = self.featureConvLayerVar)
        self.featureConvLayerForwardButton = ttk.Button(self.master, text = "\u2192", command = self.featureConvLayerNext, style = "CustomArrow.TButton", takefocus=0)
        self.featureConvLayerBackwardButton = ttk.Button(self.master, text = "\u2190", command = self.featureConvLayerBack, style = "CustomArrow.TButton", takefocus=0)
        self.featureMapStats = tk.StringVar(self.master)
        self.featureMapConvPage = tk.Label(self.master, textvariable = self.featureMapStats)
        self.featureNumChanVar = tk.StringVar(self.master)
        self.featureNumChannel = tk.Label(self.master, textvariable = self.featureNumChanVar)
        
        self.themeImage = self.imageStorage.call(9, self.themeBool)
        self.themetest_image = self.themeImage
        self.themeButton = tk.Label(self.master, image = self.themetest_image, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.themeButton.place(x = 1270, y = 30)
        self.themeButton.bind("<ButtonRelease-1>", self.switchTheme)
        self.themeButton.bind("<Enter>", self.themeEnter)
        self.themeButton.bind("<Leave>", self.themeLeave)

        self.retrainImage = self.imageStorage.call(5, self.themeBool)
        self.retrainImage_ = self.retrainImage
        self.retrain = tk.Label(self.master, image = self.retrainImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.retrain.bind("<ButtonRelease-1>", self.retrainfunc)
        self.retrain.bind("<Enter>", self.reTrainOn_enter)
        self.retrain.bind("<Leave>", self.reTrainOn_leave)

        self.trainDiffImage = self.imageStorage.call(3, self.themeBool)
        self.trainDiffImage_ = self.trainDiffImage
        self.traindifferent = tk.Label(self.master, image = self.trainDiffImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.traindifferent.bind("<ButtonRelease-1>", self.traindifferentfunc)
        self.traindifferent.bind("<Enter>", self.trainDiffOn_enter)
        self.traindifferent.bind("<Leave>", self.trainDiffOn_leave)

        self.downloadstatsLbl = tk.Label(self.master, text = "Download Statistics:", font = ("Mulish", 9))
        self.trainAnotherText = tk.Label(self.master, text = "Train Another Model: ", font = ("Mulish", 9))
        self.reTrainModelText = tk.Label(self.master, text  ="Retrain Model: ", font = ("Mulish", 9))
        self.downloadModelText = tk.Label(self.master, text = "Download Model: ", font = ("Mulish", 9))
        self.downloadCodeText = tk.Label(self.master, text = "Download Code: ", font = ("Mulish", 9))

        self.saveStateLbl = tk.Label(self.master, text = "Save Model's State:", font = ("Mulish", 9))

        self.criVar = tk.StringVar(self.master)
        self.criComboBox = ttk.Combobox(self.master, textvariable = self.criVar, style = "ComboboxStyle.TCombobox", state = "readonly",width = 16, values = ["Cross Entropy", "MSE", "L1", "BCE", "BCE w/ Logits", "Smooth L1", "KLDiv", "Poisson NLL", "Multi Label Marg", "Hinge Embed", "Multi Margin"])
        self.criComboBox.bind("<<ComboboxSelected>>", self.callCriterion)

        self.estyle = ttk.Style()

        self.downloadButton = self.imageStorage.call(7, self.themeBool)
        self.downloadButton_ = self.downloadButton
        self.downloadModel = tk.Label(self.master, image=self.downloadButton_, borderwidth=0, relief="flat", highlightthickness=0, state="normal")
        self.downloadModel.bind("<ButtonRelease-1>", self.downloadModelFunc)
        self.downloadModel.bind("<Enter>", self.downloadModelEnter)
        self.downloadModel.bind("<Leave>", self.downloadModelLeave)

        self.downloadCode = tk.Label(self.master, image = self.downloadButton_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.downloadCode.bind("<ButtonRelease-1>", self.downloadCodeFunc)
        self.downloadCode.bind("<Enter>", self.downloadCodeEnter)
        self.downloadCode.bind("<Leave>", self.downloadCodeLeave)

        self.downloadStatsButton = tk.Label(self.master, image = self.downloadButton_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.downloadStatsButton.bind("<ButtonRelease-1>", self.downloadStats)
        self.downloadStatsButton.bind("<Enter>", self.downloadStatsEnter)
        self.downloadStatsButton.bind("<Leave>", self.downloadStatsLeave)

        self.saveStateButton = tk.Label(self.master ,image = self.downloadButton_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.saveStateButton.bind("<ButtonRelease-1>",self.downloadState)
        self.saveStateButton.bind("<Enter>", self.downloadStateEnter)
        self.saveStateButton.bind("<Leave>", self.downloadStateLeave)

        self.popImage = self.imageStorage.call(1,self.themeBool)
        self.popImage_ = self.popImage
        self.popGraphOne = tk.Label(self.master, image = self.popImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.popGraphOne.bind("<ButtonRelease-1>",self.pop1)
        self.popGraphOne.bind("<Enter>", self.PopOneOn_enter)
        self.popGraphOne.bind("<Leave>", self.PopOneOn_leave)
        
        self.popGraphTwo = tk.Label(self.master, image = self.popImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.popGraphTwo.bind("<ButtonRelease-1>",self.pop2)
        self.popGraphTwo.bind("<Enter>", self.PopTwoOn_enter)
        self.popGraphTwo.bind("<Leave>", self.PopTwoOn_leave)

        self.popGraphThree = tk.Label(self.master, image = self.popImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.popGraphThree.bind("<ButtonRelease-1>", self.pop3)
        self.popGraphThree.bind("<Enter>", self.PopThreeOn_enter)
        self.popGraphThree.bind("<Leave>", self.PopThreeOn_leave)

        self.popGraphFour = tk.Label(self.master, image = self.popImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.popGraphFour.bind("<ButtonRelease-1>", self.pop4)
        self.popGraphFour.bind("<Enter>", self.PopFourOn_enter)
        self.popGraphFour.bind("<Leave>", self.PopFourOn_leave)

        self.popCM = tk.Label(self.master, image = self.popImage, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.popCM.bind("<ButtonRelease-1>", self.popCMFunc)
        self.popCM.bind("<Enter>", self.PopCM_enter)
        self.popCM.bind("<Leave>", self.PopCM_leave)

        self.popGraph1 = tk.Label(self.master, text = 'View Graph: ')
        self.popGraph2 = tk.Label(self.master, text = 'View Graph: ')
        self.popGraph3 = tk.Label(self.master, text = 'View Graph: ')
        self.popGraph4 = tk.Label(self.master, text = 'View Feature Maps: ')
        self.popConMat = tk.Label(self.master, text = 'View Matrix: ')


                                #---------Page One Utilities---------#
        #Step One Utilities
        self.estyle.element_create("plain.field", "from", "clam")
        self.estyle.layout("EntryStyle.TEntry",[('Entry.plain.field', {'children': [('Entry.background', {'children': [('Entry.padding', {'children': [('Entry.textarea', {'sticky': 'nswe'})],'sticky': 'nswe'})], 'sticky': 'nswe'})],'border':'2', 'sticky': 'nswe'})])
        self.dataLoadCanvas = tk.Canvas(self.master, width = 403, height = 253, bg = self.theme[self.themeBool], borderwidth=0, relief="flat", highlightthickness=0,)
        self.dataLoadBox = self.dataLoadCanvas.create_rectangle(3,3,400,250,outline = "black", fill = self.theme[self.themeBool], width = 3)
        self.stepOneBanner = tk.Label(self.master, text = "Step One: Setup Data", font = ("Mulish", 12, "bold"))
        self.dataPathWay = tk.Label(self.master, text = "Data's Location:", font = ("Mulish", 11))
        self.imageSize = tk.Label(self.master, text = "Resize Image:", font = ("Mulish", 11, "bold"))
        self.imageWidth = tk.Label(self.master, text = "Width: ", font = ("Mulish", 11))
        self.imageHeight = tk.Label(self.master, text = "Height: ", font = ("Mulish", 11))
        self.dataTypeLabel = tk.Label(self.master, text = "Data Type:", font = ("Mulish", 11))
        self.configureData = tk.Label(self.master, text = "Configure Data and Load Pre-trained Model", font = ("Mulish", 11, "bold"))
        self.estyle.layout("ComboboxStyle.TCombobox", [('Combobox.plain.field', {'children': [('Combobox.background', {'children': [('Combobox.padding', {'children': [('Combobox.textarea', {'sticky': 'nswe'})],'sticky': 'nswe'})],'sticky': 'nswe'})],'border': '2','sticky': 'nswe'})])
        
        self.datavar = tk.StringVar(self.master)
        self.dataCB = ttk.Combobox(self.master, textvariable=self.datavar, values = [".jpg", ".csv", ".bin"], state = "readonly", style = "ComboboxStyle.TCombobox", width = 8)
        self.dataCB.bind("<<ComboboxSelected>>", self.callDataType)

        self.archvar = tk.StringVar(self.master)
        self.archCB = ttk.Combobox(self.master, textvariable = self.archvar, values = ["None","LeNet","AlexNet", "VGGNet"], state = "readonly", style = "ComboboxStyle.TCombobox", width = 12)
        self.archCB.bind("<<ComboboxSelected>>", self.callPreArch)

        self.dataPathSearch_Bar = ttk.Entry(self.master, width=30, style="EntryStyle.TEntry")
        self.heightSizeSearch_Bar = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.widthSizeSearch_Bar = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.loadPTBoolButton = ttk.Entry(self.master, width = 30, style="EntryStyle.TEntry")
        
        self.chooseModelLabel = tk.Label(self.master, text = "Choose a Pre-Existing Model (Optional):", font = ("Mulish", 11, "bold"))
        self.archType = tk.Label(self.master, text = "Architecture:", font = ("Mulish", 11))
        self.preHeight = {"AlexNet": 227 , "LeNet": 32, "VGGNet": 224}
        self.step1ResubmitWarning = False
        self.loadPreTrained = tk.Label(self.master, text = 'Pre-Trained Model: ', font = ("Mulish", 11))

        self.estyle.theme_use("clam")
        self.estyle.configure("Custom.TButton", font = ("Mulish", 11, "bold"), width = 7, height = 2)
        self.stepOneSubmitButton = ttk.Button(self.master, text = "Submit", command = self.stepOneSubmit, style = "Custom.TButton", takefocus=0)
        self.loading_thread = None
        self.threadReTrain = None

        #Step Two Utilities
        self.dataSplitCanvas = tk.Canvas(self.master, width = 403, height = 253, bg = self.theme[self.themeBool], borderwidth=0, relief="flat", highlightthickness=0,)
        self.dataSplitBox = self.dataSplitCanvas.create_rectangle(3,3,400,250, outline = "black", fill = self.theme[self.themeBool], width = 3)
        self.datasplitLbl = tk.Label(self.master, text = "Data Split", font = ("Mulish", 11, "bold"))
        self.modelpropertieslbl = tk.Label(self.master, text = "Model's Properties", font = ("Mulish", 11, "bold"))
        self.stepTwoBanner = tk.Label(self.master, text = "Step Two: Data Split & Model Setup", font = ("Mulish", 12, "bold"))
        self.training_split = tk.Label(self.master, text = "Training Split(%)", font = ("Mulish", 11))
        self.testing_split = tk.Label(self.master, text = "Testing Split(%)", font = ("Mulish", 11))
        self.validation_split = tk.Label(self.master, text = "Validation Split(%)", font = ("Mulish", 11))
        self.training_Split_Search_Bar = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.validation_Split_Search_Bar = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.testing_Split_Search_Bar = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.batchSizeSB = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.lrSB = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.numEpochsSB = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")

        self.estyle.configure("Custom2.TButton", font=("Mulish", 7), width = 4, height = 1)
        self.estyle.configure("Custom3.TButton",font=("Mulish", 7), width = 5, height = 1)

        self.normalizeConst = False
        self.addNormalize = False
        self.checkBoxNorm = self.imageStorage.call(34, self.themeBool)
        self.checkBoxNorm_ = self.checkBoxNorm
        self.normCheckBox = tk.Label(self.master, image = self.checkBoxNorm_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.normCheckBox.bind("<ButtonRelease-1>", self.addNormalizeFunc)
        self.normCheckBox.bind("<Enter>", self.normEnter)
        self.normCheckBox.bind("<Leave>", self.normLeave)

        self.stepTwoSubmitButton = ttk.Button(self.master, text = "Submit", command = self.stepTwoSubmit, style = "Custom.TButton", takefocus=0)
        
        self.batchSizeLabel = tk.Label(self.master, text = "Batch Size", font = ("Mulish",11))
        self.learningRateLabel = tk.Label(self.master, text = "Learning Rate", font = ("Mulish",11))
        self.numEpochsLabel = tk.Label(self.master, text = "Number of Epochs", font = ("Mulish",11))
        self.optimizerLabel = tk.Label(self.master, text = "Optimizer: ", font = ("Mulish", 11))
        self.trainingOptimizationLbl = tk.Label(self.master, text = 'Training Optimization', font =  ("Mulish",11, "bold"))

        self.normalizeDataLbl = tk.Label(self.master, text = "Normalize:", font = ("Mulish", 11))
        self.weightDecay = tk.Label(self.master, text = "Weight Decay:", font = ("Mulish", 11))
        self.earlyStoppingLabel = tk.Label(self.master, text = "Early Stopping:", font = ("Mulish", 11))
        self.criterionLbl = tk.Label(self.master, text = 'Criterion:', font = ("Mulish",11))
        self.weightDecayEntry = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        self.verticalLineCanvas = tk.Canvas(self.master, width=5, height= 70, bg=self.theme[self.themeBool], borderwidth=0, relief="flat", highlightthickness=0)
        self.verticalLine = self.verticalLineCanvas.create_line(2, 2, 2, 70, fill="black", width=2)
        self.earlyStoppingEntry = ttk.Entry(self.master, width = 7, style="EntryStyle.TEntry")
        
        self.criVar = tk.StringVar(self.master)
        self.criComboBox = ttk.Combobox(self.master, textvariable = self.criVar, style = "ComboboxStyle.TCombobox", state = "readonly",width = 16, values = ["Cross Entropy", "MSE", "L1", "BCE", "BCE w/ Logits", "Smooth L1", "KLDiv", "Poisson NLL", "Multi Label Marg", "Hinge Embed", "Multi Margin"])
        self.criComboBox.bind("<<ComboboxSelected>>", self.callCriterion)

        self.optiVar = tk.StringVar(self.master)
        self.optiComboBox = ttk.Combobox(self.master, textvariable = self.optiVar, style = "ComboboxStyle.TCombobox", state = "readonly", width = 16, values = ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "AdamW", "SparseAdam", "ASGD", "Rprop", "RAdam"])
        self.optiComboBox.bind("<<ComboboxSelected>>", self.callOptimizers)
        self.threadTwo = None


        #Step Three Utilities
        self.convLayerCanvas = tk.Canvas(self.master, width = 403, height = 253, bg = self.theme[self.themeBool], borderwidth=0, relief="flat", highlightthickness=0)
        self.convLayerBox = self.convLayerCanvas.create_rectangle(3,3,400,250, outline = "black", fill = self.theme[self.themeBool], width = 3)
        self.stepThreeBanner = tk.Label(self.master, text = "Step Three: Enter Convolutional Layers", font = ("Mulish", 12, "bold"))
        self.numInChannels = tk.Label(self.master, text = "# In Channels", font = ("Mulish", 9))
        self.numOutChannels = tk.Label(self.master, text = "# Out Channels", font = ("Mulish", 9))
        self.kernelSize = tk.Label(self.master, text = "Kernel Size", font = ("Mulish", 9))
        self.strideLength = tk.Label(self.master, text = "Kernel Stride", font = ("Mulish", 9))
        self.in_Channel_Search_Bar = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.out_Channel_Search_Bar = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.paddingAndPoolingLayer = tk.Label(self.master, text = "Padding/Pooling Layers", font = ("Mulish", 11, "bold"))
        self.paddingLabel = tk.Label(self.master, text = "Add a Padding Layer:", font = ("Mulish", 11))
        self.poolingLabel = tk.Label(self.master, text = "Add a Pooling Layer:", font = ("Mulish", 11))
        self.sizeLabel = tk.Label(self.master, text = "Size", font = ("Mulish", 9))
        self.strideLengthLabel = tk.Label(self.master, text = "Stride", font = ("Mulish", 9))
        self.activationFunctions = tk.Label(self.master, text = "Activation Functions: ", font = ("Mulish", 11))

        self.actiVarFullConn = tk.StringVar(self.master)

        self.actiVar = tk.StringVar(self.master)
        self.combobox = ttk.Combobox(self.master, textvariable = self.actiVar, width = 10, style = "ComboboxStyle.TCombobox", state = "readonly", values = ["None", "ReLU", "PReLU", "ELU", "Tanh", "Sigmoid", "Softmax"])
        self.combobox.bind("<<ComboboxSelected>>", self.callActivationFunctions)

        self.curConvUp = tk.Label(self.master, text = "On Layer:")
        self.estyle.configure("CustomArrow.TButton", width = 2, height = 1)
        self.stepThreeSubmitButton = ttk.Button(self.master, text = "Submit", command = self.stepThreeSubmit, style = "Custom.TButton", takefocus=0)
        self.convLeftButton = ttk.Button(self.master, text = "\u2190", command = self.convBack, style = "CustomArrow.TButton", takefocus=0)
        self.convRightButton = ttk.Button(self.master, text = "\u2192", command = self.convForward, style = "CustomArrow.TButton", takefocus=0)

        self.trashImage = self.imageStorage.call(15, self.themeBool)
        self.trashImage_ = self.trashImage
        self.deleteConvLayer = tk.Label(self.master, image = self.trashImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.deleteConvLayer.bind("<ButtonRelease-1>", self.deleteConv)
        self.deleteConvLayer.bind("<Enter>", self.deleteConvEnter)
        self.deleteConvLayer.bind("<Leave>", self.deleteConvLeave)

        self.deleteAllConvLayer = tk.Label(self.master, image = self.trashImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.deleteAllConvLayer.bind("<ButtonRelease-1>", self.deleteAllConv)
        self.deleteAllConvLayer.bind("<Enter>", self.deleteAllConvEnter)
        self.deleteAllConvLayer.bind("<Leave>", self.deleteAllConvLeave)

        self.paddingCheckConst = False
        self.addPaddingBool = False
        self.checkBoxIm = self.imageStorage.call(34, self.themeBool)
        self.checkBoxIm_ = self.checkBoxIm
        self.paddingCheckBox = tk.Label(self.master, image = self.checkBoxIm_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.paddingCheckBox.bind("<ButtonRelease-1>", self.paddingCheckBoxFunc)
        self.paddingCheckBox.bind("<Enter>", self.paddingEnter)
        self.paddingCheckBox.bind("<Leave>", self.paddingLeave)

        self.poolingCheckConst = False
        self.addPoolingBool = False
        self.checkBoxIm2 = self.imageStorage.call(34, self.themeBool)
        self.checkBoxIm2_ = self.checkBoxIm2
        self.poolingCheckBox = tk.Label(self.master, image = self.checkBoxIm2_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.poolingCheckBox.bind("<ButtonRelease-1>", self.addPooling)
        self.poolingCheckBox.bind("<Enter>", self.poolingEnter)
        self.poolingCheckBox.bind("<Leave>", self.poolingLeave)

        self.curConvText = tk.StringVar(self.master)
        self.curConvText.set("Currently in Layer: " + str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
        self.curConvLabel = tk.Label(self.master, textvariable = self.curConvText)
        self.kernelLength = tk.Label(self.master, text = "H:")
        self.kernelWidth = tk.Label(self.master, text = ",W:")
        self.kernel_L_SB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry")
        self.kernel_W_SB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry")
        self.convStrideLength = tk.Label(self.master, text = "H:")
        self.convStrideWidth = tk.Label(self.master, text = ",W:")
        self.strideLengthconv_Search_Bar = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry") # Length
        self.convStrideWSB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry")
        self.poolingSizeLengthLabel = tk.Label(self.master, text = "H:")
        self.poolingSizeWidthLabel = tk.Label(self.master, text = ",W:")
        self.poolingStrideLengthLabel = tk.Label(self.master, text = "H:")
        self.poolingStrideWidthLabel = tk.Label(self.master, text = ",W:")

        self.paddingSizeLengthLabel = tk.Label(self.master, text = ",R:")
        self.paddingSizeWidthLabel = tk.Label(self.master, text = ",B:")
        self.paddingSizeTopLabel = tk.Label(self.master, text = ",T:")
        self.paddingSizeLeftLabel = tk.Label(self.master, text = "L:")
        self.paddingSize_Search_Bar = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry") #Length
        self.paddingSizeWidthSB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry")
        self.paddingSize_TopSB = ttk.Entry(self.master, width = 2, style = "EntryStyle.TEntry")
        self.paddingSize_LeftSB = ttk.Entry(self.master, width = 2, style = "EntryStyle.TEntry")

        self.poolingSize_Search_Bar = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry") #Length
        self.poolingSizeWidthSB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry")
        self.stride_Length_Search_Bar = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry") #Length
        self.poolStrideWidthSB = ttk.Entry(self.master, width = 2, style="EntryStyle.TEntry") #Width
        self.convLayerProp = tk.Label(self.master, text = "Convolution Layer Properties", font = ("Mulish", 11, "bold"))
        self.convDropout = tk.Label(self.master, text = "Dropout Rate:", font = ("Mulish", 11))
        self.convDropOutEntry = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.deleteAllConvLbl = tk.Label(self.master, text = "Delete All:", font = ("Mulish", 8))
        self.deleteConvLbl = tk.Label(self.master, text = "Delete Layer:", font = ("Mulish", 8))

        self.optInOut = tk.Label(self.master, text = "Opt in/out", font = ("Mulish", 9))

        self.threadThree = None

        #Step Four Utilities
        self.fullConnCanvas = tk.Canvas(self.master, width = 403, height = 253, bg = self.theme[self.themeBool], borderwidth=0, relief="flat", highlightthickness=0,)
        self.fullConnBox = self.fullConnCanvas.create_rectangle(3,3,400,250, outline = "black", fill = self.theme[self.themeBool], width = 3)
        self.stepFourBanner = tk.Label(self.master, text = "Step Four: Enter Fully Connected Layers", font = ("Mulish", 12, "bold"))
        self.fullConnLabel = tk.Label(self.master, text = "Fully Connected Layer Properties", font = ("Mulish", 11, "bold"))
        self.fullConnNumInChannels = tk.Label(self.master, text = "Number of In Channels:", font = ("Mulish", 11))
        self.fullConnNumOutChannels = tk.Label(self.master, text = "Number of Out Channels:", font = ("Mulish", 11))
        self.inChannels_Search_Bar = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.outChannels_Search_Bar = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.curFullUp = tk.Label(self.master, text = "On Layer:")
        self.fullConnActiFunctLabel = tk.Label(self.master, text = "Activation Functions: ", font = ("Mulish", 11))
        self.curFullText = tk.StringVar(self.master)
        self.curFullText.set("")
        self.curFullLabel = tk.Label(self.master, textvariable = self.curFullText)
        self.dropout = tk.Label(self.master, text = "Dropout Rate:", font = ("Mulish", 11))
        self.dropoutEntry = ttk.Entry(self.master, width = 5, style="EntryStyle.TEntry")
        self.stepFourSubmitButton = ttk.Button(self.master, text = "Submit", command = self.stepFourSubmit, style = "Custom.TButton", takefocus=0)
        self.fullLeftButton = ttk.Button(self.master, text = "\u2190", command = self.fullBack, style = "CustomArrow.TButton", takefocus=0)
        self.fullRightButton = ttk.Button(self.master, text = "\u2192", command = self.fullForward, style = "CustomArrow.TButton", takefocus=0)
        self.deleteAllFullLbl = tk.Label(self.master, text = "Delete All:", font = ("Mulish", 8))
        self.deleteFullLbl = tk.Label(self.master, text = "Delete Layer:", font = ("Mulish", 8))

        self.actiVarFullConn = tk.StringVar(self.master)
        self.comboboxFullConn = ttk.Combobox(self.master, textvariable = self.actiVarFullConn, width = 10, style = "ComboboxStyle.TCombobox", state = "readonly", values = ["None", "ReLU", "PReLU", "ELU", "Tanh", "Sigmoid", "Softmax"])
        self.comboboxFullConn.bind("<<ComboboxSelected>>", self.callActiFuncFullConn)

        self.deleteFullLayer = tk.Label(self.master, image = self.trashImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.deleteFullLayer.bind("<ButtonRelease-1>", self.deleteFull)
        self.deleteFullLayer.bind("<Enter>", self.deleteFullEnter)
        self.deleteFullLayer.bind("<Leave>", self.deleteFullLeave)

        self.deleteAllFullLayer = tk.Label(self.master, image = self.trashImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.deleteAllFullLayer.bind("<ButtonRelease-1>", self.deleteAllFull)
        self.deleteAllFullLayer.bind("<Enter>", self.deleteAllFullEnter)
        self.deleteAllFullLayer.bind("<Leave>", self.deleteAllFullLeave)
        self.threadFour = None

        #Instructions Utilities
        self.userGuideLbl = tk.Label(self.master, text = "User Guide:")

        self.UGImage = self.imageStorage.call(13, self.themeBool)
        self.UGImage_ = self.UGImage
        self.instructionButton = tk.Label(self.master, image = self.UGImage_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.instructionButton.bind("<ButtonRelease-1>", self.instruction_page)
        self.instructionButton.bind("<Enter>", self.instructionEnter)
        self.instructionButton.bind("<Leave>", self.instructionLeave)

        #Infrastructure Utlities
        self.text_widget = scrolledtext.ScrolledText(self.master, wrap = tk.WORD, bg = self.theme[self.themeBool], state = tk.NORMAL)#, borderwidth=2)#, highlightthickness=3,)#, relief="flat")
        self.text_widget.pack(padx = 15, pady = 15, expand = True, fill = tk.BOTH)

        self.estyle.configure("My.Vertical.TScrollbar", background="green", bordercolor="red", arrowcolor="white")
        self.scrollbar = tk.Scrollbar(self.master, command=self.text_widget.yview, orient="vertical")


        self.infrastructureLbl = tk.Label(self.master, text = "CNN Architecture", font = ("Mulish", 12, "bold"))
        self.perfReport = tk.Label(self.master, text = "Performance Report", font = ("Mulish", 12, "bold"))
        self.step1 = False
        self.step2 = False
        self.notification_textOne = tk.StringVar(self.master)
        self.notificationOne = tk.Label(self.master, textvariable=self.notification_textOne)
        self.notification_textTwo = tk.StringVar(self.master)
        self.notificationTwo = tk.Label(self.master, textvariable=self.notification_textTwo)
        self.notification_textThree = tk.StringVar(self.master)
        self.notificationThree = tk.Label(self.master, textvariable=self.notification_textThree)
        self.notification_textFour = tk.StringVar(self.master)
        self.notificationFour = tk.Label(self.master, textvariable=self.notification_textFour)
        self.notification_textTrain = tk.StringVar(self.master)
        self.notificationTrain = tk.Label(self.master, textvariable=self.notification_textTrain)
        self.notification_textinTrain = tk.StringVar(self.master)
        self.notification_inTrain = tk.Label(self.master, textvariable = self.notification_textinTrain)
        self.trainButton = ttk.Button(self.master, text = "Train", command = self.check_Train, style = "Custom.TButton", takefocus=0)
        self.copyModelProp = tk.Label(self.master, text = "Copy Architecture:")


        self.copyButton = self.imageStorage.call(11, self.themeBool)
        self.copyButton_ = self.copyButton
        self.copyModelButton = tk.Label(self.master, image = self.copyButton_, borderwidth = 0, relief = "flat", highlightthickness=0)
        self.copyModelButton.bind("<ButtonRelease-1>", self.copyArchitecture)
        self.copyModelButton.bind("<Enter>", self.copyEnter)
        self.copyModelButton.bind("<Leave>", self.copyLeave)

        self.estyle.configure("Custom.TButton",
            background = self.theme_fields[False],         #border around button
            foreground = self.theme_notbold[False], #words in button
            selectfieldbackground = "orange"                 #Doesnt do shit lmao
        )

        self.estyle.configure("ComboboxStyle.TCombobox",
            background = self.theme_fields[False],
            foreground = self.theme_bold[False],
            fieldbackground = self.theme_fields[False]
        )

        self.estyle.configure("Custom2.TButton",
            background = self.theme_fields[False],         #border around button
            foreground = self.theme_notbold[False], #words in button
            selectfieldbackground = "orange"                 #Doesnt do shit lmao
        )

        self.estyle.configure("CustomArrow.TButton",
            background = self.theme_fields[False],         #border around button
            foreground = self.theme_notbold[False], #words in button
            selectfieldbackground = "orange"                 #Doesnt do shit lmao
        )

        self.estyle.configure("Custom3.TButton",
            background = self.theme_fields[False],         #border around button
            foreground = self.theme_notbold[False], #words in button
            selectfieldbackground = "orange"                 #Doesnt do shit lmao
        )

        self.estyle.map("ComboboxStyle.TCombobox", fieldbackground=[('readonly',self.theme_fields[False])])
        self.estyle.map("ComboboxStyle.TCombobox", selectbackground=[('readonly', self.theme_fields[False])])
        self.estyle.map("ComboboxStyle.TCombobox", selectforeground=[('readonly', self.theme_bold[False])])
                                #---------Instruction Page Utlities---------#
        
        self.estyle.configure("CustomHelp.TButton", font = ("Mulish", 9, "bold"), width = 9, height = 2)
        self.inInstructions = False
        self.backButton = ttk.Button(self.master, text = "Back", command = self.page_one, style = "Custom.TButton", takefocus=0)
        self.introButton = ttk.Button(self.master, text = "Intro", command = self.introIns, style = "CustomHelp.TButton", takefocus=0)
        self.step1InstructionButton = ttk.Button(self.master, text = "Step One", command = self.stepOneIns, style = "CustomHelp.TButton", takefocus=0)
        self.step2InstructionButton = ttk.Button(self.master, text = "Step Two", command = self.stepTwoIns, style = "CustomHelp.TButton", takefocus=0)
        self.step3InstructionButton = ttk.Button(self.master, text = "Step Three", command = self.stepThreeIns, style = "CustomHelp.TButton", takefocus=0)
        self.step4InstructionButton = ttk.Button(self.master, text = "Step Four", command = self.stepFourIns, style = "CustomHelp.TButton", takefocus=0)
        self.stepArchInstructionButton = ttk.Button(self.master, text = "Properties", command = self.archIns, style = "CustomHelp.TButton", takefocus=0)
        self.trainingInstructionButton = ttk.Button(self.master, text = "Training", command = self.trainIns, style = "CustomHelp.TButton", takefocus=0)
        self.devNoteButton = ttk.Button(self.master, text = "Dev's Note", command = self.devNote, style = "CustomHelp.TButton", takefocus = 0)

        self.oneIns = self.imageStorage.call(18, self.themeBool)
        self.oneInsIm = tk.Label(self.master, image = self.oneIns, borderwidth = 0, relief = "flat")

        self.twoIns = self.imageStorage.call(19, self.themeBool)
        self.twoInsIm = tk.Label(self.master, image = self.twoIns, borderwidth = 0, relief = "flat")

        self.threeIns = self.imageStorage.call(23, self.themeBool)
        self.threeInsIm = tk.Label(self.master, image = self.threeIns, borderwidth = 0, relief = "flat")

        self.fourIns = self.imageStorage.call(24, self.themeBool)
        self.fourInsIm = tk.Label(self.master, image = self.fourIns, borderwidth = 0, relief = "flat")

        self.archIns1 = self.imageStorage.call(27, self.themeBool)
        self.archIns1Im = tk.Label(self.master, image = self.archIns1, borderwidth = 0, relief = "flat")

        self.introInsTitle = tk.Label(self.master, text = "Introduction", font = ("Mulish", 11, "bold"))
        self.oneInsTitle = tk.Label(self.master, text = "Step One Guide", font = ("Mulish", 11, "bold"))
        self.twoInsTitle = tk.Label(self.master, text = "Step Two Guide", font = ("Mulish", 11, "bold"))
        self.threeInsTitle = tk.Label(self.master, text = "Step Three Guide", font = ("Mulish", 11, "bold"))
        self.FourInsTitle = tk.Label(self.master, text = "Step Four Guide", font = ("Mulish", 11, "bold"))
        self.archInsTitle = tk.Label(self.master, text = "Model Properties Guide", font = ("Mulish", 11, "bold"))
        self.trainInsTitle = tk.Label(self.master, text = "Training Guide", font = ("Mulish", 11, "bold"))
        self.devNoteTitle = tk.Label(self.master, text = "Developer's Note", font = ("Mulish", 11, "bold"))
        
        self.oneIns1 = tk.Label(self.master, text = self.imageStorage.call_(20, True), font = ("Mulish", 11), justify=tk.LEFT)
        self.twoIns2 = tk.Label(self.master, text = self.imageStorage.call_(21, True), font = ("Mulish", 11), justify = tk.LEFT)
        self.threeIns3 = tk.Label(self.master, text = self.imageStorage.call_(25, True), font = ("Mulish", 11), justify = tk.LEFT)
        self.fourIns4 = tk.Label(self.master, text = self.imageStorage.call_(26,True), font = ("Mulish", 11), justify = tk.LEFT)
        self.introIns_ = tk.Label(self.master, text = self.imageStorage.call_(22,True), font = ("Mulish", 11), justify = tk.LEFT)
        self.arch1Ins = tk.Label(self.master, text = self.imageStorage.call_(28, True), font = ("Mulish", 11), justify  = tk.LEFT)
        self.trainingIns_ = tk.Label(self.master, text = self.imageStorage.call_(33, True), font = ("Mulish", 11), justify = tk.LEFT)
        self.devNoteIns_ = tk.Label(self.master, text = self.imageStorage.call_(36, True), font = ("Mulish", 11), justify = tk.LEFT)

        self.archForwardButton = ttk.Button(self.master, text = "\u2192", command = self.archNext, style = "CustomArrow.TButton", takefocus=0)
        self.archBackwardButton = ttk.Button(self.master, text = "\u2190", command = self.archBack, style = "CustomArrow.TButton", takefocus=0)
        self.archLabel = tk.Label(self.master, text = "Infrastructure Page One: Step 1 and 2")
        self.archConst = 1

        self.pageOneList = [self.text_widget, self.scrollbar, self.dataLoadCanvas, self.stepOneBanner, self.dataPathWay, self.imageSize, self.imageWidth, self.imageHeight, self.dataPathSearch_Bar,
        self.heightSizeSearch_Bar, self.widthSizeSearch_Bar, self.stepOneSubmitButton, self.dataSplitCanvas, self.stepTwoBanner, self.training_split, self.testing_split,
        self.validation_split, self.training_Split_Search_Bar, self.validation_Split_Search_Bar, self.testing_Split_Search_Bar, self.stepTwoSubmitButton, self.batchSizeLabel,
        self.learningRateLabel, self.numEpochsLabel, self.batchSizeSB, self.lrSB, self.numEpochsSB, self.optimizerLabel, self.optiComboBox, self.convLayerCanvas,
        self.stepThreeBanner, self.convLayerProp, self.numInChannels, self.numOutChannels, self.kernelSize, self.strideLength, self.in_Channel_Search_Bar,
        self.out_Channel_Search_Bar, self.strideLengthconv_Search_Bar, self.paddingAndPoolingLayer, self.paddingLabel, self.poolingLabel, self.sizeLabel, self.strideLengthLabel, self.poolingSize_Search_Bar, self.paddingSize_Search_Bar,
        self.stride_Length_Search_Bar, self.activationFunctions, self.combobox, self.stepThreeSubmitButton, self.deleteConvLayer, self.deleteAllConvLayer,
        self.convLeftButton, self.convRightButton, self.fullConnCanvas, self.stepFourBanner, self.fullConnLabel, self.fullConnNumInChannels, self.fullConnNumOutChannels, self.inChannels_Search_Bar,
        self.outChannels_Search_Bar, self.comboboxFullConn, self.fullConnActiFunctLabel, self.stepFourSubmitButton, self.fullLeftButton, self.fullRightButton, self.deleteFullLayer, self.deleteAllFullLayer,
        self.infrastructureLbl,self.trainButton,self.notificationOne, self.notificationTwo,self.notificationThree, self.notificationFour,
        self.notificationTrain, self.normalizeDataLbl, self.curConvLabel, self.kernelLength,self.kernelWidth, self.kernel_L_SB, self.kernel_W_SB,
        self.convStrideLength,self.convStrideWidth,self.strideLengthconv_Search_Bar,self.convStrideWSB, self.poolingSizeLengthLabel,self.poolingSizeWidthLabel,self.poolingStrideLengthLabel,
        self.poolingStrideWidthLabel,self.paddingSizeLengthLabel,self.paddingSizeWidthLabel,self.paddingSizeWidthSB,self.poolingSizeWidthSB,self.poolStrideWidthSB,self.curConvUp,self.curFullUp,
        self.curConvLabel,self.curFullLabel, self.datasplitLbl, self.modelpropertieslbl, self.trainingOptimizationLbl, self.weightDecay, self.weightDecayEntry, self.convDropout, self.convDropOutEntry,
        self.dropout, self.dropoutEntry, self.verticalLineCanvas, self.earlyStoppingEntry, self.earlyStoppingLabel, self.dataTypeLabel, self.archCB, self.chooseModelLabel, self.archType,
        self.configureData, self.dataCB, self.criComboBox, self.criterionLbl, self.copyModelProp, self.copyModelButton, self.loadPreTrained, self.loadPTBoolButton, self.deleteAllFullLbl, self.deleteFullLbl,
        self.deleteAllConvLbl, self.deleteConvLbl, self.optInOut, self.poolingCheckBox, self.paddingCheckBox, self.normCheckBox, self.paddingSizeTopLabel, self.paddingSizeLeftLabel, self.paddingSize_LeftSB, self.paddingSize_TopSB]
        self.training_page = [self.saveStateButton, self.saveStateLbl, self.modelPerf, self.perfReport, self.featureMaps, self.statsPageLabel, self.trainForwardButton, self.trainBackwardButton, self.statsReport, self.featureNumReport, 
        self.featureNumForwardButton, self.featureNumBackwardButton, self.featureConvLayerReport, self.featureConvLayerForwardButton, self.featureConvLayerBackwardButton, self.featureMapConvPage, 
        self.featureNumChannel, self.popGraphOne, self.popGraphTwo, self.popGraphThree, self.popGraphFour, self.retrain, self.traindifferent, self.downloadModel, self.downloadCode,self.downloadstatsLbl,
        self.downloadStatsButton, self.popGraph1, self.popGraph2, self.popGraph3, self.popGraph4, self.downloadCodeText, self.downloadModelText, self.reTrainModelText, self.trainAnotherText, self.popConMat, self.popCM]
        self.instruction_Page = [self.step1InstructionButton, self.step2InstructionButton, self.step3InstructionButton, self.step4InstructionButton, self.stepArchInstructionButton, self.introButton, self.trainingInstructionButton, self.devNoteButton]
        self.intro_instruction = [self.introInsTitle, self.introIns_]
        self.stepOne_instruction = [self.oneInsIm, self.oneInsTitle, self.oneIns1]
        self.stepTwo_instruction = [self.twoInsTitle, self.twoInsIm, self.twoIns2]
        self.stepThree_instruction = [self.threeInsTitle, self.threeInsIm, self.threeIns3]
        self.stepFour_instruction = [self.FourInsTitle, self.fourInsIm, self.fourIns4]
        self.archInstruction = [self.archInsTitle, self.archIns1Im, self.arch1Ins, self.archForwardButton, self.archBackwardButton, self.archLabel]
        self.trainingInstruction = [self.trainInsTitle, self.trainingIns_]
        self.devsNoteInstruction = [self.devNoteTitle, self.devNoteIns_]

        self.backButton.lift()
        self.step1InstructionButton.lift()
        self.step2InstructionButton.lift()
        self.step3InstructionButton.lift()
        self.step4InstructionButton.lift()
        self.stepArchInstructionButton.lift()
        self.trainingInstructionButton.lift()
        self.devNoteButton.lift()
        self.oneInsIm.lift()
        self.twoInsIm.lift()
        self.threeInsIm.lift()
        self.fourInsIm.lift()
        self.introButton.lift()
        self.archIns1Im.lift()
        self.switchTheme(event = False)


    def switchTheme(self, event):
        if self.themeButton_tag:
            if self.themeBool:
                self.themeBool = False
            else:
                self.themeBool = True
            if len(self.dataTrain) != 0:
                report = self.dataTrain[11]
            else:
                report = {}
            if not self.inInstructions:
                self.infrastructureVisualization(shapes = self.backEnd.convShapes, shapes2 = self.backEnd.fullShapes, Train = not self.inPageOne, report = report , theme_ = self.themeBool)
            
            #Other
            self.userGuideLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.themeLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])

            #Step One
            self.uiCanvas.configure(bg = self.theme[self.themeBool])
            self.uiCanvas.itemconfig(self.uiBox, fill=self.theme[self.themeBool], outline = self.theme_bold[self.themeBool])
            self.logo.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.madeBy.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.stepOneBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.stepTwoBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.stepThreeBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.stepFourBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.infrastructureLbl.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.copyModelProp.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.configureData.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.loadPreTrained.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.dataPathWay.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.dataTypeLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.imageSize.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.imageWidth.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.imageHeight.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.chooseModelLabel.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.archType.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])

            #Step Two
            self.dataSplitCanvas.configure(bg = self.theme[self.themeBool])
            self.dataSplitCanvas.itemconfig(self.dataSplitBox, fill=self.theme[self.themeBool], outline = self.theme_bold[self.themeBool])
            self.datasplitLbl.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.modelpropertieslbl.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.stepTwoBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.training_split.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.testing_split.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.validation_split.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.batchSizeLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.learningRateLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.numEpochsLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.optimizerLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.trainingOptimizationLbl.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.normalizeDataLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.weightDecay.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.earlyStoppingLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.criterionLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.verticalLineCanvas.configure(bg = self.theme[self.themeBool])
            self.verticalLineCanvas.itemconfig(self.verticalLine, fill=self.theme_bold[self.themeBool])

            #Step Three
            self.convLayerCanvas.configure(bg = self.theme[self.themeBool])
            self.convLayerCanvas.itemconfig(self.convLayerBox, fill=self.theme[self.themeBool], outline = self.theme_bold[self.themeBool])
            self.curConvUp.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.stepThreeBanner.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.numInChannels.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.numOutChannels.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.kernelSize.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.strideLength.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingAndPoolingLayer.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.paddingLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.poolingLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.sizeLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.optInOut.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.strideLengthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.activationFunctions.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.curConvLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.kernelLength.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.kernelWidth.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.convStrideLength.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.convStrideWidth.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.poolingSizeLengthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.poolingSizeWidthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.poolingStrideLengthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.poolingStrideWidthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingSizeLengthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingSizeWidthLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingSizeTopLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingSizeLeftLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.convLayerProp.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.convDropout.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.deleteAllConvLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.deleteConvLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.paddingCheckBox.configure(fg = self.theme[self.themeBool], bg = self.theme_fields[self.themeBool])

            #Step Four
            self.fullConnCanvas.configure(bg = self.theme[self.themeBool])
            self.fullConnCanvas.itemconfig(self.fullConnBox, fill=self.theme[self.themeBool], outline = self.theme_bold[self.themeBool])
            self.curFullUp.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.curFullLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.deleteAllFullLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.deleteFullLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.fullConnLabel.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.fullConnNumInChannels.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.fullConnNumOutChannels.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.dropout.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.fullConnActiFunctLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])

            #Notifications
            self.notificationOne.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.notificationTwo.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.notificationThree.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.notificationFour.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.notificationTrain.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.notification_inTrain.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])

            #Training Page
            self.modelPerf.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.featureMaps.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.statsPageLabel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.statsReport.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.featureNumReport.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.featureConvLayerReport.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.featureMapConvPage.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.featureNumChannel.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.downloadstatsLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.trainAnotherText.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.reTrainModelText.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.downloadModelText.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.downloadCodeText.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.popGraph1.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.popGraph2.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.popGraph3.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.popGraph4.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.perfReport.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.saveStateLbl.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            self.popConMat.configure(fg = self.theme_notbold[self.themeBool], bg = self.theme[self.themeBool])
            
            self.popGraphImage = self.imageStorage.call(1,self.themeBool)
            self.popGraphOne.configure(image = self.popGraphImage)
            self.popGraphTwo.configure(image = self.popGraphImage)
            self.popGraphThree.configure(image = self.popGraphImage)
            self.popGraphFour.configure(image = self.popGraphImage)
            self.popCM.configure(image = self.popGraphImage)
            self.trainDiffNew = self.imageStorage.call(3,self.themeBool)
            self.traindifferent.configure(image = self.trainDiffNew)
            self.retrainImage = self.imageStorage.call(5, self.themeBool)
            self.retrain.configure(image = self.retrainImage)

            self.downloadButton = self.imageStorage.call(7, self.themeBool)
            self.downloadModel.configure(image = self.downloadButton)
            self.downloadCode.configure(image = self.downloadButton)
            self.downloadStatsButton.configure(image = self.downloadButton)
            self.saveStateButton.configure(image = self.downloadButton)

            self.themeImage = self.imageStorage.call(9, self.themeBool)
            self.themeButton.configure(image = self.themeImage)

            self.copyButton = self.imageStorage.call(11, self.themeBool)
            self.copyModelButton.configure(image = self.copyButton)

            self.UGImage = self.imageStorage.call(13, self.themeBool)
            self.instructionButton.configure(image = self.UGImage)

            self.trashAllFull = self.imageStorage.call(15, self.themeBool)
            self.deleteAllFullLayer.configure(image = self.trashAllFull)

            self.trashImage = self.imageStorage.call(15, self.themeBool)
            self.deleteFullLayer.configure(image = self.trashImage)

            self.trashConvImage = self.imageStorage.call(15, self.themeBool)
            self.deleteConvLayer.configure(image = self.trashConvImage)

            self.trashConvAllImage = self.imageStorage.call(15, self.themeBool)
            self.deleteAllConvLayer.configure(image = self.trashConvAllImage)

            self.oneIns = self.imageStorage.call(18, self.themeBool)
            self.oneInsIm.configure(image = self.oneIns)

            self.twoIns = self.imageStorage.call(19, self.themeBool)
            self.twoInsIm.configure(image = self.twoIns)

            self.threeIns = self.imageStorage.call(23, self.themeBool)
            self.threeInsIm.configure(image = self.threeIns)

            self.fourIns = self.imageStorage.call(24, self.themeBool)
            self.fourInsIm.configure(image = self.fourIns)

            im = {1: 27, 2: 29, 3: 31}
            self.archIns1 = self.imageStorage.call(im[self.archConst], self.themeBool)
            self.archIns1Im.configure(image = self.archIns1)

            self.estyle.configure("TFrame", background='#343541' if self.themeBool else '#F0F0F0')
            self.estyle.configure("TFrame")


            self.oneInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.oneIns1.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.twoIns2.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.threeIns3.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.fourIns4.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.introIns_.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.twoInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.threeInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.introInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.FourInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.archInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.arch1Ins.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.trainingIns_.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.devNoteIns_.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.archLabel.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.trainInsTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            self.devNoteTitle.configure(fg= self.theme_bold[self.themeBool], bg= self.theme[self.themeBool])
            
            self.estyle.configure("EntryStyle.TEntry",
                background = self.theme_fields[self.themeBool],
                foreground = self.theme_bold[self.themeBool],
                fieldbackground = self.theme_fields[self.themeBool]
            )

            self.estyle.configure("Minimal.TCheckbutton",
                background = self.theme[self.themeBool],
                foreground = self.theme_bold[self.themeBool],
                fieldbackground = self.theme_fields[self.themeBool],
                hoverbackground= self.theme[self.themeBool],
                highlightcolor= 'blue'
            )

            self.estyle.configure("ComboboxStyle.TCombobox",
                background = self.theme_fields[self.themeBool],
                foreground = self.theme_bold[self.themeBool],
                fieldbackground = self.theme_fields[self.themeBool]
            )

            self.estyle.configure("Custom.TButton",
                background = self.theme_fields[self.themeBool],         #border around button
                foreground = self.theme_bold[self.themeBool], #words in button
                selectfieldbackground = "orange"                 #Doesnt do shit lmao
            )

            self.estyle.configure("Custom2.TButton",
                background = self.theme_fields[self.themeBool],         #border around button
                foreground = self.theme_notbold[self.themeBool], #words in button
                selectfieldbackground = "orange"                 #Doesnt do shit lmao
            )

            self.estyle.configure("CustomArrow.TButton",
                background = self.theme_fields[self.themeBool],         #border around button
                foreground = self.theme_notbold[self.themeBool], #words in button
                selectfieldbackground = "orange"                 #Doesnt do shit lmao
            )

            self.estyle.configure("Custom3.TButton",
                background = self.theme_fields[self.themeBool],         #border around button
                foreground = self.theme_notbold[self.themeBool], #words in button
                selectfieldbackground = "orange"                 #Doesnt do shit lmao
            )

            self.estyle.configure("CustomHelp.TButton",
                background = self.theme_fields[self.themeBool],         #border around button
                foreground = self.theme_notbold[self.themeBool], #words in button
                selectfieldbackground = "orange"                 #Doesnt do shit lmao
            )

            self.estyle.configure("arrowless.Vertical.TScrollbar", troughcolor="blue")

            self.estyle.map("ComboboxStyle.TCombobox", fieldbackground=[('readonly',self.theme_fields[self.themeBool])])
            self.estyle.map("ComboboxStyle.TCombobox", selectbackground=[('readonly', self.theme_fields[self.themeBool])])
            self.estyle.map("ComboboxStyle.TCombobox", selectforeground=[('readonly', self.theme_bold[self.themeBool])])

            self.dataLoadCanvas.configure(bg = self.theme[self.themeBool])
            self.dataLoadCanvas.itemconfig(self.dataLoadBox, fill = self.theme[self.themeBool] , outline = self.theme_bold[self.themeBool])
            self.fig.set_facecolor(self.theme[self.themeBool])
            self.fig2.set_facecolor(self.theme[self.themeBool])
            self.fig3.set_facecolor(self.theme[self.themeBool])
            self.fig4.set_facecolor(self.theme[self.themeBool])
            if self.addPaddingBool:
                self.checkBoxIm = self.imageStorage.call(35, self.themeBool)
                self.checkBoxIm_ = self.checkBoxIm
                self.paddingCheckBox.configure(image = self.checkBoxIm_)
            if not self.addPaddingBool:
                self.checkBoxIm = self.imageStorage.call(34, self.themeBool)
                self.checkBoxIm_ = self.checkBoxIm
                self.paddingCheckBox.configure(image = self.checkBoxIm_)

            if self.addPoolingBool:
                self.checkBoxIm2 = self.imageStorage.call(35, self.themeBool)
                self.checkBoxIm2_ = self.checkBoxIm2
                self.poolingCheckBox.configure(image = self.checkBoxIm2_)
            if not self.addPoolingBool:
                self.checkBoxIm2 = self.imageStorage.call(34, self.themeBool)
                self.checkBoxIm2_ = self.checkBoxIm2
                self.poolingCheckBox.configure(image = self.checkBoxIm2_)

            if self.addNormalize:
                self.checkBoxNorm = self.imageStorage.call(35, self.themeBool)
                self.checkBoxNorm_ = self.checkBoxNorm
                self.normCheckBox.configure(image = self.checkBoxNorm_)
            if not self.addNormalize:
                self.checkBoxNorm = self.imageStorage.call(34, self.themeBool)
                self.checkBoxNorm_ = self.checkBoxNorm
                self.normCheckBox.configure(image = self.checkBoxNorm_)

            if not self.inPageOne:
                self.update_gui(self.dataTrain)
    
    def addNormalizeFunc(self, event):
        if self.normalizeConst:
            self.addNormalize = not self.addNormalize
            if self.addNormalize:
                self.checkBoxNorm = self.imageStorage.call(35, self.themeBool)
                self.checkBoxNorm_ = self.checkBoxNorm
                self.normCheckBox.configure(image = self.checkBoxNorm_)
            if not self.addNormalize:
                self.checkBoxNorm = self.imageStorage.call(34, self.themeBool)
                self.checkBoxNorm_ = self.checkBoxNorm
                self.normCheckBox.configure(image = self.checkBoxNorm_)

    def addPooling(self, event):
        if self.poolingCheckConst:
            self.addPoolingBool = not self.addPoolingBool
            if self.addPoolingBool:
                self.checkBoxIm2 = self.imageStorage.call(35, self.themeBool)
                self.checkBoxIm2_ = self.checkBoxIm2
                self.poolingCheckBox.configure(image = self.checkBoxIm2_)
            if not self.addPoolingBool:
                self.checkBoxIm2 = self.imageStorage.call(34, self.themeBool)
                self.checkBoxIm2_ = self.checkBoxIm2
                self.poolingCheckBox.configure(image = self.checkBoxIm2_)

    def paddingCheckBoxFunc(self, event):
        if self.paddingCheckConst:
            self.addPaddingBool = not self.addPaddingBool
            if self.addPaddingBool:
                self.checkBoxIm = self.imageStorage.call(35, self.themeBool)
                self.checkBoxIm_ = self.checkBoxIm
                self.paddingCheckBox.configure(image = self.checkBoxIm_)
            if not self.addPaddingBool:
                self.checkBoxIm = self.imageStorage.call(34, self.themeBool)
                self.checkBoxIm_ = self.checkBoxIm
                self.paddingCheckBox.configure(image = self.checkBoxIm_)

    def normEnter(self, event):
        self.normalizeConst = True

    def normLeave(self, event):
        self.normalizeConst = False

    def poolingEnter(self, event):
        self.poolingCheckConst = True

    def poolingLeave(self, event):
        self.poolingCheckConst = False

    def paddingEnter(self, event):
        self.paddingCheckConst = True

    def paddingLeave(self, event):
        self.paddingCheckConst = False

    def deleteAllConvEnter(self, event):
        self.deleteAllConv_tag = True
        self.trashConvAllImage = self.imageStorage.call(16, self.themeBool)
        self.deleteAllConvLayer.configure(image = self.trashConvAllImage)

    def deleteAllConvLeave(self, event):
        self.deleteAllConv_tag = False
        self.trashConvAllImage = self.imageStorage.call(15, self.themeBool)
        self.deleteAllConvLayer.configure(image = self.trashConvAllImage)

    def deleteConvEnter(self, event):
        self.deleteConv_tag = True
        self.trashConvImage = self.imageStorage.call(16, self.themeBool)
        self.deleteConvLayer.configure(image = self.trashConvImage)

    def deleteConvLeave(self, event):
        self.deleteConv_tag = False
        self.trashConvImage = self.imageStorage.call(15, self.themeBool)
        self.deleteConvLayer.configure(image = self.trashConvImage)

    def deleteFullEnter(self, event):
        self.deleteFull_tag = True
        self.trashImage = self.imageStorage.call(16, self.themeBool)
        self.deleteFullLayer.configure(image = self.trashImage)

    def deleteFullLeave(self, event):
        self.deleteFull_tag = False
        self.trashImage = self.imageStorage.call(15, self.themeBool)
        self.deleteFullLayer.configure(image = self.trashImage)

    def deleteAllFullEnter(self, event):
        self.deleteAllFull_tag = True
        self.trashAllFull = self.imageStorage.call(16, self.themeBool)
        self.deleteAllFullLayer.configure(image = self.trashAllFull)

    def deleteAllFullLeave(self, event):
        self.deleteAllFull_tag = False
        self.trashAllFull = self.imageStorage.call(15, self.themeBool)
        self.deleteAllFullLayer.configure(image = self.trashAllFull)

    def instructionEnter(self, event):
        self.instructionButton_tag = True
        self.UGImage = self.imageStorage.call(14, self.themeBool)
        self.instructionButton.configure(image = self.UGImage)

    def instructionLeave(self, event):
        self.instructionButton_tag = False
        self.UGImage = self.imageStorage.call(13, self.themeBool)
        self.instructionButton.configure(image = self.UGImage)

    def copyEnter(self, event):
        self.copyButton_tag = True
        self.copyNew = self.imageStorage.call(12, self.themeBool)
        self.copyModelButton.configure(image = self.copyNew)

    def copyLeave(self, event):
        self.copyButton_tag = False
        self.copyNew = self.imageStorage.call(11, self.themeBool)
        self.copyModelButton.configure(image = self.copyNew)

    def themeEnter(self, event):
        self.themeButton_tag = True
        self.themeNew = self.imageStorage.call(10, self.themeBool)
        self.themeButton.configure(image = self.themeNew)

    def themeLeave(self, event):
        self.themeButton_tag = False
        self.themeNew = self.imageStorage.call(9, self.themeBool)
        self.themeButton.configure(image = self.themeNew)

    def downloadStateEnter(self, event):
        self.downloadState_tag = True
        self.downModel4 = self.imageStorage.call(8, self.themeBool)
        self.saveStateButton.configure(image = self.downModel4)

    def downloadStateLeave(self, event):
        self.downloadState_tag = False
        self.downModel4 = self.imageStorage.call(7, self.themeBool)
        self.saveStateButton.configure(image = self.downModel4)

    def downloadCodeEnter(self, event):
        self.downloadCode_tag = True
        self.downModel3 = self.imageStorage.call(8, self.themeBool)
        self.downloadCode.configure(image = self.downModel3)

    def downloadCodeLeave(self, event):
        self.downloadCode_tag = False
        self.downModel3 = self.imageStorage.call(7, self.themeBool)
        self.downloadCode.configure(image = self.downModel3)

    def downloadStatsEnter(self, event):
        self.downloadStats_tag = True
        self.downModel2 = self.imageStorage.call(8, self.themeBool)
        self.downloadStatsButton.configure(image = self.downModel2)

    def downloadStatsLeave(self, event):
        self.downloadStats_tag = False
        self.downModel2 = self.imageStorage.call(7, self.themeBool)
        self.downloadStatsButton.configure(image = self.downModel2)

    def downloadModelEnter(self, event):
        self.downloadModel_tag = True
        self.downModel = self.imageStorage.call(8, self.themeBool)
        self.downloadModel.configure(image = self.downModel)

    def downloadModelLeave(self, event):
        self.downloadModel_tag = False
        self.downModel = self.imageStorage.call(7, self.themeBool)
        self.downloadModel.configure(image = self.downModel)

    def reTrainOn_leave(self, event):
        self.reTrain_tag = False
        self.retrainImage = self.imageStorage.call(5, self.themeBool)
        self.retrain.configure(image = self.retrainImage)

    def reTrainOn_enter(self, event):
        self.reTrain_tag = True
        self.retrainImage = self.imageStorage.call(6, self.themeBool)
        self.retrain.configure(image = self.retrainImage)

    def trainDiffOn_enter(self,event):
        self.trainDiff_tag = True
        self.trainDiffNew = self.imageStorage.call(4,self.themeBool)
        self.traindifferent.configure(image = self.trainDiffNew)

    def trainDiffOn_leave(self, event):
        self.trainDiff_tag = False
        self.trainDiffNew = self.imageStorage.call(3,self.themeBool)
        self.traindifferent.configure(image = self.trainDiffNew)

    def PopOneOn_enter(self, event):
        self.popOne_tag = True
        self.popGraphImageNew = self.imageStorage.call(2,self.themeBool)
        self.popGraphOne.configure(image = self.popGraphImageNew)

    def PopOneOn_leave(self, event):
        self.popOne_tag = False
        self.popGraphImageNew = self.imageStorage.call(1,self.themeBool)
        self.popGraphOne.configure(image = self.popGraphImageNew)

    def PopTwoOn_enter(self, event):
        self.popTwo_tag = True
        self.popGraphImageNew2 = self.imageStorage.call(2,self.themeBool)
        self.popGraphTwo.configure(image = self.popGraphImageNew2)

    def PopTwoOn_leave(self, event):
        self.popTwo_tag = False
        self.popGraphImageNew2 = self.imageStorage.call(1,self.themeBool)
        self.popGraphTwo.configure(image = self.popGraphImageNew2)

    def PopThreeOn_enter(self, event):
        self.popThree_tag = True
        self.popGraphImageNew3 = self.imageStorage.call(2,self.themeBool)
        self.popGraphThree.configure(image = self.popGraphImageNew3)

    def PopThreeOn_leave(self, event):
        self.popThree_tag = False
        self.popGraphImageNew3 = self.imageStorage.call(1,self.themeBool)
        self.popGraphThree.configure(image = self.popGraphImageNew3)

    def PopFourOn_enter(self, event):
        self.popFour_tag = True
        self.popGraphImageNew4 = self.imageStorage.call(2,self.themeBool)
        self.popGraphFour.configure(image = self.popGraphImageNew4)

    def PopFourOn_leave(self, event):
        self.popFour_tag = False
        self.popGraphImageNew4 = self.imageStorage.call(1,self.themeBool)
        self.popGraphFour.configure(image = self.popGraphImageNew4)

    def PopCM_enter(self, event):
        self.popCM_tag = True
        self.popGraphImageNewCM = self.imageStorage.call(2,self.themeBool)
        self.popCM.configure(image = self.popGraphImageNewCM)

    def PopCM_leave(self,event):
        self.popCM_tag = False
        self.popGraphImageNewCM = self.imageStorage.call(1, self.themeBool)
        self.popCM.configure(image = self.popGraphImageNewCM)

    def downloadState(self, event):
        if self.downloadState_tag:
            if self.training_thread.is_alive():
                self.notification_textinTrain.set("Model is still being trained")
                if not self.backEnd.doingRegression:
                    self.notification_inTrain.place(x = 965, y = 420)
                else:
                    self.notification_inTrain.place(x = 965, y = 630)
                return None

            self.notification_inTrain.place_forget()
            self.backEnd.downloadModelState()
            return None

    def downloadModelFunc(self, event):
        if self.downloadModel_tag:
            if self.training_thread.is_alive():
                self.notification_textinTrain.set("Model is still being trained")
                if not self.backEnd.doingRegression:
                    self.notification_inTrain.place(x = 965, y = 420)
                else:
                    self.notification_inTrain.place(x = 965, y = 630)
                return None

            self.notification_inTrain.place_forget()
            self.backEnd.downloadModel()
            return None

    def downloadStats(self, event):
        if self.downloadStats_tag:
            self.backEnd.download_CSV()     

    def copyArchitecture(self, event):
        if self.copyButton_tag:
            copyStatus = self.backEnd.copyCNNArchitecture()
            if copyStatus:
                self.notification_textTrain.set("Architecture copied succesfully")
                self.notificationTrain.place(x = 960, y = 635)
            if not copyStatus:
                self.notification_textTrain.set("Architecture failed to copy. Finish Model first")
                self.notificationTrain.place(x = 960, y = 635)
            return None

    def callCriterion(self, event):
        self.criterion = self.criVar.get()

    def callPreArch(self, event):
        self.preArch = self.archvar.get()

    def downloadCodeFunc(self, event):
        if self.downloadCode_tag:
            #input_file = "ACNCode.py"
            input_file = self.imageStorage.call_(37, True)

            # Ask the user to select a file location and name
            file_name = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python Files", "*.py")],
                title="Save Training Code As"
            )

            conv_string = ''
            for i in self.backEnd.convLayers:
                conv_string += "train_model.convLayers[" + str(i) + "] = " + str(self.backEnd.convLayers[i]) + '\n    '

            full_string = ''
            for i in self.backEnd.fullLayers:
                full_string += "train_model.fullLayers[" + str(i) + "] = " + str(self.backEnd.fullLayers[i]) + '\n    '

            preLoad = ''
            if self.backEnd.preTrainedModel is not None:
                preLoad += 'train_model.preTrainedModel = torch.load(r"' + str(self.loadPTBoolButton.get()) + '")'

            file_content = """
if __name__ == "__main__":
    train_model = BackEnd()
    data_link =  r'""" + str(self.backEnd.datalink) + """'
    datatype = '""" + str(self.dataType) + """'
    if datatype == ".jpg":
        train_model.get_data(data_link)
        train_model.load_data(""" + str(self.backEnd.height) +""",""" + str(self.backEnd.width) + """)

    if datatype == ".csv":
        train_model.load_data_csv(data_link, """ + str(self.backEnd.height) + """,""" + str(self.backEnd.width) + """)

    if datatype == ".bin":
        train_model.load_data_binary(data_link, """ + str(self.backEnd.height) + """,""" + str(self.backEnd.width) + """)

    print("Number of Data: " + str(len(train_model.dataset)))

    # Splitting Data      
    train_model.split_data("""+ str(self.backEnd.batchSize) +""", """ + str(self.backEnd.addNormalize) + """, """ + str(self.backEnd.trainingSplit) + """, """ + str(self.backEnd.validationSplit) + """, """ + str(self.backEnd.testingSplit) + """, """ + str(self.backEnd.learningRate) + """, """ + str(self.backEnd.epochs) + """, """ + str(self.backEnd.weightdecay) + """, """ + str(self.backEnd.earlyStopping) + """ , '""" + str(self.backEnd.optimizer) + """', '""" + str(self.backEnd.critIdentifier) + """')
    
    # Add Convolutional Layers
    """ + conv_string + """
    # Add Fully connected Layers
    """ + full_string + """

    """ + preLoad + """

    # Train the model
    epochs, training_accuracy, valid_accuracy, training_loss, valid_loss, mse_plotting, mae_plotting, testing_acc, true_pos, true_neg, false_pos, false_neg, ma_precision, ma_recall, ma_f1, precision, recall, f1_score  = train_model.trainingLoop()
    
    #Plot Accuracies
    plot_acc = True

    #Plot Losses
    plot_loss = True

    #Plot Error
    plot_error = True

    #Plot True Negative/Positive
    plot_true_np = True

    #Plot False Negative/Positive
    plot_false_np = True

    #Plot Precision
    plot_precision = True

    #Plot Recall
    plot_recall = True

    #Plot F1
    plot_f1 = True

    train_model.plotting(epochs, training_accuracy, valid_accuracy, training_loss, valid_loss, mse_plotting, mae_plotting, testing_acc, true_pos, true_neg, false_pos, false_neg, ma_precision, ma_recall, ma_f1, precision, recall, f1_score, plot_acc, plot_loss, plot_error, plot_true_np, plot_false_np, plot_precision, plot_recall)
    """

            input_file += '\n' + file_content
            with open(file_name, "w") as file:
                file.write(input_file)


    def traindifferentfunc(self, event, key = False):
        if self.trainDiff_tag or key:
            if self.threadTrainDifferent is not None:
                if self.threadTrainDifferent.is_alive():
                    return None

            self.threadTrainDifferent = threading.Thread(target=lambda: self.traindifferentload())
            self.threadTrainDifferent.daemon = True
            self.threadTrainDifferent.start()

    def traindifferentload(self):
        self.notification_textinTrain.set("Stopping training process")
        self.notification_inTrain.place(x = 965, y = 420)
        self.stop_training_event.set()
        
        if self.training_thread.is_alive():
            self.training_thread.join()

        self.stop_training_event.clear()


        for i in self.training_page:
            i.pack()
            i.pack_forget()
        self.queue.queue.clear()
        self.notification_inTrain.place_forget()

        self.chart_widget.destroy()
        self.chart2_widget.destroy()
        self.chart3_widget.destroy()
        self.chart4_widget.destroy()
        self.chart_heatmap_widget.destroy()
        self.master.update()
        self.doneTraining = False
        self.dataTrain = {}
        self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
        self.curConvLabel.place(x = 362, y = 615)
        self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
        self.curFullLabel.place(x = 841, y = 615)
        self.ES = False
        self.page_one()
        return None

    def callDataType(self, event):
        self.dataType = self.datavar.get()


    def retrainfunc(self, event, key = False):
        if self.reTrain_tag or key:
            if self.threadReTrain is not None:
                if self.threadReTrain.is_alive():
                    return None

            self.threadReTrain = threading.Thread(target=lambda: self.retrainfuncLoad())
            self.threadReTrain.daemon = True
            self.threadReTrain.start()

    def retrainfuncLoad(self):
        self.notification_textinTrain.set("Stopping training process for retraining")
        self.notification_inTrain.place(x = 965, y = 420)
        self.stop_training_event.set()

        if self.training_thread.is_alive():
            self.training_thread.join()

        self.stop_training_event.clear()

        self.chart_widget.destroy()
        self.chart2_widget.destroy()
        self.chart3_widget.destroy()
        self.chart4_widget.destroy()
        self.chart_heatmap_widget.destroy()
        self.backEnd.split_data(self.backEnd.batchSize, self.addNormalize, self.backEnd.trainingSplit, self.backEnd.validationSplit, self.backEnd.testingSplit, self.backEnd.learningRate,self.backEnd.epochs, self.backEnd.weightdecay, self.backEnd.earlyStopping, self.backEnd.optimizer, self.backEnd.critIdentifier)
        self.backEnd.model = None
        self.ES = False
        self.doneTraining = False
        self.dataTrain = {}
        self.queue.queue.clear()
        self.check_Train()
        return None

    def popCMFunc(self, event, key = False):
        if self.popCM_tag or key:
            plt.close()
            data = self.dataTrain
            if len(data) != 0:

                plt.figure(figsize = (7,7))
                sns.heatmap(data[23], linewidths = 0.1, cmap = 'Greens', linecolor = 'gray', fmt = '.1f', annot = True)
                plt.xlabel('Predicted', fontsize = 9)
                plt.ylabel('Actual', fontsize = 9)
                plt.title('Confusion Matrix', fontsize = 10)
                plt.show()
            if len(data) == 0:

                plt.figure(figsize = (7,7))
                sns.heatmap(len(self.backEnd.dataPerLabel) * [[0] * len(self.backEnd.dataPerLabel)], linewidths = 0.1, cmap = 'Greens', linecolor = 'gray', fmt = '.1f', annot = True)
                plt.xlabel('Predicted', fontsize = 9)
                plt.ylabel('Actual', fontsize = 9)
                plt.title('Confusion Matrix', fontsize = 10)
                plt.show()


    def pop4(self, event, key = False):
        if self.popFour_tag or key: 
            plt.close()
            data = self.dataTrain

            if len(data) != 0:
                plt.figure(figsize = (12,3))
                for i in range(len(data[18][data[19][self.featureMapConv]][0,:])):
                    plt.subplot(math.ceil(len(data[18][data[19][self.featureMapConv]][0,:])/6), 6, i + 1)
                    plt.imshow(data[18][data[19][self.featureMapConv]][0, i].detach(), cmap='viridis')
                    plt.axis('on')
                plt.suptitle("Feature Map for Convolutional Layer " + str(self.featureMapConv))
                plt.show()
            
            if len(data) == 0:
                plt.figure(figsize = (12,3))
                for i in range(6):
                    plt.subplot(1, 6, i + 1)
                    plt.plot([0,1], [0,1], color = 'white')
                    plt.axis('on')
                plt.suptitle("Feature Map for Convolutional Layer " + str(self.featureMapConv))
                plt.show()
            return None

    def pop3(self, event, key = False):
        if self.popThree_tag or key:
            plt.close()
            data = self.dataTrain
            if len(data) != 0:
                data_use = {0: data[11], 1: {1: "Training and Validation Accuracy", 2: data[0], 3: data[1], 4:data[2], 5: data[3], 6: data[4], 7: data[5], 8: data[6], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: data[11][max(data[11])]},
                2: {1: "Micro True Positive/Negative", 2: data[0], 3: data[7], 4: data[8], 5: data[9], 6: data[10], 7: data[12], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: data[0], 3: data[16], 4: data[22], 21: data[16], 5: data[15], 6: data[13], 7: data[17], 8: data[14], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            if len(data) == 0:
                data_use = {0: {}, 1: {1: "Training and Validation Accuracy", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: 1},
                2: {1: "Micro True Positive/Negative", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: [0,1], 3: [0,1], 4: [0,1], 21: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            if len(data) != 0:
                plt.figure(figsize = (6,5))
                plt.title(data_use[self.statsPage][10], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][7], label=data_use[self.statsPage][18])
                if self.statsPage != 2:
                    plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][8], label=data_use[self.statsPage][19])
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][20], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()

            if len(data) == 0:
                plt.figure(figsize = (6,5))
                plt.title(data_use[self.statsPage][10], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][7], label=data_use[self.statsPage][18], color = 'white')
                if self.statsPage != 2:
                    plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][8], label=data_use[self.statsPage][19], color = 'white')
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][20], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()
            return None

    def pop2(self, event, key = False):
        if self.popTwo_tag or key:
            data = self.dataTrain
            if len(data) != 0:
                data_use = {0: data[11], 1: {1: "Training and Validation Accuracy", 2: data[0], 3: data[1], 4:data[2], 5: data[3], 6: data[4], 7: data[5], 8: data[6], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: data[11][max(data[11])]},
                2: {1: "Micro True Positive/Negative", 2: data[0], 3: data[7], 4: data[8], 5: data[9], 6: data[10], 7: data[12], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: data[0], 3: data[16], 4: data[22], 21: data[16], 5: data[15], 6: data[13], 7: data[17], 8: data[14], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            if len(data) == 0:
                data_use = {0: {}, 1: {1: "Training and Validation Accuracy", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: 1},
                2: {1: "Micro True Positive/Negative", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: [0,1], 3: [0,1], 4: [0,1], 21: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}


            plt.close()

            if len(data) != 0:
                fig = plt.figure(figsize=(6, 5))
                plt.title(data_use[self.statsPage][9], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][5], label=data_use[self.statsPage][15])
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][6], label=data_use[self.statsPage][16])
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][17], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()

            if len(data) == 0:
                fig = plt.figure(figsize=(6, 5))
                plt.title(data_use[self.statsPage][9], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][5], label=data_use[self.statsPage][15], color = 'white')
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][6], label=data_use[self.statsPage][16], color = 'white')
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][17], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()
            return None

    def pop1(self, event, key = False):
        if self.popOne_tag or key:
            plt.close()
            data = self.dataTrain
            if len(data) != 0:
                data_use = {0: data[11], 1: {1: "Training and Validation Accuracy", 2: data[0], 3: data[1], 4:data[2], 5: data[3], 6: data[4], 7: data[5], 8: data[6], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: data[11][max(data[11])]},
                2: {1: "Micro True Positive/Negative", 2: data[0], 3: data[7], 4: data[8], 5: data[9], 6: data[10], 7: data[12], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: data[0], 3: data[16], 4: data[22], 21: data[16], 5: data[15], 6: data[13], 7: data[17], 8: data[14], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            if len(data) == 0:
                data_use = {0: {}, 1: {1: "Training and Validation Accuracy", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 
                9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: 1},
                2: {1: "Micro True Positive/Negative", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
                14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
                3: {1: "F1 Score", 2: [0,1], 3: [0,1], 4: [0,1], 21: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
                14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            if len(data) != 0:
                plt.figure(figsize = (6,5))
                plt.title(data_use[self.statsPage][1], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][3], label=data_use[self.statsPage][11])
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][4], label=data_use[self.statsPage][12])
                if self.statsPage == 1 and type(data[11][max(data[11])]) != dict:
                    plt.plot(data_use[self.statsPage][2], [data_use[self.statsPage][21]] * len(data_use[self.statsPage][2]),linestyle = '--' , label = 'Testing Accuracy')
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][14], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()

            if len(data) == 0:
                plt.figure(figsize = (6,5))
                plt.title(data_use[self.statsPage][1], fontsize=10)
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][3], label=data_use[self.statsPage][11], color = 'white')
                plt.plot(data_use[self.statsPage][2], data_use[self.statsPage][4], label=data_use[self.statsPage][12], color = 'white')
                plt.xlabel(data_use[self.statsPage][13], fontsize=9)
                plt.ylabel(data_use[self.statsPage][14], fontsize=9)
                plt.legend(fontsize=7)
                plt.show()
            return None

    def featureConvLayerNext(self):
        if self.featureMapConv < len(self.backEnd.convLayers):
            self.featureMapConv += 1
            self.featureMapStats.set("Feature Map on Convolutional Layer " + str(self.featureMapConv))
            self.featureMapConvPage.place(x = 480, y = 430, anchor = "center")
            self.featureNumChanVar.set(str(self.backEnd.convLayers[self.featureMapConv][1]) + " Channels")
            self.featureNumChannel.place(x = 480, y = 450, anchor = "center")
            self.update_gui(self.dataTrain)

    def featureConvLayerBack(self):
        if self.featureMapConv > 1:
            self.featureMapConv -= 1
            self.featureMapStats.set("Feature Map on Convolutional Layer " + str(self.featureMapConv))
            self.featureMapConvPage.place(x = 480, y = 430, anchor = "center")
            self.featureNumChanVar.set(str(self.backEnd.convLayers[self.featureMapConv][1]) + " Channels")
            self.featureNumChannel.place(x = 480, y = 450, anchor = "center")
            self.update_gui(self.dataTrain)

    def featureNumNext(self):
        if self.dataTrain != {}:
            if self.featureMapCurrent < round(len(self.dataTrain[18][self.dataTrain[19][self.featureMapConv]][0]) / 6) - 1:
                self.featureMapCurrent += 1
                self.update_gui(self.dataTrain)

    def featureNumBack(self):
        if self.featureMapCurrent > 0:        
            self.featureMapCurrent -= 1
            self.update_gui(self.dataTrain)

    def statsNext(self):
        l = {1: "Accuracies, Losses, and Errors", 2: "True Postive/Negative, False Positive/Negative, and Time", 3: "Precision, Recall, F1 Score"}
        if self.statsPage < 3:
            self.statsPage += 1
            self.statsReportVar.set("Currently on Performance Page: " + str(self.statsPage) + "/3")
            self.statsReport.place(x = 680, y = 100)
            self.statsPageVar.set(l[self.statsPage])
            self.statsPageLabel.place(x = 480, y = 110, anchor = "center")
            self.update_gui(self.dataTrain)

    def statsBack(self):
        l = {1: "Accuracies, Losses, and Errors", 2: "True Postive/Negative, False Positive/Negative, and Time", 3: "Precision, Recall, F1 Score"}
        if self.statsPage > 1:
            self.statsPage -= 1
            self.statsReportVar.set("Currently on Performance Page: " + str(self.statsPage) + "/3 ")
            self.statsReport.place(x = 680, y = 100)
            self.statsPageVar.set(l[self.statsPage])
            self.statsPageLabel.place(x = 480, y = 110, anchor = "center")
            self.update_gui(self.dataTrain)

    def update_scroll_position(self, event):
        self.current_position = self.text_widget.yview()

    def infrastructureVisualization(self, shapes = None, shapes2 = None, Train = False, report = None, theme_ = False):
        if not Train:
            self.text_widget.place(x=965, y=140, width=350, height=480)  # Adjust x, y, width, and height as needed
        if Train:
            if not self.backEnd.doingRegression:
                self.text_widget.place(x = 965, y = 140, width = 350, height = 270)
            if self.backEnd.doingRegression:
                self.text_widget.place(x=965, y=140, width=350, height=480)
        
        self.text_widget.configure(state = tk.NORMAL, bg = self.theme[theme_])

        
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.configure(state = tk.DISABLED)

        self.text_widget.bind("<MouseWheel>", self.update_scroll_position)

        self.text_widget.tag_configure("dark_bold", font=("Mulish", 10, "bold"), foreground = "white")
        self.text_widget.tag_configure("light_bold", font = ("Mulish", 10, "bold"), foreground = "black")
        self.text_widget.tag_configure("dark_", foreground = "#D8DEE9")
        self.text_widget.tag_configure("light_", foreground = "black")
        self.text_widget.tag_configure("chose_light", foreground = "red")
        self.text_widget.tag_configure("choose_dark", foreground = "#F9914A")
        
        theme = {True: "dark_", False: "light_"}
        theme_bold = {True: "dark_bold", False: "light_bold"}
        choosen = {True: "choose_dark", False: "chose_light"}
        
        if self.step1:
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, "Image Properties and Step One Information", theme_bold[theme_])

            #self.text_widget.insert(tk.END" \nImage )
            if len(self.backEnd.imDim) > 0:
                self.text_widget.insert(tk.END, "\nImage Dimension: " + str(self.backEnd.imDim), theme[theme_])
            else:
                self.text_widget.insert(tk.END, "\nImage Dimension: [??,??,??]", theme[theme_])
            self.text_widget.insert(tk.END, "\nTotal Data Points: " + str(len(self.backEnd.dataset)), theme[theme_])
            self.text_widget.insert(tk.END, "\nTotal Number of Labels: " + str(len(self.backEnd.dataPerLabel)), theme[theme_])
            if not self.backEnd.preTrainedModel:
                self.text_widget.insert(tk.END, "\nLoaded Pre-Trained Model: False", theme[theme_])
            else:
                self.text_widget.insert(tk.END, "\nLoaded Pre-Trained Model: True", theme[theme_])
            self.text_widget.insert(tk.END, "\n\nData Quality Metrics", theme_bold[theme_])
            self.text_widget.insert(tk.END, "\nAverage Data per Label: " + str(self.backEnd.mean), theme[theme_])
            self.text_widget.insert(tk.END, "\nSmallest over largest label: " + str(self.backEnd.dataRatio), theme[theme_])
            self.text_widget.insert(tk.END, "\nGini Coefficient: " + str(self.backEnd.gini_), theme[theme_])
            self.text_widget.configure(state = tk.DISABLED)

        if self.step2:
            self.text_widget.configure(state = tk.NORMAL)
            self.text_widget.insert(tk.END, "\n\nData Split", theme_bold[theme_])
            self.text_widget.insert(tk.END, "\nTraining: " + str(self.round_to_two_sigfigs_bigger_9999(self.backEnd.trainingSplit)), theme[theme_])
            self.text_widget.insert(tk.END, "|Testing: " + str(self.round_to_two_sigfigs_bigger_9999(self.backEnd.testingSplit)) + "|Validation: " + str(self.round_to_two_sigfigs_bigger_9999(self.backEnd.validationSplit)), theme[theme_])
            self.text_widget.insert(tk.END, "\nNumber of Training Data: " + str(self.backEnd.datalist[0]), theme[theme_])
            self.text_widget.insert(tk.END, "\nNumber of Validation Data: " + str(self.backEnd.datalist[1]), theme[theme_])
            self.text_widget.insert(tk.END, "\nNumber of Testing Data: " + str(self.backEnd.datalist[2]), theme[theme_])
            self.text_widget.insert(tk.END, "\n\nModel Information", theme_bold[theme_])
            self.text_widget.insert(tk.END, "\nBatchSize: " + str(self.round_to_two_sigfigs_bigger_9999(self.backEnd.batchSize)) + "\n" + "Learning Rate: " + str(self.round_to_two_sigfigs(self.backEnd.learningRate)), theme[theme_])
            self.text_widget.insert(tk.END, "\nNumber of Epochs: " + str(self.round_to_two_sigfigs_bigger_9999(self.backEnd.epochs)), theme[theme_])
            if self.addNormalize == True: 
                norm = "Yes" 
            else: 
                norm = "No"
            self.text_widget.insert(tk.END, "\nNormalize: " + str(norm), theme[theme_])
            self.text_widget.insert(tk.END, "\nOptimizer: " + str(self.optimizers), theme[theme_])
            self.text_widget.insert(tk.END, "\nWeight Decay: " + str(round(self.backEnd.weightdecay,4)), theme[theme_])
            self.text_widget.insert(tk.END, "\nEarly Stopping: " + str(self.backEnd.earlyStopping), theme[theme_])
            self.text_widget.insert(tk.END, "\nCriterion: " + str(self.backEnd.critIdentifier), theme[theme_])
            self.text_widget.configure(state = tk.DISABLED)

        if len(self.backEnd.convLayers) >= 1:
            self.text_widget.configure(state = tk.NORMAL)
            self.text_widget.insert(tk.END, "\n")
            self.text_widget.insert(tk.END, "\nConvolutional Layer(s) Properties", theme_bold[theme_])

            for k,v in self.backEnd.convLayers.items():
                if k == self.backEnd.convLayerSteps:
                    if shapes is not None:
                        self.text_widget.insert(tk.END, "\nConvolutional Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nImage before layer: " + str(shapes[k][0]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.convLayers[k][0]) + "| Out Channels: " + str(self.backEnd.convLayers[k][1]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.convLayers[k][14]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.convLayers[k][15]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nConvolutional Kernel Properties", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nSize: " + "[" +str(self.backEnd.convLayers[k][2]) + "," + str(self.backEnd.convLayers[k][3]) + "]" +"| Stride: " + "[" +str(self.backEnd.convLayers[k][4]) + "," + str(self.backEnd.convLayers[k][5]) + "]", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nOutcome: " + str(shapes[k][1]), choosen[theme_])
                        if self.backEnd.convLayers[k][7]:
                            self.text_widget.insert(tk.END, "\nPadding Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Size: " + "[" + str(self.backEnd.convLayers[k][-2]) + "," + str(self.backEnd.convLayers[k][10]) + "," + str(self.backEnd.convLayers[k][-1]) + "," + str(self.backEnd.convLayers[k][11]) + "]", choosen[theme_]) # left -2, top -1
                            self.text_widget.insert(tk.END, "\nPadding Outcome: " + str(shapes[k][2]), choosen[theme_])
                        if self.backEnd.convLayers[k][6]:
                            self.text_widget.insert(tk.END, "\nPooling Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Size: " + "[" + str(self.backEnd.convLayers[k][8]) + "," + str(self.backEnd.convLayers[k][9]) + "]", choosen[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Stride: " + "[" + str(self.backEnd.convLayers[k][12]) + "," + str(self.backEnd.convLayers[k][13]) + "]", choosen[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Outcome: " + str(shapes[k][3]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nFinal Outcome: " + str(shapes[k][3]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\n")
                    else:
                        self.text_widget.insert(tk.END, "\nConvolutional Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nImage before layer: [??,??,??]", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.convLayers[k][0]) + "| Out Channels: " + str(self.backEnd.convLayers[k][1]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.convLayers[k][14]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.convLayers[k][15]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nConvolutional Kernel Properties", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nSize: " + "[" +str(self.backEnd.convLayers[k][2]) + "," + str(self.backEnd.convLayers[k][3]) + "]" +"| Stride: " + "[" +str(self.backEnd.convLayers[k][4]) + "," + str(self.backEnd.convLayers[k][5]) + "]", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nOutcome: [??,??,??]", choosen[theme_])
                        if self.backEnd.convLayers[k][7]:
                            self.text_widget.insert(tk.END, "\nPadding Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Size: " + "[" + str(self.backEnd.convLayers[k][-2]) + "," + str(self.backEnd.convLayers[k][10]) + "," + str(self.backEnd.convLayers[k][-1]) + "," + str(self.backEnd.convLayers[k][11]) + "]", choosen[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Outcome: [??,??,??]", choosen[theme_])
                        if self.backEnd.convLayers[k][6]:
                            self.text_widget.insert(tk.END, "\nPooling Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Size: " + "[" + str(self.backEnd.convLayers[k][8]) + "," + str(self.backEnd.convLayers[k][9]) + "]", choosen[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Stride: " + "[" + str(self.backEnd.convLayers[k][12]) + "," + str(self.backEnd.convLayers[k][13]) + "]", choosen[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Outcome: [??,??,??]", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nFinal Outcome: [??,??,??]", choosen[theme_])
                        self.text_widget.insert(tk.END, "\n")
                else:
                    if shapes is not None:
                        self.text_widget.insert(tk.END, "\nConvolutional Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nImage before layer: " + str(shapes[k][0]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.convLayers[k][0]) + "| Out Channels: " + str(self.backEnd.convLayers[k][1]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.convLayers[k][14]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.convLayers[k][15]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nConvolutional Kernel Properties", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nSize: " + "[" +str(self.backEnd.convLayers[k][2]) + "," + str(self.backEnd.convLayers[k][3]) + "]" +"| Stride: " + "[" +str(self.backEnd.convLayers[k][4]) + "," + str(self.backEnd.convLayers[k][5]) + "]", theme[theme_])
                        self.text_widget.insert(tk.END, "\nOutcome: " + str(shapes[k][1]), theme[theme_])
                        if self.backEnd.convLayers[k][7]:
                            self.text_widget.insert(tk.END, "\nPadding Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Size: " + "[" + str(self.backEnd.convLayers[k][-2]) + "," + str(self.backEnd.convLayers[k][10]) + "," + str(self.backEnd.convLayers[k][-1]) + "," + str(self.backEnd.convLayers[k][11]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Outcome: " + str(shapes[k][2]), theme[theme_])
                        if self.backEnd.convLayers[k][6]:
                            self.text_widget.insert(tk.END, "\nPooling Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Size: " + "[" + str(self.backEnd.convLayers[k][8]) + "," + str(self.backEnd.convLayers[k][9]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Stride: " + "[" + str(self.backEnd.convLayers[k][12]) + "," + str(self.backEnd.convLayers[k][13]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Outcome: " + str(shapes[k][3]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nFinal Outcome: " + str(shapes[k][3]), theme[theme_])
                        self.text_widget.insert(tk.END, "\n")
                    else:
                        self.text_widget.insert(tk.END, "\nConvolutional Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nImage before layer: [??,??,??]", theme[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.convLayers[k][0]) + "| Out Channels: " + str(self.backEnd.convLayers[k][1]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.convLayers[k][14]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.convLayers[k][15]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nConvolutional Kernel Properties", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nSize: " + "[" +str(self.backEnd.convLayers[k][2]) + "," + str(self.backEnd.convLayers[k][3]) + "]" +"| Stride: " + "[" +str(self.backEnd.convLayers[k][4]) + "," + str(self.backEnd.convLayers[k][5]) + "]", theme[theme_])
                        self.text_widget.insert(tk.END, "\nOutcome: [??,??,??]", theme[theme_])
                        if self.backEnd.convLayers[k][7]:
                            self.text_widget.insert(tk.END, "\nPadding Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Size: " + "[" + str(self.backEnd.convLayers[k][-2]) + "," + str(self.backEnd.convLayers[k][10]) + "," + str(self.backEnd.convLayers[k][-1]) + "," + str(self.backEnd.convLayers[k][11]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPadding Outcome: [??,??,??]", theme[theme_])
                        if self.backEnd.convLayers[k][6]:
                            self.text_widget.insert(tk.END, "\nPooling Properties", theme_bold[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Size: " + "[" + str(self.backEnd.convLayers[k][8]) + "," + str(self.backEnd.convLayers[k][9]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Stride: " + "[" + str(self.backEnd.convLayers[k][12]) + "," + str(self.backEnd.convLayers[k][13]) + "]", theme[theme_])
                            self.text_widget.insert(tk.END, "\nPooling Outcome: [??,??,??]", theme[theme_])
                        self.text_widget.insert(tk.END, "\nFinal Outcome: [??,??,??]", theme[theme_])
                        self.text_widget.insert(tk.END, "\n")


            self.text_widget.configure(state = tk.DISABLED)

        if len(self.backEnd.fullLayers) >= 1:
            self.text_widget.configure(state = tk.NORMAL)
            self.text_widget.insert(tk.END, "\n\nFully Connected Layer(s) Properties", theme_bold[theme_])

            for k,v in self.backEnd.fullLayers.items():

                if k == self.backEnd.fullLayerSteps:
                    if shapes2 is not None:
                        self.text_widget.insert(tk.END, "\nFully Connected Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nInput Size: " + str(shapes2[k][0]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.fullLayers[k][0]), choosen[theme_])
                        self.text_widget.insert(tk.END, "| Out Channels: " + str(self.backEnd.fullLayers[k][1]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.fullLayers[k][2]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.fullLayers[k][3]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nOutput Size: " + str(shapes2[k][1]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\n")
                    else:
                        self.text_widget.insert(tk.END, "\nFully Connected Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nInput Size: ???", choosen[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.fullLayers[k][0]), choosen[theme_])
                        self.text_widget.insert(tk.END, "| Out Channels: " + str(self.backEnd.fullLayers[k][1]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.fullLayers[k][2]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.fullLayers[k][3]), choosen[theme_])
                        self.text_widget.insert(tk.END, "\nOutput Size: ???", choosen[theme_])
                        self.text_widget.insert(tk.END, "\n")
                else:
                    if shapes2 is not None:
                        self.text_widget.insert(tk.END, "\nFully Connected Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nInput Size: " + str(shapes2[k][0]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.fullLayers[k][0]), theme[theme_])
                        self.text_widget.insert(tk.END, "| Out Channels: " + str(self.backEnd.fullLayers[k][1]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.fullLayers[k][2]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.fullLayers[k][3]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nOutput Size: " + str(shapes2[k][1]), theme[theme_])
                        self.text_widget.insert(tk.END, "\n")
                    else:
                        self.text_widget.insert(tk.END, "\nFully Connected Layer " + str(k) + ":", theme_bold[theme_])
                        self.text_widget.insert(tk.END, "\nInput Size: ???", theme[theme_])
                        self.text_widget.insert(tk.END, "\nIn Channels: " + str(self.backEnd.fullLayers[k][0]), theme[theme_])
                        self.text_widget.insert(tk.END, "| Out Channels: " + str(self.backEnd.fullLayers[k][1]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nDropout Rate: " + str(self.backEnd.fullLayers[k][2]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nActivation Function: " + str(self.backEnd.fullLayers[k][3]), theme[theme_])
                        self.text_widget.insert(tk.END, "\nOutput Size: ???", theme[theme_])
                        self.text_widget.insert(tk.END, "\n")

            self.text_widget.configure(state = tk.DISABLED)

        if Train:
            self.text_widget.configure(state = tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, "Report For Each Epoch", theme_bold[theme_])
            if self.doneTraining:
                self.text_widget.insert(tk.END, "\n\nTraining Completed", theme_bold[theme_])
                self.text_widget.insert(tk.END, "\nTesting Accuracy: " + str(round(report[max(report)],3)) + "%", theme[theme_])
            if self.ES:
                self.text_widget.insert(tk.END, "\n\nEarly Stopping at Epoch " + str(max(report) - 1), theme[theme_])
            for k,v in reversed(report.items()):
                
                if type(v) == dict:
                    self.text_widget.insert(tk.END, "\n\nEpoch " + str(k), theme_bold[theme_])
                    self.text_widget.insert(tk.END, "\nTraining Statistics:", theme_bold[theme_])
                    self.text_widget.insert(tk.END, "\nLoss/Accuracy: " + str(self.round_to_two_sigfigs(report[k][1])) + "/" + str(round(report[k][2],3)) + "%", theme[theme_])
                    self.text_widget.insert(tk.END, "\nMSE/MAE: " + str(round(report[k][5],3)) +"/" +str(round(report[k][6],3)), theme[theme_])
                    self.text_widget.insert(tk.END, "\nValidation Statistics:", theme_bold[theme_])
                    self.text_widget.insert(tk.END, "\nLoss/Accuracy: " + str(self.round_to_two_sigfigs(report[k][4])) + "/" + str(round(report[k][3],3)) + "%", theme[theme_])
                    self.text_widget.insert(tk.END, "\nGeneral Statistics:", theme_bold[theme_])
                    self.text_widget.insert(tk.END, "\nµTrue Pos/Neg: " + str(self.round_to_two_sigfigs(report[k][9])) +"/" + str(self.round_to_two_sigfigs(report[k][10])), theme[theme_])
                    self.text_widget.insert(tk.END, "\nµFalse Pos/Neg: " + str(self.round_to_two_sigfigs(report[k][11])) +"/" + str(self.round_to_two_sigfigs(report[k][12])), theme[theme_])
                    self.text_widget.insert(tk.END, "\nµ/macro Precision: " + str(self.round_to_two_sigfigs(report[k][14])) +"(s)/" + str(self.round_to_two_sigfigs(report[k][18])), theme[theme_])
                    self.text_widget.insert(tk.END, "\nµ/macro Recall: " + str(self.round_to_two_sigfigs(report[k][15])) + "/" + str(self.round_to_two_sigfigs(report[k][16])), theme[theme_])
                    self.text_widget.insert(tk.END, "\nµ/macro F1: " + str(self.round_to_two_sigfigs(report[k][17])) + "/" + str(self.round_to_two_sigfigs(report[k][7])), theme[theme_])
                    self.text_widget.insert(tk.END, "\nTraining Time: " + str(round(report[k][13], 3)) + "s", theme[theme_])

            self.text_widget.configure(state = tk.DISABLED)
        self.text_widget.config(yscrollcommand=self.scrollbar.set)
        if not Train:
            self.text_widget.yview_moveto(self.current_position[0])


    
    def check_Train(self):
        if self.threadcheck_fit is not None:
            if self.threadcheck_fit.is_alive():
                return None

        if len(self.backEnd.convLayers) == 0 and len(self.backEnd.fullLayers) == 0:
            self.notification_textTrain.set("Model is incomplete or incorrect")
            self.notificationTrain.place(x = 960, y = 635)
            return None

        if self.backEnd.fullShapes == None or self.backEnd.convShapes == None:
            self.notification_textTrain.set("Model is incomplete or incorrect")
            self.notificationTrain.place(x = 960, y = 635)
            return None

        #Checking Output Neurons
        if int(self.backEnd.fullShapes[len(self.backEnd.fullShapes)][1]) == len(self.backEnd.dataPerLabel) or int(self.backEnd.fullShapes[len(self.backEnd.fullShapes)][1]) == 1:
            
            #Checking if regression training but data is not compatible
            if int(self.backEnd.fullShapes[len(self.backEnd.fullShapes)][1]) == 1 and not self.backEnd.regression_possibility:
                self.notification_textTrain.set("This dataset is not compatible with regression")
                self.notificationTrain.place(x = 960, y = 635)
                return None

            #Checking all cases for Regression (output neurons == 1)
            if int(self.backEnd.fullShapes[len(self.backEnd.fullShapes)][1]) == 1:

                #Check if Criterion is L1 or Smooth L1 or MSE. If not return nothing
                if self.backEnd.critIdentifier == "L1" or self.backEnd.critIdentifier == "Smooth L1" or self.backEnd.critIdentifier == "MSE":
                    pass
                else:
                    self.notification_textTrain.set("Criterion not suitable for regression")
                    self.notificationTrain.place(x = 960, y = 635)
                    return None

                #Checking if output activation function is None
                if self.backEnd.fullLayers[len(self.backEnd.fullLayers)][3] != 'None':
                    self.notification_textTrain.set("Final Activation must be None for regression")
                    self.notificationTrain.place(x = 960, y = 635)
                    return None

            self.backEnd.doingRegression = True

            if int(self.backEnd.fullShapes[len(self.backEnd.fullShapes)][1]) == len(self.backEnd.dataPerLabel):
                self.backEnd.doingRegression = False

        else:
            self.notification_textTrain.set("Number of Output Neurons incorrect")
            self.notificationTrain.place(x = 960, y = 635)
            return None

        self.backEnd.check_gpu()
        
        self.threadcheck_fit = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.check_Train_queue, loadPreTrained = True))
        self.threadcheck_fit.daemon = True
        self.threadcheck_fit.start()
        if self.backEnd.preTrainedModel is not None:
            self.notification_textTrain.set("Checking Pre-trained Model's compatibility")
        else:
            self.notification_textTrain.set("Last Check before Training")
        self.notificationTrain.place(x = 960, y = 635)
        self.checkTrainLoad()

    def checkTrainLoad(self):
        if self.check_Train_queue.qsize() >= 1:
            data_get = self.check_Train_queue.get()
            if data_get:
                self.Train()
                self.notification_inTrain.place_forget()
                return None
            if not data_get:
                self.notification_textTrain.set("Architecture does not fit with Pre-trained Model")
                self.notificationTrain.place(x = 960, y = 635)
                return None
        self.master.after(10, self.checkTrainLoad)


    def Train(self):
        for i in self.pageOneList:
            i.pack()
            i.pack_forget()

        self.inPageOne = False

        self.notificationTrain.place_forget()
        self.instructionButton.pack()
        self.instructionButton.pack_forget()
        self.userGuideLbl.pack()
        self.userGuideLbl.pack_forget()
        self.master.update()  # Force GUI to update immediately

        self.downloadstatsLbl.place(x = 1138, y = 110)
        self.downloadStatsButton.place(x = 1260, y = 108)
        self.modelPerf.place(x = 20, y = 100)
        self.perfReport.place(x = 965, y = 110)
        self.featureMaps.place(x = 20, y = 395)
        self.statsPageVar.set("Accuracies, Losses, and Errors")
        self.statsPageLabel.place(x = 480, y = 130, anchor = "center")
        self.trainForwardButton.place(x = 915, y = 97)
        self.trainBackwardButton.place(x = 885, y = 97)
        self.statsReportVar.set("Currently on Performance Page: " + str(self.statsPage) + "/3")
        self.statsReport.place(x = 680, y = 100)
        self.featureNumVar.set("Change Channels ")
        self.featureNumReport.place(x = 765, y = 390)
        self.featureNumForwardButton.place(x = 915, y = 387)
        self.featureNumBackwardButton.place(x = 885, y = 387)
        self.featureConvLayerVar.set("Change Conv Layer")
        self.featureConvLayerReport.place(x = 765, y = 420)
        self.featureConvLayerForwardButton.place(x = 915, y = 417)
        self.featureConvLayerBackwardButton.place(x = 885, y = 417)
        self.featureMapStats.set("Feature Map on Convolutional Layer " + str(self.featureMapConv))
        self.featureMapConvPage.place(x = 480, y = 430, anchor = "center")
        if len(self.backEnd.convLayers) >= 1:
            self.featureNumChanVar.set(str(self.backEnd.convLayers[self.featureMapConv][1]) + " Channels")
        self.featureNumChannel.place(x = 480, y = 450, anchor = "center")
        self.popGraphOne.place(x= 108, y = 355)
        self.popGraphTwo.place(x= 418, y = 355)
        self.popGraphThree.place(x = 728, y = 355)
        self.popGraph1.place(x= 35, y = 357)
        self.popGraph2.place(x= 345, y = 357)
        self.popGraph3.place(x = 655, y = 357)

        self.popGraph4.place(x = 35, y = 662)
        self.popGraphFour.place(x = 146, y = 660)

        if not self.backEnd.doingRegression:
            self.popConMat.place(x = 960, y = 662)
            self.popCM.place(x = 1040, y = 660)

        self.traindifferent.place(x = 605, y = 710)
        self.trainAnotherText.place(x = 480, y = 713)

        self.retrain.place(x = 755, y = 710)
        self.reTrainModelText.place(x = 665, y = 713)

        self.downloadCode.place(x = 915, y = 710)
        self.downloadCodeText.place(x = 810, y = 713)

        self.downloadModel.place(x = 1260, y = 710)
        self.downloadModelText.place(x = 1150, y = 713)

        self.saveStateLbl.place(x = 975, y = 713)
        self.saveStateButton.place(x = 1090, y = 710)

        self.update_gui({})
        self.training_thread = threading.Thread(target=lambda: self.backEnd.trainingLoop(self.queue, self.stop_training_event))
        self.training_thread.daemon = True
        self.training_thread.start()
        self.infrastructureVisualization(Train = True, report = {}, theme_ = self.themeBool)
        self.get_queue()

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

    def update_gui(self, data):
        if data == {}:
            data_use = {0: {}, 1: {1: "Training and Validation Accuracy", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 
            9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: 1},
            2: {1: "Micro True Positive/Negative", 2: [0,1], 3: [0,1], 4: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
            14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
            3: {1: "F1 Score", 2: [0,1], 3: [0,1], 4: [0,1], 21: [0,1], 5: [0,1], 6: [0,1], 7: [0,1], 8: [0,1], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
            14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}

            self.chart = FigureCanvasTkAgg(self.fig, master=self.master)
            self.chart_widget = self.chart.get_tk_widget()
            self.chart2 = FigureCanvasTkAgg(self.fig2, master=self.master)
            self.chart2_widget = self.chart2.get_tk_widget()
            self.chart3 = FigureCanvasTkAgg(self.fig3, master=self.master)
            self.chart3_widget = self.chart3.get_tk_widget()
            self.chart4 = FigureCanvasTkAgg(self.fig4, master = self.master)
            self.chart4_widget = self.chart4.get_tk_widget()
            self.chart_heatmap = FigureCanvasTkAgg(self.fig_heatmap, master=self.master)
            self.chart_heatmap_widget = self.chart_heatmap.get_tk_widget()

            self.fig.suptitle(data_use[self.statsPage][1], fontsize=10, color=self.theme_bold[self.themeBool])
            self.ax.plot(data_use[self.statsPage][2], data_use[self.statsPage][3], label=data_use[self.statsPage][11], color = "white")
            self.ax.plot(data_use[self.statsPage][2], data_use[self.statsPage][4], label=data_use[self.statsPage][12], color = "white")
            self.ax.set_xlabel(data_use[self.statsPage][13], fontsize=9)
            self.ax.set_ylabel(data_use[self.statsPage][14], fontsize=9)
            for tick in self.ax.get_xticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            for tick in self.ax.get_yticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            self.ax.xaxis.label.set_color(self.theme_bold[self.themeBool])
            self.ax.yaxis.label.set_color(self.theme_bold[self.themeBool])

            self.fig2.suptitle(data_use[self.statsPage][9], fontsize=10, color=self.theme_bold[self.themeBool])
            self.ax2.plot(data_use[self.statsPage][2], data_use[self.statsPage][5], label=data_use[self.statsPage][15], color = "white")
            self.ax2.plot(data_use[self.statsPage][2], data_use[self.statsPage][6], label=data_use[self.statsPage][16], color = "white")
            self.ax2.set_xlabel(data_use[self.statsPage][13], fontsize=9)
            self.ax2.set_ylabel(data_use[self.statsPage][17], fontsize=9)
            for tick in self.ax2.get_xticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            for tick in self.ax2.get_yticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            self.ax2.xaxis.label.set_color(self.theme_bold[self.themeBool])
            self.ax2.yaxis.label.set_color(self.theme_bold[self.themeBool])

            self.fig3.suptitle(data_use[self.statsPage][10], fontsize=10, color=self.theme_bold[self.themeBool])
            self.ax3.plot(data_use[self.statsPage][2], data_use[self.statsPage][7], label=data_use[self.statsPage][18], color = "white")
            if self.statsPage != 2:
                self.ax3.plot(data_use[self.statsPage][2], data_use[self.statsPage][8], label=data_use[self.statsPage][19], color = "white")
            self.ax3.set_xlabel(data_use[self.statsPage][13], fontsize=9)
            self.ax3.set_ylabel(data_use[self.statsPage][20], fontsize=9)
            for tick in self.ax3.get_xticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            for tick in self.ax3.get_yticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            self.ax3.xaxis.label.set_color(self.theme_bold[self.themeBool])
            self.ax3.yaxis.label.set_color(self.theme_bold[self.themeBool])

            if not self.backEnd.doingRegression:
                self.ax_heatmap.clear()
                self.fig_heatmap.suptitle("Confusion Matrix", color = self.theme_bold[self.themeBool], fontsize=10)
                sns.heatmap(len(self.backEnd.dataPerLabel) * [[0] * len(self.backEnd.dataPerLabel)], linewidths=0.1, cmap='Greens', linecolor='gray', fmt='.1f', annot=False, ax=self.ax_heatmap, cbar=False)
                self.ax_heatmap.set_xlabel("Predicted", fontsize=9)
                self.ax_heatmap.set_ylabel("Actual", fontsize=9)
                self.fig_heatmap.set_facecolor(self.theme[self.themeBool])
                for tick in self.ax_heatmap.get_xticklabels():
                    tick.set_color(self.theme_bold[self.themeBool])
                for tick in self.ax_heatmap.get_yticklabels():
                    tick.set_color(self.theme_bold[self.themeBool])
                self.ax_heatmap.yaxis.label.set_color(self.theme_bold[self.themeBool])
                self.ax_heatmap.xaxis.label.set_color(self.theme_bold[self.themeBool])
                self.chart_heatmap.draw()
                self.chart_heatmap_widget.place(x=1115, y=557, anchor = 'center')
                self.fig_heatmap.tight_layout()

            for ax in self.ax4:
                ax.xaxis.label.set_color(self.theme_bold[self.themeBool])
                ax.yaxis.label.set_color(self.theme_bold[self.themeBool])
                for label in ax.get_xticklabels():
                    label.set_color(self.theme_bold[self.themeBool])
                for label in ax.get_yticklabels():
                    label.set_color(self.theme_bold[self.themeBool])
            self.fig4.tight_layout()
            self.chart4.draw()

            self.chart3_widget.place(x=790, y=255, anchor="center")
            self.chart2_widget.place(x=480, y=255, anchor="center")
            self.chart_widget.place(x=170, y=255, anchor="center")
            self.chart4_widget.place(x=480, y=557, anchor="center")
            self.fig3.tight_layout()
            self.fig4.tight_layout()
            self.fig2.tight_layout()
            self.fig.tight_layout()

            return None #Take this line out once
        
        else:
            data_use = {0: data[11], 1: {1: "Training and Validation Accuracy", 2: data[0], 3: data[1], 4:data[2], 5: data[3], 6: data[4], 7: data[5], 8: data[6], 
            9: "Training and Validation Loss", 10: "MAE and MSE", 11: "Training Accuracy", 12: "Validation Accuracy", 13: "Epochs", 14: "Accuracy (%)", 15: "Training Loss", 16: "Validation Loss", 17: "Loss", 18: "MAE", 19: "MSE", 20: "Error", 21: data[11][max(data[11])]},
            2: {1: "Micro True Positive/Negative", 2: data[0], 3: data[7], 4: data[8], 5: data[9], 6: data[10], 7: data[12], 9:"Micro False Positive/Negative", 10: "Training Time per Epoch", 11: "True Positive", 12: "True Negative", 13:"Epochs",
            14:"# of Data", 15: "False Positive", 16: "False Negative", 17: "# of Data", 18: "Time", 20: "Time (s)"},
            3: {1: "F1 Score", 2: data[0], 3: data[16], 4: data[22], 21: data[16], 5: data[15], 6: data[13], 7: data[17], 8: data[14], 9: "Precision", 10: "Recall", 11: "Micro", 12: "Macro", 22: "F1 Score", 13:"Epochs",
            14: "Ratio", 15: "Macro", 16: "Micro", 17: "Ratio", 18: "Macro", 19: "Micro", 13: "Epochs", 20: "Ratio"}}
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
        if data != {}:
            self.infrastructureVisualization(Train = True, report = data_use[0], theme_ = self.themeBool)
        if len(self.backEnd.convLayers) >= 1:
            if data != {}:
                self.infrastructureVisualization(Train = True, report = data_use[0], theme_ = self.themeBool)
                if self.featureMapCurrent > round(len(data[18][data[19][self.featureMapConv]][0]) / 6) - 1:
                    self.featureMapCurrent = round(len(data[18][data[19][self.featureMapConv]][0]) / 6) - 1
            else:
                self.featureMapCurrent = 0

            for i, x in enumerate(self.ax4.flat):
                feature_map_index = i + 6 * self.featureMapCurrent
                if feature_map_index < len(data[18][data[19][self.featureMapConv]][0]):
                    x.imshow(data[18][data[19][self.featureMapConv]][0, feature_map_index].detach(), cmap='viridis', aspect = 'auto')
                    x.set_title(f'Feature Map {feature_map_index + 1}', fontsize=8, color = self.theme_bold[self.themeBool])
                    x.axis('on')
                else:
                    x.clear()  # Clear the subplot
                    x.axis('off')
            self.fig4.set_facecolor(self.theme[self.themeBool])

        for ax in self.ax4:
            ax.xaxis.label.set_color(self.theme_bold[self.themeBool])
            ax.yaxis.label.set_color(self.theme_bold[self.themeBool])
            for label in ax.get_xticklabels():
                label.set_color(self.theme_bold[self.themeBool])
            for label in ax.get_yticklabels():
                label.set_color(self.theme_bold[self.themeBool])
        self.fig4.tight_layout()
        self.chart4.draw()
        self.chart4_widget.place(x=480, y=557, anchor="center")

        self.ax.clear()
        self.fig.suptitle(data_use[self.statsPage][1], fontsize=10, color=self.theme_bold[self.themeBool])
        self.ax.plot(data_use[self.statsPage][2], data_use[self.statsPage][3], label=data_use[self.statsPage][11])
        self.ax.plot(data_use[self.statsPage][2], data_use[self.statsPage][4], label=data_use[self.statsPage][12])
        if self.statsPage == 1 and type(data[11][max(data[11])]) != dict:
            self.ax.plot(data_use[self.statsPage][2], [data_use[self.statsPage][21]] * len(data_use[self.statsPage][2]),linestyle = '--' , label = 'Testing Accuracy')
        self.ax.set_xlabel(data_use[self.statsPage][13], fontsize=9)
        self.ax.set_ylabel(data_use[self.statsPage][14], fontsize=9)
        self.ax.legend(fontsize=7)
        for tick in self.ax.get_xticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        for tick in self.ax.get_yticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        self.fig.set_facecolor(self.theme[self.themeBool])
        self.ax.xaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax.yaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax.figure.canvas.draw()
        self.chart_widget.place(x=170, y=255, anchor="center")
        self.fig.tight_layout()
        
        self.ax2.clear()
        self.fig2.suptitle(data_use[self.statsPage][9], fontsize=10, color=self.theme_bold[self.themeBool])
        self.ax2.plot(data_use[self.statsPage][2], data_use[self.statsPage][5], label=data_use[self.statsPage][15])
        self.ax2.plot(data_use[self.statsPage][2], data_use[self.statsPage][6], label=data_use[self.statsPage][16])
        self.ax2.set_xlabel(data_use[self.statsPage][13], fontsize=9)
        self.ax2.set_ylabel(data_use[self.statsPage][17], fontsize=9)
        self.ax2.legend(fontsize=7)
        for tick in self.ax2.get_xticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        for tick in self.ax2.get_yticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        self.fig2.set_facecolor(self.theme[self.themeBool])
        self.ax2.xaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax2.yaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax2.figure.canvas.draw()
        self.chart2_widget.place(x=480, y=255, anchor="center")
        self.fig2.tight_layout()

        self.ax3.clear()
        self.fig3.suptitle(data_use[self.statsPage][10], fontsize=10, color=self.theme_bold[self.themeBool])
        self.ax3.plot(data_use[self.statsPage][2], data_use[self.statsPage][7], label=data_use[self.statsPage][18])
        if self.statsPage != 2:
            self.ax3.plot(data_use[self.statsPage][2], data_use[self.statsPage][8], label=data_use[self.statsPage][19])
        self.ax3.set_xlabel(data_use[self.statsPage][13], fontsize=9)
        self.ax3.set_ylabel(data_use[self.statsPage][20], fontsize=9)
        self.ax3.legend(fontsize=7)
        for tick in self.ax3.get_xticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        for tick in self.ax3.get_yticklabels():
            tick.set_color(self.theme_bold[self.themeBool])
        self.fig3.set_facecolor(self.theme[self.themeBool])
        self.ax3.xaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax3.yaxis.label.set_color(self.theme_bold[self.themeBool])
        self.ax3.figure.canvas.draw()
        self.chart3_widget.place(x=790, y=255, anchor="center")
        self.fig3.tight_layout()

        if not self.backEnd.doingRegression:
            self.ax_heatmap.clear()
            self.fig_heatmap.suptitle("Confusion Matrix", color = self.theme_bold[self.themeBool], fontsize=10)
            if len(self.backEnd.dataPerLabel) <= 6:
                sns.heatmap(data[23], linewidths=0.1, cmap='Greens', linecolor='gray', fmt='.1f', annot=True, ax=self.ax_heatmap, cbar=False)
            else:
                sns.heatmap(data[23], linewidths=0.1, cmap='Greens', linecolor='gray', fmt='.1f', annot=False, ax=self.ax_heatmap, cbar=False)
            self.ax_heatmap.set_xlabel("Predicted", fontsize=9)
            self.ax_heatmap.set_ylabel("Actual", fontsize=9)
            self.fig_heatmap.set_facecolor(self.theme[self.themeBool])
            for tick in self.ax_heatmap.get_xticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            for tick in self.ax_heatmap.get_yticklabels():
                tick.set_color(self.theme_bold[self.themeBool])
            self.ax_heatmap.yaxis.label.set_color(self.theme_bold[self.themeBool])
            self.ax_heatmap.xaxis.label.set_color(self.theme_bold[self.themeBool])
            self.chart_heatmap.draw()
            self.chart_heatmap_widget.place(x=1115, y=557, anchor = 'center')
            self.fig_heatmap.tight_layout()

        self.master.update_idletasks()  # Update the GUI

    def round_to_two_sigfigs(self,number):
        return "{:.1e}".format(number)

    def round_to_two_sigfigs_bigger_9999(self,number):
        if number > 9999:
            return "{:.1e}".format(number)
        else:
            return number

    def deleteAllFull(self, event, key = False):
        if self.deleteAllFull_tag or key:
            self.backEnd.fullLayers = {}
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
            if len(self.backEnd.fullLayers) > 0 or len(self.backEnd.convLayers) > 0:
                threadDeleteConv = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.deleteAllFull_queue))
                threadDeleteConv.daemon = True
                threadDeleteConv.start()
                self.deleteAllFullLoad()
            if len(self.backEnd.fullLayers) == 0 and len(self.backEnd.convLayers) == 0:
                self.backEnd.fullLayerSteps = 1
                self.curFullLabel.pack()
                self.curFullLabel.pack_forget()
                self.infrastructureVisualization({},{}, theme_ = self.themeBool)

    def deleteAllFullLoad(self):
        if self.deleteAllFull_queue.qsize() >= 1:
            data_get = self.deleteAllFull_queue.get()
            if data_get == True:
                self.backEnd.fullLayerSteps = 1
                self.curFullLabel.pack()
                self.curFullLabel.pack_forget()
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                return None
        self.master.after(10, self.deleteAllFullLoad)

    def fullBack(self):
        if self.backEnd.fullLayerSteps > 1:
            self.backEnd.fullLayerSteps -= 1
            self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
            self.curFullLabel.place(x = 841, y = 615)
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)

    def fullForward(self):
        if self.backEnd.fullLayerSteps <= len(self.backEnd.fullLayers):
            self.backEnd.fullLayerSteps += 1
            self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
            self.curFullLabel.place(x = 841, y = 615)
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)

    def callActiFuncFullConn(self, event):
        self.actiFuncFull = self.actiVarFullConn.get()

    def convBack(self):
        if self.backEnd.convLayerSteps > 1:
            self.backEnd.convLayerSteps -= 1
            self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
            self.curConvLabel.place(x = 362, y = 615)
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)

    def convForward(self):
        if self.backEnd.convLayerSteps <= len(self.backEnd.convLayers):
            self.backEnd.convLayerSteps += 1
            self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
            self.curConvLabel.place(x = 362, y = 615)
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)

    def deleteFull(self, event):
        if self.deleteFull_tag:
            if self.backEnd.fullLayers != {}:
                if self.backEnd.fullLayerSteps < max(self.backEnd.fullLayers) + 1:
                    for i in range(self.backEnd.fullLayerSteps, max(self.backEnd.fullLayers)):
                        self.backEnd.fullLayers[i] = self.backEnd.fullLayers[i + 1]
                    del self.backEnd.fullLayers[max(self.backEnd.fullLayers)]
                    if len(self.backEnd.fullLayers) > 0 or len(self.backEnd.convLayers) > 0:
                        threadDeleteFull = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.deleteFull_queue))
                        threadDeleteFull.daemon = True
                        threadDeleteFull.start()
                        self.notification_textFour.set("Deleting Fully Connected Layer")
                        self.notificationFour.place(x= 510, y=620)
                        self.deleteFullLoad()
                        return None
                    if len(self.backEnd.fullLayers) == 0 or len(self.backEnd.convLayers) == 0:
                        self.curFullLabel.place_forget()
                        self.infrastructureVisualization({}, {}, theme_ = self.themeBool)
                        self.notificationFour.place_forget()
                        return None
            
    def deleteFullLoad(self):
        if self.deleteFull_queue.qsize() >= 1:
            data_get = self.deleteFull_queue.get()
            if data_get == True:
                self.fullBack()
                if len(self.backEnd.fullLayers) != 0:
                    self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
                    self.curFullLabel.place(x = 841, y = 615)
                else:
                    self.curFullLabel.place_forget()
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                self.notificationFour.place_forget()
                return None
        self.master.after(10, self.deleteFullLoad)

    def deleteConv(self, event):
        if self.deleteConv_tag:
            if self.backEnd.convLayers != {}:
                if self.backEnd.convLayerSteps < max(self.backEnd.convLayers) + 1:
                    for i in range(self.backEnd.convLayerSteps, max(self.backEnd.convLayers)):
                        self.backEnd.convLayers[i] = self.backEnd.convLayers[i + 1]
                    del self.backEnd.convLayers[max(self.backEnd.convLayers)]
                    if len(self.backEnd.fullLayers) > 0 or len(self.backEnd.convLayers) > 0:
                        threadDeleteConv = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.deleteConv_queue))
                        threadDeleteConv.daemon = True
                        threadDeleteConv.start()
                        self.notification_textThree.set("Deleting Convolutional Layer")
                        self.notificationThree.place(x= 30, y=620)
                        self.deleteConvLoad()
                        return None
                    if len(self.backEnd.fullLayers) == 0 or len(self.backEnd.convLayers) == 0:
                        self.curConvLabel.place_forget()
                        self.infrastructureVisualization({}, {}, theme_ = self.themeBool)
                        self.notificationThree.place_forget()
                        return None

    def deleteConvLoad(self):
        if self.deleteConv_queue.qsize() >= 1:
            data_get = self.deleteConv_queue.get()
            if data_get == True:
                self.convBack()
                if len(self.backEnd.convLayers) != 0:
                    self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
                    self.curConvLabel.place(x = 362, y = 615)
                else:
                    self.curConvLabel.place_forget()
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                self.notificationThree.place_forget()
                return None
        self.master.after(10, self.deleteConvLoad)

    def deleteAllConv(self, event, key = False):
        if self.deleteAllConv_tag or key:
            self.backEnd.convLayers = {}
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
            if len(self.backEnd.fullLayers) > 0 or len(self.backEnd.convLayers) > 0:
                threadDeleteConv = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.deleteAllConv_queue))
                threadDeleteConv.daemon = True
                threadDeleteConv.start()
                self.deleteAllConvLoad()
            if len(self.backEnd.fullLayers) == 0 or len(self.backEnd.convLayers) == 0:
                self.backEnd.convLayerSteps = 1
                self.curConvLabel.pack()
                self.curConvLabel.pack_forget()
                self.infrastructureVisualization({},{}, theme_ = self.themeBool)

    def deleteAllConvLoad(self):
        if self.deleteAllConv_queue.qsize() >= 1:
            data_get = self.deleteAllConv_queue.get()
            if data_get == True:
                self.backEnd.convLayerSteps = 1
                self.curConvLabel.pack()
                self.curConvLabel.pack_forget()
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                return None
        self.master.after(10, self.deleteAllConvLoad)

    def callOptimizers(self, event):
        self.optimizers = self.optiVar.get()

    def callActivationFunctions(self, event):
        self.actiFuncConv = self.actiVar.get()

    def stepOneSubmit(self):
        height = self.heightSizeSearch_Bar.get()
        width = self.widthSizeSearch_Bar.get()
        datalink = self.dataPathSearch_Bar.get()
        pretrainedMod = str(self.loadPTBoolButton.get())

        if self.loading_thread is not None:
            if self.loading_thread.is_alive():
                self.notification_textOne.set("Data is loading. Please wait")
                self.notificationOne.place(x=30, y=367)
                return None

        if self.threadTwo is not None:
            if self.threadTwo.is_alive():
                self.notification_textOne.set("Step 2 is loading. Please wait")
                self.notificationOne.place(x=30, y=367)
                return None

        if self.threadThree is not None:
            if self.threadThree.is_alive():
                self.notification_textOne.set("Step 3 is loading. Please wait")
                self.notificationOne.place(x=30, y=367)
                return None

        if self.threadFour is not None:
            if self.threadFour.is_alive():
                self.notification_textOne.set("Step 4 is loading. Please wait")
                self.notificationOne.place(x=30, y=367)
                return None

        if self.step1ResubmitWarning:
            self.notification_textOne.set("Step 1 is completed. Click 'Submit' to resubmit it")
            self.notificationOne.place(x=30, y=367)
            self.step1ResubmitWarning = False
            return None

        if datalink == "" or self.dataType == None:
            self.notification_textOne.set("One or more fields is empty. Please fill them in first")
            self.notificationOne.place(x=30, y=367)
            self.step1 = False
            return None
        self.step1 = False
        self.step2 = False
        self.infrastructureVisualization(theme_ = self.themeBool)

        pretrainedResult = False
        if pretrainedMod != "":
            pretrainedResult = self.backEnd.checkPreTrained(pretrainedMod)

            if not pretrainedResult:
                self.notification_textOne.set("Failed to load Pre-trained Model")
                self.notificationOne.place(x=30, y=367)
                self.step1 = False
                return None

        if pretrainedMod == "":
            self.backEnd.preTrainedModel = None

        if self.preArch == "None":
            if height == "" or width == "":
                self.notification_textOne.set("One or more fields is empty. Please fill them in first")
                self.notificationOne.place(x=30, y=367)
                self.step1 = False
                return None
        else:
            height = self.preHeight[self.preArch]
            width = self.preHeight[self.preArch]
        try:
            height = int(height)
            width = int(width)
            if not isinstance(height, int) or not isinstance(width, int):
                self.notificationOne.place_forget()                
                self.notificationOne.place(x=30, y=367)
        except ValueError:
            self.notification_textOne.set("Height and Width must be an integer")            
            self.notificationOne.place(x=30, y=367)
            self.step1 = False
            return None

        if height < 0 or width < 0:
            self.notification_textOne.set("Height and Width must be a positive integer")
            self.notificationOne.place(x=30, y=367)
            self.step1 = False
            return None

        if self.dataType == ".jpg":
            folders = self.backEnd.get_data(datalink)

            if folders is None or len(folders) == 0:
                self.notification_textOne.set("Data directory not found. Try another link")            
                self.notificationOne.place(x=30, y=367)
                self.step1 = False
                return None

            self.loading_thread = threading.Thread(target=lambda: self.backEnd.load_data(self.loading_queue, height, width))
            self.loading_thread.daemon = True
            self.loading_thread.start()
            self.notification_textOne.set("Loaded 0% of data")            
            self.notificationOne.place(x=30, y=367)
            self.get_training_queue()
        
        if self.dataType == ".bin":
            self.loading_thread = threading.Thread(target = lambda: self.backEnd.load_data_binary(self.loading_queue, datalink, height, width))
            self.loading_thread.daemon = True
            self.loading_thread.start()
            self.notification_textOne.set("Loading Binary Data")
            self.notificationOne.place(x = 30, y = 367)
            self.get_training_queue()

        if self.dataType == ".csv":
            self.loading_thread = threading.Thread(target = lambda: self.backEnd.load_data_csv(self.loading_queue, datalink, height, width))
            self.loading_thread.daemon = True
            self.loading_thread.start()
            self.notification_textOne.set("Loading .csv Data")
            self.notificationOne.place(x = 30, y = 367)
            self.get_training_queue()

        self.step1ResubmitWarning = True
        self.deleteAllFull(event = False, key = True)
        self.deleteAllConv(event = False, key = True)

    def get_training_queue(self):
        if self.loading_queue.qsize() >= 1:
            data_get = self.loading_queue.get()
            self.update_loading(data_get)
            if data_get == "Finished" or data_get == "Fail1" or data_get == "Fail2" or data_get == "ImNotSquare":
                return None
        self.master.after(10, self.get_training_queue)

    def update_loading(self,data):
        if self.dataType == ".jpg":
            try:
                self.notification_textOne.set("Loaded " + str(round(float(data / len(self.backEnd.folders)) * 100,2)) + "% of data")            
                self.notificationOne.place(x=30, y=367)
            except:
                pass

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

        if self.dataType == ".csv":
            if data == "Failed":
                self.notification_textOne.set("Failed to load .csv file")
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


        if data == "Finished":
            self.step1 = True
            self.step1ResubmitWarning = True
            self.backEnd.calculate_quality()
            self.notificationThree.place_forget()
            self.notificationFour.place_forget()
            self.notificationOne.place_forget()
            self.notificationTwo.place_forget()
            if len(self.backEnd.dataset) > 0:
                width, height = self.backEnd.dataset[0][0].size
                self.backEnd.imDim = [len(self.backEnd.dataset[0][0].getbands()), height, width]
            self.infrastructureVisualization(theme_ = self.themeBool)
        return None

    def stepTwoSubmit(self):
        self.notification_textTwo.set("")
        if self.step1 == False:
            self.notification_textTwo.set("Step One has not been completed")
            self.notificationTwo.place(x= 510, y=367)
            return None

        trainingSplit = self.training_Split_Search_Bar.get()
        testingSplit = self.testing_Split_Search_Bar.get()
        validationSplit = self.validation_Split_Search_Bar.get()
        batchSize = self.batchSizeSB.get()
        learningRate = self.lrSB.get()
        epochs = self.numEpochsSB.get()
        weightdecay = self.weightDecayEntry.get()
        earlyStopping = self.earlyStoppingEntry.get()
        criterion = self.criterion


        if trainingSplit == "" or validationSplit == "" or testingSplit == "" or batchSize == "" or learningRate == "" or epochs == "" or self.optimizers == "" or weightdecay == "" or earlyStopping == "" or criterion == None:
            self.notification_textTwo.set("One or more fields is empty. Please fill them in first")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None

        if self.loading_thread is not None:
            if self.loading_thread.is_alive():
                self.notification_textTwo.set("Step 1 is loading. Please wait")
                self.notificationTwo.place(x= 510, y=367)
                return None

        if self.threadTwo is not None:
            if self.threadTwo.is_alive():
                self.notification_textTwo.set("Model is loading. Please Wait")
                self.notificationTwo.place(x= 510, y=367)
                return None

        if self.threadThree is not None:
            if self.threadThree.is_alive():
                self.notification_textTwo.set("Step 3 is loading. Please wait")
                self.notificationTwo.place(x= 510, y=367)
                return None

        if self.threadFour is not None:
            if self.threadFour.is_alive():
                self.notification_textTwo.set("Step 4 is loading. Please wait")
                self.notificationTwo.place(x= 510, y=367)
                return None

        try:
            trainingSplit = int(trainingSplit)
            validationSplit = int(validationSplit)
            testingSplit = int(testingSplit)
            batchSize = int(batchSize)
            learningRate = float(learningRate)
            epochs = int(epochs)
            weightdecay = float(weightdecay)
            earlyStopping = int(earlyStopping)

        except:
            self.notification_textTwo.set("The entered fields must be a positive integer")
            self.notificationTwo.place(x = 520, y = 367)
            self.step2 = False
            return None

        if weightdecay > 1 or weightdecay < 0:
            self.notification_textTwo.set("Weight Decay must be between 0 and 1")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None            

        if int(trainingSplit) + int(validationSplit) + int(testingSplit) != 100.0:
            self.notification_textTwo.set("The three data splits must add up to 100%")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None

        if int(trainingSplit) <= 0 or int(validationSplit) <= 0 or int(testingSplit) <= 0 or int(batchSize) <= 0 or float(learningRate) <= 0 or int(epochs) <= 0 or int(earlyStopping) <= 0:
            self.notification_textTwo.set("The entered fields must be a positive integer")
            self.notificationTwo.place(x = 520, y = 367)
            self.step2 = False
            return None

        try:
            self.backEnd.split_data(batchSize, self.addNormalize, trainingSplit, validationSplit, testingSplit, learningRate, epochs, weightdecay, earlyStopping, self.optimizers, criterion)
        except:
            self.notification_textTwo.set("An error has occured. Ensure dataset is big enough")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None

        if batchSize >= self.backEnd.datalist[0]:
            self.notification_textTwo.set("Batch size must be smaller than training data")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None

        if batchSize >= self.backEnd.datalist[1]:
            self.notification_textTwo.set("Batch size must be smaller than validation data")
            self.notificationTwo.place(x= 510, y=367)
            self.step2 = False
            return None

        self.notificationTwo.place_forget()
        if self.preArch != "None":
            preArchConv = {"LeNet":{1: [self.backEnd.imDim[0], 6, 5, 5, 1, 1, True, False, 2, 2, '', '', 2, 2, 0.0, 'Sigmoid'], 2: [6, 16, 5, 5, 1, 1, True, False, 2, 2, '', '', 2, 2, 0.0, 'Sigmoid']}, "AlexNet": {1: [self.backEnd.imDim[0], 96, 11, 11, 4, 4, True, False, 3, 3, '', '', 2, 2, 0.0, 'ReLU', '', ''], 2: [96, 256, 5, 5, 1, 1, True, True, 3, 3, 2, 2, 2, 2, 0.0, 'ReLU', 2, 2], 3: [256, 384, 3, 3, 1, 1, False, True, '3', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 4: [384, 384, 3, 3, 1, 1, False, True, '3', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 5: [384, 256, 3, 3, 1, 1, True, True, 3, 3, 1, 1, 2, 2, 0.5, 'ReLU', 1, 1]},"VGGNet": {1: [3, 64, 3, 3, 1, 1, False, True, '', '', 1, 1, '', '', 0.0, 'ReLU', 1, 1], 2: [64, 64, 3, 3, 1, 1, True, True, 2, 2, 1, 1, 2, 2, 0.0, 'ReLU', 1, 1], 3: [64, 128, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 4: [128, 128, 3, 3, 1, 1, True, True, 2, 2, 1, 1, 2, 2, 0.0, 'ReLU', 1, 1], 5: [128, 256, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 6: [256, 256, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 7: [256, 256, 3, 3, 1, 1, True, True, 2, 2, 1, 1, 2, 2, 0.0, 'ReLU', 1, 1], 8: [256, 512, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 9: [512, 512, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 10: [512, 512, 3, 3, 1, 1, True, True, 2, 2, 1, 1, 2, 2, 0.0, 'ReLU', 1, 1], 11: [512, 512, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 12: [512, 512, 3, 3, 1, 1, False, True, '2', '', 1, 1, '2', '', 0.0, 'ReLU', 1, 1], 13: [512, 512, 3, 3, 1, 1, True, True, 2, 2, 1, 1, 2, 2, 0.0, 'ReLU', 1, 1]}}
            preArchFull = {"LeNet":{1: [400, 120, 0.0, 'Sigmoid'], 2: [120, 84, 0.0, 'Sigmoid'], 3: [84, len(self.backEnd.dataPerLabel), 0.0, 'Softmax']}, "AlexNet": {1: [9216, 4096, 0.5, 'ReLU'], 2: [4096, 4096, 0.5, 'ReLU'], 3: [4096, len(self.backEnd.dataPerLabel), 0.0, 'Softmax']}, "VGGNet": {1: [25088, 4096, 0.5, 'ReLU'], 2: [4096, 4096, 0.5, 'ReLU'], 3: [4096, len(self.backEnd.dataPerLabel), 0.0, 'Softmax']}}
            self.backEnd.convLayers = preArchConv[self.preArch]
            self.backEnd.fullLayers = preArchFull[self.preArch]
            self.threadTwo = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.stepTwo_queue))
            self.threadTwo.daemon = True
            self.threadTwo.start()
            self.notification_textTwo.set("Loading Pre-selected Model")
            self.notificationTwo.place(x= 510, y=367)
            self.stepTwoLoad()
        else:
            self.step2 = True
            self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
        return None

    def stepTwoLoad(self):
        if self.stepTwo_queue.qsize() >= 1:
            data_get = self.stepTwo_queue.get()
            if data_get == True:
                self.step2 = True
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                self.notificationTwo.place_forget()
                self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
                self.curConvLabel.place(x = 362, y = 615)
                self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
                self.curFullLabel.place(x = 841, y = 615)
                self.notificationThree.place_forget()
                self.notificationFour.place_forget()
                self.notificationOne.place_forget()
                self.notificationTwo.place_forget()
                return None
        self.master.after(10, self.stepTwoLoad)

    def stepThreeSubmit(self):
        num_in = self.in_Channel_Search_Bar.get()
        num_out = self.out_Channel_Search_Bar.get()
        kernel_size_L = self.kernel_L_SB.get()
        kernel_size_W = self.kernel_W_SB.get()
        conv_stride_L = self.strideLengthconv_Search_Bar.get()
        conv_stride_W = self.convStrideWSB.get()
        pooling_size_L = self.poolingSize_Search_Bar.get()
        pooling_size_W = self.poolingSizeWidthSB.get()
        pool_stride_L = self.stride_Length_Search_Bar.get()
        pool_stride_W = self.poolStrideWidthSB.get()
        acti_func = self.actiFuncConv
        dropoutrate = self.convDropOutEntry.get()
        

        padding_left = self.paddingSize_LeftSB.get()        #Left
        padding_right = self.paddingSize_Search_Bar.get()   #Right
        padding_top = self.paddingSize_TopSB.get()          #Top
        padding_bottom = self.paddingSizeWidthSB.get()      #Bottom
        
        if self.step2 == False:
            self.notification_textThree.set("Step Two has not been completed")
            self.notificationThree.place(x= 30, y=620)
            return None

        if self.loading_thread is not None:
            if self.loading_thread.is_alive():
                self.notification_textThree.set("Step 1 is loading. Please wait")
                self.notificationThree.place(x= 30, y=620)
                return None

        if self.threadTwo is not None:
            if self.threadTwo.is_alive():
                self.notification_textThree.set("Step 2 is loading. Please wait")
                self.notificationThree.place(x= 30, y=620)
                return None

        if self.threadThree is not None:
            if self.threadThree.is_alive():
                self.notification_textThree.set("Model is updating. Please wait")
                self.notificationThree.place(x= 30, y=620)
                return None

        if self.threadFour is not None:
            if self.threadFour.is_alive():
                self.notification_textThree.set("Step 4 is loading. Please wait")
                self.notificationThree.place(x= 30, y=620)
                return None

        if num_in == "" or num_out == "" or kernel_size_L == "" or conv_stride_L == "" or acti_func == "" or acti_func == "" or dropoutrate == "":
            self.notification_textThree.set("One or more fields is empty. Please fill them in first")
            self.notificationThree.place(x= 30, y=620)
            return None
        
        if kernel_size_W == "": kernel_size_W = kernel_size_L
        if conv_stride_W == "": conv_stride_W = conv_stride_L

        if self.addPoolingBool == True:
            if pooling_size_L == "" or pool_stride_L == "":
                self.notification_textThree.set("One or more fields is empty. Please fill them in first")
                self.notificationThree.place(x= 30, y=620)
                return None
            if pool_stride_W == "": pool_stride_W = pool_stride_L
            if pooling_size_W == "": pooling_size_W = pooling_size_L

        if self.addPaddingBool == True:
            if padding_left == "":
                self.notification_textThree.set("One or more fields is empty. Please fill them in first")
                self.notificationThree.place(x= 30, y=620)
                return None
            if padding_bottom == "": padding_bottom = padding_left
            if padding_top == "":padding_top = padding_left
            if padding_right == "": padding_right = padding_left

        try:
            num_in = int(num_in)
            num_out = int(num_out)
            kernel_size_L = int(kernel_size_L)
            kernel_size_W = int(kernel_size_W)
            conv_stride_L = int(conv_stride_L)
            conv_stride_W = int(conv_stride_W)
            dropoutrate = float(dropoutrate)
            if self.addPoolingBool == True:
                pooling_size_L = int(pooling_size_L)
                pooling_size_W = int(pooling_size_W)
                pool_stride_L = int(pool_stride_L)
                pool_stride_W = int(pool_stride_W)
            if self.addPaddingBool == True:
                padding_right = int(padding_right)
                padding_bottom = int(padding_bottom)
                padding_top = int(padding_top)
                padding_left = int(padding_left)
        except:
            self.notification_textThree.set("The entered fields must be a positive integer")
            self.notificationThree.place(x= 30, y=620)
            return None

        if dropoutrate < 0 or dropoutrate > 1:
            self.notification_textThree.set("The Dropout rate must be between 0 and 1")
            self.notificationThree.place(x= 30, y=620)
            return None            

        if num_in < 0 or num_out < 0 or kernel_size_L < 0 or kernel_size_W < 0 or conv_stride_L < 0 or conv_stride_W < 0:
            self.notification_textThree.set("The entered fields must be a positive integer")
            self.notificationThree.place(x= 30, y=620)
            return None

        if self.addPoolingBool == True:
            if pooling_size_L < 0 or pooling_size_W < 0 or pool_stride_L < 0 or pool_stride_W < 0:
                self.notification_textThree.set("The entered fields must be a positive integer")
                self.notificationThree.place(x= 30, y=620)
                return None

        if self.addPaddingBool == True:
            if padding_right < 0 or padding_bottom < 0 or padding_top <0 or padding_left <0:
                self.notification_textThree.set("The entered fields must be a positive integer")
                self.notificationThree.place(x= 30, y=620)
                return None

        self.backEnd.convLayers[self.backEnd.convLayerSteps] = [num_in, num_out, kernel_size_L, kernel_size_W, conv_stride_L, conv_stride_W, self.addPoolingBool, self.addPaddingBool, pooling_size_L, pooling_size_W, padding_right, padding_bottom, pool_stride_L, pool_stride_W, dropoutrate ,acti_func, padding_left, padding_top]
        self.threadThree = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.stepTwo_queue))
        self.threadThree.daemon = True
        self.threadThree.start()
        self.notification_textThree.set("Updating Model")
        self.notificationThree.place(x= 30, y=620)
        self.stepThreeLoad()
        return None

    def stepThreeLoad(self):
        if self.stepTwo_queue.qsize() >= 1:
            data_get = self.stepTwo_queue.get()
            if data_get == True:
                self.step3 = True
                self.backEnd.convLayerSteps += 1
                self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
                self.curConvLabel.place(x = 362, y = 615)
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                self.notificationThree.place_forget()
                self.notificationFour.place_forget()
                self.notificationOne.place_forget()
                self.notificationTwo.place_forget()
                return None
        self.master.after(10, self.stepThreeLoad)


    def stepFourSubmit(self):
        num_in = self.inChannels_Search_Bar.get()
        num_out = self.outChannels_Search_Bar.get()
        acti_func = self.actiFuncFull
        dropoutrate = self.dropoutEntry.get()

        self.notification_textFour.set("")
        if not self.step2:
            self.notification_textFour.set("Step Two has not been completed")
            self.notificationFour.place(x= 510, y=620)
            return None
#
        if self.loading_thread is not None:
            if self.loading_thread.is_alive():
                self.notification_textFour.set("Step 1 is loading. Please wait")
                self.notificationFour.place(x= 510, y=620)
                return None

        if self.threadTwo is not None:
            if self.threadTwo.is_alive():
                self.notification_textFour.set("Step 2 is loading. Please wait")
                self.notificationFour.place(x= 510, y=620)
                return None

        if self.threadThree is not None:
            if self.threadThree.is_alive():
                self.notification_textFour.set("Step 3 is loading. Please wait")
                self.notificationFour.place(x= 510, y=620)
                return None

        if self.threadFour is not None:
            if self.threadFour.is_alive():
                self.notification_textFour.set("Model is updating. Please wait")
                self.notificationFour.place(x= 510, y=620)
                return None
# 

        if num_in == "" or num_out == "" or acti_func == "" or dropoutrate == "":
            self.notification_textFour.set("One or more fields is empty. Please fill them in first")
            self.notificationFour.place(x= 510, y=620)
            return None
        try:
            num_in = int(num_in)
            num_out = int(num_out)
            dropoutrate = float(dropoutrate)
        except:
            self.notification_textFour.set("The entered fields must be a positive integer")
            self.notificationFour.place(x= 510, y=620)
            return None

        if num_in < 0 or num_out < 0:
            self.notification_textFour.set("The entered fields must be a positive integer")
            self.notificationFour.place(x= 510, y=620)
            return None

        if dropoutrate < 0 or dropoutrate > 1:
            self.notification_textFour.set("Dropout Rate must be between 0 and 1")
            self.notificationFour.place(x= 510, y=620)
            return None

        self.backEnd.fullLayers[self.backEnd.fullLayerSteps] = [num_in, num_out, dropoutrate, acti_func]
        self.threadFour = threading.Thread(target=lambda: self.backEnd.singleForwardPass(self.stepTwo_queue))
        self.threadFour.daemon = True
        self.threadFour.start()
        self.notification_textFour.set("Updating Model")
        self.notificationFour.place(x= 510, y=620)
        self.stepFourLoad()
        return None

    def stepFourLoad(self):
        if self.stepTwo_queue.qsize() >= 1:
            data_get = self.stepTwo_queue.get()
            if data_get == True:
                self.backEnd.fullLayerSteps += 1
                self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
                self.curFullLabel.place(x = 841, y = 615)
                self.infrastructureVisualization(self.backEnd.convShapes, self.backEnd.fullShapes, theme_ = self.themeBool)
                self.notificationThree.place_forget()
                self.notificationFour.place_forget()
                self.notificationOne.place_forget()
                self.notificationTwo.place_forget()
                return None
        self.master.after(10, self.stepFourLoad)

    def page_one(self):
        self.inPageOne = True
        self.infrastructureVisualization(shapes = self.backEnd.convShapes,shapes2 = self.backEnd.fullShapes, theme_ = self.themeBool)
        if len(self.backEnd.convLayers) > 0:
            self.curConvText.set(str(self.backEnd.convLayerSteps) + "/" + str(len(self.backEnd.convLayers)))
            self.curConvLabel.place(x = 362, y = 615)
        if len(self.backEnd.fullLayers) > 0:
            self.curFullText.set(str(self.backEnd.fullLayerSteps) + "/" + str(len(self.backEnd.fullLayers)))
            self.curFullLabel.place(x = 841, y = 615)

        #Step One Utilities
        self.configureData.place(x = 25, y = 145)
        self.dataLoadCanvas.place(x = 20, y = 140)
        self.stepOneBanner.place(x = 20, y = 110)
        self.loadPreTrained.place(x = 40, y = 169)
        self.loadPTBoolButton.place(x = 175, y = 169)
        self.dataPathWay.place(x = 40, y = 193)
        self.dataPathSearch_Bar.place(x = 175, y = 193)
        self.dataTypeLabel.place(x = 40, y = 218)
        self.dataCB.place(x = 175, y = 218)
        self.imageSize.place(x = 25, y = 240)

        self.imageHeight.place(x = 80, y = 265)            #x = 240, y = 265
        self.heightSizeSearch_Bar.place(x = 140, y = 267)    #x = 300, y= 267
        self.imageWidth.place(x = 240, y = 265)              #x = 80, y = 265
        self.widthSizeSearch_Bar.place(x = 300, y= 267)    #x = 140, y = 267

        self.stepOneSubmitButton.place(x = 335, y = 350)
        self.chooseModelLabel.place(x = 25, y = 300)
        self.archType.place(x = 40, y = 325)
        self.archCB.place(x = 135, y = 327)
       
        #Step Two Utilities
        self.dataSplitCanvas.place(x = 500, y = 140)
        self.stepTwoBanner.place(x = 500, y = 110)
        self.training_split.place(x = 505, y = 165)
        self.testing_split.place(x = 640, y = 165)
        self.validation_split.place(x = 770, y = 165)
        self.datasplitLbl.place(x = 505, y = 145)
        self.training_Split_Search_Bar.place(x = 537, y = 186)
        self.testing_Split_Search_Bar.place(x = 673, y = 186)
        self.validation_Split_Search_Bar.place(x = 807, y = 186)
        self.modelpropertieslbl.place(x = 505, y = 206)
        self.numEpochsLabel.place(x = 770, y = 226)
        self.numEpochsSB.place(x = 807, y = 250)
        self.batchSizeLabel.place(x = 517, y = 224)
        self.batchSizeSB.place(x = 537, y = 247)
        self.learningRateLabel.place(x = 645, y = 226)
        self.lrSB.place(x = 673, y = 250)
        self.trainingOptimizationLbl.place(x = 505, y = 268)
        self.optimizerLabel.place(x = 520, y = 290)
        self.optiComboBox.place(x = 600, y = 290)
        self.normalizeDataLbl.place(x = 520, y = 340)
        self.normCheckBox.place(x = 610, y = 346)
        self.criterionLbl.place(x = 520, y = 315)
        self.criComboBox.place(x = 600, y = 315)
        self.stepTwoSubmitButton.place(x = 815, y = 350)
        self.verticalLineCanvas.place(x = 725, y = 292)

        self.weightDecay.place(x = 735, y = 290)
        self.weightDecayEntry.place(x = 840, y = 293)
        self.earlyStoppingEntry.place(x = 840, y = 317)
        self.earlyStoppingLabel.place(x = 735, y = 315)
        
        #Step Three Utilities
        self.convLayerProp.place(x = 25, y = 435)
        self.paddingAndPoolingLayer.place(x = 25, y = 497)
        self.optInOut.place(x = 197, y = 500)           #x = 210
        self.poolingCheckBox.place(x = 217, y = 526)    #x = 230
        self.paddingCheckBox.place(x = 217, y = 551)    #x = 230
        self.stepThreeBanner.place(x = 20, y = 400)
        self.convLayerCanvas.place(x = 20, y = 430)
        self.numInChannels.place(x = 40, y = 455)
        self.numOutChannels.place(x = 140, y = 455)
        self.kernelSize.place(x = 245, y = 455)
        self.strideLength.place(x = 330, y = 455)
        self.in_Channel_Search_Bar.place(x = 60, y = 475)
        self.out_Channel_Search_Bar.place(x = 160, y = 475)
        self.kernelLength.place(x = 243, y = 475)
        self.kernel_L_SB.place(x = 260, y = 475)
        self.kernelWidth.place(x = 275, y = 475)
        self.kernel_W_SB.place(x = 295, y = 475)
        self.convStrideLength.place(x = 334, y = 475)
        self.convStrideWidth.place(x = 364, y = 475)
        self.strideLengthconv_Search_Bar.place(x = 349, y = 475)
        self.convStrideWSB.place(x = 385, y = 475)


        self.poolingSizeLengthLabel.place(x = 266, y = 522) #x = 233
        self.poolingSize_Search_Bar.place(x = 282, y = 522) #x = 249 
        self.poolingSizeWidthLabel.place(x = 298, y = 522) #x = 265
        self.poolingSizeWidthSB.place(x = 319, y = 522) # x = 286

        self.poolingStrideLengthLabel.place(x = 342, y = 522) #x = 309
        self.poolStrideWidthSB.place(x = 394, y = 522) #x = 361
        self.poolingStrideWidthLabel.place(x = 373, y = 522) #x = 340
        self.stride_Length_Search_Bar.place(x = 358, y = 522) #x = 325

        self.paddingSizeLeftLabel.place(x = 235, y = 547)
        self.paddingSize_LeftSB.place(x = 249, y = 547)

        self.paddingSizeLengthLabel.place(x = 265, y = 547)
        self.paddingSize_Search_Bar.place(x = 282, y = 547)

        self.paddingSizeTopLabel.place(x = 300, y = 547)
        self.paddingSize_TopSB.place(x = 319, y = 547)

        self.paddingSizeWidthLabel.place(x = 335, y = 547)
        self.paddingSizeWidthSB.place(x = 352, y = 547)

        self.paddingLabel.place(x = 40, y = 545)
        self.poolingLabel.place(x = 40, y = 520)
        self.sizeLabel.place(x = 296, y = 500) #x = 288
        self.strideLengthLabel.place(x = 366, y = 500) # x = 358
        self.stepThreeSubmitButton.place(x = 340, y = 643)
        self.curConvUp.place(x = 347, y = 597)
        self.convDropout.place(x = 40, y = 595)
        self.convDropOutEntry.place(x = 150, y = 598)
        self.activationFunctions.place(x = 40, y = 570)
        self.combobox.place(x = 190,y = 572)
        self.deleteAllConvLbl.place(x = 95, y = 651)
        self.deleteAllConvLayer.place(x = 150, y = 645)
        self.deleteConvLbl.place(x = 185, y = 651)
        self.deleteConvLayer.place(x = 255, y = 645)
        self.convLeftButton.place(x = 30, y = 645)
        self.convRightButton.place(x = 60, y = 645)

        
        #Step Four Utilities
        self.stepFourBanner.place(x = 500, y = 400)
        self.fullConnCanvas.place(x = 500, y = 430)
        self.fullConnLabel.place(x = 505, y = 435)
        self.fullConnNumInChannels.place(x = 520, y = 460)
        self.fullConnNumOutChannels.place(x = 520, y = 485)
        self.inChannels_Search_Bar.place(x = 700, y = 460)
        self.outChannels_Search_Bar.place(x = 700, y = 485)
        self.comboboxFullConn.place(x = 700, y = 535)
        self.fullConnActiFunctLabel.place(x = 520, y = 535)
        self.stepFourSubmitButton.place(x = 820, y = 643)
        self.curFullUp.place(x = 827, y = 597)
        self.dropout.place(x = 520, y = 510)
        self.dropoutEntry.place(x = 700, y = 510)
        self.fullLeftButton.place(x = 510,y= 645)
        self.fullRightButton.place(x = 540, y = 645)
        self.deleteAllFullLbl.place(x = 575, y = 651)
        self.deleteAllFullLayer.place(x = 630, y = 645)
        self.deleteFullLbl.place(x = 665, y = 651)
        self.deleteFullLayer.place(x = 735, y = 645)

        
        #Infrastructure Utilities
        self.infrastructureLbl.place(x = 965, y = 110)
        self.copyModelProp.place(x = 1140, y = 110)
        self.copyModelButton.place(x = 1250, y = 108)

        #Instruction Utilities
        self.removeInstruction_page()
        self.instructionButton.place(x = 1165, y = 30)
        self.userGuideLbl.place(x = 1095, y = 33)

        #Training
        self.trainButton.place(x = 1220,y = 630)

    def instruction_page(self, event):
        if self.instructionButton_tag:
            self.inInstructions = True
            for i in self.pageOneList:
                i.pack()
                i.pack_forget()
            self.instructionButton.pack()
            self.instructionButton.pack_forget()
            self.userGuideLbl.pack()
            self.userGuideLbl.pack_forget()
            self.backButton.place(x = 1237, y = 680)
            self.introButton.place(x = 1237, y = 100)
            self.step1InstructionButton.place(x = 1237, y = 140)
            self.step2InstructionButton.place(x = 1237, y = 180)
            self.step3InstructionButton.place(x = 1237, y = 220)
            self.step4InstructionButton.place(x = 1237, y = 260)
            self.stepArchInstructionButton.place(x = 1237, y = 300)
            self.trainingInstructionButton.place(x = 1237, y = 340)
            self.devNoteButton.place(x = 1237, y = 380)
            self.introIns()

    def removeInstruction_page(self):
        self.backButton.pack()
        self.backButton.pack_forget()
        self.inInstructions = False
        for i in (self.instruction_Page + self.stepOne_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.stepTwo_instruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()

    def introIns(self):
        for i in (self.stepTwo_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.stepOne_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.introInsTitle.place(x = 20, y = 100)
        self.introIns_.place(x = 20, y = 120)

    def stepOneIns(self):
        for i in (self.stepTwo_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.oneInsIm.place(x = 820, y = 100)
        self.oneInsTitle.place(x = 20, y = 100)
        self.oneIns1.place(x = 20, y = 120)

    def stepTwoIns(self):
        for i in (self.stepOne_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.twoInsIm.place(x = 820, y = 100)
        self.twoInsTitle.place(x = 20, y = 100)
        self.twoIns2.place(x = 20, y = 120)

    def stepThreeIns(self):
        for i in (self.stepTwo_instruction + self.stepOne_instruction + self.stepFour_instruction + self.archInstruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.threeInsIm.place(x = 820, y = 100)
        self.threeInsTitle.place(x = 20, y = 100)
        self.threeIns3.place(x = 20, y = 120)

    def stepFourIns(self):
        for i in (self.stepTwo_instruction + self.stepThree_instruction + self.stepOne_instruction + self.archInstruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.fourInsIm.place(x = 820, y = 100)
        self.FourInsTitle.place(x = 20, y = 100)
        self.fourIns4.place(x = 20, y = 120)

    def archIns(self):
        for i in (self.stepTwo_instruction + self.stepThree_instruction + self.stepFour_instruction + self.stepOne_instruction + self.intro_instruction + self.trainingInstruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.archConst = 1
        title = {1: "Infrastructure Page One: Step 1 and 2", 2: "Infrastructure Page Two: Step 3", 3: "Infrastructure Page Three: Step 4"}
        im = {1: 27, 2: 29, 3: 31}
        desc = {1: 28, 2: 30, 3: 32}
        self.archLabel.configure(text = title[self.archConst])
        self.arch1Ins.configure(text = self.imageStorage.call_(desc[self.archConst],True))
        self.archIns1 = self.imageStorage.call(im[self.archConst], self.themeBool)
        self.archIns1Im.configure(image = self.archIns1)

        self.archInsTitle.place(x = 20, y = 100)
        self.archIns1Im.place(x = 850, y = 100)
        self.arch1Ins.place(x = 20, y = 120)
        self.archLabel.place(x = 350, y = 100)
        self.archForwardButton.place(x = 800, y = 100)
        self.archBackwardButton.place(x = 770, y = 100)

    def archNext(self):
        title = {1: "Infrastructure Page One: Step 1 and 2", 2: "Infrastructure Page Two: Step 3", 3: "Infrastructure Page Three: Step 4"}
        im = {1: 27, 2: 29, 3: 31}
        desc = {1: 28, 2: 30, 3: 32}
        if self.archConst < 3:
            self.archConst += 1
            self.archLabel.configure(text = title[self.archConst])
            self.arch1Ins.configure(text = self.imageStorage.call_(desc[self.archConst],True))
            self.archIns1 = self.imageStorage.call(im[self.archConst], self.themeBool)
            self.archIns1Im.configure(image = self.archIns1)

    def archBack(self):
        title = {1: "Infrastructure Page One: Step 1 and 2", 2: "Infrastructure Page Two: Step 3", 3: "Infrastructure Page Three: Step 4"}
        im = {1: 27, 2: 29, 3: 31}
        desc = {1: 28, 2: 30, 3: 32}
        if self.archConst > 1:
            self.archConst -= 1
            self.archLabel.configure(text = title[self.archConst])
            self.arch1Ins.configure(text = self.imageStorage.call_(desc[self.archConst],True))
            self.archIns1 = self.imageStorage.call(im[self.archConst], self.themeBool)
            self.archIns1Im.configure(image = self.archIns1)

    def trainIns(self):
        for i in (self.stepOne_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.stepTwo_instruction + self.intro_instruction + self.devsNoteInstruction):
            i.pack()
            i.pack_forget()
        self.trainInsTitle.place(x = 20, y= 100)
        self.trainingIns_.place(x = 20, y = 120)

    def devNote(self):
        for i in (self.stepOne_instruction + self.stepThree_instruction + self.stepFour_instruction + self.archInstruction + self.stepTwo_instruction + self.intro_instruction + self.trainingInstruction):
            i.pack()
            i.pack_forget()
        self.devNoteTitle.place(x = 20, y= 100)
        self.devNoteIns_.place(x = 20, y = 120)

if __name__ == "__main__":
    window = tk.Tk()

    app = UI(window)
    app.page_one()

    window.geometry("1400x750")
    window.resizable(False,False)

    window.mainloop()