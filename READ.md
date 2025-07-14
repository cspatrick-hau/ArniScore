# ArniScore

This repository contains the core project files. Due to large file size limitations, some folders (such as `venv`, `datasets`, and `models`) are **not included** in the GitHub repository. Please follow the instructions below to set up your environment and download the necessary resources.

---

## ✅ Install Virtual Environment

To isolate your project dependencies and ensure compatibility, it's recommended to use a virtual environment. Follow the steps below:

Make sure you have Python installed. Type "python -m venv venv" in terminal to install virtual environment. It will generate a folder like this:
![venv folder](images/venv.png)

This folder contains the project dependencies such as tensorflow, matplotlib, and etc.


## 📁 Download Dataset

Download the dataset manually from the link below:

📥 [Click here to download the dataset](https://drive.google.com/drive/folders/1CAuES_XsapAbSjg2op5Imr8i3q1P2rVJ)

After downloading, extract and place the dataset folder into the project root directory as shown:

![dataset folder](images/dataset.png)



## Detailed Instructions For LSTM + CNN

Note: Datasets were not pushed in github due to storage restrictions

=> Install the Libraries from requirements.txt

Outline
Step 1: Download Visualize the Data with its Labels
Step 2: Preprocess the Dataset
Step 3: Split the Data into Train and Test Set
Step 4: Implement the ConvLSTM Approach
Step 4.1: Construct the Model
Step 4.2: Compile & Train the Model
Step 4.3: Plot Model’s Loss & Accuracy Curves
Step 5: Test the Model

Step 1: Download and Visualize the Data with its Labels
In the first step, we will download and visualize the data along with labels to get an idea about what we will be dealing with. We will be using the actual dataset from the GDrive. The Dataset contains:

2 Class Categories(Valid and Invalid)

Step 2: Preprocess the Dataset
Next, we will perform some preprocessing on the dataset. First, we will read the video files from the dataset and resize the frames of the videos to a fixed width and height, to reduce the computations and normalized the data to range [0-1] by dividing the pixel values with 255, which makes convergence faster while training the network.

=> Create a Function to Extract, Resize & Normalize Frames
We will create a function frames_extraction() that will create a list containing the resized and normalized frames of a video whose path is passed to it as an argument. The function will read the video file frame by frame, although not all frames are added to the list as we will only need an evenly distributed sequence length of frames.

=> Create a Function for Dataset Creation
Now we will create a function create_dataset() that will iterate through all the classes specified in the CLASSES_LIST constant and will call the function frame_extraction() on every video file of the selected classes and return the frames (features), class index ( labels), and video file path (video_files_paths).

Step 3: Split the Data into Train and Test Set
As of now, we have the required features (a NumPy array containing all the extracted frames of the videos) and one_hot_encoded_labels (also a Numpy array containing all class labels in one hot encoded format). So now, we will split our data to create training and testing sets. We will also shuffle the dataset before the split to avoid any bias and get splits representing the overall distribution of the data.

Step 4: Implement the ConvLSTM Approach
In this step, we will implement the first approach by using a combination of ConvLSTM cells. A ConvLSTM cell is a variant of an LSTM network that contains convolutions operations in the network. it is an LSTM with convolution embedded in the architecture, which makes it capable of identifying spatial features of the data while keeping into account the temporal relation.

=> For video classification, this approach effectively captures the spatial relation in the individual frames and the temporal relation across the different frames. As a result of this convolution structure, the ConvLSTM is capable of taking in 3-dimensional input (width, height, num_of_channels) whereas a simple LSTM only takes in 1-dimensional input hence an LSTM is incompatible for modeling Spatio-temporal data on its own.

=> You can read the paper Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting by Xingjian Shi (NIPS 2015), to learn more about this architecture.

Step 4.1: Construct the Model
To construct the model, we will use Keras ConvLSTM2D recurrent layers. The ConvLSTM2D layer also takes in the number of filters and kernel size required for applying the convolutional operations. The output of the layers is flattened in the end and is fed to the Dense layer with softmax activation which outputs the probability of each action category.

=> We will also use MaxPooling3D layers to reduce the dimensions of the frames and avoid unnecessary computations and Dropout layers to prevent overfitting the model on the data. The architecture is a simple one and has a small number of trainable parameters. This is because we are only dealing with a small subset of the dataset which does not require a large-scale model.

Step 4.2: Compile & Train the Model
Next, we will add an early stopping callback to prevent overfitting and start the training after compiling the model.

Step 4.3: Plot Model’s Loss & Accuracy Curves
Now we will create a function plot_metric() to visualize the training and validation metrics. We already have separate metrics from our training and validation steps so now we just have to visualize them.

Step 5: Test the Model
Next, we will create a function predict_on_video() that will simply read a video frame by frame from the path passed in as an argument and will perform action recognition on video and save the results.

=> Now we will utilize the function predict_on_video() created above to perform action recognition on the test video we had downloaded using the function download_youtube_videos() and display the output video with the predicted action overlayed on it.



