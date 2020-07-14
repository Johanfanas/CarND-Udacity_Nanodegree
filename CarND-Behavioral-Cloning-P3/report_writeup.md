# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of one convolution neural network with 5x5 filter sizes and depths between 32 and 128 (model.py lines 94-102). I used a simple achitecture because everytime I tried using a powerful architecture, it overfitted the data.

The data is first normalized using a Keras lambda layer (code line 95) with an input shape of 160,320,3. The input data is then cropped using Keras Cropping 2D layer (code line 96). The images are then inputted to a
convolutional neural network followed by a RELU activation layer (code line 97) and a dropout of 60% to avoid overfitting. The layer is then flatten to obtain one output.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98). The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. An Early Stop callback was used in the fit generator so the model stops training when it detects overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data was gathered by driving in the center of the lane at low speed. In my case, this was sufficient to make my model drive successfully one lap. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was starting with the Nvidia architecture and modifying layers while observing its performance until I obtained good predictions.

I tried using LeNet but it kept overfitting the data and it didn't drive the vehicle that well. Then I tried to removing layers and adding dropout and max pooling layers. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. For my model I used 3 epochs. The training loss and validation loss both decreased in every epoch which means our model is not overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle got very close to the wall but the vehicle never stepped on the lane lines. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-102) consists of the following architecture: 

Image Input Shape 160x320x3
Convolution neural network of 15 with a 5x5 kernel
Max Pooling Layer
60% Dropout Layer
Output Layer of 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][https://github.com/udacity/CarND-Behavorial-Clonning-P3/blob/master/report_images/center_camera.JPG]

I drove the vehicle at a low speed to make sure the vehicle was approximately centered. For my model I also used images from the left and right cameras and used a correction value for the steering angle. To augment the data and avoid recording driving on the opposite direction I flipped the images and inverted the sign on the steering angle. Below is shown an example of the image from the left camera flipped.

![alt text][https://github.com/udacity/CarND-Behavorial-Clonning-P3/blob/master/report_images/left_camera.jpg]
![alt text][https://github.com/udacity/CarND-Behavorial-Clonning-P3/blob/master/report_images/flip_left_camera.jpg]

In a function called data I stored the image paths and steering angle in two lists. The data of both lists are then shuffled. I split the data set and put 20% of the data into the validation set. After the collection process, I had 13644 number of trainning data. I then preprocessed this data by normalizing it and mean centering the data. Also, the image is cropped to eliminate irrelevant information from the image.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs I found worked the best for my model was three. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I saved my model as 'model.h5' and the simulation video is called run.mp4. In this video it seems as the vehicle steps on the lane lines and bumps into the wall on the bridge but it doesn't. Since the video is taken from the front centered camera, I made a video of my screen (called behavorial_cloning.mp4) so you can verify it does not violate any of the project rules.
