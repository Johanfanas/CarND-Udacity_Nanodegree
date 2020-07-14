# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). All images are in the HTML file just in case.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used common python built in functions to determine the size of each data set, number of classes/labels in the data set and shape of the images.

Data set information:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I start by plotting a random image from the dataset in order to assure the images are imported correctly. Next, I plot the histrograms of the training, validation and testing data set. These histograms give us information of how many images are of a certain Class ID in each data set.

![alt text][https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/data_visualization.JPG]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is easier for computers to analyze images because they only contain one channel the color space ranges from 0-255. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/gray_image.JPG)

As a last step, I normalized the image data because it allows each pixel to have a similar data distribution.  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| RELU and dropout 		| Dropout of 0.7    							|
| Max pooling	      	|  2x2 stride,  outputs 14x14x6 				|
|Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU and dropout		| Dropout of 0.7 								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input 400, Output 120   						|
| Fully connected		| Input 120, Output 84   						|
| Fully connected		| Input 84, Output 43   						|
| Softmax				| Followed by minimizing cross entropy 			|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer (Stochastic Optimization Method), a batch size os 128, 10 epochs and a learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.947 
* test set accuracy of 0.921

If an iterative approach was chosen:
The first architecture I tried was Lenet. Using this achitecture with no data processing techniques we obtained a validation accuracy of 0.88; therefore, processing techniques had to be applied in order for it to be easier for the NN. Initially, only normalization was applied but we were still obtainin a validation below 0.93 and the NN was over fitting the data. To avoid over fitting we added a dropout after each RELU activation function. Also, all images were converted to grayscale.
 
Adding dropout after every RELU activation helped a lot because the validation accuracy increased rapidly instead of having an oscillating behavior as before. Before adding dropouts, I was using 20 Epochs and still couldn't reach successfully the validation accuracy target. A validation accuracy of 94.7% and a test accuracy of 92.1% indicate our model is working well. Also, the continous increase of the validation accuracy with 10 Epochs indicate the model is not over fitting the data.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt5.jpg) 
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt1.jpg) 
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt2.jpg)
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt3.jpg) 
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt4.jpg)
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt7.jpg)
![alt text](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/test_images/gt8.jpg)

I successfully downloaded and processed the images from on the web. They were plotted to assure they were imported correctly. The images were saved in a folder called test_images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I used my model to predict seven images downloaded from the web and achieved 33%.

| Label         		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 80km/h 	| Speed limit 50km/h   							| 
| Turn right 			| Turn right        							|
| Stop sign	   		   	|  Stop sign            						|
| Speed limit 60km/h  	| Speed limit 100km/h 							|
| Road work	   		   	|  Road work             						|
| Caution sign 		 	| Dangerous curve to the right 					|

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 6 softmax probabilities for each image were provided and can be seen in the tables below. For the first image we got:

| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.77              	| Speed limit (50km/h)  						| 
| 0.17       			| Speed limit (30km/h) 							|
| 0.036  	   		   	|  Wild animals crossing   						|
| 0.049              	| Speed limit (80km/h) 							|
| 0.0034  	   		   	|  Dangerous curve to the left  	 			|
| 0.00074    		 	| Right-of-way at the next intersection 		|

Our model was uncertai for the first image. The correct prediction would have been the fourth probability. For the second image we have

| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.37              	| Ahead only            						| 
| 0.16       			| No passing for vehicles over 3.5 metric tons 	|
| 0.089  	   		   	|  Right-of-way at the next intersection   			|
| 0.084              	| Turn right ahead  							|
| 0.059  	   		   	|  Right-of-way at the next intersection  		|
| 0.052     		 	| Speed limit (80km/h)         					|

The model was also very uncertain with our second image. Again, the correct prediction would have been the fourth probatility. For the third image we have

| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.85              	| Stop                  						| 
| 0.045       			| Go straight or right 							|
| 0.017  	   		   	|  Speed limit (70km/h)   						|
| 0.013              	| Speed limit (50km/h) 							|
| 0.011  	   		   	| Speed limit (30km/h)          	 			|
| 0.00966    		 	| Bumpy road                					|

The model achieved the correct prediction for the third image with a probability of 85%

| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.84              	| No entry               						| 
| 0.078       			| Keep right           							|
| 0.017  	   		   	|  No passing            						|
| 0.0092              	| Turn left ahead    							|
| 0.0085  	   		   	|  Beware of ice/snow            	 			|
| 0.0069    		 	| Dangerous curve to the right 					|


| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.80              	| Road work              						| 
| 0.062       			| Bicycles crossing 							|
| 0.043  	   		   	|  Bumpy road           						|
| 0.029              	| Road narrows on the right 					|
| 0.011  	   		   	|  Traffic signals               	 			|
| 0.089      		 	| Double curve               					|

The model correctly predicted the fifth image with a probabilitiy of 80%. And for the sixth image

| Probabilities    		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.33              	| No entry              						| 
| 0.15       			| Stop              							|
| 0.089  	   		   	|  Speed limit (30km/h)   						|
| 0.071              	| Yield             							|
| 0.042  	   		   	|  Roundabout mandatory         	 			|
| 0.037     		 	| Traffic signals             					|