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


[//]: # (Image References)

[image1]: ./figures/class_distribution_in_traffic_sign_dataset.png
[image2]: ./figures/example_rbg.png
[image3]: ./figures/random_sampels_from_data_augmentation.png
[image4]: ./figures/random_images_from_training_set.png
[image5]: ./figures/2-stage_ConvNet.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
In the upcoming figure you can see an exploratory visualization of the dataset. The first thing that stands out is that the size of the categorize is differnt. But it seems that the difference between the catogarize is represented in both sets.

![figure][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In this ![whitepaper][http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] there is a descirption of how to pre-process the data set.

* Every image is converted from RGB to YUV and only the Y channel is is keeped
* The Y channel is prepocessed with contrast normalization
* Global normalization with frist centering each image around tis mean vlaue and then local normalization emphasizeing edeges

Also i have generated with the Keras library additional train images. The generated training images variate through rotation, width shift and height shift from the original one. 

![original image][image2]
![data augmentation][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

At first i had ne Lenet 5 Convolutional network, but it was not so good with the validation accuracy. So i searched for better solutions and found [whitepaper][http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] from Yann Lecun. In this white paper there is a description from enhanced lenet with a preprocessing methode for the pictures and description for the convoltional network. The convolutional network have 4 layers. The first 2 layers are convolutional and after using max pool and dropout I concatinating both than the product is feeded to the other 2 fully connected layers. Between the fully connected layers there is a dropout.

![Convolutional Network][image5]



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y image   							| 
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x64 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 outputs 16x16x64 				            |
| Dropout2	            |               						        |
| Convolution 3x3	    | 3x3 stride, same padding, outputs 4x4x128     |
| RELU                  |                                               |
| Max pooling			| 2x2 outputs 4x4x128 	       				    |
| Dropout1       		|												|
| Concat				| Concatinating layer 1 and layer 2				|
| Matmul				|                                  				|
| Dropout				|                          		    			|
| Matmul				| 			                        	        |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the training i used *Adam optimizer* becauce it was shown in the previous quiz and it seems doing a great job. Due to
limited system resources i had to use a *batch size* of 128 and iteratet 5000 times each. For preventing overfitting i used 
*dropout* with the rate of 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.936
* test set accuracy of 0.889

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

My pictures are bright and might occupy a differnt range in the color space than the given pictures. So it is possible that i did not train my dataset on similar pictures. 

Here are five German traffic signs that was randomly pick from validation set:

![random pictures][image4] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy on the new traffic signs is 80%, while it was 94.2% on the test set. This is a sign of underfitting. By looking at the virtualized *class distribution in traffic sign dataset*, I think this can be addressed by using more image preprocessing techniques on the training set.

Here are the results of the prediction:

| Image			        |     Prediction	                     | 
|:---------------------:|:--------------------------------------:| 
| Priority Road         | Priority Road                          | 
| Ahead only            | Ahead only						     |
| Yield                 | Yield 					             |
| Keep right            | Keep right				 		     |
| Speed Limit (60 km/h) | Speed Limit (60 km/h)     			 |


The model was able to correctly predict 4 of the 5 traffic signs, which gives an accuracy of 80%. The softmax prediction shows that for the image *Turn left ahead* there was a chance of 34%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

All images are predicted on point but the category speed limit sign was accurate, however the limits number was predicted with lower accurace (0.48) by the mache. The reason for the low accuracy is that the speedlimit signs are looking similiar.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Priority Road   		                        | 
| 1.    				| Ahead only 							     	|
| 1.					| Yield 							            |
| 1.	      			| Keep right				 			    	|
| 0.48			        | Speed Limit (60 km/h)      	    			|
