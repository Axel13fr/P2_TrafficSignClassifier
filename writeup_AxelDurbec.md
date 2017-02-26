#**Traffic Sign Recognition** 

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

[distrib]: ./report_img/distrib.png "Distributions"
[random_pick]: ./report_img/random_pick.png "Random Pick"
[oneperclass]: ./report_img/oneperclass.png "One Ex. Per Class"

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Axel13fr/P2_TrafficSignClassifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.

First of all, the distribution of the classes among the different sets:

![Class Distributions][distrib]

One can see that the distributions are close enough to expect no accuracy deformation due to different sampling of the classes among the sets. However, some categories are largely underrepresented, for example on the training set, some classes have 2000 samples while other have as low as 200 which is 10 times less ! It would be interested to compute the accuracy per class later and see if that correlates with a low number of samples available at training.

Let's take 4 random samples from the training set to see the images:
![Randomly picked Images][random_pick]

And now lets have a look at one picture from each class:
![One per class][oneperclass]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook. I have choosen a tensorflow implementation to use my GPU speed and easily make sure that whatever the source, as long as its 32x32 RGB, the same chain will always be applied. Another advantage is that the tf functions will randomly change the images so every epochs will generate slightly new samples !

Interestingly enough because counter intuitive at first, in his paper treating road sign classification, Yann Lecun mentions that the color information didn't improve the performance vs Grayscale, which is very interesting to reduce the number of parameters (input depth becomes 1 instead of 3), so I went for this approach. 

On the top of that, I've tried various random transformations for data augmentation, which shall help to prevent overfit and get better results on the test set. I've choosen transformations which shall not affect the meaning of the sign nor deteriorate some important characteristics (ex: horizontal flip was not good because road sign texts are not symmetric). 

However in practice, this didn't reveal any improvements at least for the ones I have tried and my choosen architecture, there must be something to do with this so more investigation would be required.

Finally, standardization is applied to make sure all images are on the same scale, this will help the weights to treat the same range of data and ensure numerical stability. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data was already split into a nice 70%/10%/20% for train/valid/test set. Furthermore, as shown in the first part of this writeup, the distribution of the different classes along the sets is the same so no performance bias shall be noticed due to a bad split.

As for Data augmentation, I treated that previously by integrating it directly into my preprocessing chain.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 GrayScale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 				|
| Convolution 3x3     	 | 1x1 stride, valid padding, outputs 3x3x20 	|
| RELU					|												|
| Max pooling	      	   | 2x2 stride,  outputs 2x2x20 				|
| Flatten	      	   | outputs 1x80 				|
| Fully connected		| 1x80, outputs 1x80|
| Fully connected		| 1x80, outputs 1xNumber Of Classes|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13th-17th cells of the ipython notebook. 

I have rate, epochs and batch size as the classical hyperparameters but I've included as well 3 others worth mentionning: the drop out rates (keep probability) applied to the convolutional layers (C1 for conv1 and C2 for conv2 and conv3) and to the fully connected layers (FC_Prob). As I will detail below, these were critical to manage to reduce the overfitting problem.

To train the model, I used the AdamOptimizer which has the advantage to take care of learning rate decaying over epochs so that helps having one less hyperparameter to tune (possibly at the price of missing some accuracy against a fine tuned gradient descent). Softmax was used to normalize the output to probabilities and one hot was used to compare to the only valid class per sample.

The training code is self explanatory, with 2 useful additions of my own: a remaining time estimator based on the average train time for 1 epoch and a small logger class to plot the accuracy rates over the epochs at the end of the training.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
