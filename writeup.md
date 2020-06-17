# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output-images/example.png "Traffic Sign Example"
[image2]: ./output-images/histograms.png "Histograms"
[image3]: ./output-images/preprocessed-image.png "Pre-processed image"
[image4]: ./output-images/Augmented-image.png "Augmented image"
[image5]: ./output-images/placeholder.png "Traffic Sign 2"
[image6]: ./output-images/placeholder.png "Traffic Sign 3"
[image7]: ./output-images/placeholder.png "Traffic Sign 4"
[image8]: ./output-images/placeholder.png "Traffic Sign 5"

## Writeup summary

In this report I will address all steps of this project, explaining my approach and presenting some results obtained.

---
### Step 0 - Load the data set

After downloading the data sets to the "traffic-signs-data" folder, I dowNload all training, validation and testing data set to the notebook.

### Step 1 - Data Set Summary & Exploration

#### Data Set Summary

I used the python library to calculate summary statistics of the traffic signs data set. The results obtained are:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

To visualize an example of the traffic sign image, I defined the function 'show_traffic_sign_example(X, y)' which plots an ramdom image of the data set. Here is an image example:

![alt text][image1]


I also create an image with the histograms of training, validation and testing data set to look the distribution of all 43 classes in all data set. The result is presented in the following image:

![alt text][image2]

It is possible to conclude that the distribution is not uniform through the classes of all data set. So we are towards a kind of imbalanced classification problem. Analysing the histograms image, we can also conclude that the histograms are similiar between the three data sets.

### Step 2 - Design and Test a Model Architecture

#### Pre-processing  

As a first step, I decided to create the function 'normalize_image' which normalizes a 3 color image using the formula (pixel - 128)/ 128. I normalized all data set and trained the model with that input. However the obtained results were not satisfied so I decided to follow a different approach.

I created the function 'normalize_grayscale' which converts an image to YUV color space and normalize the Y channel with Min-Max scaling to a range of [0.1, 0.9]. Training the model with this pre-processing results in better accuracy.


Here is an example of a traffic sign image before and after the pre-processing method.

![alt text][image3]


It is important to normalize the image data to center the data around zero mean and have a small variance. This process helps the neural network to learn faster and have a similiar activation behaviour when each input is multiplyed by the weights and added to biases values of the neural network.


#### Generate additional data 

After training the data set, the model starts to overfit to the training data set. So I decided to generate additional data to avoid overfitting. I build the new dataset by adding 4 transformed versions of the original training set, yielding 173995 samples in total. I follow the same process proposed in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The new samples are randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees)

I created 4 functions: 'transform_image', 'resize_image', 'rotate_image' and 'translate_image' for that purposes. A description of each function is presented below:
* 'translate_image' - randomly perturbe an image in position ([-2,2] pixels)
* 'rotate_image'    - randomly perturbe an image in rotation ([-15,+15] degrees) 
* 'resize_image'    - randomly perturbe an image in scale ([.9,1.1] ratio) 
* 'transform_image' - loop over all images of a data set and apply the previous process to each image

The reason for applying this process is that ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set.


Here is an example of an original image and an augmented image:

![alt text][image4]


#### Model Architecture


After several error trials, my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y-channel image						| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### Model Training

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### Solution Approach

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Step 3 - Use the model to make predictions on new images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


