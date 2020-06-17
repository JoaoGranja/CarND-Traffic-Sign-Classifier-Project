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
[image5]: ./traffic-sign-images/Traffic-sign-1.png "Traffic Sign 1"
[image6]: ./traffic-sign-images/Traffic-sign-2.png "Traffic Sign 2"
[image7]: ./traffic-sign-images/Traffic-sign-3.png "Traffic Sign 3"
[image8]: ./traffic-sign-images/Traffic-sign-4.png "Traffic Sign 4"
[image9]: ./traffic-sign-images/Traffic-sign-5.png "Traffic Sign 5"

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
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x32 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32  				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x64 		    		|
| Flatten				| outputs 1600  								|
| Fully connected 		| outputs 512 									|
| RELU					|												|
| Dropout				|												|
| Fully connected 		| outputs 256 									|
| RELU					|												|
| Dropout				|												|
| Fully connected 		| outputs 43 									|
 


#### Model Training

To train the model, I created a training pipeline that uses the model to classify the traffic sign data. First I calculated the cross-entropy between the logits and labels. Then I calculate the mean of this cross-entropy to be used on the optimizer. I used the AdamOptimizer with a learning rate of 0.001 with the goal to minimize the mean of the cross-entropy.

Running over 15 times the training data set (EPOCHS = 15), I trained the model using inputs size of 128 images (BATCH_SIZE = 128). 

#### Solution Approach

The approach followed to find the best solution for this project was to take the infamous LeNet neural network architecture, discussed on previous lessons and improve the architeture changing some neural network parameters and adding more features. I decided to use this kind of neural network because it has a very simple architecture and it yielded very good results on previous lesson for the handwritten digit images classifier. Although the problem is not the same, many principles are. In fact the best way to recognize the traffic sign is to look to the shapes of the images, like on the handwritten digit images.

Using the same LeNet neural network architecture as on previous lessons, the obtained results were not satisfied. So I follow an iterative approach until find a good solution for this problem. Below I summary the steps followed:

* First I decided to change the dimensions of the LeNet layers by adding more filters to the convolution layers which improves the results. This was a very important step to take because the number of the ouptut classes for this problem is 43 instead of 10 on lesson problem, So to have better results I have to increase the number of parameters on the last fully-connected layers, which means increase the deep of the convolution layers. 
* Second step was to deal with overfitting. After the first step I achieve good results for the training set accuracy but not for the validation set acuracy which indicates overfitting issue. To deal with that I augment the training data by translating, rotating and shifting images as explained on Step 2 of this report. However that was not enough, so I add the regularization feature Dropout in each layer. This improve considerably the validation set accuracy (higher than 0.98).


My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.986
* test set accuracy of 0.909

These results were calculated on 19 and 21 cells of the notebook.

### Analyse the misclassified labels of the validation set

During the iterative approach to find a good solution for the neural network architecture, I analysed the misclassified labels of the validation set to verify what are the classes with most incorrect predictions and to check if the validation set accuracy is uniform through the classes. After this analysis, I can conclude that the neural network has different accuracies through the classes. Actually, the classes 0 and 41 just have an accuracy of 20% and 40% respectively. A possible explanation for this result can be that these classes have few training data set as other classes. 

### Step 3 - Use the model to make predictions on new images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

I decided to chose 2 pairs of similar images to verify how the model behaves. The image 1 and 3 have both a red circle with shapes inside of the circle. The image 4 and 5 have both blue circles and white signs inside of it. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing truck 		| No passing truck 	   							| 
| Turn right only		| Turn right only								|
| Ahead only			| Ahead only									|
| Pedestrians      		| Pedestrians					 				|
| 30 km/h       		| 30 km/h      							        |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60 		            | No passing truck 	   							| 
| .20		            | Turn right only								|
| .05			        | Ahead only									|
| .04      		        | Pedestrians					 				|
| .01       		    | 30 km/h      							        |

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


