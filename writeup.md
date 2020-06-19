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

[image1]: ./output-images/Example_traffic_sign.png "Traffic Sign Example"
[image2]: ./output-images/Histograms.png "Histograms"
[image3]: ./output-images/Preprocessed-image.png "Pre-processed image"
[image4]: ./output-images/Augmented-image.png "Augmented image"
[image5]: ./output-images/Traffic-Signs.png "Traffic Signs"
[image6]: ./output-images/BarChart-Softmax.png "Bar Chart"
[image7]: ./output-images/FeatureMap.png "Feature Map"

## Writeup summary

In this report I will address all steps of this project, explaining my approach and presenting some results obtained.

---
### Step 0 - Load the data set

After downloading the datasets to the "traffic-signs-data" folder, I load all training, validation and testing datasets to the notebook.

### Step 1 - Data Set Summary & Exploration

#### Data Set Summary

I used the python library to calculate some summary statistics of the traffic signs dataset. The results obtained are:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

To visualize an example of the traffic sign image, I defined the function 'show_traffic_sign()' which selects and plots a ramdom image of the data set. Here is an example:

![alt text][image1]


I also built an image with histograms of training, validation and testing datasets. The goal of this image is to look to the distribution of all 43 classes in all data set. The result is presented in the following image:

![alt text][image2]

It is possible to conclude that the distribution is not uniform through the classes of all data set. So we are towards a kind of imbalanced classification problem. Analysing the three histograms, we can also conclude that the data distributions are similar in the three datasets.

### Step 2 - Design and Test a Model Architecture

#### Pre-processing  

As a first step, I decided to create the function 'normalize_image' which normalizes a 3 color image using the formula (pixel - 128)/ 128. I normalized all datasets and trained the model with that normalization. However the obtained results were not satisfied so I decided to follow a different approach.

I created the function 'normalize_grayscale' which converts an image to YUV color space and take and normalize the Y channel with Min-Max scaling to a range of [0.1, 0.9]. Training the model with this pre-processing method results a much better accuracy.


Here is an example of a traffic sign image before and after the preprocessing method.

![alt text][image3]


It is important to normalize the image data to center the data around zero mean and have a small variance. This process helps the neural network to learn faster and have a similar activation behaviour when each input is multiplied by the weights and added to biases values of the neural network.


#### Generate additional data 

After training the data set, the model starts to overfit to the training dataset (Accuracy of the training dataset was higher than validation dataset). So I decided to generate additional data to avoid overfitting. I build the new dataset by adding 4 transformed versions of the original training set, yielding 173995 samples in total. I follow the same process proposed in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The new samples are randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees)

I created 4 functions: 'transform_image', 'resize_image', 'rotate_image' and 'translate_image' for that purpose. A description of each function is presented below:
* 'translate_image' - randomly perturb an image in position ([-2,2] pixels)
* 'rotate_image'    - randomly perturb an image in rotation ([-15,+15] degrees) 
* 'resize_image'    - randomly perturb an image in scale ([.9,1.1] ratio) 
* 'transform_image' - loop over all training dataset and apply the previous process to each image

The reason for applying this process is that ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set.


Here is an example of an original image and an augmented preprocessed image :

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

To train the model, I created a training pipeline that uses the model to classify the traffic sign data. First I calculated the cross-entropy between the logits and labels. Then I calculate the mean of this cross-entropy to be used on the optimizer. I used the AdamOptimizer with a learning rate of 0.001 with the goal to minimize the mean of the cross-entropy. This pipeline was built on 11th cell of the Ipython notebook 'Traffic_Sign_Classifier.ipynb'. 

Running over 15 times the training data set (EPOCHS = 15), I trained the model using input batchs with size of 128 images (BATCH_SIZE = 128). 

#### Solution Approach

The approach followed to find the best solution for this project was to take the infamous LeNet neural network architecture, discussed on previous lessons and improve the architecture changing some neural network parameters and adding more features. I decided to use this kind of neural network because it has a very simple architecture and it yielded very good results on previous lesson for the handwritten digit images classifier. Although the problem is not the same, many principles are. In fact the best way to recognize the traffic sign is to look to the shapes of the images, like on the handwritten digit images classification problem.

Using the same LeNet neural network architecture as on previous lessons, the obtained results were not satisfied. So I follow an iterative approach until find a good solution for this problem. Below I summary the steps followed:

* First I decided to change the dimensions of the LeNet layers by adding more filters to the convolution layers which improves the output results. This was a very important step because the number of the output classes for this problem is 43 instead of 10 on lesson problem. So to have better results I have to increase the number of parameters on the last fully-connected layers, which means increase the deep of the convolution filters. 
* Second step was to deal with overfitting. After the first step I achieve good results for the training set accuracy but not for the validation set accuracy which indicates overfitting issue. To deal with that I augmented the training data by translating, rotating and shifting images as explained on Step 2 of this report. However that was not enough, so I add the regularization feature Dropout in each layer. This improve considerably the validation set accuracy (higher than 0.98).


My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.984
* test set accuracy of 0.960

These results were calculated on 14th and 16th cells of the Ipython notebook 'Traffic_Sign_Classifier.ipynb'.

### Analyse the misclassified labels of the validation set

During the iterative approach to find a good solution for the neural network architecture, I analysed the misclassified labels of the validation set to verify what are the classes with most incorrect predictions and to check if the validation set accuracy is uniform through the classes. I could conclude that the neural network has different accuracies through the classes. Actually, the classes 0 and 21 have an accuracy lower than 80%. A possible explanation for this result can be that these classes have few training data set as other classes. 

### Step 3 - Use the model to make predictions on new images

#### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and load to the notebook:

![alt text][image5] 

I decided to chose 2 pairs of similar images to verify how the model behaves.  The image 1 and 2 have both a blue circle and white signs inside of it. The image 3 and 5 have both a red circle with shapes inside of the circle.

It is important to note that the image 4 has a different shape than others and all of them need to be preprocessed to resize them to 32x32.

#### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right only		| Turn right only								|
| Ahead only			| Ahead only									|
| 30 km/h       		| 30 km/h      							        |
| Pedestrians      		| Traffic signals         		 				|
| No passing truck 		| No passing truck 	   							| 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavorably to the accuracy on the test set of around 96%. This 5 traffic signs is a very small sample to make any conclusion about the accuracy of the model. However analyzing the image with the incorrect prediction (image 4), the prediction ("Traffic signals") is a traffic sign with similar aspects of the label one ("Pedestrians"). Actually both traffic signs has a red triangle with some shapes inside.  

### Step 4 - Analyze the softmax probabilities of the new images

#### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for calculating the softmax probabilities of the new images is located in the 20th cell of the Ipython notebook.

The result of the top five softmax probabilities for each image can be visualized on the following image:

![alt text][image6] 

A table of the best softmax probabilities for each image is:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99		            | Turn right only								|
| .99		            | Ahead only									|
| .99		            | 30 km/h      							        |
| .31		            | Traffic signals           	 				|
| .96		            | No passing truck 	   							| 


Except for the image 4, the model is pretty sure about its predictions (probability higher than 0.95). For the image 4, the model just has 31% sure about its prediction. 

For the images 1, 2, 3 and 5, the model predicts correctly the class with high certainty but for image 4 the prediction has lower certainty and it is incorrect. 

Analyzing the Bar chart, it is possible to verify that the model just results comparable top five softmax probabilities for the image 4 (yellow color). Actually for this image, the second high softmax probability value (0.21) is close to the first one and its prediction is the correct label. The third high softmax probability is 0.18, the forth high softmax probability is 0.16 and the fifht high softmax probability is 0.08. For the other images, the first softmax probability is much higher than the remaining ones.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

To visualize the Neural Network's state, I create a global variable "conv_1" and pass it as argument to the function 'outputFeatureMap'.

The result obtainged for the first convolution layer is:

![alt text][image7]

From the image, we can see 32 feature maps representing each convolution filter. Looking to each feature map, it is possible to note that this layer tries to capture simple shapes like circles. 


