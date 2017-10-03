# **Traffic Sign Recognition**   


[//]: # (Image References)

[image1]: ./dataset_exploration.png "Exploration"
[image2]: ./class_distribution.png "Distribution"
[image3]: ./normalization.png "Normalization"
[image4]: ./augmentation.png "Augmentation"
[image5]: ./download.png "New Images"
[image6]: ./visualization.png "Visualization"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used basic numpy functions to examine the dimension of the dataset. 

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32*32
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

It's a good idea to first takae a look at the images in each class. I randomly picked one image from each unique class and display them in an array. 

![alt text][image1]

It is immediately clear to me that normalization would be necessary because there's a big difference in brightness and contrast between images. 

I plotted the number in each class to understand the distribution of training dataset. 

![alt text][image2]

The distribution of classes is quite uneven. Some classes have only around 250 pictures. This will lead to insufficient training of the neural network on those classes. Data augmentation would be necessary to help with training. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

The preprocessing step I took involves normalizing the pictures to zero mean and unit standard deviation of the mini-batch. This way of preprocessing is easy and already quite effective. By normalizing the inputs, we limit the range of input that the neural net has to learn from so it will learn much faster. This way of normalization preserves the relative characteristics in the image. Thought it may look funny to human eyes, it indeeds helps with training. I didn't turn images into grayscale images because I think color is useful in this case. While color doesn't seem to be a necessary factor to distinguish between traffic signs, color does separate traffic signs from their background. 

![alt text][image3]

Because of the class imbalance mentioned before, some classes actually have very few examples. To circumvent this problem, I applied data augmentation to the training example using imgaug library. It might make more sense to target augmentation to the classes that have the least number of examples, but it takes more time to implement. As a first trial, I applied data augmentation randomly and indiscriminately across the entire training dataset. The processes I used are : cropping, affine transformation, Gaussian blur and changing color hue. This is an example of an original image and the augmented images.  

![alt text][image4]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Dropout				| keep_prob = 0.8								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Dropout				| keep_prob = 0.8								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Dropout				| keep_prob = 0.65								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 					|
| Dropout				| keep_prob = 0.65								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256 					|
| Dropout				| keep_prob = 0.5								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x512 	|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x512 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x512 					|
| Dropout				| keep_prob = 0.5								|
| Flatten				|												|
| Fully connected		| 512 hidden units								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| 256 hidden units								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
|						|												|
|						|												|
 
I designed the architecture in the VGG style. I didn't use as many layers as this dataset is not that sophisticated. I used higher keep probability in the earlier convolution layers because otherwise the neural net will train very very slowly. 

#### 3. Describe how you trained your model. 

I used Adam optimizer to train my model as it uses adaptive learning rate and momentum. It works very nicely for most of the problem. I chose the batch size of 64 because of GPU memory constraints. The number of epochs was chose by observing the training loss and validation accuracy. I chose a number of epochs at which both numbers don't change much further. If I see training loss and validation accuracy both go down, I would terminate my training and use smaller number of epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

To get the entire work flow right, I began with a rather small convolutional neural nets with four conv layers without dropout, batch normalization or other performance enhancement components. I chose to use convolutional neural nets because it's been shown to work well with image classification. By sharing weights, conv nets greatly reduces parameter number and increases robustness. After I made sure that the first simple net works, I then added one performance enhancment component each time. This is to make troubleshooting easier. If things go wrong, I know it's the one component I added that is not right. Random shuffling of training dataset turns out to be quite important. Without it, there's wild fluctuation between each mini-batch and training was very difficult. After I successfully added random shuffling, batch normalization and dropout, I then moved on to increase the depth of the network. Given that I've already achieved good results with only four convolution layers, I didn't really go too deep in my final solution. I made my neural net very similar to VGG, but with only 8 convolutional layers and 2 fully connected layers. I tried different numbers of convolution layers and eventually settled at 8 because it gave the best results. To further improve the training, I applied data augmentation. I began with gentle augmentation, applying only cropping at 30% examples to see how it works. After I verified that it works, I increased data augmentation and also epochs of training. Finaly I was able to achieve testing accuracy of 98.3 %. 

For the hyperparameters of the neural net, I chose the batch size 64 because a larger number will yield out of memory error. I used three different keep probability for drop out because I found that applying agressive dropout at earlier convolutional layers made learning very difficult. I used Adam optimizer because it uses adaptive learning rate and also takes into account momentum. These qualities make it a superior optimizer for a lot of task with just its default parameters. I chose epoch number so that before which validation accuracy rise to a plateau. I also tuned data augmentation scheme by adding types of image manipulation and increasing probablity of manipulation. I also tried different numbers of hidden units of the fully connected layers. 

Finally, I applied label smoothing. Label smoothing was reported to boost neural network performance because it punished extreme predictions. Though I didn't find the effect to be very dramatic, it did lead to the best testing accuracy I was able to achieve. Also, it closes the gap between validation accuracy and testing accuracy. One downside I found though, is that it made training loss harder to interpret. 

My final model results were:
* training loss of : 0.87
* validation set accuracy of : 98.7 %  
* test set accuracy of : 98.5 % 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:   

![alt text][image5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction using my final model:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road    		| Slippery road									| 
| Priority road			| Priority road									|
| Children crossing		| Children crossing								|
| Roundabout mandatory	| Roundabout mandatory			 				|
| No passing			| No passing									|


It's interesting to see how this compares with the simplest model (4 conv layers) I first made. Here I showed the predicted probabilty of the five new images. The simple model achieved 80% accuracy on these new images. For the third one, it confidentily gave a wrong prediction. Here's the prediction result of the simple model on the third new image.   


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9990887    			| Road work   									| 
| 0.00068052   			| Right-of-way at the next intersection			|
| 0.00020422			| Children crossing								|
| 0.00001141   			| Beware of ice/snow			 				|
| 0.00000694			| Priority road      							|

 It is conceivable that the neural net made the mistake because the road work sign, and Right-of-way at the next intersection sign both two components in its center, just like the children crossing sign. 

Children crossing sign turns out to be quite a tricky one to classify. When I tried to improve the performance of my neural net by adding more layers, changing number of hidden units, changing data augmentation scheme, etc., I would still sometimes ended up with wrong classification result for this particular sign even when testing accrucay is near 98 %. 


### (Optional) Visualizing the Neural Network 
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image6]

Because I chose to use a small kernel size for the conv layers, it's hard to tell what each kernel is doing. I instead tried to visualize the activation of the first conv layer. It's not easy to interpret what features the convolutional layer actually picks up. The most apparent one would be edges, which can be clearly seen in FeatureMap 4, 7, 11, 13, 24, 25, 27, 39. 