

![Alt text ](https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/biased_data_1.png)



![A representation of Octdrey Catburn](https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/biased_data_1.png)





Overview
---
In this project, deep neural networks and convolutional neural networks are used to clone the driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

A simulator provided by Udacity is used to generate training data by steering a car around a track. I had used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

Data collection strategy
---
I had quantified my data collection into 5 bins, each bin consist of a CSV file with images name, steering angle  and their corresponding images.

1. clockwise bin : Data taken by driving in clockwise track
2. counter clockwise bin : Data taken by driving in counter-clockwise track
3. Off road : Data gathered mostly from driving back to the road from the Off rode
4. Smooth curve : Data gathered while making smooth turns along the curves
5. Missed curves : Data gathered where the trained model fails

I am checking the data distribution in each bin and creating a CSV file with more balanced steering data. Later I am combining balanced CSV files from all the bin and creating a final analysis bin. This helps in creating a much balanced training dataset


70 percentage of the data is used for training while the rest is used for testing.

Data distribution histogram
---

Data preprocessing
---
* Images read by using openCV are read as BGR instead of RGB so they are first converted in to RGB
* Images are cropped to remove the lower deck of the car, trees and sky from the images.
* Images are resized to 64x64 to reduce the computation load
* making the data unbiased by analyzing the histogram of steering data

Data Augmentation:
---
Data augmentation is used to  generate more data from the known possibilities of assumed situations. Techniques used fir my project were

* Flipping the images left to right and negating the steering values
* using corrected steering values for left and right cameras




Data Generations
---
Data for models are generated as one by one and feed directly to the model to avoid usage of vast memory consumption of the models. Python  generators are of great help.

Model Selections
---
I had initially started with a single input images layer and tried out different models.




Techniques learned:
---
