Overview
---
In this project, deep neural networks and convolutional neural networks are used to clone the driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

A simulator provided by Udacity is used to generate training data by steering a car around a track. I had used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

Data collection strategy
---
I had quantified my data collection into 5 bins

1) clockwise bin
2) counter clockwise bin
3) Off road
4) smooth curve
5) missed curves

I am checking the data distribution in each bin and creating a new csv file with unbiased data. This helps in creating a much better training dataset

* Data collected on clockwise laps up to three along the track.
* To avoid Driving left bias Counter-Clockwise driving data is also collected for up to three laps while maintaining the car on the center of the laps.
* Data of getting on to the road from off the track is also added in to the data set up to two laps.
* Data focused on smooth drive around the steep turns

70 percentage of the data is used for training a=while the rest is used for testing.

Data distribution histogram
---

Data preprocessing
---
* Images read by using openCV are read as BGR instead of RGB so they are first converted in to RGB
* Images are cropped to remove the lower deck of the car, trees and sky from the images.
* Images are resized to 64x64 to reduce the training complexity on
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
I had initially started with a single input images layer and


Techniques learned:
---
