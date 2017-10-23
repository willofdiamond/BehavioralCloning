
Overview
---
In this project, deep neural networks and convolutional neural networks are used to clone the driving behavior of the driver. The model has been trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle.



Data collection strategy
---
I had quantified my data collection into 5 bins, each bin consist of a CSV file with images name, steering angle  and their corresponding images.

1. clockwise directory : Data taken by driving in clockwise track
2. counter clockwise directory : Data taken by driving in counter-clockwise track
3. Off road directory : Data gathered mostly from driving back to the road from the Off rode
4. Smooth curve directory: Data gathered while making smooth turns along the curves
5. Missed curves directory: Data gathered where the trained model fails

I am checking the data distribution in each bin and creating a CSV file with more balanced steering data. Later I am combining balanced CSV files from all the bin and creating a final analysis bin. This helps in creating a much balanced training dataset


I am using 70 percentage of the data is used for training while the rest is used for testing.

Data cleanup
---
All the directories mentioned in clockwise has a csv file, images directory.Most of the data is found to be biased in the  data_distribution_check.py file is used to clean up data, this file also generate a new CSV file with all the unbiased data. All the new csv files are combined and there corresponding images are kept in a separate folder for training and validation.

#### Data distribution histograms

[Cleaned data]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/Cleaneddata.png
[uncleaned Biased Data]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/uncleanedBiasedData.png
[uncleaned data]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/uncleaneddata.png
[biased data]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/biased_data_1.png
[final unbiased]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/final_unbiased_data.png
[Data used for Analysis]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/full_data_1.png
[test data histogram]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/test_histogram_1.png
[train data histogram]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/train_histogram_1.png
[unbiased data]:https://github.com/willofdiamond/BehavioralCloning/blob/master/Images/unbiased_data_1.png

[test result video]:https://www.youtube.com/watch?v=oh_4m896i4Q&feature=youtu.be
#### parameters used to control the amount of biased data in the final selected set
biased_data_ratio = 0.2 # determines the percentage of the diased data to exist in the final dataset
biased_left_slab  = 0 # left threshold from the center of biased data
biased_right_slab = 0.005 # right threshold  from the center of biased data
Data generated from the  simulator steering angle is highly biased as visible in the below image.
[Full data histogram][uncleaned data]
![Full data histogram][uncleaned Biased Data]
Cleaned data with 20 % of the biased data is shown below
![Full data histogram][Cleaned data]






Histogram distribution of the cleaned data
![Full data histogram][Data used for Analysis]

![Biased data histogram ][biased data]

![Unbiased data histogram][unbiased data]

![Final unbiased histogram][final unbiased]

![Trained histogram ][train data histogram]

![Test histogram][test data histogram]





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
* used the images from left and right cameras of the car with a correction to the steering angle to obtain their corresponding steering angle.
* using corrected steering values for left and right cameras




Data Generations
---
Data for models are generated as one by one and feed directly to the model to avoid usage of vast memory consumption of the models. Python  generators are of great help.

Model Selections
---
I had initially started with a single input images layer and tried out different models. I had tried  NVIDIA and Lenet models. Lenet is not does not give the desired result Nvidia architecture give good result. But I had further changed the model. My current model has four convolution layers.

|Layers                | window   | strides   |features|  activation |
|:-------------:|:-------------:|:-----:|:--------:|:---------:|
| convolution layer   | 5       | 1     |    25    |     relu      |
| convolution layer   | 5       | 1     |   34     |   relu        |
| convolution layer   | 5       |2      |   46     |    relu       |
| convolution layer   | 5       |2      |   64    |    relu       |
|Dense layer                |  none      |    none   |   200     |    relu      |
|Dense layer                 |   none     |   none   |     150    |    relu       |
|Dense layer                 |    none     |  none    |   80      |    relu       |
|Dense layer                 |   none     |  none    |     10    |   relu        |








Techniques learned:
---

Test Result:
---
I am failing at the corners, lane with no white boundary and sharp curves
[link to the project video][test result video]
