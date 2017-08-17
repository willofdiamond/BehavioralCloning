import csv
import numpy as np
import cv2
import sys
import keras as ks
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lines_list = []
steering_angle_list = []
augmented_steering_angle_list = []
train_steering_angle_array = []
test_steering_angle_array = []
correction = 0.2
file_directory = "/Users/hemanth/Udacity/behavioralData/data_images/"
#file_directory = "/Users/hemanth/Udacity/behavioralData/data_images_2/"
images_directory = "/Users/hemanth/Udacity/CarND-Behavioral-Cloning-P3/images/"

with open(file_directory+"driving_log.csv") as csvfile:
#with open(file_directory+"new_unbiased_driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines_list.append(row)
            angle=float(row[3])
            angle_flipped = -1*float(row[3])
            angle_left = angle+correction
            angle_right = angle-correction
            steering_angle_list.append(angle)
            augmented_steering_angle_list.extend([angle,angle_flipped,angle_left,angle_right])
        itr+=1


def create_histogram(angle_array,bins=200,filename="histogram.png",range1=(-1,1)):
    n,bins,patches=plt.hist(angle_array,bins,range1)
    plt.title(filename.split('.')[0])
    #plt.savefig(images_directory+filename)
    plt.show()

np.random.seed(10)
Split_data_ratio  = 0.8
biased_data_ratio = 0.2
biased_left_slab  = 0
biased_right_slab = 0.005

steering_angle_array = np.array(steering_angle_list)
#biased_index = np.where(abs(steering_angle_array)<biased_slab)
biased_index = np.where(np.array([index >= biased_left_slab and index <= biased_right_slab  for index in steering_angle_array]))
biased_index = biased_index[0]
unbiased_index = [index for index in range(len(steering_angle_array)) if index not in biased_index]
unbiased_index = np.array(unbiased_index)
biased_steering_data = steering_angle_array[biased_index]
# randomized biased index
np.random.shuffle(biased_index)
# get some percentage of the higly biased data in to the analysis data
allowed_biased_index = biased_index[[i for i in range(int(biased_data_ratio*len(biased_index)))]]
final_data_index = np.concatenate((allowed_biased_index,unbiased_index))
unbiased_steering_data = np.delete(steering_angle_array,biased_index)

 # Visualize or save histogram data
create_histogram(steering_angle_array,300,"full_data_1.png")
create_histogram(augmented_steering_angle_list,300,"Augmnented_full_data_1.png")
create_histogram(biased_steering_data,300,"biased_data_1.png")
create_histogram(unbiased_steering_data,300,"unbiased_data_1.png")
create_histogram(steering_angle_array[final_data_index],300,"final_unbiased_data.png")

lines_array = np.array(lines_list)
lines_array = lines_array[final_data_index]
np.random.shuffle(lines_array)
split_limit = int(Split_data_ratio * lines_array.shape[0])
train_lines = lines_array[:split_limit]
test_lines = lines_array[split_limit:]

# Visualize or save histogram data
train_steering_angle_array = [float(line[3]) for line in train_lines]
test_steering_angle_array = [float(line[3]) for line in test_lines]
#create_histogram(steering_angle_array,"full_data_histogram_1.png",100)
create_histogram(train_steering_angle_array,300,"train_histogram_1")
create_histogram(test_steering_angle_array,300,"test_histogram_1")

with open(file_directory+"new_unbiased_driving_log.csv",'w') as csvfile1:
    newWriter = csv.writer(csvfile1,delimiter=',')
    for row in lines_array:
        newWriter.writerow(row)
