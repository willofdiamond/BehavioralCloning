import csv
import numpy as np
import cv2
import sys
import keras as ks

lines = []
file_directory = "/Users/hemanth/Udacity/behavioralData/data_images/"
with open(file_directory+"driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines.append(row)
        itr+=1

print("hi")

Split_data=0.8
np.random.shuffle(lines)
train_lines = lines[:int(0.8*len(lines))]
test_lines = lines[int(0.8*len(lines)):]

def data_generation(file_directory,train_lines):
    correction=0.2
    print("in function")

    for itr in range(len(train_lines)):
        image_data=[]
        steering_data=[]
        image=[]
        image_left=[]
        image_right=[]
        steering=[]
        steering_left=[]
        steering_right=[]
        steering_flipped=[]
        print("Iteration is : ",itr,train_lines[itr][0:3])
        if(not '' in train_lines[itr]):
            try:
                image = cv2.imread(train_lines[itr][0])
                #print(image.shape)
                try:
                    image_flipped=np.fliplr(image)
                except:
                    print("flipping of image failed: checking image shape")
                    print("Image not read"+"Image path:",train_lines[itr])
                    print(image.shape)
                    break
            except:
                print("Image not read"+"Image path:"+train_lines[itr][0])
            try:
                image_left = cv2.imread(train_lines[itr][1])
            except:
                print("Image not read"+"Image path:"+train_lines[itr][1])
                break
            try:
                image = cv2.imread(train_lines[itr][2])
            except:
                print("Image not read"+"Image path:"+train_lines[itr][2])
                break
            try:
                steering=float(train_lines[itr][3])
                steering_flipped=-1*steering
            except:
                print("steering value failed:  ")
                print(train_lines[itr])
                print(train_lines[itr][3])
                break
            steering_left = steering+correction
            steering_right = steering-correction
            image_data.extend([image,image_flipped,image_left,image_right])
            steering_data.extend([steering,steering_flipped,steering_left,steering_right])
            if('' in [image,image_flipped,image_left,image_right] or '' in [steering,steering_flipped,steering_left,steering_right]):
                print([image,image_flipped,image_left,image_right])
                print([steering,steering_flipped,steering_left,steering_right])
                break
            if(image.shape==(160,320,3) and image_left.shape==(160,320,3) and image_right.shape==(160,320,3) and image_flipped.shape==(160,320,3)):
                print("image")
                print(image.shape)
                print("image_left")
                print(image_left.shape)
                print("image_right")
                print(image_right.shape)
                print("image_flipped")
                print(image_flipped.shape)
                break
