# image_save.py
import csv
import cv2

file_directory = "/Users/hemanth/Udacity/behavioralData/1_smoothcurve/"
#file_directory = "/Users/hemanth/Udacity/behavioralData/1_left/"
#file_directory = "/Users/hemanth/Udacity/behavioralData/1_right/"
#file_directory = "/Users/hemanth/Udacity/behavioralData/1_offroad/"
#file_directory = "/Users/hemanth/Udacity/behavioralData/1_missed_curves/"


save_directory = "/Users/hemanth/Udacity/behavioralData/data_images_2/IMG/"
lines_list=[]
with open(file_directory+"new_unbiased_driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines_list.append(row)
            #steering_angle_list.append(float(row[3]))
        itr+=1

def save_image_to_folder(file_directory_withName,save_directory):
    name = file_directory_withName.split('/')[-1]
    image = cv2.imread(file_directory+'IMG/'+name)

    cv2.imwrite(save_directory+name,image)

image_list1 = [line[0] for line in lines_list]
image_list2 = [line[1] for line in lines_list]
image_list3 = [line[2] for line in lines_list]
[save_image_to_folder(image_directory,save_directory) for  image_directory in image_list1]
[save_image_to_folder(image_directory,save_directory) for  image_directory in image_list2]
[save_image_to_folder(image_directory,save_directory) for  image_directory in image_list3]
