import csv
import numpy as np
import cv2
import sys
import keras as ks
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lines = []
steering_angle_array=[]
file_directory = "/Users/hemanth/Udacity/behavioralData/dataset_3/"
with open(file_directory+"driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines.append(row)
            steering_angle_array.append(float(row[3]))
        itr+=1

print("hi")

Split_data=0.8
np.random.shuffle(lines)
train_lines = lines[:int(0.8*len(lines))]
test_lines = lines[int(0.8*len(lines)):]


'''
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
'''


# data_generation(file_directory,lines)

def create_histogram(steering_angle_array,filename="histogram.png",bins=20,range=(-1,1)):
    n,bins,patches=plt.hist(steering_angle_array,bins,range)
    #plt.show()
    plt.savefig(filename)

create_histogram(steering_angle_array,"data_histogram_1.png",40)

def trim_sky_hood(image):
    return image[50:160-20]

def resize_image(image,size=(64,64)):
    image_resize=cv2.resize(image,size,interpolation= cv2.INTER_AREA)
    return image_resize

def image_show(image,name="sample"):
    if ( not isinstance(name, str)):
        name ="display_name"
    cv2.imshow(name, image)
    cv2.waitKey(1)

def image_show_plt(image,name="sample"):
    plt.imshow(image)
    plt.show()


def save_image(image,file_directory,name="save_image.png"):
    cv2.imwrite(file_directory+name,image)

def read_steering_angle(train_lines):
    try:
        steering=float(train_lines[3])
    except:
        print("steering value failed:  ")
        print(train_lines)
        print(train_lines[3])
    return steering






def generateData(file_directory,train_lines,batch_size =1):
    correction = 0.2
    count = 0
    train_lines  = train_lines
    train_lines2 = train_lines
    train_lines3 = train_lines
    train_lines4 = train_lines
    # trying to completely randomize the training data for data augmentation as well
    np.random.shuffle(train_lines2)
    np.random.shuffle(train_lines3)
    np.random.shuffle(train_lines4)
    print("Started getting data")
    while(1):
        entire_steering_Data=[]
        np.random.shuffle(train_lines)
        for batch_itr in range(0,int(len(train_lines)/batch_size)):
            #image_data=[]
            #steering_data=[]
            if(batch_itr!=int(len(train_lines)/batch_size)-1):
                cur_itr_start = batch_itr*batch_size
                cur_itr_end = cur_itr_start+batch_size
            else:
                cur_itr_start = batch_itr*batch_size
                cur_itr_end = len(train_lines)
            for itr in range(cur_itr_start,cur_itr_end):
                #print(itr)
                image_data=[]
                steering_data=[]
                image=[]
                image_left=[]
                image_right=[]
                steering=[]
                steering_left=[]
                steering_right=[]
                steering_flipped=[]
                if(not '' in train_lines[itr][0:4]):
                    #image = cv2.imread(file_directory+train_lines[itr][0])
                    try:
                        image = cv2.imread(file_directory+'IMG/'+train_lines[itr][0].split('/')[-1])
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="read_image.png")
                        image = trim_sky_hood(image)
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="trim_sky_hood.png")
                        image = resize_image(image,(64,64))
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="resize_image.png")
                            count+=1
                        #print(image.shape)
                        if image is None:
                            print("image shown as None"+"Image path:",train_lines[itr])
                            continue
                        steering = read_steering_angle(train_lines[itr])
                        '''
                        try:
                            steering=float(train_lines[itr][3])
                        except:
                            print("steering value failed:  ")
                            print(train_lines[itr])
                            print(train_lines[itr][3])
                        '''
                    except:
                        print("Center Image not read"+"Image path:",train_lines[itr])
                    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    #cv2.imshow('image',image)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #image_left = cv2.imread(file_directory+"IMG/"+train_lines[itr][1].split('/')[-1])
                    #image_right = cv2.imread(file_directory+"IMG/"+train_lines[itr][2].split('/')[-1])
                    try:
                        image_left = cv2.imread(file_directory+'IMG/'+train_lines2[itr][1].split('/')[-1])
                        image_left = trim_sky_hood(image_left)
                        image_left = resize_image(image_left)
                        if image_left is None:
                            print("Left image read None"+"Image path:",train_lines2[itr])
                            continue
                        steering_left = read_steering_angle(train_lines2[itr])+correction
                    except:
                        print("Left image not read"+"Image path:",train_lines2[itr])
                    try:
                        image_right = cv2.imread(file_directory+'IMG/'+train_lines3[itr][2].split('/')[-1])
                        image_right = trim_sky_hood(image_right)
                        image_right = resize_image(image_right)
                        if image_right is None:
                            print("Left image not read None"+"Image path:",train_lines3[itr])
                            continue
                        steering_right = read_steering_angle(train_lines3[itr])-correction
                    except:
                        print("Right image not read"+"Image path:",train_lines3[itr])
                    #print(file_directory+"IMG/"+train_lines[itr][1].ddsplit('/')[-1])
                    try:
                        image_flipped = cv2.imread(file_directory+'IMG/'+train_lines4[itr][0].split('/')[-1])
                        image_flipped=np.fliplr(image_flipped)
                        if image_right is None:
                            print("flipped image not read None"+"Image path:",train_lines4[itr])
                            continue
                        steering_flipped = read_steering_angle(train_lines4[itr])
                    except:
                        print("flipping of image failed: checking image shape",train_lines4[itr])
                        #print(image.shape)
                    '''
                    image_data.append(image)
                    #flipping the image to train car for right curve turns
                    image_data.append(image_flipped)
                    #Add left side image
                    image_data.append(image_left)
                    #Add right side image
                    image_data.append(image_right)
                    '''
                    image_data.extend([image,image_flipped,image_left,image_right])
                    '''
                    steering_data.append(steering)
                    steering_data.append(steering_flipped)
                    steering_data.append(steering_left)
                    steering_data.append(steering_right)
                    '''
                    steering_data.extend([steering,steering_flipped,steering_left,steering_right])
                    #entire_steering_Data.extend([steering,steering_flipped,steering_left,steering_right])
                    yield (np.array(image_data),np.array(steering_data))



batch_size1 = 1

from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Activation,Reshape
from keras.layers.convolutional import Convolution2D,Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras import losses
#rows,col,channels=160,320,3
rows,col,channels=64,64,3
 # NVIDIA Architecture
model = Sequential()
#read a  3@160x320 input planes
#Crop the image to eliminate the other objects
#model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(rows,col,channels)))


# Feature Map of shape 3@90x320
#rows,col,channels=90,320,3
# Normalize the data
model.add(Lambda(lambda x: x/127.5 - 1.,output_shape=(rows,col,channels),input_shape=(rows,col,channels)))
# Trying resining to save time
#model.add(Reshape((64, 64)))
# Convolution layer 1
model.add(Conv2D(24,5,2))
# Feature map 24@43x158
model.add(Activation('relu'))
# Feature map 24@43x158
# Convolution layer 2
model.add(Conv2D(36,5,2))
# Feature Map of shape 36@20x77
model.add(Activation('relu'))
# Feature Map of shape 36@20x77
# Convolution layer 3
model.add(Conv2D(48,5,2))
# Feature Map of shape 48@8x37
model.add(Activation('relu'))
# Feature Map of shape 48@8x37
# Convolution layer 4
model.add(Conv2D(64,3,1))
# Feature Map of shape 64@6x35
model.add(Activation('relu'))
# Feature Map of shape 64@6x35
# Convolution layer 5
model.add(Conv2D(64,3,1))
# Feature Map of shape 64@4x33
model.add(Activation('relu'))
# Feature Map of shape 64@4x33
# Flatten the planes
model.add(Flatten())
# Feature Map of shape 8448
model.add(Dense(100))
# Feature Map of shape 100
model.add(Activation('relu'))
model.add(Dense(50))
# Feature Map of shape 50
model.add(Activation('relu'))
model.add(Dense(10))
# Feature Map of shape 10
model.add(Activation('relu'))
model.add(Dense(1))


#optimizing the loss functon
#adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer='adam')
print("No of training images:" +str((len(train_lines))))
model.fit_generator(generateData(file_directory,train_lines,batch_size = 1), int((4*len(train_lines))/batch_size1), 1,1,None,generateData(file_directory,test_lines,batch_size = 1),int((4*len(test_lines))/batch_size1) )

model.save('model_dataset_3.h5')
print('model saved')


'''
# Used for the testing inputdata is read properly
for i in range(int((4*len(train_lines))/batch_size1)):
    print(i)
    vales_retured=(next(generateData(file_directory,train_lines,batch_size = 1)))
    images=vales_retured[0]
    if((images[0].shape !=(64,64,3)) and (images[1].shape !=(64,64,3)) and (images[2].shape !=(64,64,3)) and (images[3].shape !=(64,64,3))):
        print(i)
        print("error")
        print(images)
        print(images.shape)
        print(vales_retured[1])
        break

'''





'''
for i,j in generateData(file_directory,lines):
    print(len(i))
    cv2.imshow("hi",i[0])
    cv2.waitKey()


'''
