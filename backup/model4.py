import csv
import numpy as np
import cv2
import sys
import keras as ks
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
import matplotlib.image as mpimg


lines_list = []
steering_angle_list = []
train_steering_angle_array = []
test_steering_angle_array = []

file_directory = "/Users/hemanth/Udacity/behavioralData/data_images_2/"
#file_directory = "/Users/hemanth/Udacity/data/"
images_directory = "/Users/hemanth/Udacity/BehavioralCloning/images/"

#images_directory = "/Users/hemanth/Udacity/CarND-Behavioral-Cloning-P3/images/"

#with open(file_directory+"driving_log.csv") as csvfile:
with open(file_directory+"new_unbiased_driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines_list.append(row)
            steering_angle_list.append(float(row[3]))
        itr+=1


def create_histogram(angle_array,bins=200,filename="histogram.png",range1=(-1,1)):
    n,bins,patches=plt.hist(angle_array,bins,range1)
#    plt.title(filename.split('.')[0])
#    plt.savefig(images_directory+filename)
#    plt.show()


#print("hi")
np.random.seed(10)
Split_data_ratio  = 0.70
biased_data_ratio = 1.0
biased_slab      = 0.005
batch_size1 = 1

steering_angle_array = np.array(steering_angle_list)
biased_index = np.where(abs(steering_angle_array)<biased_slab)
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

#train_steering_angle_array=list(map(lambda x: float(x[3]),train_lines))
#test_steering_angle_array=list(msplit_limitap(lambda x: float(x[3]),test_lines))
# data_generation(file_directory,lines)

def trim_sky_hood(image):
    return image[50:160-20]

def resize_image(image,size=(64,64)):
    image_resize=cv2.resize(image,size,interpolation= cv2.INTER_AREA)
    return image_resize

def image_show(image,name="sample"):
    if ( not isinstance(name, str)):
        name ="display_name"
    cv2.imshow(name, image)
    cv2.waitKey(0)

def image_show_plt(image,name="sample"):
    plt.imshow(image)
    plt.show()


def save_image(image,file_directory,name="save_image.png"):
    cv2.imwrite(file_directory+name,image)

def get_steering_angle(train_lines):
    try:
        steering=float(train_lines[3])
    except:
        print("steering value failed:  ")
        print(train_lines)
        print(train_lines[3])
    return steering

def Change_color(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def Change_colorSpace(image,color_space = "RGB"):
    img_features = np.zeros(image.shape)
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            img_features = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img_features = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img_features = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            img_features = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img_features = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: img_features = np.copy(image)

    return img_features

def preprocessing(image):
    color_space = "RGB"

#    print(image)
#    image = Change_color(image)
#    save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="model_RGB_image.png")
    image = trim_sky_hood(image)
#    save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="model_trim_image1.png")
    image = Change_colorSpace(image, color_space)
#    save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="model_HLS_image1.png")

#    image = resize_image(image,(64,64))
#    print("image size",image.shape)
    image = resize_image(image,(64,64))
#    save_image(image,"/Users/hemanth/Udacity/behsavioralData/",name="model_resize_image1.png")
#    print("image size",image.shape)
    return image




def generateData(file_directory,train_lines,batch_size =1):
    correction = 0.2
    count = 0
    #train_lines  = train_lines
    train_lines2 = train_lines
    train_lines3 = train_lines
    train_lines4 = train_lines
    # trying to completely randomize the training data for data augmentation as well
    #print("Started getting data")
    while(1):
        #entire_steering_Data=[]
        np.random.shuffle(train_lines)
        np.random.shuffle(train_lines2)
        np.random.shuffle(train_lines3)
        np.random.shuffle(train_lines4)
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
                    image = cv2.imread(file_directory+'IMG/'+train_lines[itr][0].split('/')[-1])
                    image = preprocessing(image)
                    try:
                        #image = cv2.imread(train_lines[itr][0])
#                        print("file_directory: ",file_directory+'IMG/'+train_lines[itr][0].split('/')[-1],"::",train_lines[itr][0].split('/')[-1],"::",file_directory)
                        image = cv2.imread(file_directory+'IMG/'+train_lines[itr][0].split('/')[-1])
                        image = preprocessing(image)
                        '''
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="read_image.png")
                        image = trim_sky_hood(image)
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="trim_sky_hood.png")
                        image = resize_image(image,(64,64))
                        if(count<1):
                            save_image(image,"/Users/hemanth/Udacity/behavioralData/",name="resize_image.png")
                            count+=1
                        '''
                        #print(image.shape)
                        if image is None:
                            print("image shown as None"+"Image path:",train_lines[itr])
                            continue
                        steering = get_steering_angle(train_lines[itr])
                        '''
                        try:
                            steering=float(train_lines[itr][3])
                        except:
                            print("steering value failed:  ")
                            print(train_lines[itr])
                            print(train_lines[itr][3])
                        '''
                    except:
                        print("Center Image not read "+"Image path:",train_lines[itr])
                    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    #cv2.imshow('image',image)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #image_left = cv2.imread(file_directory+"IMG/"+train_lines[itr][1].split('/')[-1])
                    #image_right = cv2.imread(file_directory+"IMG/"+train_lines[itr][2].split('/')[-1])
                    try:
                        #image_left = cv2.imread(train_lines2[itr][1])
                        image_left = cv2.imread(file_directory+'IMG/'+train_lines2[itr][1].split('/')[-1])
                        image_left = preprocessing(image_left)
                        '''
                        image_left = Change_color(image_left)
                        image_left = trim_sky_hood(image_left)
                        image_left = resize_image(image_left)
                        '''
                        if image_left is None:
                            print("Left image read None"+"Image path:",train_lines2[itr])
                            continue
                        steering_left = get_steering_angle(train_lines2[itr])+correction
                    except:
                        print("Left image not read"+"Image path:",train_lines2[itr])
                    try:
                        #image_right = cv2.imread(train_lines2[itr][2])
                        image_right = cv2.imread(file_directory+'IMG/'+train_lines3[itr][2].split('/')[-1])
                        image_right = preprocessing(image_right)
                        '''
                        image_right = Change_color(image_right)
                        image_right = trim_sky_hood(image_right)
                        image_right = resize_image(image_right)
                        '''
                        if image_right is None:
                            print("Left image not read None"+"Image path:",train_lines3[itr])
                            continue
                        steering_right = get_steering_angle(train_lines3[itr])-correction
                    except:
                        print("Right image not read"+"Image path:",train_lines3[itr])
                    #print(file_directory+"IMG/"+train_lines[itr][1].ddsplit('/')[-1])
                    try:
                        #image_flipped = cv2.imread(train_lines4[itr][0])
                        image_flipped = cv2.imread(file_directory+'IMG/'+train_lines4[itr][0].split('/')[-1])
                        image_flipped = np.fliplr(image_flipped)
                        image_flipped = preprocessing(image_flipped)
                        '''
                        image_flipped = Change_color(image_flipped)
                        image_flipped = trim_sky_hood(image_flipped)
                        image_flipped = resize_image(image_flipped)
                        '''
                        if image_flipped is None:
                            print("flipped image not read None"+"Image path:",train_lines4[itr])
                            continue
                        steering_flipped = get_steering_angle(train_lines4[itr])
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





from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Activation,Reshape,Dropout
from keras.layers.convolutional import Convolution2D,Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras import losses
#rows,col,channels=160,320,3
rows,col,channels=64,64,3
#rows,col,channels=90,320,3
 # NVIDIA Architecture
model = Sequential()
#read a  3@160x320 input planes
#Crop the image to eliminate the other objects
#model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(rows,col,channels)))


# Feature Map of shape 3@90x320
#rows,col,channels=90,320,3
# Normalize the data
model.add(Lambda(lambda x: x/127.5 - 1.,output_shape=(rows,col,channels),input_shape=(rows,col,channels)))
# Trying resizing to save time
#model.add(Reshape((64, 64)))

# Convolution layer 1
#'''
# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

    # Add a flatten layer
model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
model.add(Dense(1))
#'''
'''

model.add(Conv2D(24, 5, 2,border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(36, 5, 2,border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(48, 5, 2, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

#model.add(Dropout(0.50))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100,W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
# Add a fully connected output layer
model.add(Dense(1))

'''





#'''mode
#optimizing the loss functon
adam = ks.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer=adam)
print("No of training images:" +str((len(train_lines))))
epochs = 5
model.fit_generator(generateData(file_directory,train_lines,batch_size = batch_size1), int((4*len(train_lines))/batch_size1),epochs,1,None,generateData(file_directory,test_lines,batch_size = batch_size1),int((4*len(test_lines))/batch_size1) )

model.save('model4_dataset_1.h5')
print('model saved')

#'''
'''
# Used for the testing inputdata is read properly
batch_size1 = 1
for i in range(28398,int((4*len(train_lines))/batch_size1)):
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
