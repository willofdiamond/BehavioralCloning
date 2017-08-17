

* model_Nvidia_2epoc : Trained on Nvidia Architecture for 2 epoch with out resizing the image

* model_Nvidia_5epoc : Trained on Nvidia Architecture for 5 epoch with out resizing the image

* model_Nvidia_2epoc_resized.h5 : Trained on Nvidia Architecture for 2 epoch with resizing the image

* model_dataset_6.h5 : train and test data were balanced Modified Nvidia model with resized images and 2 epochs.
Analysis: Model trained very  was very well trained. trying to align on the center of the road. Unable to perform sharp turns and deviating  the road with no lane marks.  

* model_dataset_7.h5 : train and test data were balanced Modified Nvidia model with resized images and 4 epochs. 2 epochs are enough as no significant improvements after 2 epochs.
Analysis: performing bad on right side detection model biased towards negative steering data

* model_dataset_8.h5 : train and test data were balanced Modified Nvidia model with resized images and 4epochs. angle correction increased to 0.4 from 0.2, Convolution Architecture is modified, learning rate is decreased and dropouts rate were also changed.
Analysis:

* model_dataset_9.h5  :train and test data were balanced Modified Nvidia model with resized images and 1 epochs.
Analysis:

* model_dataset_10.h5 :train and test data were balanced Modified Nvidia model with resized images and 4epochs. angle correction increased to 0.4 from 0.2, Convolution Architecture is modified, learning rate is decreased and dropouts rate were also changed.
Analysis:

* model_dataset_11.h5 :train and test data were balanced Modified Nvidia model with resized images and 4epochs. angle correction increased to 0.4 from 0.2, Convolution Architecture is modified, learning rate is decreased and dropouts rate were also changed.
Analysis:

* model_dataset_11.h5 : reduced to two convolution layers. train and test data were balanced Modified Nvidia model with resized images and 4epochs. angle correction increased to 0.4 from 0.2, Convolution Architecture is modified, learning rate is decreased and dropouts rate were also changed.
Analysis:

* model_dataset_12.h5 reduced convolution vernal size and removed most of the dropouts . changed data divide to 50%  to reduce training load and increased more central data and add more data from the missing spots
