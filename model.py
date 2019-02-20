import sklearn
import numpy as np
import csv
import matplotlib.image as mpimg


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Constants
data_path = "data/"
image_path = data_path + "IMG/"
cvs_file=data_path+'driving_log.csv'

left_image_angle_correction = 0.20
right_image_angle_correction = -0.20
lines_csv = []
processed_csv_data = []

# Reading the content of csv file
with open(cvs_file) as input:
    csv_reader = csv.reader(input)
    # Skipping the headers if exist 
    # next(csv_reader, None)
    for each_line in csv_reader:
        lines_csv.append(each_line)



# Generator for fit data
def generator(input_data, batch_size=32):

    num_samples = len(input_data)

    while True:
     # Shuffling the csv entries
        sklearn.utils.shuffle(input_data)     
        
        for offset in range(0, num_samples, batch_size):
            # Splitting the data set into required batch size
            batch_data = input_data[offset:offset + batch_size]
            image_data = []
            steering_angle = []

            # Iterating over each image in batch_data
            for each_entry in batch_data:
                center_image_path = image_path + each_entry[0].split('/')[-1]
                center_image = mpimg.imread(center_image_path)
                steering_angle_for_centre_image = float(each_entry[3])
                # Processing the center image
                if center_image is not None:
                    image_data.append(center_image)
                    steering_angle.append(steering_angle_for_centre_image)
                    # Flipping the image
                    image_data.append(np.copy( np.fliplr( center_image ) ))
                    steering_angle.append(- steering_angle_for_centre_image)

                # Processing the left image
                left_image_path = image_path + each_entry[1].split('/')[-1]
                left_image = mpimg.imread(left_image_path)
                if left_image is not None:
                    image_data.append(left_image)
                    steering_angle.append(steering_angle_for_centre_image + left_image_angle_correction)

                # Processing the right image
                right_image_path = image_path + each_entry[2].split('/')[-1]
                right_image = mpimg.imread(right_image_path)
                if right_image is not None:
                    image_data.append(right_image)
                    steering_angle.append(steering_angle_for_centre_image + right_image_angle_correction)

            # Shuffling and returning the image data back to the calling function
            yield sklearn.utils.shuffle(np.array(image_data), np.array(steering_angle))


# Split lines of driving_log.csv into training and validation samples
# 80% of the data will be used for training.
train_data, validation_data = train_test_split(lines_csv, test_size=0.2)
# Creating generator for train and validation data set
train_generator = generator(train_data,batch_size=32)
validation_generator = generator(validation_data,batch_size=32)


# Getting shape of image
first_img_path = image_path + lines_csv[0][0].split('/')[-1]
first_image = mpimg.imread(first_img_path)
print(first_image.shape);


# My final model architecture
model = Sequential()
#Normalize the data.
model.add( Lambda( lambda x: x/255. - 0.5, input_shape=(160,320,3) ))
# Crop the hood of the car and the higher parts of the images 
# which contain irrelevant sky/horizon/trees
model.add( Cropping2D( cropping=( (70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))                      
model.add(Convolution2D(64, 3, 3, activation="relu"))                        
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))                      
                     
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_data) , verbose=1, validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=3)
model.save('model.h5')
