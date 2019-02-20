# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./examples/center_2019_02_19_14_19_09_706.jpg
[image3]: ./examples/left_2019_02_19_14_19_09_706.jpg
[image4]: ./examples/right_2019_02_19_14_19_09_706.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My final model is a Keras implementation of the Nvidia convolutional neural network designed specifically to generate steering data for self-driving cars based on camera inputs. See "Final model architecture" below for a description of the layers. 

#### 2. Attempts to reduce overfitting in the model

I split the data into training and validation sets to diagnose overfitting, but when I used the fully augmented data set (described in "Creation of the Training Set" below), overfitting did not appear to be a significant problem. Loss on the validation set was comparable to loss on the test set at the end of training. Apparently, the (shuffled and augmented) training set was large enough to allow the model to generalize to the validation data as well, even without dropout layers.

I also made sure to monitor loss while the network was training to make sure validation loss was not increasing for later epochs.



#### 3. Model parameter tuning

I used an Adams optimizer, so tuning learning rate was not necessary. The one parameter I did tune was the correction angle added to (subtracted from) the driving angle to pair with an image from the left (right) camera.

After trying several outlier values, I found a range of correction angles that resulted in good driving performance. I trained the network for correction angles of 0.2, 0.25, 0.3. Training with larger correction angles resulted in snappier response to tight turns, but also a tendency to overcorrect on shallower turns, which makes sense. The model.h5 file accompanying this submission was trained with a correction angle of 0.2. Sometimes it approaches the side of the road, or sways side to side, but corrects itself robustly. I actually like the mild swaying, because it shows the car knows how to recover.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I drived 2 time on the track to gether more datas. Specially on the straight bridge with no lane I tried to keep controlling left and right even though car was go straight. Because when training time did not need to control in straight load but these no control datas caused car not to move in straight load with autonomouse driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First I implemented a cropping layer as the first layer in my network. This removed the top 70 and bottom 25 pixels from each input image before passing the image on to the convolution layers. The top  pixels tended to contain sky/trees/horizon, and the bottom 20 pixels contained the car's hood, all of which are irrelevant to steering and might confuse the model.
I then decided to augment the training dataset by additionally using images from the left and right cameras, as well as a left-right flipped version of the center camera's image.
 I implemented Python generators to serve training and validation data to model.fit_generator(). This made model.py run much faster and more smoothly. 
 
 I then implemented the Nvidia neural network architecture found here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. This network is purpose-built for end-to-end training of self-driving car steering based on input from cameras, so it is ideal for the simulator.

The only remaining step was to tune the correction applied to the angle associated with the right and left camera images, as described in "Model parameter tuning" above. I found that the trained network reliably steered the car all the way around the track for several different choices of correction angle. It was really cool to see how the choice of correction angle influenced the car's handling. As I noted earlier, training the network with high correction angles resulted in quick, sharp response to turns, but also a tendency to overcorrect. Training with smaller correction angles resulted in less swaying back and forth across the road, but also a gentler (sometimes too gentle) response to sharp turns.

#### 2. Final Model Architecture


For informational purposes, output dimensions of convolution layers are shown, with output heights computed according to out_height = ceil( ( in_height - kernel_height + 1 )/stride_height. Output widths are computed similarly.

When adding a layer, Keras automatically computes the output shape of the previous layer, so it is not necessary to compute output dimensions manually in the code.


| Layer                         |     Description                       |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                                      A
| Cropping              | Crop top 70 pixels and bottom 25 pixels; output shape = 65x320x3 |
| Normalization         | Each new pixel value = old pixel value/255 - 0.5      |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 24 output channels, output shape = 31x158x24  |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 36 output channels, output shape = 14x76x36   |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 48 output channels, output shape = 5x36x48    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 3x34x64    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 1x32x64    |
| RELU                  |                                                       |
| Flatten               | Input 1x32x64, output 2048    |
| Fully connected       | Input 2048, output 100        |
| Dropout               | Set units to zero with probability 0.5 |
| Fully connected       | Input 100, output 50          |
| Fully connected       | Input 50, output 10           |
| Fully connected       | Input 10, output 1 (labels)   |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
To save GPU time I trained my mac and uploaded the data to workspace

Here's an example image from the center camera.

![alt text][image2]

Here's an image at the same time sample from the left camera.
This image also approximates what the center camera would see if the car were too far to the left.

![alt text][image3]

Here's an image at the same time sample from the the right camera.
This image also approximates what the center camera would see if the car were too far to the right.

![alt text][image4]

Adding the left and right images to the training set paired with corrected angles should help the car recover when the center-camera's image veers too far to the left or right.

The angle associated with each flipped image is the negative of the current driving angle, because it is the angle the car should steer if the road were flipped left<->right. The track is counterclockwise, so unaugmented training data contains more left turns than right turns. Flipping the center-camera image and pairing it with a corresponding flipped angle adds more right-turn data, which should help the model generalize.

Images were read in from files, and the flipped image added, using a Python generator. The generator processed lines of the file that stored image locations along with angle data (driving_log.csv)

The data set provided 4390 samples, each of which had a path to a center, left, and right image. sklearn.model_selection.train_test_split() was used to split off 20% of the samples to use for validation. For each sample, the center-flipped image was created on the fly within the generator.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by loss :0.0042 and val_loss:0.0443. I used an adam optimizer so that manually training the learning rate wasn't necessary.
