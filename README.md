# Image Classfication With Long Tailed Dataset

# This Task was Given by Analytics Vidya Here is the Problem Statement

#P roblem Statement

In the insurance industry, processing claims for vehicle damage is a common and most essential task. With the advancement in AI and Computer Vision, the users can settle the claims online instantly by uploading the images of the damaged car with the insurance company.

Now, insurance companies face the constant challenge of identifying fraudulent claims. It's a common practice for the users to submit the fraudulent images as a part of the claim settlement process. This brings out the threat/challenge to the insurance companies to identify the fraudulent claims which leads to significant financial losses.

Fraudulent claims often involve exaggerating the extent of damage or submitting false claims altogether. In this problem, we will focus on the first type of problem i.e. exaggerating the extent of damage. To mitigate these losses and maintain the integrity of their operations, insurance companies must develop effective methods to flag out these claims most accurately and efficiently. 

The hackathon challenges you to develop a robust and high performance model for classifying an image of a car into different types of damages automatically with the help of computer vision techniques. By accurately identifying the damages, the insurance company can assess the legitimacy of the claim and make informed decisions regarding payouts.


# Dataset

You are provided with 3 files: 

Training set (train.zip)
Test set (test.zip)


# Training Dataset

The training set contains a diverse dataset of car images of damaged vehicles from various angles, lighting conditions along with labels indicating the specific type of damage (e.g., dents, scratches, cracks, collision damage, etc)

# Dataset Description

# train.zip contains 2 files: images folder and train.csv

images folder contains the images to be used for training the model
train.csv contains the 3 columns: image_id, filename and target class of the images present in the training dataset.

train.csv

#Column 	Description
image_id	Unique identifier of the image
filename	Filename of the image
label	
Type of the damage present in the car

1: crack
2: scratch
3: tire flat
4: dent
5: glass shatter
6: lamp broken


# Test Dataset

In the test set, you are provided with only the images and you need to predict the type of damage for each image present in the test set.

Dataset Description

test.zip contains only images folder and test.csv

images folder contains all the test images for which the prediction is to be done.
test.csv contains 2 columns: image_id and filename and you need to predict the label for each present in the test set.
 test.csv

Column

Description

image_id

