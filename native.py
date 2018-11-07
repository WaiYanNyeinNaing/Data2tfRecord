#https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/

import os 

import glob

import random

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

images = glob.glob('datasets/test/Arkar/*.png')

# Shuffling the dataset to remove the bias - if present
random.shuffle(images)
# Creating Labels. Consider apple = 0 and orange = 1

labels = [ 0 if 'apple' in image else 1 for image in images ]
data = list(zip(images, labels))

# Ratio

data_size = len(data)
split_size = int(0.6 * data_size)

# Splitting the dataset

training_images, training_labels = zip(*data[:split_size])
testing_images, testing_labels = zip(*data[split_size:])

print(training_images.shape)
print(training_labels.shape)
