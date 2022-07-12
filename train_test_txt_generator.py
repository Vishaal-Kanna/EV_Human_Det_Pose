#!/usr/bin/evn python

import glob, os

dataset_path = 'PoseTrack/training_data/images_bak'

# Percentage of images to be used for the test set
percentage_test = 1

# Create and/or truncate train.txt and test.txt
file_train = open('PoseTrack/training_data/train.txt', 'w')
file_test = open('PoseTrack/training_data/test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.png")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test+1:
        counter = 1
        file_test.write(dataset_path + "/" + title + '.png' + "\n")
    else:
        file_train.write(dataset_path + "/" + title + '.png' + "\n")
        counter = counter + 1