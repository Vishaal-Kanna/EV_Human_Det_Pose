import matplotlib.pyplot as plt
import cv2
import glob

path = glob.glob('/home/vishaal/git/DHP19/PoseTrack/images/train/000027_bonn_train/*.jpg')

for file in path:
    img = cv2.imread(file)
    plt.imshow(img)
    plt.show()