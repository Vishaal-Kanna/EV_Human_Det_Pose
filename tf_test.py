import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import os
from os.path import join
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mse2D(y_true, y_pred):
    mean_over_ch = K.mean(K.square(y_pred - y_true), axis=-1)
    mean_over_w = K.mean(mean_over_ch, axis=-1)
    mean_over_h = K.mean(mean_over_w, axis=-1)
    return mean_over_h


def predict_CNN_extract_skeleton_2d(img, verbose=False):
    """ Predict with CNN, extract predicted 2D coordinates,
    and return image, 2D label and 2D prediction for imgidx sample """
    # Cropping rightmost pixels
    x_selected = img #x_h5['DVS'][imgidx:imgidx + 1, :, :344, ch_idx]
    # print(x_selected[0,56,67])
    # quit()
    pred = trained_model.predict(np.expand_dims(np.expand_dims(x_selected, axis=0), axis=-1))
    # print(pred[0].shape)
    # y_2d, gt_mask, y_heatmaps = get_2Dcoords_and_heatmaps_label(y_h5['XYZ'][imgidx], ch_idx)
    # p_coords_max = np.zeros(y_2d.shape)
    # confidence = np.zeros(y_2d.shape[0])  # confidence is not used in this example.

    # for j_idx in range(y_2d.shape[0]):
    #     pred_j_map = pred[0, :, :, j_idx]
    #     # Predicted max value for each heatmap. Keep only the first one if more are present.
    #     p_coords_max_tmp = np.argwhere(pred_j_map == np.max(pred_j_map))
    #     p_coords_max[j_idx] = p_coords_max_tmp[0]
    #     # Confidence of the joint
    #     confidence[j_idx] = np.max(pred_j_map)
    #     if verbose:
    #         print('j{} GT: {} -- PRED: {} (mask: {})'.format(j_idx, y_2d[j_idx], p_coords_max[j_idx], gt_mask[j_idx]))
    # y_2d_float = y_2d.astype(np.float)
    # # where mask is 0, set gt back to NaN
    # y_2d_float[gt_mask == 0] = np.nan
    # dist_2d = np.linalg.norm((y_2d_float - p_coords_max), axis=-1)
    # mpjpe = np.nanmean(dist_2d)
    # print('sample mpjpe: {}'.format(mpjpe))
    return x_selected, pred[0]


def plot_2d(dvs_frame,sample_pred):
    " To plot image and 2D ground truth and prediction "
    # plt.figure()
    plt.imshow(dvs_frame,cmap='gray')
    # for q in range(0,13):
    plt.imshow(np.sum(sample_pred, axis=-1),alpha=.5)

    # plt.plot(sample_gt[:, 1], sample_gt[:, 0], '.', c='red', label='gt')
    # plt.plot(sample_pred[:, 1], sample_pred[:, 0], '.', c='blue', label='pred')
    plt.show()
    # plt.legend()


# x_h5 = h5py.File('/home/vishaal/git/DHP19/S1_session4_mov1_7500events.h5', 'r')
# imgidx = 54

trained_model=load_model('/home/vishaal/git/DHP19/DHP_CNN.model', custom_objects={'mse2D': mse2D})
print(trained_model.summary())
quit()

for i in range(1,10000):
    img = cv2.imread('/home/vishaal/git/DHP19/images/img_human_occ_{}.png'.format(i),0)
    img_cam2, pred = predict_CNN_extract_skeleton_2d(img, verbose=True)

    plot_2d(img_cam2,pred)

