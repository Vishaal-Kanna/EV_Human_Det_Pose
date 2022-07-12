import os.path as path
import os
import json
from random import randint
# import time

import numpy as np
import matplotlib.pyplot as plt
# from IPython import display

from skimage.draw import polygon
import skimage.io as sio
from pycocotools.coco import COCO
import cv2
import glob
import zarr

posetrack_home_fp = path.expanduser('/home/vishaal/git/DHP19/PoseTrack')

# the annotations folder (must contain 'train', 'val' and 'test' subfolders)
posetrack_annotations_fp = os.path.join(posetrack_home_fp, 'annotations')

assert(os.path.exists(posetrack_home_fp))
assert(os.path.exists(posetrack_annotations_fp))

json_folder = '/home/vishaal/git/DHP19/PoseTrack/annotations/train'

for json_file in sorted(os.listdir(json_folder)):

    print(json_file)


    name_j, extension = path.splitext(json_file)

    # Read a PoseTrack sequence.
    coco = COCO(path.join(posetrack_annotations_fp, 'train/' + json_file))
    #coco = COCO(path.join(posetrack_annotations_fp, 'test/023732_mpii_test.json'))

    # or load the full database.
    # coco = COCO(path.join(posetrack_annotations_fp, 'posetrack.json'))

    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)

    # Execute this line to see all available sequence IDs.
    np.unique([img['vid_id'] for img in imgs])

    posetrack_images = []
    for img in imgs:
        if not img['is_labeled']:  # or img['vid_id'] != '000015':  # Uncomment to filter for a sequence.
            pass
        else:
            posetrack_images.append(img)

    # print(len(posetrack_images))


    def showAnns(img_name,img, anns, coco):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """

        kp_ev = ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

        from matplotlib.collections import PatchCollection
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            #         ax = plt.gca()
            #         ax.set_autoscale_on(False)
            polygons = []
            color = []
            np.random.seed(1)
            color_coeffs = np.random.random((31, 3))

            name, extension = path.splitext(img_name)
            name1 = os.path.basename(name)

            img_rgb = cv2.imread('/home/vishaal/git/DHP19/PoseTrack/rgb_images/train/' + name_j + '/' + name1 + '.jpg')
            # img = cv2.resize(img, (img_rgb.shape[1], img_rgb.shape[0]))
            w_rgb = img_rgb.shape[1]
            h_rgb = img_rgb.shape[0]
            img_rgb = cv2.resize(img_rgb, (640, 480))
            for ann_idx, ann in enumerate(anns):

                c_assoc = ann['track_id'] * 97 % 31
                c = (color_coeffs[c_assoc:c_assoc + 1, :] * 0.6 + 0.4).tolist()[0]

                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])


                    if kp_ev != coco.loadCats(ann['category_id'])[0]['keypoints']:
                        print("error")

                    x = kp[0::3]*img.shape[1]/w_rgb
                    y = kp[1::3]*img.shape[0]/h_rgb
                    v = kp[2::3]

                    if v.max() == 0:
                        continue

                    x_min = int(x[v > 0].min())
                    y_min = int(y[v > 0].min())
                    x_max = int(x[v > 0].max())
                    y_max = int(y[v > 0].max())

                    # w = int((x_max - x_min) * 1.5)
                    h = int((y_max - y_min) * 1.25)
                    w = int(h*0.66)

                    x_mid = int((x_max + x_min) * 0.5)
                    y_mid = int((y_max + y_min) * 0.5)

                    roi_resize_h = 344
                    roi_resize_w = 180
                    roi = img[y_mid - int(h / 2):y_mid + int(h / 2), x_mid - int(w / 2):x_mid + int(w / 2)]
                    # roi_rgb = img_rgb[y_mid - int(h / 2):y_mid + int(h / 2), x_mid - int(w / 2):x_mid + int(w / 2)]

                    # for sk in sks:
                    #     if np.all(v[sk] > 0):
                    #         print(x[sk], y[sk])
                    # quit()

                    if roi.shape[0]<100 or roi.shape[1]<60:
                        continue

                    if roi.size==0:
                        continue
                    roi = cv2.resize(roi, (roi_resize_w, roi_resize_h))

                    if np.mean(roi)<15 or np.mean(roi)>70:
                        continue

                    heatmap = np.zeros((roi.shape[0], roi.shape[1], 17))
                    for i in range(0, x.shape[0]):
                        if v[i]==0:
                            continue
                        if int((y[i] - y_mid + int(h / 2))*roi_resize_h/h)>=344 or int((y[i] - y_mid + int(h / 2))*roi_resize_h/h)<0 or int((x[i] - x_mid + int(w / 2))*roi_resize_w/w)>=180 or int((x[i] - x_mid + int(w / 2))*roi_resize_w/w)<0:
                            continue
                        heatmap[int((y[i] - y_mid + int(h / 2))*roi_resize_h/h), int((x[i] - x_mid + int(w / 2))*roi_resize_w/w), i] = 1

                        # cv2.circle(heatmap, (int(x[i] - x_mid + int(w / 2)), int(y[i] - y_mid + int(h / 2))), 5, (0,0,255), 10)  # ,'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                        heatmap[:,:,i] = cv2.GaussianBlur(heatmap[:,:,i], (0, 0), 6)
                        heatmap[:, :, i] = heatmap[:, :, i]/heatmap[:, :, i].max()
                        # print(heatmap[:, :, i].max())

                    # vis = np.sum(heatmap,axis=2)
                    # print(vis.max())
                    # # quit()
                    # # print(vis.shape)
                    # # quit()
                    # cv2.imshow("image", roi)
                    # cv2.imshow("vis",vis)
                    # cv2.waitKey(0)

                    # added = cv2.addWeighted(roi, 0.4, heatmap, 0.5, 0)
                    # # cv2.imshow("Image", roi)
                    # # cv2.imshow("heatmap", heatmap)
                    # cv2.imshow("Overlayed Heatmap", added)

                    # roi = cv2.resize(roi,(344,260))
                    # heatmap = heatmap.reshape(heatmap.shape[0], -1)
                    # print(heatmap.shape)
                    # quit()
                    # print(name_j+ '_frame_{}_{}'.format(name1,ann_idx),':',np.mean(roi))

                    cv2.imwrite('/home/vishaal/git/DHP19/PoseTrack/training_data/images/' + name_j+ '/'+name_j+'_frame_{}_{}.png'.format(name1,ann_idx),roi)
                    # cv2.imwrite('/home/vishaal/git/DHP19/PoseTrack/training_data/heatmap_vis/' + name_j + '_frame_{}_{}.png'.format(name1, ann_idx), vis)
                    zarr.save('/home/vishaal/git/DHP19/PoseTrack/training_data/labels/' + name_j + '_frame_{}_{}.zarr'.format(name1, ann_idx), heatmap)
                    # cv2.imwrite('/home/vishaal/git/DHP19/PoseTrack/labels/train/' + name_j + '/rgb_frame/rgb_frame_{}_{}.png'.format(name1,ann_idx),roi_rgb)
                    # cv2.imwrite('/home/vishaal/git/DHP19/PoseTrack/labels/train/' + name_j + '/heatmap/heatmap_{}_{}.png'.format(name1, ann_idx),added)
                    # cv2.waitKey(1)

        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])


    os.mkdir('/home/vishaal/git/DHP19/PoseTrack/training_data/images/' + name_j)
    # os.mkdir('/home/vishaal/git/DHP19/PoseTrack/labels/train/' + name_j + '/rgb_frame')
    # os.mkdir('/home/vishaal/git/DHP19/PoseTrack/labels/train/' + name_j + '/event_frame')
    # os.mkdir('/home/vishaal/git/DHP19/PoseTrack/labels/train/' + name_j + '/heatmap')

    for image_idx, selected_im in enumerate(posetrack_images[:len(posetrack_images)]):
        ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
        anns = coco.loadAnns(ann_ids)
        # Load image.
        img = cv2.imread(path.join(posetrack_home_fp, selected_im['file_name']))
        # print(selected_im['file_name'])
        if img is None:
            continue

        # Visualize ignore regions if present.
        # if 'ignore_regions_x' in selected_im.keys():
        #     for region_x, region_y in zip(selected_im['ignore_regions_x'], selected_im['ignore_regions_y']):
        #         rr, cc = polygon(region_y, region_x, img.shape)
        #         img[rr, cc, 1] = 128 + img[rr, cc, 1] / 2
        # Display.
        #     plt.clf()
        #     plt.axis('off')
        showAnns(selected_im['file_name'],img, anns, coco)
    #     print(heatmap.shape)
    #     cv2.imshow("heatmap",heatmap)
    #     cv2.waitKey(0)
    # Visualize keypoints.


    # If you want to save the visualizations somewhere:
    # plt.savefig("vis_{:04d}.png".format(image_idx))
    # Frame updates.
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
    #     time.sleep(1. / 10.)
    # If you want to just look at the first image, uncomment:
    # break
    # plt.close()
    # cv2.destroyAllWindows()


