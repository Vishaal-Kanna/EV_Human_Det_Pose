import os

from event_utils.lib.data_loaders.hdf5_dataset import DynamicH5Dataset
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
from os import path
from filters import bg_filter

path_to_events_file= '/home/vishaal/git/DHP19/h5_files/events.h5'

path_to_frame_txt_file= '/home/vishaal/git/DHP19/h5_files/dvs-video-frame_times.txt'

test_img_path= '/home/vishaal/git/DHP19/h5_files/imgs/'


def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The HDF5 dataset
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def binary_search_h5_timestamp(file, x, h5_len ,side='left'):
    f = file
    return binary_search_h5_dset(f['events'][:, 0], x, l=0, r= h5_len, side=side)

def read_frame_ts_from_text(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
        lines= [l.replace('\t', '') for l in lines]
        lines = [l.replace('\n', '') for l in lines]
        lines = [l.split() for l in lines]

    lines= np.array(lines[2: len(lines)]).astype('float')
    lines[:, 1] = lines[:, 1] * 1e6
    lines[:, 0]= lines[:, 0] + 1

    return lines



def read_gt_from_text(gt_txt_file, width, height):
    with open(gt_txt_file) as f:
        lines = f.readlines()
        lines= [l.replace('\t', '') for l in lines]
        lines = [l.replace('\n', '') for l in lines]
        lines = [l.split(',') for l in lines]

    lines = np.array(lines).astype(float)
    lines= np.array(lines).astype(int)

    lines[:, 2] = np.int0((lines[:, 2] / width) * 640)
    lines[:, 3] = np.int0((lines[:, 3] / height) * 480)
    lines[:, 4] = np.int0((lines[:, 4] / width) * 640)
    lines[:, 5] = np.int0((lines[:, 5] / height) * 480)



    return lines





def viz_events(events, height= 480, width= 640):
    img = np.full((height, width, 3), 0, dtype=np.uint8)
    img[events[1], events[0]] = 255 #* events[2][:, None]
    return img




def drawGT(frame, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)






# with h5py.File(h5_file, "r") as f:
#     # List all groups
#     print("Keys: %s" % f.keys())
#     a_group_key = list(f.keys())[0]
#
#     print(a_group_key)
#
#     # Get the data
#     data = f['events'][:, 0]
#
#     print(len(data))
#
#     exit(0)


def main(filename):

    dt =30000
    h5_file = '/home/vishaal/git/DHP19/PoseTrack/h5_files/'+filename+'/events.h5'


    frame_ts= read_frame_ts_from_text('/home/vishaal/git/DHP19/PoseTrack/h5_files/'+filename+'/dvs-video-frame_times.txt')


    #set up sampling factor here
    up_sampling_factor= 4

    num_iter= int(len(frame_ts) / up_sampling_factor)



    print(num_iter)


    f = h5py.File(h5_file, 'r')

    h5_len= len(f['events'][:, 0])

    ev_arr= []

    for i in tqdm(range(num_iter)):
        ts= frame_ts[up_sampling_factor * i, 1]
        # if i == 0:
        #     ts= ts + 2600
        ev_ts_s= binary_search_h5_timestamp(f, ts, h5_len)
        ev_ts_e = binary_search_h5_timestamp(f, ts + 25000, h5_len)

        cur_evs= f['events'][ev_ts_s: ev_ts_e]

        cur_evs= cur_evs[:, [1, 2, 3, 0]]

        lst=[]

        lastTimesMap = np.zeros((480, 640))
        index = np.zeros(cur_evs.shape[0])
        t = cur_evs[:,3]
        x = cur_evs[:,0]
        y = cur_evs[:,1]
        pol = cur_evs[:,2]
        for i in range(cur_evs.shape[0]):
            ts = t[i]
            xs = int(x[i])
            ys = int(y[i])

            if ys >= 480 or xs >= 640 or ys < 0 or xs < 0:
                index[i] = 1
                continue
            deltaT = ts - lastTimesMap[ys, xs]

            if deltaT > dt:
                index[i] = 1

            if xs == 1 or xs == 640 - 1 or ys == 1 or ys == 480 - 1:
                continue
            else:
                lastTimesMap[ys, xs - 1] = ts
                lastTimesMap[ys, xs + 1] = ts
                lastTimesMap[ys - 1, xs] = ts
                lastTimesMap[ys + 1, xs] = ts
                lastTimesMap[ys - 1, xs - 1] = ts
                lastTimesMap[ys + 1, xs + 1] = ts
                lastTimesMap[ys + 1, xs - 1] = ts
                lastTimesMap[ys - 1, xs + 1] = ts

        x[index == 1] = -10
        y[index == 1] = -10
        t[index == 1] = -10
        pol[index == 1] = -10

        for i in range(0, cur_evs.shape[0]):
            if index[i]==1:
                continue
            else:
                lst.append([int(x[i]), int(y[i]), int(pol[i]), t[i]])
        lst = np.array(lst)


        ev_arr.append(lst)


    ev_arr= np.array(ev_arr)

    print(ev_arr.shape)


    ev_all_save= []

    for i in range(len(ev_arr)):
        ev_all_save += list(ev_arr[i].flatten())


    ev_all_save= np.array(ev_all_save)
    ev_all_save= ev_all_save.reshape((-1, 4))


    np.save('/home/vishaal/git/DHP19/PoseTrack/npy_files/train/'+filename+'.npy', ev_arr)
    # np.save('ev_v2e_all.npy', ev_all_save)


def test(filename, offset, w, h):
    ev_arr= np.load('/home/vishaal/git/DHP19/PoseTrack/npy_files/train/'+filename, allow_pickle= True)

    # ev_all= np.load('ev_v2e_all.npy', allow_pickle= True)

    # save video in opencv
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video_path = 'ev_gen_box_video.mp4'

    # writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))


    # gt_box= read_gt_from_text(path_to_gt_txt_file, width= w, height= h)




    # img_files= sorted(os.listdir(test_img_path))[0:90]


    #The start frames
    offset_frame= offset


    print(len(ev_arr))

    name, extension = path.splitext(filename)
    os.mkdir('/home/vishaal/git/DHP19/PoseTrack/images/train/' + name)

    for i in range(len(ev_arr)):
        # frame= plt.imread(test_img_path + img_files[i])
        # frame= cv2.resize(frame, (640, 480), cv2.INTER_AREA)
        frame= viz_events(ev_arr[i].T)

        # box_indx= np.where(gt_box[:, 0] == (offset_frame + i + 1))

        # cur_gt_boxes= gt_box[box_indx][:, [2, 3, 4, 5]]

        # for j in range(len(cur_gt_boxes)):
        #     l, t, w, h =cur_gt_boxes[j]
        #     drawGT(frame, l, t, l + w, t + h)

        cv2.imwrite('/home/vishaal/git/DHP19/PoseTrack/images/train/'+name+'/'f"{i:06d}.jpg", frame)
        # writer.write(frame)
        # cv2.waitKey(50)

if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-m', type=int, required=True)
    Parser.add_argument('-offset', type=int, default=0, required=False)
    Parser.add_argument('-width', type=int,  default=640, required=False)
    Parser.add_argument('-height', type=int,  default=480, required=False)

    args = Parser.parse_args()

    if args.m == 0:
        h5_folder = '/home/vishaal/git/DHP19/PoseTrack/h5_files'
        # npy_files = sorted(npy_files)
        for filename in sorted(os.listdir(h5_folder)):
            print(filename)
            json_file = filename + '.json'
            if json_file in os.listdir('/home/vishaal/git/DHP19/PoseTrack/annotations/train'):
                main(filename)
            else:
                print(filename," annotation does does not exist")
    if args.m == 1:
        npy_folder = '/home/vishaal/git/DHP19/PoseTrack/npy_files/train/'
        # npy_files = sorted(npy_files)
        for filename in sorted(os.listdir(npy_folder)):
            print(filename)
            test(filename, args.offset, args.width, args.height)