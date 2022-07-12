import scipy.io
import c3d

# mat = scipy.io.loadmat("/home/vishaal/Downloads/YouTube_Pose_dataset_1.0/YouTube_Pose_dataset.mat")
with open('/home/vishaal/git/DHP19/aspset510/aspset510_v1_trainval-joints_3d/ASPset-510/trainval/joints_3d/04ac/04ac-0003.c3d', 'rb') as handle:
    reader = c3d.Reader(handle)
    print(reader)
    for i, points in enumerate(reader.read_frames()):
        print(points[0])


