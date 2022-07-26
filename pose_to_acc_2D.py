import scipy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from mpl_toolkits import mplot3d
from skimage.draw import disk

def skeleton(x, y, z):
    " Definition of skeleton edges "
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR, x_elbowL, x_hipR, x_hipL, \
    x_handR, x_handL, x_kneeR, x_kneeL, x_footR, x_footL = x[0], x[1], x[2], x[3], \
                                                           x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
    y_head, y_shoulderR, y_shoulderL, y_elbowR, y_elbowL, y_hipR, y_hipL, \
    y_handR, y_handL, y_kneeR, y_kneeL, y_footR, y_footL = y[0], y[1], y[2], y[3], \
                                                           y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12]
    z_head, z_shoulderR, z_shoulderL, z_elbowR, z_elbowL, z_hipR, z_hipL, \
    z_handR, z_handL, z_kneeR, z_kneeL, z_footR, z_footL = z[0], z[1], z[2], z[3], \
                                                           z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12]
    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 3, 2))
    skeleton[0, :, :] = [[x_head, x_shoulderR], [y_head, y_shoulderR], [z_head, z_shoulderR]]
    skeleton[1, :, :] = [[x_head, x_shoulderL], [y_head, y_shoulderL], [z_head, z_shoulderL]]
    skeleton[2, :, :] = [[x_elbowR, x_shoulderR], [y_elbowR, y_shoulderR], [z_elbowR, z_shoulderR]]
    skeleton[3, :, :] = [[x_elbowL, x_shoulderL], [y_elbowL, y_shoulderL], [z_elbowL, z_shoulderL]]
    skeleton[4, :, :] = [[x_elbowR, x_handR], [y_elbowR, y_handR], [z_elbowR, z_handR]]
    skeleton[5, :, :] = [[x_elbowL, x_handL], [y_elbowL, y_handL], [z_elbowL, z_handL]]
    skeleton[6, :, :] = [[x_hipR, x_shoulderR], [y_hipR, y_shoulderR], [z_hipR, z_shoulderR]]
    skeleton[7, :, :] = [[x_hipL, x_shoulderL], [y_hipL, y_shoulderL], [z_hipL, z_shoulderL]]
    skeleton[8, :, :] = [[x_hipR, x_kneeR], [y_hipR, y_kneeR], [z_hipR, z_kneeR]]
    skeleton[9, :, :] = [[x_hipL, x_kneeL], [y_hipL, y_kneeL], [z_hipL, z_kneeL]]
    skeleton[10, :, :] = [[x_footR, x_kneeR], [y_footR, y_kneeR], [z_footR, z_kneeR]]
    skeleton[11, :, :] = [[x_footL, x_kneeL], [y_footL, y_kneeL], [z_footL, z_kneeL]]
    skeleton[12, :, :] = [[x_shoulderR, x_shoulderL], [y_shoulderR, y_shoulderL], [z_shoulderR, z_shoulderL]]
    skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL], [z_hipR, z_hipL]]
    return skeleton


def plotSingle3Dframe(y_true_pred, key_l, inter_pts , t, acc_field1, acc_field2, acc_field3, acc_field4, heatmap, c='black' , limits=[[-1, 1], [-500, 500], [0, 1500]], plot_lines=True):
    " 3D plot of single frame. Can be both label or prediction "

    # img1 = np.zeros((260, 346))
    # img2 = np.zeros((260, 346))
    # img3 = np.zeros((260, 346))
    # img4 = np.zeros((260, 346))
    img1 = np.zeros((2 * 260, 2 * 346, 3))

    for i in range(y_true_pred.shape[0]):

        heatmap_i= heatmap[i][0] / 255.0

        color = (heatmap_i[2], heatmap_i[1], heatmap_i[0])
        # color_255 = (int(heatmap[i][0][0]), int(heatmap[i][0][1]), int(heatmap[i][0][2]))

        color_255_1 = int(acc_field1[i])
        color_255_2 = int(acc_field2[i])
        color_255_3 = int(acc_field3[i])
        color_255_4 = int(acc_field4[i])

        x = y_true_pred[i, 0]
        y = y_true_pred[i, 1]
        z = y_true_pred[i, 2]
        hom_coord = np.matrix([x, y, z, 1])
        coord_pix_all_cam2_homog = np.matmul(P_mat_cam3, hom_coord.T)
        coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
        u = coord_pix_all_cam2_homog_norm[0]
        v = 260 - coord_pix_all_cam2_homog_norm[1]
        # # cv2.circle(img1, (int(u),int(v)), 3, color_255_1, 5)
        # rr, cc = disk((int(v), int(u)), radius=5, shape=img1.shape)
        # img1[rr, cc] = acc_field1[i]
        # coord_pix_all_cam2_homog = np.matmul(P_mat_cam2, hom_coord.T)
        # coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
        # u = coord_pix_all_cam2_homog_norm[0]
        # v = 260 - coord_pix_all_cam2_homog_norm[1]
        # # cv2.circle(img2, (int(u),int(v)), 3, color_255_2,5)
        # rr, cc = disk((int(v), int(u)), radius=5, shape=img2.shape)
        # img2[rr, cc] = acc_field2[i]
        # coord_pix_all_cam2_homog = np.matmul(P_mat_cam3, hom_coord.T)
        # coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
        # u = coord_pix_all_cam2_homog_norm[0]
        # v = 260 - coord_pix_all_cam2_homog_norm[1]
        # # cv2.circle(img3, (int(u),int(v)), 3, color_255_3,5)
        # rr, cc = disk((int(v), int(u)), radius=5, shape=img3.shape)
        # img3[rr, cc] = acc_field3[i]
        # coord_pix_all_cam2_homog = np.matmul(P_mat_cam4, hom_coord.T)
        # coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
        # u = coord_pix_all_cam2_homog_norm[0]
        # v = 260 - coord_pix_all_cam2_homog_norm[1]
        # # cv2.circle(img4, (int(u),int(v)), 3, color_255_4, 5)
        # rr, cc = disk((int(v), int(u)), radius=5, shape=img4.shape)
        # img4[rr, cc] = acc_field4[i]
        if t ==0:
            continue
        if i==0:
            ax1.scatter(t, acc_field3[0], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[0,t] = acc_field3[0]
        if i==1:
            ax2.scatter(t, acc_field3[1], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[1, t] = acc_field3[1]
        if i == 2:
            ax3.scatter(t, acc_field3[2], s=5, color='r', marker='o')
            cv2.circle(img1, (2 * int(u), 2 * int(v)), 3, (255, 255, 255), 5)
            acceleration[2, t] = acc_field3[2]
        if i==3:
            ax4.scatter(t, acc_field3[3], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[3, t] = acc_field3[3]
        if i == 4:
            ax5.scatter(t, acc_field3[4], s=5, color='r', marker='o')
            cv2.circle(img1, (2 * int(u), 2 * int(v)), 3, (255, 255, 255), 5)
            acceleration[4, t] = acc_field3[4]
        if i==5:
            ax6.scatter(t, acc_field3[5], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[5, t] = acc_field3[5]
        if i==6:
            ax7.scatter(t, acc_field3[6], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[6, t] = acc_field3[6]
        if i==7:
            ax8.scatter(t, acc_field3[7], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[7, t] = acc_field3[7]
        if i==8:
            ax9.scatter(t, acc_field3[8], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[8, t] = acc_field3[8]
        if i==9:
            ax10.scatter(t, acc_field3[9], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[9, t] = acc_field3[9]
        if i==10:
            ax11.scatter(t, acc_field3[10], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[10, t] = acc_field3[10]
        if i==11:
            ax12.scatter(t, acc_field3[11], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[11, t] = acc_field3[11]
        if i==12:
            ax13.scatter(t, acc_field3[12], s=5, color='r', marker='o')
            cv2.circle(img1, (2*int(u), 2*int(v)), 3, (255, 255, 255), 5)
            acceleration[12, t] = acc_field3[12]

    # plot skeleton

    # for i in range(13, len(key_l)):
    #
    #     heatmap_i = heatmap[i][0] / 255.0
    #
    #     color = (heatmap_i[2], heatmap_i[1], heatmap_i[0])
    #     # color_255 = (int(heatmap[i][0][0]), int(heatmap[i][0][1]), int(heatmap[i][0][2]))
    #
    #     color_255_1 = int(acc_field1[i])
    #     color_255_2 = int(acc_field2[i])
    #     color_255_3 = int(acc_field3[i])
    #     color_255_4 = int(acc_field4[i])
    #
    #     name= key_l[i]
    #
    #     x = inter_pts[name][t][0]
    #     y = inter_pts[name][t][1]
    #     z = inter_pts[name][t][2]
    #     hom_coord = np.matrix([x, y, z, 1])
    #     coord_pix_all_cam2_homog = np.matmul(P_mat_cam1, hom_coord.T)
    #     coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
    #     u = coord_pix_all_cam2_homog_norm[0]
    #     v = 260 - coord_pix_all_cam2_homog_norm[1]
    #     # cv2.circle(img1, (int(u),int(v)), 3, color_255_1, 5)
    #     rr, cc = disk((int(v), int(u)), radius=5, shape=img1.shape)
    #     img1[rr, cc] = acc_field1[i]
    #     coord_pix_all_cam2_homog = np.matmul(P_mat_cam2, hom_coord.T)
    #     coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
    #     u = coord_pix_all_cam2_homog_norm[0]
    #     v = 260 - coord_pix_all_cam2_homog_norm[1]
    #     # cv2.circle(img2, (int(u),int(v)), 3, color_255_2,5)
    #     rr, cc = disk((int(v), int(u)), radius=5, shape=img2.shape)
    #     img2[rr, cc] = acc_field2[i]
    #     coord_pix_all_cam2_homog = np.matmul(P_mat_cam3, hom_coord.T)
    #     coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
    #     u = coord_pix_all_cam2_homog_norm[0]
    #     v = 260 - coord_pix_all_cam2_homog_norm[1]
    #     # cv2.circle(img3, (int(u),int(v)), 3, color_255_3,5)
    #     rr, cc = disk((int(v), int(u)), radius=5, shape=img3.shape)
    #     img3[rr, cc] = acc_field3[i]
    #     coord_pix_all_cam2_homog = np.matmul(P_mat_cam4, hom_coord.T)
    #     coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]
    #     u = coord_pix_all_cam2_homog_norm[0]
    #     v = 260 - coord_pix_all_cam2_homog_norm[1]
    #     # cv2.circle(img4, (int(u),int(v)), 3, color_255_4, 5)
    #     rr, cc = disk((int(v), int(u)), radius=5, shape=img4.shape)
    #     img4[rr, cc] = acc_field4[i]
    #     # ax.scatter(x, y, z, zdir='z', s=50, color=color, marker='o', depthshade=True)

    # plt.pause(0.00001)

    # img1 = cv2.GaussianBlur(img2, (51, 51), 0)
    # img1 = img1.astype(np.uint8)
    # img1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)

    # img2 = cv2.GaussianBlur(img2, (51, 51), 0)
    # img2 = img2.astype(np.uint8)
    # img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)
    #
    # img3 = cv2.GaussianBlur(img3, (51, 51), 0)
    # img3 = img3.astype(np.uint8)
    # img3 = cv2.applyColorMap(img3, cv2.COLORMAP_JET)
    #
    # img4 = cv2.GaussianBlur(img4, (51, 51), 0)
    # img4 = img4.astype(np.uint8)
    # img4 = cv2.applyColorMap(img4, cv2.COLORMAP_JET)

    # if t==0:
    #     cv2.imshow("Cam1", img1)
    #     cv2.waitKey(0)
    # cv2.imshow("Cam1", img1)
    # cv2.waitKey(1)

    # axarr[0, 0].imshow(img1, cmap='gray')
    # axarr[0, 1].imshow(img2, cmap='gray')
    # axarr[1, 0].imshow(img3, cmap='gray')
    # axarr[1, 1].imshow(img4, cmap='gray')

    # cv2.imshow("Cam1", img1)
    # cv2.imshow("Cam2", img2)
    # cv2.imshow("Cam3", img3)
    # cv2.imshow("Cam4", img4)

    x = y_true_pred[:, 0]
    y = y_true_pred[:, 1]
    z = y_true_pred[:, 2]

    # plt.pause(0.0001)

    # lines_skeleton = skeleton(x, y, z)
    # if plot_lines:
    #     for l in range(len(lines_skeleton)):
    #         ax.plot(lines_skeleton[l, 0, :], lines_skeleton[l, 1, :], lines_skeleton[l, 2, :], c)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # # ax.set_aspect('equal')
    # # set same scale for all the axis
    # x_limits = limits[0]
    # y_limits = limits[1]
    # z_limits = limits[2]
    # x_range = np.abs(x_limits[1] - x_limits[0])
    # x_middle = np.mean(x_limits)
    # y_range = np.abs(y_limits[1] - y_limits[0])
    # y_middle = np.mean(y_limits)
    # z_range = np.abs(z_limits[1] - z_limits[0])
    # z_middle = np.mean(z_limits)
    # # The plot bounding box is a sphere in the sense of the infinity
    # # norm, hence I call half the max range the plot radius.
    # plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    # ax.set_xlim3d([-1, 1])
    # ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    # ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    # #
    # plt.pause(0.0001)
    #
    # ax.clear()



key_l= ['head', 'shoulderR', 'shoulderL', 'elbowR', 'elbowL',
        'hipR', 'hipL', 'handR', 'handL', 'kneeR', 'kneeL', 'footR', 'footL',
        'head1', 'upper_R_arm', 'lower_R_arm', 'upper_L_arm', 'lower_L_arm',
        'mid_shoulder', 'center_chest', 'upper_L_leg', 'lower_L_leg'
        , 'upper_R_leg', 'lower_R_leg']

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')


vicon_data = glob.glob('/home/vishaal/git/prophesee_event_demo_bak/Vicon_data/S2_4_1.mat')
vicon_data = sorted(vicon_data)



P_mat_cam1 = np.load('/home/vishaal/git/prophesee_event_demo_bak/Vicon_bak/P1.npy')
P_mat_cam2 = np.load('/home/vishaal/git/prophesee_event_demo_bak/Vicon_bak/P2.npy')
P_mat_cam3 = np.load('/home/vishaal/git/prophesee_event_demo_bak/Vicon_bak/P3.npy')
P_mat_cam4 = np.load('/home/vishaal/git/prophesee_event_demo_bak/Vicon_bak/P4.npy')

# f, axarr = plt.subplots(2, 2)
fig = plt.figure()
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.1     # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 1   #
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)
ax1 = fig.add_subplot(13,1,1)
ax2 = fig.add_subplot(13,1,2)
ax3 = fig.add_subplot(13,1,3)
ax4 = fig.add_subplot(13,1,4)
ax5 = fig.add_subplot(13,1,5)
ax6 = fig.add_subplot(13,1,6)
ax7 = fig.add_subplot(13,1,7)
ax8 = fig.add_subplot(13,1,8)
ax9 = fig.add_subplot(13,1,9)
ax10 = fig.add_subplot(13,1,10)
ax11 = fig.add_subplot(13,1,11)
ax12 = fig.add_subplot(13,1,12)
ax13 = fig.add_subplot(13,1,13)
ax1.title.set_text('Head vs t')
ax2.title.set_text('shoulderR v vs t')
ax3.title.set_text('shoulderL vs t')
ax4.title.set_text('elbowR vs t')
ax5.title.set_text('elbowL vs t')
ax6.title.set_text('hipR vs t')
ax7.title.set_text('hipL vs t')
ax8.title.set_text('handR vs t')
ax9.title.set_text('handL vs t')
ax10.title.set_text('kneeR vs t')
ax11.title.set_text('kneeL vs t')
ax12.title.set_text('footR v vs t')
ax13.title.set_text('footL vs t')

acceleration = np.zeros((13,740))

for filename in vicon_data:
    mat = scipy.io.loadmat(filename)
    print(filename)
    # print(mat)
    # quit()
    #
    # print(mat['XYZPOS'][0, 0]['head'].shape)
    # quit()



    MAX_ACC_3D = 4
    MAX_ACC_2D = 0.5

    inter_pts= {}

    for i in range(13, len(key_l)):
        name= key_l[i]
        if key_l[i] == 'head1':
            inter_pts[name]= mat['XYZPOS'][0,0][key_l[0]] + mat['XYZPOS'][0,0][key_l[1]] + mat['XYZPOS'][0,0][key_l[2]]
            inter_pts[name]= inter_pts[name] / 3
        if key_l[i] == 'upper_R_arm':
            inter_pts[name] = mat['XYZPOS'][0, 0]['shoulderR'] + mat['XYZPOS'][0, 0]['elbowR']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'lower_R_arm':
            inter_pts[name] = mat['XYZPOS'][0, 0]['elbowR'] + mat['XYZPOS'][0, 0]['handR']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'upper_L_arm':
            inter_pts[name] = mat['XYZPOS'][0, 0]['shoulderL'] + mat['XYZPOS'][0, 0]['elbowL']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'lower_L_arm':
            inter_pts[name] = mat['XYZPOS'][0, 0]['elbowL'] + mat['XYZPOS'][0, 0]['handL']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'mid_shoulder':
            inter_pts[name] = mat['XYZPOS'][0, 0]['shoulderR'] + mat['XYZPOS'][0, 0]['shoulderL']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'center_chest':
            inter_pts[name] = mat['XYZPOS'][0, 0]['shoulderR'] + mat['XYZPOS'][0, 0]['shoulderL'] \
                              + mat['XYZPOS'][0, 0]['hipR'] + mat['XYZPOS'][0, 0]['hipL']
            inter_pts[name] = inter_pts[name] / 4
        if key_l[i] == 'upper_L_leg':
            inter_pts[name] = mat['XYZPOS'][0, 0]['hipL'] + mat['XYZPOS'][0, 0]['kneeL']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'lower_L_leg':
            inter_pts[name] = mat['XYZPOS'][0, 0]['kneeL'] + mat['XYZPOS'][0, 0]['footL']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'upper_R_leg':
            inter_pts[name] = mat['XYZPOS'][0, 0]['hipR'] + mat['XYZPOS'][0, 0]['kneeR']
            inter_pts[name] = inter_pts[name] / 2
        if key_l[i] == 'lower_R_leg':
            inter_pts[name] = mat['XYZPOS'][0, 0]['kneeR'] + mat['XYZPOS'][0, 0]['footR']
            inter_pts[name] = inter_pts[name] / 2


    for t in range(0, 740, 1):
        print(t)
        x_l= []
        y_l= []
        z_l= []

        acc_field = []
        acc_field1 = []
        acc_field2 = []
        acc_field3 = []
        acc_field4 = []

        for itr in range(len(key_l)):

            i= key_l[itr]

            if itr <= 12:

                hom_coord_1t = np.matrix([mat['XYZPOS'][0, 0][i][t-1][0], mat['XYZPOS'][0, 0][i][t-1][1], mat['XYZPOS'][0, 0][i][t-1][2], 1])
                hom_coord_t = np.matrix([mat['XYZPOS'][0,0][i][t][0], mat['XYZPOS'][0,0][i][t][1], mat['XYZPOS'][0,0][i][t][2], 1])
                hom_coord_t1 = np.matrix([mat['XYZPOS'][0, 0][i][t+1][0], mat['XYZPOS'][0, 0][i][t+1][1], mat['XYZPOS'][0, 0][i][t+1][2], 1])

                coord_pix_all_cam2_homog1_1t = np.matmul(P_mat_cam1, hom_coord_1t.T)
                coord_pix_all_cam2_homog2_1t = np.matmul(P_mat_cam2, hom_coord_1t.T)
                coord_pix_all_cam2_homog3_1t = np.matmul(P_mat_cam3, hom_coord_1t.T)
                coord_pix_all_cam2_homog4_1t = np.matmul(P_mat_cam4, hom_coord_1t.T)

                coord_pix_all_cam2_homog1_t = np.matmul(P_mat_cam1, hom_coord_t.T)
                coord_pix_all_cam2_homog2_t = np.matmul(P_mat_cam2, hom_coord_t.T)
                coord_pix_all_cam2_homog3_t = np.matmul(P_mat_cam3, hom_coord_t.T)
                coord_pix_all_cam2_homog4_t = np.matmul(P_mat_cam4, hom_coord_t.T)

                coord_pix_all_cam2_homog1_t1 = np.matmul(P_mat_cam1, hom_coord_t1.T)
                coord_pix_all_cam2_homog2_t1 = np.matmul(P_mat_cam2, hom_coord_t1.T)
                coord_pix_all_cam2_homog3_t1 = np.matmul(P_mat_cam3, hom_coord_t1.T)
                coord_pix_all_cam2_homog4_t1 = np.matmul(P_mat_cam4, hom_coord_t1.T)

                coord_pix_all_cam2_homog1_1t_norm = coord_pix_all_cam2_homog1_1t / coord_pix_all_cam2_homog1_1t[-1]
                coord_pix_all_cam2_homog2_1t_norm = coord_pix_all_cam2_homog2_1t / coord_pix_all_cam2_homog2_1t[-1]
                coord_pix_all_cam2_homog3_1t_norm = coord_pix_all_cam2_homog3_1t / coord_pix_all_cam2_homog3_1t[-1]
                coord_pix_all_cam2_homog4_1t_norm = coord_pix_all_cam2_homog4_1t / coord_pix_all_cam2_homog4_1t[-1]

                coord_pix_all_cam2_homog1_t_norm = coord_pix_all_cam2_homog1_t / coord_pix_all_cam2_homog1_t[-1]
                coord_pix_all_cam2_homog2_t_norm = coord_pix_all_cam2_homog2_t / coord_pix_all_cam2_homog2_t[-1]
                coord_pix_all_cam2_homog3_t_norm = coord_pix_all_cam2_homog3_t / coord_pix_all_cam2_homog3_t[-1]
                coord_pix_all_cam2_homog4_t_norm = coord_pix_all_cam2_homog4_t / coord_pix_all_cam2_homog4_t[-1]

                coord_pix_all_cam2_homog1_t1_norm = coord_pix_all_cam2_homog1_t1 / coord_pix_all_cam2_homog1_t1[-1]
                coord_pix_all_cam2_homog2_t1_norm = coord_pix_all_cam2_homog2_t1 / coord_pix_all_cam2_homog2_t1[-1]
                coord_pix_all_cam2_homog3_t1_norm = coord_pix_all_cam2_homog3_t1 / coord_pix_all_cam2_homog3_t1[-1]
                coord_pix_all_cam2_homog4_t1_norm = coord_pix_all_cam2_homog4_t1 / coord_pix_all_cam2_homog4_t1[-1]

                v_t_1_1 = coord_pix_all_cam2_homog1_t_norm - coord_pix_all_cam2_homog1_1t_norm
                v_t_2_1 = coord_pix_all_cam2_homog1_t1_norm - coord_pix_all_cam2_homog1_t_norm
                v_t_1_2 = coord_pix_all_cam2_homog2_t_norm - coord_pix_all_cam2_homog2_1t_norm
                v_t_2_2 = coord_pix_all_cam2_homog2_t1_norm - coord_pix_all_cam2_homog2_t_norm
                v_t_1_3 = coord_pix_all_cam2_homog3_t_norm - coord_pix_all_cam2_homog3_1t_norm
                v_t_2_3 = coord_pix_all_cam2_homog3_t1_norm - coord_pix_all_cam2_homog3_t_norm
                v_t_1_4 = coord_pix_all_cam2_homog4_t_norm - coord_pix_all_cam2_homog4_1t_norm
                v_t_2_4 = coord_pix_all_cam2_homog4_t1_norm - coord_pix_all_cam2_homog4_t_norm

                v_t_1= mat['XYZPOS'][0,0][i][t] - mat['XYZPOS'][0,0][i][t - 1]
                v_t_2= mat['XYZPOS'][0,0][i][t + 1] - mat['XYZPOS'][0,0][i][t]
                x_l.append(mat['XYZPOS'][0, 0][i][t][0])
                y_l.append(mat['XYZPOS'][0, 0][i][t][1])
                z_l.append(mat['XYZPOS'][0, 0][i][t][2])
            else:


                hom_coord_1t = np.matrix([inter_pts[i][t-1][0], inter_pts[i][t-1][1], inter_pts[i][t-1][2], 1])
                hom_coord_t = np.matrix([inter_pts[i][t][0], inter_pts[i][t][1], inter_pts[i][t][2], 1])
                hom_coord_t1 = np.matrix([inter_pts[i][t+1][0], inter_pts[i][t+1][1], inter_pts[i][t+1][2], 1])

                coord_pix_all_cam2_homog1_1t = np.matmul(P_mat_cam1, hom_coord_1t.T)
                coord_pix_all_cam2_homog2_1t = np.matmul(P_mat_cam2, hom_coord_1t.T)
                coord_pix_all_cam2_homog3_1t = np.matmul(P_mat_cam3, hom_coord_1t.T)
                coord_pix_all_cam2_homog4_1t = np.matmul(P_mat_cam4, hom_coord_1t.T)

                coord_pix_all_cam2_homog1_t = np.matmul(P_mat_cam1, hom_coord_t.T)
                coord_pix_all_cam2_homog2_t = np.matmul(P_mat_cam2, hom_coord_t.T)
                coord_pix_all_cam2_homog3_t = np.matmul(P_mat_cam3, hom_coord_t.T)
                coord_pix_all_cam2_homog4_t = np.matmul(P_mat_cam4, hom_coord_t.T)

                coord_pix_all_cam2_homog1_t1 = np.matmul(P_mat_cam1, hom_coord_t1.T)
                coord_pix_all_cam2_homog2_t1 = np.matmul(P_mat_cam2, hom_coord_t1.T)
                coord_pix_all_cam2_homog3_t1 = np.matmul(P_mat_cam3, hom_coord_t1.T)
                coord_pix_all_cam2_homog4_t1 = np.matmul(P_mat_cam4, hom_coord_t1.T)

                coord_pix_all_cam2_homog1_1t_norm = coord_pix_all_cam2_homog1_1t / coord_pix_all_cam2_homog1_1t[-1]
                coord_pix_all_cam2_homog2_1t_norm = coord_pix_all_cam2_homog2_1t / coord_pix_all_cam2_homog2_1t[-1]
                coord_pix_all_cam2_homog3_1t_norm = coord_pix_all_cam2_homog3_1t / coord_pix_all_cam2_homog3_1t[-1]
                coord_pix_all_cam2_homog4_1t_norm = coord_pix_all_cam2_homog4_1t / coord_pix_all_cam2_homog4_1t[-1]

                coord_pix_all_cam2_homog1_t_norm = coord_pix_all_cam2_homog1_t / coord_pix_all_cam2_homog1_t[-1]
                coord_pix_all_cam2_homog2_t_norm = coord_pix_all_cam2_homog2_t / coord_pix_all_cam2_homog2_t[-1]
                coord_pix_all_cam2_homog3_t_norm = coord_pix_all_cam2_homog3_t / coord_pix_all_cam2_homog3_t[-1]
                coord_pix_all_cam2_homog4_t_norm = coord_pix_all_cam2_homog4_t / coord_pix_all_cam2_homog4_t[-1]

                coord_pix_all_cam2_homog1_t1_norm = coord_pix_all_cam2_homog1_t1 / coord_pix_all_cam2_homog1_t1[-1]
                coord_pix_all_cam2_homog2_t1_norm = coord_pix_all_cam2_homog2_t1 / coord_pix_all_cam2_homog2_t1[-1]
                coord_pix_all_cam2_homog3_t1_norm = coord_pix_all_cam2_homog3_t1 / coord_pix_all_cam2_homog3_t1[-1]
                coord_pix_all_cam2_homog4_t1_norm = coord_pix_all_cam2_homog4_t1 / coord_pix_all_cam2_homog4_t1[-1]

                # print(coord_pix_all_cam2_homog1_t1_norm)

                v_t_1_1 = coord_pix_all_cam2_homog1_t_norm - coord_pix_all_cam2_homog1_1t_norm
                v_t_2_1 = coord_pix_all_cam2_homog1_t1_norm - coord_pix_all_cam2_homog1_t_norm
                v_t_1_2 = coord_pix_all_cam2_homog2_t_norm - coord_pix_all_cam2_homog2_1t_norm
                v_t_2_2 = coord_pix_all_cam2_homog2_t1_norm - coord_pix_all_cam2_homog2_t_norm
                v_t_1_3 = coord_pix_all_cam2_homog3_t_norm - coord_pix_all_cam2_homog3_1t_norm
                v_t_2_3 = coord_pix_all_cam2_homog3_t1_norm - coord_pix_all_cam2_homog3_t_norm
                v_t_1_4 = coord_pix_all_cam2_homog4_t_norm - coord_pix_all_cam2_homog4_1t_norm
                v_t_2_4 = coord_pix_all_cam2_homog4_t1_norm - coord_pix_all_cam2_homog4_t_norm

                v_t_1 = inter_pts[i][t] - inter_pts[i][t - 1]
                v_t_2 = inter_pts[i][t + 1] - inter_pts[i][t]

            a_t = v_t_2 - v_t_1
            a_t_1 = v_t_2_1 - v_t_1_1
            a_t_2 = v_t_2_2 - v_t_1_2
            a_t_3 = v_t_2_3 - v_t_1_3
            a_t_4 = v_t_2_4 - v_t_1_4
            # print(a_t)
            # hom_coord = np.matrix([a_t[0], a_t[1], a_t[2], 1])
            # coord_pix_all_cam2_homog1 = np.matmul(P_mat_cam1, hom_coord.T)
            # coord_pix_all_cam2_homog2 = np.matmul(P_mat_cam2, hom_coord.T)
            # coord_pix_all_cam2_homog3 = np.matmul(P_mat_cam3, hom_coord.T)
            # coord_pix_all_cam2_homog4 = np.matmul(P_mat_cam4, hom_coord.T)
            #
            # print("Acc: ",coord_pix_all_cam2_homog1)
            #
            # coord_pix_all_cam2_homog_norm1 = coord_pix_all_cam2_homog1 / coord_pix_all_cam2_homog1[-1]
            # coord_pix_all_cam2_homog_norm2 = coord_pix_all_cam2_homog2 / coord_pix_all_cam2_homog2[-1]
            # coord_pix_all_cam2_homog_norm3 = coord_pix_all_cam2_homog3 / coord_pix_all_cam2_homog3[-1]
            # coord_pix_all_cam2_homog_norm4 = coord_pix_all_cam2_homog4 / coord_pix_all_cam2_homog4[-1]
            #
            # norm_a = np.linalg.norm(a_t)
            # norm_a1 = np.linalg.norm(coord_pix_all_cam2_homog_norm1)
            # norm_a2 = np.linalg.norm(coord_pix_all_cam2_homog_norm2)
            # norm_a3 = np.linalg.norm(coord_pix_all_cam2_homog_norm3)
            # norm_a4 = np.linalg.norm(coord_pix_all_cam2_homog_norm4)

            # print("Cam1: ", a_t_1)
            # print("Cam2: ", a_t_2)
            # print("Cam3: ", a_t_3)
            # print("Cam4: ", a_t_4)

            norm_a = np.linalg.norm(a_t)
            norm_a1 = np.linalg.norm(a_t_1)
            norm_a2 = np.linalg.norm(a_t_2)
            norm_a3 = np.linalg.norm(a_t_3)
            norm_a4 = np.linalg.norm(a_t_4)

            acc_field.append(norm_a)
            acc_field1.append(norm_a1)
            acc_field2.append(norm_a2)
            acc_field3.append(norm_a3)
            acc_field4.append(norm_a4)

        acc_field = np.array(acc_field)
        acc_field1 = np.array(acc_field1)
        acc_field2 = np.array(acc_field2)
        acc_field3 = np.array(acc_field3)
        acc_field4 = np.array(acc_field4)

        # print(np.max(acc_field))

        # acc_field = acc_field * 255 / MAX_ACC_3D
        # acc_field1 = acc_field1 * 255 / MAX_ACC_2D
        # acc_field2 = acc_field2 * 255 / MAX_ACC_2D
        # acc_field3 = acc_field3 * 255 / MAX_ACC_2D
        # acc_field4 = acc_field4 * 255 / MAX_ACC_2D

        acc_field = np.int0(acc_field).astype(np.uint8)

        heatmap = cv2.applyColorMap(acc_field, cv2.COLORMAP_JET)

        x_l= np.array(x_l).reshape(-1, 1)
        y_l = np.array(y_l).reshape(-1, 1)
        z_l = np.array(z_l).reshape(-1, 1)


        human_t= np.hstack((x_l, y_l))
        human_t= np.hstack((human_t, z_l))

        plotSingle3Dframe(human_t, key_l, inter_pts, t, acc_field1, acc_field2, acc_field3, acc_field4, heatmap)
    plt.show()
    np.save('/home/vishaal/git/prophesee_event_demo_bak/acceleration/acc_running',acceleration)

