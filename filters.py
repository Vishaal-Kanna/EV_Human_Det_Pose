import numpy as np
import cv2
import math

def bg_filter(filename):
    lines = np.loadtxt(filename)

    k=1
    dt = 30000
    # with open('/home/vishaal/git/DHP19/events_filtered.txt', 'w') as text_file:
    while k<10000000:
        print(k)
        lastTimesMap = np.zeros((480, 640))
        index = np.zeros(40000)
        t = lines[(k - 1) * 40000: k * 40000, 0]
        # print(t.shape)
        x = lines[(k - 1) * 40000: k * 40000, 1]
        y = lines[(k - 1) * 40000: k * 40000, 2]
        pol = lines[(k - 1) * 40000: k * 40000, 3]
        for i in range((k - 1) * 40000, k * 40000):
            ts = t[i-(k - 1) *40000]
            xs = int(x[i-(k - 1) *40000])
            ys = int(y[i-(k - 1) *40000])
            if ys>=480 or xs>=640 or ys<0 or xs<0:
                index[i- (k - 1) *40000] = -10
                continue
            deltaT = ts - lastTimesMap[ys, xs]

            if deltaT > dt:
                index[i-(k - 1) *40000] = 1

            if xs == 1 or xs == 640-1 or ys == 1 or ys == 480-1:
                continue
            else:
                lastTimesMap[ys, xs - 1] = ts
                lastTimesMap[ys,xs + 1] = ts
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

        for i in range(0,40000):
            lst.append([t[i],int(x[i]),int(y[i]),int(pol[i])])

        k+=1

# lines = np.loadtxt('/home/vishaal/git/DHP19/events_filtered.txt')

# k=1
# while k < 10000:
#     ev_frame = np.full((480, 640), 0, dtype=np.uint8)
#     for i in range((k-1)*20000,k*20000):
#         if int(lines[i,1])>=640 or int(lines[i,2])>=480 or int(lines[i,1])<0 or int(lines[i,2])<0 or lines[i,2]==-10:
#             continue
#         ev_frame[int(lines[i,2]), int(lines[i,1])] = 255
#
#     img = cv2.resize(ev_frame,(344,260))
#     cv2.imshow("Frame", img)
#     cv2.imwrite('/home/vishaal/git/DHP19/images/img_chair_{}.png'.format(k),img)
#     cv2.waitKey(0)
#     k+=1


