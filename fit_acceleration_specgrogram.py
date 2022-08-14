import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.fft import fftshift
from scipy.optimize import least_squares

def pos_all_plot(pos_all):
    # plt.cla()

    for i in [3]:
        plt.scatter(pos_all[i, 0], pos_all[i,1], s=5, color='r', marker='o')


    plt.axis('equal')

    plt.pause(0.01)


def body_skeleton(pos_all):
    plt.cla()

    for i in range(pos_all.shape[0]):
        plt.scatter(pos_all[i, 0], pos_all[i,1], s=5, color='r', marker='o')

    #head-lshoulder-rshoulder
    plt.plot([pos_all[0, 0], pos_all[1, 0]], [pos_all[0, 1], pos_all[1, 1]], color='black')
    plt.plot([pos_all[0, 0], pos_all[2, 0]], [pos_all[0, 1], pos_all[2, 1]], color='black')
    plt.plot([pos_all[1, 0], pos_all[2, 0]], [pos_all[1, 1], pos_all[2, 1]], color='black')

    #shoulder-hip
    plt.plot([pos_all[1, 0], pos_all[5, 0]], [pos_all[1, 1], pos_all[5, 1]], color='black')
    plt.plot([pos_all[2, 0], pos_all[6, 0]], [pos_all[2, 1], pos_all[6, 1]], color='black')
    plt.plot([pos_all[5, 0], pos_all[6, 0]], [pos_all[5, 1], pos_all[6, 1]], color='black')

    #shoulder-elbow-hand
    plt.plot([pos_all[1, 0], pos_all[3, 0],  pos_all[7, 0]], [pos_all[1, 1], pos_all[3, 1],  pos_all[7, 1]], color='black')
    plt.plot([pos_all[2, 0], pos_all[4, 0],  pos_all[8, 0]], [pos_all[2, 1], pos_all[4, 1],  pos_all[8, 1]], color='black')

    # hip-knee-feet
    plt.plot([pos_all[5, 0], pos_all[9, 0], pos_all[11, 0]], [pos_all[5, 1], pos_all[9, 1], pos_all[11, 1]],
             color='black')
    plt.plot([pos_all[6, 0], pos_all[10, 0], pos_all[12, 0]], [pos_all[6, 1], pos_all[10, 1], pos_all[12, 1]],
             color='black')

    plt.axis('equal')

    plt.pause(0.01)

def fit_acc(pos_all):
    t = np.linspace(0, 8, 9)

    acc_res= []
    for i in range(pos_all.shape[0]):
        acc_i= []
        for j in range(4, pos_all.shape[1] - 5):
            pos_x_arr= pos_all[i, j- 4: j + 5, 0]
            pos_y_arr = pos_all[i, j- 4: j + 5, 1]

            x_param= np.polyfit(t, pos_x_arr, deg= 3)
            y_param = np.polyfit(t, pos_y_arr, deg=3)

            x_poly= np.poly1d(x_param)
            y_poly = np.poly1d(y_param)

            acc_x_poly= np.polyder(x_poly, m= 2)
            acc_y_poly = np.polyder(y_poly, m=2)


            scale= 9

            offset_arr= np.linspace(-0.5, 0.5, 9)

            offset_arr= offset_arr[:-1]

            for k in range(len(offset_arr)):
             acc_i.append(np.linalg.norm([acc_x_poly(4 + offset_arr[k]), acc_y_poly(4 + offset_arr[k])]))


        acc_res.append(acc_i)

    acc_res= np.array(acc_res)

    print(acc_res.shape)

    np.save('accelaration/DHP19_fit_acc.npy', acc_res)



def fun(T, x, sr):
    # sampling rate
    sr = 2000
    # sampling interval
    ts = 1.0 / sr
    t = np.arange(0, 10, ts)


    num_repeat= (len(t) / (T * sr)) + 1

    window= int(T * sr)

    x_w= x[0:window]

    num_repeat= int(num_repeat)

    x_w= np.tile(x_w, num_repeat)

    x_w= x_w[0:len(t)]

    fig = plt.figure()
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.1  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 1  #
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)
    axs= []
    num_plot= 2
    for i in range(num_plot):
        axs.append(fig.add_subplot(num_plot, 1, i + 1))



    axs[0].plot(t, x, 'r')
    axs[1].plot(t, x_w, 'r')
    plt.show()

    corr = signal.correlate(x_w, x)
    lags = signal.correlation_lags(len(x), len(x_w))

    plt.plot(lags, corr)
    plt.show()


    return np.max(corr)










def auto_corr():
    # sampling rate
    sr = 2000
    # sampling interval
    ts = 1.0 / sr
    t = np.arange(0, 10, ts)

    freq = 2.
    x = 3 * np.sin(2 * np.pi * freq * t)

    x0= np.array([0.5])

    res_lsq = least_squares(fun, x0, args=(x, sr))

    print(res_lsq.x[0])


    # plt.figure(figsize=(8, 6))
    # plt.plot(t, x, 'r')
    # plt.ylabel('Amplitude')
    #
    # plt.show()














if __name__ == '__main__':
    pos_all= np.load('accelaration/super_points_chair.npy')

    for i in range(pos_all.shape[0]):
        for j in range(pos_all.shape[1]):
            if np.sum(pos_all[i, j]) == 0.:
                pos_all[i, j] = pos_all[i, j - 1]

    for i in range(pos_all.shape[0]):
        for j in range(pos_all.shape[1] - 1, -1, -1):
            if np.sum(pos_all[i, j]) == 0.:
                pos_all[i, j] = pos_all[i, j + 1]



    t = np.linspace(0, pos_all.shape[1], pos_all.shape[1])

    # print(pos_all.shape)





    # pos_all[:, :, 1]= 240 - pos_all[:, :, 1]

    # for t in range(pos_all.shape[1]):
    #     #
    #     #     pos_all_t= pos_all[:, t ,:]
    #     #     body_skeleton(pos_all_t)

    # for t in range(pos_all.shape[1]):
    #     pos_all_t= pos_all[:, t ,:]
    #     pos_all_plot(pos_all_t)


    fit_acc(pos_all)

    # auto_corr()




    exit(0)







    SAMPLE_RATE= 100
    N= acc_all.shape[1]



    fs= SAMPLE_RATE

    joint_name= ['head', 'shoulderR', 'shoulderL', 'elbowR', 'elbowL', 'hipR', 'hipL', 'handR', 'handL', 'kneeR', 'kneeL',
                 'footR', 'footL']


    for i in range(13):
        plt.subplot(2, 7, i + 1)
        plt.gca().set_title(joint_name[i])
        f, t, Sxx = signal.spectrogram(acc_all[i], fs)


        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.show()



