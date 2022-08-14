import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.fft import fftshift



if __name__ == '__main__':
    acc_all= np.load('accelaration/DHP19_fit_acc.npy')




    # acc_all= acc_all.transpose(1, 0, 2)


    # acc_all= acc_all[:, 150:]






    t= np.linspace(0, acc_all.shape[1], acc_all.shape[1])


    # Display

    fig = plt.figure()
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.1  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 1  #
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)
    ax1 = fig.add_subplot(13, 1, 1)
    ax2 = fig.add_subplot(13, 1, 2)
    ax3 = fig.add_subplot(13, 1, 3)
    ax4 = fig.add_subplot(13, 1, 4)
    ax5 = fig.add_subplot(13, 1, 5)
    ax6 = fig.add_subplot(13, 1, 6)
    ax7 = fig.add_subplot(13, 1, 7)
    ax8 = fig.add_subplot(13, 1, 8)
    ax9 = fig.add_subplot(13, 1, 9)
    ax10 = fig.add_subplot(13, 1, 10)
    ax11 = fig.add_subplot(13, 1, 11)
    ax12 = fig.add_subplot(13, 1, 12)
    ax13 = fig.add_subplot(13, 1, 13)
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

    ax1.scatter(t, acc_all[0], s=5, color='r', marker='o')

    ax2.scatter(t, acc_all[1], s=5, color='r', marker='o')

    ax3.scatter(t, acc_all[2], s=5, color='r', marker='o')

    ax4.scatter(t, acc_all[3], s=5, color='r', marker='o')

    ax5.scatter(t, acc_all[4], s=5, color='r', marker='o')

    ax6.scatter(t, acc_all[5], s=5, color='r', marker='o')

    ax7.scatter(t, acc_all[6], s=5, color='r', marker='o')

    ax8.scatter(t, acc_all[7], s=5, color='r', marker='o')

    ax9.scatter(t, acc_all[8], s=5, color='r', marker='o')

    ax10.scatter(t, acc_all[9], s=5, color='r', marker='o')

    ax11.scatter(t, acc_all[10], s=5, color='r', marker='o')

    ax12.scatter(t, acc_all[11], s=5, color='r', marker='o')

    ax13.scatter(t, acc_all[12], s=5, color='r', marker='o')

    # plt.show()





    SAMPLE_RATE= 120
    N= acc_all.shape[1]



    fig = plt.figure()
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.1  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 1  #
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=hspace)
    ax1 = fig.add_subplot(13, 1, 1)
    ax2 = fig.add_subplot(13, 1, 2)
    ax3 = fig.add_subplot(13, 1, 3)
    ax4 = fig.add_subplot(13, 1, 4)
    ax5 = fig.add_subplot(13, 1, 5)
    ax6 = fig.add_subplot(13, 1, 6)
    ax7 = fig.add_subplot(13, 1, 7)
    ax8 = fig.add_subplot(13, 1, 8)
    ax9 = fig.add_subplot(13, 1, 9)
    ax10 = fig.add_subplot(13, 1, 10)
    ax11 = fig.add_subplot(13, 1, 11)
    ax12 = fig.add_subplot(13, 1, 12)
    ax13 = fig.add_subplot(13, 1, 13)
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

    # Note the extra 'r' at the front
    yf = rfft(acc_all[0])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax1.plot(xf, np.abs(yf))


    yf = rfft(acc_all[1])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax2.plot(xf, np.abs(yf))


    yf = rfft(acc_all[2])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax3.plot(xf, np.abs(yf))


    yf = rfft(acc_all[3])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax4.plot(xf, np.abs(yf))


    yf = rfft(acc_all[4])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax5.plot(xf, np.abs(yf))


    yf = rfft(acc_all[5])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax6.plot(xf, np.abs(yf))


    yf = rfft(acc_all[6])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax7.plot(xf, np.abs(yf))

    yf = rfft(acc_all[7])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax8.plot(xf, np.abs(yf))

    yf = rfft(acc_all[8])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax9.plot(xf, np.abs(yf))

    yf = rfft(acc_all[9])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax10.plot(xf, np.abs(yf))

    yf = rfft(acc_all[10])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax11.plot(xf, np.abs(yf))

    yf = rfft(acc_all[11])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax12.plot(xf, np.abs(yf))

    yf = rfft(acc_all[12])
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    ax13.plot(xf, np.abs(yf))

    plt.show()

    fs= SAMPLE_RATE

    joint_name= ['head', 'shoulderR', 'shoulderL', 'elbowR', 'elbowL', 'hipR', 'hipL', 'handR', 'handL', 'kneeR', 'kneeL',
                 'footR', 'footL'] * 20


    for i in range(12):
        plt.subplot(2, 7, i + 1)
        plt.gca().set_title(joint_name[i])
        f, t, Sxx = signal.spectrogram(acc_all[i], fs, nperseg= 50)


        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.show()
