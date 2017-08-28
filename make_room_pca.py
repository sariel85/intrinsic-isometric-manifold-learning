from scipy.io import loadmat
#import h5py
import numpy
import cv2
from cv2 import *
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
from data_generation import print_process, create_color_map, phase_invariance, print_dynamics, non_int_roll

sim_dir_name = "2D Apartment New - Array"
sim_dir = './' + sim_dir_name
video_file = sim_dir + '/' + 'video_input.avi'

n_components = 50
n_frames_pca = 3*1000
smooth_sigma = 3
smoothing_kernel_size = 6*smooth_sigma+1
downsample_factor = 4

ds_type = 'PCA'
cap = cv2.VideoCapture(video_file)
n_frames = numpy.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = numpy.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = numpy.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
n_pixels = numpy.int(frame_height * frame_width)

movie_frames_for_pca = None

while not cap.isOpened():
    cap = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

i_frame_pca = 0

while True and i_frame_pca < n_frames_pca:

    flag, frame = cap.read()

    #lin_phase = 40

    #X, Y = numpy.meshgrid(range(n_pixels_x), range(n_pixels_y))
    #frame_gray_fft_changed = frame_gray_fft*[numpy.exp(-1j*(2*numpy.pi/n_pixels_x)*lin_phase*X)]
    #recon = numpy.real(numpy.fft.ifft2(frame_gray_fft_changed[0, :, :]))

    if flag:
        frame = numpy.roll(frame, shift=0, axis=1)
        '''
        plt.figure()
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
        plt.axis("off")
        plt.show(block=False)
        '''

        #r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #gray_shifted = numpy.roll(gray, 20, axis=1)

        #n_pixels_x = gray.shape[1]
        #n_pixels_y = gray.shape[0]

        #plt.figure()
        #plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
        #plt.show(block=False)

        # plt.figure()
        # plt.imshow(gray_shifted, cmap='gray', vmin=0, vmax=255)
        # plt.show(block=False)

        # phase_invariance(gray_shifted)
        ''''
        # The frame is ready and already captured
        frame_3 = numpy.concatenate((gray, gray, gray), axis=1)
        dst_gray = cv2.GaussianBlur(frame_3, (smoothing_kernel_size, smoothing_kernel_size), sigmaX=3, sigmaY=3, borderType=BORDER_REPLICATE)
        dst_gray = dst_gray[:, frame_width:2*frame_width]

        lin_phase_low = phase_invariance(dst_gray)
        lin_phase_low = 0
        #lin_phase = 10

        r_rolled = numpy.roll(r, shift=int(numpy.round(lin_phase_low)), axis=1)
        g_rolled = numpy.roll(g, shift=int(numpy.round(lin_phase_low)), axis=1)
        b_rolled = numpy.roll(b, shift=int(numpy.round(lin_phase_low)), axis=1)

        #r_rolled = non_int_roll(r, -lin_phase_low)
        #g_rolled = non_int_roll(g, -lin_phase_low)
        #b_rolled = non_int_roll(b, -lin_phase_low)
        '''

        #gray_rolled = 0.2989 * r_rolled + 0.5870 * g_rolled + 0.1140 * b_rolled

        #frame_3 = numpy.asarray([r_rolled, g_rolled, b_rolled])
        #frame_3 = frame_3.transpose([1, 2, 0])

        #lin_phase = phase_invariance(gray_rolled)

        frame_rgb = numpy.concatenate((frame, frame, frame), axis=1)

        dst = cv2.GaussianBlur(frame_rgb, (smoothing_kernel_size, smoothing_kernel_size), sigmaX=smooth_sigma, sigmaY=smooth_sigma, borderType=BORDER_REPLICATE)
        dst = dst[:, frame_width:2*frame_width, :]
        dst = dst[::downsample_factor, :, :][:, ::downsample_factor, :]

        if (i_frame_pca==0):
            plt.figure()
            plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), interpolation='none')
            plt.title('Smoothed image')
            plt.show(block=False)

        #plt.figure()
        #plt.subplot(121), plt.imshow(dst_gray, cmap='gray'), plt.title('Original')
        #plt.subplot(122), plt.imshow(cv2.cvtColor(numpy.asarray(dst, dtype=numpy.uint8), cv2.COLOR_BGR2RGB), interpolation='none'), plt.title('Averaging')
        #plt.show(block=False)

        if movie_frames_for_pca is None:
            movie_frames_for_pca = numpy.empty([n_frames_pca, dst.shape[0]*dst.shape[1]*dst.shape[2]])

        movie_frames_for_pca[i_frame_pca, :] = numpy.asarray(dst).reshape([dst.shape[0]*dst.shape[1]*dst.shape[2]])
        i_frame_pca = i_frame_pca + 1
        print(str(numpy.int(i_frame_pca)) + " frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == n_frames:
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        cv2.CAP_PROP_POS_FRAMES
        break


if ds_type == 'PCA':
    pca = PCA(n_components=n_components, whiten=False)
    movie_frames_base = movie_frames_for_pca[::3, :]
    pca.fit(movie_frames_base)
    pca_base = pca.components_
    explained_variance = pca.explained_variance_
    plt.plot(explained_variance)
    plt.show(block=False)
    plt.ylabel('Explained variance')
    plt.xlabel('Principal components')

    movie_pca = numpy.zeros((n_frames, n_components))

    cap = cv2.VideoCapture(video_file)

    while not cap.isOpened():
        cap = cv2.VideoCapture(video_file)
        cv2.waitKey(1000)
        print("Wait for the header")

    n_frames_per_batch = 500
    n_batches = round(n_frames/n_frames_per_batch)
    for i in numpy.arange(round(n_frames/n_frames_per_batch)):
        print('%d out of %d' % (i+1, n_batches))
        movie_frames_for_batch = None
        i_frame = 0
        while True and i_frame < n_frames_per_batch:
            flag, frame = cap.read()
            if flag:
                #Transform image
                frame_rgb = numpy.concatenate((frame, frame, frame), axis=1)
                dst = cv2.GaussianBlur(frame_rgb, (smoothing_kernel_size, smoothing_kernel_size), sigmaX=smooth_sigma, sigmaY=smooth_sigma,
                                       borderType=BORDER_REPLICATE)
                dst = dst[:, frame_width:2 * frame_width, :]
                dst = dst[::downsample_factor, :, :][:, ::downsample_factor, :]

                if movie_frames_for_batch is None:
                    movie_frames_for_batch = numpy.empty([n_frames_per_batch, dst.shape[0] * dst.shape[1] * dst.shape[2]])

                movie_frames_for_batch[i_frame, :] = numpy.asarray(dst).reshape([dst.shape[0] * dst.shape[1] * dst.shape[2]])

                i_frame += 1

            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)
            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == n_frames:
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                cv2.CAP_PROP_POS_FRAMES
                break

        movie_pca[i*n_frames_per_batch:(i+1)*n_frames_per_batch, :] = pca.transform(movie_frames_for_batch)

    #numpy.savetxt(sim_dir + '/' + 'pca_vects.txt', pca_base, delimiter=',')
    #pca_base = numpy.loadtxt(sim_dir + '/' + 'pca_vects.txt', delimiter=',')
    #movie_pca = pca.fit_transform(movie_frames)
    #movie_pca = numpy.dot(numpy.linalg.pinv(pca_base.T), (movie_frames-numpy.mean(movie_frames.T, 1)).T)


intrinsic_states = numpy.loadtxt(sim_dir + '/' + 'intrinsic_process_to_measure.txt', delimiter=',').T

numpy.savetxt(sim_dir + '/' + 'intrinsic_states.txt', intrinsic_states, delimiter=',')

#color_map = create_color_map(intrinsic_process)
#print_process(movie_pca, bounding_shape=None, color_map=color_map, titleStr="Feature Space")
#plt.show(block=False)
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#sensor_noisy = min_max_scaler.fit_transform(movie_pca.T)
sensor_noisy = movie_pca.T
numpy.savetxt(sim_dir + '/' + 'observed_states_noisy.txt', sensor_noisy, delimiter=',')
plt.show(block=True)
