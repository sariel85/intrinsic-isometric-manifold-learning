from scipy.io import loadmat
import h5py
import numpy
from moviepy.editor import *
import cv2
from cv2 import *
from sklearn.decomposition.pca import PCA

sim_dir_name = "Room3D"
sim_dir = './' + sim_dir_name
video_file = sim_dir + '/' + 'test.avi'

cap = cv2.VideoCapture(video_file)

n_frames = numpy.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_hight = numpy.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = numpy.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

test_frames = numpy.empty([n_frames, 3*frame_hight*frame_width])

while not cap.isOpened():
    cap = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print ("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        cv2.CAP_PROP_POS_FRAMES
        break
    test_frames[pos_frame, :] = numpy.asarray(frame[:, :, :]).reshape([3 * frame_hight * frame_width])


numpy.savetxt(sim_dir + '/' + 'test_mat.txt', test_frames, delimiter=',')
test_frames = numpy.loadtxt(sim_dir + '/' + 'movie_mat.txt', delimiter=',')

pca_base = numpy.loadtxt(sim_dir + '/' + 'pca_base.txt', delimiter=',')

test_noisy = numpy.dot(numpy.linalg.pinv(pca_base.T), test_frames.T)

numpy.savetxt(sim_dir + '/' + 'test_noisy.txt', sensor_noisy_base, delimiter=',')





