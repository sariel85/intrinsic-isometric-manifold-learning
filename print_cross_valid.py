from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy


sim_dir_name = "2D Unit Disk - Bursts - Fishbowl"
cross_dir_name = sim_dir_name + "/Cross/"
n = 20
w = 0.00001

cross_dir_name = cross_dir_name + "n_" + str(n) + "_w_" + str(w)


f = []
for (dirpath, dirnames, filenames) in walk(cross_dir_name):
    for file in filenames:
        f.append(dirpath + '/' + file)

log_total = []
for file in f:
    logs_temp = numpy.load(file).astype(dtype=numpy.float64).T
    log_total.append(logs_temp)

log_total_numpy = numpy.asarray(log_total)
log_total_numpy = log_total_numpy.mean(axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(log_total_numpy[1][0:], c='k', label='Train Error')
ax.plot(log_total_numpy[2][0:], c='g', label='Valid Error')
ax.plot(log_total_numpy[3][0:], c='r', label='Test Error')
ax.set_xlabel('Training epochs')
ax.set_ylabel('Log-likelihood')
#ax.set_title('Nodes=%d, Weight Decay=%s' % (n, w))
plt.ylim([-18.5, -17.0])
#plt.xlim([1500, 2800])
#plt.legend()
plt.show(block="True")


#os.listdir(cross_dir_name)



