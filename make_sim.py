# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
from data_generation import BoundingShape, ItoGenerator, print_process, create_color_map
from util import print_potential
import numpy
import os
import pickle

intrinsic_process_file_name = 'intrinsic_process'
intrinsic_variance_file_name = 'intrinsic_variance'
dist_potential_file_name = 'dist_potential'
bounding_shape_file_name = 'bounding_shape'
precision = 'float64'
intrinsic_variance = None
subsample_factor = None

'''
sim_dir_name = "2D Square"
process_mode = "Static"
n_points_simulated = 2000
subsample_factor = 10
added_dim_limits = None
bounding_shape = BoundingShape(vertices=[(-1, -1), (1, -1), (1, 1), (-1, 1)])
'''

'''
sim_dir_name = "2D Unit Disk"
process_mode = "Static"
n_points_simulated = 1000
intrinsic_variance = 0.00001**2
n_legs = 24
r = 0.5
added_dim_limits = None
subsample_factor = 10
bounding_shape = BoundingShape(vertices=[(numpy.cos(2*numpy.pi/n_legs*x)*r, numpy.sin(2*numpy.pi/n_legs*x)*r) for x in range(0, n_legs+1)])
'''
'''
sim_dir_name = "2D Unit Disk Punctured by Square"
process_mode = "Static"
n_points_simulated = 10000
intrinsic_variance = 0.00001**2
n_legs = 24
r = 1
added_dim_limits = None
subsample_factor = 10
#bounding_shape = BoundingShape(vertices=[(numpy.cos(2*numpy.pi/n_legs*x)*r, numpy.sin(2*numpy.pi/n_legs*x)*r) for x in range(0, n_legs+1)], hole=[(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)])
cross_size = 0.5
cross_width = 0.2
bounding_shape = BoundingShape(vertices=[(numpy.cos(2*numpy.pi/n_legs*x)*r, numpy.sin(2*numpy.pi/n_legs*x)*r) for x in range(0, n_legs+1)], hole=[(-cross_size, cross_width), (-cross_width, cross_width), (-cross_width, cross_size), (-cross_width, cross_size), (cross_width, cross_size), (cross_width, cross_width), (cross_size, cross_width), (cross_size, -cross_width), (cross_width, -cross_width), (cross_width, -cross_size), (-cross_width, -cross_size), (-cross_width, -cross_width), (-cross_size, -cross_width)])
'''

'''
sim_dir_name = "2D Unit Square Punctured by Cross"
process_mode = "Static"
n_points_simulated = 1500
intrinsic_variance = 0.00001**2
n_legs = 24
r = 1
added_dim_limits = None
subsample_factor = 10
cross_size = 0.5
cross_width = 0.2
bounding_shape = BoundingShape(vertices=[(-1, -1), (1, -1), (1, 1), (-1, 1)], hole=[(-cross_size, cross_width), (-cross_width, cross_width), (-cross_width, cross_size), (-cross_width, cross_size), (cross_width, cross_size), (cross_width, cross_width), (cross_size, cross_width), (cross_size, -cross_width), (cross_width, -cross_width), (cross_width, -cross_size), (-cross_width, -cross_size), (-cross_width, -cross_width), (-cross_size, -cross_width)])
'''


'''
sim_dir_name = "Punctured 2D Unit Disk"
process_mode = "Static"
n_points_simulated = 10000
intrinsic_variance = 0.00001**2
n_legs = 24
r = 0.5
r_hole = 0.2
added_dim_limits = None
subsample_factor = 10
bounding_shape = BoundingShape(vertices=[(numpy.cos(2*numpy.pi/n_legs*x)*r, numpy.sin(2*numpy.pi/n_legs*x)*r) for x in range(0, n_legs+1)], hole=[(numpy.cos(2*numpy.pi/n_legs*x)*r_hole, numpy.sin(2*numpy.pi/n_legs*x)*r_hole) for x in range(0, n_legs+1)])
'''

'''
sim_dir_name = "Punctured 2D Unit Square"
process_mode = "Static"
intrinsic_variance = 0.00001**2
n_points_simulated = 10000
subsample_factor = 10
added_dim_limits = None
bounding_shape = BoundingShape(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)], hole=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
'''

'''
sim_dir_name = "2D Rectangle"
process_mode = "Static"
intrinsic_variance = 0.00001**2
n_points_simulated = 10000
added_dim_limits = None
subsample_factor = 10
bounding_shape = BoundingShape(vertices=[(0, 0), (2, 0), (2, 3), (0, 3)])
'''

'''
sim_dir_name = "Punctured 2D Rectangle"
process_mode = "Static"
intrinsic_variance = 0.00001**2
n_points_simulated = 10000
added_dim_limits = None
subsample_factor = 10
bounding_shape = BoundingShape(vertices=[(0, 0), (2, 0), (2, 3), (0, 3)], hole=[(0.5, 1.25), (1.5, 1.25), (1.5, 2.50), (0.5, 2.50)])
'''


'''
sim_dir_name = "2D Non-Convex H"
process_mode = "Dynamic"
intrinsic_variance = 0.02**2
added_dim_limits = None
subsample_factor = 10
bounding_shape = BoundingShape(vertices_in=[(0, 0), (0.4, 0), (0.4, 0.3), (0.7, 0.3), (0.7, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.7), (0.2, 0.7), (0.2, 1), (0, 1)])
'''

sim_dir_name = "2D Apartment to Print"
process_mode = "Static"
n_points_simulated = 10000
bounding_shape = BoundingShape(vertices=[(2.6, 0.2), (2.6, 1.33), (0.2, 1.33), (0.2, 3.25), (0.93, 3.25), (0.93, 5.11), (1.69, 5.11), (1.69, 4.87), (2.8, 4.87), (2.8, 4), (4.37, 4), (4.37, 6.3), (7.28, 6.3), (7.28, 5.82), (8.69, 5.82), (8.69, 8.25), (9.54, 8.25), (9.54, 4.65), (12.8, 4.65), (12.8, 3.18), (10.65, 3.18), (10.65, 2.3), (11.8, 2.3), (11.8, 0.2), (9.17, 0.2), (9.17, 2.3), (9.75, 2.3),  (9.75, 3.63), (8.52, 4.25), (7.28, 4.25), (7.28, 1.63), (4.69, 1.63), (4.69, 0.2)])
#added_dim_limits = numpy.asarray([[0.5, 2.5]]).T
added_dim_limits = None
subsample_factor = 10


'''
sim_dir_name = "2D Non Convex Room - Static"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (15, 6), (15, 4.1), (14.1, 4.1), (14.1, 2), (15, 2), (15, 0), (10.4, 0), (10.4, 3), (7.5, 3), (7.5, 0)])
#boundary_threshold = 0.2
'''

'''
sim_dir_name = "3D Small Room"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (7.5, 6), (7.5, 0), (7.5, 0)])
added_dim_limits = numpy.asarray([[0., 6.]]).T
boundary_threshold = 0.2
'''

'''
sim_dir_name = "3D Non Convex Room"
process_mode = "Static"
intrinsic_variance = 0.1**2
bounding_shape = BoundingShape(vertices=[(2.4, 0), (2.4, 1.13), (1, 1.13), (1, 3.15), (0, 3.15), (0, 6), (3, 6), (3, 4.2), (4.2, 4.2), (4.2, 6), (7.5, 6), (7.5, 0), (7.5, 0)])
added_dim_limits = numpy.asarray([[0., 6.]]).T
boundary_threshold = 0.2
'''

sim_dir = './' + sim_dir_name + ' - ' + process_mode

if not(os.path.isdir(sim_dir)):
    os.makedirs(sim_dir)

intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name

ito_generator = ItoGenerator(bounding_shape=bounding_shape, added_dim_limits=added_dim_limits)

intrinsic_simulated_process, dist_potential = ito_generator.gen_process(n_trajectory_points=n_points_simulated, process_var=intrinsic_variance, process_mode=process_mode, added_dim_limits=added_dim_limits, subsample_factor=subsample_factor)

numpy.save(sim_dir + '/' + intrinsic_process_file_name, intrinsic_simulated_process.T)

if process_mode == "Dynamic":
    numpy.save(sim_dir + '/' + intrinsic_variance_file_name, intrinsic_variance)

#numpy.save(sim_dir + '/' + dist_potential_file_name, dist_potential)

f = open(sim_dir + '/' + 'bounding_shape.pckl', 'wb')
pickle.dump(bounding_shape, f)
f.close()


numpy.save(sim_dir + '/' + bounding_shape_file_name, bounding_shape.vertices)
#numpy.savetxt(sim_dir + '/' + 'bounding_shape.txt', bounding_shape.vertices, delimiter=',')

color_map = create_color_map(intrinsic_simulated_process)
n_plot_points = 20000
n_plot_points = min(n_points_simulated, n_plot_points)
points_plot_index = numpy.random.choice(n_points_simulated, size=n_plot_points, replace=False)
print_process(intrinsic_simulated_process, bounding_shape=bounding_shape, indexes=points_plot_index, titleStr="Intrinsic Space", color_map=color_map)
#color_map[numpy.where(dist_potential < boundary_threshold), :] = [0, 0, 0]
#color_map[numpy.where(dist_potential > boundary_threshold), :] = color_map[numpy.where(dist_potential > boundary_threshold), :]
#print_process(intrinsic_simulated_process, bounding_shape=bounding_shape, indexs=points_plot_index, titleStr="Intrinsic Space + Boundaries", color_map=color_map)
plt.show(block=True)

