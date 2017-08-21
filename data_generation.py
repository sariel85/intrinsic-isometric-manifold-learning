import numpy
import math
from shapely.geometry import Polygon as PolygonShapely, Point, LinearRing
#import polyhedron
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpl_toolkits.mplot3d as a3
from matplotlib.patches import Polygon as Polygon, Ellipse as Ellipse
from util import rigid_transform
import sys


def create_color_map(intrinsic_process, bounding_shape=None):
    if intrinsic_process.shape[0] == 2 or intrinsic_process.shape[0] == 3:
        # Get data limits
        if bounding_shape is not None:
            box_origin, box_width = bounding_shape.bounding_box()
        else:
            min_val = numpy.min(intrinsic_process, axis=1)
            max_val = numpy.max(intrinsic_process, axis=1)
            box_origin = min_val
            box_width = max_val - min_val

        if intrinsic_process.shape[0] == 2:

            color_map_part = ((intrinsic_process - box_origin.reshape(2, 1)) / box_width.reshape(2, 1))
            color_map = 0 * numpy.ones([3, intrinsic_process.shape[1]])
            color_map[(0, 1), :] = color_map_part
            color_map = numpy.asarray(color_map).T.tolist()

        if intrinsic_process.shape[0] == 3:

            color_map_part = ((intrinsic_process - box_origin.reshape(3, 1)) / box_width.reshape(3, 1))
            color_map = 0 * numpy.ones([3, intrinsic_process.shape[1]])
            color_map[(0, 1, 2), :] = color_map_part
            color_map = numpy.asarray(color_map).T.tolist()

        return numpy.asarray(color_map)

    else:
        return None
    # return matplotlib.colors.ColorConverter.to_rgb(arg=color_map)


def angle_from_cov(cov_mat):

    U, s, V = numpy.linalg.svd(cov_mat)
    width = numpy.sqrt(s[0])
    height = numpy.sqrt(s[1])
    angle = numpy.angle(V[0, 0]+V[0, 1]*1j, deg=False)
    return height, width, angle


def print_process(input_process, indexes=None, bounding_shape=None, color_map=None, ax=None, titleStr=None, covs=None, ax_limits=None, align_points=None, el=1, azi=1):

    if indexes is None:
        indexes = numpy.arange(0, input_process.shape[1])
    else:
        color_map = color_map[indexes, :]
    input_process = input_process[:, indexes]

    if align_points is not None:
        n_required_align_points = 100000
        n_available_align_points = input_process.shape[1]
        n_used_align_points = min(n_available_align_points, n_required_align_points)
        points_align_index = numpy.random.choice(n_available_align_points, size=n_used_align_points, replace=False)
        R, t = rigid_transform(input_process[:, points_align_index], align_points[:, points_align_index])
        process = (numpy.dot(R, input_process).T + t.T).T
    else:
        process = input_process

    if process.shape[0] == 2:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')

        if color_map is None:
            ax.scatter(process[0, :], process[1, :], s=36, linewidths=1)
        else:
            ax.scatter(process[0, :], process[1, :], c=color_map, s=36, linewidths=1, edgecolors=0.5*color_map)

        if covs is not None:
            for i_cluster_point in range(0, process.shape[1]):
                height, width, angle = angle_from_cov(covs[i_cluster_point][:, :])
                e = Ellipse(xy=(process[0, i_cluster_point], process[1, i_cluster_point]), width=width, height=height, angle=angle*180/numpy.pi)
                e.set_color(color_map[i_cluster_point, :])
                e.set_edgecolor('k')
                e.set_alpha(0.5)
                ax.add_patch(e)
        #plt.axis('equal')
        #ax.set_xlabel('x')
        #x.set_ylabel('y')
        if bounding_shape is not None:
            if bounding_shape.shape_type == 'Box':
                bounding_shape_vertices = numpy.asarray([bounding_shape.origin, bounding_shape.origin+[bounding_shape.scale[0], [0]], bounding_shape.origin + bounding_shape.scale, bounding_shape.origin+[[0], bounding_shape.scale[1]]], dtype=numpy.float64).squeeze()
                polygon = Polygon(bounding_shape.vertices, fill=False)
            else:
                polygon = Polygon(bounding_shape.vertices, fill=False)
                ax.add_patch(polygon)
                if bounding_shape.hole is not None:
                    polygon = Polygon(bounding_shape.hole, fill=False)
                    ax.add_patch(polygon)

        if ax_limits is not None:
            ax.set_xlim([ax_limits[0], ax_limits[1]])
            ax.set_ylim([ax_limits[2], ax_limits[3]])
        else:
            x_size = numpy.max(process[0, :])-numpy.min(process[0, :])
            y_size = numpy.max(process[1, :])-numpy.min(process[1, :])
            ax.set_xlim([numpy.min(process[0, :])-0.1*x_size, numpy.max(process[0, :])+0.1*x_size])
            ax.set_ylim([numpy.min(process[1, :])-0.1*y_size, numpy.max(process[1, :])+0.1*y_size])

        plt.show(block=False)
        plt.axis('equal')

        return ax

    elif process.shape[0] == 3:

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', aspect='equal')
        else:
            ax.cla()

        if color_map is None:
            ax.scatter(process[0, :], process[1, :], process[2, :], s=36, linewidths=1)
        else:
            ax.scatter(process[0, :], process[1, :], process[2, :], c=color_map, s=36, linewidths=1, edgecolors=0.5*color_map)

        if bounding_shape is not None:
            for i in numpy.arange(len(bounding_shape.tri)):
                square = [bounding_shape.vertices[bounding_shape.tri[i][0]], bounding_shape.vertices[bounding_shape.tri[i][1]], bounding_shape.vertices[bounding_shape.tri[i][2]]]
                face = a3.art3d.Poly3DCollection([square])
                face.set_color(colors.rgb2hex(numpy.rand(3)))
                face.set_edgecolor('k')
                face.set_alpha(0.5)
                ax.add_collection3d(face)

        if ax_limits is not None:
            ax.set_xlim([ax_limits[0], ax_limits[1]])
            ax.set_ylim([ax_limits[2], ax_limits[3]])
            ax.set_zlim([ax_limits[4], ax_limits[5]])

        max_range = numpy.array([process[0].max() - process[0].min(), process[1].max() - process[1].min(), process[2].max() - process[2].min()]).max() / 2.0

        mid_x = (process[0].max() + process[0].min()) * 0.5
        mid_y = (process[1].max() + process[1].min()) * 0.5
        mid_z = (process[2].max() + process[2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.canvas.set_window_title(titleStr)
        ax.view_init(el, azi)

        fig.canvas.set_window_title(titleStr)
        plt.show(block=False)
        return ax

    else:

        return None

def print_drift(process, drift,  indexs=None, bounding_shape=None, color_map=None, ax=None, titleStr=None, covs=None):

    if indexs is None:
        indexs = numpy.arange(0, process.shape[1])
    else:
        color_map = color_map[indexs, :]

    process = process[:, indexs]
    drift = drift[:, indexs]
    if process.shape[0] == 2:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        else:
            ax.cla()
        #plt.plot(process[0, :], process[1, :])

        if color_map is None:
            plt.quiver(process[0, :].T, process[1, :].T,
                       drift[0, :] - process[0, :].T,
                       drift[1, :] - process[1, :].T, angles='xy', scale_units='xy', scale=1, pivot='tail')
        else:
            plt.quiver(process[0, :].T, process[1, :].T,
                       drift[0, :] - process[0, :].T,
                       drift[1, :] - process[1, :].T, angles='xy', scale_units='xy', scale=1, color=color_map)

        if covs is not None:
            covs = covs[indexs, :]
            for i_cluster_point in range(0, process.shape[1]):
                height, width, angle = angle_from_cov(covs[i_cluster_point][:, :])
                e = Ellipse(xy=(process[0, i_cluster_point], process[1, i_cluster_point]), width=width, height=height, angle=angle*180/numpy.pi)
                e.set_color(color_map[i_cluster_point, :])
                e.set_edgecolor('k')
                e.set_alpha(0.5)
                ax.add_patch(e)
        #plt.axis('equal')
        #ax.set_xlabel('x')
        #x.set_ylabel('y')
        if bounding_shape is not None:
            if bounding_shape.shape_type == 'Box':
                bounding_shape_vertices = numpy.asarray([bounding_shape.origin, bounding_shape.origin+[bounding_shape.scale[0], [0]], bounding_shape.origin + bounding_shape.scale, bounding_shape.origin+[[0], bounding_shape.scale[1]]], dtype=numpy.float64).squeeze()
                polygon = Polygon(bounding_shape_vertices, fill=False)
            else:
                polygon = Polygon(bounding_shape.vertices, fill=False)
                ax.add_patch(polygon)

    elif process.shape[0] == 3:

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', aspect='equal')
        else:
            ax.cla()

        if color_map is None:
            ax.quiver(process[0, :].T, process[1, :].T, process[2, :].T,
                      (drift[0, :] - process[0, :].T),
                      (drift[1, :] - process[1, :].T),
                      (drift[2, :] - process[2, :].T), units='xy')
        else:
            ax.quiver(process[0, :].T, process[1, :].T, process[2, :].T,
                      (drift[0, :] - process[0, :].T),
                      (drift[1, :] - process[1, :].T),
                      (drift[2, :] - process[2, :].T), units='xy', color=color_map)

        if bounding_shape is not None:
            for i in numpy.arange(len(bounding_shape.tri)):
                square = [bounding_shape.vertices[bounding_shape.tri[i][0]], bounding_shape.vertices[bounding_shape.tri[i][1]], bounding_shape.vertices[bounding_shape.tri[i][2]]]
                face = a3.art3d.Poly3DCollection([square])
                face.set_color(colors.rgb2hex(numpy.rand(3)))
                face.set_edgecolor('k')
                face.set_alpha(0.5)
                ax.add_collection3d(face)

    ax.set_aspect('equal')
    plt.title(titleStr)
    plt.show(block=False)
    return ax

def print_dynamics(process_base, process_step,  indexs=None, bounding_shape=None, color_map=None, ax=None, titleStr=None):

    n_points = indexs.shape[0]
    if indexs is None:
        indexs = numpy.arange(0, process_base.shape[1])
    else:
        color_map = color_map[indexs, :]

    process = process_base[:, indexs]
    drift = process_step[:, indexs]-process_base[:, indexs]
    if process.shape[0] == 2:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        else:
            ax.cla()
        #plt.plot(process[0, :], process[1, :])

        if color_map is None:
            plt.quiver(process[0, :].T, process[1, :].T,
                       drift[0, :],
                       drift[1, :], units='xy', scale=1, pivot='tail')
        else:
            plt.quiver(process[0, :].T, process[1, :].T,
                       drift[0, :],
                       drift[1, :], units='xy', scale=1, pivot='tail', color=color_map)

        if bounding_shape is not None:
            if bounding_shape.shape_type == 'Box':
                bounding_shape_vertices = numpy.asarray([bounding_shape.origin, bounding_shape.origin+[bounding_shape.scale[0], [0]], bounding_shape.origin + bounding_shape.scale, bounding_shape.origin+[[0], bounding_shape.scale[1]]], dtype=numpy.float64).squeeze()
                polygon = Polygon(bounding_shape_vertices, fill=False)
            else:
                polygon = Polygon(bounding_shape.vertices, fill=False)
                ax.add_patch(polygon)

    elif process.shape[0] == 3:

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', aspect='equal')
        else:
            ax.cla()

        if color_map is None:
            for i_point in range(0, n_points):
                ax.quiver(process[0, i_point], process[1, i_point], process[2, i_point], (drift[0, i_point]), (drift[1, i_point]), (drift[2, i_point]),
                          length=numpy.linalg.norm(drift[:, i_point]), pivot='tail')
        else:
            for i_point in range(0, n_points):
                ax.quiver(process[0, i_point], process[1, i_point], process[2, i_point], (drift[0, i_point]), (drift[1, i_point]), (drift[2, i_point]),
                          length=numpy.linalg.norm(drift[:, i_point]), pivot='tail', color=color_map[i_point, :])

        if bounding_shape is not None:
            for i in numpy.arange(len(bounding_shape.tri)):
                square = [bounding_shape.vertices[bounding_shape.tri[i][0]], bounding_shape.vertices[bounding_shape.tri[i][1]], bounding_shape.vertices[bounding_shape.tri[i][2]]]
                face = a3.art3d.Poly3DCollection([square])
                face.set_color(colors.rgb2hex(numpy.rand(3)))
                face.set_edgecolor('k')
                face.set_alpha(0.5)
                ax.add_collection3d(face)

    ax.set_aspect('equal')
    plt.title(titleStr)
    plt.show(block=False)
    return ax





def bounding_potential(point, bounding_shape, added_dim_limits):
    d_shape = bounding_shape.dist_from_point(point)

    if added_dim_limits is not None:
        added_cords = (numpy.random.rand(added_dim_limits.shape[1]) - added_dim_limits[0, :]) * (
        added_dim_limits[1, :] - added_dim_limits[0, :])
        d_added = numpy.min([added_cords - added_dim_limits[0, :], added_dim_limits[1, :] - added_cords])
    else:
        d_added = numpy.inf

    d = min(d_shape, d_added)

    if d <= 0:
        potential = 0
    else:
        potential = 0.5 * math.pow(d, 2)
    return potential


def compute_numerical_gradient(f, point, step=0.0001):
    dim_function = point.shape[0]
    f_num_grad = numpy.zeros([dim_function, 1])
    #f_base = f(point=point.reshape((dim_function, 1)))
    for i_dim in range(0, dim_function):
        step_vect = numpy.zeros([dim_function, 1])
        step_vect[i_dim] = step
        f_num_grad[i_dim] = (f(point=(point.reshape([dim_function, 1]) + step_vect)) - f(point=(point.reshape([dim_function, 1]) - step_vect))) / (2*step)
    return f_num_grad


class BoundingShape(object):
    #def __init__(self):
    #    self.shape_type = 'nan'
    #    self.origin = 'nan'
    #    self.scale = 'nan'
    #    self.vertices = 'nan'
    #    self.hole = 'nan'
    #    self.tri = 'nan'

    def __init__(self, vertices='nan', hole=None, tri='nan', radius='nan', size='nan', predef_type='Not Specified', k=5, r_in=0.5, r_out=1, origin=[0, 0]):

        self.shape_type = 'nan'
        self.origin = 'nan'
        self.scale = 'nan'
        self.vertices = 'nan'
        self.hole = None
        self.tri = 'nan'

        if predef_type == '1D Unit Segment':
            self.shape_type = "Box"
            self.origin = numpy.asarray([[0]], dtype=numpy.float64)
            self.scale = numpy.asarray([[1]], dtype=numpy.float64)
            self.dim = 1
        elif predef_type == '2D Unit Square':
            self.shape_type = "Box"
            self.origin = numpy.asarray([[0], [0]], dtype=numpy.float64)
            self.scale = numpy.asarray([[1], [1]], dtype=numpy.float64)
            self.dim = 2
        elif predef_type == '3D Unit Cube':
            self.shape_type = "Box"
            self.origin = numpy.asarray([[0], [0], [0]], dtype=numpy.float64)
            self.scale = numpy.asarray([[1], [1], [1]], dtype=numpy.float64)
            self.dim = 3
        elif predef_type == '2D Unit Circle':
            self.shape_type = "Ball"
            self.scale = 0.5
            self.dim = 2
            self.origin = numpy.asarray([[0], [0]], dtype=numpy.float64)
        elif predef_type == '3D Unit Ball':
            self.shape_type = "Ball"
            self.scale = 1
            self.dim = 3
            self.origin = numpy.asarray([[0], [0], [0]], dtype=numpy.float64)
        elif predef_type == 'Star':
            self.shape_type = "Polygon"
            self.vertices = []
            self.dim = 2
            deg_space = (2*numpy.pi)/k
            for i_leg in range(0, k):
                self.vertices.append((0.5+r_in*numpy.cos(i_leg*deg_space), 0.5+r_in*numpy.sin(i_leg*deg_space))+numpy.asarray(origin))
                self.vertices.append((0.5+r_out*numpy.cos(i_leg*deg_space+deg_space/2), 0.5+r_out*numpy.sin(i_leg*deg_space+deg_space/2))+numpy.asarray(origin))

        else:
            if vertices[0].__len__() == 1:
                self.dim = 1
                self.shape_type = "Ball"
            elif vertices[0].__len__() == 2:
                self.dim = 2
                self.vertices = vertices
                self.hole = hole

                self.shape_type = "Polygon"
            elif vertices[0].__len__() == 3:
                self.dim = 3
                self.shape_type = "Polyhedra"
                self.vertices = vertices
                self.hole = hole
                self.tri = tri
            else:
                # Unknown Bounding Shape
                assert False

    def dist_from_point(self, point):
        point = numpy.asarray(point, dtype=numpy.float64)
        if self.shape_type == 'Box':
            dx = numpy.amax(numpy.concatenate((self.origin - point.reshape([self.dim, 1]), numpy.zeros([self.dim, 1], dtype=numpy.float64), point.reshape([self.dim, 1]) - (self.origin + self.scale)), axis=1), axis=1)
            return numpy.linalg.norm(dx)
        elif self.shape_type == 'Ball':
            return numpy.amax([numpy.linalg.norm(point - self.origin) - self.scale, 0])
        elif self.shape_type == 'Polygon':

            poly_in = PolygonShapely(self.vertices)
            point_sh = (Point(point[0], point[1]))
            #d_in = poly_in_ext.project(point_sh)
            #p_in = poly_in_ext.interpolate(d_in)
            #closest_point_in = numpy.asarray(list(p_in.coords)[0])
            #d_shape_in = numpy.linalg.norm(closest_point_in - point.T)
            dist_in = poly_in.distance(Point(point[0], point[1]))

            if self.hole is not None:
                poly_out = PolygonShapely(self.hole)
                poly_out_ext = LinearRing(poly_out.exterior.coords)
                d_out = poly_out_ext.project(point_sh)
                p_out = poly_out_ext.interpolate(d_out)
                closest_point_out = numpy.asarray(list(p_out.coords)[0])
                d_shape_out = numpy.linalg.norm(closest_point_out - point.T)
                dist_out = poly_out.distance(Point(point[0], point[1]))
                return numpy.max([dist_in, (dist_out == 0) * d_shape_out])
            else:
                return dist_in

        elif self.shape_type == 'Polyhedra':
            limit_polyhedron = polyhedron.Polyhedron(self.tri, self.vertices)
            if limit_polyhedron.winding_number((float(point[0]), float(point[1]), float(point[2]))) == 0:
                n_tri = self.tri.__len__()
                dist_tri = numpy.zeros([n_tri, 1])
                for i_tri in range(0, n_tri):
                    a = numpy.asarray(self.vertices[self.tri[i_tri][0]], dtype=numpy.float64)
                    b = numpy.asarray(self.vertices[self.tri[i_tri][1]], dtype=numpy.float64)
                    c = numpy.asarray(self.vertices[self.tri[i_tri][2]], dtype=numpy.float64)
                    vect_a = b - a
                    vect_b1 = vect_a / numpy.linalg.norm(vect_a)
                    vect_b1 = vect_b1.reshape(3, 1)
                    vect_b = c - a
                    vect_b2 = vect_b / numpy.linalg.norm(vect_b)
                    vect_b2 = vect_b2.reshape(3, 1)
                    vect_b2 = vect_b2 - vect_b1 * numpy.dot(vect_b1.T, vect_b2)
                    vect_b2 = vect_b2 / numpy.linalg.norm(vect_b2)
                    vect_p = point - a.T.reshape(3, 1)
                    mat = numpy.asarray([vect_b1.reshape(3), vect_b2.reshape(3)], dtype=numpy.float64).T

                    vect_a_2d = numpy.dot(mat.T, vect_a)
                    vect_b_2d = numpy.dot(mat.T, vect_b)
                    vect_o_2d = numpy.asarray([0, 0], dtype=numpy.float64)
                    vect_p_2d = numpy.dot(mat.T, vect_p)
                    poly_tri = PolygonShapely(numpy.asarray([vect_a_2d.T, vect_b_2d.T, vect_o_2d.T]))
                    dist_plane = poly_tri.distance(Point(vect_p_2d[0], vect_p_2d[1]))
                    dist_plane_orth = numpy.linalg.norm(vect_p - numpy.dot(mat, vect_p_2d))
                    dist_tri[i_tri] = numpy.sqrt(numpy.power(dist_plane, 2) + numpy.power(dist_plane_orth, 2))

                return numpy.min(dist_tri)
            else:
                return 0
        else:
            assert False

    def bounding_box(self):

        if self.shape_type == 'Box':
            box_origin = self.origin
            box_width = self.scale
        elif self.shape_type == 'Ball':
            box_origin = self.origin - self.scale
            box_width = 2 * self.scale
        elif self.shape_type == 'Polygon':
            box_origin = numpy.amin(numpy.asarray(self.vertices), 0).reshape(2, 1)
            box_width = (numpy.amax(numpy.asarray(self.vertices), 0) - numpy.amin(numpy.asarray(self.vertices), 0)).reshape(2, 1)
        elif self.shape_type == 'Polyhedra':
            box_origin = numpy.amin(numpy.asarray(self.vertices), 0).reshape(3, 1)
            box_width = (numpy.amax(numpy.asarray(self.vertices), 0) - numpy.amin(numpy.asarray(self.vertices), 0)).reshape(3, 1)
        else:
            assert False
        return box_origin, box_width

    def contains(self, point):
        d = self.dist_from_point(point)
        if d > 0:
            return False
        else:
            return True


class ItoGenerator(object):
    def __init__(self, bounding_shape, added_dim_limits):


        self.bounding_shape = bounding_shape
        def intrinsic_potential(point): return bounding_potential(point, bounding_shape=bounding_shape, added_dim_limits=added_dim_limits)
        self.intrinsic_potential = intrinsic_potential

        if added_dim_limits is not None:
            self.dim_intrinsic = bounding_shape.dim + added_dim_limits.shape[1]
        else:
            self.dim_intrinsic = bounding_shape.dim

        '''
        if bounding_shape is None:
            self.intrinsic_potential = intrinsic_potential
            self.dim_intrinsic = dim_intrinsic
            self.bounding_shape = None
        else:
            self.bounding_shape = bounding_shape
            def intrinsic_potential(point): return bounding_potential(point, bounding_shape=bounding_shape, added_dim_limits=added_dim_limits)
            self.intrinsic_potential = intrinsic_potential
            self.dim_intrinsic = dim_intrinsic
        '''

    def gen_process(self, n_trajectory_points, process_var, process_mode, added_dim_limits, subsample_factor=10):

        toolbar_width = 100

        if process_mode == 'Dynamic':
            # Setup toolbar
            n_simulation_points = n_trajectory_points * subsample_factor
            milestones = numpy.arange(0, n_simulation_points, n_simulation_points / toolbar_width, dtype=numpy.float64)
            milestones = numpy.round(milestones)
            milestones2 = milestones
            milestones2[0:-1] = milestones[1:]
            milestones2[-1] = n_simulation_points
            process = numpy.empty([self.dim_intrinsic, n_simulation_points], dtype=numpy.float64)
            dist_potential = numpy.empty([n_simulation_points], dtype=numpy.float64)
            dist_potential[:] = numpy.NAN
            process[:] = numpy.NAN
            point_start = numpy.empty((self.dim_intrinsic, 1))
            point_start[:, 0] = numpy.NAN
            poly = PolygonShapely(self.bounding_shape.vertices)
            pol_ext = LinearRing(poly.exterior.coords)
            sim_var = process_var / subsample_factor
            # Find bounding box of bounding shape
            [box_origin, box_width] = self.bounding_shape.bounding_box()
            in_bounds_flag = False
            # Randomly select point with in the bounding shape

            if added_dim_limits is None:
                while not in_bounds_flag:
                    point_start = box_origin + numpy.multiply(box_width, numpy.random.rand(self.dim_intrinsic, 1))
                    if self.bounding_shape.contains(point_start):
                        in_bounds_flag = True
            else:
                while not in_bounds_flag:
                    point_start = box_origin + numpy.multiply(box_width, numpy.random.rand(self.dim_intrinsic, 1))
                    if self.bounding_shape.contains(point_start):
                        in_bounds_flag = True

                if added_dim_limits is not None:
                    added_cord = numpy.random.rand(self.dim_intrinsic, added_dim_limits)
                    point_start = point_start, added_cord

            # Save selected starting point and generate the rest of the process
            process[:, 0] = point_start[:, 0]
            dist_potential[0] = numpy.inf
            # Used to track progress
            i_prog = 0

            for i_point in range(1, n_simulation_points):

                if i_point > milestones[i_prog]:
                    # update the bar
                    #sys.stdout.write("%s \n" % ("-"*i_prog))
                    printProgress(i_prog, toolbar_width, prefix='Progress:', suffix='Complete', barLength=50)
                    i_prog = i_prog+1

                num_grad = compute_numerical_gradient(self.intrinsic_potential, point=process[:, i_point-1])

                test_point = process[:, i_point-1].reshape(self.dim_intrinsic, 1) - num_grad + math.sqrt(sim_var) * numpy.random.randn(self.dim_intrinsic, 1)
                #temp = process[:, 0].reshape(dim_process, 1) - num_grad + math.sqrt(sim_var) * numpy.random.randn(
                #    dim_process, 1)
                test_point = test_point[:, 0]

                if added_dim_limits is not None:
                    added_cords = (numpy.random.rand(added_dim_limits.shape[1]) - added_dim_limits[0, :])*(added_dim_limits[1, :] - added_dim_limits[0, :])
                    test_point = numpy.append(test_point, added_cords)
                    d_added = numpy.min([added_cords - added_dim_limits[0, :], added_dim_limits[1, :] - added_cords])
                else:
                    d_added = numpy.inf

                # Add point
                process[:, i_point] = test_point.reshape((self.dim_intrinsic,))

                test_point_low = test_point[:self.bounding_shape.dim]
                #d_shape = self.bounding_shape.dist_from_point(test_point)

                point = Point(test_point_low.reshape((self.bounding_shape.dim,)))

                d = pol_ext.project(point)
                p = pol_ext.interpolate(d)
                closest_point_coords = numpy.asarray(list(p.coords)[0])

                d_shape = numpy.linalg.norm(closest_point_coords-test_point[:self.bounding_shape.dim].T)

                dist_potential[i_point] = min(d_shape, d_added)

                if (numpy.linalg.norm(num_grad)>0):
                    dist_potential[i_point-1] = -dist_potential[i_point-1]

        elif process_mode == 'Static':

            subsample_factor=1

            # setup toolbar
            n_simulation_points = n_trajectory_points * subsample_factor
            milestones = numpy.arange(0, n_simulation_points, n_simulation_points / toolbar_width, dtype=numpy.float64)
            milestones = numpy.round(milestones)
            milestones2 = milestones
            milestones2[0:-1] = milestones[1:]
            milestones2[-1] = n_simulation_points
            process = numpy.empty([self.dim_intrinsic, n_simulation_points], dtype=numpy.float64)
            process[:] = numpy.NAN
            dist_potential = numpy.empty([n_simulation_points], dtype=numpy.float64)
            dist_potential[:] = numpy.NAN
            point_start = numpy.empty((self.dim_intrinsic, 1))
            point_start[:, 0] = numpy.NAN
            poly = PolygonShapely(self.bounding_shape.vertices)
            pol_ext = LinearRing(poly.exterior.coords)
            # Find bounding box of bounding shape
            [box_origin, box_width] = self.bounding_shape.bounding_box()
            # Used to track progress
            i_prog = 0
            for i_point in range(0, n_simulation_points):
                in_bounds_flag = False
                # Randomly select point within the bounding shape
                while not in_bounds_flag:
                    test_point = box_origin + numpy.multiply(box_width, numpy.random.rand(self.bounding_shape.dim, 1))
                    if self.bounding_shape.contains(test_point):
                        in_bounds_flag = True
                test_point_low = numpy.asarray(test_point)

                if added_dim_limits is not None:
                    added_cords = (numpy.random.rand(added_dim_limits.shape[1]))*(added_dim_limits[1, :] - added_dim_limits[0, :]) + added_dim_limits[0, :]
                    test_point = numpy.append(test_point, added_cords)
                    d_added = numpy.min([added_cords - added_dim_limits[0, :], added_dim_limits[1, :] - added_cords])
                else:
                    d_added = numpy.inf

                # Add point
                process[:, i_point] = test_point.reshape((self.dim_intrinsic,))
                #d_shape = self.bounding_shape.dist_from_point(test_point)
                point = Point(test_point_low.reshape((self.bounding_shape.dim,)))
                d = pol_ext.project(point)
                p = pol_ext.interpolate(d)
                closest_point_coords = numpy.asarray(list(p.coords)[0])
                d_shape = numpy.linalg.norm(closest_point_coords-test_point[:self.bounding_shape.dim].T)
                dist_potential[i_point] = min(d_shape, d_added)

                if i_point > milestones[i_prog]:
                    # update the bar
                    # sys.stdout.write("%s \n" % ("-"*i_prog))
                    printProgress(i_prog, toolbar_width, prefix='Progress:', suffix='Complete', barLength=50)
                    i_prog = i_prog + 1

        else:
            assert("Intrinsic Process is Neither Static or Dynamic")

        printProgress(toolbar_width, toolbar_width, prefix='Progress:', suffix='Complete', barLength=50)

        return numpy.asarray(process[:, 0::subsample_factor], dtype=numpy.float64), dist_potential[0::subsample_factor]

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def phase_invariance(gray):
    n_pixels_x = gray.shape[1]
    n_pixels_y = gray.shape[0]
    frame_gray_fft = numpy.fft.fft2(gray)
    ang = numpy.angle(frame_gray_fft, deg=False)
    lin_phase = ((ang[0, 1])/(2*numpy.pi))*gray.shape[1]
    #lin_phase = 0
    '''
    T = numpy.fft.fftshift(2*numpy.pi*numpy.arange(n_pixels_x)/n_pixels_x)
    T[:numpy.floor(T.shape[0]/2)] = T[:numpy.floor(T.shape[0]/2)]-2*numpy.pi
    T = numpy.fft.fftshift(T)
    #lin_phase = 0.5
    X, Y = numpy.meshgrid(T, range(n_pixels_y))
    frame_gray_fft_changed = frame_gray_fft*[numpy.exp(1j*lin_phase*X)]
    recon = numpy.abs(numpy.fft.ifft2(frame_gray_fft_changed[0, :, :]))
    '''
    #plt.figure()
    #plt.subplot(121), plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(recon, cmap='gray', vmin=0, vmax=255)
    #plt.title('Linear Phase Removed'), plt.xticks([]), plt.yticks([])
    #plt.show(block=False)
    #plt.figure(), plt.imshow(numpy.abs(numpy.abs(gray)-numpy.abs(recon)), cmap='gray', vmin=0, vmax=255)
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    return lin_phase

def non_int_roll (img, lin_phase):
    #lin_phase = 0.5
    n_pixels_x = img.shape[1]
    n_pixels_y = img.shape[0]
    frame_gray_fft = numpy.fft.fft2(img)
    T = numpy.fft.fftshift(2*numpy.pi*numpy.arange(n_pixels_x)/n_pixels_x)
    T[:numpy.floor(T.shape[0]/2)] = T[:numpy.floor(T.shape[0]/2)]-2*numpy.pi
    T = numpy.fft.fftshift(T)
    X, Y = numpy.meshgrid(T, range(n_pixels_y))
    frame_gray_fft_changed = frame_gray_fft*[numpy.exp(1j*lin_phase*X)]
    recon = numpy.abs(numpy.fft.ifft2(frame_gray_fft_changed[0, :, :]))
    return recon





