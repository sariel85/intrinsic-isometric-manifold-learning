from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

class IntrinsicMetricNet(object):

    def __init__(self, dim_measurements=2, local_noise=1e-5, dim_intrinsic=2, n_hidden_tangent=10, n_hidden_drift=10):

        self.input_base_Theano = T.dmatrix('input_base_Theano')
        self.input_step_Theano = T.dmatrix('input_step_Theano')
        self.init_cov_Theano = T.dtensor3('init_cov_Theano')
        self.init_drift_Theano = T.dmatrix('init_drift_Theano')
        self.reg_Theano = T.dtensor3('reg_Theano')
        self.intrinsic_variance_Theano = T.dscalar('intrinsic_variance_Theano')
        self.dim_intrinsic = dim_intrinsic
        self.dim_measurements = dim_measurements
        self.n_hidden_tangent = n_hidden_tangent
        self.n_hidden_drift = n_hidden_drift
        self.dim_jacobian = self.dim_intrinsic*self.dim_measurements
        self.dim_jacobian = self.dim_intrinsic*self.dim_measurements
        self.dim_jacobian_int = self.dim_intrinsic*self.dim_intrinsic

        self.local_noise = theano.shared(value=local_noise, name='local_noise', borrow=True)

        initial_W_tangent_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.dim_measurements)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.dim_measurements)),
                size=(self.n_hidden_tangent, self.dim_measurements)
            ),
            dtype=theano.config.floatX
        )

        initial_W_tangent_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.n_hidden_tangent)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.n_hidden_tangent)),
                size=(self.n_hidden_tangent, self.n_hidden_tangent)
            ),
            dtype=theano.config.floatX
        )

        initial_W_tangent_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_jacobian + self.n_hidden_tangent)),
                high=4 * numpy.sqrt(6. / (self.dim_jacobian + self.n_hidden_tangent)),
                size=(self.dim_jacobian, self.n_hidden_tangent)
            ),
            dtype=theano.config.floatX
        )

        initial_W_drift_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_drift + self.dim_measurements)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_drift + self.dim_measurements)),
                size=(self.n_hidden_drift, self.dim_measurements)
            ),
            dtype=theano.config.floatX
        )

        initial_W_drift_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                size=(self.n_hidden_drift, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )

        initial_W_drift_3 = numpy.asarray(
            numpy.random.uniform(
                low=-0.004 * numpy.sqrt(6. / (self.dim_measurements + self.n_hidden_drift)),
                high=0.004 * numpy.sqrt(6. / (self.dim_measurements + self.n_hidden_drift)),
                size=(self.dim_measurements, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )

        self.W_tangent_1 = theano.shared(value=initial_W_tangent_1, name='W_tangent_1', borrow=True)

        self.W_tangent_2 = theano.shared(value=initial_W_tangent_2, name='W_tangent_2', borrow=True)

        self.W_tangent_3 = theano.shared(value=initial_W_tangent_3, name='W_tangent_3', borrow=True)

        self.b_tangent_1 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_1', borrow=True)

        self.b_tangent_2 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_2', borrow=True)

        self.b_tangent_3 = theano.shared(value=numpy.zeros((self.dim_jacobian, ), dtype=theano.config.floatX), name='b_tangent_3', borrow=True)

        self.W_drift_1 = theano.shared(value=initial_W_drift_1, name='W_drift_1', borrow=True)

        self.W_drift_2 = theano.shared(value=initial_W_drift_2, name='W_drift_2', borrow=True)

        self.W_drift_3 = theano.shared(value=initial_W_drift_3, name='W_drift_3', borrow=True)

        self.b_drift_1 = theano.shared(value=numpy.zeros((self.n_hidden_drift,), dtype=theano.config.floatX), name='b_drift_1', borrow=True)

        self.b_drift_2 = theano.shared(value=numpy.zeros((self.n_hidden_drift,), dtype=theano.config.floatX), name='b_drift_2', borrow=True)

        self.b_drift_3 = theano.shared(value=numpy.zeros((self.dim_measurements,), dtype=theano.config.floatX), name='b_drift_3', borrow=True)

        self.measurement_variance = theano.shared(value=local_noise, name='measurement_variance', borrow=True)

        self.get_jacobian_val = theano.function(
            inputs=[self.input_base_Theano],
            outputs=self.get_jacobian(self.input_base_Theano),
        )

        self.get_drift_val = theano.function(
            inputs=[self.input_base_Theano],
            outputs=self.get_drift(self.input_base_Theano),
        )

    def get_drift_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_drift_1, inputs).T + self.b_drift_1.T).T

    def get_drift_hidden_2(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_drift_2, inputs).T + self.b_drift_2.T).T

    def get_drift_output(self, inputs):
        return (T.dot(self.W_drift_3, inputs).T + self.b_drift_3.T).T

    def get_drift(self, inputs_base):
        drift_hidden_1 = self.get_drift_hidden_1(inputs_base)
        drift_hidden_2 = self.get_drift_hidden_2(drift_hidden_1)
        return self.get_drift_output(drift_hidden_2).T

    def get_tangent_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_tangent_1, inputs).T + self.b_tangent_1.T).T

    def get_tangent_hidden_2(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_tangent_2, inputs).T + self.b_tangent_2.T).T

    def get_tangent_output(self, inputs):
        return (T.dot(self.W_tangent_3, inputs).T + self.b_tangent_3.T).T

    def get_jacobian(self, inputs_base):
        tengent_hidden_1 = self.get_tangent_hidden_1(inputs_base)
        tengent_hidden_2 = self.get_tangent_hidden_2(tengent_hidden_1)
        jacobian = T.reshape(self.get_tangent_output(tengent_hidden_2).T, (inputs_base.shape[1], self.dim_measurements, self.dim_intrinsic))
        return jacobian

    def get_cost(self, intrinsic_variance, inputs_base, inputs_step, drift_list, S, R, weight_decay):
        drift = self.get_drift(inputs_base.T)
        jacobian = self.get_jacobian(inputs_base.T)
        JJT = T.batched_dot(jacobian, jacobian.dimshuffle((0, 2, 1)))
        C_temp = JJT*intrinsic_variance
        diag_var = theano.tensor.tile(T.sqr(self.measurement_variance), [inputs_base.shape[0], 3, 3], 3)
        C_temp2 = R*diag_var

        C = C_temp + C_temp2
        C_inv, updates = theano.scan(fn=lambda C_mat: T.nlinalg.matrix_inverse(C_mat), sequences=[C], strict=True)
        C_det, updates = theano.scan(fn=lambda C_mat: T.nlinalg.det(C_mat), sequences=[C], strict=True)
        C_log_det = T.log(C_det)

        S_det, updates = theano.scan(fn=lambda S_mat: T.nlinalg.det(S_mat), sequences=[S], strict=True)
        S_log_det = T.log(S_det)

        before_trace = T.batched_dot(C_inv, S)
        traced, updates = theano.scan(fn=lambda mat: T.nlinalg.trace(mat), sequences=[before_trace], strict=True)
        #cost_drift_init = ((drift_list.T-drift.T)**2).mean()+0.0000001*(T.sum(self.W_drift_1**2)+T.sum(self.W_drift_2**2))
        cost_drift_init = ((drift_list.T-drift.T)**2).mean()+0*(T.sum(self.W_drift_1**2)+T.sum(self.W_drift_2**2))
        #cost_drift = (T.sum(((inputs_step-inputs_base)-drift) ** 2, 1)).mean()

        cost = 0
        cost_cov_init_valid = (T.abs_((C - S)).mean())/(T.abs_(S).mean())
        cost_cov_init_train = cost_cov_init_valid + 0 * (T.sum(self.W_tangent_1 ** 2) + T.sum(self.W_tangent_2 ** 2))

        cost_cov_valid = T.mean(C_log_det - S_log_det + traced - self.dim_measurements)
        cost_cov_train = cost_cov_valid + weight_decay * (T.sum(self.W_tangent_1 ** 2) + T.sum(self.W_tangent_2 ** 2))
        det_part = T.mean(C_log_det)
        trace_part = T.mean(traced)
        return cost, cost_cov_init_valid, cost_cov_init_train, cost_drift_init, cost_cov_train, cost_cov_valid, det_part, trace_part

    @staticmethod
    def gradient_updates_momentum(cost, params, learning_rate, momentum):
        b1 = momentum
        updates = []
        zero = numpy.zeros(1).astype(theano.config.floatX)[0]
        i = theano.shared(zero)
        i_t = i + 1.

        for p in params:

            v = theano.shared(numpy.zeros(p.get_value().shape).astype(dtype=theano.config.floatX))
            g_t = T.grad(cost, p)
            v_t = ((1-b1)*g_t) + (b1 * v)
            p_t = p - learning_rate*v_t

            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))

        return updates


    @staticmethod
    def gradient_updates_adam(cost, params, learning_rate, momentum, momentum2):
        b1 = momentum
        b2 = momentum2
        e = 1e-8
        updates = []
        zero = numpy.zeros(1).astype(theano.config.floatX)[0]
        i = theano.shared(zero)
        i_t = i + 1.

        for p in params:

            m = theano.shared(numpy.zeros(p.get_value().shape).astype(dtype=theano.config.floatX))
            v = theano.shared(numpy.zeros(p.get_value().shape).astype(dtype=theano.config.floatX))
            g_t = T.grad(cost, p)
            m_t = ((1. - b1) * g_t) + (b1 * m)
            m_t_hat = m_t/(1.-b1**i_t)
            v_t = ((1. - b2) * T.sqr(g_t)) + (b2 * v)
            v_t_hat = v_t/(1.-b2**i_t)
            p_t = p - learning_rate * m_t_hat/(T.sqrt(v_t_hat) + e)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))

        return updates

    def train_net(self, intrinsic_variance, batch_size, init_cluster_points_train, init_cluster_points_valid, init_cluster_points_true, init_cov_list_train, init_cov_list_valid, init_cov_list_true, reg_list_train, reg_list_valid, reg_list_true, init_drift_list_train, init_drift_list_valid, init_drift_list_true, drift_iter, cov_init_iter, cov_iter, weight_decay, learning_rate, momentum, momentum2, slowdown, train_var=False):

        n_init_cluster_points_train = init_cluster_points_train.shape[1]
        n_init_cluster_points_valid = init_cluster_points_valid.shape[1]
        n_init_cluster_points_total = init_cluster_points_true.shape[1]

        max_epoch_drift_init = drift_iter
        max_epoch_cov_init = cov_init_iter
        max_epoch_cov = cov_iter

        if train_var:
            params = [self.W_tangent_1, self.W_tangent_2,
                      self.W_tangent_3, self.b_tangent_1,
                      self.b_tangent_2, self.b_tangent_3, self.measurement_variance]
        else:
            params = [self.W_tangent_1, self.W_tangent_2,
                      self.W_tangent_3, self.b_tangent_1,
                      self.b_tangent_2, self.b_tangent_3]

        params3 = [self.W_drift_1, self.W_drift_2,
                   self.W_drift_3, self.b_drift_1,
                   self.b_drift_2, self.b_drift_3]

        cost, cost_cov_init_valid, cost_cov_init_train, cost_drift_init, cost_cov_train, cost_cov_valid, det_part, trace_part = \
            self.get_cost(
            self.intrinsic_variance_Theano, self.input_base_Theano,
            self.input_step_Theano, self.init_drift_Theano, self.init_cov_Theano, self.reg_Theano, weight_decay)

        if max_epoch_drift_init > 0:
            learning_rate = theano.shared(1e-3)
            momentum = theano.shared(momentum)

            updates = self.gradient_updates_momentum(cost_drift_init, params3, learning_rate, momentum)

            train = theano.function(inputs=[self.input_base_Theano, self.init_drift_Theano], outputs=cost_drift_init,
                                    updates=updates)
            train_valid = theano.function(inputs=[self.input_base_Theano, self.init_drift_Theano],
                                          outputs=cost_drift_init, updates=None)

            cost_term = []
            cost_term_valid = []

            iteration = 0
            n_batch = n_init_cluster_points_train
            n_batch = min(n_batch, n_init_cluster_points_train)

            max_iteration_drift_init = (n_init_cluster_points_train/n_batch)*max_epoch_drift_init

            while iteration < max_iteration_drift_init:
                points_in_batch = numpy.random.choice(n_init_cluster_points_train, size=n_batch, replace=False)
                current_cost = train(init_cluster_points_train[:, points_in_batch].reshape((self.dim_measurements, n_batch)).T, init_drift_list_train[points_in_batch, :])
                if n_init_cluster_points_valid > 0:
                    current_valid_cost = train_valid(init_cluster_points_valid[:, :].reshape((self.dim_measurements, n_init_cluster_points_valid)).T, init_drift_list_valid)
                else:
                    current_valid_cost = 0
                print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "cost_batch=", current_cost.mean(), "cost_valid=", current_valid_cost.mean())
                cost_term.append(current_cost)
                cost_term_valid.append(current_valid_cost)
                if iteration % 1000 == 0:
                    learning_rate.set_value(0.98 * learning_rate.get_value())
                iteration += 1

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(cost_term[0:], c='b', label='Train Error')
            ax.plot(cost_term_valid[0:], c='r', label='Valid Error')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Cost')
            ax.set_title("Cost vs Epochs - Drift")
            plt.show(block=False)
            plt.legend()

        if max_epoch_cov_init > 0:
            learning_rate = theano.shared(learning_rate)
            momentum = theano.shared(momentum)
            updates = self.gradient_updates_momentum(cost_cov_init_train, params, learning_rate, momentum)
            train = theano.function(inputs=[self.intrinsic_variance_Theano,
                                            self.input_base_Theano, self.init_cov_Theano, self.reg_Theano],
                                    outputs=cost_cov_init_train,
                                    updates=updates)
            train_valid = theano.function(inputs=[self.intrinsic_variance_Theano,
                                                  self.input_base_Theano, self.init_cov_Theano, self.reg_Theano],
                                          outputs=cost_cov_init_valid,
                                          updates=None)

            cost_term = []
            cost_term_valid = []

            iteration = 0
            n_batch = n_init_cluster_points_train
            n_batch = min(n_batch, n_init_cluster_points_train)

            max_iteration_cov_init = (n_init_cluster_points_train/n_batch)*max_epoch_cov_init

            slow_down_factor = slowdown**(1/max_iteration_cov_init)

            while iteration < max_iteration_cov_init:
                points_in_batch = numpy.random.choice(n_init_cluster_points_train, size=n_batch, replace=False)
                current_cost = train(intrinsic_variance,
                                     init_cluster_points_train[:, points_in_batch].reshape(
                                         (self.dim_measurements, n_batch)).T,
                                     init_cov_list_train[points_in_batch, :, :], reg_list_train[points_in_batch, :, :])
                if n_init_cluster_points_valid > 0:
                    points_in_batch = numpy.random.choice(n_init_cluster_points_valid, size=n_batch, replace=False)
                    current_valid_cost = train_valid(intrinsic_variance,
                                                     init_cluster_points_valid[:, points_in_batch].reshape(
                                                         (self.dim_measurements, n_batch)).T,
                                                     init_cov_list_valid[points_in_batch, :], reg_list_valid[points_in_batch, :])
                else:
                    current_valid_cost = 0

                cost_term.append(current_cost)
                cost_term_valid.append(current_valid_cost)

                iteration += 1

                print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "cost_batch=", current_cost, "cost_valid=", current_valid_cost)

                cost_term.append(current_cost)
                cost_term_valid.append(current_valid_cost)

                learning_rate.set_value(slow_down_factor * learning_rate.get_value())

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(cost_term[0:], c='b', label='Train Error')
            ax.plot(cost_term_valid[0:], c='r', label='Valid Error')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Cost')
            ax.set_title("Cost vs Epochs - Cov Init")
            plt.legend()

        if max_epoch_cov > 0:
            learning_rate = theano.shared(learning_rate)
            momentum = theano.shared(momentum)
            if momentum2 > 0:
                momentum2 = theano.shared(momentum2)
                updates = self.gradient_updates_adam(cost_cov_train, params, learning_rate, momentum, momentum2)
            else:
                updates = self.gradient_updates_momentum(cost_cov_train, params, learning_rate, momentum)

            train_func = theano.function(inputs=[self.intrinsic_variance_Theano,
                                            self.input_base_Theano, self.init_cov_Theano, self.reg_Theano],
                                    outputs=cost_cov_train,
                                    updates=updates)
            test_func = theano.function(inputs=[self.intrinsic_variance_Theano,
                                                  self.input_base_Theano, self.init_cov_Theano, self.reg_Theano],
                                          outputs=cost_cov_valid, updates=None)

            iteration = 0
            n_batch = batch_size
            n_batch = min(n_batch, n_init_cluster_points_train)
            max_iteration_cov = (n_init_cluster_points_train / n_batch) * max_epoch_cov
            slow_down_factor = slowdown**(1/max_iteration_cov)

            train_cost_log = []
            train_error_log = []
            valid_error_log = []
            test_error_log = []

            while iteration < max_iteration_cov:
                points_in_batch = numpy.random.choice(n_init_cluster_points_train, size=n_batch, replace=False)
                if n_init_cluster_points_valid > 0:
                    current_valid_error = test_func(intrinsic_variance, init_cluster_points_valid.reshape((self.dim_measurements, n_init_cluster_points_valid)).T, init_cov_list_valid, reg_list_valid)
                    current_train_error = test_func(intrinsic_variance,
                                                    init_cluster_points_train.reshape(
                                                        (self.dim_measurements, n_init_cluster_points_train)).T,
                                                    init_cov_list_train, reg_list_train)
                    current_true_error = test_func(intrinsic_variance,
                                                   init_cluster_points_true.reshape(
                                                       (self.dim_measurements, n_init_cluster_points_total)).T,
                                                   init_cov_list_true, reg_list_true)
                else:
                    current_valid_error = 0
                    current_train_error = 0
                    current_true_error = 0

                current_batch_train_cost = train_func(intrinsic_variance, init_cluster_points_train[:, points_in_batch].reshape((self.dim_measurements, n_batch)).T, init_cov_list_train[points_in_batch, :, :], reg_list_train[points_in_batch, :, :])

                train_cost_log.append(current_batch_train_cost)
                train_error_log.append(current_train_error)
                valid_error_log.append(current_valid_error)
                test_error_log.append(current_true_error)

                iteration += 1

                print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "train_cost_batch=", current_batch_train_cost,
                      "train_error=", current_train_error, "valid_error=", current_valid_error, "test_error=", current_true_error)

                learning_rate.set_value(slow_down_factor * learning_rate.get_value())

            if n_init_cluster_points_valid > 0:
                current_valid_error = test_func(intrinsic_variance,
                                                init_cluster_points_valid.reshape(
                                                    (self.dim_measurements, n_init_cluster_points_valid)).T,
                                                init_cov_list_valid, reg_list_valid)
            else:
                current_valid_error = 0

            current_train_error = test_func(intrinsic_variance,
                                                init_cluster_points_train.reshape(
                                                    (self.dim_measurements, n_init_cluster_points_train)).T,
                                                init_cov_list_train, reg_list_train)
            current_true_error = test_func(intrinsic_variance,
                                       init_cluster_points_true.reshape(
                                           (self.dim_measurements, n_init_cluster_points_total)).T,
                                       init_cov_list_true, reg_list_true)

            current_batch_train_cost = train_func(intrinsic_variance,
                                                  init_cluster_points_train[:, points_in_batch].reshape(
                                                      (self.dim_measurements, n_batch)).T,
                                                  init_cov_list_train[points_in_batch, :, :],
                                                  reg_list_train[points_in_batch, :, :])

            train_cost_log.append(current_batch_train_cost)
            train_error_log.append(current_train_error)
            valid_error_log.append(current_valid_error)
            test_error_log.append(current_true_error)

            print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "train_cost_batch=",
                  current_batch_train_cost,
                  "train_error=", current_train_error, "valid_error=", current_valid_error, "test_error=",
                  current_true_error)

            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(train_error_log[0:], c='k', label='Train Error')
            if n_init_cluster_points_valid > 0:
                ax.plot(valid_error_log[0:], c='g', label='Valid Error')
            ax.plot(test_error_log[0:], c='r', label='Test Error')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Cost')
            ax.set_title("Cost vs Epochs - Cov")
            plt.legend()
            '''
            logs = numpy.asarray([train_cost_log, train_error_log, valid_error_log, test_error_log])

            return logs

    plt.show(block=False)
