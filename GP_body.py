from scipy import optimize
import tensorflow as tf
import copy
from GP_tools import *
import scipy.io
from scipy.io import savemat
from scipy.io import loadmat
import time
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

@tf.function
def scale_func(pos, theta):
    return (theta[1] + theta[2]) / 2 - (theta[1] - theta[2]) / 2 * tf.math.tanh((pos - theta[3]) / theta[4])

def scale_func_NUTS(pos, l1, l2, x0, lw):
    return (l1 + l2) / 2 - (l1 - l2) / 2 * tf.math.tanh((pos - x0) / lw)

def kernel(pos_row, pos_col, err, theta):
    row = np.shape(pos_row)[0]
    col = np.shape(pos_col)[0]
    pos_row_in = tf.constant(copy.deepcopy(pos_row), dtype=tf.float64)
    pos_col_in = tf.constant(copy.deepcopy(pos_col), dtype=tf.float64)

    l_x1 = scale_func(pos_row_in, theta)
    l_x1_box = tf.broadcast_to(l_x1, [row, col])
    l_x2 = scale_func(pos_col_in, theta)
    l_x2_box = tf.broadcast_to(tf.transpose(l_x2), [row, col])

    l_multi = tf.math.multiply(l_x1_box, l_x2_box)
    l_x_sqr_final = tf.square(l_x1_box) + tf.square(l_x2_box)
    input_diff = tf.square(
        tf.broadcast_to(pos_row_in, [row, col]) - tf.broadcast_to(tf.transpose(pos_col_in), [row, col]))

    first_component = tf.sqrt(tf.math.divide_no_nan(2 * l_multi, l_x_sqr_final))
    second_component = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff, l_x_sqr_final))
    k = (theta[0] ** 2 * tf.multiply(first_component, second_component)) + err
    return k

def kernel_NUTS(pos_row, pos_col, err, sig, l1, l2, x0, lw):
    row = np.shape(pos_row)[0]
    col = np.shape(pos_col)[0]
    pos_row_in = tf.constant(copy.deepcopy(pos_row), dtype=tf.float64)
    pos_col_in = tf.constant(copy.deepcopy(pos_col), dtype=tf.float64)

    l_x1 = scale_func_NUTS(pos_row_in, l1, l2, x0, lw)
    l_x1_box = tf.broadcast_to(l_x1, [row, col])
    l_x2 = scale_func_NUTS(pos_col_in, l1, l2, x0, lw)
    l_x2_box = tf.broadcast_to(tf.transpose(l_x2), [row, col])

    l_multi = tf.math.multiply(l_x1_box, l_x2_box)
    l_x_sqr_final = tf.square(l_x1_box) + tf.square(l_x2_box)
    input_diff = tf.square(
        tf.broadcast_to(pos_row_in, [row, col]) - tf.broadcast_to(tf.transpose(pos_col_in), [row, col]))

    first_component = tf.sqrt(tf.math.divide_no_nan(2 * l_multi, l_x_sqr_final))
    second_component = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff, l_x_sqr_final))

    k = (sig ** 2 * tf.multiply(first_component, second_component)) + err
    return k

def Grad_Kernel(pos_grad, pos_temp, theta):
    row = np.shape(pos_grad)[0]
    col = np.shape(pos_temp)[0]
    pos_grad_tf = tf.constant(copy.deepcopy(pos_grad), dtype=tf.float64)
    pos_temp_tf = tf.constant(copy.deepcopy(pos_temp), dtype=tf.float64)

    l_x1 = scale_func(pos_grad_tf, theta)
    grad_l_x1 = -0.5 * (theta[1] - theta[2]) / (theta[4] * tf.square(tf.math.cosh((pos_grad_tf - theta[3])/theta[4])))
    grad_l_x1 = tf.broadcast_to(grad_l_x1, [row, col])
    l_x2 = scale_func(pos_temp_tf, theta)

    l_x1_box = tf.broadcast_to(l_x1, [row, col])
    l_x2_box = tf.broadcast_to(tf.transpose(l_x2), [row, col])

    l_multi = tf.math.multiply(l_x1_box, l_x2_box)
    l_x_sqr_final = tf.math.square(l_x1_box) + tf.math.square(l_x2_box)

    input_diff = tf.broadcast_to(pos_grad_tf, [row, col]) - tf.broadcast_to(tf.transpose(pos_temp_tf), [row, col])
    input_diff_sqr = tf.math.square(input_diff)

    first_comp = tf.math.sqrt(tf.math.divide_no_nan(2 * l_multi, l_x_sqr_final))
    second_comp = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff_sqr, l_x_sqr_final))

    k = theta[0] ** 2 * tf.math.multiply(first_comp, second_comp)

    third = tf.math.divide_no_nan(-2 * input_diff, l_x_sqr_final)
    forth = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), l_x1_box)
    fifth = tf.math.multiply(tf.math.divide_no_nan(-2 * l_x1_box, l_x_sqr_final),
                             1 - tf.math.divide_no_nan(2 * input_diff_sqr, l_x_sqr_final))

    grad_kernel = tf.math.multiply(k, third + tf.math.multiply(0.5 * grad_l_x1, forth + fifth))

    return grad_kernel

def Grad_Kernel_NUTS(pos_grad, pos_temp, sig, l1, l2, x0, lw):
    row = np.shape(pos_grad)[0]
    col = np.shape(pos_temp)[0]
    pos_grad_tf = tf.constant(copy.deepcopy(pos_grad), dtype=tf.float64)
    pos_temp_tf = tf.constant(copy.deepcopy(pos_temp), dtype=tf.float64)

    l_x1 = scale_func_NUTS(pos_grad_tf, l1, l2, x0, lw)
    grad_l_x1 = -0.5 * (l1 - l2) / (lw * tf.square(tf.math.cosh((pos_grad_tf - x0)/lw)))
    grad_l_x1 = tf.broadcast_to(grad_l_x1, [row, col])
    l_x2 = scale_func_NUTS(pos_temp_tf, l1, l2, x0, lw)

    l_x1_box = tf.broadcast_to(l_x1, [row, col])
    l_x2_box = tf.broadcast_to(tf.transpose(l_x2), [row, col])

    l_multi = tf.math.multiply(l_x1_box, l_x2_box)
    l_x_sqr_final = tf.math.square(l_x1_box) + tf.math.square(l_x2_box)

    input_diff = tf.broadcast_to(pos_grad_tf, [row, col]) - tf.broadcast_to(tf.transpose(pos_temp_tf), [row, col])
    input_diff_sqr = tf.math.square(input_diff)

    first_comp = tf.math.sqrt(tf.math.divide_no_nan(2 * l_multi, l_x_sqr_final))
    second_comp = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff_sqr, l_x_sqr_final))

    k = sig ** 2 * tf.math.multiply(first_comp, second_comp)

    third = tf.math.divide_no_nan(-2 * input_diff, l_x_sqr_final)
    forth = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), l_x1_box)
    fifth = tf.math.multiply(tf.math.divide_no_nan(-2 * l_x1_box, l_x_sqr_final),
                             1 - tf.math.divide_no_nan(2 * input_diff_sqr, l_x_sqr_final))

    grad_kernel = tf.math.multiply(k, third + tf.math.multiply(0.5 * grad_l_x1, forth + fifth))

    return grad_kernel

def Dual_Grad(Pos_grad_row, Pos_grad_col, Err, theta):
    row = np.shape(Pos_grad_row)[0]
    col = np.shape(Pos_grad_col)[0]
    pos_grad_row_tf = tf.constant(copy.deepcopy(Pos_grad_row), dtype=tf.float64)
    pos_grad_col_tf = tf.constant(copy.deepcopy(Pos_grad_col), dtype=tf.float64)

    L_x1 = scale_func(pos_grad_row_tf, theta)
    grad_L_x1 = -0.5 * (theta[1] - theta[2]) / (theta[4] * tf.square(tf.math.cosh((pos_grad_row_tf - theta[3])/theta[4])))
    grad_L_x1 = tf.broadcast_to(grad_L_x1, [row,col])

    L_x2 = scale_func(pos_grad_col_tf, theta)
    grad_L_x2 = -0.5 * (theta[1] - theta[2]) / (theta[4] * tf.square(tf.math.cosh((pos_grad_col_tf - theta[3])/theta[4])))
    grad_L_x2 = tf.broadcast_to(tf.transpose(grad_L_x2), [row, col])

    # 원래 kernel matrix 구하기
    L_multi = tf.linalg.matmul(L_x1, tf.transpose(L_x2))
    L_x_sqr_final = tf.broadcast_to(tf.math.square(L_x1), [row, col]) + tf.broadcast_to(
        tf.transpose(tf.math.square(L_x2)), [row, col])

    input_diff = tf.broadcast_to(pos_grad_row_tf, [row, col]) - tf.broadcast_to(tf.transpose(pos_grad_col_tf),
                                                                                [row, col])
    input_diff_sqr = tf.math.square(input_diff)

    First_comp = tf.math.sqrt(tf.math.divide_no_nan(2 * L_multi, L_x_sqr_final))
    Second_comp = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff_sqr, L_x_sqr_final))

    K = theta[0] ** 2 * tf.math.multiply(First_comp, Second_comp)

    # Hessian의 첫 번째 comp 구하기
    third = tf.math.divide_no_nan(-2 * input_diff, L_x_sqr_final)
    forth = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), tf.broadcast_to(L_x1, [row, col]))
    fifth = tf.math.multiply(tf.math.divide_no_nan(-2 * tf.broadcast_to(L_x1, [row, col]), L_x_sqr_final),
                             1 - tf.math.divide_no_nan(2 * input_diff_sqr, L_x_sqr_final))

    third_2 = tf.math.divide_no_nan(2 * input_diff, L_x_sqr_final)
    forth_2 = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), tf.broadcast_to(tf.transpose(L_x2), [row, col]))
    fifth_2 = tf.math.multiply(
        tf.math.divide_no_nan(-2 * tf.broadcast_to(tf.transpose(L_x2), [row, col]), L_x_sqr_final),
        1 - tf.math.divide_no_nan(2 * input_diff_sqr, L_x_sqr_final))

    dual_grad_first = tf.math.multiply(tf.math.multiply(K, third + tf.math.multiply(0.5 * grad_L_x1, forth + fifth)),
                                       third_2 + tf.math.multiply(0.5 * grad_L_x2, forth_2 + fifth_2))

    # Hessian의 두 번째 comp 구하기
    l1_sqr_deriva = tf.math.multiply(tf.broadcast_to(L_x1, [row, col]), grad_L_x1)
    l2_sqr_deriva = tf.math.multiply(tf.broadcast_to(tf.transpose(L_x2), [row, col]), grad_L_x2)
    alpha = L_x_sqr_final + 2 * tf.math.multiply(input_diff, l2_sqr_deriva)
    beta = tf.math.multiply(tf.math.multiply(l1_sqr_deriva, l2_sqr_deriva),
                            1 - tf.math.divide_no_nan(4 * input_diff_sqr, L_x_sqr_final))
    gamma = tf.math.multiply(2 * l1_sqr_deriva, -1 * input_diff)

    dual_grad_second = tf.math.divide_no_nan(tf.math.multiply(2 * K, alpha + beta + gamma),
                                             tf.math.square(L_x_sqr_final))
    dual_grad = (dual_grad_first + dual_grad_second) + Err
    return dual_grad

def Dual_Grad_NUTS(Pos_grad_row, Pos_grad_col, Err, sig, l1, l2, x0, lw):
    row = np.shape(Pos_grad_row)[0]
    col = np.shape(Pos_grad_col)[0]
    pos_grad_row_tf = tf.constant(copy.deepcopy(Pos_grad_row), dtype=tf.float64)
    pos_grad_col_tf = tf.constant(copy.deepcopy(Pos_grad_col), dtype=tf.float64)

    L_x1 = scale_func_NUTS(pos_grad_row_tf, l1, l2, x0, lw)
    grad_L_x1 = -0.5 * (l1 - l2) / (lw * tf.square(tf.math.cosh((pos_grad_row_tf - x0)/lw)))
    grad_L_x1 = tf.broadcast_to(grad_L_x1, [row,col])

    L_x2 = scale_func_NUTS(pos_grad_col_tf, l1, l2, x0, lw)
    grad_L_x2 = -0.5 * (l1 - l2) / (lw * tf.square(tf.math.cosh((pos_grad_col_tf - x0)/lw)))
    grad_L_x2 = tf.broadcast_to(tf.transpose(grad_L_x2), [row, col])

    # 원래 kernel matrix 구하기
    L_multi = tf.linalg.matmul(L_x1, tf.transpose(L_x2))
    L_x_sqr_final = tf.broadcast_to(tf.math.square(L_x1), [row, col]) + tf.broadcast_to(
        tf.transpose(tf.math.square(L_x2)), [row, col])

    input_diff = tf.broadcast_to(pos_grad_row_tf, [row, col]) - tf.broadcast_to(tf.transpose(pos_grad_col_tf),
                                                                                [row, col])
    input_diff_sqr = tf.math.square(input_diff)

    First_comp = tf.math.sqrt(tf.math.divide_no_nan(2 * L_multi, L_x_sqr_final))
    Second_comp = tf.math.exp(-1 * tf.math.divide_no_nan(input_diff_sqr, L_x_sqr_final))

    K = sig ** 2 * tf.math.multiply(First_comp, Second_comp)

    # Hessian의 첫 번째 comp 구하기
    third = tf.math.divide_no_nan(-2 * input_diff, L_x_sqr_final)
    forth = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), tf.broadcast_to(L_x1, [row, col]))
    fifth = tf.math.multiply(tf.math.divide_no_nan(-2 * tf.broadcast_to(L_x1, [row, col]), L_x_sqr_final),
                             1 - tf.math.divide_no_nan(2 * input_diff_sqr, L_x_sqr_final))

    third_2 = tf.math.divide_no_nan(2 * input_diff, L_x_sqr_final)
    forth_2 = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64),
                                    tf.broadcast_to(tf.transpose(L_x2), [row, col]))
    fifth_2 = tf.math.multiply(
        tf.math.divide_no_nan(-2 * tf.broadcast_to(tf.transpose(L_x2), [row, col]), L_x_sqr_final),
        1 - tf.math.divide_no_nan(2 * input_diff_sqr, L_x_sqr_final))

    dual_grad_first = tf.math.multiply(tf.math.multiply(K, third + tf.math.multiply(0.5 * grad_L_x1, forth + fifth)),
                                       third_2 + tf.math.multiply(0.5 * grad_L_x2, forth_2 + fifth_2))

    # Hessian의 두 번째 comp 구하기
    l1_sqr_deriva = tf.math.multiply(tf.broadcast_to(L_x1, [row, col]), grad_L_x1)
    l2_sqr_deriva = tf.math.multiply(tf.broadcast_to(tf.transpose(L_x2), [row, col]), grad_L_x2)
    alpha = L_x_sqr_final + 2 * tf.math.multiply(input_diff, l2_sqr_deriva)
    beta = tf.math.multiply(tf.math.multiply(l1_sqr_deriva, l2_sqr_deriva),
                            1 - tf.math.divide_no_nan(4 * input_diff_sqr, L_x_sqr_final))
    gamma = tf.math.multiply(2 * l1_sqr_deriva, -1 * input_diff)

    dual_grad_second = tf.math.divide_no_nan(tf.math.multiply(2 * K, alpha + beta + gamma),
                                             tf.math.square(L_x_sqr_final))
    dual_grad = (dual_grad_first + dual_grad_second) + Err
    return dual_grad

class GPR():
    def __init__(self, Pos_regression, Pos_Temp, Temp, err, Pos_Grad, Grad, Grad_Err,
                 alpha0, beta0, alpha1, beta1, alpha2, beta2, alpha3, beta3,
                 x0_start, alpha4, beta4, bounds, mode, theta_0, target, burn, KSTAR_shot, diag_time):
        self.Pos_regression = Pos_regression
        self.Pos_Temp = Pos_Temp
        self.Temp = Temp
        self.err = err
        self.Pos_Grad = Pos_Grad
        self.Grad = Grad
        self.Grad_Err = Grad_Err
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.alpha3 = alpha3
        self.beta3 = beta3
        self.x0_start = x0_start
        self.alpha4 = alpha4
        self.beta4 = beta4
        self.bounds = bounds
        self.mode = mode
        self.theta_0 = theta_0
        self.target = target
        self.burn = burn
        self.KSTAR_shot = KSTAR_shot
        self.diag_time = diag_time

    def get_kernel_MAP(self, theta):
        Hess = Dual_Grad(self.Pos_Grad, self.Pos_Grad, self.Grad_Err, theta).numpy()
        Kernel_Origianl = kernel(self.Pos_Temp, self.Pos_Temp, self.err, theta).numpy()
        Operated_Kernel = Grad_Kernel(self.Pos_Grad, self.Pos_Temp, theta).numpy()
        Temp1 = np.concatenate((Hess, Operated_Kernel), axis=1)
        Temp2 = np.concatenate((Operated_Kernel.T, Kernel_Origianl), axis=1)
        Kernel = np.concatenate((Temp1, Temp2), axis=0)
        return Kernel

    def get_kernel_NUTS(self, sig, l1, l2, x0, lw):
        Hess = Dual_Grad_NUTS(self.Pos_Grad, self.Pos_Grad, self.Grad_Err, sig, l1, l2, x0, lw)
        Kernel_Origianl = kernel_NUTS(self.Pos_Temp, self.Pos_Temp, self.err, sig, l1, l2, x0, lw)
        Operated_Kernel = Grad_Kernel_NUTS(self.Pos_Grad, self.Pos_Temp, sig, l1, l2, x0, lw)
        temp1 = tf.concat((Hess, Operated_Kernel), 1)
        temp2 = tf.concat((tf.transpose(Operated_Kernel), Kernel_Origianl), 1)
        Kernel = tf.concat((temp1, temp2), 0)
        return Kernel

    def log_likelihood_MAP(self, theta):
        Kernel = self.get_kernel_MAP(theta)
        likelihood = (np.log(np.absolute(np.linalg.det(Kernel)))
                      + np.matmul(np.concatenate((self.Grad, self.Temp), axis=0).T,
                                  np.matmul(np.linalg.inv(Kernel), np.concatenate((self.Grad, self.Temp), axis=0))))

        U = (likelihood[0, 0]
             - (self.alpha0 - 1) * np.log(theta[0]) + self.beta0 * theta[0]
             - (self.alpha1 - 1) * np.log(theta[1]) + self.beta1 * theta[1]
             - (self.alpha2 - 1) * np.log(theta[2]) + self.beta2 * theta[2]
             - (self.alpha3 - 1) * np.log(theta[3] - self.x0_start) + self.beta3 * theta[3]
             - (self.alpha4 - 1) * np.log(theta[4]) + self.beta4 * theta[4]
             )
        return U

    def log_likelihood_NUTS(self, sig, l1, l2, x0, lw):
        Kernel = self.get_kernel_NUTS(sig, l1, l2, x0, lw)
        likelihood = 0.5 * (tf.math.log(tf.abs(tf.linalg.det(Kernel)))
                            + tf.linalg.matmul(tf.transpose(tf.concat((tf.constant(self.Grad, dtype=tf.float64),
                                                                       tf.constant(self.Temp, dtype=tf.float64)), 0)),
                                               tf.linalg.matmul(tf.linalg.inv(Kernel),
                                                                tf.concat((tf.constant(self.Grad, dtype=tf.float64),
                                                                           tf.constant(self.Temp, dtype=tf.float64)),
                                                                          0))))
        U = (likelihood[0, 0]
             - (self.alpha0 - 1) * tf.math.log(sig) + self.beta0 * sig
             - (self.alpha1 - 1) * tf.math.log(l1) + self.beta1 * l1
             - (self.alpha2 - 1) * tf.math.log(l2) + self.beta2 * l2
             - (self.alpha3 - 1) * tf.math.log(x0 - self.x0_start) + self.beta3 * x0
             - (self.alpha4 - 1) * tf.math.log(lw) + self.beta4 * lw
             )

        return -1 * U

    def prediction(self, sig, l1, l2, x0, lw):
        # Temperature prediction
        K = np.concatenate((Grad_Kernel_NUTS(self.Pos_Grad, self.Pos_regression, sig, l1, l2, x0, lw).numpy(),
                            kernel_NUTS(self.Pos_Temp, self.Pos_regression, 0, sig, l1, l2, x0, lw).numpy()), axis=0)
        Kernel = self.get_kernel_NUTS(sig, l1, l2, x0, lw).numpy()
        mu = np.matmul(K.T, np.matmul(np.linalg.inv(Kernel), np.concatenate((self.Grad, self.Temp), axis=0)))
        sigma = np.absolute(kernel_NUTS(self.Pos_regression, self.Pos_regression, 0, sig, l1, l2, x0, lw).numpy()
                            - np.matmul(K.T, np.matmul(np.linalg.inv(Kernel), K)))
        # Gradient prediction
        dual_grad1 = Dual_Grad_NUTS(self.Pos_Grad, self.Pos_regression, 0, sig, l1, l2, x0, lw).numpy()
        dual_grad2 = (Grad_Kernel_NUTS(self.Pos_regression, self.Pos_Temp, sig, l1, l2, x0, lw).numpy()).T
        dual_grad = np.concatenate((dual_grad1, dual_grad2), axis=0)
        g_mu = np.matmul(dual_grad.T, np.matmul(np.linalg.inv(Kernel), np.concatenate((self.Grad, self.Temp), axis=0)))
        g_sigma = np.absolute(Dual_Grad_NUTS(self.Pos_regression, self.Pos_regression, 0, sig, l1, l2, x0, lw).numpy()
                              - np.matmul(dual_grad.T, np.matmul(np.linalg.inv(Kernel), dual_grad)))
        return (mu, sigma, g_mu, g_sigma)

    def MAP(self):
        start = time.time()
        if self.mode == 'H':
            def l1_l2(theta):
                return theta[1] - theta[2] - 0.01

        elif self.mode == 'L':
            def l1_l2(theta):
                return theta[1] - theta[2]

        else:
            print("You typed the wrong mode.")

        cons = ({'type': 'ineq', 'fun': l1_l2})

        Results = optimize.shgo(self.log_likelihood_MAP, self.bounds, n=32, iters=3, constraints=cons)
        # Temperature prediction
        K = np.concatenate((Grad_Kernel(self.Pos_Grad, self.Pos_regression, Results['x']).numpy(),
                            kernel(self.Pos_Temp, self.Pos_regression, 0, Results['x']).numpy()), axis=0)
        Kernel = self.get_kernel_MAP(Results['x'])
        mu = np.matmul(K.T, np.matmul(np.linalg.inv(Kernel), np.concatenate((self.Grad, self.Temp), axis=0)))
        sigma = np.absolute(kernel(self.Pos_regression, self.Pos_regression, 0, Results['x']).numpy()
                            - np.matmul(K.T, np.matmul(np.linalg.inv(Kernel), K)))
        # Gradient prediction
        dual_grad1 = Dual_Grad(self.Pos_Grad, self.Pos_regression, 0, Results['x']).numpy()
        dual_grad2 = (Grad_Kernel(self.Pos_regression, self.Pos_Temp, Results['x']).numpy()).T
        dual_grad = np.concatenate((dual_grad1, dual_grad2), axis=0)
        g_mu = np.matmul(dual_grad.T, np.matmul(np.linalg.inv(Kernel), np.concatenate((self.Grad, self.Temp), axis=0)))
        g_sigma = np.absolute(Dual_Grad(self.Pos_regression, self.Pos_regression, 0, Results['x']).numpy()
                              - np.matmul(dual_grad.T, np.matmul(np.linalg.inv(Kernel), dual_grad)))

        data_file = {}
        data_file['MAP_param_Ti'] = Results
        data_file['MAP_mu_Ti'] = mu
        data_file['MAP_sig_Ti'] = sigma
        data_file['MAP_gmu_Ti'] = g_mu
        data_file['MAP_gsig_Ti'] = g_sigma

        savemat('./data/MAP_result.mat', data_file)

        print("Time for MAP : %.3f" % (time.time() - start))
        print('Ti Sig_f = {0:>.2e}, l1 = {1:>.2e}, l2 = {2:>.2e}, x0 = {3:>.2e}, lw = {4:>.2e}'
              .format(Results['x'][0], Results['x'][1], Results['x'][2], Results['x'][3], Results['x'][4]))

        return ()

    def NUTS(self):
        num_results = self.burn + self.target

        ####### 모든 Hyper-parameter를 0보다 크게 만들어주는 constraint이다. ##########################################
        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

        ####### L을 정해줄 필요가 없는 NUTS를 정의한 코드이다. #########################################################
        sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.log_likelihood_NUTS,
                step_size=tf.cast(0.1, tf.float64)),
            bijector=[constrain_positive, constrain_positive, constrain_positive,
                      constrain_positive, constrain_positive])

        ####### eps를 정해줄 필요가 없는 dual averaging step size adaptation를 정의한 코드이다. #######################
        adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler,
            num_adaptation_steps=int(0.8 * self.burn),
            target_accept_prob=tf.cast(0.90, tf.float64))

        ####### 실제 sampling을 해줄 코드이다. #####################################################################
        @tf.function(autograph=False)
        def do_sampling():
            return tfp.mcmc.sample_chain(
                kernel=adaptive_sampler,
                current_state=[self.theta_0[0], self.theta_0[1], self.theta_0[2], self.theta_0[3], self.theta_0[4]],
                num_results=num_results,
                num_burnin_steps=self.burn,
                trace_fn=lambda current_state, kernel_results: kernel_results)

        t0 = time.time()
        samples, kernel_results = do_sampling()
        t1 = time.time()
        print("Time for NUTS : {:.2f}s.".format(t1 - t0))
        (sig_f, l1, l2, x0, lw) = samples
        sig_f = sig_f.numpy()
        l1 = l1.numpy()
        l2 = l2.numpy()
        x0 = x0.numpy()
        lw = lw.numpy()

        U_result = (kernel_results[0][1][0]).numpy() * -1
        step_size = np.array(kernel_results[0][1][3])
        accp = (tf.math.exp(kernel_results[0][1][4])).numpy()
        is_accepted = (kernel_results[0][1][6]).numpy()

        temp = 0
        for i in range(np.shape(is_accepted)[0]):
            if is_accepted[i] == True:
                temp += 1
        rate = temp / np.shape(is_accepted)[0] * 100

        Sample_result = {}
        Sample_result['NUTS_Samples'] = samples
        Sample_result['NUTS_Potential'] = U_result
        Sample_result['NUTS_Acceptence'] = accp
        Sample_result['NUTS_Accep_rate'] = rate
        Sample_result['NUTS_is_accp'] = is_accepted
        Sample_result['NUTS_Eps'] = step_size
        savemat('./Data/NUTS_samples.mat', Sample_result)

        Sampling_Plot(U_result, sig_f, l1, l2, x0, lw,
                      './NUTS_figure/Sampling.png')

        plot_Acceptence(accp[:], rate, './NUTS_figure/Accptence.png')

        B_Sig_f = sig_f[self.burn:]
        B_l1 = l1[self.burn:]
        B_l2 = l2[self.burn:]
        B_x0 = x0[self.burn:]
        B_lw = lw[self.burn:]

        ###### FFT를 사용하여 autocorrelation을 계산하는 코드이다. ####################################################
        (Sig_rho, Sig_eff_rho_idx, Sig_IAT, Sig_ESS) = get_ESS(B_Sig_f, 0.05)
        (l1_rho, l1_eff_rho_idx, l1_IAT, l1_ESS) = get_ESS(B_l1, 0.05)
        (l2_rho, l2_eff_rho_idx, l2_IAT, l2_ESS) = get_ESS(B_l2, 0.05)
        (x0_rho, x0_eff_rho_idx, x0_IAT, x0_ESS) = get_ESS(B_x0, 0.05)
        (lw_rho, lw_eff_rho_idx, lw_IAT, lw_ESS) = get_ESS(B_lw, 0.05)
        #########################################################################################################

        rho_result = [Sig_rho, l1_rho, l2_rho, x0_rho, lw_rho]
        eff_rho_idx = [Sig_eff_rho_idx, l1_eff_rho_idx, l2_eff_rho_idx,
                       x0_eff_rho_idx, lw_eff_rho_idx]
        FFT_result = [Sig_IAT, l1_IAT, l2_IAT, x0_IAT, lw_IAT]
        ESS_result = [Sig_ESS, l1_ESS, l2_ESS, x0_ESS, lw_ESS]

        Sample_result['NUTS_rho'] = rho_result
        Sample_result['NUTS_eff_rho_idx'] = eff_rho_idx
        Sample_result['NUTS_IAT'] = FFT_result
        Sample_result['NUTS_ESS'] = ESS_result
        Sample_result['NUTS_burn'] = self.burn
        Sample_result['NUTS_target'] = self.target

        savemat('./Data/NUTS_total.mat', Sample_result)
        return ()