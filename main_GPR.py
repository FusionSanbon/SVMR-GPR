from GP_body import *
from GP_tools import *
import warnings
import os

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    os.mkdir('./Figure/')
    os.mkdir('./Data/')
    os.mkdir('./NUTS_figure/')
except:
    print("The folders already exist.")

num_of_prediction = 100
axis_pos = 1.8

function_pos = np.round(np.atleast_2d(np.linspace(1.8, 2.30, 30)).T, 3)
function_val = np.zeros([30, 1])
function_err = np.zeros([30, 1])

for i in range(30):
    if function_pos[i, 0] <= 2.2:
        function_val[i, 0] = 1+2*np.cos((function_pos[i, 0] - 1.8)*(5*np.pi/4)) + np.random.normal(0, 0.1)
        function_err[i, 0] = 0.1
    else:
        function_val[i, 0] = 1 - np.cos((function_pos[i, 0] - 2.3) * (5 * np.pi)) + np.random.normal(0, 0.02)
        function_err[i, 0] = 0.02

pos_pred = np.round(np.atleast_2d(np.linspace(1.8, 2.30, num_of_prediction)).T, 3)

syn_data = {}
syn_data['position'] = function_pos
syn_data['value'] = function_val
syn_data['error'] = function_err
syn_data['position_prediction'] = pos_pred
savemat('./Data/synthetic_data.mat', syn_data)

######### Prior ###############################################################################
l1_mode = 0.25; l1_std = 0.2
alpha1, beta1 = get_l1_l2_prior(l1_mode, l1_std ** 2)

l2_mode = 0.25; l2_std = 0.2
alpha2, beta2 = get_l1_l2_prior(l2_mode, l2_std ** 2)

x0_mode = 0.25; x0_std = 0.2
x0_start = 1.80
alpha3, beta3 = get_l1_l2_prior(x0_mode, x0_std ** 2)

lw_mode = 10 ** (-1); lw_std = 10 ** (-1)
alpha4, beta4 = get_l1_l2_prior(lw_mode, lw_std ** 2)

sig_mode_Ti = 0.5 * np.max(function_val); sig_std = 1.0
alpha0_Ti, beta0_Ti = get_l1_l2_prior(sig_mode_Ti, sig_std ** 2)

######## Model uncertainty ###################################################################
GP_Err_Ti = np.diagflat(np.square(function_err[:, 0]))

######## Searching boundary for MAP estimator ################################################
bounds_Ti = [(0.01, np.max(function_val)), (0.01, 0.5), (0.01, 0.5), (1.80, 2.30), (10**(-3), 0.3)]

############## Gradient information ##########################################################
Pos_Grad = np.atleast_2d([axis_pos, 2.30]).T
Grad = np.atleast_2d([0.1, 0.1]).T
Grad_Err = np.array([[1, 0], [0, 1]])

############# Settings for the marginalization ###############################################
burn = 1000
target = 10000

theta0_Ti = tf.constant([np.random.uniform(0.01, sig_mode_Ti + sig_std),
                       np.random.uniform(0.01, l1_mode + l1_std),
                       np.random.uniform(0.01, l2_mode + l2_std),
                       np.random.uniform(x0_start + x0_mode - x0_std, x0_start + x0_mode + x0_std),
                       np.random.uniform(10**(-3), 1)], tf.float64)

### Setting our model ######################################################################
Minseok_GPR = GPR(pos_pred, function_pos, function_val, GP_Err_Ti, Pos_Grad, Grad, Grad_Err,
            alpha0_Ti, beta0_Ti, alpha1, beta1, alpha2, beta2, alpha3, beta3,
            x0_start, alpha4, beta4, bounds_Ti, 'H', theta0_Ti, target, burn, 0000, 0.0)

Minseok_GPR.MAP()
Minseok_GPR.NUTS()
