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

syn_data = loadmat('./Data/synthetic_data.mat')
function_pos = copy.deepcopy(syn_data['position'])
function_val = copy.deepcopy(syn_data['value'])
function_err = copy.deepcopy(syn_data['error'])
pos_pred = copy.deepcopy(syn_data['position_prediction'])

######### Prior ###############################################################################
l1_mode = 0.25
l1_std = 0.2
alpha1, beta1 = get_l1_l2_prior(l1_mode, l1_std ** 2)

l2_mode = 0.25
l2_std = 0.2
alpha2, beta2 = get_l1_l2_prior(l2_mode, l2_std ** 2)

x0_mode = 0.25
x0_std = 0.2
x0_start = 1.80
alpha3, beta3 = get_l1_l2_prior(x0_mode, x0_std ** 2)

lw_mode = 10 ** (-1)
lw_std = 10 ** (-1)
alpha4, beta4 = get_l1_l2_prior(lw_mode, lw_std ** 2)

sig_mode_Ti = 0.5 * np.max(function_val)
sig_std = 1.0
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

### Setting our model #######################################################################
Minseok_GPR = GPR(pos_pred, function_pos, function_val, GP_Err_Ti, Pos_Grad, Grad, Grad_Err,
            alpha0_Ti, beta0_Ti, alpha1, beta1, alpha2, beta2, alpha3, beta3,
            x0_start, alpha4, beta4, bounds_Ti, 'H', theta0_Ti, target, burn, 0000, 0.0)

########## Loading MAP and NUTS results #####################################################
MAP_data = loadmat('./Data/MAP_result.mat')
MAP_param = MAP_data['MAP_param_Ti']['x'][0, 0]
MAP_mu = MAP_data['MAP_mu_Ti']
MAP_sig= MAP_data['MAP_sig_Ti']
MAP_gmu = MAP_data['MAP_gmu_Ti']
MAP_gsig= MAP_data['MAP_gsig_Ti']

NUTS_data = loadmat('./Data/NUTS_total.mat')
rho = NUTS_data['NUTS_rho']
eff_rho_idx = NUTS_data['NUTS_eff_rho_idx']
samples = NUTS_data['NUTS_Samples']
IAT = NUTS_data['NUTS_IAT']

Sig_f = samples[0, :]
l1 = samples[1, :]
l2 = samples[2, :]
x0 = samples[3, :]
lw = samples[4, :]

B_Sig_f = Sig_f[burn:]
B_l1 = l1[burn:]
B_l2 = l2[burn:]
B_x0 = x0[burn:]
B_lw = lw[burn:]

Sig_rho = rho[0, :]
l1_rho = rho[1, :]
l2_rho = rho[2, :]
x0_rho = rho[3, :]
lw_rho = rho[4, :]

Plot_FFT(B_Sig_f, Sig_rho[:eff_rho_idx[0, 0]], B_l1, l1_rho[:eff_rho_idx[0, 1]],
         B_l2, l2_rho[:eff_rho_idx[0, 2]], B_x0, x0_rho[:eff_rho_idx[0, 3]],
         B_lw, lw_rho[:eff_rho_idx[0, 4]],
         IAT[0, 0], IAT[0, 1], IAT[0, 2], IAT[0, 3], IAT[0, 4],'./NUTS_figure/FFT.png')

thin_kms = 10

Thinned_idx = np.arange(0, target, thin_kms) + burn
T_Sig_f = Sig_f[Thinned_idx]
T_l1 = l1[Thinned_idx]
T_l2 = l2[Thinned_idx]
T_x0 = x0[Thinned_idx]
T_lw = lw[Thinned_idx]

posterior_plot(B_Sig_f, Sig_rho[:eff_rho_idx[0, 0]], B_l1, l1_rho[:eff_rho_idx[0, 1]],
                 B_l2, l2_rho[:eff_rho_idx[0, 2]], B_x0, x0_rho[:eff_rho_idx[0, 3]],
                 B_lw, lw_rho[:eff_rho_idx[0, 4]],
                 IAT[0, 0], IAT[0, 1], IAT[0, 2], IAT[0, 3], IAT[0, 4],
                MAP_param, T_Sig_f, T_l1, T_l2, T_x0, T_lw,
                alpha0_Ti, beta0_Ti, alpha1, beta1, alpha2, beta2, alpha3, beta3, x0_start, alpha4, beta4)

####### NUTS prediction #############################################################
mu = np.zeros([np.shape(pos_pred)[0], int(target/thin_kms)])
sig = np.zeros([np.shape(pos_pred)[0], np.shape(pos_pred)[0]])
G_mu = np.zeros([np.shape(pos_pred)[0], int(target/thin_kms)])
G_sig = np.zeros([np.shape(pos_pred)[0], np.shape(pos_pred)[0]])

for j in range(int(target/thin_kms)):
    temp = Minseok_GPR.prediction(T_Sig_f[j], T_l1[j], T_l2[j], T_x0[j], T_lw[j])
    mu[:, j] = temp[0][:, 0]
    G_mu[:, j] = temp[2][:, 0]
    sig += temp[1]
    G_sig += temp[3]

sig /= int(target/thin_kms)
G_sig /= int(target/thin_kms)
sig += np.cov(mu)
G_sig += np.cov(G_mu)
mu = np.atleast_2d(np.mean(mu, axis=1)).T
G_mu = np.atleast_2d(np.mean(G_mu, axis=1)).T

NUTS_pred = {}
NUTS_pred['NUTS_mu'] = mu
NUTS_pred['NUTS_sig'] = sig
NUTS_pred['NUTS_gmu'] = G_mu
NUTS_pred['NUTS_gsig'] = G_sig
savemat('./Data/NUTS_profile_prediction.mat',NUTS_pred)

############### Figure: Ti profile from MAP estimator #####################################
plt.figure(figsize=(9, 15))
plt.subplot(2, 1, 1)
GP_Err_diag = np.atleast_2d(np.sqrt(np.diag(MAP_sig))).T

for i in range(30):
    plt.plot(pos_pred[:, 0], np.random.multivariate_normal(MAP_mu[:, 0], MAP_sig),
             color='lime', markersize=0.5, zorder=i+1)

plt.plot(pos_pred[:, 0], MAP_mu, c='g', linestyle='-',markersize=30, zorder=101)
plt.plot(pos_pred[:, 0], MAP_mu-GP_Err_diag, c='g', linestyle='--', markersize=30, zorder=102)
plt.plot(pos_pred[:, 0], MAP_mu+GP_Err_diag, c='g', linestyle='--', markersize=30, zorder=103)

plt.errorbar(function_pos[:, 0], function_val[:, 0], function_err[:, 0], fmt='g.', markersize=10,zorder=108)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('T$_i$ [keV]', fontsize=35)
plt.grid()
plt.xlim(1.78, 2.32)
plt.ylim(-0.02, np.max(function_val) * 1.3)
plt.text(1.83, 3.3, '(a)', fontsize = 35)
plt.tight_layout()
plt.ioff()
plt.legend(loc='upper right', fontsize=25)

plt.subplot(2, 1, 2)
Grad_Err_diag = np.atleast_2d(np.sqrt(np.diag(MAP_gsig))).T
for i in range(30):
    plt.plot(pos_pred, np.random.multivariate_normal(MAP_gmu[:, 0], MAP_gsig),
             color='lime', markersize=5, zorder=i+1)

plt.plot(pos_pred, MAP_gmu, c='g', linestyle='-', markersize=30, label = 'With outlier', zorder = 100)
plt.plot(pos_pred, MAP_gmu - Grad_Err_diag, c='g', linestyle='--', markersize=30, zorder = 101)
plt.plot(pos_pred, MAP_gmu + Grad_Err_diag, c='g', linestyle='--', markersize=30, zorder = 102)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('R [m]', fontsize=35)
plt.ylabel('dT$_i$/dR [keV/m]', fontsize=35)
plt.grid()
plt.xlim(1.78, 2.32)
plt.text(1.83, 0.5, '(b)', fontsize = 35)
plt.tight_layout()
plt.ioff()

plt.savefig('./Figure/fig_MAP.png')
plt.close()

######## Figure: NUTS MAP compare ###########################################################
NUTS_prediction = loadmat('./Data/NUTS_profile_prediction.mat')

NUTS_mu = NUTS_prediction['NUTS_mu']
NUTS_sig = NUTS_prediction['NUTS_sig']
NUTS_gmu = NUTS_prediction['NUTS_gmu']
NUTS_gsig = NUTS_prediction['NUTS_gsig']

plt.figure(figsize=(9, 13))

plt.subplot(2, 1, 1)
GP_Err_diag_SVMR = np.atleast_2d(np.sqrt(np.diag(MAP_sig))).T
GP_Err_diag_NUTS = np.atleast_2d(np.sqrt(np.diag(NUTS_sig))).T

plt.plot(pos_pred[:, 0], MAP_mu, c='b', linestyle='-', markersize=30, zorder=104, label='MAP')
plt.plot(pos_pred[:, 0], MAP_mu-GP_Err_diag_SVMR, c='b', linestyle='--', markersize=30, zorder=105)
plt.plot(pos_pred[:, 0], MAP_mu+GP_Err_diag_SVMR, c='b', linestyle='--', markersize=30, zorder=106)

plt.plot(pos_pred[:, 0], NUTS_mu, c='r', linestyle='-', markersize=30, zorder=107, label='NUTS')
plt.plot(pos_pred[:, 0], NUTS_mu-GP_Err_diag_NUTS, c='r', linestyle='--', markersize=30, zorder=108)
plt.plot(pos_pred[:, 0], NUTS_mu+GP_Err_diag_NUTS, c='r', linestyle='--', markersize=30, zorder=109)

plt.errorbar(function_pos[:, 0], function_val[:, 0], function_err[:, 0], fmt='k.', markersize=20, zorder=111)

plt.xticks(fontsize=25)
plt.yticks([0, 1, 2, 3, 4], fontsize=25)
plt.ylabel('T$_i$ [keV]', fontsize=35)
plt.grid()
plt.xlim(1.78, 2.32)
plt.ylim(-0.02, np.max(function_val) * 1.3)
plt.text(1.83, 3.3, '(a)', fontsize = 35)
plt.tight_layout()
plt.ioff()
plt.legend(loc='upper right', fontsize=25)

plt.subplot(2, 1, 2)
Grad_Err_diag_SVMR = np.atleast_2d(np.sqrt(np.diag(MAP_gsig))).T
Grad_Err_diag_NUTS = np.atleast_2d(np.sqrt(np.diag(NUTS_gsig))).T

plt.plot(pos_pred[:, 0], MAP_gmu, c='b', linestyle='-', markersize=30, zorder = 103)
plt.plot(pos_pred[:, 0], MAP_gmu - Grad_Err_diag_SVMR, c='b', linestyle='--', markersize=30, zorder = 104)
plt.plot(pos_pred[:, 0], MAP_gmu + Grad_Err_diag_SVMR, c='b', linestyle='--', markersize=30, zorder = 105)

plt.plot(pos_pred[:, 0], NUTS_gmu, c='r', linestyle='-', markersize=30, zorder = 106)
plt.plot(pos_pred[:, 0], NUTS_gmu - Grad_Err_diag_NUTS, c='r', linestyle='--', markersize=30, zorder = 107)
plt.plot(pos_pred[:, 0], NUTS_gmu + Grad_Err_diag_NUTS, c='r', linestyle='--', markersize=30, zorder = 108)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('R [m]', fontsize=35)
plt.ylabel('dT$_i$/dR [keV/m]', fontsize=35)
plt.grid()
plt.xlim(1.78, 2.32)
plt.text(1.83, 0.5, '(b)', fontsize=35)
plt.tight_layout()
plt.ioff()

plt.savefig('./NUTS_figure/fig_MAP_NUTS_compare.png')
plt.close()