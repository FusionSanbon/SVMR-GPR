import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import copy
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    os.mkdir('./Figure/')
except:
    print("The folders already exist.")

def Outlier_idx_SVR(Pos, Temp, Const, eps):
    Pos2 = np.atleast_2d(Pos).T
    SVMR = make_pipeline(StandardScaler(), SVR(C=Const, epsilon=eps))
    SVMR.fit(Pos2, Temp)
    SVMR_pred = SVMR.predict(Pos2)
    diff = np.abs(SVMR_pred - Temp)

    SVM_Err = np.zeros(np.shape(SVMR_pred)[0])
    for i in range(np.shape(SVMR_pred)[0]):
        SVM_Err[i] = eps

    result = (np.where(diff > (eps + eps*0.2))[0], SVMR_pred)
    return result

syn_data = loadmat('./Data/synthetic_data.mat')
function_pos = copy.deepcopy(syn_data['position'])
function_val = copy.deepcopy(syn_data['value'])
function_err = copy.deepcopy(syn_data['error'])
pos_pred = copy.deepcopy(syn_data['position_prediction'])

function_val[3, 0] += 3.0

#################### Plotting for the Training dataset ###################################################
Param = [0, 0]
Param[0] = 13.1
Param[1] = 0.443

dot_size = 200
xtick_size = 45
ytick_size = 45

eps = (np.median(function_val[:24, 0])-function_val[23, 0]) * Param[1]
(rough_tune, SVMR_PRED1) = Outlier_idx_SVR(np.atleast_2d(function_pos[:24, 0]), function_val[:24, 0], Param[0], eps)

plt.figure(figsize=(8, 6))
plt.plot(function_pos[:24, 0], SVMR_PRED1, c='b', zorder=1, label='SVMR')
plt.plot(function_pos[:24, 0], SVMR_PRED1 + eps, linestyle='--', c='b', zorder=2)
plt.plot(function_pos[:24, 0], SVMR_PRED1 - eps, linestyle='--', c='b', zorder=3)
plt.scatter(function_pos[:24, 0], function_val[:24, 0], c='g', s=dot_size, zorder=7, label='T$_i$')
plt.scatter(function_pos[3, 0], function_val[3, 0], s=dot_size, c='r', zorder=8)
plt.xlabel('R [m]', fontsize=45)
plt.ylabel('T$_i$ [keV]', fontsize=45)
plt.legend(fontsize=25)
plt.xlim([1.8, 2.18])
plt.ylim([-0.05, np.max(function_val[:24, 0]) * 1.3])
plt.xticks([1.8, 1.9, 2.0, 2.1, 2.2], fontsize=xtick_size)
plt.yticks(fontsize=ytick_size)
plt.grid()
plt.tight_layout()

plt.savefig('./Figure/SVMR_result.png')
plt.close()