import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
import scipy.special as sps
from matplotlib import colors
from scipy.io import loadmat
from scipy.io import savemat

def get_l1_l2_prior(mode, var):
    beta = (mode + np.sqrt(mode ** 2 + 4 * var)) / (2 * var)
    alpha = beta * mode + 1

    return (alpha, beta)

def Sampling_Plot(U1, Sig_f, l1, l2, x0, lw, PATH):
    plt.figure(figsize=(16, 18))
    plt.subplot(3, 2, 1)
    plt.plot(U1, c='r')
    plt.ylabel(r'U(${\theta}$)', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 2, 2)
    plt.plot(Sig_f, c='g')
    plt.ylabel(r'${\sigma_f}$', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 2, 3)
    plt.plot(l1, c='b')
    plt.ylabel(r'${l_1}$', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 2, 4)
    plt.plot(l2, c='c')
    plt.ylabel(r'${l_2}$', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 2, 5)
    plt.plot(x0, c='m')
    plt.xlabel('Iteration', fontsize=25)
    plt.ylabel(r'${x_0}$', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 2, 6)
    plt.plot(lw, c='k')
    plt.xlabel('Iteration', fontsize=25)
    plt.ylabel(r'${l_w}$', fontsize=30)
    plt.xticks(np.arange(0, np.shape(Sig_f)[0]+1, np.shape(Sig_f)[0]/5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.savefig(PATH)
    plt.close()

def plot_Acceptence(Accept, accprate, PATH):
    domain = np.arange(len(Accept)) + 1
    plt.figure(figsize=(21, 18))
    plt.scatter(domain, Accept, color='c')
    plt.ylim([0, 1.1])
    plt.xlabel('Iter', fontsize=45)
    plt.ylabel(r'Acceptence', fontsize=45)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    txt = 'Accepetence rate : ' + format(accprate,'.3f')
    plt.text(len(Accept) * 0.65, 1.02, txt, fontsize=35)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH)
    plt.close()

def get_ESS(param, thres):
    mean = np.mean(param)
    var = np.var(param)

    # Normalized data
    ndata = param - mean

    rho = np.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
    rho = rho / var / len(ndata)

    Eff_rho_idx = np.where(rho <= thres)[0][0]
    IAT = 1 + 2 * np.sum(rho[1:Eff_rho_idx])
    ESS = len(ndata) / IAT
    return (rho, Eff_rho_idx, IAT, ESS)

def Plot_FFT(Sig_f, Sig_rho, l1, l1_rho, l2, l2_rho, x0, x0_rho, lw, lw_rho,
             Sig_IAT, l1_IAT, l2_IAT, x0_IAT, lw_IAT,Path):
    xtick_size = 20
    ytick_size = 20
    xlabel_size = 30
    ylabel_size = 30

    plt.figure(figsize=(25, 9))
    plt.subplot(2, 5, 1)
    plt.plot(Sig_f, c='g')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$\sigma_f$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 6)
    plt.plot(Sig_rho, c='g')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{\sigma_f}$', fontsize=ylabel_size)
    plt.text((np.shape(Sig_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(Sig_IAT, 2)), fontsize=25)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 2)
    plt.plot(l1, c='b')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_1$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 7)
    plt.plot(l1_rho, c='b')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_1}$', fontsize=ylabel_size)
    plt.text((np.shape(l1_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(l1_IAT, 2)), fontsize=25)
    plt.ylim(0, 1.1)
    plt.xticks([0, 1, 2, 3], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 3)
    plt.plot(l2, c='c')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_2$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 8)
    plt.plot(l2_rho, c='c')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_2}$', fontsize=ylabel_size)
    plt.text((np.shape(l2_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(l2_IAT, 1)), fontsize=25)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 4)
    plt.plot(x0, c='m')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$x_0$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 9)
    plt.plot(x0_rho, c='m')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{x_0}$', fontsize=ylabel_size)
    plt.text((np.shape(x0_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(x0_IAT, 1)), fontsize=25)
    plt.ylim(0, 1.1)
    plt.xticks([0, 4, 8, 12], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 5)
    plt.plot(lw, c='k')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_w$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(2, 5, 10)
    plt.plot(lw_rho, c='k')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_w}$', fontsize=ylabel_size)
    plt.text((np.shape(lw_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(lw_IAT, 1)), fontsize=25)
    plt.ylim(0, 1.1)
    plt.xticks([0, 5, 10], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.savefig(Path)
    plt.close()

def posterior_plot(Sig_f, Sig_rho, l1, l1_rho, l2, l2_rho, x0, x0_rho, lw, lw_rho,
                   Sig_IAT, l1_IAT, l2_IAT, x0_IAT, lw_IAT,
                   MAP_param, T_Sig_f, T_l1, T_l2, T_x0, T_lw,
                   alpha0, beta0, alpha1, beta1, alpha2, beta2, alpha3, beta3,
                   x0_start, alpha4, beta4):
    xtick_size = 35
    ytick_size = 35
    xlabel_size = 50
    ylabel_size = 50

    plt.figure(figsize=(35, 19))
    plt.subplot(3, 5, 1)
    plt.plot(Sig_f, c='g')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$\sigma_f$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 6)
    plt.plot(Sig_rho, c='g')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{\sigma_f}$', fontsize=ylabel_size)
    plt.text((np.shape(Sig_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(Sig_IAT, 2)), fontsize=35)
    plt.ylim(0, 1.1)
    #plt.xticks([0.0, 1.0, 2.0, 3.0], fontsize=xtick_size)
    plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 2)
    plt.plot(l1, c='b')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_1$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 7)
    plt.plot(l1_rho, c='b')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_1}$', fontsize=ylabel_size)
    plt.text((np.shape(l1_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(l1_IAT, 2)), fontsize=35)
    plt.ylim(0, 1.1)
    plt.xticks([0, 1.0, 2.0, 3, 4], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 3)
    plt.plot(l2, c='c')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_2$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 8)
    plt.plot(l2_rho, c='c')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_2}$', fontsize=ylabel_size)
    plt.text((np.shape(l2_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(l2_IAT, 2)), fontsize=35)
    plt.ylim(0, 1.1)
    plt.xticks([0, 5, 10, 15, 20], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 4)
    plt.plot(x0, c='m')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$x_0$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 9)
    plt.plot(x0_rho, c='m')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{x_0}$', fontsize=ylabel_size)
    plt.text((np.shape(x0_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(x0_IAT, 2)), fontsize=35)
    plt.ylim(0, 1.1)
    plt.xticks([0, 4, 8, 12], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 5)
    plt.plot(lw, c='k')
    plt.xlabel('Iteration', fontsize=xlabel_size)
    plt.ylabel('$l_w$', fontsize=ylabel_size)
    plt.xticks([0, 5000, 10000], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 10)
    plt.plot(lw_rho, c='k')
    plt.xlabel('Lag', fontsize=xlabel_size)
    plt.ylabel(r'$\rho_{l_w}$', fontsize=ylabel_size)
    plt.text((np.shape(lw_rho)[0] - 1) * 0.5, 0.9, 'IAT=' + str(np.round(lw_IAT, 2)), fontsize=35)
    plt.ylim(0, 1.1)
    plt.xticks([0, 4, 8, 12], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ioff()
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(3, 5, 11)
    count, bins, ignored = plt.hist(T_Sig_f, bins=np.arange(min(T_Sig_f), max(T_Sig_f) + np.nanstd(T_Sig_f) * 0.4,
                                                            np.nanstd(T_Sig_f) * 0.4),
                                    density=True, histtype="step", color='g')
    shape, scale = alpha0, 1 / beta0
    y = bins ** (shape - 1) * (np.exp(-bins / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins, y, linewidth=2, color='r',label = 'Prior')
    plt.vlines(MAP_param[0, 0], 0, 100, colors = 'orange', linewidth = 8, label = 'MAP')
    plt.xlabel(r'${\sigma_f}$', fontsize=xlabel_size)
    plt.ylabel(r'${p(\sigma_f|D)}$', fontsize=ylabel_size)
    plt.legend(fontsize=30, loc='upper right')
    plt.xticks([1.5, 3.0, 4.5], fontsize=xtick_size)
    #plt.xticks(fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ylim([-0.01, np.max(count)*1.2])
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 5, 12)
    count2, bins2, ignored2 = plt.hist(T_l1, bins=np.arange(min(T_l1), max(T_l1) + np.nanstd(T_l1) * 0.4, np.nanstd(T_l1) * 0.4),
                                       density=True, histtype="step", color='b')
    shape, scale = alpha1, 1 / beta1
    y = bins2 ** (shape - 1) * (np.exp(-bins2 / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins2, y, linewidth=2, color='r')
    plt.vlines(MAP_param[0, 1], 0, 100, colors = 'orange', linewidth = 8)
    plt.xlabel(r'${l_1}$', fontsize=xlabel_size)
    plt.ylabel(r'${p(l_1|D)}$', fontsize=ylabel_size)
    plt.xticks([0.2, 0.5, 0.8, 1.1], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ylim([-0.01, np.max(count2) * 1.2])
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 5, 13)
    count3, bins3, ignored3 = plt.hist(T_l2, bins=np.arange(min(T_l2), max(T_l2) + np.nanstd(T_l2) * 0.4,
                                                          np.nanstd(T_l2) * 0.4), density=True, histtype="step",
                                       color='c')
    shape, scale = alpha2, 1 / beta2
    y = bins3 ** (shape - 1) * (np.exp(-bins3 / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins3, y, linewidth=2, color='r')
    plt.vlines(MAP_param[0, 2], 0, 100, colors='orange', linewidth=8)
    plt.xlabel(r'${l_2}$', fontsize=xlabel_size)
    plt.ylabel(r'${p(l_2|D)}$', fontsize=ylabel_size)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=xtick_size)
    #plt.xticks([0.0, 0.4, 0.8, 1.2], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ylim([-0.01, np.max(count3) * 1.2])
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 5, 14)
    count4, bins4, ignored4 = plt.hist(T_x0, bins=np.arange(min(T_x0), max(T_x0) + np.nanstd(T_x0) * 0.4,
                                                          np.nanstd(T_x0) * 0.4), density=True, histtype="step",
                                       color='m')
    shape, scale = alpha3, 1 / beta3
    y = (bins4 - x0_start) ** (shape - 1) * (np.exp(-(bins4 - x0_start) / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins4, y, linewidth=2, color='r')
    plt.vlines(MAP_param[0, 3], 0, 100, colors='orange', linewidth=8)
    plt.ylim([0, np.max(count4) * 1.2])
    plt.xlabel(r'${x_0}$', fontsize=xlabel_size)
    plt.ylabel(r'${p(x_0|D)}$', fontsize=ylabel_size)
    plt.xticks([1.9, 2.1, 2.3], fontsize=xtick_size)
    #plt.xticks([1.8, 2.2, 2.6, 3.00], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ylim([-0.01, np.max(count4) * 1.2])
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.subplot(3, 5, 15)
    count5, bins5, ignored5 = plt.hist(T_lw, bins=np.arange(min(T_lw), max(T_lw) + np.nanstd(T_lw) * 0.1,
                                                          np.nanstd(T_lw) * 0.1), density=True, histtype="step",
                                       color='k')
    shape, scale = alpha4, 1 / beta4
    y = bins5 ** (shape - 1) * (np.exp(-bins5 / scale) / (sps.gamma(shape) * scale ** shape))
    plt.plot(bins5, y, linewidth=2, color='r')
    plt.vlines(MAP_param[0, 4], 0, 100, colors='orange', linewidth=8)
    plt.xlabel(r'${l_w}$', fontsize=xlabel_size)
    plt.ylabel(r'${p(l_w|D)}$', fontsize=ylabel_size)
    plt.xticks([0.10, 0.20, 0.30], fontsize=xtick_size)
    #plt.xticks([0.0, 0.2, 0.4, 0.6], fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.ylim([-0.1, np.max(count5) * 1.2])
    plt.grid(True)
    plt.ioff()
    plt.tight_layout()

    plt.savefig('./NUTS_figure/fig_posterior_total.png')

def MAP_SVMR_plot(CES_pos_raw, CES_pos, Pred_pos, Ti, Ti_Err, Ti_raw, Ti_Err_raw,
                  mu, sig, gmu, gsig, SVMR_mu, SVMR_sig, SVMR_gmu, SVMR_gsig):
    plt.figure(figsize=(9, 13))
    plt.subplot(2, 1, 1)
    GP_Err_diag = np.atleast_2d(np.sqrt(np.diag(sig))).T
    GP_Err_diag_SVMR = np.atleast_2d(np.sqrt(np.diag(SVMR_sig))).T

    for i in range(30):
        plt.plot(Pred_pos[:, 0], np.random.multivariate_normal(mu[:,0], sig),
                 color='pink', markersize=0.5, zorder=i+1)

    for i in range(30):
        plt.plot(Pred_pos[:, 0],np.random.multivariate_normal(SVMR_mu[:,0], SVMR_sig),
                 color='lime', markersize=0.5, zorder=i+31)

    plt.plot(Pred_pos[:, 0], mu, c='purple', linestyle='-',markersize=30,zorder=101, label='GPR')
    plt.plot(Pred_pos[:, 0], mu-GP_Err_diag, c='purple', linestyle='--',markersize=30,zorder=102)
    plt.plot(Pred_pos[:, 0], mu+GP_Err_diag, c='purple', linestyle='--',markersize=30,zorder=103)

    plt.plot(Pred_pos[:, 0], SVMR_mu, c='g', linestyle='-',markersize=30,zorder=104, label='GPR+SVMR')
    plt.plot(Pred_pos[:, 0], SVMR_mu-GP_Err_diag_SVMR, c='g', linestyle='--',markersize=30,zorder=105)
    plt.plot(Pred_pos[:, 0], SVMR_mu+GP_Err_diag_SVMR, c='g', linestyle='--',markersize=30,zorder=106)

    plt.errorbar(CES_pos_raw[0, :], Ti_raw[0, :], Ti_Err_raw[0, :], fmt='.', color='silver', markersize=15, zorder=107)
    plt.errorbar(CES_pos[:, 0], Ti[:, 0], np.sqrt(np.diag(Ti_Err)), fmt='k.', markersize=20, zorder=108)
    plt.errorbar(CES_pos_raw[0, 4], Ti_raw[0, 4], Ti_Err_raw[0, 4], fmt='b.', markersize=20, zorder=109, label='Outlier')
    #plt.scatter(Fake_pos, Fake_Temp_Ti, s=30, color='violet', marker='.',zorder=110)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('T$_i$ [keV]', fontsize=35)
    plt.grid()
    plt.xlim(1.78, 2.32)
    plt.ylim(-0.02, np.max(Ti_raw) * 1.3)
    plt.text(1.81, 4.2, '(a)', fontsize = 35)
    plt.tight_layout()
    plt.ioff()
    plt.legend(loc='upper right', fontsize=25)

    plt.subplot(2, 1, 2)
    Grad_Err_diag = np.atleast_2d(np.sqrt(np.diag(gsig))).T
    Grad_Err_diag_SVMR = np.atleast_2d(np.sqrt(np.diag(SVMR_gsig))).T
    for i in range(30):
        plt.plot(Pred_pos, np.random.multivariate_normal(gmu[:, 0], gsig),
                 color = 'pink', markersize=5, zorder=i+1)

    for i in range(30):
        plt.plot(Pred_pos[:, 0], np.random.multivariate_normal(SVMR_gmu[:, 0], SVMR_gsig),
                 color='lime', markersize=0.5, zorder=i+31)

    plt.plot(Pred_pos, gmu, c='purple', linestyle='-', markersize=30, label = 'With outlier', zorder = 100)
    plt.plot(Pred_pos, gmu - Grad_Err_diag, c='purple', linestyle='--', markersize=30, zorder = 101)
    plt.plot(Pred_pos, gmu + Grad_Err_diag, c='purple', linestyle='--', markersize=30, zorder = 102)

    plt.plot(Pred_pos, SVMR_gmu , c='g', linestyle='-', markersize=30, label='Without outlier', zorder = 103)
    plt.plot(Pred_pos, SVMR_gmu - Grad_Err_diag_SVMR, c='g', linestyle='--', markersize=30, zorder = 104)
    plt.plot(Pred_pos, SVMR_gmu + Grad_Err_diag_SVMR, c='g', linestyle='--', markersize=30, zorder = 105)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('R [m]', fontsize=35)
    plt.ylabel('dT$_i$/dR [keV/m]', fontsize=35)
    plt.grid()
    plt.xlim(1.78, 2.32)
    plt.text(1.81, 140, '(b)', fontsize = 35)
    plt.tight_layout()
    plt.ioff()

    plt.savefig('./Figure/fig_SVMR_MAP.eps', format='eps', dpi=3000)
    #plt.savefig('./Figure/fig_SVMR_MAP.png')
    plt.close()
