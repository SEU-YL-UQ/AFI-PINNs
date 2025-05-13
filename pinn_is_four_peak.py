import sys
sys.path.append('adaptive_restart_pinn')

from adaptive.subset_simulation import Subset_simulation  
from adaptive.prior_density import Uniform
from models.PINN_four_peak import Four_peak_torch
import numpy as np
import torch
import os
import pickle
from config.four_peak_save_path import create_path, delete_img_path, rar_save_path, rar_save_path, rar_save_path
from functools import partial
import scipy.stats as stat
import matplotlib.pyplot as plt 
from utils.early_stopping import EarlyStopping
from utils.generate_data import generate_peak4_samples



create_path(rar_save_path)
delete_img_path(rar_save_path)


N_b = 800
N_f = [2000]
lb = np.array([-1, -1])
ub = np.array([1, 1])
for i in range(len(N_f)):
    for j in range(1):
        delete_img_path(rar_save_path)
        X_f_train, X_b_train, u_b = generate_peak4_samples(N_b, N_f[i], lb, ub)
        prior = Uniform(lb, ub)
        SS = Subset_simulation(prior, 1, 0.1)
        early_stopping = EarlyStopping(10, 1e-3)
        pinn = Four_peak_torch(X_f_train, X_b_train, u_b, early_stopping=early_stopping, save_path = rar_save_path, device = torch.device('cuda:0'))
        pinn.train(max_epoches= 50000, ss  = SS, tol_p = 0.01, choice = 'r(x)', method = 'rar')
        with open(os.path.join(rar_save_path.data_save_path, 'proportion_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.proportion, f)
        with open(os.path.join(rar_save_path.data_save_path, 'loss_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.loss, f)
        with open(os.path.join(rar_save_path.data_save_path, 'error_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.error, f)
        with open(os.path.join(rar_save_path.data_save_path, 'failure_probability_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.p_f, f)
        with open(os.path.join(rar_save_path.data_save_path, 'final_samples_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.X_f_train, f)
        with open(os.path.join(rar_save_path.data_save_path, 'predicted_error_' + str(j) + '_' + str(N_f[i])), 'wb') as f:
            pickle.dump(pinn.Error, f)


