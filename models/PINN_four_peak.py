import torch
import torch.nn as nn
import os
import numpy as np
import sys 
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sympy import *
from torch.utils.data import TensorDataset, DataLoader 
from utils.early_stopping import EarlyStopping
from pyDOE import lhs

sys.path.append('pinn_is')
from utils.freeze_weights import freeze_by_idxs
from utils.mod_lbfgs import ModLBFGS


class DNN(nn.Module):
    """This class carrys out DNN"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hiddens):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hiddens = num_hiddens
        self.nn = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh()
        ]
        for _ in range(self.num_hiddens):
            self.nn += self.block()
        self.nn.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.nn = nn.Sequential(*self.nn)
        
        # self.nn.apply(self.init_weights)

    def block(self):
        """This block implements a hidden block"""
        return [nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()]

    def forward(self, x):
        return self.nn(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

class Four_peak_torch:
    """This script carrys out unbounded pinn pdes"""
    def __init__(self, X_f_train, X_b_train, u_b, early_stopping, save_path, device) -> None:
        self.device = device
        self.save_path = save_path 
        
        self.p_f = []
        self.net = DNN(2, 128, 1, 5).to(device)
        # self.net.init_weights()
        

        self.u_b = torch.tensor(u_b, dtype = torch.float32).to(device)
        self.x_b = torch.tensor(X_b_train[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.y_b = torch.tensor(X_b_train[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.X_f_train = X_f_train
        self.loss = []
        self.error = []
        self.Error = []
        self.proportion = []
        self.stalled_epoch = 0
        self.iter = 0
        self.Iter = 0
        

        u_true = lambda x, y: np.exp(-1000*((x-0.5)**2 + (y - 0.5)**2)) +\
            np.exp(-1000*((x+0.5)**2 + (y + 0.5)**2))+ np.exp(-1000*((x+0.5)**2 + (y - 0.5)**2)) +\
            np.exp(-1000*((x-0.5)**2 + (y + 0.5)**2))
        x = np.linspace(-1,1,100)
        y = np.linspace(-1,1,100)
        X, Y = np.meshgrid(x, y)
        self.points = np.array([X.flatten(), Y.flatten()]).T
        self.true_u = u_true(X.flatten(), Y.flatten()).reshape(X.shape)
        self.X, self.Y = X, Y

        self.early_stopping = early_stopping
        
    

    def net_u(self, x, y):
        u = self.net(torch.hstack((x, y)))
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        
        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True)[0]

        f =  -4*u - 2 * x * u_x - 2 * y * u_y + u_yy + u_xx - self.source_function(x, y)

        return f
    
    def source_function(self, x, y):
        temp1 = -1000*((x-0.5)**2 + (y - 0.5)**2)
        temp2 = -1000*((x+0.5)**2 + (y + 0.5)**2)
        temp3 = -1000*((x+0.5)**2 + (y - 0.5)**2)
        temp4 = -1000*((x-0.5)**2 + (y + 0.5)**2)
        s = -2*x*((1000 - 2000*x)*torch.exp(temp1) + (-2000*x - 1000)*torch.exp(temp2) + (1000-2000*x)*torch.exp(temp4) + (-2000*x-1000)*torch.exp(temp3))- \
            2*y*((1000 - 2000*y)*torch.exp(temp1) + (-2000*y - 1000)*torch.exp(temp2) + (1000 - 2000*y)*torch.exp(temp3) + (-2000*y - 1000)*torch.exp(temp4)) +\
            2000 * (2000*(x - 0.5)**2 * torch.exp(temp1) + 2000 * (x+0.5)**2*torch.exp(temp2) - torch.exp(temp1) - torch.exp(temp2) + 2000*(x - 0.5)**2 * torch.exp(temp4) + 2000 * (x+0.5)**2*torch.exp(temp3) - torch.exp(temp3) - torch.exp(temp4)) +\
            2000 * (2000*(y - 0.5)**2 * torch.exp(temp1) + 2000 * (y+0.5)**2*torch.exp(temp2) - torch.exp(temp1) - torch.exp(temp2) + 2000*(y - 0.5)**2 * torch.exp(temp3) + 2000 * (y+0.5)**2*torch.exp(temp4) - torch.exp(temp3) - torch.exp(temp4)) -\
            4 *(torch.exp(temp1) + torch.exp(temp2) + torch.exp(temp3) + torch.exp(temp4))
        return s

    def train_one_epoch(self):
        self.net.train()
        batch_size = self.x_f.shape[0]
        n_batches = self.x_f.shape[0]//batch_size
        def closure():
            for j in range(n_batches):
            # for data,_ in self.dataloader:
                x_f_batch = self.x_f[j*batch_size:(j*batch_size + batch_size),]
                y_f_batch = self.y_f[j*batch_size:(j*batch_size + batch_size),]
                # x_f_batch = data[:,0:1]
                # y_f_batch = data[:,1:2]
                # y_f_batch = self.y_f[j*batch_size:(j*batch_size + batch_size),]
                self.optimizer.zero_grad()
                # u & f predictions:
                u_b_prediction = self.net_u(self.x_b, self.y_b)
                f_prediction = self.net_f(x_f_batch, y_f_batch)

                # losses:
                u_b_loss = ((u_b_prediction - self.u_b)**2).mean()
                f_loss = (f_prediction**2).mean()
                ls = f_loss + u_b_loss
                # derivative with respect to net's weights:
                ls.backward()
            # increase iteration count:
                if not self.Iter%10:
                    # print('current loss', ls.item())
                    self.Error.append(self.calculate_error())
                self.Iter += 1
                self.loss.append(ls.item())
                
            return ls
        self.optimizer.step(closure)
        # self.CosineLR.step()
        
    
    def update(self):
        self.x_f = torch.tensor(self.X_f_train[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(self.device)
        self.x_f = torch.vstack([self.x_f, self.x_b])
        
        self.y_f = torch.tensor(self.X_f_train[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(self.device)
        self.y_f = torch.vstack([self.y_f, self.y_b])
        self.optimizer = ModLBFGS(self.net.parameters(), lr = 0.3, max_iter=1, max_eval=None,
                            history_size=50, tolerance_grad=1e-7, tolerance_change=1e-7,
                            line_search_fn="strong_wolfe")
        self.adam_optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-4)
        # self.CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000, eta_min=0.2)
        # self.X_f_train = torch.tensor(self.X_f_train, requires_grad=True).float().to(self.device)
        # dataset = TensorDataset(self.X_f_train, torch.zeros((self.X_f_train.shape[0], 1)).to(self.device))
        # self.dataloader = DataLoader(dataset, shuffle=True, batch_size=len(self.X_f_train)//2)
    
    def train(self, max_epoches, ss, tol_p, choice, method, save_final = True):
        self.update()
        batch_size = self.X_f_train.shape[0]
        n_batches = self.X_f_train.shape[0]//batch_size
        for i in range(max_epoches):
            self.train_one_epoch()
            # for j in range(n_batches):
            #     x_f_batch = self.x_f[j*batch_size:(j*batch_size + batch_size),]
            #     y_f_batch = self.y_f[j*batch_size:(j*batch_size + batch_size),]
            #     self.adam_optimizer.zero_grad()
            #     # u & f predictions:
            #     u_b_prediction = self.net_u(self.x_b, self.y_b)
            #     f_prediction = self.net_f(x_f_batch, y_f_batch)

            #     # losses:
            #     u_b_loss = ((u_b_prediction - self.u_b)**2).mean()
            #     f_loss = (f_prediction**2).mean()
            #     ls = f_loss + 10*u_b_loss
            #     # derivative with respect to net's weights:
            #     ls.backward()
            #     self.adam_optimizer.step()
            #     self.loss.append(ls.item())
            if not i%10:
                print('[current epoch: %d][current loss: %.6f][Early stopping counter: %d]'%(i,self.loss[-1], self.early_stopping.counter))
                # self.Error.append(self.calculate_error())
                self.early_stopping(self.loss[-1])
            if self.early_stopping.early_stop:
                # print(np.mean(abs(self.predict(self.points)[1])))
                # print(np.mean(abs(self.predict(self.points)[2])))
                # freeze_by_idxs(self.net, np.arange(1))
                self.update()
                print('early stop')
                current_error = self.calculate_error()
                self.error.append(current_error)
                print(self.error)
                self.early_stopping.reset()
                self.plot_error(prefix='add_points' + str(self.iter))
                eta = self.Annealing(i*2, max_epoches*2)
                self.resample(ss, eta, tol_p, choice, method)
                if self.p_f[-1] < tol_p:
                    self.plot_error(prefix='full_training.png')
                    self.save_model(prefix='full_model' + str(len(self.X_f_train)))
                    print('current error:', self.calculate_error())
                    print('training complete, current failure probability is: %.6f'%(self.p_f[-1]))
                    break
                self.plot_error(self.samples, prefix='add_points' + str(self.iter))
                self.plot(self.error, 'error' + str(self.iter))
                self.plot(self.p_f, 'failure' + str(self.iter))
                self.update()
                self.save_model('model' + str(self.iter) + '_' + str(len(self.X_f_train)))
                self.iter += 1
                print('current_error', self.calculate_error())
        print('current error', self.calculate_error())

        if save_final:
            self.plot_error(prefix='full_training.png')
            self.save_model(prefix='full_model' + str(len(self.X_f_train)))
    
    def power_function(self, choice, tol):
        if choice == 'r(x)':
            return lambda x: abs(self.predict(x)[1].squeeze()) - tol
        else:
            return lambda x: abs(self.predict(x)[2].squeeze()) - tol
    
    def resample(self, ss, eta, tol_p, choice, method):
        print('resampling proportion: ', eta)
        if method == "residual":
            power_function = self.power_function(choice, ss.final_level)
            num_init_samples = (self.X_f_train.shape[0]*eta).astype(np.int32)
            self.samples, p_f = ss.sample(num_init_samples, power_function, tol_p)
            self.p_f.append(p_f)
            print('current failure probability: %.4f'%(p_f))
            index = np.random.choice(np.arange(self.X_f_train.shape[0]), self.X_f_train.shape[0] - num_init_samples, replace=False)
            select_x_f = self.X_f_train[index]
            new_x_f = ss.prior.sample(num_init_samples - self.samples.shape[0])
            self.X_f_train = np.vstack([self.samples, select_x_f, new_x_f])
        else:
            new_samples = ss.prior.sample(self.X_f_train.shape[0])
            power_function = self.power_function(choice, ss.final_level)
            num_init_samples = (self.X_f_train.shape[0]*eta).astype(np.int32)
            self.samples = new_samples[np.argsort(power_function(self.X_f_train))[-num_init_samples:]]
            index = np.random.choice(np.arange(self.X_f_train.shape[0]), self.X_f_train.shape[0] - num_init_samples)
    
            select_x_f = self.X_f_train[index]
            self.X_f_train = np.vstack([self.samples, select_x_f])
            p_f = np.sum(power_function(self.X_f_train)>0)/len(self.X_f_train)
            self.p_f.append(p_f)
    
    def plot(self,data, prefix):
        fig = plt.figure()
        plt.plot(data)
        plt.yscale('log')
        plt.savefig(os.path.join(self.save_path.img_save_path, prefix))
        

    def Annealing(self, epoch, max_epoch):
        cur_epoch = epoch - self.stalled_epoch 
        self.stalled_epoch = epoch
        eta = 0.5 * (1 + np.cos(cur_epoch/(max_epoch - self.stalled_epoch) * np.pi))
        self.proportion.append(eta)
        return eta

    
    def predict(self, points):
        x = torch.tensor(points[:, 0:1], requires_grad = True).float().to(self.device)
        y = torch.tensor(points[:, 1:2], requires_grad = True).float().to(self.device)

        self.net.eval()
        u = self.net_u(x, y)
        f = self.net_f(x, y)
        f_x  = torch.autograd.grad(
            f, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        u = u.to('cpu').detach().numpy()
        f = f.to('cpu').detach().numpy()
        f_x = f_x.to('cpu').detach().numpy()
        return u, f, f_x

    def plot_error(self, add_points = None, prefix = None):
        """ plot the solution on new data """
        u_predict, f_predict, u_x = self.predict(self.points)
    
        u_predict = u_predict.reshape(self.true_u.shape)
        f_predict = f_predict.reshape(self.true_u.shape)
        u_x = u_x.reshape(self.true_u.shape)

        fig = plt.figure(figsize=(15,10))
        fig.suptitle('Initial points:' + str(len(self.y_f)))
        ax1 = fig.add_subplot(221)
        im1 = ax1.contourf(self.X, self.Y, abs(f_predict), cmap = "winter")
        if add_points is not None:
            fig.suptitle('Initial points: ' + str(len(self.x_f)) +  ' ' + 'add points: ' + str(len(add_points)))
            ax1.scatter(self.X_f_train[:,0], self.X_f_train[:,1], marker = '+', edgecolors = 'black', facecolors = 'white', s = 3)
            ax1.scatter(add_points[:,0], add_points[:,1], marker = 'o', edgecolors = 'red', facecolors = 'white')
        ax1.set_title("Equation error")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')
    
        ax2 = fig.add_subplot(222)
        im2 = ax2.contourf(self.X, self.Y, abs(u_predict - self.true_u), cmap = "winter")
        ax2.set_title("Solution error")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='2%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax3 = fig.add_subplot(223)
        ax3.plot(self.loss, label = "L_2 error")
        ax3.set_yscale('log')
        ax3.legend()
        ax3 = fig.add_subplot(224)
        ax3.plot(self.Error, label = "L_2 error")
        ax3.set_yscale('log')
        ax3.legend()
        plt.savefig(os.path.join(self.save_path.img_save_path, prefix + '.png'))
        # plt.show()
    
    def calculate_error(self):
        u_predict, _,_ = self.predict(self.points)
        error = np.linalg.norm(u_predict.squeeze() - self.true_u.flatten())/np.linalg.norm(self.true_u.flatten())
        return error
    
    
    def save_model(self, prefix):
        torch.save(self.net, os.path.join(self.save_path.model_save_path, prefix))
    
    def save_data(self, target, prefix):
        np.savetxt(os.path.join(self.save_path.data_save_path, prefix), target)

