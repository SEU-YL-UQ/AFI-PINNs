import numpy as np 
from pyDOE import lhs


def generate_peak2_samples(N_b, N_f, lb, ub):
    np.random.seed(1)
    u_true = lambda x, y: np.exp(-1000*((x-0.5)**2 + (y - 0.5)**2)) + np.exp(-1000*((x+0.5)**2 + (y + 0.5)**2))
    X_f = lb + (ub-lb)*lhs(2, N_f)
    x_lb = np.random.uniform(-1,1,(N_b//4, 1))
    u_lb = u_true(-1, x_lb)
    x_rb = np.random.uniform(-1,1,(N_b//4, 1))
    u_rb = u_true(1, x_rb)
    x_ub = np.random.uniform(-1,1,(N_b//4, 1))
    u_ub = u_true(x_ub, 1)
    x_bb = np.random.uniform(-1,1, (N_b//4, 1))
    u_bb = u_true(x_bb, -1)
    X_lb = np.hstack([-np.ones((N_b//4, 1)), x_lb])
    X_ub = np.hstack([x_ub, np.ones((N_b//4, 1))])
    X_rb = np.hstack([np.ones((N_b//4, 1)), x_rb])
    X_bb = np.hstack([x_bb, -np.ones((N_b//4, 1))])
    X_b_train = np.vstack([X_lb, X_ub, X_rb, X_bb])
    u_b = np.vstack([u_lb, u_ub, u_rb, u_bb])
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_b_train = X_b_train[index]
    # X_f = np.vstack([X_f, X_b_train])
    u_b = u_b[index]
    return X_f, X_b_train, u_b

def generate_peak4_samples(N_b, N_f, lb, ub):
    np.random.seed(1)
    u_true = lambda x, y: np.exp(-1000*((x-0.5)**2 + (y - 0.5)**2)) + np.exp(-1000*((x+0.5)**2 + (y + 0.5)**2)) + np.exp(-1000*((x+0.5)**2 + (y - 0.5)**2)) + np.exp(-1000*((x-0.5)**2 + (y + 0.5)**2))
    X_f = lb + (ub-lb)*lhs(2, N_f)
    x_lb = np.random.uniform(-1,1,(N_b//4, 1))
    u_lb = u_true(-1, x_lb)
    x_rb = np.random.uniform(-1,1,(N_b//4, 1))
    u_rb = u_true(1, x_rb)
    x_ub = np.random.uniform(-1,1,(N_b//4, 1))
    u_ub = u_true(x_ub, 1)
    x_bb = np.random.uniform(-1,1, (N_b//4, 1))
    u_bb = u_true(x_bb, -1)
    X_lb = np.hstack([-np.ones((N_b//4, 1)), x_lb])
    X_ub = np.hstack([x_ub, np.ones((N_b//4, 1))])
    X_rb = np.hstack([np.ones((N_b//4, 1)), x_rb])
    X_bb = np.hstack([x_bb, -np.ones((N_b//4, 1))])
    X_b_train = np.vstack([X_lb, X_ub, X_rb, X_bb])
    u_b = np.vstack([u_lb, u_ub, u_rb, u_bb])
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_b_train = X_b_train[index]
    # X_f = np.vstack([X_f, X_b_train])
    u_b = u_b[index]
    return X_f, X_b_train, u_b

def generate_ac_samples(N_b, N_0, N_f, lb, ub):
    np.random.seed(2)
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    t = np.random.rand(N_b//2,1)
    x_lb = np.hstack([-np.ones_like(t), t])
    x_rb = np.hstack([np.ones_like(t), t])
    u_lb = -np.ones_like(t)
    u_rb = -np.ones_like(t)
    X_0_train = np.random.uniform(lb[0], ub[0], (N_0,1))
    u_0 = X_0_train**2*np.cos(np.pi*X_0_train)
    X_0_train = np.hstack([X_0_train, np.zeros_like(X_0_train)])
    return X_f_train, x_lb, x_rb, X_0_train, u_lb, u_rb, u_0




def g(t):
    x = np.cos(t) - np.cos(5*t) * np.cos(t)/4
    y = np.sin(t) - np.cos(5*t) * np.sin(t)/4
    return np.array([x, y])

def f(x):
    if x[0] == 0:
            temp1 = g(np.pi/2)
            temp2 = g(np.pi * 1.5)
            if (x[1] - temp1[1]) * (x[1] - temp2[1]) < 0:
                return True 
            else:
                return False  
    t = np.arctan(x[1]/x[0])
    temp1 = g(t)
    temp2 = g(t + np.pi)
    if (x[1] - temp1[1]) * (x[1] - temp2[1]) < 0:
        return True 
    else:
        return False

def generate_ubounded_2d_samples(N_0, N_f, lb, ub):
    np.random.seed(1)
    u_true = lambda x, y: np.exp(-((x - 4)**2 + (y - 4)**2)) + np.exp(-((x + 4)**2 + (y + 4)**2))
    X = lb + (ub - lb) * lhs(2, 3*N_f)
    X_f_train = np.zeros_like(X)
    X_f_train[:,0] = X[:,0] * np.cos(X[:,1])
    X_f_train[:,1] = X[:,0] * np.sin(X[:,1])
    index = np.apply_along_axis(f, 1, X_f_train)
    X_f_train = X_f_train[~index]
    choice = np.random.choice(X_f_train.shape[0], N_f, replace=False)
    X_f_train = X_f_train[choice]
    X_0 = np.random.uniform(0, 2*np.pi, N_0)
    X_0_train = np.zeros((N_0, 2))
    X_0_train[:,0] = np.cos(X_0) - np.cos(5*X_0)*np.cos(X_0)/4
    X_0_train[:,1] = np.sin(X_0) - np.cos(5*X_0)*np.sin(X_0)/4
    index_f = np.arange(0, N_f)
    index_b = np.arange(0, N_0)
    np.random.shuffle(index_f)
    np.random.shuffle(index_b)
    u_train = u_true(X_0_train[:,0], X_0_train[:,1])[:,None]
    X_f_train = X_f_train[index_f,:]
    X_0_train = X_0_train[index_b,:]
    u_train = u_train[index_b,:]
    return X_0_train, u_train, X_f_train

def generate_wave_samples(N_f, N_b, N_0, lb, ub):
    np.random.seed(10)
    u_true = lambda x,t: 0.5/np.cosh(2*(x - np.sqrt(3)*t)) - 0.5/np.cosh(2*(x - 10 + np.sqrt(3)*t)) + 0.5/np.cosh(2*(x + np.sqrt(3)*t)) - 0.5/np.cosh(2*(x + 10 - np.sqrt(3)*t))
    X_0_train = np.random.uniform(lb[0], ub[0], (N_0,1))
    u_0 = u_true(X_0_train, 0)
    X_0_train = np.hstack([X_0_train, np.zeros_like(X_0_train)])
    t = np.random.uniform(lb[0], ub[1], (N_b,1))
    X_lb_train = np.hstack([-5*np.ones_like(t), t])
    X_rb_train = np.hstack([5*np.ones_like(t),t])
    u_lb = u_true(-5, t)
    u_rb = u_true(5, t)
    X_f_train = lb + (ub - lb)*lhs(2, N_f)
    return X_f_train, X_lb_train, X_rb_train, u_lb, u_rb, X_0_train, u_0

def generate_wave_samples(N_f, N_b, N_0, lb, ub):
    np.random.seed(1)
    u_true = lambda x,t: 0.5/np.cosh(2*(x - np.sqrt(3)*t)) - 0.5/np.cosh(2*(x - 10 + np.sqrt(3)*t)) + 0.5/np.cosh(2*(x + np.sqrt(3)*t)) - 0.5/np.cosh(2*(x + 10 - np.sqrt(3)*t))
    X_0_train = np.random.uniform(lb[0], ub[0], (N_0,1))
    u_0 = u_true(X_0_train, 0)
    X_0_train = np.hstack([X_0_train, np.zeros_like(X_0_train)])
    t = np.random.uniform(lb[0], ub[1], (N_b,1))
    X_lb_train = np.hstack([-5*np.ones_like(t), t])
    X_rb_train = np.hstack([5*np.ones_like(t),t])
    u_lb = u_true(-5, t)
    u_rb = u_true(5, t)
    X_f_train = lb + (ub - lb)*lhs(2, N_f)
    return X_f_train, X_lb_train, X_rb_train, u_lb, u_rb, X_0_train, u_0

def generate_nls_samples(N_f, N_b,N_0,lb,ub):
    # np.random.seed(1)
    u_true = lambda t,x: (1-4*(1+(2j)*t)/(4*(x**2+t**2)+1))*np.exp((1j)*t)
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    x_0 = np.random.uniform(lb[1], ub[1], N_0)
    u_0 = u_true(lb[0], x_0)
    X_0_train = np.array([np.ones_like(x_0)*lb[0], x_0]).T 
    t = np.random.uniform(lb[0], ub[0], N_b//2)
    u_lb = u_true(t,lb[1])
    u_rb = u_true(t,ub[1])
    X_lb_train = np.array([t,np.ones_like(t)*lb[1]]).T
    X_rb_train = np.array([t,np.ones_like(t)*ub[1]]).T
    return X_f_train, X_lb_train, X_rb_train, X_0_train, u_lb, u_rb, u_0
    


def generate_high_dimensioanl_samples(N_f, N_b, lb, ub):
    np.random.seed(2)
    X_f_train = lb + (ub-lb)* lhs(len(lb), N_f)
    exact_u = lambda x: np.exp(-10*np.linalg.norm(x, axis = 1).reshape(-1,1)**2)
    u = np.zeros((0,1))
    X_u_train = np.zeros((0,len(lb)))
    for i in range(len(ub)):
        x_hat = -np.ones(len(ub) - 1) + 2 * np.ones(len(ub) - 1) * lhs(len(ub) - 1, N_b//(len(ub)*2))
        x_hat_upper = np.insert(x_hat, i, values = 1, axis = 1)
        x_hat_lower = np.insert(x_hat, i, values = -1, axis = 1)
        u_upper = exact_u(x_hat_upper)
        u_lower = exact_u(x_hat_lower)
        X_u_train = np.append(X_u_train, x_hat_upper, axis = 0)
        X_u_train = np.append(X_u_train, x_hat_lower, axis = 0)
        u = np.append(u, u_upper, axis = 0)
        u = np.append(u, u_lower, axis = 0)
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_u_train = X_u_train[index,:]
    u = u[index, :]
    return X_f_train, X_u_train, u
    

# X_f_train, X_lb_train, X_rb_train, u_lb, u_rb, X_0_train, u_0 = generate_nls_samples(1000,100,100,np.array([-2,-2]),np.array([2,2]))
# print(X_f_train)
# import matplotlib.pyplot as plt
# plt.scatter(X_f_train[:,0], X_f_train[:,1])
# plt.scatter(X_0_train[:,0], X_0_train[:,1])
# plt.scatter(X_rb_train[:,0], X_rb_train[:,1])
# plt.scatter(X_lb_train[:,0], X_lb_train[:,1])
# plt.xlabel('t')
# plt.ylabel('x')
# plt.savefig('/home/gaozhiwei/python/adaptive_restart_pinn/1.png')