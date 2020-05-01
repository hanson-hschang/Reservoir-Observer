"""
Created on Wed Apr. 03, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Signal import Signal, Rossler, Lorenz

class Reservoir(object):
    def __init__(self, dt, n, m, d=400, alpha=1., beta=1e-10, D=20, rho=1., w=1.):
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        
        self.d = d
        self.n = n
        self.m = m
        
        self.W_in = np.zeros([self.d, self.n])
        self.b_in = np.ones(self.d)
        self.A = np.zeros([self.d, self.d])
        self.W_out = np.zeros([self.m, self.d])
        self.b_out = np.zeros(self.m)

        self.D = D
        self.rho = rho
        self.w = w

        self.U_mean = np.zeros(self.n)
        self.U_variance = np.zeros(self.n)
        self.r = np.zeros(self.d)
        self.Y_mean = np.zeros(self.m)
        self.Y_variance = np.zeros(self.m)

        self.initialization()

    def initialization(self):
        probability = self.D/self.d
        for di in range(self.d):
            for dj in range(di, self.d):
                if np.random.rand()<probability:
                    self.A[di,dj] = np.random.uniform(-1,1)
                    self.A[dj,di] = self.A[di,dj]
        eigenvalues, _ = np.linalg.eig(self.A)
        self.A = self.A/np.abs(np.max(eigenvalues))*self.rho
        for n, d in enumerate(np.random.choice(self.n, self.d)):
            self.W_in[n,d] = np.random.uniform(-self.w, self.w)
        return
    
    def reservoir_dynamics(self, r, U):
        return r+self.alpha*(-r + np.tanh(np.matmul(self.A,r)+np.matmul(self.W_in,U)+self.b_in))

    def train(self, U, Y, show_time=False):
        
        self.U_mean = np.reshape(np.mean(U, axis=1), [-1,1])
        self.U_variance = np.reshape(np.var(U, axis=1), [-1,1])
        self.Y_mean = np.reshape(np.mean(Y, axis=1), [-1,1])
        self.Y_variance = np.reshape(np.var(Y, axis=1), [-1,1])
        normalized_U = (U-self.U_mean)/self.U_variance
        normalized_Y = (Y-self.Y_mean)/self.Y_variance

        r = np.zeros([self.d, Y.shape[1]])
        for k in range(Y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            if not k==(Y.shape[1]-1):
                r[:,k+1] = self.reservoir_dynamics(r[:,k], normalized_U[:,k])

        r_mean = np.reshape(np.mean(r, axis=1), [-1,1])
        diff_r = r-r_mean
        Y_mean = np.reshape(np.mean(normalized_Y, axis=1), [-1,1])
        diff_Y = normalized_Y - Y_mean
        numerator =  np.einsum('mk,kn->mn',diff_Y, np.transpose(diff_r))
        denominator = np.einsum('ik,kj->ij',diff_r, np.transpose(diff_r))+self.beta*np.identity(self.d)
        self.W_out = np.matmul(numerator,np.linalg.inv(denominator))
        self.b_out = -np.reshape(np.matmul(self.W_out,r_mean)-Y_mean, [-1])
        
        self.r = r[:,-1]
        self.U_mean = np.reshape(self.U_mean, [-1])
        self.U_variance = np.reshape(self.U_variance, [-1])
        return 
    
    def predict(self, U):
        Y = np.reshape(np.matmul(self.W_out, self.r)+self.b_out, [-1,1])
        normalized_U = np.reshape(np.squeeze((U-self.U_mean)/self.U_variance), [-1])
        self.r = self.reservoir_dynamics(self.r, normalized_U)
        return np.reshape(Y*self.Y_variance+self.Y_mean, [-1])
    
    def run(self, U, show_time=False):
        Y_hat = np.zeros([self.m, U.shape[1]])
        for k in range(U.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            Y_hat[:,k] = self.predict(U[:,k])
        return Y_hat


def rossler_example():
    '''Time Setting'''
    T = 20
    fs = 20
    dt = 1/fs
    
    '''Training Signal Setting'''
    signal_type = Rossler(dt=dt, X0=[0.,0.,0.])
    signal = Signal(signal_type=signal_type, T=T)
    observer = Reservoir(dt=signal.dt, n=signal.Y.shape[0], m=signal.X.shape[0], alpha=0.2, beta=1e-8)
    observer.train(signal.Y, signal.X, show_time=False)

    '''Testing Signal Setting'''
    signal_type = Rossler(dt=dt, X0=signal.X[:,-1])
    # signal_type = Rossler(dt=dt, X0=np.random.uniform(-1,1,3))
    signal = Signal(signal_type=signal_type, T=10*T)
    X_hat = observer.run(signal.Y, show_time=False)

    fontsize = 20
    fig, axes = plt.subplots(3,1,figsize=(8,8), sharex=True)
    axes[0].plot(signal.t, signal.X[0], label=r'$U=x_1$')
    axes[0].legend(fontsize=fontsize-5, loc='lower right')
    axes[0].tick_params(labelsize=fontsize)
    for n in range(2):
        axes[n+1].plot(signal.t, signal.X[n+1], label=r'$x_{}$'.format(n+2))
        axes[n+1].plot(signal.t, X_hat[n+1], label=r'$\hat x_{}$'.format(n+2))
        axes[n+1].legend(fontsize=fontsize-5, loc='lower right', ncol=2)
        axes[n+1].tick_params(labelsize=fontsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('\ntime $t$ [s]', fontsize=fontsize)
    plt.show()
    pass

def lorenz_example():
    '''Time Setting'''
    T = 20
    fs = 20
    dt = 1/fs
    
    '''Training Signal Setting'''
    signal_type = Lorenz(dt=dt, X0=[1.,1.,1.])
    signal = Signal(signal_type=signal_type, T=T)
    observer = Reservoir(dt=signal.dt, n=signal.Y.shape[0], m=signal.X.shape[0], alpha=0.2, beta=1e-8)
    observer.train(signal.Y, signal.X, show_time=False)

    '''Testing Signal Setting'''
    signal_type = Lorenz(dt=dt, X0=signal.X[:,-1])
    # signal_type = Lorenz(dt=dt, X0=np.random.uniform(-1,1,3))
    signal = Signal(signal_type=signal_type, T=T)
    X_hat = observer.run(signal.Y, show_time=False)

    fontsize = 20
    fig, axes = plt.subplots(3,1,figsize=(8,8), sharex=True)
    axes[0].plot(signal.t, signal.X[0], label=r'$U=x_1$')
    axes[0].legend(fontsize=fontsize-5, loc='lower right')
    axes[0].tick_params(labelsize=fontsize)
    for n in range(2):
        axes[n+1].plot(signal.t, signal.X[n+1], label=r'$x_{}$'.format(n+2))
        axes[n+1].plot(signal.t, X_hat[n+1], label=r'$\hat x_{}$'.format(n+2))
        axes[n+1].legend(fontsize=fontsize-5, loc='lower right', ncol=2)
        axes[n+1].tick_params(labelsize=fontsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('\ntime $t$ [s]', fontsize=fontsize)
    plt.show()
    pass


if __name__=='__main__':
    case_dict = dict(
        rossler=rossler_example,
        lorenz=lorenz_example,
    )
    if len(sys.argv) == 2:
        example = case_dict.get(sys.argv[1], None)
        if example is not None:
            example()
            quit()
        else:
            print("Error: Parameters provided incorrectly.")
    else:
        print("Error: Number of parameters is incorrect.")    
    print("\n  Examples:")
    for key in case_dict.keys():
        print("$ python ReservoirObserver.py "+key)
    pass
