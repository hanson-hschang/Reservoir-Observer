"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import sys

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Signal(object):
    """Signal:
        Notation Notes:
            Nd: dimension of states(X)
            M: number of observations(Y)
        Initialize = Signal(signal_type, T)
            signal_type: signal class
            T: float or int, end time in sec
        Members =
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            f(X, t): state tansition function maps states(X) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            X: numpy array with the shape of (Nd,T/dt+1), states in time series
            Y: numpy array with the shape of (M,T/dt+1), observations in time series
    """
    def __init__(self, signal_type, T):
        self.dt = signal_type.dt
        self.T = self.dt*int(T/self.dt)
        self.t = np.arange(0, self.T+self.dt, self.dt)

        self.f = signal_type.f
        self.h = signal_type.h
        
        self.X = np.transpose(odeint(func=self.f, y0=signal_type.X0, t=self.t))
        self.Y = np.reshape(self.h(self.X), [-1, self.t.shape[0]])

class Rossler(object):
    def __init__(self, dt=0.1, X0=[0.,0.,0.], a1=0.5, a2=2., a3=4.):
        self.sigma_B = [0,0,0]
        self.sigma_W = [0]
        self.X0 = X0
        self.dt = dt
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        X_dot[0] = -X[1] - X[2]
        X_dot[1] =  X[0] + self.a1*X[1]
        X_dot[2] =  self.a2 + X[2]*(X[0]-self.a3)
        return X_dot
    
    def h(self, X):
        return X[0]

class Lorenz(object):
    def __init__(self, dt=0.05, X0=[1.,1.,1.], a1=10., a2=28., a3=8/3):
        self.sigma_B = [0,0,0]
        self.sigma_W = [0]
        self.X0 = X0
        self.dt = dt
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        X_dot[0] = -self.a1*X[0] + self.a2*X[1]
        X_dot[1] =  self.a2*X[0] - X[1] - X[0]*X[2]
        X_dot[2] =  -self.a3*X[2] + X[0]*X[1]
        return X_dot
    
    def h(self, X):
        return X[0]

def rossler_example():
    T = 200.
    fs = 20
    dt = 1/fs
    signal_type = Rossler(dt)
    signal = Signal(signal_type=signal_type, T=T)

    fontsize = 20
    _, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()
    pass

def lorenz_example():
    T = 20.
    fs = 20
    dt = 1/fs
    signal_type = Lorenz(dt=dt, X0=[1,1,1])
    signal = Signal(signal_type=signal_type, T=T)

    fontsize = 20
    _, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()
    pass

if __name__ == "__main__":
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
        print("$ python Signal.py "+key)
    pass
    

