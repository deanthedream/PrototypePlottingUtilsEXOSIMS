""" Kalman Filter from https://www.kalmanfilter.net/kalman1d.html 
and https://en.wikipedia.org/wiki/Kalman_filter
There are 5 basicl Kalman Filter Equations
By: Dean Keithly
"""


import numpy as np

#### https://www.kalmanfilter.net/kalman1d.html
def stateUpdate(xhat_nnm1, Kn, zn):
    xhat_nn = xhat_nnm1 + Kn*(zn - xhat_nnm1)
    return xhat_nn

def stateExtrapolation(dt, xhat_nn, xdothat_nn, xddothat_nn):
    xhat_np1n = xhat_nn + dt*xdothat_nn + 1./2.*xddothat_nn*dt**2.
    xdothat_np1n = xdothat_nn + xddothat_nn*dt
    xddothat_np1n = xddothat_nn
    return xhat_np1n, xdothat_np1n, xddothat_np1n

def kalmanGain(p_nnm1,r_n):
    Kn = p_nnm1/(p_nnm1+r_n)
    return Kn

def covarianceUpdate(p_nnm1,Kn):
    p_nn = (1-Kn)*p_nnm1
    return p_nn

def covarianceExtrapolation(p_nn):
    p_np1n = p_nn
    return p_np1n


#### https://en.wikipedia.org/wiki/Kalman_filter
#wk is the unknown drag occuring on the spacecraft

Fk = np.asarray([[1., dt, 0.],[0.,1.,0.],[0.,0.,0.]])
Bk = np.asarray([[0.,0.,0.5*dt**2.],[0.,0.,dt],[1.]])
wk = np.asarray([[0.],[0.],[sigma_thdd]]) #assumes covariance Qk
Hk = np.asarray([[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
vk = np.asarray([[sigma_theta],[0.],[0.]]) # this contains the process noise Rk

def stateUpdate(Fk,x_km1, Bk, uk, wk):
    """
    Fk is the state transition model which is applied to the previous state xk−1;
    x_km1 - the previous state
    Bk is the control-input model which is applied to the control vector uk;
    uk - the control vector
    wk is the process noise, which is assumed to be drawn from a zero mean 
    """
    xk = np.matmul(Fk,x_km1) + np.matmul(Bk,uk) + wk
    return xk


#Left off with zk, Predict, update, and optimal control gain kn