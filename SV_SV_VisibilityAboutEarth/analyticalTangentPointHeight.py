
import numpy as np
import matplotlib.pyplot as plt

"""
From SV_SV_visibilityAboutEarth.ipynb
h
"""
def tangentHeight(a_p,e_p,i_p,W_p,w_p,nu_p,a_q,e_q,i_q,W_q,w_q,nu_q):
    A_p = a_p*(1.-e_p**2.)
    B_p = np.sin(W_p)*np.cos(i_p)*np.cos(w_p)
    C_p = np.sin(W_p)*np.sin(w_p)*np.cos(i_p)
    D_p = np.sin(w_p)*np.cos(W_p)
    E_p = np.cos(W_p)*np.cos(w_p)
    F_p = np.sin(W_p)*np.sin(w_p)
    G_p = np.sin(W_p)*np.cos(w_p)
    H_p = np.cos(W_p)*np.cos(i_p)*np.cos(w_p)
    I_p = np.sin(w_p)*np.cos(W_p)*np.cos(i_p)
    J_p = np.sin(i_p)*np.cos(w_p)
    K_p = np.sin(i_p)*np.sin(w_p)

    A_q = a_q*(1.-e_q**2.)
    B_q = np.sin(W_q)*np.cos(i_q)*np.cos(w_q)
    C_q = np.sin(W_q)*np.sin(w_q)*np.cos(i_q)
    D_q = np.sin(w_q)*np.cos(W_q)
    E_q = np.cos(W_q)*np.cos(w_q)
    F_q = np.sin(W_q)*np.sin(w_q)
    G_q = np.sin(W_q)*np.cos(w_q)
    H_q = np.cos(W_q)*np.cos(i_q)*np.cos(w_q)
    I_q = np.sin(w_q)*np.cos(W_q)*np.cos(i_q)
    J_q = np.sin(i_q)*np.cos(w_q)
    K_q = np.sin(i_q)*np.sin(w_q)



    h = (-(A_p*J_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*K_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.))*(-A_p*J_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*K_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
     + A_q*J_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*K_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)) + (-A_p*B_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*C_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
      - A_p*D_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*E_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.))*(-A_p*B_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*C_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
       - A_p*D_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*E_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_q*B_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*C_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)\
        + A_q*D_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) - A_q*E_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)) - (-A_p*F_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*G_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
         + A_p*H_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*I_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.))*(A_p*F_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*G_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
          - A_p*H_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*I_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_q*F_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*G_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)\
           + A_q*H_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*I_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)))/np.sqrt((-A_p*J_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*K_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
            + A_q*J_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*K_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.))**2 + (A_p*B_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) + A_p*C_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
             + A_p*D_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*E_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_q*B_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) - A_q*C_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)\
              - A_q*D_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*E_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.))**2 + (A_p*F_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*G_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.)\
               - A_p*H_p*np.sin(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_p*I_p*np.cos(nu_p)/(e_p*np.cos(nu_p) + 1.) - A_q*F_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*G_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.)\
                + A_q*H_q*np.sin(nu_q)/(e_q*np.cos(nu_q) + 1.) + A_q*I_q*np.cos(nu_q)/(e_q*np.cos(nu_q) + 1.))**2)

    return h


a_p = 6371+5000
e_p=0.
i_p = np.pi/4.
W_p = 1.5
w_p = 5
nu_p = 0.75
a_q = 6371+1200.
e_q = 0.15
i_q = np.pi/2.
W_q = 6
w_q = 2.5
nu_q = 2.

nus = np.linspace(start=0.,stop=2.*np.pi,num=300)
nu1s,nu2s = np.meshgrid(nus,nus)

h = tangentHeight(a_p,e_p,i_p,W_p,w_p,nu_p,a_q,e_q,i_q,W_q,w_q,nu_q)

hs = np.zeros(nu1s.shape)
for i in np.arange(nu1s.shape[0]):
    for j in np.arange(nu1s.shape[1]):
        hs[i,j] = tangentHeight(a_p,e_p,i_p,W_p,w_p,nu1s[i,j],a_q,e_q,i_q,W_q,w_q,nu2s[i,j])

plt.figure(num=1)
plt.contourf(nu1s,nu2s,hs,cmap='bwr',levels=100)
plt.colorbar()
plt.show(block=False)




from scipy.optimize import minimize_scalar
def erf(x,h_avg,a_p,e_p,i_p,W_p,w_p,nu1,a_q,e_q,i_q,W_q,w_q):
    return np.abs(h_avg- tangentHeight(a_p,e_p,i_p,W_p,w_p,nu1,a_q,e_q,i_q,W_q,w_q,x))

h_avg = np.mean([np.max(hs),np.min(hs)])
pts = list()
pts2 = list()
for i in np.arange(len(nus)):
    for j in np.arange(len(nus)-1):
        if (tangentHeight(a_p,e_p,i_p,W_p,w_p,nus[i],a_q,e_q,i_q,W_q,w_q,nus[j]) < h_avg and \
            tangentHeight(a_p,e_p,i_p,W_p,w_p,nus[i],a_q,e_q,i_q,W_q,w_q,nus[j+1]) > h_avg) or\
            (tangentHeight(a_p,e_p,i_p,W_p,w_p,nus[i],a_q,e_q,i_q,W_q,w_q,nus[j]) > h_avg and \
            tangentHeight(a_p,e_p,i_p,W_p,w_p,nus[i],a_q,e_q,i_q,W_q,w_q,nus[j+1]) < h_avg):
            pts2.append([nus[i],nus[j]])
            # out = minimize_scalar(erf,bounds=(nus[j],nus[j+1]),args=(h_avg,a_p,e_p,i_p,W_p,w_p,nus[i],a_q,e_q,i_q,W_q,w_q))
            # if out.x > 2.*np.pi:
            #     out.x = out.x-2.*np.pi
            # elif out.x < 0.:
            #     out.x = out.x+2.*np.pi
            # pts.append([nus[i],out.x])
#pts = np.asarray(pts)
pts2 = np.asarray(pts2)

plt.figure(num=2)
plt.contourf(nu1s,nu2s,hs,cmap='bwr',levels=100)
plt.colorbar()
#plt.scatter(pts[:,0],pts[:,1],color='blue')
plt.scatter(pts2[:,0],pts2[:,1],color='purple')
plt.show(block=False)



# A_p**2*(e_q*x + 1)**2
# *(
#     -(J_p*sqrt(1 - y**2) + K_p*y)
#     *(A_p*(J_p*sqrt(1 - y**2) + K_p*y)*(e_q*x + 1) - A_q*(J_q*sqrt(1 - x**2) + K_q*x)*(e_p*y + 1))
#     -(
#         A_p*(e_q*x + 1)*(B_p*sqrt(1 - y**2)+ C_p*y+ D_p*sqrt(1 - y**2)- E_p*y)
#         + A_q*(e_p*y + 1)*(-B_q*sqrt(1 - x**2)- C_q*x- D_q*sqrt(1 - x**2)+ E_q*x)
#     )
#     *(
#         B_p*sqrt(1 - y**2)
#         + C_p*y
#         + D_p*sqrt(1 - y**2)
#         - E_p*y
#     )
#     + (A_p*(e_q*x + 1)*(F_p*sqrt(1 - y**2) - G_p*y - H_p*sqrt(1 - y**2) - I_p*y) + A_q*(e_p*y + 1)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))
#     *(-F_p*sqrt(1 - y**2) + G_p*y + H_p*sqrt(1 - y**2) + I_p*y)
# )
# *(
#     -2*(J_p*y - K_p*sqrt(1 - y**2))
#     *(A_p*(J_p*sqrt(1 - y**2) + K_p*y)*(e_q*x + 1) - A_q*(J_q*sqrt(1 - x**2) + K_q*x)*(e_p*y + 1))
#     - 2*(J_p*sqrt(1 - y**2) + K_p*y)
#     *(A_p*(J_p*y - K_p*sqrt(1 - y**2))*(e_q*x + 1)+ A_q*e_p*sqrt(1 - y**2)*(J_q*sqrt(1 - x**2) + K_q*x))
#     - 2*(A_p*(e_q*x + 1)*(B_p*y - C_p*sqrt(1 - y**2) + D_p*y + E_p*sqrt(1 - y**2)) - A_q*e_p*sqrt(1 - y**2)*(-B_q*sqrt(1 - x**2) - C_q*x - D_q*sqrt(1 - x**2) + E_q*x))
#     *(B_p*sqrt(1 - y**2) + C_p*y + D_p*sqrt(1 - y**2) - E_p*y)
#     - 2*(A_p*(e_q*x + 1)*(B_p*sqrt(1 - y**2) + C_p*y + D_p*sqrt(1 - y**2) - E_p*y) + A_q*(e_p*y + 1)*(-B_q*sqrt(1 - x**2) - C_q*x - D_q*sqrt(1 - x**2) + E_q*x))
#     *(B_p*y - C_p*sqrt(1 - y**2) + D_p*y + E_p*sqrt(1 - y**2))
#     + 2*(A_p*(e_q*x + 1)*(F_p*y + G_p*sqrt(1 - y**2) - H_p*y + I_p*sqrt(1 - y**2)) - A_q*e_p*sqrt(1 - y**2)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))
#     *(-F_p*sqrt(1 - y**2) + G_p*y + H_p*sqrt(1 - y**2) + I_p*y)
#     + 2*(A_p*(e_q*x + 1)*(F_p*sqrt(1 - y**2) - G_p*y - H_p*sqrt(1 - y**2) - I_p*y) + A_q*(e_p*y + 1)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))
#     *(-F_p*y - G_p*sqrt(1 - y**2) + H_p*y - I_p*sqrt(1 - y**2))
# )
# + 2*R**2*e_p*sqrt(1 - y**2)*(e_p*y + 1)*(e_q*x + 1)**2
# *(
#     (A_p*(J_p*sqrt(1 - y**2) + K_p*y)*(e_q*x + 1) - A_q*(J_q*sqrt(1 - x**2) + K_q*x)*(e_p*y + 1))**2
#     + (A_p*(e_q*x + 1)*(B_p*sqrt(1 - y**2) + C_p*y + D_p*sqrt(1 - y**2) - E_p*y) + A_q*(e_p*y + 1)*(-B_q*sqrt(1 - x**2) - C_q*x - D_q*sqrt(1 - x**2) + E_q*x))**2
#     + (A_p*(e_q*x + 1)*(F_p*sqrt(1 - y**2) - G_p*y - H_p*sqrt(1 - y**2) - I_p*y) + A_q*(e_p*y + 1)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))**2
# )
# - R**2*(e_p*y + 1)**2*(e_q*x + 1)**2
# *(
#     (2*A_p*(J_p*y - K_p*sqrt(1 - y**2))*(e_q*x + 1) + 2*A_q*e_p*sqrt(1 - y**2)*(J_q*sqrt(1 - x**2) + K_q*x))
#     *(A_p*(J_p*sqrt(1 - y**2) + K_p*y)*(e_q*x + 1) - A_q*(J_q*sqrt(1 - x**2) + K_q*x)*(e_p*y + 1))
#     + (2*A_p*(e_q*x + 1)*(B_p*y - C_p*sqrt(1 - y**2) + D_p*y + E_p*sqrt(1 - y**2)) - 2*A_q*e_p*sqrt(1 - y**2)*(-B_q*sqrt(1 - x**2) - C_q*x - D_q*sqrt(1 - x**2) + E_q*x))
#     *(A_p*(e_q*x + 1)*(B_p*sqrt(1 - y**2) + C_p*y + D_p*sqrt(1 - y**2) - E_p*y) + A_q*(e_p*y + 1)*(-B_q*sqrt(1 - x**2) - C_q*x - D_q*sqrt(1 - x**2) + E_q*x))
#     + (2*A_p*(e_q*x + 1)*(F_p*y + G_p*sqrt(1 - y**2) - H_p*y + I_p*sqrt(1 - y**2)) - 2*A_q*e_p*sqrt(1 - y**2)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))
#     *(A_p*(e_q*x + 1)*(F_p*sqrt(1 - y**2) - G_p*y - H_p*sqrt(1 - y**2) - I_p*y)
#     + A_q*(e_p*y + 1)*(-F_q*sqrt(1 - x**2) + G_q*x + H_q*sqrt(1 - x**2) + I_q*x))
# )








# (1 - cos(2*nu_p))**2*(1 - cos(2*nu_q))**2
# + (1 - cos(2*nu_p))**2*sin(nu_q)
# - (1 - cos(2*nu_p))**2*sin(2*nu_q)
# - (1 - cos(2*nu_p))**2*sin(3*nu_q)
# - (1 - cos(2*nu_p))**2*sin(4*nu_q)
# + (1 - cos(2*nu_p))**2*cos(nu_q)
# + (1 - cos(2*nu_p))**2*cos(2*nu_q)
# + (1 - cos(2*nu_p))**2*cos(3*nu_q)
# - (1 - cos(2*nu_p))**2
# + (1 - cos(2*nu_q))**2*sin(nu_p)
# + (1 - cos(2*nu_q))**2*sin(2*nu_p)
# + (1 - cos(2*nu_q))**2*sin(3*nu_p)
# + (1 - cos(2*nu_q))**2*sin(4*nu_p)
# - (1 - cos(2*nu_q))**2*cos(nu_p)
# - (1 - cos(2*nu_q))**2*cos(2*nu_p)
# - (1 - cos(2*nu_q))**2*cos(3*nu_p)
# - (1 - cos(2*nu_q))**2
# - sin(nu_p)
# - sin(2*nu_p)
# - sin(3*nu_p)
# - sin(4*nu_p)
# + sin(nu_q)
# - sin(2*nu_q)
# - sin(3*nu_q)
# - sin(4*nu_q)
# + sin(nu_p - 4*nu_q)
# + sin(nu_p - 3*nu_q)
# + sin(nu_p - 2*nu_q)
# + sin(nu_p - nu_q)
# + sin(nu_p + nu_q)
# + sin(nu_p + 2*nu_q)
# - sin(nu_p + 3*nu_q)
# - sin(nu_p + 4*nu_q)
# + sin(2*nu_p - 4*nu_q)
# + sin(2*nu_p - 3*nu_q)
# + sin(2*nu_p - 2*nu_q)
# + sin(2*nu_p - nu_q)
# + sin(2*nu_p + nu_q)
# + sin(2*nu_p + 2*nu_q)
# + sin(2*nu_p + 3*nu_q)
# - sin(2*nu_p + 4*nu_q)
# + sin(3*nu_p - 4*nu_q)
# + sin(3*nu_p - 3*nu_q)
# + sin(3*nu_p - 2*nu_q)
# + sin(3*nu_p - nu_q)
# + sin(3*nu_p + nu_q)
# + sin(3*nu_p + 2*nu_q)
# + sin(3*nu_p + 3*nu_q)
# - sin(3*nu_p + 4*nu_q)
# + sin(4*nu_p - 3*nu_q)
# + sin(4*nu_p - 2*nu_q)
# + sin(4*nu_p - nu_q)
# + sin(4*nu_p + nu_q)
# + sin(4*nu_p + 2*nu_q)
# + sin(4*nu_p + 3*nu_q)
# - cos(nu_p)
# - cos(2*nu_p)
# + cos(3*nu_p)
# + cos(nu_q)
# - cos(2*nu_q)
# - cos(3*nu_q)
# + cos(nu_p - 4*nu_q)
# - cos(nu_p - 3*nu_q)
# + cos(nu_p - 2*nu_q)
# + cos(nu_p - nu_q)
# + cos(nu_p + nu_q)
# - cos(nu_p + 2*nu_q)
# - cos(nu_p + 3*nu_q)
# - cos(nu_p + 4*nu_q)
# + cos(2*nu_p - 4*nu_q)
# + cos(2*nu_p - 3*nu_q)
# + cos(2*nu_p - 2*nu_q)
# + cos(2*nu_p - nu_q)
# + cos(2*nu_p + nu_q)
# - cos(2*nu_p + 2*nu_q)
# - cos(2*nu_p + 3*nu_q)
# - cos(2*nu_p + 4*nu_q)
# + cos(3*nu_p - 4*nu_q)
# + cos(3*nu_p - 3*nu_q)
# + cos(3*nu_p - 2*nu_q)
# + cos(3*nu_p - nu_q)
# + cos(3*nu_p + nu_q)
# + cos(3*nu_p + 2*nu_q)
# - cos(3*nu_p + 3*nu_q)
# - cos(3*nu_p + 4*nu_q)
# + cos(4*nu_p - 4*nu_q)
# + cos(4*nu_p - 3*nu_q)
# + cos(4*nu_p - 2*nu_q)
# + cos(4*nu_p - nu_q)
# - cos(4*nu_p + nu_q)
# - cos(4*nu_p + 2*nu_q)
# - cos(4*nu_p + 3*nu_q)
# - cos(4*nu_p + 4*nu_q)
# - 894184.586379905


