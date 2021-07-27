#Plot 3D cone Intersecting Orbit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as axes3d

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

u = np.linspace(-1, 2, 60)
v = np.linspace(0, 2*np.pi, 60)
U, V = np.meshgrid(u, v)

X = U
#Y1 = (U**2 + 1)*np.cos(V)
#Z1 = (U**2 + 1)*np.sin(V)

Y2 = (U + 3)*np.cos(V)
Z2 = (U + 3)*np.sin(V)

#ax.plot_surface(X, Y1, Z1, alpha=0.3, color='red', rstride=6, cstride=12)
ax.plot_surface(X, Y2, Z2, alpha=0.3, color='blue', rstride=6, cstride=12, edgecolor='black')
plt.show(block=False)



r_gs = np.asarray([0.25,0.3,0.75])
r_look = np.asarray([0.3,0.1,-0.75])/np.linalg.norm(np.asarray([0.3,0.1,-0.75]))
t=1

num=8646873213547
plt.close(num)
fig2 = plt.figure(num=num)
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')

ax2.scatter(r_gs[0],r_gs[1],r_gs[2],color='black') #GS location
ax2.scatter(r_gs[0],r_gs[1],0,color='black') #projected GS location
ax2.plot([0,r_gs[0]],[0,r_gs[1]],[0,0],linestyle='--',color='black') #line from origin to projected GS location
ax2.plot([0,r_gs[0]],[0,r_gs[1]],[0,r_gs[2]],color='black') #line from origin to GS location

ax2.plot([r_gs[0],r_gs[0]+r_look[0]*t],[r_gs[1],r_gs[1]+r_look[1]*t],[r_gs[2],r_gs[2]+r_look[2]*t],color='purple') #look vector
ax2.plot([r_gs[0]+r_look[0]*t],[r_gs[1]+r_look[1]*t],[r_gs[2]+r_look[2]*t],color='purple') #look vector endpoint

th = np.linspace(start=0,stop=2.*np.pi,num=100)
xe = 0.8*np.cos(th)
ye = 0.6*np.sin(th)
ax2.plot(xe,ye,np.zeros(len(xe)),color='blue')


#From https://stackoverflow.com/questions/48703275/3d-truncated-cone-in-python
n = 80
t = np.linspace(0, mag, n)
theta = np.linspace(0, 2 * np.pi, n)
# use meshgrid to make 2d arrays
t, theta = np.meshgrid(t, theta)
R = np.linspace(R0, R1, n)
# generate coordinates for surface
X, Y, Z = [p0[i] + v[i] * t + R *
           np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]


plt.show(block=False)
