#Testing multivariate Kernel Densite Estimation
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import threading

### Original
# mu=np.array([4,10,20])
# #sigma=np.matrix([[4,10,0],[10,25,0],[0,0,100]])
# sigma=np.matrix([[20,10,10],
#                  [10,25,1],
#                  [10,1,50]])
# data=np.random.multivariate_normal(mu,sigma,125000)
# values = data.T

# kde = stats.gaussian_kde(values)

# xmin, ymin, zmin = data.min(axis=0)
# xmax, ymax, zmax = data.max(axis=0)
# xi, yi, zi = np.mgrid[xmin:xmax:10j, ymin:ymax:10j, zmin:zmax:10j]

# # Evaluate the KDE on a regular grid...
# coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

# density = kde(coords)



#THIS PRODUCES A CRAPPY PLOT MAYAVI LOOKS MUCH NICER
# fig, ax = plt.subplots(num=1,subplot_kw=dict(projection='3d'))
# #DELETE x, y, z = values
# #cmap = mpl.cm.bwr
# #DELETEnorm = mpl.colors.Normalize(vmin=0.,vmax=np.max(density))

# #DELETEc=density[i],
# #DELETEcolors = cm.rainbow
# cNorm  = colors.Normalize(vmin=0, vmax=np.max(density))
# scalarMap = cm.ScalarMappable(norm=cNorm, cmap='bwr')

# for i in np.arange(int(coords.shape[1])):
#     color = scalarMap.to_rgba(density[i])
#     ax.scatter(coords[0,i], coords[1,i], coords[2,i], edgecolor=None, cmap='bwr', alpha=density[i]/np.max(density), c='blue')#, c=[list(color[0:3])])#, vmin=0., vmax=np.max(density))
#     #print(saltyburrito)
# #ax.scatter(xi, yi, zi, c=density, alpha=density)
# plt.show(block=False)
# #plt.draw()
# plt.pause(10)


#### Example 2
import numpy as np
from scipy import stats
from mayavi import mlab

mu=np.array([4,10,20])
# Let's change this so that the points won't all lie in a plane...
sigma=np.matrix([[20,10,10],
                 [10,25,1],
                 [10,1,50]])
mu2=np.array([4+5,10-3,20-12])
sigma2=np.matrix([[20,10,10],
                 [10,5,4],
                 [10,4,20]])

data2=np.random.multivariate_normal(mu,sigma,50000)#125000)
data3 =np.random.multivariate_normal(mu2,sigma2,50000)#125000)
data4 = np.concatenate((data2,data3),axis=0)

values2 = data4.T#data2.T

kde = stats.gaussian_kde(values2)
bw_silv = kde.silverman_factor()
bw_scotts = kde.scotts_factor()
print('SCIPY KDE bw silverman: ' + str(bw_silv))
print('SCIPY KDE bw scotts: ' + str(bw_scotts))

# Create a regular 3D grid with 50 points in each dimension
xmin, ymin, zmin = data2.min(axis=0)
xmax, ymax, zmax = data2.max(axis=0)
xi, yi, zi = np.mgrid[xmin:xmax:20j, ymin:ymax:20j, zmin:zmax:20j]

# Evaluate the KDE on a regular grid...
coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
density = kde(coords).reshape(xi.shape)

def visualize3Dcontour(xi,yi,zi,density):
    # Visualize the density estimate as isosurfaces
    out1 = mlab.contour3d(xi, yi, zi, density, opacity=0.2, contours=5, colormap='bwr')
    out2 = mlab.axes()
    #mlab.show(stop=False)
    #mayavi.mlab.savefig(filename, size=None, figure=None, magnification='auto', **kwargs)
    out3 = mlab.close(scene=None, all=False)

# Visualize the density estimate as isosurfaces
fig1 = mlab.figure(size=(1280, 720))
out1 = mlab.contour3d(xi, yi, zi, density, opacity=0.2, contours=10, colormap='bwr', figure=fig1)
out2 = mlab.axes()#xlabel,ylabel,zlabel
#mlab.show(stop=False)
#mayavi.mlab.savefig(filename, size=None, figure=None, magnification='auto', **kwargs)
mlab.savefig('tmp.png', size=None, figure=None, magnification='auto')
#mlab.close()#scene=None, all=False) # creates no output

#threading.Thread(target=visualize3Dcontour(xi,yi,zi,density)).start()

#### Doing Stuff with KDE
low_bounds = [0.9*xmin, 0.9*ymin, 0.9*zmin]
high_bounds = [1.1*xmax, 1.1*ymax, 1.1*zmax]
should_be_one = kde.integrate_box(low_bounds, high_bounds)
print('scipy integral total: ' + str(should_be_one)) #Success, the integral over the full 3D region is 1.


#### SKLEARN KDE ##################################
from sklearn.neighbors import KernelDensity
#DELETE xyz = np.vstack([xi,yi,zi])

#original
d = values2.shape[0]#num dimensions? should be 3 here
n = values2.shape[1]#num samples?
bwsklearn = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
#bw = n**(-1./(d+4)) # scott
print('SKLEARN bw (silverman): {}'.format(bwsklearn))

kde2 = KernelDensity(bandwidth=bw, metric='minkowski',#'euclidean',#
                    kernel='gaussian', algorithm='ball_tree').fit(values2.T, y=None, sample_weight=None) #Should have shape (n_samples, n_features)
#out42 = kde2.fit(values2.T, y=None, sample_weight=None) #Should have shape (n_samples, n_features)

# xmin = np.min(xi)
# xmax = np.max(xi)
# ymin = np.min(yi)
# ymax = np.max(yi)
# zmin = np.min(zi)
# zmax = np.max(zi)
#positions = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])

#DELETE X, Y, Z = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]
#DELETE positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

#density2 = kde2.score_samples(coords.T).reshape(xi.shape)#Inherently flipped from the scipy data input order
density2 = np.reshape(np.exp(kde2.score_samples(coords.T)), xi.shape) #gives ok results but blobs are distinclty separated
#density2 = np.reshape(np.log(kde2.score_samples(coords.T)), xi.shape) #can't be this since
#.reshape(xi.shape) #coords has shape (n_samples, n_features)


# Visualize the density estimate as isosurfaces
fig2 = mlab.figure(size=(1280, 720))
out11 = mlab.contour3d(xi, yi, zi, density2, opacity=0.2,  colormap='bwr', figure=fig2, contours=10)
out12 = mlab.axes()#xlabel,ylabel,zlabel
mlab.savefig('tmp2.png', size=None, figure=None, magnification='auto')



#### Statsmodels KDE
import statsmodels.api as sm
nobs = 300
np.random.seed(1234)  # Seed random generator
c1 = np.random.normal(size=(nobs,1))
c2 = np.random.normal(2, 1, size=(nobs,1))

kde3 = sm.nonparametric.KDEMultivariate(data=values2.T,\
        var_type='ccc', bw='normal_reference')
print('statsmodels KDE bw: ' + str(kde3.bw))

dens_u = np.reshape(kde3.pdf(data_predict=coords.T), xi.shape)

# Visualize the density estimate as isosurfaces
fig3 = mlab.figure(size=(1280, 720))
out21 = mlab.contour3d(xi, yi, zi, dens_u, opacity=0.2,  colormap='bwr', figure=fig3, contours=10)
out22 = mlab.axes()#xlabel,ylabel,zlabel
mlab.savefig('tmp3.png', size=None, figure=None, magnification='auto')

