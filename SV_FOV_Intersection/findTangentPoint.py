import numpy as np


"""
Finds point R lying on 3D line PQ that is closest to the origin, aka tangent to circle about origin in plane PQO
"""
def findR_givenPQ(P,Q):
    PQdist = np.sqrt((Q[0]-P[0])**2+(Q[1]-P[1])**2+(Q[2]-P[2])**2)
    a = (Q[0]-P[0])/PQdist
    b = (Q[1]-P[1])/PQdist
    c = (Q[2]-P[2])/PQdist

    #compute t, the length of the unit vector from P to R
    t = -(a*P[0]+b*P[1]+c*P[2])/(a**2+b**2+c**2)

    #reconstruct vector
    x = P[0]+a*t
    y = P[1]+b*t
    z = P[2]+c*t
    return (x,y,z)

P = (0,1,0)
Q = (0,0,1)
R = findR_givenPQ(P,Q)
print(R)





P = (0,3,0)
Q = (0,0,4)
R = findR_givenPQ(P,Q)
print(R)



import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d')

ax.plot([P[0],P[0]],[P[1],Q[1]],[P[2],Q[2]])
ax.scatter(R[0],R[1],R[2])
ax.scatter(0,0,0)
# # Plot a sin curve using the x and y axes.
# x = np.linspace(0, 1, 100)
# y = np.sin(x * 2 * np.pi) / 2 + 0.5
# ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

# # Plot scatterplot data (20 2D points per colour) on the x and z axes.
# colors = ('r', 'g', 'b', 'k')

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# x = np.random.sample(20 * len(colors))
# y = np.random.sample(20 * len(colors))
# c_list = []
# for c in colors:
#     c_list.extend([c] * 20)
# # By using zdir='y', the y value of these points is fixed to the zs value 0
# # and the (x, y) points are plotted on the x and z axes.
#ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35, roll=0)

plt.show(block=False)



plt.figure()
plt.plot([P[1],Q[1]],[P[2],Q[2]])
plt.scatter(R[1],R[2])
plt.scatter(0,0)
plt.gca().axis('equal')
plt.show(block=False)