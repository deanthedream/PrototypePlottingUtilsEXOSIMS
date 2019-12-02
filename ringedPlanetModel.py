#### Ringed Planet Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Cylinder Equation
def planet_cyl(r_kstar,d_z,R,r_ksunbot1,r_ksunbot2,phi):
    """ Produces a point on the surface of a 3D cylinder
    Args:
        r_kstar (numpy array) - 3D array describing unit vector from star to planet
        d_z (float) - distance along direction of r_kstar
        R (float) - radius of the planet
        r_ksunbot1 (numpy array) - 3D array describing a vector mutually orthogonal to r_ksunbot2 and r_kstar
        r_ksunbot2 (numpy array) - 3D array describing a vector mutually orthogonal to r_ksunbot1 and r_kstar
        phi (float) - angle
    """
    S_cyl = r_kstar * d_z + r_ksunbot1 * R * np.sin(phi) + r_ksunbot2 * R * np.cos(phi)
    return S_cyl

# Circle Equation
def ring_circ(R_r,r_Rbot1,r_Rbot2,phi):
    """ Produces a point on the 3D circle
    Args:
        R_r (float) - circle radius
        r_Rbot1 (numpy array) - 3D array describing a vector mutually orthogonal to r_Rbot2 and the ring plane vector
        r_Rbot2 (numpy array) - 3D array describing a vector mutually orthogonal to r_Rbot1 and the ring plane vector
        phi (float) - angle
    returns:
        S_circ (numpy array) - 3d numpy array defining a single point in space
    """
    S_circ = R_r * r_Rbot1 * np.sin(phi)  + R_r * r_Rbot1 * np.sin(phi)
    return S_circ

def generate_twoVectPerpToVect(vect):
    """ Generate two 3D vectors perpendicular to a 3D vector
    Args:
        vect (nmupy array) - a 3D vector
    Retunrs:
        vect1 (numpy array) - a 3D vector mutually orthogonal to vect and vect2 
        vect2 (numpy array) - a 3D vector mutually orthogonal to vect and vect1
    """
    vect = vect/np.linalg.norm(vect)
    vect_tmp = np.asarray([0.,0.,1.]) # vector along pole axis
    if not (vect == vect_tmp).all(): # Ensure vect_tmp is not coincident with vect
        vect1 = np.cross(vect_tmp,vect)/np.linalg.norm(np.cross(vect_tmp,vect)) # vector along lon. line
    else:
        vect1 = np.asarray([1.,0.,0.])
    vect1 = vect1/np.linalg.norm(vect1)
    vect2 = np.cross(vect,vect1)/np.linalg.norm(np.cross(vect,vect1))
    return vect1, vect2

#### Plot Cylinder and circle

#Genetate Cylinder
r_kstar = np.asarray([1.,0.,0.]) #Vector from the star to the planet
R  = 10. #planet radius
d_z = np.linspace(start=-2.5*R, stop=2.5*R,num=30,endpoint=True) #various distances along r_kstar_hat direction to define distance
r_ksunbot1, r_ksunbot2 = generate_twoVectPerpToVect(r_kstar) #Two perp vect perp to R_kstar
phi1 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_kstar to define the circle



#Generate ring circle
r_r = np.asarray([0.,np.sqrt(2.)/2.,np.sqrt(2.)/2.]) #ring plane normal vector
R_r= 1.5*R #a ring radius
r_Rbot1, r_Rbot2 = generate_twoVectPerpToVect(r_r) #Two perp vect in ring plane perp to r_r
phi2 = np.linspace(start=0.,stop=2*np.pi,num=30) #various angles about r_r

S_circs = list()
for phi in phi2:
    S_circs.append(ring_circ(R_r,r_Rbot1, r_Rbot2,phi))




plt.close(1)
fig1 = plt.figure(num=1)
ax1= fig1.add_subplot(111, projection= '3d')
ax1.plot()

plt.show(block=False)



