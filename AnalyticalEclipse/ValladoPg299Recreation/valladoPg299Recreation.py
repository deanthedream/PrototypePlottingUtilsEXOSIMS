import numpy as np
import matplotlib.pyplot as plt


rs = 0.4
Rp = 1 #x of the Earth from the sun
re = 0.15
th = np.linspace(start=0.,stop=2.*np.pi,num=1000)
phi = np.arccos((-re+rs)/Rp)

plt.figure()
#sun circle
xs = rs*np.cos(th)
ys = rs*np.sin(th)
plt.plot(xs,ys,color='black',zorder=50)
#Sun Center
plt.scatter(0,0,color='black',zorder=50)
#sun vertical
plt.plot([0,0],[0,rs],color='black',zorder=50)

#Earth
xs = re*np.cos(th) + Rp
ys = re*np.sin(th)
plt.plot(xs,ys,color='black',zorder=50)
#Earth Center
plt.scatter(Rp,0,color='black',zorder=50)


#Sun Center to Earth Center
plt.plot([0,Rp],[0,0],color='black',zorder=50)

#Sun Center to Left of Sun
plt.plot([0,-rs],[0,0],color='black',zorder=50)

#Sun triangle
sunTangentPoint = np.asarray([rs*np.cos(phi),rs*np.sin(phi)])
plt.plot([0,rs*np.cos(phi)],[0,rs*np.sin(phi)],color='black',zorder=50) #hypotenuse
#plt.plot([rs*np.cos(phi),rs*np.cos(phi)],[0,rs*np.sin(phi)],color='black') #vertical

#Earth triangle
plt.plot([Rp,Rp+re*np.cos(phi)],[0,re*np.sin(phi)],color='black',zorder=50) #hypotenuse
#plt.plot([Rp+re*np.cos(phi),Rp+re*np.cos(phi)],[0,re*np.sin(phi)],color='black') #vertical

#Earth Center to Vertex
y = rs/np.cos(phi) - Rp #distance from Earth to Vertex
Vertex = np.asarray([y+Rp,0]) #Vertex
plt.plot([Rp,Rp+y],[0,0],color='black',zorder=50)


#Sun Tangent +y to Vertex
plt.plot([sunTangentPoint[0],Vertex[0]],[sunTangentPoint[1],Vertex[1]],color='black',zorder=50)
plt.plot([sunTangentPoint[0],Vertex[0]],[-sunTangentPoint[1],Vertex[1]],color='black',zorder=50)

#Penumbra Angle
gamma = np.pi - (np.arcsin(re/Rp + rs/Rp) + np.pi/2.)
sunPenumbraTangentmy = np.asarray([rs*np.cos(gamma),-rs*np.sin(gamma)])
sunPenumbraTangentpy = np.asarray([rs*np.cos(gamma),rs*np.sin(gamma)])
#Sun Center to Sun Penumbra Tangent Point negative y
plt.plot([0,sunPenumbraTangentmy[0]],[0,sunPenumbraTangentmy[1]],color='black',zorder=50)
#Sun Center to Sun Penumbra Tangent Point positive y
plt.plot([0,sunPenumbraTangentpy[0]],[0,sunPenumbraTangentpy[1]],color='black',zorder=50)
#Earth Center to Earth Penumbra Tangent Point positive y
plt.plot([Rp,Rp-re*np.cos(gamma)],[0,re*np.sin(gamma)],color='black',zorder=50)
#Earth Center to Earth Penumbra Tangent Point negative y
plt.plot([Rp,Rp-re*np.cos(gamma)],[0,-re*np.sin(gamma)],color='black',zorder=50)
#Negative y Sun Penumbra Tanget Point to Positive y Earth Penumbra tangent point
plt.plot([sunPenumbraTangentmy[0],Rp-re*np.cos(gamma)],[sunPenumbraTangentmy[1],re*np.sin(gamma)],color='black',zorder=50)
#Positive y Sun Penumbra Tanget Point to Negative y Earth Penumbra tangent point
plt.plot([sunPenumbraTangentpy[0],Rp-re*np.cos(gamma)],[sunPenumbraTangentpy[1],-re*np.sin(gamma)],color='black',zorder=50)
#Positive y Earth Penumbra tangent point to VERTEX X, positive Y
yPenumbraAtVertexX = np.sin(np.pi/2. - gamma)*(y+rs*np.sin(gamma))
plt.plot([Rp-re*np.cos(gamma),Rp+y],[re*np.sin(gamma),yPenumbraAtVertexX+re*np.sin(gamma)],color='black',zorder=50)
plt.plot([Rp+y,Rp+y],[0,yPenumbraAtVertexX+re*np.sin(gamma)],color='black',zorder=50) #far right vertical line positive
#Negative Y Earth Penumbra tangent point to Vertex X, positive Y
plt.plot([Rp-re*np.cos(gamma),Rp+y],[-re*np.sin(gamma),-yPenumbraAtVertexX-re*np.sin(gamma)],color='black',zorder=50)
plt.plot([Rp+y,Rp+y],[0,-yPenumbraAtVertexX-re*np.sin(gamma)],color='black',zorder=50) #far right vertical line negative


#from matplotlib.patches import Polygon
#from matplotlib.collections import PatchCollection

#Plot Light Grey Triangles, Penumbra
#Rp-re*np.cos(gamma),Rp+y],[-re*np.sin(gamma),-yPenumbraAtVertexX-re*np.sin(gamma)
penumbraPositiveYcorners = np.asarray([[Rp-re*np.cos(gamma),re*np.sin(gamma)],[Rp+y,0],[Rp+y,yPenumbraAtVertexX+re*np.sin(gamma)]])
t1 = plt.Polygon(penumbraPositiveYcorners, color='lightgrey',zorder=5)
plt.gca().add_patch(t1)
penumbraNegativeYcorners = np.asarray([[Rp-re*np.cos(gamma),-re*np.sin(gamma)],[Rp+y,0],[Rp+y,-yPenumbraAtVertexX-re*np.sin(gamma)]])
t2 = plt.Polygon(penumbraNegativeYcorners, color='lightgrey',zorder=5)
plt.gca().add_patch(t2)
#Plot Umbra
umbraCorners = np.asarray([[Rp+re*np.cos(phi),re*np.sin(phi)],[Rp+re*np.cos(phi),-re*np.sin(phi)],[Rp+y,0]])
t3 = plt.Polygon(umbraCorners, color='grey',zorder=10)
plt.gca().add_patch(t3)
#Plot Earth Circle
#from matplotlib.patches import Circle
t4 = plt.Circle((Rp,0),radius=re,color='white',zorder=20)
plt.gca().add_patch(t4)

plt.axes().set_aspect('equal')
plt.show(block=False)
