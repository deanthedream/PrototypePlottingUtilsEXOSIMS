####
import numpy as np
from numba import jit, cuda
#pip3 install numba
#https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
#pip3 install jax
#pip3 install jaxlib
import jax
#import jax.numpy as jxnp

def projected_apbpPsipsi(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        dmajorp (numpy array):
            Semi-major axis of the projected ellipse in AU
        dminorp (numpy array):
            Semi-minor axis of the projected ellipse in AU
        Psi (numpy array):
            Angle between Op1 and OpQp
        psi (numpy array):
            Angle between OpQ and x-axis
    """

    #sqrt to np.sqrt
    #Abs to np.abs
    #sin to np.sin
    #cos to np.cos
    #atan to np.arctan
    #1.0* to 
    #3.14159265358979 to np.pi
    Gamma = e*(1 - e**2)
    gamma = (np.sin(W)*np.cos(w) + np.sin(w)*np.cos(W)*np.cos(inc))
    Phi = np.sqrt((e+1)/(1-e)) #np.sqrt(e + 1)*np.sqrt(1/(1 - e))
    phi = a**2*(e**2 - 1)**2
    #DELETE lam1 = np.sin(W)*np.sin(w)*np.cos(inc) - np.cos(W)*np.cos(w)
    lam2 = (-np.sin(W)*np.sin(w)*np.cos(inc) + np.cos(W)*np.cos(w))
    Omicron = (np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))
    Eeps = (e + 1)**2
    Eeps2 = a*(1 - e**2)
    Gorgon = (-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))
    Gemini = (e*np.cos(2*np.arctan(Phi)) + 1)

    #Specific Calc Substitutions
    Ramgy = ((e + 1)*np.sqrt(phi*gamma**2/Eeps + phi*(np.sin(W)*np.sin(w)*np.cos(inc) - np.cos(W)*np.cos(w))**2/Eeps + phi*np.sin(inc)**2*np.sin(w)**2/Eeps))
    Affinity1 = a**2*Gamma*gamma/Ramgy
    Yolo1 = (np.sin(W)*np.cos(w + np.pi) + np.sin(w + np.pi)*np.cos(W)*np.cos(inc))
    Kolko1 = (-np.sin(W)*np.sin(w + np.pi)*np.cos(inc) + np.cos(W)*np.cos(w + np.pi))
    #Semi-major axis length
    dmajorp = np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #Semi-minor axis length
    dminorp = -np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #Angle between OpQ and OpQp
    Psi = np.arccos(((Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)) + (a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
             - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)))/(np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1\
             + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)*np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2\
             + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)))
    #Angle between X-axis and Op!
    psi = np.arccos((a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))/np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
         - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2))


    #Arctan could be +pi
    Omicron_v2 = (np.sin(W)*np.cos(w + 2*(np.arctan(Phi) + np.pi)) + np.sin(w + 2*(np.arctan(Phi) + np.pi))*np.cos(W)*np.cos(inc)) #the arctan here may be between -pi/2 to pi/2 or pi/2 and 3pi/2
    #Semi-major axis length v2
    dmajorp_v2 = np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #Semi-minor axis length v2
    dminorp_v2 = -np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #Angle between OpQ and OpQp v2
    Psi_v2 = np.arccos(((Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)) + (a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
             - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)))/(np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1\
             + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)*np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2\
             + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)))
    #Angle between X-axis and Op! v2
    psi_v2 = np.arccos((a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))/np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
         - (-Affinity1 - Eeps2*Omicron_v2/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron_v2/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2))




    #theta_OpQ_X
    theta_OpQ_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2\
         + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2), a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1\
          - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2))

    #theta_OpQp_X
    theta_OpQp_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy\
         + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2), a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi))\
          + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2))

    return dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X, dmajorp_v2, dminorp_v2, Psi_v2, psi_v2

def xyz_3Dellipse(a,e,W,w,inc,v):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
        v (numpy array):
            true anomaly in Radians
    Returns:
        r (numpy array):
            x, y, z by n
    """
    eqnr = a*(1-e**2)/(1+e*np.cos(v))
    eqnX = eqnr*(np.cos(W)*np.cos(w+v) - np.sin(W)*np.sin(w+v)*np.cos(inc))
    eqnY = eqnr*(np.sin(W)*np.cos(w+v) + np.cos(W)*np.sin(w+v)*np.cos(inc))
    eqnZ = eqnr*(np.sin(inc)*np.sin(w+v))
    return np.asarray([[eqnX],[eqnY],[eqnZ]])

def projected_Op(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        r_center (numpy array):
            Projected ellipse center in AU. [x_center, y_center]
    """
    # eqnr = a*(1-e**2)/(1+e)
    # eqnX = eqnr*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(inc))
    # eqnY = eqnr*(np.sin(W)*np.cos(w) + np.cos(W)*np.sin(w)*np.cos(inc))
    # eqnZ = eqnr*(np.sin(inc)*np.sin(w))
    # c_ae = a*e #linear eccentricity
    # rhat = np.asarray([[eqnX],[eqnY],[eqnZ]])/np.linalg.norm(np.asarray([[eqnX],[eqnY],[eqnZ]]), ord=2, axis=1, keepdims=True)
    # O = -rhat*c_ae
    # #r_perigee3D = sp.Matrix([[eqnX], [eqnY], [eqnZ]])
    # #rhat_perigee3D = r_perigee3D/r_perigee3D.norm()
    # #DELETErhat = np.asarray([[eqnX],[eqnY],[eqnZ]])/np.asarray([eqnX**2+eqnY**2+eqnZ**2])
    # # r_3Dellipsecenter = -rhat_perigee3D*c_ae
    # # O = -rhat_perigee3D*c_ae
    # return np.asarray([O[0][0], O[1][0]])
    r1 = xyz_3Dellipse(a,e,W,w,inc,0.)
    r2 = xyz_3Dellipse(a,e,W,w,inc,np.pi)
    r_center = (r1+r2)/2
    return np.asarray([r_center[0][0],r_center[1][0]])

def projected_BpAngle(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        theta (numpy array):
            Angle of semi-major axis from X-axis in Radians
    """
    eqnr = a*(1-e**2)/(1+e)
    eqnX = eqnr*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(inc))
    eqnY = eqnr*(np.sin(W)*np.cos(w) + np.cos(W)*np.sin(w)*np.cos(inc))
    eqnZ = eqnr*(np.sin(inc)*np.sin(w))
    rhat = np.asarray([[eqnX],[eqnY],[eqnZ]])/np.linalg.norm(np.asarray([[eqnX],[eqnY],[eqnZ]]), ord=2, axis=0, keepdims=True)

    return np.arctan(rhat[1][0],rhat[0][0])

def projected_projectedLinearEccentricity(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        c_3D_projected (numpy array):
            linear eccentricity of 3D ellipse's projected distance on plane of 2D ellipse in AU
    """
    O = projected_Op(a,e,W,w,inc)
    c_3D_projected = np.linalg.norm(O, ord=2, axis=0, keepdims=True)
    return c_3D_projected[0]

def derotate_arbitraryPoint(x,y,phi):
    """
    Args:
        x (numpy array):
            distance from focus (star) to Op along X-axis
        y (numpy array):
            distance from focus (star) to Op along Y-axis
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 
    Returns:
        x2 (numpy array):
            de-rotated distance from origin to star focus along semi-major axis
        y2 (numpy array):
            de-rotated distance from origin to star focus along semi-minor axis
    """
    ahat_x = np.cos(phi)
    ahat_y = np.sin(phi)
    bhat_x = -np.sin(phi)
    bhat_y = np.cos(phi)

    #portion of FOp in direction of ahat
    x2 = ahat_x*x + ahat_y*y
    y2 = bhat_x*x + bhat_y*y

    return -x2, -y2

def derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op):
    """ Take the position and angle of the geometric center of the projected ellipse (relative to the orbiting Foci located at 0,0)
    and, assuming the geometric center of the ellipse is now at (0,0) return the location of the orbiting foci relative to (0,0)
    of the derotated ellipse
    Args:
        theta_OpQ_X (numpy array):
            angle of OpQ from x-axis
        theta_OpQp_X (numpy array):
            angle of OpQp from x-axis
        Op (numpy array):
            projected ellipse position [x,y] relative to star (origin)
    Return:
        x (numpy array):
            x coordinates of host star relative to projected ellipse center
        y (numpy array):
            y coordinates of host star relative to projected ellipse center
        Phi (numpy array):
            Angle of projected ellipse semi-major axis from x-axis
    """
    Phi = (theta_OpQ_X+theta_OpQp_X)/2
    x, y = derotate_arbitraryPoint(Op[0],Op[1],Phi)
    return x, y, Phi

def rerotateEllipsePoints(xpts,ypts,Phi,Ox,Oy):
    """ A function designed to take a point on the derotated ellipse and convert it to the corresponding point on the rotated ellipse
    Args:
        xpts (numpy array):
            the x points in the derotated frame to rerotate
        ypts (numpy array):
            the y points in the derotated frame to rerotate
        Phi (numpy array):
            the angle to rotate these points by
        Ox (numpy array):
            the x component of the geometric origin of the projected ellipse
        Oy (numpy array):
            the y component of the geometric origin of the projected ellipse
    Returns:
        ux (numpy array):
            the rerotated x points
        uy (numpy array):
            the rerotated y points
    """
    ux = np.cos(Phi)*xpts - np.sin(Phi)*ypts + Ox
    uy = np.sin(Phi)*xpts + np.cos(Phi)*ypts + Oy
    #DELETE ux = np.cos(Phi[ind])*minSepPoints_x[ind] - np.sin(Phi[ind])*minSepPoints_y[ind] + Op[0][ind] 
    #DELETE uy = np.sin(Phi[ind])*minSepPoints_x[ind] + np.cos(Phi[ind])*minSepPoints_y[ind] + Op[1][ind] 
    return ux, uy



def roots_vec(p):
    p = np.atleast_1d(p)
    n = p.shape[-1]
    A = np.zeros(p.shape[:1] + (n-1, n-1), float)
    A[...,1:,:-1] = np.eye(n-2)
    A[...,0,:] = -p[...,1:]/p[...,None,0]
    return np.linalg.eigvals(A)

def roots_loop(p):
    r = []
    for pp in p:
        r.append(np.roots(pp))
    return r

def quarticCoefficients(a,b,mx,my):
    """ Calculates coefficients to the quartic polynomial from projected ellipse semi parameters and
    projected strictly positive star coordinates
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        parr (numpy array):
            quartic coefficients for each star with shape n planets by 5 coefficients
    """
    A = -(2 - 2*b**2/a**2)**2/a**2
    B = 4*mx*(2 - 2*b**2/a**2)/a**2
    C = -4*my**2*b**2/a**4 - 4*mx**2/a**2 + (2 - 2*b**2/a**2)**2
    D = -4*mx*(2-2*b**2/a**2)
    E = 4*mx**2
    return np.asarray([A, B, C, D, E]).T

def quarticSolutions(a,b,mx,my):
    """ Runs and separates out real and imaginary components of the quartic solutions
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        xreal (numpy array):
            real component of x solutions to the quartic
        imag (numpy array):
            imaginary component of x solutions to the quartic
    """
    #DELETEparr = np.asarray([A,B,C,D,E]).T #[B/A,C/A,D/A,E/A]
    #DELETEparr = quarticCoefficients(a,b,mx,my)
    out = np.asarray(roots_loop(quarticCoefficients(a,b,mx,my)))
    xreal = np.real(out)
    # print('Number of nan in x0: ' + str(np.count_nonzero(np.isnan(xreal[:,0]))))
    # print('Number of nan in x1: ' + str(np.count_nonzero(np.isnan(xreal[:,1]))))
    # print('Number of nan in x2: ' + str(np.count_nonzero(np.isnan(xreal[:,2]))))
    # print('Number of nan in x3: ' + str(np.count_nonzero(np.isnan(xreal[:,3]))))
    # print('Number of non-zero in x0: ' + str(np.count_nonzero(xreal[:,0] != 0)))
    # print('Number of non-zero in x1: ' + str(np.count_nonzero(xreal[:,1] != 0)))
    # print('Number of non-zero in x2: ' + str(np.count_nonzero(xreal[:,2] != 0)))
    # print('Number of non-zero in x3: ' + str(np.count_nonzero(xreal[:,3] != 0)))
    # print('Number of non-zero in x0+x1+x2+x3: ' + str(np.count_nonzero((xreal[:,0] != 0)*(xreal[:,1] != 0)*(xreal[:,2] != 0)*(xreal[:,3] != 0))))
    imag = np.imag(out)
    # x0_i = imag[:,0]
    # x1_i = imag[:,1]
    # x2_i = imag[:,2]
    # x3_i = imag[:,3]
    # print('Number of non-zero in x0_i: ' + str(np.count_nonzero(x0_i != 0)))
    # print('Number of non-zero in x1_i: ' + str(np.count_nonzero(x1_i != 0)))
    # print('Number of non-zero in x2_i: ' + str(np.count_nonzero(x2_i != 0)))
    # print('Number of non-zero in x3_i: ' + str(np.count_nonzero(x3_i != 0)))
    # print('Number of non-zero in x0_i+x1_i+x2_i+x3_i: ' + str(np.count_nonzero((x0_i != 0)*(x1_i != 0)*(x2_i != 0)*(x3_i != 0))))
    
    return xreal, imag

def ellipseYFromX(xreal, a, b):
    """ Calculates y values in the positive quadrant 
    Args:
        xreal (numpy array):
            shape n planets by 4
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
    return:
        yreal (numpy array):
            numpy array of ellipse quadrant 1 y values, shape n planets by 4
    """
    return np.asarray([np.sqrt(b**2*(1-xreal[:,0]**2/a**2)), np.sqrt(b**2*(1-xreal[:,1]**2/a**2)), np.sqrt(b**2*(1-xreal[:,2]**2/a**2)), np.sqrt(b**2*(1-xreal[:,3]**2/a**2))]).T #yreal

def calculateSeparations(xreal, yreal, mx, my):
    """ Calculate absolute and local minimums and maximums
    Args:
        xreal (numpy array):
            x positions of zero solutions to the quartic solely in quadrant 1 with shape n planets by 4
        xreal (numpy array):
            x positions of zero solutions to the quartic solely in quadrant 1 with shape n planets by 4
        mx (numpy array):
            x position of the star in the derotated, projected ellipse with length n planets
        my (numpy array):
            y position of the star in the derotated, projected ellipse with length n planets
    """ 
    #Local min and Local max must lie in the same quadrant immediately above or below the quadrant the star belongs in
    s_mp = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]+my)**2)]).T
    s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T


    #Using Abs because some terms are negative???
    s_absmin = np.asarray([np.sqrt((np.abs(xreal[:,0])-mx)**2 + (np.abs(yreal[:,0])-my)**2), np.sqrt((np.abs(xreal[:,1])-mx)**2 + (np.abs(yreal[:,1])-my)**2), np.sqrt((np.abs(xreal[:,2])-mx)**2 + (np.abs(yreal[:,2])-my)**2), np.sqrt((np.abs(xreal[:,3])-mx)**2 + (np.abs(yreal[:,3])-my)**2)]).T
    s_absmax = np.asarray([np.sqrt((np.abs(xreal[:,0])+mx)**2 + (np.abs(yreal[:,0])+my)**2), np.sqrt((np.abs(xreal[:,1])+mx)**2 + (np.abs(yreal[:,1])+my)**2), np.sqrt((np.abs(xreal[:,2])+mx)**2 + (np.abs(yreal[:,2])+my)**2), np.sqrt((np.abs(xreal[:,3])+mx)**2 + (np.abs(yreal[:,3])+my)**2)]).T

    return s_mp, s_pm, s_absmin, s_absmax

def sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y):
    """ Calculates Minimum, Maximum, Local Minimum, Local Maximum for Each star
    Args:
        s_absmin (numpy array):
        s_absmax (numpy array):
        s_mp (numpy array):
        xreal (numpy array):
            x points of solutions on ellipse shape n planets by 4
        yreal (numpy array):
            y points of solutions on ellipse shape n planets by 4
        x (numpy array):
            x positions of projected star in derotated ellipse with length n
        y (numpy array):
            y positions of projected star in derotated ellipse with length n
    Returns:
        minSepPoints_x (numpy array):
        minSepPoints_y (numpy array):
        maxSepPoints_x (numpy array):
        maxSepPoints_y (numpy array):
        lminSepPoints_x (numpy array):
        lminSepPoints_y (numpy array):
        lmaxSepPoints_x (numpy array):
        lmaxSepPoints_y (numpy array):
    """
    #### Minimum Separations and x, y of minimum separation
    minSepInd = np.nanargmin(s_absmin,axis=1)
    minSep = s_absmin[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations
    minSep_x = xreal[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations x coord
    minSep_y = yreal[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations y coord
    # minSep_x = xreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations x coord
    # minSep_y = yreal[:,minSepInd][:,0] #Minimum Planet-StarSeparations y coord
    #DELETEminSepMask = np.zeros((len(minSepInd),4), dtype=bool) 
    #DELETEminSepMask[np.arange(len(minSepInd)),minSepInd] = 1 #contains 1's where the minimum separation occurs
    #DELETEminNanMask = np.isnan(s_absmin) #places true where value is nan
    #
    #DELETEcountMinNans = np.sum(minNanMask,axis=0) #number of Nans in each 4
    #DELETEfreqMinNans = np.unique(countMinNans, return_counts=True)
    #DELETEminAndNanMask = minSepMask + minNanMask #Array of T/F of minSep and NanMask
    #DELETEcountMinAndNanMask = np.sum(minAndNanMask,axis=1) #counting number of minSep and NanMask for each star
    #DELETEfreqMinAndNanMask = np.unique(countMinAndNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted

    #### Maximum Separations and x,y of maximum separation
    maxSepInd = np.nanargmax(s_absmax,axis=1)
    maxSep = s_absmax[np.arange(len(maxSepInd)),maxSepInd] #Maximum Planet-StarSeparations
    maxSep_x = xreal[np.arange(len(minSepInd)),maxSepInd] #Maximum Planet-StarSeparations x coord
    maxSep_y = yreal[np.arange(len(minSepInd)),maxSepInd] #Maximum Planet-StarSeparations y coord
    # maxSep_x = xreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations x coord
    # maxSep_y = yreal[:,maxSepInd][:,0] #Maximum Planet-StarSeparations y coord
    #DELETEmaxSepMask = np.zeros((len(maxSepInd),4), dtype=bool) 
    #DELETEmaxSepMask[np.arange(len(maxSepInd)),maxSepInd] = 1 #contains 1's where the maximum separation occurs
    #DELETEmaxNanMask = np.isnan(s_absmax)
    #
    #DELETEcountMaxNans = np.sum(maxNanMask,axis=0) #number of Nans in each 4
    #DELETEfreqMaxNans = np.unique(countMaxNans, return_counts=True)
    #DELETEmaxAndNanMask = maxSepMask + maxNanMask #Array of T/F of maxSep and NanMask
    #DELETEcountMaxAndNanMask = np.sum(maxAndNanMask,axis=1) #counting number of maxSep and NanMask for each star
    #DELETEfreqMaxAndNanMask = np.unique(countMaxAndNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted

    #Sort arrays Minimum
    sminOrderInds = np.argsort(s_absmin, axis=1) #argsort sorts from min to max with last terms being nan
    minSeps = s_absmin[np.arange(len(minSepInd)),sminOrderInds[:,0]]
    minSeps_x = xreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]
    minSeps_y = yreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]

    #Sort arrays Maximum
    smaxOrderInds = np.argsort(-s_absmax, axis=1) #-argsort sorts from max to min with last terms being nan #Note: Last 2 indicies will be Nan
    maxSeps = s_absmax[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]
    maxSeps_x = np.abs(xreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]])
    maxSeps_y = yreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]

    #Masking
    mask = np.zeros((len(maxSepInd),4), dtype=bool)
    #assert ~np.any(sminOrderInds[:,0] == smaxOrderInds[:,0]), 'Exception: A planet has smin == smax'
    print('Number of smaxInd == sminInd: ' + str(np.count_nonzero(sminOrderInds[:,0] == smaxOrderInds[:,0])))
    mask[np.arange(len(minSepInd)),sminOrderInds[:,0]] = 1 #contains 1's where the minimum separation occurs
    mask[np.arange(len(maxSepInd)),smaxOrderInds[:,0]] = 1 #contains 1's where the maximum separation occurs
    assert np.all(np.isnan(s_absmin) == np.isnan(s_absmax)), 'Exception: absmin and absmax have different nan values'
    mask += np.isnan(s_absmin) #Adds nan solutions to mask
    #DELETEcountMinMaxNanMask = np.sum(mask,axis=1) #counting number of stars with min, max, and nan for each star
    #freqMinMaxNanMask = np.unique(countMinMaxNanMask,return_counts=True) #just gives the quantity of 1,2,3 accounted

    #### Account For Stars with Local Minimum and Maximum
    # inds_accnt2 = np.where(countMinMaxNanMask == 2)[0] #planetInds where 2 of the 4 soultions are accounted for
    # inds_accnt4 = np.where(countMinMaxNanMask == 4)[0] #planetInds where all 4 solutions are accounted for
    #For Stars with 2 Inds Accounted For
    s_mptmp = ~mask*s_mp
    s_mptmp[mask] = np.nan

    #Sort arrays Local Minimum
    s_mpOrderInds = np.argsort(s_mptmp, axis=1) #argsort sorts from min to max with last terms being nan
    s_mplminSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    lminSeps_x = xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    lminSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    s_mplmaxSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]
    lmaxSeps_x = np.abs(xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]])
    lmaxSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]

    #Quadrant Star Belongs to
    bool1 = x > 0
    bool2 = y > 0
    #Quadrant 1 if T,T
    #Quadrant 2 if F,T
    #Quadrant 3 if F,F
    #Quadrant 4 if T,F

    #### Min Sep Point (Points on plot of Min Sep)
    minSepPoints_x = minSeps_x*(2*bool1-1)
    minSepPoints_y = minSeps_y*(2*bool2-1)

    #### Max Sep Point (Points on plot of max sep)
    maxSepPoints_x = maxSeps_x*(-2*bool1+1)
    maxSepPoints_y = maxSeps_y*(-2*bool2+1)

    #### Local Min Sep Points
    lminSepPoints_x = lminSeps_x*(2*bool1-1)
    lminSepPoints_y = lminSeps_y*(-2*bool2+1)

    #### Local Max Sep Points
    lmaxSepPoints_x = lmaxSeps_x*(2*bool1-1)
    lmaxSepPoints_y = lmaxSeps_y*(-2*bool2+1)

    return minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps


#### Point to ellipse
#def pointToEllipseDistance(a, b, x, y):
    """ Calculated distance between point (x,y) and
    Ellipse with major axis a and minor axis b,
    centered at origin, and major axis along X-axis
    Args:
        a ():
        b ():
        x ():
        y ():
    """

# #Intermediate Calcs. For Efficiency
# phi = (a**2 - b**2)
# theta = (-a**4 + 2*a**2*b**2 + a**2*x**2 - b**4 + b**2*y**2)
# gamma = (a**4 - 2*a**2*b**2 + b**4)
# beta = a**2*x**2
# alpha = -3*beta/(2*phi**2)
# epsilon = a*x/(2*phi)
# #delta = -2*a*x*(3*beta/(64*phi**2)
# gorg = (alpha + theta/gamma)
# mini = beta/(2*phi**2)
# macro = (mini - theta/(2*gamma))
# tacos = 3*beta/(64*phi**2)
# burritos = 2*a*x*(-2*a*x*(tacos - theta/(16*gamma))/phi + epsilon)/phi
# curry = (-2*a*x*macro/phi + 2*a*x/phi)
# matcha =  gorg*(-beta/gamma + burritos)/6
# regular = 2*(beta/gamma - burritos - gorg**2/12)
# #sqrt np.sqrt
# manta = (gorg**3/216 -matcha + curry**2/16 + np.sqrt((beta/gamma - burritos - gorg**2/12)**3/27 + (-gorg**3/108 + gorg*(-beta/gamma + burritos)/3 - curry**2/8)**2/4))
# cata = 2*(-gorg**3/108 + gorg*(-beta/gamma + burritos)/3 - curry**2/8)**(1/3)
# mellon = np.sqrt(beta/phi**2 - regular/(3*manta**(1/3)) + 2*manta**(1/3) - 2*theta/(3*gamma))
# sampson = (-4*a*x*macro/phi + 4*a*x/phi)/np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))
# ifbool =  -beta/gamma + burritos + gorg**2/12 < 0
# ifboolIndexes = np.nonzero(ifbool)

# #Note: Eq(a,b) represents a < b

# #Lambda Solutions to a*x*sqrt(1 - lambda**2) - b*lambda*y + lambda*sqrt(1 - lambda**2)*(-a**2 + b**2) from point_ellipse.ipynb
# soln0 = epsilon - mellon/2 - np.sqrt(2*beta/phi**2 + (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2 #NOMINALLY
# #IFBOOL IS TRUE USE THE FOLLOWING INSTEAD
# soln0[ifboolIndexes] = epsilon - np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 - np.sqrt(2*beta/phi**2 + sampson + cata - 4*theta/(3*gamma))/2
# #soln0 raw
# # Piecewise((epsilon - np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 - np.sqrt(2*beta/phi**2 + sampson + cata - 4*theta/(3*gamma))/2
# # Eq(-beta/gamma + burritos + gorg**2/12 < 0))
# # (epsilon - mellon/2 - np.sqrt(2*beta/phi**2 + (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# # True))
# soln1 = epsilon - mellon/2 + np.sqrt(2*beta/phi**2 + (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# #IFBOOL IS TRUE USE THE FOLLOWING INSTEAD
# soln1[ifboolIndexes] = epsilon - np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 + np.sqrt(2*beta/phi**2 + sampson + cata - 4*theta/(3*gamma))/2
# #soln1 raw
# # Piecewise((epsilon - np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 + np.sqrt(2*beta/phi**2 + sampson + cata - 4*theta/(3*gamma))/2
# # Eq(-beta/gamma + burritos + gorg**2/12
# # 0))
# # (epsilon - mellon/2 + np.sqrt(2*beta/phi**2 + (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# # True))
# soln2 = epsilon + mellon/2 - np.sqrt(2*beta/phi**2 - (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# #IFBOOL IS TRUE USE THE FOLLOWING INSTEAD
# soln2[ifboolIndexes] = epsilon + np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 - np.sqrt(2*beta/phi**2 - sampson + cata - 4*theta/(3*gamma))/2
# #soln2 raw
# # Piecewise((epsilon + np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 - np.sqrt(2*beta/phi**2 - sampson + cata - 4*theta/(3*gamma))/2
# # Eq(-beta/gamma + burritos + gorg**2/12
# # 0))
# # (epsilon + mellon/2 - np.sqrt(2*beta/phi**2 - (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# # True))
# soln3 = epsilon + mellon/2 + np.sqrt(2*beta/phi**2 - (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# #IFBOOL IS TRUE USE THE FOLLOWING INSTEAD
# soln3[ifboolIndexes] = epsilon + np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 + np.sqrt(2*beta/phi**2 - sampson + cata - 4*theta/(3*gamma))/2
# #soln3 raw
# # Piecewise((epsilon + np.sqrt(beta/phi**2 - cata - 2*theta/(3*gamma))/2 + np.sqrt(2*beta/phi**2 - sampson + cata - 4*theta/(3*gamma))/2
# # Eq(-beta/gamma + burritos + gorg**2/12
# # 0))
# # (epsilon + mellon/2 + np.sqrt(2*beta/phi**2 - (-4*a*x*macro/phi + 4*a*x/phi)/mellon + regular/(3*manta**(1/3)) - 2*manta**(1/3) - 4*theta/(3*gamma))/2
# # True))

# #### Check soln are within valid range
# is_soln0Valid = np.abs(soln0) <= 1
# is_soln1Valid = np.abs(soln1) <= 1
# is_soln2Valid = np.abs(soln2) <= 1
# is_soln3Valid = np.abs(soln3) <= 1
# #I am expecting 0% 0's or 50% 0's or 100% 0's
# print('Frac soln0 in Valid Range: ' + str(np.count_nonzero(is_soln0Valid)/len(soln0)))
# print('Frac soln1 in Valid Range: ' + str(np.count_nonzero(is_soln1Valid)/len(soln1)))
# print('Frac soln2 in Valid Range: ' + str(np.count_nonzero(is_soln2Valid)/len(soln2)))
# print('Frac soln3 in Valid Range: ' + str(np.count_nonzero(is_soln3Valid)/len(soln3)))
# ####

# #### Verify soln are indeed solutions
# #Calculate
# val0 = a*x*np.sqrt(1 - soln0**2) - b*soln0*y + soln0*np.sqrt(1 - soln0**2)*(-a**2 + b**2)
# val1 = a*x*np.sqrt(1 - soln1**2) - b*soln1*y + soln1*np.sqrt(1 - soln1**2)*(-a**2 + b**2)
# val2 = a*x*np.sqrt(1 - soln2**2) - b*soln2*y + soln2*np.sqrt(1 - soln2**2)*(-a**2 + b**2)
# val3 = a*x*np.sqrt(1 - soln3**2) - b*soln3*y + soln3*np.sqrt(1 - soln3**2)*(-a**2 + b**2)
# #Check if zeros
# isVal0soln = np.abs(val0) < 1e-8
# isVal1soln = np.abs(val1) < 1e-8
# isVal2soln = np.abs(val2) < 1e-8
# isVal3soln = np.abs(val3) < 1e-8
# print('Frac soln0 is verified Solution: ' + str(np.count_nonzero(isVal0soln)/len(val0)))
# print('Frac soln1 is verified Solution: ' + str(np.count_nonzero(isVal1soln)/len(val1)))
# print('Frac soln2 is verified Solution: ' + str(np.count_nonzero(isVal2soln)/len(val2)))
# print('Frac soln3 is verified Solution: ' + str(np.count_nonzero(isVal3soln)/len(val3)))
# ####


# #convert from lambda to w
# #Note: Solns are either w or -w
# soln0w = np.arccos(soln0)
# soln1w = np.arccos(soln1)
# soln2w = np.arccos(soln2)
# soln3w = np.arccos(soln3)
# #These are the eccentric anomalies of the projected ellipse potentially producing the local minimum and local maximum

# #Element-wise addition using validation results. (could be faster if using valid range results)
# numSols = np.add(np.add(np.add(isVal0soln,isVal1soln),isVal2soln),isVal3soln) #i think this is how it works
# is_validNumSols = np.all(numSols > 1)
# print('Number of Solutions is valid: ' + str(is_validNumSols))

# #How to easly get min and max sols? 2nd derivative and value at min/max comparisons.

# #return numSols, sminSolind, smaxSolind, soln0w, soln1w, soln2w, soln3w






# #### MATLAB SOLUTIONS TO QUARTIC
# #^ from ** #sed -i 's/\^/**/g' lambdaVar2txt #otherwise this operation is too slow
# alpha = (a**4 + b**4 - 2*a**2*b**2)
# beta = (a**16 + b**16 - 8*a**2*b**14 + 28*a**4*b**12 - 56*a**6*b**10 + 70*a**8*b**8 - 56*a**10*b**6 + 28*a**12*b**4 - 8*a**14*b**2)
# taco = (a**8 + b**8 - 4*a**2*b**6 + 6*a**4*b**4 - 4*a**6*b**2)
# mercy = (a**12 + b**12 - 6*a**2*b**10 + 15*a**4*b**8 - 20*a**6*b**6 + 15*a**8*b**4 - 6*a**10*b**2)
# dengue = (a**4*b**4*x**2*y**2)
# charlie = (9*a**8*b**4*x**4)
# mark = (3*a**2*b**4*x**2)
# mfart = (a**4*b**6*x**2)
# dart = (3*a**4*b**8*x**4)
# mango = (a**2*b**6*x**2*y**2)
# sangi = (2*a**4*b**2*x**2)
# puts = (3*a**10*b**2*x**4)
# gorgon = ((a**7*x)/taco + (a**9*x**3)/mercy - (2*a**3*x)/alpha - (a**5*x**3)/taco + (2*a*b**2*x)/alpha + (3*a**3*b**4*x)/taco - (3*a**5*b**2*x)/taco - (a**3*b**6*x**3)/mercy + (3*a**5*b**4*x**3)/mercy - (3*a**7*b**2*x**3)/mercy + (a**3*b**2*x**3)/taco - (a*b**6*x)/taco - (a**3*b**2*x*y**2)/taco + (a*b**4*x*y**2)/taco)
# pansy = (a**4/alpha + b**4/alpha - (2*a**2*b**2)/alpha - (a**2*x**2)/alpha - (b**2*y**2)/alpha + (3*a**6*x**2)/(2*taco) + mark/(2*taco) - (3*a**4*b**2*x**2)/taco)
# redline = ((a**2*x**2)/alpha + (3*a**12*x**4)/(16*beta) - (a**8*x**4)/(4*mercy) + (a**10*x**2)/(4*mercy) - (a**6*x**2)/taco + dart/(16*beta) - (3*a**6*b**6*x**4)/(4*beta) + charlie/(8*beta) - puts/(4*beta) + (a**2*b**8*x**2)/(4*mercy) - (a**4*b**4*x**4)/(4*mercy) - mfart/mercy + (a**6*b**2*x**4)/(2*mercy) + (3*a**6*b**4*x**2)/(2*mercy) - (a**8*b**2*x**2)/mercy - (a**2*b**4*x**2)/taco + sangi/taco - mango/(4*mercy) + dengue/(2*mercy) - (a**6*b**2*x**2*y**2)/(4*mercy))
# osaka = (256*redline**3 - 4*pansy**3*gorgon**2 + 27*gorgon**4 + 16*pansy**4*redline + 128*pansy**2*redline**2 - 144*pansy*gorgon**2*redline)
# cream = (6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy + pansy**2 + 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3) - (12*a**2*x**2)/alpha - (9*a**12*x**4)/(4*beta) + (3*a**8*x**4)/mercy - (3*a**10*x**2)/mercy + (12*a**6*x**2)/taco - (9*a**4*b**8*x**4)/(4*beta) + (9*a**6*b**6*x**4)/beta - (27*a**8*b**4*x**4)/(2*beta) + (9*a**10*b**2*x**4)/beta - (3*a**2*b**8*x**2)/mercy + (3*a**4*b**4*x**4)/mercy + (12*a**4*b**6*x**2)/mercy - (6*a**6*b**2*x**4)/mercy - (18*a**6*b**4*x**2)/mercy + (12*a**8*b**2*x**2)/mercy + (12*a**2*b**4*x**2)/taco - (24*a**4*b**2*x**2)/taco + (3*a**2*b**6*x**2*y**2)/mercy - (6*a**4*b**4*x**2*y**2)/mercy + (3*a**6*b**2*x**2*y**2)/mercy)

# #ROUND1
# # soln0 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# # soln1 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# # soln2 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# # soln3 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))

# #ROUND2
# #out2.lambda
 
# #ans =
 
# soln0 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))

# soln1 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))

# soln2 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))

# soln3 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
 
# #out2.conditions
 
# #ans =
 
# soln0cond0 = a**4 + b**4 != 2*a**2*b**2
# soln0cond1 = -((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha
# soln0cond2 = 0 <= ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha - ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6))

# #soln1cond0 = a**4 + b**4 != 2*a**2*b**2 #identical to soln0cond0
# soln1cond1 = 0 <= (a**3*x + 2*a**4 + 2*b**4 - 4*a**2*b**2 + (2*((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)*alpha)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) - a*b**2*x)/alpha
# soln1cond2 = ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha

# #soln2cond0 = a**4 + b**4 != 2*a**2*b**2 #identical to soln0cond0
# soln2cond1 = 0 <= ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha - ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)
# soln2cond2 = -((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha

# #soln3cond0 = a**4 + b**4 != 2*a**2*b**2 #identical to soln0cond0
# soln3cond1 = ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha
# soln3cond2 = 0 <= ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha
 
# #DELETEdiary off



# #MATLAB ROUND3
# #### REPLACEMENT MACHINERY
# # #^ from ** #sed -i 's/\^/**/g' lambdaVar3.txt #otherwise this operation is too slow
# sed -i 's/\^/**/g' lambdaVar4.txt
# sed -i 's/(a\*\*4\s+\sb\*\*4\s-\s2\*a\*\*2\*b\*\*2)/alpha/g' lambdaVar4.txt
# sed -i 's/(a\*\*16\s+\sb\*\*16\s-\s8\*a\*\*2\*b\*\*14\s+\s28\*a\*\*4\*b\*\*12\s-\s56\*a\*\*6\*b\*\*10\s+\s70\*a\*\*8\*b\*\*8\s-\s56\*a\*\*10\*b\*\*6\s+\s28\*a\*\*12\*b\*\*4\s-\s8\*a\*\*14\*b\*\*2)/beta/g' lambdaVar4.txt
# sed -i 's/(a\*\*8\s+\sb\*\*8\s-\s4\*a\*\*2\*b\*\*6\s+\s6\*a\*\*4\*b\*\*4\s-\s4\*a\*\*6\*b\*\*2)/taco/g' lambdaVar3.txt
# sed -i 's/(a\*\*12\s+\sb\*\*12\s-\s6\*a\*\*2\*b\*\*10\s+\s15\*a\*\*4\*b\*\*8\s-\s20\*a\*\*6\*b\*\*6\s+\s15\*a\*\*8\*b\*\*4\s-\s6\*a\*\*10\*b\*\*2)/mercy/g' lambdaVar4.txt
# sed -i 's/(a\*\*4\*b\*\*4\*x\*\*2\*y\*\*2)/dengue/g' lambdaVar4.txt
# sed -i 's/(9\*a\*\*8\*b\*\*4\*x\*\*4)/charlie/g' lambdaVar4.txt
# sed -i 's/(3\*a\*\*2\*b\*\*4\*x\*\*2)/mark/g' lambdaVar4.txt
# sed -i 's/(a\*\*4\*b\*\*6\*x\*\*2)/mfart/g' lambdaVar4.txt
# sed -i 's/(3\*a\*\*4\*b\*\*8\*x\*\*4)/dart/g' lambdaVar4.txt
# sed -i 's/(a\*\*2\*b\*\*6\*x\*\*2\*y\*\*2)/mango/g' lambdaVar4.txt
# sed -i 's/(2\*a\*\*4\*b\*\*2\*x\*\*2)/sangi/g' lambdaVar4.txt
# sed -i 's/(3\*a\*\*10\*b\*\*2\*x\*\*4)/puts/g' lambdaVar4.txt


# sed -i 's/((a\*\*7\*x)\/taco\s+\s(a\*\*9\*x\*\*3)\/mercy\s-\s(2\*a\*\*3\*x)\/alpha\s-\s(a\*\*5\*x\*\*3)\/taco\s+\s(2\*a\*b\*\*2\*x)\/alpha\s+\s(3\*a\*\*3\*b\*\*4\*x)\/taco\s-\s(3\*a\*\*5\*b\*\*2\*x)\/taco\s-\s(a\*\*3\*b\*\*6\*x\*\*3)\/mercy\s+\s(3\*a\*\*5\*b\*\*4\*x\*\*3)\/mercy\s-\s(3\*a\*\*7\*b\*\*2\*x\*\*3)\/mercy\s+\s(a\*\*3\*b\*\*2\*x\*\*3)\/taco\s-\s(a\*b\*\*6\*x)\/taco\s-\s(a\*\*3\*b\*\*2\*x\*y\*\*2)\/taco\s+\s(a\*b\*\*4\*x\*y\*\*2)\/taco)/gorgon/g' lambdaVar4.txt
# sed -i 's/(a\*\*4\/alpha\s+\sb\*\*4\/alpha\s-\s(2\*a\*\*2\*b\*\*2)\/alpha\s-\s(a\*\*2\*x\*\*2)\/alpha\s-\s(b\*\*2\*y\*\*2)\/alpha\s+\s(3\*a\*\*6\*x\*\*2)\/(2\*taco)\s+\smark\/(2\*taco)\s-\s(3\*a\*\*4\*b\*\*2\*x\*\*2)\/taco)/pansy/g' lambdaVar4.txt
# sed -i 's/((a\*\*2\*x\*\*2)\/alpha\s+\s(3\*a\*\*12\*x\*\*4)\/(16\*beta)\s-\s(a\*\*8\*x\*\*4)\/(4\*mercy)\s+\s(a\*\*10\*x\*\*2)\/(4\*mercy)\s-\s(a\*\*6\*x\*\*2)\/taco\s+\sdart\/(16\*beta)\s-\s(3\*a\*\*6\*b\*\*6\*x\*\*4)\/(4\*beta)\s+\scharlie\/(8\*beta)\s-\sputs\/(4\*beta)\s+\s(a\*\*2\*b\*\*8\*x\*\*2)\/(4\*mercy)\s-\s(a\*\*4\*b\*\*4\*x\*\*4)\/(4\*mercy)\s-\smfart\/mercy\s+\s(a\*\*6\*b\*\*2\*x\*\*4)\/(2\*mercy)\s+\s(3\*a\*\*6\*b\*\*4\*x\*\*2)\/(2\*mercy)\s-\s(a\*\*8\*b\*\*2\*x\*\*2)\/mercy\s-\s(a\*\*2\*b\*\*4\*x\*\*2)\/taco\s+\ssangi\/taco\s-\smango\/(4\*mercy)\s+\sdengue\/(2\*mercy)\s-\s(a\*\*6\*b\*\*2\*x\*\*2\*y\*\*2)\/(4\*mercy))/redline/g' lambdaVar4.txt
# sed -i 's/(256\*redline\*\*3\s-\s4\*pansy\*\*3\*gorgon\*\*2\s+\s27\*gorgon\*\*4\s+\s16\*pansy\*\*4\*redline\s+\s128\*pansy\*\*2\*redline\*\*2\s-\s144\*pansy\*gorgon\*\*2\*redline)/osaka/g' lambdaVar4.txt
# sed -i 's/(6\*((3\*\*(1\/2)\*osaka\*\*(1\/2))\/18\s-\s(4\*pansy\*redline)\/3\s+\sgorgon\*\*2\/2\s-\spansy\*\*3\/27)\*\*(1\/3)\*pansy\s+\spansy\*\*2\s+\s9\*((3\*\*(1\/2)\*osaka\*\*(1\/2))\/18\s-\s(4\*pansy\*redline)\/3\s+\sgorgon\*\*2\/2\s-\spansy\*\*3\/27)\*\*(2\/3)\s-\s(12\*a\*\*2\*x\*\*2)\/alpha\s-\s(9\*a\*\*12\*x\*\*4)\/(4\*beta)\s+\s(3\*a\*\*8\*x\*\*4)\/mercy\s-\s(3\*a\*\*10\*x\*\*2)\/mercy\s+\s(12\*a\*\*6\*x\*\*2)\/taco\s-\s(9\*a\*\*4\*b\*\*8\*x\*\*4)\/(4\*beta)\s+\s(9\*a\*\*6\*b\*\*6\*x\*\*4)\/beta\s-\s(27\*a\*\*8\*b\*\*4\*x\*\*4)\/(2\*beta)\s+\s(9\*a\*\*10\*b\*\*2\*x\*\*4)\/beta\s-\s(3\*a\*\*2\*b\*\*8\*x\*\*2)\/mercy\s+\s(3\*a\*\*4\*b\*\*4\*x\*\*4)\/mercy\s+\s(12\*a\*\*4\*b\*\*6\*x\*\*2)\/mercy\s-\s(6\*a\*\*6\*b\*\*2\*x\*\*4)\/mercy\s-\s(18\*a\*\*6\*b\*\*4\*x\*\*2)\/mercy\s+\s(12\*a\*\*8\*b\*\*2\*x\*\*2)\/mercy\s+\s(12\*a\*\*2\*b\*\*4\*x\*\*2)\/taco\s-\s(24\*a\*\*4\*b\*\*2\*x\*\*2)\/taco\s+\s(3\*a\*\*2\*b\*\*6\*x\*\*2\*y\*\*2)\/mercy\s-\s(6\*a\*\*4\*b\*\*4\*x\*\*2\*y\*\*2)\/mercy\s+\s(3\*a\*\*6\*b\*\*2\*x\*\*2\*y\*\*2)\/mercy)/cream/g' lambdaVar4.txt
# ####


# soln0 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# soln1 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) - cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# soln2 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) - (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
# soln3 = (2*a**3*x - 2*a*b**2*x)/(4*alpha) + cream**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + (12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)*cream**(1/4))
 

# soln0cond0 = a**4 + b**4 != 2*a**2*b**2
# soln0cond1 = -((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha 
# soln0cond2 = a**2*y**2 + b**2*x**2 < a**2*b**2
# soln0cond3 = 0 <= ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha - ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6))

# soln1cond0 = a**4 + b**4 != 2*a**2*b**2
# soln1cond1 = 0 <= (a**3*x + 2*a**4 + 2*b**4 - 4*a**2*b**2 + (2*((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)*alpha)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) - a*b**2*x)/alpha
# soln1cond2 = a**2*y**2 + b**2*x**2 < a**2*b**2
# soln1cond3 = ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) - 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha

# soln2cond0 = a**4 + b**4 != 2*a**2*b**2
# soln2cond1 = 0 <= ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha - ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)
# soln2cond2 = a**2*y**2 + b**2*x**2 < a**2*b**2
# soln2cond3 = -((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/(6*cream**(1/4)) - cream**(1/2)/6)/((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha

# soln3cond0 = a**4 + b**4 != 2*a**2*b**2
# soln3cond1 = ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) <= (a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2)/alpha
# soln3cond2 = a**2*y**2 + b**2*x**2 < a**2*b**2
# soln3cond3 = 0 <= ((12*redline*cream**(1/2) - pansy**2*cream**(1/2) - 9*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(2/3)*cream**(1/2) + 3*6**(1/2)*(3*3**(1/2)*osaka**(1/2) - 72*pansy*redline + 27*gorgon**2 - 2*pansy**3)**(1/2)*gorgon + 12*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/3)*pansy*cream**(1/2))**(1/2)/cream**(1/4) + cream**(1/2))/(6*((3**(1/2)*osaka**(1/2))/18 - (4*pansy*redline)/3 + gorgon**2/2 - pansy**3/27)**(1/6)) + ((a**3*x)/2 + a**4 + b**4 - 2*a**2*b**2 - (a*b**2*x)/2)/alpha
 
# def quarticRoots_wiki(a,b,c,d,e):
#     #### QUARTIC ROOTS
#     #From Graphical Discussion of the Roots of a Quartic Equation"
#     #Solutions to 0 = a*x**4 + b*x**3 + c*X**2 + d*X + e

#     #discriminant
#     delta = 256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 144*a**2*c*d**2*e - 27*a**2*d**4\
#             + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 16*a*c**4*e\
#             - 4*a*c**3*d**2 - 27*b**4*e**2 + 18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2 #verified against wikipedia multiple times
#     P = 8*a*c - 3*b**2
#     R = b**3 + 8*d*a**2 - 4*a*b*c
#     delta_0 = c**2 - 3*b*d + 12*a*e
#     D = 64*a**3*e - 16*a**2*c**2 + 16*a*b**2*c - 16*a**2*b*d - 3*b**4 #is 0 if the quartic has 2 double roots 
#     """
#     if delta < 0: 2 distinct real roots and 2 complex conjugate non-real roots
#     if delta > 0: either 4 real roots or 4 complex roots
#         if P < 0 and D < 0: 4 real roots
#         if P > 0 or D > 0: 2 pairs of complex conjugate roots
#     if delta = 0: the polynomial has a multiple root with the following different cases
#         if P < 0 and D < 0 and delta_0 != 0: real double root and 2 simple real roots
#         if D > 0 or (P > 0 and (D != 0 or R != 0)): real double root and 2 complex conjugate roots
#         if delta_0 = 0 and D != 0: real triple root and a real simple root
#         if D = 0:
#             if P < 0: 2 real double roots
#             if P > 0 and R = 0: 2 complex conjugate double roots
#             if delta_0 = 0: all 4 roots are -b/4a
#     """
#     p = (8*a*c - 3*b**2)/(8*a**2)
#     q = (b**3 - 4*a*b*c + 8*a**2*d)/(8*a**3)
#     delta_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
#     Q = ((delta_1+np.sqrt(delta_1**2 - 4*delta_0**3))/2)**(1/3)
#     S = 0.5*np.sqrt(-2/3*p + (Q+delta_0/Q)/(3*a))
    
#     #2 real 2 complex
#     inds_2real2complex = np.where(delta < 0)[0]


#     #USE IF DELTA > 0, P < 0, D < 0
#     inds_allreal = np.where((delta > 0)*(P < 0)*(D < 0))[0]
#     phi = np.arccos(delta_1/(2*np.sqrt(delta_0**3)))
#     S[inds_allreal] = 0.5*np.sqrt(-2/3*p[inds_allreal]+2/3/a[inds_allreal]*np.sqrt(delta_0[inds_allreal])*np.cos(phi[inds_allreal]/3))
#     print('QUARTIC Num inds all real: ' + str(len(inds_allreal)))

#     x1 = -b/(4*a) - S + 0.5*np.sqrt(-4*S**2 - 2*p + q/S)
#     x2 = -b/(4*a) - S - 0.5*np.sqrt(-4*S**2 - 2*p + q/S)
#     x3 = -b/(4*a) + S + 0.5*np.sqrt(-4*S**2 - 2*p - q/S)
#     x4 = -b/(4*a) + S - 0.5*np.sqrt(-4*S**2 - 2*p - q/S)
#     #note delta_1**2 - 4*delta_0**3 = -27*delta
#     return x1, x2, x3, x4, delta, P, S, Q, delta_0, delta_1

# def quarticRoots_ia(A,B,C,D,E):
#     #http://web.cs.iastate.edu/~cs577/handouts/polyroots.pdf
#     #x**4 + p*x**3 + q*x**2 + r*x + s
#     #z1?????

#     R = np.sqrt(p**2/4 - q + z1)
#     #if R != 0
#     D = np.sqrt(3/4*p**2 - R**2 - 2*q + 1/4*(4*p*q - 8*r - p**3)/R)
#     E = np.sqrt(3/4*p**2 - R**2 - 2*q - 1/4*(4*p*q - 8*r - p**3)/R)
#     #if R = 0
#     Rinds = np.where(R == 0)[0]
#     D[Rinds] = np.sqrt(3/4*p[Rinds]**2 - 2*q[Rinds] + 2*np.sqrt(z1[Rinds]**2 - 4*s[Rinds]))
#     E[Rinds] = np.sqrt(3/4*p**2 - 2*q - 2*np.sqrt(z1**2 - 4*s))
    

#     x1 = -p/4 + 0.5*(R + D)
#     x2 = -p/4 + 0.5*(R - D)
#     x3 = -p/4 - 0.5*(R - E)
#     x4 = -p/4 - 0.5*(R + E)
#     return x1, x2, x3, x4

# def cubicRoots(a2, a1, a0): #p,q,r):
#     #solutions to z**3 + a2*x**2 + a1*z + a0 = 0
#     #DELETE?solutions to x**3 + p*x**2 + q*x + r = 0
#     #FROM Jupyter Notebook z1 = -a2/3 - (-3*a1 + a2**2)/(3*(27*a0/2 - 9*a1*a2/2 + a2**3 + np.sqrt(-4*(-3*a1 + a2**2)**3 + (27*a0 - 9*a1*a2 + 2*a2**3)**2)/2)**(1/3)) - (27*a0/2 - 9*a1*a2/2 + a2**3 + np.sqrt(-4*(-3*a1 + a2**2)**3 + (27*a0 - 9*a1*a2 + 2*a2**3)**2)/2)**(1/3)/3
    
#     # p = (3*a1-a2**2)/3
#     # q = (9*a1*a2 - 27*a0 - 2*a2**3)/27
#     # Q = p/3
#     # R = q/2
#     # D = Q**3 + R**2

#     #if D > 0: 1 root is real 2 are complex conjugates
#     #if D = 0: all roots real and at least 2 are equal
#     #if D < 0: all roots are real and unequal
#         # theta = np.arccos(R/np.sqrt(-Q**3))
#         # z1 = 2*np.sqrt(-Q)*np.cos(theta/3)-a2/3
#         # z2 = 2*np.sqrt(-Q)*np.cos((theta+2*np.pi)/3)-a2/3
#         # z3 = 2*np.sqrt(-Q)*np.cos((theta+4*np.pi)/3)-a2/3

#     Q = (3*a1 - a2**2)/9
#     R = (9*a2*a1 - 27*a0 - 2*a2**3)/54
#     D = Q**3 + R**2
#     S = (R + np.sqrt(D))**(1/3)
#     T = (R - np.sqrt(D))**(1/3)

#     z1 = -a2/3 + S + T
#     #NONREALx0 = -0.5*B + 0.5*np.sqrt(B**2-4*(B**2+3*Q))
#     #NONREALx1 = -0.5*B - 0.5*np.sqrt(B**2-4*(B**2+3*Q))
#     #x0 = -p/3 - (p**2 - 3*q)/(3*(p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)) - (p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)/3
#     #NONREAL x1 = -p/3 - (p**2 - 3*q)/(3*(-1/2 - np.sqrt(3)*j/2)*(p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)) - (-1/2 - np.sqrt(3)*j/2)*(p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)/3
#     #NONREAL x2 = -p/3 - (p**2 - 3*q)/(3*(-1/2 + np.sqrt(3)*j/2)*(p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)) - (-1/2 + np.sqrt(3)*j/2)*(p**3 - 9*p*q/2 + 27*r/2 + np.sqrt(-4*(p**2 - 3*q)**3 + (2*p**3 - 9*p*q + 27*r)**2)/2)**(1/3)/3
#     return z1 #, x1, x2


# def quarticRoots_wolfram(A,B,C,D,E):
#     #https://mathworld.wolfram.com/QuarticEquation.html
#     #solutions to z**4 + a3*z**3 + a2*z**2 + z1*z + z0 = 0
#     a3 = B/A
#     a2 = C/A
#     a1 = D/A
#     a0 = E/A

#     #???
#     # #Standard Form x**4 + p*x**2 + q*x + r = 0
#     # p = a2 - 3/8*a3**2
#     # q = a1 - a2*a3/2 + a3**3/8
#     # r = a0 - a1*a3/4 + a2*a3**2/16 - 3*a3**4/256

#     #Eq 34 from https://mathworld.wolfram.com/QuarticEquation.html
#     p = -a2
#     q = a1*a3 - 4*a0
#     r = 4*a2*a0 - a1**2 - a3**2*a0
#     y1 = cubicRoots(p,q,r) #only y0 is real

#     #Take the real root, y0, y1, y2 above and subs into y1
#     #trying y1 for now (assume all real)
#     # y1nanInds = np.where(np.isnan(y1))[0]
#     # y1[y1nanInds] = y0[y1nanInds]
#     # y1nanInds = np.where(np.isnan(y1))[0]
#     # y1[y1nanInds] = y2[y1nanInds]
#     # assert ~np.isnan(y1), 'Not all are non-nan'

#     R = np.sqrt(a3**2/4 - a2 + y1)
#     #if R != 0
#     D = np.sqrt(3/4*a3**2 - R**2 - 2*a2 + 1/4*(4*a3*a2 - 8*a1 - a3**3)/R)
#     E = np.sqrt(3/4*a3**2 - R**2 - 2*a2 - 1/4*(4*a3*a2 - 8*a1 - a3**3)/R)
#     #if R = 0
#     Rinds = np.where(R == 0)[0]
#     if len(Rinds) != 0:
#         D[Rinds] = np.sqrt(3/4*a3[Rinds]**2 - 2*a2[Rinds] + 2*np.sqrt(y1[Rinds]**2 - 4*a0[Rinds]))
#         E[Rinds] = np.sqrt(3/4*a3**2 - 2*a2 - 2*np.sqrt(y1**2 - 4*a0))
#     print('Number of Nan in R: ' + str(np.count_nonzero(np.isnan(R))))
#     print('Number of Nan in D: ' + str(np.count_nonzero(np.isnan(D))))
#     print('Number of Nan in E: ' + str(np.count_nonzero(np.isnan(E))))

#     z1 = -a3/4 + R/2 + D/2
#     z2 = -a3/4 + R/2 - D/2
#     z3 = -a3/4 - R/2 + E/2
#     z4 = -a3/4 - R/2 - E/2

#     return z1, z2, z3, z4

# def quarticRoots_stackExchange(a,b,c,d,e):
#     #https://math.stackexchange.com/questions/785/is-there-a-general-formula-for-solving-4th-degree-equations-quartic

#     p1 = 2*c**3 - 9*b*c*d + 27*a*d**2 + 27*b**2*e - 72*a*c*e
#     p2 = p1 + np.sqrt(-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)
#     p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*(p2/2)**(1/3)) + (p2/2)**(1/3)/(3*a)
#     p4 = np.sqrt(b**2/(4*a**2) - 2*c/(3*a) + p3)
#     p5 = b**2/(2*a**2) - 4*c/(3*a) - p3
#     p6 = (-b**3/a**3 + a*b*c/a**2 - 8*d/a)/(4*p4)

#     x0 = -b/(4*a) - p4/2 - np.sqrt(p5 + p6)/2
#     x1 = -b/(4*a) - p4/2 + np.sqrt(p5 + p6)/2
#     x2 = -b/(4*a) + p4/2 - np.sqrt(p5 + p6)/2
#     x3 = -b/(4*a) + p4/2 + np.sqrt(p5 + p6)/2
#     return x0, x1, x2, x3

# # NEW MATLAB REPLACEMENT lamdaVar4.txt
# # sed -i 's/\^/**/g' lambdaVar4.txt
# # sed -i 's/(a\*\*4\s+\sb\*\*4\s-\s2\*a\*\*2\*b\*\*2)/alpha/g' lambdaVar4.txt
# # sed -i 's/(a\*\*16\s+\sb\*\*16\s-\s8\*a\*\*2\*b\*\*14\s+\s28\*a\*\*4\*b\*\*12\s-\s56\*a\*\*6\*b\*\*10\s+\s70\*a\*\*8\*b\*\*8\s-\s56\*a\*\*10\*b\*\*6\s+\s28\*a\*\*12\*b\*\*4\s-\s8\*a\*\*14\*b\*\*2)/beta/g' lambdaVar4.txt
# # #sed -i 's/(a\*\*8\s+\sb\*\*8\s-\s4\*a\*\*2\*b\*\*6\s+\s6\*a\*\*4\*b\*\*4\s-\s4\*a\*\*6\*b\*\*2)/taco/g' lambdaVar3.txt
# # sed -i 's/(a\*\*12\s+\sb\*\*12\s-\s6\*a\*\*2\*b\*\*10\s+\s15\*a\*\*4\*b\*\*8\s-\s20\*a\*\*6\*b\*\*6\s+\s15\*a\*\*8\*b\*\*4\s-\s6\*a\*\*10\*b\*\*2)/mercy/g' lambdaVar4.txt

# #sed -i 's/(a\*\*8\s+\sb\*\*8\s-\s4\*a\*\*2\*b\*\*6\s+\s6\*a\*\*4\*b\*\*4\s-\s4\*a\*\*6\*b\*\*2)/hull/g' lambdaVar4.txt
# #hull = (a**8 + b**8 - 4*a**2*b**6 + 6*a**4*b**4 - 4*a**6*b**2)
# # sed -i 's/(2\*a\*\*4\*b\*\*2)/ham/g' lambdaVar4.txt
# # ham = (2*a**4*b**2)
# # sed -i 's/(a\*\*6\*b\*\*8\*x\*\*2)/lolo/g' lambdaVar4.txt
# # lolo = (a**6*b**8*x**2)
# # sed -i 's/(a\*\*8\*b\*\*4\*x\*\*4)/papa/g' lambdaVar4.txt
# # papa = (a**8*b**4*x**4)
# # sed -i 's/(3\*a\*\*10\*b\*\*2\*x\*\*3)/jop/g' lambdaVar4.txt
# # jop = (3*a**10*b**2*x**3)
# # sed -i 's/(2\*a\*\*4\*x\s-\s2\*a\*\*2\*b\*\*2\*x)/gob/g' lambdaVar4.txt
# # gob = (2*a**4*x - 2*a**2*b**2*x)
# # sed -i 's/(3\*a\*\*4\*b\*\*4\*x\*\*2)/bing/g' lambdaVar4.txt
# # bing = (3*a**4*b**4*x**2)
# # sed -i 's/(a\*\*2\*b\*\*2\*y\*\*2)/sing/g' lambdaVar4.txt
# # sing = (a**2*b**2*y**2)
# # sed -i 's/(3\*a\*\*6\*b\*\*2\*x\*\*2)/goya/g' lambdaVar4.txt
# # goya = (3*a**6*b**2*x**2)
# # sed -i 's/(3\*a\*\*8\*x\*\*2)/meh/g' lambdaVar4.txt
# # meh = (3*a**8*x**2)
# # sed -i 's/(3\*a\*\*6\*b\*\*4\*x)/gem/g' lambdaVar4.txt
# # gem = (3*a**6*b**4*x)
# # sed -i 's/(a\*\*6\*b\*\*6\*x\*\*3)/dame/g' lambdaVar4.txt
# # dame = (a**6*b**6*x**3)
# # sed -i 's/(a\*\*6\/alpha\s+\s(a\*\*2\*b\*\*4)\/alpha\s-\sham\/alpha\s-\s(a\*\*4\*x\*\*2)\/alpha\s+\smeh\/(2\*hull)\s+\sbing\/(2\*hull)\s-\sgoya\/hull\s+\ssing\/alpha)/eleph/g' lambdaVar4.txt
# # eleph = (a**6/alpha + (a**2*b**4)/alpha - ham/alpha - (a**4*x**2)/alpha + meh/(2*hull) + bing/(2*hull) - goya/hull + sing/alpha)
# # sed -i 's/((a\*\*10\*x)\/hull\s+\s(a\*\*12\*x\*\*3)\/mercy\s-\s(2\*a\*\*6\*x)\/alpha\s-\s(a\*\*8\*x\*\*3)\/hull\s-\s(a\*\*4\*b\*\*6\*x)\/hull\s+\sgem\/hull\s-\s(3\*a\*\*8\*b\*\*2\*x)\/hull\s-\sdame\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*3)\/mercy\s-\sjop\/mercy\s+\s(2\*a\*\*4\*b\*\*2\*x)\/alpha\s+\s(a\*\*6\*b\*\*2\*x\*\*3)\/hull\s-\s(a\*\*4\*b\*\*4\*x\*y\*\*2)\/hull\s+\s(a\*\*6\*b\*\*2\*x\*y\*\*2)\/hull)/kale/g' lambdaVar4.txt
# # kale = ((a**10*x)/hull + (a**12*x**3)/mercy - (2*a**6*x)/alpha - (a**8*x**3)/hull - (a**4*b**6*x)/hull + gem/hull - (3*a**8*b**2*x)/hull - dame/mercy + (3*a**8*b**4*x**3)/mercy - jop/mercy + (2*a**4*b**2*x)/alpha + (a**6*b**2*x**3)/hull - (a**4*b**4*x*y**2)/hull + (a**6*b**2*x*y**2)/hull)
# # sed -i 's/(3\*a\*\*10\*b\*\*6\*x\*\*4)/tempest/g' lambdaVar4.txt
# # tempest = (3*a**10*b**6*x**4)
# # sed -i 's/(a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)/cleric/g' lambdaVar4.txt
# # cleric = (a**6*b**6*x**2*y**2)
# # sed -i 's/(3\*a\*\*14\*b\*\*2\*x\*\*4)\/(4\*beta)\s+\slolo\/(4\*mercy)\s-\spapa\/(4\*mercy)\s-\s(a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s+\s(a\*\*10\*b\*\*2\*x\*\*4)\/(2\*mercy)\s+\s(3\*a\*\*10\*b\*\*4\*x\*\*2)\/(2\*mercy)\s-\s(a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s-\s(a\*\*6\*b\*\*4\*x\*\*2)\/hull\s+\s(2\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s+\scleric\/(4\*mercy)\s-\s(a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/(2\*mercy)/ford/g' lambdaVar4.txt
# # ford = (3*a**14*b**2*x**4)/(4*beta) + lolo/(4*mercy) - papa/(4*mercy) - (a**8*b**6*x**2)/mercy + (a**10*b**2*x**4)/(2*mercy) + (3*a**10*b**4*x**2)/(2*mercy) - (a**12*b**2*x**2)/mercy - (a**6*b**4*x**2)/hull + (2*a**8*b**2*x**2)/hull + cleric/(4*mercy) - (a**8*b**4*x**2*y**2)/(2*mercy)
# # sed -i 's/((a\*\*6\*x\*\*2)\/alpha\s+\s(3\*a\*\*16\*x\*\*4)\/(16\*beta)\s-\s(a\*\*12\*x\*\*4)\/(4\*mercy)\s+\s(a\*\*14\*x\*\*2)\/(4\*mercy)\s-\s(a\*\*10\*x\*\*2)\/hull\s+\s(3\*a\*\*8\*b\*\*8\*x\*\*4)\/(16\*beta)\s-\stempest\/(4\*beta)\s+\s(9\*a\*\*12\*b\*\*4\*x\*\*4)\/(8\*beta)\s-\sford\s+\s(a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/(4\*mercy))/juan/g' lambdaVar4.txt
# # juan = ((a**6*x**2)/alpha + (3*a**16*x**4)/(16*beta) - (a**12*x**4)/(4*mercy) + (a**14*x**2)/(4*mercy) - (a**10*x**2)/hull + (3*a**8*b**8*x**4)/(16*beta) - tempest/(4*beta) + (9*a**12*b**4*x**4)/(8*beta) - ford + (a**10*b**2*x**2*y**2)/(4*mercy))
# # sed -i 's/(eleph\*\*2\s+\s9\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(2\/3)\s+\s6\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(1\/3)\*eleph\s-\s(12\*a\*\*6\*x\*\*2)\/alpha\s-\s(9\*a\*\*16\*x\*\*4)\/(4\*beta)\s+\s(3\*a\*\*12\*x\*\*4)\/mercy\s-\s(3\*a\*\*14\*x\*\*2)\/mercy\s+\s(12\*a\*\*10\*x\*\*2)\/hull\s-\s(9\*a\*\*8\*b\*\*8\*x\*\*4)\/(4\*beta)\s+\s(9\*a\*\*10\*b\*\*6\*x\*\*4)\/beta\s-\s(27\*a\*\*12\*b\*\*4\*x\*\*4)\/(2\*beta)\s+\s(9\*a\*\*14\*b\*\*2\*x\*\*4)\/beta\s-\s(3\*a\*\*6\*b\*\*8\*x\*\*2)\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*4)\/mercy\s+\s(12\*a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s-\s(6\*a\*\*10\*b\*\*2\*x\*\*4)\/mercy\s-\s(18\*a\*\*10\*b\*\*4\*x\*\*2)\/mercy\s+\s(12\*a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s+\s(12\*a\*\*6\*b\*\*4\*x\*\*2)\/hull\s-\s(24\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s-\s(3\*a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)\/mercy\s+\s(6\*a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/mercy\s-\s(3\*a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/mercy)/crenshaw/g' lambdaVar4.txt
# # crenshaw = (eleph**2 + 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*(256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)**(1/2))/18 - (4*eleph*juan)/3)**(2/3) + 6*(kale**2/2 - eleph**3/27 + (3**(1/2)*(256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph - (12*a**6*x**2)/alpha - (9*a**16*x**4)/(4*beta) + (3*a**12*x**4)/mercy - (3*a**14*x**2)/mercy + (12*a**10*x**2)/hull - (9*a**8*b**8*x**4)/(4*beta) + (9*a**10*b**6*x**4)/beta - (27*a**12*b**4*x**4)/(2*beta) + (9*a**14*b**2*x**4)/beta - (3*a**6*b**8*x**2)/mercy + (3*a**8*b**4*x**4)/mercy + (12*a**8*b**6*x**2)/mercy - (6*a**10*b**2*x**4)/mercy - (18*a**10*b**4*x**2)/mercy + (12*a**12*b**2*x**2)/mercy + (12*a**6*b**4*x**2)/hull - (24*a**8*b**2*x**2)/hull - (3*a**6*b**6*x**2*y**2)/mercy + (6*a**8*b**4*x**2*y**2)/mercy - (3*a**10*b**2*x**2*y**2)/mercy)
# # sed -i 's/(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)/moody/g' lambdaVar4.txt
# # moody = (256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)
# # sed -i 's/(12\*juan\*crenshaw\*\*(1\/2)\s-\s9\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*moody\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(2\/3)\*crenshaw\*\*(1\/2)\s-\seleph\*\*2\*crenshaw\*\*(1\/2)\s-\s3\*6\*\*(1\/2)\*(27\*kale\*\*2\s-\s2\*eleph\*\*3\s+\s3\*3\*\*(1\/2)\*moody\*\*(1\/2)\s-\s72\*eleph\*juan)\*\*(1\/2)\*kale\s+\s12\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*moody\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(1\/3)\*eleph\*crenshaw\*\*(1\/2))/bigP/g' lambdaVar4.txt
# # bigP = (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) - 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))


# def quarticRoots_matlab(a, b, x, y):

#     # sed -i 's/\^/**/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*4\s+\sb\*\*4\s-\s2\*a\*\*2\*b\*\*2)/alpha/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*16\s+\sb\*\*16\s-\s8\*a\*\*2\*b\*\*14\s+\s28\*a\*\*4\*b\*\*12\s-\s56\*a\*\*6\*b\*\*10\s+\s70\*a\*\*8\*b\*\*8\s-\s56\*a\*\*10\*b\*\*6\s+\s28\*a\*\*12\*b\*\*4\s-\s8\*a\*\*14\*b\*\*2)/beta/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*12\s+\sb\*\*12\s-\s6\*a\*\*2\*b\*\*10\s+\s15\*a\*\*4\*b\*\*8\s-\s20\*a\*\*6\*b\*\*6\s+\s15\*a\*\*8\*b\*\*4\s-\s6\*a\*\*10\*b\*\*2)/mercy/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*8\s+\sb\*\*8\s-\s4\*a\*\*2\*b\*\*6\s+\s6\*a\*\*4\*b\*\*4\s-\s4\*a\*\*6\*b\*\*2)/hull/g' lambdaVar4.txt
#     #sed -i 's/(2\*a\*\*4\*b\*\*2)/ham/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*6\*b\*\*8\*x\*\*2)/lolo/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*8\*b\*\*4\*x\*\*4)/papa/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*10\*b\*\*2\*x\*\*3)/jop/g' lambdaVar4.txt
#     #sed -i 's/(2\*a\*\*4\*x\s-\s2\*a\*\*2\*b\*\*2\*x)/gob/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*4\*b\*\*4\*x\*\*2)/bing/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*2\*b\*\*2\*y\*\*2)/sing/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*6\*b\*\*2\*x\*\*2)/goya/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*8\*x\*\*2)/meh/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*6\*b\*\*4\*x)/gem/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*6\*b\*\*6\*x\*\*3)/dame/g' lambdaVar4.txt

#     #DELETEsed -i 's/(a\*\*6\/alpha\s+\s(a\*\*2\*b\*\*4)\/alpha\s-\sham\/alpha\s-\s(a\*\*4\*x\*\*2)\/alpha\s+\smeh\/(2\*hull)\s+\sbing\/(2\*hull)\s-\sgoya\/hull\s+\ssing\/alpha)/eleph/g' lambdaVar4.txt


#     alpha = (a**4 + b**4 - 2*a**2*b**2)
#     beta = (a**16 + b**16 - 8*a**2*b**14 + 28*a**4*b**12 - 56*a**6*b**10 + 70*a**8*b**8 - 56*a**10*b**6 + 28*a**12*b**4 - 8*a**14*b**2)
#     mercy = (a**12 + b**12 - 6*a**2*b**10 + 15*a**4*b**8 - 20*a**6*b**6 + 15*a**8*b**4 - 6*a**10*b**2)
#     hull = (a**8 + b**8 - 4*a**2*b**6 + 6*a**4*b**4 - 4*a**6*b**2)
#     ham = (2*a**4*b**2)
#     lolo = (a**6*b**8*x**2)
#     papa = (a**8*b**4*x**4)
#     jop = (3*a**10*b**2*x**3)
#     gob = (2*a**4*x - 2*a**2*b**2*x)
#     bing = (3*a**4*b**4*x**2)
#     sing = (a**2*b**2*y**2)
#     goya = (3*a**6*b**2*x**2)
#     meh = (3*a**8*x**2)
#     gem = (3*a**6*b**4*x)
#     dame = (a**6*b**6*x**3)

    
#     # eleph = (a**6/alpha + (a**2*b**4)/alpha - ham/alpha - (a**4*x**2)/alpha + meh/(2*hull) + bing/(2*hull) - goya/hull + sing/alpha)
#     # #sed -i 's/((a\*\*10\*x)\/hull\s+\s(a\*\*12\*x\*\*3)\/mercy\s-\s(2\*a\*\*6\*x)\/alpha\s-\s(a\*\*8\*x\*\*3)\/hull\s-\s(a\*\*4\*b\*\*6\*x)\/hull\s+\sgem\/hull\s-\s(3\*a\*\*8\*b\*\*2\*x)\/hull\s-\sdame\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*3)\/mercy\s-\sjop\/mercy\s+\s(2\*a\*\*4\*b\*\*2\*x)\/alpha\s+\s(a\*\*6\*b\*\*2\*x\*\*3)\/hull\s-\s(a\*\*4\*b\*\*4\*x\*y\*\*2)\/hull\s+\s(a\*\*6\*b\*\*2\*x\*y\*\*2)\/hull)/kale/g' lambdaVar4.txt
#     # kale = ((a**10*x)/hull + (a**12*x**3)/mercy - (2*a**6*x)/alpha - (a**8*x**3)/hull - (a**4*b**6*x)/hull + gem/hull - (3*a**8*b**2*x)/hull - dame/mercy + (3*a**8*b**4*x**3)/mercy - jop/mercy + (2*a**4*b**2*x)/alpha + (a**6*b**2*x**3)/hull - (a**4*b**4*x*y**2)/hull + (a**6*b**2*x*y**2)/hull)
#     # #sed -i 's/(3\*a\*\*10\*b\*\*6\*x\*\*4)/tempest/g' lambdaVar4.txt
#     # tempest = (3*a**10*b**6*x**4)
#     # #sed -i 's/(a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)/cleric/g' lambdaVar4.txt
#     # cleric = (a**6*b**6*x**2*y**2)
#     # #sed -i 's/(3\*a\*\*14\*b\*\*2\*x\*\*4)\/(4\*beta)\s+\slolo\/(4\*mercy)\s-\spapa\/(4\*mercy)\s-\s(a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s+\s(a\*\*10\*b\*\*2\*x\*\*4)\/(2\*mercy)\s+\s(3\*a\*\*10\*b\*\*4\*x\*\*2)\/(2\*mercy)\s-\s(a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s-\s(a\*\*6\*b\*\*4\*x\*\*2)\/hull\s+\s(2\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s+\scleric\/(4\*mercy)\s-\s(a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/(2\*mercy)/ford/g' lambdaVar4.txt
#     # ford = (3*a**14*b**2*x**4)/(4*beta) + lolo/(4*mercy) - papa/(4*mercy) - (a**8*b**6*x**2)/mercy + (a**10*b**2*x**4)/(2*mercy) + (3*a**10*b**4*x**2)/(2*mercy) - (a**12*b**2*x**2)/mercy - (a**6*b**4*x**2)/hull + (2*a**8*b**2*x**2)/hull + cleric/(4*mercy) - (a**8*b**4*x**2*y**2)/(2*mercy)
#     # #sed -i 's/((a\*\*6\*x\*\*2)\/alpha\s+\s(3\*a\*\*16\*x\*\*4)\/(16\*beta)\s-\s(a\*\*12\*x\*\*4)\/(4\*mercy)\s+\s(a\*\*14\*x\*\*2)\/(4\*mercy)\s-\s(a\*\*10\*x\*\*2)\/hull\s+\s(3\*a\*\*8\*b\*\*8\*x\*\*4)\/(16\*beta)\s-\stempest\/(4\*beta)\s+\s(9\*a\*\*12\*b\*\*4\*x\*\*4)\/(8\*beta)\s-\sford\s+\s(a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/(4\*mercy))/juan/g' lambdaVar4.txt
#     # juan = ((a**6*x**2)/alpha + (3*a**16*x**4)/(16*beta) - (a**12*x**4)/(4*mercy) + (a**14*x**2)/(4*mercy) - (a**10*x**2)/hull + (3*a**8*b**8*x**4)/(16*beta) - tempest/(4*beta) + (9*a**12*b**4*x**4)/(8*beta) - ford + (a**10*b**2*x**2*y**2)/(4*mercy))
#     # #sed -i 's/(eleph\*\*2\s+\s9\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(2\/3)\s+\s6\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(1\/3)\*eleph\s-\s(12\*a\*\*6\*x\*\*2)\/alpha\s-\s(9\*a\*\*16\*x\*\*4)\/(4\*beta)\s+\s(3\*a\*\*12\*x\*\*4)\/mercy\s-\s(3\*a\*\*14\*x\*\*2)\/mercy\s+\s(12\*a\*\*10\*x\*\*2)\/hull\s-\s(9\*a\*\*8\*b\*\*8\*x\*\*4)\/(4\*beta)\s+\s(9\*a\*\*10\*b\*\*6\*x\*\*4)\/beta\s-\s(27\*a\*\*12\*b\*\*4\*x\*\*4)\/(2\*beta)\s+\s(9\*a\*\*14\*b\*\*2\*x\*\*4)\/beta\s-\s(3\*a\*\*6\*b\*\*8\*x\*\*2)\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*4)\/mercy\s+\s(12\*a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s-\s(6\*a\*\*10\*b\*\*2\*x\*\*4)\/mercy\s-\s(18\*a\*\*10\*b\*\*4\*x\*\*2)\/mercy\s+\s(12\*a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s+\s(12\*a\*\*6\*b\*\*4\*x\*\*2)\/hull\s-\s(24\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s-\s(3\*a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)\/mercy\s+\s(6\*a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/mercy\s-\s(3\*a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/mercy)/crenshaw/g' lambdaVar4.txt
#     # crenshaw = (eleph**2 + 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*(256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)**(1/2))/18 - (4*eleph*juan)/3)**(2/3) + 6*(kale**2/2 - eleph**3/27 + (3**(1/2)*(256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph - (12*a**6*x**2)/alpha - (9*a**16*x**4)/(4*beta) + (3*a**12*x**4)/mercy - (3*a**14*x**2)/mercy + (12*a**10*x**2)/hull - (9*a**8*b**8*x**4)/(4*beta) + (9*a**10*b**6*x**4)/beta - (27*a**12*b**4*x**4)/(2*beta) + (9*a**14*b**2*x**4)/beta - (3*a**6*b**8*x**2)/mercy + (3*a**8*b**4*x**4)/mercy + (12*a**8*b**6*x**2)/mercy - (6*a**10*b**2*x**4)/mercy - (18*a**10*b**4*x**2)/mercy + (12*a**12*b**2*x**2)/mercy + (12*a**6*b**4*x**2)/hull - (24*a**8*b**2*x**2)/hull - (3*a**6*b**6*x**2*y**2)/mercy + (6*a**8*b**4*x**2*y**2)/mercy - (3*a**10*b**2*x**2*y**2)/mercy)
#     # #sed -i 's/(256\*juan\*\*3\s-\s4\*eleph\*\*3\*kale\*\*2\s+\s16\*eleph\*\*4\*juan\s+\s128\*eleph\*\*2\*juan\*\*2\s+\s27\*kale\*\*4\s-\s144\*eleph\*kale\*\*2\*juan)/moody/g' lambdaVar4.txt
#     # moody = (256*juan**3 - 4*eleph**3*kale**2 + 16*eleph**4*juan + 128*eleph**2*juan**2 + 27*kale**4 - 144*eleph*kale**2*juan)
#     # #sed -i 's/(12\*juan\*crenshaw\*\*(1\/2)\s-\s9\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*moody\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(2\/3)\*crenshaw\*\*(1\/2)\s-\seleph\*\*2\*crenshaw\*\*(1\/2)\s-\s3\*6\*\*(1\/2)\*(27\*kale\*\*2\s-\s2\*eleph\*\*3\s+\s3\*3\*\*(1\/2)\*moody\*\*(1\/2)\s-\s72\*eleph\*juan)\*\*(1\/2)\*kale\s+\s12\*(kale\*\*2\/2\s-\seleph\*\*3\/27\s+\s(3\*\*(1\/2)\*moody\*\*(1\/2))\/18\s-\s(4\*eleph\*juan)\/3)\*\*(1\/3)\*eleph\*crenshaw\*\*(1\/2))/bigP/g' lambdaVar4.txt
#     # bigP = (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) - 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))




#     #Solutions
#     x0 = gob/(4*alpha) - crenshaw**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) - bigP**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)*crenshaw**(1/4))
#     x1 = gob/(4*alpha) - crenshaw**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) + bigP**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)*crenshaw**(1/4))
#     x2 = crenshaw**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) + gob/(4*alpha) - (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)*crenshaw**(1/4))
#     x3 = crenshaw**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) + gob/(4*alpha) + (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)*crenshaw**(1/4))

#     #Conditions
#     c00 = a**4 + b**4 != 2*a**2*b**2
#     c01 = 0 < (a**2*x*(2*a**2 - 2*b**2))/(4*alpha) - (crenshaw**(1/2) + bigP**(1/2)/crenshaw**(1/4))/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6))
#     c02 = -(crenshaw**(1/2) + bigP**(1/2)/crenshaw**(1/4))/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
#     c10 = a**4 + b**4 != 2*a**2*b**2
#     c11 = -(crenshaw**(1/2)/6 - bigP**(1/2)/(6*crenshaw**(1/4)))/(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
#     c12 = 0 < (a**2*x*(2*a**2 - 2*b**2))/(4*alpha) - (crenshaw**(1/2)/6 - bigP**(1/2)/(6*crenshaw**(1/4)))/(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)
#     c20 = a**4 + b**4 != 2*a**2*b**2
#     c21 = (crenshaw**(1/2)/6 - (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/(6*crenshaw**(1/4)))/(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
#     c22 = 0 < (a**2*x*(2*a**2 - 2*b**2) + (4*(crenshaw**(1/2)/6 - (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/(6*crenshaw**(1/4)))*alpha)/(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6))/alpha
#     c30 = a**4 + b**4 != 2*a**2*b**2
#     c31 = 0 < (crenshaw**(1/2) + (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/crenshaw**(1/4))/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) + (a**2*x*(2*a**2 - 2*b**2))/(4*alpha)
#     c32 = (crenshaw**(1/2) + (12*juan*crenshaw**(1/2) - 9*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(2/3)*crenshaw**(1/2) - eleph**2*crenshaw**(1/2) + 3*6**(1/2)*(27*kale**2 - 2*eleph**3 + 3*3**(1/2)*moody**(1/2) - 72*eleph*juan)**(1/2)*kale + 12*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/3)*eleph*crenshaw**(1/2))**(1/2)/crenshaw**(1/4))/(6*(kale**2/2 - eleph**3/27 + (3**(1/2)*moody**(1/2))/18 - (4*eleph*juan)/3)**(1/6)) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha

#     return x0, x1, x2, x3, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32

# def quarticRoots_matlab_2(a, b, x, y):
#     # sed -i 's/\^/**/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*4\s+\sb\*\*4\s-\s2\*a\*\*2\*b\*\*2)/alpha/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*16\s+\sb\*\*16\s-\s8\*a\*\*2\*b\*\*14\s+\s28\*a\*\*4\*b\*\*12\s-\s56\*a\*\*6\*b\*\*10\s+\s70\*a\*\*8\*b\*\*8\s-\s56\*a\*\*10\*b\*\*6\s+\s28\*a\*\*12\*b\*\*4\s-\s8\*a\*\*14\*b\*\*2)/beta/g' lambdaVar4.txt
#     # sed -i 's/(a\*\*12\s+\sb\*\*12\s-\s6\*a\*\*2\*b\*\*10\s+\s15\*a\*\*4\*b\*\*8\s-\s20\*a\*\*6\*b\*\*6\s+\s15\*a\*\*8\*b\*\*4\s-\s6\*a\*\*10\*b\*\*2)/mercy/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*8\s+\sb\*\*8\s-\s4\*a\*\*2\*b\*\*6\s+\s6\*a\*\*4\*b\*\*4\s-\s4\*a\*\*6\*b\*\*2)/hull/g' lambdaVar4.txt
#     #sed -i 's/(2\*a\*\*4\*b\*\*2)/ham/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*6\*b\*\*8\*x\*\*2)/lolo/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*8\*b\*\*4\*x\*\*4)/papa/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*10\*b\*\*2\*x\*\*3)/jop/g' lambdaVar4.txt
#     #sed -i 's/(2\*a\*\*4\*x\s-\s2\*a\*\*2\*b\*\*2\*x)/gob/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*4\*b\*\*4\*x\*\*2)/bing/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*2\*b\*\*2\*y\*\*2)/sing/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*6\*b\*\*2\*x\*\*2)/goya/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*8\*x\*\*2)/meh/g' lambdaVar4.txt
#     #sed -i 's/(3\*a\*\*6\*b\*\*4\*x)/gem/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*6\*b\*\*6\*x\*\*3)/dame/g' lambdaVar4.txt
#     #sed -i 's/(a\*\*6\/alpha\s+\s(a\*\*2\*b\*\*4)\/alpha\s-\sham\/alpha\s-\s(a\*\*4\*x\*\*2)\/alpha\s+\smeh\/(2\*hull)\s+\sbing\/(2\*hull)\s-\sgoya\/hull\s-\ssing\/alpha)/ranch/g' lambdaVar5.txt
#     #sed -i 's/((a\*\*10\*x)\/hull\s+\s(a\*\*12\*x\*\*3)\/mercy\s-\s(2\*a\*\*6\*x)\/alpha\s-\s(a\*\*8\*x\*\*3)\/hull\s-\s(a\*\*4\*b\*\*6\*x)\/hull\s+\sgem\/hull\s-\s(3\*a\*\*8\*b\*\*2\*x)\/hull\s-\sdame\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*3)\/mercy\s-\sjop\/mercy\s+\s(2\*a\*\*4\*b\*\*2\*x)\/alpha\s+\s(a\*\*6\*b\*\*2\*x\*\*3)\/hull\s+\s(a\*\*4\*b\*\*4\*x\*y\*\*2)\/hull\s-\s(a\*\*6\*b\*\*2\*x\*y\*\*2)\/hull)/kembo/g' lambdaVar5.txt
#     #sed -i 's/((a\*\*6\*x\*\*2)\/alpha\s+\s(3\*a\*\*16\*x\*\*4)\/(16\*beta)\s-\s(a\*\*12\*x\*\*4)\/(4\*mercy)\s+\s(a\*\*14\*x\*\*2)\/(4\*mercy)\s-\s(a\*\*10\*x\*\*2)\/hull\s+\s(3\*a\*\*8\*b\*\*8\*x\*\*4)\/(16\*beta)\s-\s(3\*a\*\*10\*b\*\*6\*x\*\*4)\/(4\*beta)\s+\s(9\*a\*\*12\*b\*\*4\*x\*\*4)\/(8\*beta)\s-\s(3\*a\*\*14\*b\*\*2\*x\*\*4)\/(4\*beta)\s+\slolo\/(4\*mercy)\s-\spapa\/(4\*mercy)\s-\s(a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s+\s(a\*\*10\*b\*\*2\*x\*\*4)\/(2\*mercy)\s+\s(3\*a\*\*10\*b\*\*4\*x\*\*2)\/(2\*mercy)\s-\s(a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s-\s(a\*\*6\*b\*\*4\*x\*\*2)\/hull\s+\s(2\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s-\s(a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)\/(4\*mercy)\s+\s(a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/(2\*mercy)\s-\s(a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/(4\*mercy))/yak/g' lambdaVar5.txt
#     #sed -i 's/(256\*yak\*\*3\s-\s4\*ranch\*\*3\*kembo\*\*2\s+\s16\*ranch\*\*4\*yak\s+\s128\*ranch\*\*2\*yak\*\*2\s+\s27\*kembo\*\*4\s-\s144\*ranch\*kembo\*\*2\*yak)/hail/g' lambdaVar5.txt
#     #sed -i 's/(kembo\*\*2\/2\s-\sranch\*\*3\/27\s+\s(3\*\*(1\/2)\*hail\*\*(1\/2))\/18\s-\s(4\*ranch\*yak)\/3)/ride/g' lambdaVar5.txt
#     #sed -i 's/(ranch\*\*2\s+\s9\*ride\*\*(2\/3)\s+\s6\*ride\*\*(1\/3)\*ranch\s-\s(12\*a\*\*6\*x\*\*2)\/alpha\s-\s(9\*a\*\*16\*x\*\*4)\/(4\*beta)\s+\s(3\*a\*\*12\*x\*\*4)\/mercy\s-\s(3\*a\*\*14\*x\*\*2)\/mercy\s+\s(12\*a\*\*10\*x\*\*2)\/hull\s-\s(9\*a\*\*8\*b\*\*8\*x\*\*4)\/(4\*beta)\s+\s(9\*a\*\*10\*b\*\*6\*x\*\*4)\/beta\s-\s(27\*a\*\*12\*b\*\*4\*x\*\*4)\/(2\*beta)\s+\s(9\*a\*\*14\*b\*\*2\*x\*\*4)\/beta\s-\s(3\*a\*\*6\*b\*\*8\*x\*\*2)\/mercy\s+\s(3\*a\*\*8\*b\*\*4\*x\*\*4)\/mercy\s+\s(12\*a\*\*8\*b\*\*6\*x\*\*2)\/mercy\s-\s(6\*a\*\*10\*b\*\*2\*x\*\*4)\/mercy\s-\s(18\*a\*\*10\*b\*\*4\*x\*\*2)\/mercy\s+\s(12\*a\*\*12\*b\*\*2\*x\*\*2)\/mercy\s+\s(12\*a\*\*6\*b\*\*4\*x\*\*2)\/hull\s-\s(24\*a\*\*8\*b\*\*2\*x\*\*2)\/hull\s+\s(3\*a\*\*6\*b\*\*6\*x\*\*2\*y\*\*2)\/mercy\s-\s(6\*a\*\*8\*b\*\*4\*x\*\*2\*y\*\*2)\/mercy\s+\s(3\*a\*\*10\*b\*\*2\*x\*\*2\*y\*\*2)\/mercy)/finale/g' lambdaVar5.txt
    
#     #DELETEsed -i 's/(a\*\*6\/alpha\s+\s(a\*\*2\*b\*\*4)\/alpha\s-\sham\/alpha\s-\s(a\*\*4\*x\*\*2)\/alpha\s+\smeh\/(2\*hull)\s+\sbing\/(2\*hull)\s-\sgoya\/hull\s+\ssing\/alpha)/eleph/g' lambdaVar4.txt


#     alpha = (a**4 + b**4 - 2*a**2*b**2)
#     beta = (a**16 + b**16 - 8*a**2*b**14 + 28*a**4*b**12 - 56*a**6*b**10 + 70*a**8*b**8 - 56*a**10*b**6 + 28*a**12*b**4 - 8*a**14*b**2)
#     mercy = (a**12 + b**12 - 6*a**2*b**10 + 15*a**4*b**8 - 20*a**6*b**6 + 15*a**8*b**4 - 6*a**10*b**2)
#     hull = (a**8 + b**8 - 4*a**2*b**6 + 6*a**4*b**4 - 4*a**6*b**2)
#     ham = (2*a**4*b**2)
#     lolo = (a**6*b**8*x**2)
#     papa = (a**8*b**4*x**4)
#     jop = (3*a**10*b**2*x**3)
#     gob = (2*a**4*x - 2*a**2*b**2*x)
#     bing = (3*a**4*b**4*x**2)
#     sing = (a**2*b**2*y**2)
#     goya = (3*a**6*b**2*x**2)
#     meh = (3*a**8*x**2)
#     gem = (3*a**6*b**4*x)
#     dame = (a**6*b**6*x**3)
#     ranch = (a**6/alpha + (a**2*b**4)/alpha - ham/alpha - (a**4*x**2)/alpha + meh/(2*hull) + bing/(2*hull) - goya/hull - sing/alpha)
#     kembo = ((a**10*x)/hull + (a**12*x**3)/mercy - (2*a**6*x)/alpha - (a**8*x**3)/hull - (a**4*b**6*x)/hull + gem/hull - (3*a**8*b**2*x)/hull - dame/mercy + (3*a**8*b**4*x**3)/mercy - jop/mercy + (2*a**4*b**2*x)/alpha + (a**6*b**2*x**3)/hull + (a**4*b**4*x*y**2)/hull - (a**6*b**2*x*y**2)/hull)
#     yak = ((a**6*x**2)/alpha + (3*a**16*x**4)/(16*beta) - (a**12*x**4)/(4*mercy) + (a**14*x**2)/(4*mercy) - (a**10*x**2)/hull + (3*a**8*b**8*x**4)/(16*beta) - (3*a**10*b**6*x**4)/(4*beta) + (9*a**12*b**4*x**4)/(8*beta) - (3*a**14*b**2*x**4)/(4*beta) + lolo/(4*mercy) - papa/(4*mercy) - (a**8*b**6*x**2)/mercy + (a**10*b**2*x**4)/(2*mercy) + (3*a**10*b**4*x**2)/(2*mercy) - (a**12*b**2*x**2)/mercy - (a**6*b**4*x**2)/hull + (2*a**8*b**2*x**2)/hull - (a**6*b**6*x**2*y**2)/(4*mercy) + (a**8*b**4*x**2*y**2)/(2*mercy) - (a**10*b**2*x**2*y**2)/(4*mercy))
#     hail = (256*yak**3 - 4*ranch**3*kembo**2 + 16*ranch**4*yak + 128*ranch**2*yak**2 + 27*kembo**4 - 144*ranch*kembo**2*yak)
#     ride = (kembo**2/2 - ranch**3/27 + (3**(1/2)*hail**(1/2))/18 - (4*ranch*yak)/3)
#     finale = (ranch**2 + 9*ride**(2/3) + 6*ride**(1/3)*ranch - (12*a**6*x**2)/alpha - (9*a**16*x**4)/(4*beta) + (3*a**12*x**4)/mercy - (3*a**14*x**2)/mercy + (12*a**10*x**2)/hull - (9*a**8*b**8*x**4)/(4*beta) + (9*a**10*b**6*x**4)/beta - (27*a**12*b**4*x**4)/(2*beta) + (9*a**14*b**2*x**4)/beta - (3*a**6*b**8*x**2)/mercy + (3*a**8*b**4*x**4)/mercy + (12*a**8*b**6*x**2)/mercy - (6*a**10*b**2*x**4)/mercy - (18*a**10*b**4*x**2)/mercy + (12*a**12*b**2*x**2)/mercy + (12*a**6*b**4*x**2)/hull - (24*a**8*b**2*x**2)/hull + (3*a**6*b**6*x**2*y**2)/mercy - (6*a**8*b**4*x**2*y**2)/mercy + (3*a**10*b**2*x**2*y**2)/mercy)

#     x0 = gob/(4*alpha) - finale**(1/2)/(6*ride**(1/6)) - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*ride**(1/6)*finale**(1/4))
#     x1 = gob/(4*alpha) - finale**(1/2)/(6*ride**(1/6)) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*ride**(1/6)*finale**(1/4))
#     x2 = finale**(1/2)/(6*ride**(1/6)) + gob/(4*alpha) - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*ride**(1/6)*finale**(1/4))
#     x3 = finale**(1/2)/(6*ride**(1/6)) + gob/(4*alpha) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*ride**(1/6)*finale**(1/4))
 
#     c00 = a**4 + b**4 != 2*a**2*b**2
#     c01 = 0 < (a**2*x*(2*a**2 - 2*b**2))/(4*alpha) - (finale**(1/2) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/finale**(1/4))/(6*ride**(1/6))
#     c02 = -(finale**(1/2) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/finale**(1/4))/(6*ride**(1/6)) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
                        
#     c10 = a**4 + b**4 != 2*a**2*b**2
#     c11 = -(finale**(1/2)/6 - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*finale**(1/4)))/ride**(1/6) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
#     c12 = 0 < (a**2*x*(2*a**2 - 2*b**2))/(4*alpha) - (finale**(1/2)/6 - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) - 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*finale**(1/4)))/ride**(1/6)
 
#     c20 = a**4 + b**4 != 2*a**2*b**2
#     c21 = (finale**(1/2)/6 - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*finale**(1/4)))/ride**(1/6) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
#     c22 = 0 < (a**2*x*(2*a**2 - 2*b**2) + (4*(finale**(1/2)/6 - (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/(6*finale**(1/4)))*alpha)/ride**(1/6))/alpha
                             
#     c30 = a**4 + b**4 != 2*a**2*b**2
#     c31 = 0 < (finale**(1/2) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/finale**(1/4))/(6*ride**(1/6)) + (a**2*x*(2*a**2 - 2*b**2))/(4*alpha)
#     c32 = (finale**(1/2) + (12*yak*finale**(1/2) - 9*ride**(2/3)*finale**(1/2) - ranch**2*finale**(1/2) + 3*6**(1/2)*(27*kembo**2 - 2*ranch**3 + 3*3**(1/2)*hail**(1/2) - 72*ranch*yak)**(1/2)*kembo + 12*ride**(1/3)*ranch*finale**(1/2))**(1/2)/finale**(1/4))/(6*ride**(1/6)) <= (a*(a**4 - (a**3*x)/2 + b**4 - 2*a**2*b**2 + (a*b**2*x)/2))/alpha
 
#     return x0, x1, x2, x3, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32


# def mathematicaRoots(a, b, x, y):
#     #From Arnaldo, the best roomie ever
#     port = (-a**4+2*a**2*b**2-b**4)
#     ham = (a**4*x-a**2*b**2*x)
#     havarti = (a**6-2*a**4*b**2+a**2*b**4-a**4*x**2-a**2*b**2*y**2)
#     irk = (-a**6+2*a**4*b**2-a**2*b**4+a**4*x**2+a**2*b**2*y**2)
#     mave = (a**6*x-a**4*b**2*x)
#     rim = (a**2-b**2)
#     mater = (a**4-2*a**2*b**2+b**4-a**2*x**2-b**2*y**2)

#     x0 = ((1/2)*a**2*rim**(-1)*x+(-1/2)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+\
#         36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(1/2)+(-1/2)*(2*a**4*rim**(-2)*x**2-1/3*port**(-1)*havarti-rim**(-2)*irk-1/3*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+\
#         (-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)-1/3*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3)+\
#         (-1/4)*(-16*a**4*rim**(-1)*x+8*a**6*rim**(-3)*x**3-8*a**2*rim**(-3)*x*irk)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(-1/2))**(1/2))

#     x1 = ((1/2)*a**2*rim**(-1)*x+(-1/2)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+\
#         36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(1/2)+(1/2)*(2*a**4*rim**(-2)*x**2-1/3*port**(-1)*havarti-rim**(-2)*irk-1/3*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+\
#         (-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)-1/3*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3)+\
#         (-1/4)*(-16*a**4*rim**(-1)*x+8*a**6*rim**(-3)*x**3-8*a**2*rim**(-3)*x*irk)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(-1/2))**(1/2))

#     x2 = ((1/2)*a**2*rim**(-1)*x+(1/2)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+\
#         36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(1/2)+(-1/2)*(2*a**4*rim**(-2)*x**2-1/3*port**(-1)*havarti-rim**(-2)*irk-1/3*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+\
#         (-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)-1/3*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3)+\
#         (1/4)*(-16*a**4*rim**(-1)*x+8*a**6*rim**(-3)*x**3-8*a**2*rim**(-3)*x*irk)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(-1/2))**(1/2))

#     x3 = ((1/2)*a**2*rim**(-1)*x+(1/2)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+\
#         36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(1/2)+(1/2)*(2*a**4*rim**(-2)*x**2-1/3*port**(-1)*havarti-rim**(-2)*irk-1/3*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+\
#         (-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)-1/3*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3)+\
#         (1/4)*(-16*a**4*rim**(-1)*x+8*a**6*rim**(-3)*x**3-8*a**2*rim**(-3)*x*irk)*(a**4*rim**(-2)*x**2+(1/3)*port**(-1)*havarti-rim**(-2)*irk+(1/3)*2**(1/3)*a**4*port**(-1)*mater**2*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(-1/3)+(1/3)*2**(-1/3)*port**(-1)*(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3+(-4*(12*a**6*port*x**2+12*ham*mave+havarti**2)**3+(108*a**6*x**2*ham**2+108*port*mave**2-72*a**6*port*x**2*havarti+36*ham*mave*havarti+2*havarti**3)**2)**(1/2))**(1/3))**(-1/2))**(1/2))

#     return x0, x1, x2, x3

# def mathematicaRoots_dean(a,b,x,y):

#     porter = (a**6 - 2*a**4*b**2 + a**2*b**4 - a**4*x**2 - a**2*b**2*y**2)
#     marble = (-a**4 + 2*a**2*b**2 - b**4)
#     hand = (a**4*x - a**2*b**2*x)
#     yelp = (-a**6 + 2*a**4*b**2 - a**2*b**4 + a**4*x**2 + a**2*b**2*y**2)
#     jak = (a**6*x - a**4*b**2*x)
#     dip = (a**4 - 2*a**2*b**2 + b**4 - a**2*x**2 - b**2*y**2)
#     randal = 108*a**6*x**2*hand**2 + 108*marble*jak**2 - 72*a**6*marble*x**2*porter + 36*hand*jak*porter + 2*porter**3
#     sqrt1 = -4*(12*a**6*marble*x**2 + 12*hand*jak + porter**2)**3 + (randal)**2
#     sqrt2 = (a**4*x**2)/(a**2 - b**2)**2 + porter/(3*marble) - yelp/(a**2 - b**2)**2 + (2**(1/3)*a**4*dip**2)/(3*marble*(randal + np.sqrt(sqrt1))**(1/3)) + (randal + np.sqrt(sqrt1))**(1/3)/(3*2**(1/3)*marble)
#     print('MATHEMATICA DEAN')
#     print(np.count_nonzero(sqrt1 < 0))
#     print(np.count_nonzero(sqrt2 < 0))

#     x0 = (a**2*x)/(2*(a**2 - b**2)) - np.sqrt(sqrt2)/2 - np.sqrt((2*a**4*x**2)/(a**2 - b**2)**2 - porter/(3*marble) - yelp/(a**2 - b**2)**2 - (2**(1/3)*a**4*dip**2)/(3*marble*(randal + np.sqrt(sqrt1))**(1/3)) - (randal + np.sqrt(sqrt1))**(1/3)/(3*2**(1/3)*marble) - ((-16*a**4*x)/(a**2 - b**2) + (8*a**6*x**3)/(a**2 - b**2)**3 - (8*a**2*x*yelp)/(a**2 - b**2)**3)/(4*np.sqrt(sqrt2)))/2

#     x1 = (a**2*x)/(2*(a**2 - b**2)) - np.sqrt(sqrt2)/2 + np.sqrt((2*a**4*x**2)/(a**2 - b**2)**2 - porter/(3*marble) - yelp/(a**2 - b**2)**2 - (2**(1/3)*a**4*dip**2)/(3*marble*(randal + np.sqrt(sqrt1))**(1/3)) - (randal + np.sqrt(sqrt1))**(1/3)/(3*2**(1/3)*marble) - ((-16*a**4*x)/(a**2 - b**2) + (8*a**6*x**3)/(a**2 - b**2)**3 - (8*a**2*x*yelp)/(a**2 - b**2)**3)/(4*np.sqrt(sqrt2)))/2

#     x2 = (a**2*x)/(2*(a**2 - b**2)) + np.sqrt(sqrt2)/2 - np.sqrt((2*a**4*x**2)/(a**2 - b**2)**2 - porter/(3*marble) - yelp/(a**2 - b**2)**2 - (2**(1/3)*a**4*dip**2)/(3*marble*(randal + np.sqrt(sqrt1))**(1/3)) - (randal + np.sqrt(sqrt1))**(1/3)/(3*2**(1/3)*marble) + ((-16*a**4*x)/(a**2 - b**2) + (8*a**6*x**3)/(a**2 - b**2)**3 - (8*a**2*x*yelp)/(a**2 - b**2)**3)/(4*np.sqrt(sqrt2)))/2

#     x3 =  (a**2*x)/(2*(a**2 - b**2)) + np.sqrt(sqrt2)/2 + np.sqrt((2*a**4*x**2)/(a**2 - b**2)**2 - porter/(3*marble) - yelp/(a**2 - b**2)**2 - (2**(1/3)*a**4*dip**2)/(3*marble*(randal + np.sqrt(sqrt1))**(1/3)) - (randal + np.sqrt(sqrt1))**(1/3)/(3*2**(1/3)*marble) + ((-16*a**4*x)/(a**2 - b**2) + (8*a**6*x**3)/(a**2 - b**2)**3 - (8*a**2*x*yelp)/(a**2 - b**2)**3)/(4*np.sqrt(sqrt2)))/2

#     return x0, x1, x2, x3

def checkResiduals(A,B,C,D,xreals2,inds,numSols):
    residual_0 = xreals2[inds,0]**4 + A[inds]*xreals2[inds,0]**3 + B[inds]*xreals2[inds,0]**2 + C[inds]*xreals2[inds,0] + D[inds]
    residual_1 = xreals2[inds,1]**4 + A[inds]*xreals2[inds,1]**3 + B[inds]*xreals2[inds,1]**2 + C[inds]*xreals2[inds,1] + D[inds]
    residual_2 = np.zeros(residual_0.shape)
    residual_3 = np.zeros(residual_0.shape)
    if numSols > 2:
        residual_2 = xreals2[inds,2]**4 + A[inds]*xreals2[inds,2]**3 + B[inds]*xreals2[inds,2]**2 + C[inds]*xreals2[inds,2] + D[inds]
        if numSols > 3:
            residual_3 = xreals2[inds,3]**4 + A[inds]*xreals2[inds,3]**3 + B[inds]*xreals2[inds,3]**2 + C[inds]*xreals2[inds,3] + D[inds]
    residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
    isAll = np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7))
    maxRealResidual = np.max(np.real(residual))
    maxImagResidual = np.max(np.imag(residual))
    return residual, isAll, maxRealResidual, maxImagResidual

def quarticCoefficients_ellipse_to_Quarticipynb(a, b, x, y, r):
    """ Calculates coefficients of the quartic expression solving for the intersection between a circle with radius r and ellipse with semi-major axis a
    semi-minor axis b, and the center of the circle at x and y.
    Coefficients for the quartic of form x**4 + A*x**3 + B*x**2 + C*x + D = 0
    """
    A = -4*a**2*x/(a**2 - b**2)
    B = 2*a**2*(a**2*b**2 - a**2*r**2 + 3*a**2*x**2 + a**2*y**2 - b**4 + b**2*r**2 - b**2*x**2 + b**2*y**2)/(a**4 - 2*a**2*b**2 + b**4)
    C = 4*a**4*x*(-b**2 + r**2 - x**2 - y**2)/(a**4 - 2*a**2*b**2 + b**4)
    D = a**4*(b**4 - 2*b**2*r**2 + 2*b**2*x**2 - 2*b**2*y**2 + r**4 - 2*r**2*x**2 - 2*r**2*y**2 + x**4 + 2*x**2*y**2 + y**4)/(a**4 - 2*a**2*b**2 + b**4)
    return A, B, C, D

def quarticCoefficients_smin_smax_lmin_lmax(a, b, x, y):
    """ Calculates coefficients of the quartic equation solving where ds2/dxe = 0 for the distance between a point and the ellipse
    for an ellipse with semi-major axis a, semi-minor axis b, and point at x, y
    """
    Gamma = np.zeros(len(a),dtype='complex128')
    Gamma = (4*a**4 - 8*a**2*b**2 + 4*b**4)/a**2
    A = (-8*a**2*x + 8*b**2*x)/Gamma
    B = (-4*a**4 + 8*a**2*b**2 + 4*a**2*x**2 - 4*b**4 + 4*b**2*y**2)/Gamma
    C = (8*a**4*x - 8*a**2*b**2*x)/Gamma
    D = (-4*a**4*x**2)/Gamma
    return A, B, C, D

def quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D):
    """ Equations from ellipse_to_Quartic.ipynb
    """
    p0 = (-3*A**2/8+B)**3
    p1 = (A*(A**2/8-B/2)+C)**2
    p2 = -A*(A*(3*A**2/256-B/16)+C/4)+D
    p3 = -3*A**2/8+B
    p4 = 2*A*(A**2/8-B/2)
    p5 = -p0/108-p1/8+p2*p3/3
    p6 = (p0/216+p1/16-p2*p3/6+np.sqrt(p5**2/4+(-p2-p3**2/12)**3/27))**(1/3)
    p7 = A**2/4-2*B/3
    p8 = (2*p2+p3**2/6)/(3*p6)
    #, (-2*p2-p3**2/6)/(3*p6)
    p9 = np.sqrt(-2*p5**(1/3)+p7)
    p10 = np.sqrt(2*p6+p7+p8)
    p11 = A**2/2-4*B/3

    #otherwise case
    x0 = -A/4 - p10/2 - np.sqrt(p11 - 2*p6 - p8 + (2*C + p4)/p10)/2
    x1 = -A/4 - p10/2 + np.sqrt(p11 - 2*p6 - p8 + (2*C + p4)/p10)/2
    x2 = -A/4 + p10/2 - np.sqrt(p11 - 2*p6 - p8 + (-2*C - p4)/p10)/2
    x3 = -A/4 + p10/2 + np.sqrt(p11 - 2*p6 - p8 + (-2*C - p4)/p10)/2
    zeroInds = np.where(p2 + p3**2/12 == 0)[0] #piecewise condition
    if len(zeroInds) != 0:
        x0[zeroInds] = -A[zeroInds]/4 - p9[zeroInds]/2 - np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (2*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2
        x1[zeroInds] = -A[zeroInds]/4 - p9[zeroInds]/2 + np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (2*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2
        x2[zeroInds] = -A[zeroInds]/4 + p9[zeroInds]/2 - np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (-2*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2
        x3[zeroInds] = -A[zeroInds]/4 + p9[zeroInds]/2 + np.sqrt(p11[zeroInds] + 2*np.cbrt(p5[zeroInds]) + (-2*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2

    delta = 256*D**3 - 192*A*C*D**2 - 128*B**2*D**2 + 144*B*C**2*D - 27*C**4\
        + 144*A**2*B*D**2 - 6*A**2*C**2*D - 80*A*B**2*C*D + 18*A*B*C**3 + 16*B**4*D\
        - 4*B**3*C**2 - 27*A**4*D**2 + 18*A**3*B*C*D - 4*A**3*C**3 - 4*A**2*B**3*D + A**2*B**2*C**2 #verified against wikipedia multiple times
    assert 0 == np.count_nonzero(np.imag(delta)), 'All delta are real'
    delta = np.real(delta)
    P = 8*B - 3*A**2
    assert 0 == np.count_nonzero(np.imag(P)), 'Not all P are real'
    P = np.real(P)
    D2 = 64*D - 16*B**2 + 16*A**2*B - 16*A*C - 3*A**4 #is 0 if the quartic has 2 double roots 
    assert 0 == np.count_nonzero(np.imag(D2)), 'Not all D2 are real'
    D2 = np.real(D2)
    R = A**3 + 8*C - 4*A*B
    assert 0 == np.count_nonzero(np.imag(R)), 'Not all R are real'
    R = np.real(R)
    delta_0 = B**2 - 3*A*C + 12*D
    assert 0 == np.count_nonzero(np.imag(delta_0)), 'Not all delta_0 are real'
    delta_0 = np.real(delta_0)

    return np.asarray([x0, x1, x2, x3]).T, delta, P, D2, R, delta_0



### LEAVING THIS HERE, HAVE ALTERNATIVE
# #### SOLUTIONS TO ds2_dxe = 0
# phi = (a**2 - b**2)
# theta = (-a**6 + 2*a**4*b**2 + a**4*x**2 - a**2*b**4 + a**2*b**2*y**2)
# gamma = (a**4*x**2/(2*phi**2) - theta/(2*(a**4 - 2*a**2*b**2 + b**4)))
# beta = (3*a**4*x**2/(64*phi**2) - theta/(16*(a**4 - 2*a**2*b**2 + b**4)))
# alpha = (-3*a**4*x**2/(2*phi**2) + theta/(a**4 - 2*a**2*b**2 + b**4))
# epsilon = (-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi)
# gamma = (-(2*a**4*x/phi - 2*a**2*x*gamma/phi)**2/8 - alpha**3/108 + alpha*epsilon/3)
# gorg = (a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) - 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi - alpha**2/12)
# mini = ((2*a**4*x/phi - 2*a**2*x*gamma/phi)**2/16 + alpha**3/216 - alpha*epsilon/6 + sqrt(gorg**3/27 + gamma**2/4))

# Piecewise((a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)),
# (a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True)),


# Piecewise((a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True)),


# Piecewise((a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True))


# Piecewise((a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True))


def smin_smax_slmin_slmax(n, xreal, yreal, mx, my, x, y):
    """

    Returns:
        yrealAllRealInds - these indicies must have min, max, local min, local max
        yrealImagInds - these indicies must have min, max
    """
    yrealAllRealInds = np.where(np.all(np.abs(np.imag(yreal)) < 1e-5,axis=1))[0]
    yreal[np.abs(np.imag(yreal)) < 1e-5] = np.real(yreal[np.abs(np.imag(yreal)) < 1e-5]) #eliminate any unreasonably small imaginary components
    yrealImagInds = np.where(np.any(np.abs(np.imag(yreal)) >= 1e-5,axis=1))[0] #inds where any of the values are imaginary
    assert len(yrealImagInds) + len(yrealAllRealInds) == n, 'For some reason, this sum does not account for all planets'
    assert len(np.intersect1d(yrealImagInds,yrealAllRealInds)) == 0, 'For some reason, this sum does not account for all planets'
    #The following 7 lines can be deleted. it just says the first 2 cols of yreal have the smallest imaginary component
    yrealImagArgsortInds = np.argsort(np.abs(np.imag(yreal[yrealImagInds])),axis=1)
    assert len(yrealImagInds) == np.count_nonzero(yrealImagArgsortInds[:,0] == 0), "Not all first indicies have smallest Imag component"
    assert len(yrealImagInds) == np.count_nonzero(yrealImagArgsortInds[:,1] == 1), "Not all first indicies have second smallest Imag component"
    #maxImagFirstCol = np.max(np.imag(yreal[yrealImagInds,0]))
    assert np.max(np.imag(yreal[yrealImagInds,0])) == 0, 'max y imag component of column 0 is not 0'
    #maxImagSecondCol = np.max(np.imag(yreal[yrealImagInds,1]))
    assert np.max(np.imag(yreal[yrealImagInds,1])) == 0, 'max y imag component of column 1 is not 0'
    # np.max(np.imag(yreal[yrealImagInds,2])) #this is quite large
    # np.max(np.imag(yreal[yrealImagInds,3])) #this is quite large

    #Initialize minSep, lminSep, lmaxSep, maxSep
    minSep = np.zeros(xreal.shape[0])
    maxSep = np.zeros(xreal.shape[0])
    lminSep = np.zeros(len(yrealAllRealInds))
    lmaxSep = np.zeros(len(yrealAllRealInds))
    #Initialize minSepPoints_x, minSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, maxSepPoints_x, maxSepPoints_y
    minSepPoints_x = np.zeros(xreal.shape[0])
    minSepPoints_y = np.zeros(xreal.shape[0])
    lminSepPoints_x = np.zeros(len(yrealAllRealInds))
    lminSepPoints_y = np.zeros(len(yrealAllRealInds))
    lmaxSepPoints_x = np.zeros(len(yrealAllRealInds))
    lmaxSepPoints_y = np.zeros(len(yrealAllRealInds))
    maxSepPoints_x = np.zeros(xreal.shape[0])
    maxSepPoints_y = np.zeros(xreal.shape[0])
    ###################################################################################
    #### Smin and Smax Two Real Solutions Two Imaginary Solutions #####################
    #Smin and Smax need to be calculated separately for x,y with imaginary solutions vs those without
    #For yrealImagInds. Smin and Smax must be either first column of second column
    assert np.all(np.real(xreal[yrealImagInds,0]) > 0), 'not all xreal components are strictly negative'
    #assert np.all(np.real(xreal[yrealImagInds,0]) < 0), 'not all xreal components are strictly negative'
    assert np.all(np.real(yreal[yrealImagInds,0]) > 0), 'not all yreal components are strictly positive'
    assert np.all(np.real(xreal[yrealImagInds,1]) > 0), 'not all xreal components are strictly positive'
    assert np.all(np.real(yreal[yrealImagInds,1]) > 0), 'not all yreal components are strictly positive'
    smm0 = np.sqrt((np.real(xreal[yrealImagInds,0])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])-my[yrealImagInds])**2)
    smp0 = np.sqrt((np.real(xreal[yrealImagInds,0])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])+my[yrealImagInds])**2)
    spm0 = np.sqrt((np.real(xreal[yrealImagInds,0])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])-my[yrealImagInds])**2)
    spp0 = np.sqrt((np.real(xreal[yrealImagInds,0])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])+my[yrealImagInds])**2)
    smm1 = np.sqrt((np.real(xreal[yrealImagInds,1])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])-my[yrealImagInds])**2)
    smp1 = np.sqrt((np.real(xreal[yrealImagInds,1])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])+my[yrealImagInds])**2)
    spm1 = np.sqrt((np.real(xreal[yrealImagInds,1])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])-my[yrealImagInds])**2)
    spp1 = np.sqrt((np.real(xreal[yrealImagInds,1])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])+my[yrealImagInds])**2)

    #Search for Smallest
    smm = np.asarray([smm0,smm1])
    assert np.all(np.argmin(smm,axis=0) == 1), 'mins are not all are smm1'
    smp = np.asarray([smp0,smp1])
    assert np.all(np.argmin(smp,axis=0) == 0), 'mins are not all are smp0'
    spm = np.asarray([spm0,spm1])
    assert np.all(np.argmin(spm,axis=0) == 1), 'mins are not all are spm1'
    spp = np.asarray([spp0,spp1])
    assert np.all(np.argmin(spp,axis=0) == 1), 'mins are not all are spp1'
    #above says smallest must be one of these: smm1, smp0, spm1, spp1
    #The following are where each of these separations are 0
    smm1Inds = np.where((smm1 < smp0)*(smm1 < spm1)*(smm1 < spp1))[0]
    smp0Inds = np.where((smp0 < smm1)*(smp0 < spm1)*(smp0 < spp1))[0]
    #spp1Inds = np.where((smp0 < smm1)*(smp0 < spm1)*(smp0 < spp1))[0]
    spm1Inds = np.where((spm1 < smp0)*(spm1 < smm1)*(spm1 < spp1))[0]
    spp1Inds = np.where((spp1 < smp0)*(spp1 < spm1)*(spp1 < smm1))[0]
    assert len(yrealImagInds) == len(smm1Inds) + len(smp0Inds) + len(spm1Inds) + len(spp1Inds), 'Have not covered all cases'
    if len(smm1Inds) > 0:
        minSep[yrealImagInds[smm1Inds]] = smm1[smm1Inds]
        minSepPoints_x[yrealImagInds[smm1Inds]] = np.real(xreal[yrealImagInds[smm1Inds],1])
        minSepPoints_y[yrealImagInds[smm1Inds]] = np.real(yreal[yrealImagInds[smm1Inds],1])
    if len(smp0Inds) > 0:
        minSep[yrealImagInds[smp0Inds]] = smp0[smp0Inds]
        minSepPoints_x[yrealImagInds[smp0Inds]] = np.real(xreal[yrealImagInds[smp0Inds],0])
        minSepPoints_y[yrealImagInds[smp0Inds]] = np.real(yreal[yrealImagInds[smp0Inds],0])
    if len(spm1Inds) > 0:
        minSep[yrealImagInds[spm1Inds]] = smp0[spm1Inds]
        minSepPoints_x[yrealImagInds[spm1Inds]] = np.real(xreal[yrealImagInds[spm1Inds],1])
        minSepPoints_y[yrealImagInds[spm1Inds]] = np.real(yreal[yrealImagInds[spm1Inds],1])
    if len(spp1Inds) > 0:
        minSep[yrealImagInds[spp1Inds]] = spp1[spp1Inds]
        minSepPoints_x[yrealImagInds[spp1Inds]] = np.real(xreal[yrealImagInds[spp1Inds],1])
        minSepPoints_y[yrealImagInds[spp1Inds]] = np.real(yreal[yrealImagInds[spp1Inds],1])
    #above says largest must be one of these: smm0, smp0, spm1, spp1
    smm0Inds = np.where((smm0 > smp0)*(smm0 > spm1)*(smm0 > spp0))[0]
    smp0Inds = np.where((smp0 > smm0)*(smp0 > spm1)*(smp0 > spp0))[0]
    spm1Inds = np.where((spm1 > smp0)*(spm1 > smm0)*(spm1 > spp0))[0]
    spp0Inds = np.where((spp0 > smp0)*(spp0 > spm1)*(spp0 > smm0))[0]
    if len(smm0Inds) > 0:
        maxSep[yrealImagInds[smm0Inds]] = smm0[smm0Inds]
        maxSepPoints_x[yrealImagInds[smm0Inds]] = np.real(xreal[yrealImagInds[smm0Inds],0])
        maxSepPoints_y[yrealImagInds[smm0Inds]] = np.real(yreal[yrealImagInds[smm0Inds],0])
    if len(smp0Inds) > 0:
        maxSep[yrealImagInds[smp0Inds]] = smp0[smp0Inds]
        maxSepPoints_x[yrealImagInds[smp0Inds]] = np.real(xreal[yrealImagInds[smp0Inds],0])
        maxSepPoints_y[yrealImagInds[smp0Inds]] = np.real(yreal[yrealImagInds[smp0Inds],0])
    if len(spm1Inds) > 0:
        maxSep[yrealImagInds[spm1Inds]] = spm1[spm1Inds]
        maxSepPoints_x[yrealImagInds[spm1Inds]] = np.real(xreal[yrealImagInds[spm1Inds],1])
        maxSepPoints_y[yrealImagInds[spm1Inds]] = np.real(yreal[yrealImagInds[spm1Inds],1])
    if len(spp0Inds) > 0:
        maxSep[yrealImagInds[spp0Inds]] = spp0[spp0Inds]
        maxSepPoints_x[yrealImagInds[spp0Inds]] = np.real(xreal[yrealImagInds[spp0Inds],0])
        maxSepPoints_y[yrealImagInds[spp0Inds]] = np.real(yreal[yrealImagInds[spp0Inds],0])

    #not currentyl assigning x,y values or lmin lmax for 2 solutions with 2 complex
    ########################################################
    #### 4 Real Solutions ##################################
    smm = np.zeros((4,len(yrealAllRealInds)))
    smp = np.zeros((4,len(yrealAllRealInds)))
    #spm = np.zeros((4,len(yrealAllRealInds))) #removed for efficiency
    spp = np.zeros((4,len(yrealAllRealInds)))
    for i in [0,1,2,3]:
        smm[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))-my[yrealAllRealInds])**2)
        smp[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))+my[yrealAllRealInds])**2)
        #spm[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])+mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))-my[yrealAllRealInds])**2) #removed for efficiency
        spp[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])+mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))+my[yrealAllRealInds])**2)
    smm = smm.T
    smp = smp.T
    #spm = spm.T #removed for efficiency
    spp = spp.T



    #KEEP
    #### minSep
    minSep[yrealAllRealInds] = smm[:,1]
    minSepPoints_x[yrealAllRealInds] = xreal[yrealAllRealInds,1]
    minSepPoints_y[yrealAllRealInds] = yreal[yrealAllRealInds,1]
    ####

    #KEEP
    #### maxSep
    maxSep[yrealAllRealInds] = spp[:,0]
    maxSepPoints_x[yrealAllRealInds] = xreal[yrealAllRealInds,0]
    maxSepPoints_y[yrealAllRealInds] = yreal[yrealAllRealInds,0]
    ####

    #KEEP
    smp2 = np.sqrt((np.real(xreal[yrealAllRealInds,2])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,2]))+my[yrealAllRealInds])**2)
    smp3 = np.sqrt((np.real(xreal[yrealAllRealInds,3])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,3]))+my[yrealAllRealInds])**2)
    smp = np.asarray([smp2,smp3]).T
    #### slmax
    slmaxInds = np.argmax(smp,axis=1)
    lmaxSep = smp[np.arange(len(yrealAllRealInds)),slmaxInds]
    lmaxSepPoints_x = xreal[yrealAllRealInds, 2+slmaxInds]
    lmaxSepPoints_y = yreal[yrealAllRealInds, 2+slmaxInds]
    #### slmin
    slminInds = np.argmin(smp,axis=1)
    lminSep = smp[np.arange(len(yrealAllRealInds)),slminInds]
    lminSepPoints_x = xreal[yrealAllRealInds, 2+slminInds]
    lminSepPoints_y = yreal[yrealAllRealInds, 2+slminInds]
    ####

    #Comment
    #Checks on minSep <= lminSep <= lmaxSep <= maxSep
    assert ~np.any(minSep == 0), 'Oops, a minSep was missed'
    assert ~np.any(maxSep == 0), 'Oops, a maxSep was missed'
    assert np.all(minSep[yrealAllRealInds] <= lminSep), 'Not all minSep < lminSep'
    assert np.all(maxSep[yrealAllRealInds] >= lmaxSep), 'Not all maxSep > lmaxSep'
    assert np.all(lminSep <= lmaxSep), 'Not all lminSep < lmaxSep'
    ################################################################################################# Done with separations

    #KEEP
    #Quadrant Star Belongs to
    bool1 = x > 0
    bool2 = y > 0
    #Quadrant 1 if T,T
    #Quadrant 2 if F,T
    #Quadrant 3 if F,F
    #Quadrant 4 if T,F
    #### Min Sep Point (Points on plot of Min Sep)
    minSepPoints_x = minSepPoints_x*(2*bool1-1)
    minSepPoints_y = minSepPoints_y*(2*bool2-1)
    #### Max Sep Point (Points on plot of max sep)
    maxSepPoints_x = maxSepPoints_x*(-2*bool1+1)
    maxSepPoints_y = maxSepPoints_y*(-2*bool2+1)
    #### Local Min Sep Points
    lminSepPoints_x = lminSepPoints_x*(2*bool1[yrealAllRealInds]-1)
    lminSepPoints_y = lminSepPoints_y*(-2*bool2[yrealAllRealInds]+1)
    #### Local Max Sep Points
    lmaxSepPoints_x = lmaxSepPoints_x*(2*bool1[yrealAllRealInds]-1)
    lmaxSepPoints_y = lmaxSepPoints_y*(-2*bool2[yrealAllRealInds]+1)

    return minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds

def trueAnomalyFromXY(X,Y,W,w,inc):
    """ Calculated true anomaly from X, Y, and KOE
    Args:
        X (numpy array): x component of body in 3D elliptical orbit
        Y (numpy array): y component of body in 3D elliptical orbit
        W (numpy array): Longitude of the ascending node of the body
        w (numpy array): argument of periapsis of the body
        inc (numpy array): inclination of the body's orbit
    Returns:
        nu (numpy array):
    """
    nu = np.arctan2( -X/Y*np.sin(W)*np.cos(w) -X/Y*np.cos(W)*np.cos(inc)*np.sin(w) + np.cos(W)*np.cos(w) - np.sin(W)*np.cos(inc)*np.cos(w),\
                -X/Y*np.sin(W)*np.sin(w) + X/Y*np.cos(W)*np.cos(inc)*np.cos(w) + np.cos(W)*np.sin(w) + np.sin(W)*np.cos(inc)*np.cos(w) )
    return nu

