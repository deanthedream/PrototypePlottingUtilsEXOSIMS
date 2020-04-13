####
import numpy as np

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
    #DELETEdmajorp = np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
    #     + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #DELETEdel Ramgy, Affinity1, Yolo1, Kolko1

    #Specific Calc Substitutions
    #DELETE Ramgy2 = (e + 1)*np.sqrt(phi*(gamma)**2/Eeps + phi*(np.sin(W)*np.sin(w)*np.cos(inc) - np.cos(W)*np.cos(w))**2/Eeps + phi*np.sin(inc)**2*np.sin(w)**2/Eeps)
    #DELETEAffinity2 = a**2*Gamma*(gamma)/Ramgy2
    #DELETEYolo2 = np.sin(W)*np.cos(w + np.pi/2) + np.sin(w + np.pi/2)*np.cos(W)*np.cos(inc)
    #DELETEKolko2 = -np.sin(W)*np.sin(w + np.pi/2)*np.cos(inc) + np.cos(W)*np.cos(w + np.pi/2)
    #Semi-minor axis length
    dminorp = -np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)/2
    #DELETEdminorp = -np.sqrt(np.abs(Affinity2 + Eeps2*Yolo2/(1 - e) - (a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)*np.sqrt(np.abs(Affinity2 + Eeps2*(Omicron)/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*Kolko2/(1 - e) - (-Affinity2 - Eeps2*(Omicron)/Gemini)*np.sqrt(np.abs(Affinity2 + Eeps2*(Omicron)/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)**2))**2)/2\
    #     + np.sqrt(np.abs(Affinity2 + Eeps2*Yolo2/(1 - e) + (a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)*np.sqrt(np.abs(Affinity2 + Eeps2*(Omicron)/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)**2))**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*Kolko2/(1 - e) + (-Affinity2 - Eeps2*(Omicron)/Gemini)*np.sqrt(np.abs(Affinity2 + Eeps2*(Omicron)/Gemini)**2 + np.abs(a**2*Gamma*(lam2)/Ramgy2 + Eeps2*(Gorgon)/Gemini)**2))**2)/2
    #DELETEdel Ramgy2, Affinity2, Yolo2, Kolko2

    #Angle between OpQ and OpQp
    Psi = np.arccos(((Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)) + (a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
             - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))*(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)))/(np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1\
             + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)*np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2\
             + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2)))

    #Angle between X-axis and Op!
    psi = np.arccos((a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))/np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e)\
         - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*Gorgon/Gemini)**2))**2))

    #theta_OpQ_X
    theta_OpQ_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1 - e) - (a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2\
         + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2), a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) - (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1\
          - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2))

    #theta_OpQp_X
    theta_OpQp_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1 - e) + (a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy\
         + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2), a**2*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1 - e) + (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi)) + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2*np.arctan(Phi))\
          + np.sin(w + 2*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2*np.arctan(Phi)) + 1))**2 + np.abs(a**2*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2*np.arctan(Phi)))/(e*np.cos(2*np.arctan(Phi)) + 1))**2))

    return dmajorp, dminorp, Psi, psi, theta_OpQ_X, theta_OpQp_X

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
