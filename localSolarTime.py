
import numpy as np

def SolarTime_from_dtUTCLon(dUTC,LT,lon,d):
    """
    dUTC in hours
    LT in hours
    """
    LSTM = 15.*dUTC #LSTM is in degrees
    B = 360./365.*(d+dUTC/60.-81.)*np.pi/180.
    EoT = 9.87*np.sin(2.*B) - 7.53*np.cos(B) - 1.5*np.sin(B) #in minutes
    TC = 4.*(lon-LSTM) + EoT #TC is in minutes
    LST = LT + TC/60. #is in hours
    return LST, LT, TC, EoT, B #local solar time

# #localSolarTime
# dUTC = 10.
# d = 5.
# lon = 150.
# LT = 12.5

# ds = np.linspace(start=0.,stop=365.,num=1000)
# Bs = (ds-81.)*360./365.*np.pi/180.
# EoTs = 9.87*np.sin(2.*Bs) - 7.53*np.cos(Bs) - 1.5*np.sin(Bs)
# print(EoTs)
# B = 0.09472139
# EoT = 9.87*np.sin(2.*B) - 7.53*np.cos(B) - 1.5*np.sin(B)
# print(Bs)

# LST, LT, TC, EoT, B = SolarTime_from_dtUTCLon(dUTC,LT,lon,d)
# print(LST)
# print(LT)
# print(TC)
# print(EoT)
# print(B)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(ds, EoTs)
# plt.show(block=False)

