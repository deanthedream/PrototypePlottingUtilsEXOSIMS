# Testing data volume and speed of cubic spline and RBS interpolators

import numpy as np
from scipy.interpolate import CubicSpline as CubicSpline
from scipy.interpolate import RectBivariateSpline as RBS
from EXOSIMS.util.phaseFunctions import quasiLambertPhaseFunction
import sys
import psutil
import itertools
import time
from scipy.signal import argrelextrema
import os
try:
    import cPickle as pickle
except:
    import pickle
from itertools import combinations
from scipy.optimize import minimize

w = np.asarray(np.linspace(start=0.,stop=2.*np.pi,num=5))
inc = np.linspace(start=0.,stop=np.pi/2.,num=5)
v = np.linspace(start=0.,stop=2.*np.pi,num=30)
e = np.linspace(start=0.,stop=0.9,num=5)


#beta = np.arccos(np.sin(inc)*np.sin(v+w))

#The Left hand side of the expression
#lhs = 10.**(-0.4*dMag)*a**2.*(1.+e**2.)**2./p/Rp**2.


def rhs(inc,v,w,e):
    """
    The right hand side containing the variables of the planet dmag expression 
    """
    beta = np.arccos(np.sin(inc)*np.sin(v+w))
    rhs = quasiLambertPhaseFunction(beta)*(e*np.cos(v)+1.)**2.
    return rhs

# #### Calculate CubicSpline
# #Get initial memory usage
# initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
# t0 = time.time()
# #cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

# cbss = dict()
# for i,j,k in itertools.product(np.arange(len(inc)),np.arange(len(w)),np.arange(len(e))):
#     cbss[(i,j,k)] = CubicSpline(v,rhs(inc[i],v,w[j],e[k]))


# finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
# t1 = time.time()
# usedMemory = finalMemory - initMemory
# usedTime = t1-t0

# print('For CubicSpline')
# print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
# print('Memory Used Per cbs (MB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e))*10.**3.))
# print('Time Used (s): ' + str(usedTime))
# print('Time Used Per cbs (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
# ####



# #### Calculate InterpolationGrid
# #Get initial memory usage
cachedir = './'
dfilename = 'interpgrid' + '_' + str(len(w)) + '_' + str(len(inc)) + '_' + str(len(v)) + '_' + str(len(e))+ '.pkl'
path = os.path.join(cachedir, dfilename)
# if the 2D completeness update array exists as a .dcomp file load it
if os.path.exists(path):
    print('Loading cached Interpolation Grid array from "%s".' % path)
    try:
        with open(path, "rb") as ff:
            interpArray = pickle.load(ff)
    except UnicodeDecodeError:
        with open(path, "rb") as ff:
            interpArray = pickle.load(ff,encoding='latin1')
    print('Interpolation Grid array loaded from cache.')
else:
    initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
    t0 = time.time()
    #cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

    interpArray = np.zeros((len(w),len(inc),len(e),len(v)))
    for i,j,k,l in itertools.product(np.arange(len(inc)),np.arange(len(w)),np.arange(len(e)),np.arange(len(v))):
        interpArray[i,j,k,l] = rhs(inc[i],v[l],w[j],e[k])


    finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
    t1 = time.time()
    usedMemory = finalMemory - initMemory
    usedTime = t1-t0

    # store dynamic completeness array as .dcomp file
    with open(path, 'wb') as ff:
        pickle.dump(interpArray, ff)
    print('Interpolation Grid calculations finished')
    print('Interpolation Grid array stored in %r' % path)

    print('For 4D InterpolationGrid Data')
    print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
    print('Memory Used Per cbs equivalent (GB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e))))
    print('Time Used (s): ' + str(usedTime))
    print('Time Used Per cbs equivalent (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
####



#### Calculate RBS
#Get initial memory usage
initMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t0 = time.time()
#cbs = CubicSpline(v,rhs(inc[0],v,w[0],e[0]))

rbsDict = dict()
extremaDict = dict()
extremaDict['min'] = dict() 
extremaDict['lmin'] = dict() 
extremaDict['lmax'] = dict() 
extremaDict['max'] = dict() 
for i,j in itertools.product(np.arange(len(inc)),np.arange(len(w))):
    #rhsArray = np.zeros((len(e),len(yarray)))
    #for k,l in itertools.product(np.arange(len(v)),np.arange(len(e))):
    #    rhsArray[k,l] = rhs(inc[i],v,w[j],e[l])
    rhsArray = np.zeros((len(e),len(v)))
    #for k,l in itertools.product(np.arange(len(e)),np.arange(len(v))):
    for k in np.arange(len(e)):
        rhsArray[k] = rhs(inc[i],v,w[j],e[k])
        lmaxInd = None
        lminInd = None
        arglmax = argrelextrema(rhsArray[k], np.greater)[0] #gives local max
        if len(arglmax) == 1:
            maxInd = np.argmax(rhsArray[k])
            lmax = np.nan
        elif len(arglmax) > 1: #there is more than one local maximum
            maxInd = np.argmax(rhsArray[k]) #ind where maximum occurs
            lmaxInd = list(arglmax).remove(maxInd)
            lmax = rhsArray[lmaxInd]
        elif len(arglmax) == 0: #all values are the same
            lmax = np.nan
            maxInd = 0
        else:
            print(error)
        arglmin = argrelextrema(rhsArray[k], np.less)[0] #gives local min
        if len(arglmin) == 1:
            minInd = np.argmin(rhsArray[k])
            lmin = np.nan
        elif len(arglmin) > 1: #there is more than one local minimum
            minInd = np.argmin(rhsArray[k]) #ind where minimum occurs
            lminInd = list(arglmin).remove(minInd)
            lmin = rhsArray[lminInd]
        elif len(arglmin) == 0: #all values are the same
            lmin = np.nan
            minInd = 0
        else:
            print(error)
        extremaDict['min'][(i,j,k)] = np.min(rhsArray[k,maxInd])
        extremaDict['lmin'][(i,j,k)] = lmin
        extremaDict['lmax'][(i,j,k)] = lmax
        extremaDict['max'][(i,j,k)] = np.max(rhsArray[k,minInd])
    rbsDict[(i,j)] = RBS(e,v,rhsArray)

    #Finds min, max, lmin, lmax of RBS vs nu at specific e
    
    


finalMemory = dict(psutil.virtual_memory()._asdict())['used']/(1024.0 ** 3.)
t1 = time.time()
usedMemory = finalMemory - initMemory
usedTime = t1-t0

print('For RBS')
print('Memory Used (GB): ' + str(usedMemory) + ' for ' + str(len(w)*len(inc)*len(e)) + ' points')
print('Memory Used Per vbs equivalent (MB/cbs): ' + str(usedMemory/(len(w)*len(inc)*len(e)*10.**3.)))
print('Time Used (s): ' + str(usedTime))
print('Time Used Per cbs equivalent (): ' + str(usedTime/(len(w)*len(inc)*len(e))))
####


#we need to calculate the total number of variables we will encounter for a quartic polynomial of nu
# A Python program to print all combinations of given length with unsorted input. 

comb1 = combinations([0, 1, 2, 3], 1) 
comb2 = combinations([0, 1, 2, 3], 2)
comb3 = combinations([0, 1, 2, 3], 3)
comb4 = combinations([0, 1, 2, 3], 4) 

#Total number is
noVars = 1 #the constant
oneVars = len(list(comb1))*4 #1 term mixed 4=4**1
twoVars = len(list(comb2))*16 #2 terms mixed 16=4**2
threeVars = len(list(comb3))*64 #3 terms mixed 64=4**3
fourVars = len(list(comb4))*256 #all 4 terms mixed 256=4**4
totalCoeffs = noVars + oneVars + twoVars + threeVars + fourVars #total number of coeffs to create



def rhs_model(x,inc,nu,omega,e):
    """ This is the right hand equation model
    """
    ####
    rhs = x[0]+x[1]*omega+x[2]*omega**2+x[3]*omega**3+x[4]*omega**4+\
    x[5]*inc+x[6]*inc**2+x[7]*inc**3+x[8]*inc**4+x[9]*nu+x[10]*nu**2+x[11]*nu**3+\
    x[12]*nu**4+x[13]*e+x[14]*e**2+x[15]*e**3+x[16]*e**4+x[17]+x[18]*inc+x[19]*inc**2+\
    x[20]*inc**3+x[21]*omega+x[22]*inc*omega+x[23]*inc**2*omega+x[24]*inc**3*omega+\
    x[25]*omega**2+x[26]*inc*omega**2+x[27]*inc**2*omega**2+x[28]*inc**3*omega**2+x[29]*omega**3+\
    x[30]*inc*omega**3+x[31]*inc**2*omega**3+x[32]*inc**3*omega**3+x[33]+x[34]*nu+x[35]*nu**2+\
    x[36]*nu**3+x[37]*omega+x[38]*nu*omega+x[39]*nu**2*omega+x[40]*nu**3*omega+x[41]*omega**2+\
    x[42]*nu*omega**2+x[43]*nu**2*omega**2+x[44]*nu**3*omega**2+x[45]*omega**3+x[46]*nu*omega**3+\
    x[47]*nu**2*omega**3+x[48]*nu**3*omega**3+x[49]+x[50]*e+x[51]*e**2+x[52]*e**3+x[53]*omega+x[54]*e*omega+\
    x[55]*e**2*omega+x[56]*e**3*omega+x[57]*omega**2+x[58]*e*omega**2+x[59]*e**2*omega**2+x[60]*e**3*omega**2+\
    x[61]*omega**3+x[62]*e*omega**3+x[63]*e**2*omega**3+x[64]*e**3*omega**3+x[65]+x[66]*nu+x[67]*nu**2+x[68]*nu**3+\
    x[69]*inc+x[70]*inc*nu+x[71]*inc*nu**2+x[72]*inc*nu**3+x[73]*inc**2+x[74]*inc**2*nu+x[75]*inc**2*nu**2+\
    x[76]*inc**2*nu**3+x[77]*inc**3+x[78]*inc**3*nu+x[79]*inc**3*nu**2+x[80]*inc**3*nu**3+x[81]+x[82]*e+\
    x[83]*e**2+x[84]*e**3+x[85]*inc+x[86]*e*inc+x[87]*e**2*inc+x[88]*e**3*inc+x[89]*inc**2+x[90]*e*inc**2+\
    x[91]*e**2*inc**2+x[92]*e**3*inc**2+x[93]*inc**3+x[94]*e*inc**3+x[95]*e**2*inc**3+x[96]*e**3*inc**3+x[97]+\
    x[98]*e+x[99]*e**2+x[100]*e**3+x[101]*nu+x[102]*e*nu+x[103]*e**2*nu+x[104]*e**3*nu+x[105]*nu**2+x[106]*e*nu**2+\
    x[107]*e**2*nu**2+x[108]*e**3*nu**2+x[109]*nu**3+x[110]*e*nu**3+x[111]*e**2*nu**3+x[112]*e**3*nu**3+x[113]+x[114]*nu+\
    x[115]*nu**2+x[116]*nu**3+x[117]*inc+x[118]*inc*nu+x[119]*inc*nu**2+x[120]*inc*nu**3+x[121]*inc**2+x[122]*inc**2*nu+\
    x[123]*inc**2*nu**2+x[124]*inc**2*nu**3+x[125]*inc**3+x[126]*inc**3*nu+x[127]*inc**3*nu**2+x[128]*inc**3*nu**3+x[129]*omega+\
    x[130]*nu*omega+x[131]*nu**2*omega+x[132]*nu**3*omega+x[133]*inc*omega+x[134]*inc*nu*omega+x[135]*inc*nu**2*omega+\
    x[136]*inc*nu**3*omega+x[137]*inc**2*omega+x[138]*inc**2*nu*omega+x[139]*inc**2*nu**2*omega+x[140]*inc**2*nu**3*omega+\
    x[141]*inc**3*omega+x[142]*inc**3*nu*omega+x[143]*inc**3*nu**2*omega+x[144]*inc**3*nu**3*omega+x[145]*omega**2+\
    x[146]*nu*omega**2+x[147]*nu**2*omega**2+x[148]*nu**3*omega**2+x[149]*inc*omega**2+x[150]*inc*nu*omega**2+\
    x[151]*inc*nu**2*omega**2+x[152]*inc*nu**3*omega**2+x[153]*inc**2*omega**2+x[154]*inc**2*nu*omega**2+\
    x[155]*inc**2*nu**2*omega**2+x[156]*inc**2*nu**3*omega**2+x[157]*inc**3*omega**2+x[158]*inc**3*nu*omega**2+\
    x[159]*inc**3*nu**2*omega**2+x[160]*inc**3*nu**3*omega**2+x[161]*omega**3+x[162]*nu*omega**3+x[163]*nu**2*omega**3+\
    x[164]*nu**3*omega**3+x[165]*inc*omega**3+x[166]*inc*nu*omega**3+x[167]*inc*nu**2*omega**3+x[168]*inc*nu**3*omega**3+\
    x[169]*inc**2*omega**3+x[170]*inc**2*nu*omega**3+x[171]*inc**2*nu**2*omega**3+x[172]*inc**2*nu**3*omega**3+\
    x[173]*inc**3*omega**3+x[174]*inc**3*nu*omega**3+x[175]*inc**3*nu**2*omega**3+x[176]*inc**3*nu**3*omega**3+\
    x[177]+x[178]*e+x[179]*e**2+x[180]*e**3+x[181]*inc+x[182]*e*inc+x[183]*e**2*inc+x[184]*e**3*inc+x[185]*inc**2+\
    x[186]*e*inc**2+x[187]*e**2*inc**2+x[188]*e**3*inc**2+x[189]*inc**3+x[190]*e*inc**3+x[191]*e**2*inc**3+\
    x[192]*e**3*inc**3+x[193]*omega+x[194]*e*omega+x[195]*e**2*omega+x[196]*e**3*omega+x[197]*inc*omega+x[198]*e*inc*omega+\
    x[199]*e**2*inc*omega+x[200]*e**3*inc*omega+x[201]*inc**2*omega+x[202]*e*inc**2*omega+x[203]*e**2*inc**2*omega+\
    x[204]*e**3*inc**2*omega+x[205]*inc**3*omega+x[206]*e*inc**3*omega+x[207]*e**2*inc**3*omega+x[208]*e**3*inc**3*omega+\
    x[209]*omega**2+x[210]*e*omega**2+x[211]*e**2*omega**2+x[212]*e**3*omega**2+x[213]*inc*omega**2+x[214]*e*inc*omega**2+\
    x[215]*e**2*inc*omega**2+x[216]*e**3*inc*omega**2+x[217]*inc**2*omega**2+x[218]*e*inc**2*omega**2+x[219]*e**2*inc**2*omega**2+\
    x[220]*e**3*inc**2*omega**2+x[221]*inc**3*omega**2+x[222]*e*inc**3*omega**2+x[223]*e**2*inc**3*omega**2+\
    x[224]*e**3*inc**3*omega**2+x[225]*omega**3+x[226]*e*omega**3+x[227]*e**2*omega**3+x[228]*e**3*omega**3+x[229]*inc*omega**3+\
    x[230]*e*inc*omega**3+x[231]*e**2*inc*omega**3+x[232]*e**3*inc*omega**3+x[233]*inc**2*omega**3+x[234]*e*inc**2*omega**3+\
    x[235]*e**2*inc**2*omega**3+x[236]*e**3*inc**2*omega**3+x[237]*inc**3*omega**3+x[238]*e*inc**3*omega**3+\
    x[239]*e**2*inc**3*omega**3+x[240]*e**3*inc**3*omega**3+x[241]+x[242]*e+x[243]*e**2+x[244]*e**3+x[245]*nu+\
    x[246]*e*nu+x[247]*e**2*nu+x[248]*e**3*nu+x[249]*nu**2+x[250]*e*nu**2+x[251]*e**2*nu**2+x[252]*e**3*nu**2+\
    x[253]*nu**3+x[254]*e*nu**3+x[255]*e**2*nu**3+x[256]*e**3*nu**3+x[257]*omega+x[258]*e*omega+x[259]*e**2*omega+\
    x[260]*e**3*omega+x[261]*nu*omega+x[262]*e*nu*omega+x[263]*e**2*nu*omega+x[264]*e**3*nu*omega+x[265]*nu**2*omega+\
    x[266]*e*nu**2*omega+x[267]*e**2*nu**2*omega+x[268]*e**3*nu**2*omega+x[269]*nu**3*omega+x[270]*e*nu**3*omega+\
    x[271]*e**2*nu**3*omega+x[272]*e**3*nu**3*omega+x[273]*omega**2+x[274]*e*omega**2+x[275]*e**2*omega**2+\
    x[276]*e**3*omega**2+x[277]*nu*omega**2+x[278]*e*nu*omega**2+x[279]*e**2*nu*omega**2+\
    x[280]*e**3*nu*omega**2+x[281]*nu**2*omega**2+x[282]*e*nu**2*omega**2+x[283]*e**2*nu**2*omega**2+\
    x[284]*e**3*nu**2*omega**2+x[285]*nu**3*omega**2+x[286]*e*nu**3*omega**2+x[287]*e**2*nu**3*omega**2+\
    x[288]*e**3*nu**3*omega**2+x[289]*omega**3+x[290]*e*omega**3+x[291]*e**2*omega**3+x[292]*e**3*omega**3+\
    x[293]*nu*omega**3+x[294]*e*nu*omega**3+x[295]*e**2*nu*omega**3+x[296]*e**3*nu*omega**3+\
    x[297]*nu**2*omega**3+x[298]*e*nu**2*omega**3+x[299]*e**2*nu**2*omega**3+x[300]*e**3*nu**2*omega**3+\
    x[301]*nu**3*omega**3+x[302]*e*nu**3*omega**3+x[303]*e**2*nu**3*omega**3+x[304]*e**3*nu**3*omega**3+\
    x[305]+x[306]*e+x[307]*e**2+x[308]*e**3+x[309]*nu+x[310]*e*nu+x[311]*e**2*nu+x[312]*e**3*nu+x[313]*nu**2+\
    x[314]*e*nu**2+x[315]*e**2*nu**2+x[316]*e**3*nu**2+x[317]*nu**3+x[318]*e*nu**3+x[319]*e**2*nu**3+\
    x[320]*e**3*nu**3+x[321]*inc+x[322]*e*inc+x[323]*e**2*inc+x[324]*e**3*inc+x[325]*inc*nu+x[326]*e*inc*nu+\
    x[327]*e**2*inc*nu+x[328]*e**3*inc*nu+x[329]*inc*nu**2+x[330]*e*inc*nu**2+x[331]*e**2*inc*nu**2+\
    x[332]*e**3*inc*nu**2+x[333]*inc*nu**3+x[334]*e*inc*nu**3+x[335]*e**2*inc*nu**3+x[336]*e**3*inc*nu**3+\
    x[337]*inc**2+x[338]*e*inc**2+x[339]*e**2*inc**2+x[340]*e**3*inc**2+x[341]*inc**2*nu+x[342]*e*inc**2*nu+\
    x[343]*e**2*inc**2*nu+x[344]*e**3*inc**2*nu+x[345]*inc**2*nu**2+x[346]*e*inc**2*nu**2+\
    x[347]*e**2*inc**2*nu**2+x[348]*e**3*inc**2*nu**2+x[349]*inc**2*nu**3+\
    x[350]*e*inc**2*nu**3+x[351]*e**2*inc**2*nu**3+x[352]*e**3*inc**2*nu**3+\
    x[353]*inc**3+x[354]*e*inc**3+x[355]*e**2*inc**3+x[356]*e**3*inc**3+x[357]*inc**3*nu+\
    x[358]*e*inc**3*nu+x[359]*e**2*inc**3*nu+x[360]*e**3*inc**3*nu+x[361]*inc**3*nu**2+\
    x[362]*e*inc**3*nu**2+x[363]*e**2*inc**3*nu**2+x[364]*e**3*inc**3*nu**2+x[365]*inc**3*nu**3+\
    x[366]*e*inc**3*nu**3+x[367]*e**2*inc**3*nu**3+x[368]*e**3*inc**3*nu**3+x[369]+x[370]*e+\
    x[371]*e**2+x[372]*e**3+x[373]*nu+x[374]*e*nu+x[375]*e**2*nu+x[376]*e**3*nu+x[377]*nu**2+\
    x[378]*e*nu**2+x[379]*e**2*nu**2+x[380]*e**3*nu**2+x[381]*nu**3+x[382]*e*nu**3+x[383]*e**2*nu**3+\
    x[384]*e**3*nu**3+x[385]*inc+x[386]*e*inc+x[387]*e**2*inc+x[388]*e**3*inc+x[389]*inc*nu+\
    x[390]*e*inc*nu+x[391]*e**2*inc*nu+x[392]*e**3*inc*nu+x[393]*inc*nu**2+x[394]*e*inc*nu**2+\
    x[395]*e**2*inc*nu**2+x[396]*e**3*inc*nu**2+x[397]*inc*nu**3+x[398]*e*inc*nu**3+\
    x[399]*e**2*inc*nu**3+x[400]*e**3*inc*nu**3+x[401]*inc**2+x[402]*e*inc**2+\
    x[403]*e**2*inc**2+x[404]*e**3*inc**2+x[405]*inc**2*nu+x[406]*e*inc**2*nu+\
    x[407]*e**2*inc**2*nu+x[408]*e**3*inc**2*nu+x[409]*inc**2*nu**2+x[410]*e*inc**2*nu**2+\
    x[411]*e**2*inc**2*nu**2+x[412]*e**3*inc**2*nu**2+x[413]*inc**2*nu**3+\
    x[414]*e*inc**2*nu**3+x[415]*e**2*inc**2*nu**3+x[416]*e**3*inc**2*nu**3+\
    x[417]*inc**3+x[418]*e*inc**3+x[419]*e**2*inc**3+x[420]*e**3*inc**3+\
    x[421]*inc**3*nu+x[422]*e*inc**3*nu+x[423]*e**2*inc**3*nu+x[424]*e**3*inc**3*nu+\
    x[425]*inc**3*nu**2+x[426]*e*inc**3*nu**2+x[427]*e**2*inc**3*nu**2+x[428]*e**3*inc**3*nu**2+\
    x[429]*inc**3*nu**3+x[430]*e*inc**3*nu**3+x[431]*e**2*inc**3*nu**3+x[432]*e**3*inc**3*nu**3+\
    x[433]*omega+x[434]*e*omega+x[435]*e**2*omega+x[436]*e**3*omega+x[437]*nu*omega+x[438]*e*nu*omega+\
    x[439]*e**2*nu*omega+x[440]*e**3*nu*omega+x[441]*nu**2*omega+x[442]*e*nu**2*omega+x[443]*e**2*nu**2*omega+\
    x[444]*e**3*nu**2*omega+x[445]*nu**3*omega+x[446]*e*nu**3*omega+x[447]*e**2*nu**3*omega+\
    x[448]*e**3*nu**3*omega+x[449]*inc*omega+x[450]*e*inc*omega+x[451]*e**2*inc*omega+\
    x[452]*e**3*inc*omega+x[453]*inc*nu*omega+x[454]*e*inc*nu*omega+x[455]*e**2*inc*nu*omega+\
    x[456]*e**3*inc*nu*omega+x[457]*inc*nu**2*omega+x[458]*e*inc*nu**2*omega+x[459]*e**2*inc*nu**2*omega+\
    x[460]*e**3*inc*nu**2*omega+x[461]*inc*nu**3*omega+x[462]*e*inc*nu**3*omega+x[463]*e**2*inc*nu**3*omega+\
    x[464]*e**3*inc*nu**3*omega+x[465]*inc**2*omega+x[466]*e*inc**2*omega+x[467]*e**2*inc**2*omega+\
    x[468]*e**3*inc**2*omega+x[469]*inc**2*nu*omega+x[470]*e*inc**2*nu*omega+\
    x[471]*e**2*inc**2*nu*omega+x[472]*e**3*inc**2*nu*omega+x[473]*inc**2*nu**2*omega+\
    x[474]*e*inc**2*nu**2*omega+x[475]*e**2*inc**2*nu**2*omega+x[476]*e**3*inc**2*nu**2*omega+\
    x[477]*inc**2*nu**3*omega+x[478]*e*inc**2*nu**3*omega+x[479]*e**2*inc**2*nu**3*omega+\
    x[480]*e**3*inc**2*nu**3*omega+x[481]*inc**3*omega+x[482]*e*inc**3*omega+x[483]*e**2*inc**3*omega+\
    x[484]*e**3*inc**3*omega+x[485]*inc**3*nu*omega+x[486]*e*inc**3*nu*omega+x[487]*e**2*inc**3*nu*omega+\
    x[488]*e**3*inc**3*nu*omega+x[489]*inc**3*nu**2*omega+x[490]*e*inc**3*nu**2*omega+\
    x[491]*e**2*inc**3*nu**2*omega+x[492]*e**3*inc**3*nu**2*omega+x[493]*inc**3*nu**3*omega+\
    x[494]*e*inc**3*nu**3*omega+x[495]*e**2*inc**3*nu**3*omega+x[496]*e**3*inc**3*nu**3*omega+\
    x[497]*omega**2+x[498]*e*omega**2+x[499]*e**2*omega**2+x[500]*e**3*omega**2+x[501]*nu*omega**2+\
    x[502]*e*nu*omega**2+x[503]*e**2*nu*omega**2+x[504]*e**3*nu*omega**2+x[505]*nu**2*omega**2+\
    x[506]*e*nu**2*omega**2+x[507]*e**2*nu**2*omega**2+x[508]*e**3*nu**2*omega**2+x[509]*nu**3*omega**2+\
    x[510]*e*nu**3*omega**2+x[511]*e**2*nu**3*omega**2+x[512]*e**3*nu**3*omega**2+x[513]*inc*omega**2+\
    x[514]*e*inc*omega**2+x[515]*e**2*inc*omega**2+x[516]*e**3*inc*omega**2+x[517]*inc*nu*omega**2+\
    x[518]*e*inc*nu*omega**2+x[519]*e**2*inc*nu*omega**2+x[520]*e**3*inc*nu*omega**2+\
    x[521]*inc*nu**2*omega**2+x[522]*e*inc*nu**2*omega**2+x[523]*e**2*inc*nu**2*omega**2+\
    x[524]*e**3*inc*nu**2*omega**2+x[525]*inc*nu**3*omega**2+x[526]*e*inc*nu**3*omega**2+\
    x[527]*e**2*inc*nu**3*omega**2+x[528]*e**3*inc*nu**3*omega**2+x[529]*inc**2*omega**2+\
    x[530]*e*inc**2*omega**2+x[531]*e**2*inc**2*omega**2+x[532]*e**3*inc**2*omega**2+\
    x[533]*inc**2*nu*omega**2+x[534]*e*inc**2*nu*omega**2+x[535]*e**2*inc**2*nu*omega**2+\
    x[536]*e**3*inc**2*nu*omega**2+x[537]*inc**2*nu**2*omega**2+x[538]*e*inc**2*nu**2*omega**2+\
    x[539]*e**2*inc**2*nu**2*omega**2+x[540]*e**3*inc**2*nu**2*omega**2+x[541]*inc**2*nu**3*omega**2+\
    x[542]*e*inc**2*nu**3*omega**2+x[543]*e**2*inc**2*nu**3*omega**2+x[544]*e**3*inc**2*nu**3*omega**2+\
    x[545]*inc**3*omega**2+x[546]*e*inc**3*omega**2+x[547]*e**2*inc**3*omega**2+x[548]*e**3*inc**3*omega**2+\
    x[549]*inc**3*nu*omega**2+x[550]*e*inc**3*nu*omega**2+x[551]*e**2*inc**3*nu*omega**2+\
    x[552]*e**3*inc**3*nu*omega**2+x[553]*inc**3*nu**2*omega**2+x[554]*e*inc**3*nu**2*omega**2+\
    x[555]*e**2*inc**3*nu**2*omega**2+x[556]*e**3*inc**3*nu**2*omega**2+x[557]*inc**3*nu**3*omega**2+\
    x[558]*e*inc**3*nu**3*omega**2+x[559]*e**2*inc**3*nu**3*omega**2+x[560]*e**3*inc**3*nu**3*omega**2+\
    x[561]*omega**3+x[562]*e*omega**3+x[563]*e**2*omega**3+x[564]*e**3*omega**3+x[565]*nu*omega**3+\
    x[566]*e*nu*omega**3+x[567]*e**2*nu*omega**3+x[568]*e**3*nu*omega**3+x[569]*nu**2*omega**3+\
    x[570]*e*nu**2*omega**3+x[571]*e**2*nu**2*omega**3+x[572]*e**3*nu**2*omega**3+\
    x[573]*nu**3*omega**3+x[574]*e*nu**3*omega**3+x[575]*e**2*nu**3*omega**3+\
    x[576]*e**3*nu**3*omega**3+x[577]*inc*omega**3+x[578]*e*inc*omega**3+\
    x[579]*e**2*inc*omega**3+x[580]*e**3*inc*omega**3+x[581]*inc*nu*omega**3+\
    x[582]*e*inc*nu*omega**3+x[583]*e**2*inc*nu*omega**3+x[584]*e**3*inc*nu*omega**3+\
    x[585]*inc*nu**2*omega**3+x[586]*e*inc*nu**2*omega**3+x[587]*e**2*inc*nu**2*omega**3+\
    x[588]*e**3*inc*nu**2*omega**3+x[589]*inc*nu**3*omega**3+x[590]*e*inc*nu**3*omega**3+\
    x[591]*e**2*inc*nu**3*omega**3+x[592]*e**3*inc*nu**3*omega**3+x[593]*inc**2*omega**3+\
    x[594]*e*inc**2*omega**3+x[595]*e**2*inc**2*omega**3+x[596]*e**3*inc**2*omega**3+\
    x[597]*inc**2*nu*omega**3+x[598]*e*inc**2*nu*omega**3+x[599]*e**2*inc**2*nu*omega**3+\
    x[600]*e**3*inc**2*nu*omega**3+x[601]*inc**2*nu**2*omega**3+x[602]*e*inc**2*nu**2*omega**3+\
    x[603]*e**2*inc**2*nu**2*omega**3+x[604]*e**3*inc**2*nu**2*omega**3+x[605]*inc**2*nu**3*omega**3+\
    x[606]*e*inc**2*nu**3*omega**3+x[607]*e**2*inc**2*nu**3*omega**3+x[608]*e**3*inc**2*nu**3*omega**3+\
    x[609]*inc**3*omega**3+x[610]*e*inc**3*omega**3+x[611]*e**2*inc**3*omega**3+x[612]*e**3*inc**3*omega**3+\
    x[613]*inc**3*nu*omega**3+x[614]*e*inc**3*nu*omega**3+x[615]*e**2*inc**3*nu*omega**3+\
    x[616]*e**3*inc**3*nu*omega**3+x[617]*inc**3*nu**2*omega**3+x[618]*e*inc**3*nu**2*omega**3+\
    x[619]*e**2*inc**3*nu**2*omega**3+x[620]*e**3*inc**3*nu**2*omega**3+x[621]*inc**3*nu**3*omega**3+\
    x[622]*e*inc**3*nu**3*omega**3+x[623]*e**2*inc**3*nu**3*omega**3+x[624]*e**3*inc**3*nu**3*omega**3
    return rhs

def rhs_error(x,inc,v,w,e):
    #interpArray = np.zeros((len(w),len(inc),len(e),len(v)))
    error = 0
    for i,j,k in itertools.product(np.arange(len(inc)),np.arange(len(w)),np.arange(len(e))):#,np.arange(len(v))):
        #interpArray[i,j,k,l] = rhs(inc[i],v[l],w[j],e[k])
        tmp = rhs_model(x,inc[i],v,w[j],e[k])
        error = error + np.sum((interpArray[i,j,k]-tmp)**2.)
    print(error)
    return error

#Load x0 from file
dfilename2 = 'rhsmodelx0' + '_9_14_2020.pkl' #Optimization over a 5,5,5,5 shaped grid. Did not successfully terminate due to precision loss. Final Error 571.08 or 0.9137 error per grid-point
path2 = os.path.join(cachedir, dfilename2)

x0 = 0.*np.ones(totalCoeffs)
if os.path.exists(path2):
    print('Loading cached Interpolation Grid array from "%s".' % path2)
    try:
        with open(path2, "rb") as ff:
            x0 = pickle.load(ff)
    except UnicodeDecodeError:
        with open(path2, "rb") as ff:
            x0 = pickle.load(ff,encoding='latin1')
out = minimize(rhs_error,x0,args=(inc,v,w,e))

# store dynamic completeness array as .dcomp file
with open(path2, 'wb') as ff:
    pickle.dump(out.x, ff)
