# For determining cost of old US Govt. Projects

import pandas as pd 
import matplotlib.pyplot as plt

# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("inflation_data.csv") 


vehicles = dict()

vehicles['USSOlympia'] = {'cost':1796000,'currency':'USD','FY':1893, 'FY2020':1.}
vehicles['HMSQueenElizabeth'] = {'cost':3014103,'currency':'pound','FY':1913, 'FY2020':1.}
vehicles['Derfflinger'] = {'cost':56000000,'currency':'gold marks','FY':1914, 'FY2020':1.}
vehicles['zero'] = {'cost':55000,'currency':'yen','FY':1945, 'FY2020':1.}#https://forum.axishistory.com/viewtopic.php?t=149803&start=45


currencies = ['USD','pound','gold marks','yen']
currecyConversionDict = dict() #From X to Y (X,Y). To convert from Currency X to currency Y, multiply X by the associated index (X,Y)
currecyConversionDict[(currencies[0],currencies[1])] = 1.2678 #https://www.majorexchangerates.com/usd/1914-gbp.html
currecyConversionDict[(currencies[0],currencies[2])] = 4.198 #http://marcuse.faculty.history.ucsb.edu/projects/currency.htm
currecyConversionDict[(currencies[0],currencies[3])] = 1/0.2344
#currecyConversionDict[(currencies[0],currencies[1])] = 

#Convert All Costs into USD
vehicles['HMSQueenElizabeth']['cost'] = vehicles['HMSQueenElizabeth']['cost']*currecyConversionDict[(currencies[0],currencies[1])]
vehicles['Derfflinger']['cost'] = vehicles['Derfflinger']['cost']*currecyConversionDict[(currencies[0],currencies[2])]


inflations = list(data.loc['year'==vehicles['USSOlympia']['FY']:2020]['inflation rate'])
for i in np.arange(len(inflations)):
    vehicles['USSOlympia']['FY2020'] = vehicles['USSOlympia']['FY2020']*(1.+inflations[i])

inflations = list(data.loc['year'==vehicles['HMSQueenElizabeth']['FY']:2020]['inflation rate'])
for i in np.arange(len(inflations)):
    vehicles['HMSQueenElizabeth']['FY2020'] = vehicles['HMSQueenElizabeth']['FY2020']*(1.+inflations[i])

inflations = list(data.loc['year'==vehicles['Derfflinger']['FY']:2020]['inflation rate'])
for i in np.arange(len(inflations)):
    vehicles['Derfflinger']['FY2020'] = vehicles['Derfflinger']['FY2020']*(1.+inflations[i])


vehicles['USSOlympia']['FY2020']
vehicles['HMSQueenElizabeth']['FY2020']
vehicles['Derfflinger']['FY2020']

