#For opening and testing Thermal Desktop Execution From the API

import sys
print sys.path

import clr
TAV = clr.AddReference('TdApiV1') #this is the imported TdApiV1 module

#Create Instance of Thermal Desktop
from TdApiV1 import *
#from TdApiV1 import ThermalDesktop
td = ThermalDesktop()#TdApiV1.ThermalDesktop() #instantiate an instance of thermal desktop object
td.Connect() # this turns on autodesk and thermal desktop
td.Print("Hello World") #this prints Hello World to the Autodesk command interface

node1 = td.CreateNode()#This creates a node
node2 = td.CreateNode()

#Add Node Properties



submod = td.CreateSubmodel('HelloWorld')
node1.Submodel = submod.get_Name()
node2.Submodel = submod.get_Name()

#Create Instance of RcConductorData
#cond = TdApiV1.RcConductorData()

#Create a Connection Object For both objects that will be connected
con1 = Connection()#TdApiV1.Connection()
con1.set_Handle(node1.Handle)#Have to specify a handle for connection
con2 = Connection()#TdApiV1.Connection()
con2.set_Handle(node2.Handle)#Have to specify a handle for connection

#Create Conductor between the two connections
cond1 = td.CreateConductor(con1.Handle, con2.Handle)#cond numbering is arbitrary
cond1.Submodel = submod.get_Name()#Set submodel name
cond1.ValueExp.set_Value("100.0 + 100.0")

#UnitsData = TdApiV1.UnitsData()
Units = Units#TdApiV1.Units
cond1.ValueExp.set_units(UnitsData(Units.SI))#(TdApiV1.UnitsData(Units.SI))


td.SetRcConductor(con1)#actually pushes the conductor created to thermal desktop



td.ZoomExtents()#zooms to the edges of the TD model in autodesk