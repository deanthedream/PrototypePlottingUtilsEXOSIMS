"""
The purpose of this script is to take a template json script and make a series of "similar" json scripts from an original json script

This script is designed for two modes:
	Sweep Single Parameter
	Sweep Multiple Parameters over Range

#makeSimilarScripts.py is designed to run from the 'EXOSIMS/Scripts/' Folder

Another example
 %run makeSimilarScripts.py


 
Written by Dean Keithly on 6/28/2018
"""

import os
import numpy as np
import argparse
import json
import re
import ntpath
import ast
import string
import copy
import datetime

def createScriptFolder(makeSimilarInst,sourcefile):
    """This method creates a 'Script Folder' - a new folder with name 'makeSimilarInst_sourcefile' in 'EXOSIMS/Scripts/'
    """
    myString = os.getcwd() + '/' + makeSimilarInst + '_' + sourcefile
    try:
        os.mkdir(myString)#will fail if directory exists
        print('MADE DIR: ' + myString)
    except:
        print('DID NOT MAKE DIR: ' + myString + ' It already exists.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of scripts and a queue. all files are relocated to a new folder.")
    parser.add_argument('--makeSimilarInst',nargs=1,type=str, help='Full path to the makeSimilar.json instruction script (string).')

    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

    
    #(default) If no makeSimilarScripts instruction file is provided, default use makeSimilar.json
    if makeSimilarInst is None:
        with open('./makeSimilar.json') as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#This script contains the instructions for precisely how to modify the base file
        makeSimilarInst = 'makeSimilar.json'
    else:#else: use the provided instructions
        assert os.path.exists(makeSimilarInst), "%s is not a valid filepath" % (makeSimilarInst)
        with open(makeSimilarInst) as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#This script contains the instructions for precisely how to modify the base file


    sourcefile = jsonDataInstruction['scriptName']#the filename of the script to be copied
    with open('../Scripts/' + sourcefile) as f:#Load source script json file
        jsonDataSource = json.load(f)#This script contains the information to be slightly modified

    #Create Script Folder
    createScriptFolder(makeSimilarInst,sourcefile)#Create 'Script Folder' - a new folder with name 'makeSimilarInst_sourcefile' in 'EXOSIMS/Scripts/'
    

    namesOfScriptsCreated = list()
    # Case 1
    """
    Here we want to sweep parameters A and B such that  A = [1,2,3] and B = [4,5,6]
    In this case, the first script will have A=1, B=4. The second script will have A=2, B=5.
    It is required that len(A) == len(B).
    You can sweep an arbitrarily large number of parameters A,B,C,D,...,Z,AA,... so long as the number of values you have for each is constant
    """
    if jsonDataInstruction['sweepType'] == "SweepParameters":
        sweepParameters = jsonDataInstruction['sweepParameters']# Grab Parameter to Sweep
        sweepValues = jsonDataInstruction['sweepValues']# retrieve manually defined sweep values

        #Check Each Parameter has the same number of values to sweep
        for ind in range(len(sweepValues)-1):
            assert(len(sweepValues[ind]) == len(sweepValues[ind+1]))

        for ind in range(len(sweepValues[0])):#Number of values to sweep over
            #Create Filename Substring using parameters and values
            paramNameSet = ''
            for ind2 in range(len(sweepParameters)):#Iterate over all parameters
                paramNameSet = paramNameSet + sweepParameters[ind2] + str(sweepValues[ind2][ind])
            #Now strip all invalid filename parts
            scriptName = os.path.splitext(sourcefile)[0] + ''.join(c for c in paramNameSet if c in valid_chars and not(c == '.')) + '.json'
            namesOfScriptsCreated.append(scriptName)#Append to master list of all scripts created
            jsonDataOutput = copy.deepcopy(jsonDataSource)#Create a deepCopy of the original json script

            for ind3 in range(len(sweepParameters)):#Iterate over all parameters
                jsonDataOutput[sweepParameters[ind3]] = sweepValues[ind3][ind] #replace value

            #Write out json file
            with open('../Scripts/' + scriptName, 'w') as g:
                json.dump(jsonDataOutput, g, indent=1)


        # Create queue.json script from namesOfScriptsCreated
        queueOut = {}
        date = unicode(datetime.datetime.now())
        date2 = ''.join(c for c in date if c in valid_chars and not(c == '.'))
        with open('../run/queue' + date2 + '.json', 'w') as g:
            queueOut['scriptNames'] = namesOfScriptsCreated
            queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
            json.dump(queueOut, g, indent=1)

        # Case 2
        """
        Here we want to take a set of parameters A,B,C,...,Z
        and set them at +/- a,b,c,...,z% from theic current value
        """
    elif (jsonDataInstruction['sweepType'] == "SweepParametersPercentage"):
        sweepPercentage = jsonDataInstruction['sweepPercentage']# retrieve manually defined sweep percentage
        sweepParameters = jsonDataInstruction['sweepParameters']# Grab Parameter to Sweep

        #for ind in range(2):#Number of values to sweep over
        #Create Filename Substring using parameters and values
        paramNameSet = ''
        for ind2 in range(len(sweepParameters)):#Iterate over all parameters
            paramNameSet = paramNameSet + sweepParameters[ind2] + str(1. + sweepPercentage[ind2])
        #Now strip all invalid filename parts
        scriptName = os.path.splitext(sourcefile)[0] + ''.join(c for c in paramNameSet if c in valid_chars and not(c == '.')) + '.json'
        namesOfScriptsCreated.append(scriptName)#Append to master list of all scripts created
        jsonDataOutput = copy.deepcopy(jsonDataSource)#Create a deepCopy of the original json script

        for ind3 in range(len(sweepParameters)):#Iterate over all parameters
            jsonDataOutput[sweepParameters[ind3]] = 1. + sweepPercentage[ind3] #replace value

        #Write out json file
        with open('../Scripts/' + scriptName, 'w') as g:
            json.dump(jsonDataOutput, g, indent=1)

        # Create queue.json script from namesOfScriptsCreated
        queueOut = {}
        date = unicode(datetime.datetime.now())
        date2 = ''.join(c for c in date if c in valid_chars and not(c == '.'))
        with open('../run/queue' + date2 + '.json', 'w') as g:
            queueOut['scriptNames'] = namesOfScriptsCreated
            queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
            json.dump(queueOut, g, indent=1)
    else:
        print('not a valid instruction script')

