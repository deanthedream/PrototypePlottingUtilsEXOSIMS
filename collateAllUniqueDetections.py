"""Collate All Unique Detections
The purpose of this script is to search through all pkl files in SUBFOLDERS of a FOLDER and search parse them for unique detections. Each Unique Detection will be appended to a csv file with folder name, pkl file name, 

To run from within IPython
%run collateAllUniqueDetections.py --searchFolder '/full path to/' --outFolder '/outFolder to put csv file'

A specific example
%run collateAllUniqueDetections.py --searchFolder '/home/dean/Documents/SIOSlab' --outFolder '/outFolder to put csv file/collatedData_collationDate.csv'


 
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
from itertools import combinations
import shutil
import glob
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of scripts and a queue. all files are relocated to a new folder.")
    parser.add_argument('--searchFolder',nargs=1,type=str, help='Path to Folder to Search Through (string).')
    parser.add_argument('--outFolder',nargs=1,type=str, help='Path to Folder to Place collatedData_collationDate.csv (string).')
    args = parser.parse_args()


    searchFolder = args.searchFolder[0]
    outFolder = args.outFolder[0]

    #### Get List of All run_dir containing pkl files
    #searchFolder = '/home/dean/Documents/SIOSlab/'  
    searchFolder += '*/'
    pklfiles = glob.glob(os.path.join(searchFolder,'*.pkl'))
    pklfiles2 = list()
    for f in pklfiles:
        myStr = '/'.join(f.split('/')[0:-1])
        if not myStr in pklfiles2:#Ensures no duplicates added
            pklfiles2.append(myStr)
    ##########

    #### Get Date
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    ####################################3

    #### Search Through All Files
    for f in pklfiles2:
        out = gen_summary(f, includeUniversePlanetPop=False)

        Rps = list()
        detected = list()       
        Mps = list()
        #tottime = list()
        starinds = list()
        smas = list()
        ps = list()
        es = list()
        WAs = list()
        SNRs = list()
        fZs = list()
        fEZs = list()
        dMags = list()
        rs = list()

        #Parse through out for planets with R<Rneptune
        for ind1 in range(len(out['detected'])):

            for ind2 in range(len(out['detected'][ind1])):
                if out['Rps'][ind1][ind2] < 24764.0/6371.0: #Radius of neptune in earth Radii
                    Rps.append(out['Rps'][ind1][ind2])
                    detected.append(out['detected'][ind1][ind2])
                    Mps.append(out['Mps'][ind1][ind2])
                    #tottime.append(out['tottime'][ind1][ind2])
                    starinds.append(out['starinds'][ind1][ind2])
                    smas.append(out['smas'][ind1][ind2])
                    ps.append(out['ps'][ind1][ind2])
                    es.append(out['es'][ind1][ind2])
                    WAs.append(out['WAs'][ind1][ind2])
                    SNRs.append(out['SNRs'][ind1][ind2])
                    fZs.append(out['fZs'][ind1][ind2])
                    fEZs.append(out['fEZs'][ind1][ind2])
                    dMags.append(out['dMags'][ind1][ind2])
                    rs.append(out['rs'][ind1][ind2])

        outString = list()
        for i in range(len(Rps)):
            outString.append(str(Rps[i]) + ',' + str(detected[i]) + ',' + str(Mps[i]) + ',' +str(starinds[i]) + ',' +str(smas[i]) + ',' +str(ps[i]) + ',' +str(es[i]) + ',' +str(WAs[i]) + ',' +str(SNRs[i]) + ',' +str(fZs[i]) + ',' +str(fEZs[i]) + ',' + str(dMags[i]) + ',' +str(rs[i]) + '\n')
        outString = ''.join(outString)

        with open(outFolder + 'NEIDinfo.txt', 'a+') as g:
            g.write(outString)