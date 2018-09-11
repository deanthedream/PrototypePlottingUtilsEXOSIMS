"""singleRunPostProcessing.py


#To Run singleRunPostProcessing.py within an ipython session
%run singeRunPostPorcessing.py --onMultiRun '/pathToDirContainingRunTypes/'
--onRunFolder

Note: the onMultiRun path should contain a series of run_type folders containing individual runs

Written By: Dean Keithly
Written On: 9/10/2018
"""
import os
if not 'DISPLAY' in os.environ.keys():
    import matplotlib
    matplotlib.use('Agg')

import shutil
import numpy as np
import argparse
import json
import datetime





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of scripts and a queue. all files are relocated to a new folder.")
    parser.add_argument('--onMultiRun',nargs=1,type=str, help='Folder to run Single Run Post Porcessing On (string).')

    args = parser.parse_args()
    onMultiRun = args.onMultiRun[0]

    #Open singlePlotInst.json if it exists
    if args.singlePlotInst is None:
        singlePlotInst = './singlePlotInst.json'
        assert os.path.exists(singlePlotInst), "%s is not a valid filepath" % (singlePlotInst)
        with open(singlePlotInst.json) as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#T
    if os.path.isfile(onMultiRun + 'singlePlotInst.json'):
        with open(singlePlotInst.json) as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#T

    for item in os.listdir(onMultiRun):#Iterate over all items at this level
        if os.path.isdir(item):
            pass
            #Do a thing if the item is a directory