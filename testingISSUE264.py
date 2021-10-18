#testing same stars in different catalogs having different KOMaps


import sys, os.path, EXOSIMS, EXOSIMS.MissionSim, json
import numpy as np
import copy
from copy import deepcopy
import astropy.units as u


outspec1 = {"missionFinishAbs": 61729.75,
    "missionLife": 3.0,
    "missionPortion": 0.195,
    "missionSchedule": None,
    "missionStart": 60634.0,
    "scienceInstruments": [
    {
      "name": "imagingEMCCD",
      "QE": 0.9,
      "optics": 0.28,
      "FoV": 0.75,
      "pixelNumber": 105,
      "pixelSize": 13e-6,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 1.3e-3,
      "texp": 100,
      "ENF": 1,
      "PCeff": 0.75
    },
    {
      "name": "imagingRedEMCCD",
      "QE": 0.9,
      "optics": 0.42,
      "FoV": 1.1,
      "pixelNumber": 104,
      "pixelSize": 13e-6,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 1.3e-3,
      "texp": 100,
      "PCeff": 0.75,
      "ENF": 1
    },
    {
      "name": "spectroEMCCD",
      "QE": 0.9,
      "FoV": 0.8,
      "optics": 0.27,
      "pixelNumber": 102,
      "pixelSize": 12e-6,
      "core_platescale": 0.1,
      "lenslSamp": 2,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 2.1e-3,
      "texp": 300,
      "ENF": 1,
      "Rs": 140,
      "PCeff": 0.9
    }
  ],
  "starlightSuppressionSystems": [
    {
      "name": "VVC500",
      "lam": 500,
      "IWA": 0.045,
      "OWA": 2.127,
      "ohTime": 0.1,
      "BW": 0.20,
      "optics": 0.95,
      "optics_comment": "contamination",
      "core_platescale": 0.1,
      "koAngles_Earth": [
            45.0,
            180.0
        ],
        "koAngles_Moon": [
            45.0,
            180.0
        ],
        "koAngles_Small": [
            1.0,
            180.0
        ],
        "koAngles_Sun": [
            45.0,
            124.0
        ]
    }
 ],
      "observingModes": [
    {
      "instName": "imagingEMCCD",
      "systName": "VVC500",
      "detection": 1,
      "SNR": 7
    },
    {
      "instName": "imagingRedEMCCD",
      "systName": "VVC500",
      "detection": 1,
      "lam": 745,
      "SNR": 7
    }
  ],
  "modules": {
    "PlanetPopulation": "KeplerLike2",
    "StarCatalog": "EXOCAT1",
    "OpticalSystem": " ",
    "ZodiacalLight": " ",
    "BackgroundSources": " ",
    "PlanetPhysicalModel": "FortneyMarleyCahoyMix1",
    "Observatory": "WFIRSTObservatoryL2",
    "TimeKeeping": " ",
    "PostProcessing": " ",
    "Completeness": " ",
    "TargetList": " ",
    "SimulatedUniverse": "KeplerLikeUniverse",
    "SurveySimulation": " ",
    "SurveyEnsemble": " "
  }}

outspec2 = {"missionFinishAbs": 61729.75,
    "missionLife": 3.0,
    "missionPortion": 0.195,
    "missionSchedule": None,
    "missionStart": 60634.0,
    "scienceInstruments": [
    {
      "name": "imagingEMCCD",
      "QE": 0.9,
      "optics": 0.28,
      "FoV": 0.75,
      "pixelNumber": 105,
      "pixelSize": 13e-6,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 1.3e-3,
      "texp": 100,
      "ENF": 1,
      "PCeff": 0.75
    },
    {
      "name": "imagingRedEMCCD",
      "QE": 0.9,
      "optics": 0.42,
      "FoV": 1.1,
      "pixelNumber": 104,
      "pixelSize": 13e-6,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 1.3e-3,
      "texp": 100,
      "PCeff": 0.75,
      "ENF": 1
    },
    {
      "name": "spectroEMCCD",
      "QE": 0.9,
      "FoV": 0.8,
      "optics": 0.27,
      "pixelNumber": 102,
      "pixelSize": 12e-6,
      "core_platescale": 0.1,
      "lenslSamp": 2,
      "sread": 0,
      "idark": 3e-5,
      "CIC": 2.1e-3,
      "texp": 300,
      "ENF": 1,
      "Rs": 140,
      "PCeff": 0.9
    }
  ],
  "starlightSuppressionSystems": [
    {
      "name": "VVC500",
      "lam": 500,
      "IWA": 0.045,
      "OWA": 2.127,
      "ohTime": 0.1,
      "BW": 0.20,
      "optics": 0.95,
      "optics_comment": "contamination",
      "core_platescale": 0.1,
      "koAngles_Earth": [
            45.0,
            180.0
        ],
        "koAngles_Moon": [
            45.0,
            180.0
        ],
        "koAngles_Small": [
            1.0,
            180.0
        ],
        "koAngles_Sun": [
            45.0,
            124.0
        ]
    }
    ],
      "observingModes": [
    {
      "instName": "imagingEMCCD",
      "systName": "VVC500",
      "detection": 1,
      "SNR": 7
    },
    {
      "instName": "imagingRedEMCCD",
      "systName": "VVC500",
      "detection": 1,
      "lam": 745,
      "SNR": 7
    }
  ],
  "modules": {
    "PlanetPopulation": "KnownRVPlanets",
    "StarCatalog": "EXOCAT1",
    "OpticalSystem": " ",
    "ZodiacalLight": " ",
    "BackgroundSources": " ",
    "PlanetPhysicalModel": " ",
    "Observatory": "WFIRSTObservatoryL2",
    "TimeKeeping": " ",
    "PostProcessing": " ",
    "Completeness": " ",
    "TargetList": "KnownRVPlanetsTargetList",
    "SimulatedUniverse": "KnownRVPlanetsUniverse",
    "SurveySimulation": " ",
    "SurveyEnsemble": " "
  }}

sim1 = EXOSIMS.MissionSim.MissionSim(**outspec1)#scriptfile=scriptfile,nopar=True)
sim2 = EXOSIMS.MissionSim.MissionSim(**outspec2)

# Notice that HIP 74500 = HD 134987
indIn1 = np.where(sim1.TargetList.Name == 'HIP 74500')[0]
indIn2 = np.where(sim2.TargetList.Name == 'HD 134987')[0]

len(sim1.TargetList.Name)
len(sim2.TargetList.Name)

sim1.SurveySimulation.koMaps['VVC500'].shape
sim2.SurveySimulation.koMaps['VVC500'].shape

sim1.TargetList.coords[indIn1]
sim2.TargetList.coords[indIn2]

#verifies the two targets have the same keepout map
np.all(np.logical_not(sim1.SurveySimulation.koMaps['VVC500'][indIn1] == sim2.SurveySimulation.koMaps['VVC500'][indIn2]))

