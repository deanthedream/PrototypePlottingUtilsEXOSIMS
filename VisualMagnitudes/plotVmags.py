## Plots Visual Magnitudes of Objects

import matplotlib.pyplot as plt
import numpy as np
#DELETE import matplotlib.dates as mdates
import os


PPoutpath = './'

VmagLims = [-27.,32.]
#Saving original
# Vmags = [
# #{'Vmag':    -67.57  , 'Name':'gamma-ray burst GRB 080319B',   'Observer':' seen from 1 AU away ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -44.00  , 'Name':'star R136a1',                   'Observer':' seen from 1 AU away ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -40.07  , 'Name':'star Zeta1 Scorpii',            'Observer':' seen from 1 AU away ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -38.00  , 'Name':'star Rigel',                    'Observer':' seen from 1 AU away ', 'Notes':'It would be seen as a large very bright bluish disk of 35° apparent diameter.   ', 'plot':'yes'},
# #{'Vmag':    -30.30  , 'Name':'star Sirius A',                 'Observer':' seen from 1 AU away ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -29.30  , 'Name':'Sun',                           'Observer':' seen from Mercury at perihelion ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -27.40  , 'Name':'Sun',                           'Observer':' seen from Venus at perihelion   ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    -26.74  , 'Name':'Sun',                           'Observer':' seen from Earth[17] ', 'Notes':'About 400,000 times brighter than mean full moon    ', 'plot':'yes'},
# #{'Vmag':    -25.60  , 'Name':'Sun',                           'Observer':' seen from Mars at aphelion  ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    -25.00  , 'Name':'Minimum brightness that causes the typical eye slight pain to look at   ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -23.00  , 'Name':'Sun',                           'Observer':' seen from Jupiter at aphelion   ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -21.70  , 'Name':'Sun',                           'Observer':' seen from Saturn at aphelion    ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -20.20  , 'Name':'Sun',                           'Observer':' seen from Uranus at aphelion    ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -19.30  , 'Name':'Sun',                           'Observer':' seen from Neptune   ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -18.20  , 'Name':'Sun',                           'Observer':' seen from Pluto at aphelion ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -16.70  , 'Name':'Sun',                           'Observer':' seen from Eris at aphelion  ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -14.20  , 'Name':'An illumination level of 1 lux[18][19]', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    -12.90  , 'Name':'full moon',                     'Observer':' seen from Earth at perihelion   ', 'Notes':'maximum brightness of perigee + perihelion + full moon (mean distance value is -12.74,[20] though values are about 0.18 magnitude brighter when including the opposition effect)    ', 'plot':'yes'},
# {'Vmag':    -11.20  , 'Name':'Sun',                           'Observer':' seen from Sedna at aphelion ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    -10.00  , 'Name':'Comet Ikeya–Seki (1965)',       'Observer':' seen from Earth ', 'Notes':'which was the brightest Kreutz Sungrazer of modern times[21]    ', 'plot':'yes'},
# {'Vmag':    -9.50   , 'Name':'Iridium (satellite) flare',     'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
# {'Vmag':    -7.50   , 'Name':'supernova of 1006',             'Observer':' seen from Earth ', 'Notes':'the brightest stellar event in recorded history (7200 light-years away)[22] ', 'plot':'yes'},
# #{'Vmag':    -6.50   , 'Name':'The total integrated magnitude of the night sky ', 'Observer':' seen from Earth ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -6.00   , 'Name':'Crab Supernova of 1054',        'Observer':' seen from Earth ', 'Notes':'(6500 light-years away)[23] ', 'plot':'yes'},
# {'Vmag':    -5.90   , 'Name':'International Space Station',   'Observer':' seen from Earth ', 'Notes':'when the ISS is at its perigee and fully lit by the Sun[24] ', 'plot':'yes'},
# {'Vmag':    -4.92   , 'Name':'Venus',                         'Observer':' seen from Earth ', 'Notes':'maximum brightness[25] when illuminated as a crescent   ', 'plot':'yes'},
# #{'Vmag':    -4.14   , 'Name':'Venus',                         'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# {'Vmag':    -4.     , 'Name':'Faintest objects observable during the day with naked eye when Sun is high  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -3.99   , 'Name':'star Epsilon Canis Majoris  ',  'Observer':' seen from Earth ', 'Notes':'maximum brightness of 4.7 million years ago, the historical brightest star of the last and next five million years  ', 'plot':'yes'},
# #{'Vmag':    -2.98   , 'Name':'Venus',                         'Observer':' seen from Earth ', 'Notes':'minimum brightness when it is on the far side of the Sun[25]    ', 'plot':'yes'},
# {'Vmag':    -2.94   , 'Name':'Jupiter',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes'},
# {'Vmag':    -2.94   , 'Name':'Mars',                          'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes'},
# #{'Vmag':    -2.5    , 'Name':'Faintest objects visible during the day with naked eye when Sun is less than 10° above the horizon  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    -2.50   , 'Name':'new moon',                      'Observer':' seen from Earth ', 'Notes':'minimum brightness  ', 'plot':'yes'},
# {'Vmag':    -2.48   , 'Name':'Mercury',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness at superior conjunction (unlike Venus, Mercury is at its brightest when on the far side of the Sun, the reason being their different phase curves)[25]   ', 'plot':'yes'},
# #{'Vmag':    -2.20   , 'Name':'Jupiter',                       'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# {'Vmag':    -1.66   , 'Name':'Jupiter',                       'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# {'Vmag':    -1.47   , 'Name':'Sirius',                        'Observer':' seen from Earth ', 'Notes':'Brightest star except for the Sun at visible wavelengths[26]    ', 'plot':'yes'},
# #{'Vmag':    -0.83   , 'Name':'star Eta Carinae',              'Observer':' seen from Earth ', 'Notes':'apparent brightness as a supernova impostor in April 1843   ', 'plot':'yes'},
# {'Vmag':    -0.72   , 'Name':'star Canopus',                  'Observer':' seen from Earth ', 'Notes':'2nd brightest star in night sky[27] ', 'plot':'yes'},
# {'Vmag':    -0.55   , 'Name':'Saturn',                        'Observer':' seen from Earth ', 'Notes':'maximum brightness near opposition and perihelion when the rings are angled toward Earth[25]    ', 'plot':'yes'},
# {'Vmag':    -0.3    , 'Name':'Halleys comet',                 'Observer':' seen from Earth ', 'Notes':'Expected apparent magnitude at 2061 passage ', 'plot':'yes'},
# #{'Vmag':    -0.27   , 'Name':'star system Alpha Centauri AB', 'Observer':' seen from Earth ', 'Notes':'Combined magnitude (3rd brightest star in night sky)    ', 'plot':'yes'},
# #{'Vmag':    -0.04   , 'Name':'star Arcturus',                 'Observer':' seen from Earth ', 'Notes':'4th brightest star to the naked eye[28] ', 'plot':'yes'},
# #{'Vmag':    -0.01   , 'Name':'star Alpha Centauri A',         'Observer':' seen from Earth ', 'Notes':'4th brightest individual star visible telescopically in the night sky   ', 'plot':'yes'},
# {'Vmag':    0.03    , 'Name':'star Vega',                     'Observer':' seen from Earth ', 'Notes':'which was originally chosen as a definition of the zero point[29]   ', 'plot':'yes'},
# #{'Vmag':    0.23    , 'Name':'Mercury',                       'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# #{'Vmag':    0.5     , 'Name':'Sun',                           'Observer':' seen from Alpha Centauri    ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    0.46    , 'Name':'Saturn',                        'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# #{'Vmag':    0.71    , 'Name':'Mars',                          'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# #{'Vmag':    1.17    , 'Name':'Saturn',                        'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# #{'Vmag':    1.86    , 'Name':'Mars',                          'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# {'Vmag':    3.03    , 'Name':'supernova SN 1987A',            'Observer':' seen from Earth', 'Notes':'in the Large Magellanic Cloud (160,000 light-years away)    ', 'plot':'yes'},
# {'Vmag':    3.      , 'Name':'Faintest stars visible in an urban neighborhood with naked eye', 'Observer':' ', 'Notes':' ', 'plot':'yes', 'lower':3., 'upper':4.},
# {'Vmag':    3.44    , 'Name':'Andromeda Galaxy',              'Observer':' seen from Earth ', 'Notes':'M31[30] ', 'plot':'yes'},
# {'Vmag':    4.      , 'Name':'Orion Nebula',                  'Observer':' seen from Earth ', 'Notes':'M42 ', 'plot':'yes'},
# #{'Vmag':    4.38    , 'Name':'moon Ganymede',                 'Observer':' seen from Earth ', 'Notes':'maximum brightness[31] (moon of Jupiter and the largest moon in the Solar System)   ', 'plot':'yes'},
# {'Vmag':    4.5     , 'Name':'open cluster M41',              'Observer':' seen from Earth ', 'Notes':'an open cluster that may have been seen by Aristotle[32]    ', 'plot':'yes'},
# {'Vmag':    4.5     , 'Name':'Sagittarius Dwarf Spheroidal Galaxy', 'Observer':' seen from Earth ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    5.2     , 'Name':'asteroid Vesta',                'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
# {'Vmag':    5.38    , 'Name':'Uranus',                        'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes', 'source':'[33]'},
# #{'Vmag':    5.68    , 'Name':'Uranus',                        'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# {'Vmag':    5.72    , 'Name':'spiral galaxy M33',             'Observer':' seen from Earth ', 'Notes':'which is used as a test for naked eye seeing under dark skies[34][35]   ', 'plot':'yes'},
# {'Vmag':    5.8     , 'Name':'gamma-ray burst GRB 080319B',   'Observer':' seen from Earth ', 'Notes':'Peak visual magnitude (the "Clarke Event") seen on Earth on March 19, 2008 from a distance of 7.5 billion light-years.  ', 'plot':'yes'},
# #{'Vmag':    6.03    , 'Name':'Uranus',                        'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# {'Vmag':    6.49    , 'Name':'asteroid Pallas',               'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
# {'Vmag':    6.5     , 'Name':'Approximate limit of stars observed by a mean naked eye observer under very good conditions. There are about 9,500 stars visible to mag 6.5.[4] ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    6.64    , 'Name':'dwarf planet Ceres',            'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
# {'Vmag':    6.75    , 'Name':'asteroid Iris',                 'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
# {'Vmag':    6.9     , 'Name':'spiral galaxy M81',             'Observer':' seen from Earth ', 'Notes':'This is an extreme naked-eyetarget that pushes human eyesight and the Bortle scale to the limit[36] ', 'plot':'yes'},
# {'Vmag':    7.      , 'Name':'Extreme naked-eye limit, Class 1 on Bortle scale, the darkest skies available on Earth[37]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes', 'lower':7.,'upper':8.},
# #{'Vmag':    7.25    , 'Name':'Mercury',                       'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# {'Vmag':    7.67    , 'Name':'Neptune',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes', 'source':'[38]'},
# #{'Vmag':    7.78    , 'Name':'Neptune',                       'Observer':' seen from Earth ', 'Notes':'mean brightness[25] ', 'plot':'yes'},
# #{'Vmag':    8.      , 'Name':'Neptune',                       'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
# #{'Vmag':    8.1     , 'Name':'moon Titan',                    'Observer':' seen from Earth ', 'Notes':'maximum brightness; largest moon of Saturn;[39][40] mean opposition magnitude 8.4[41]   ', 'plot':'yes'},
# #{'Vmag':    8.29    , 'Name':'star UY Scuti',                 'Observer':' seen from Earth ', 'Notes':'Maximum brightness; largest known star by radius    ', 'plot':'yes'},
# {'Vmag':    8.94    , 'Name':'asteroid 10 Hygiea',            'Observer':' seen from Earth ', 'Notes':'maximum brightness[42]  ', 'plot':'yes'},
# {'Vmag':    9.5     , 'Name':'Faintest objects visible using common 7×50 binoculars under typical conditions[43]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    10.2    , 'Name':'moon Iapetus',                  'Observer':' seen from Earth ', 'Notes':'maximum brightness,[40] brightest when west of Saturn and takes 40 days to switch sides ', 'plot':'yes'},
# {'Vmag':    10.7    , 'Name':'Luhman 16',                     'Observer':' seen from Earth ', 'Notes':'Closest brown dwarfs    ', 'plot':'yes'},
# {'Vmag':    11.05   , 'Name':'star Proxima Centauri',         'Observer':' seen from Earth ', 'Notes':'2nd closest star    ', 'plot':'yes'},
# #{'Vmag':    11.8    , 'Name':'moon Phobos',                   'Observer':' seen from Earth ', 'Notes':'Maximum brightness; brightest moon of Mars  ', 'plot':'yes'},
# #{'Vmag':    12.23   , 'Name':'star R136a1',                   'Observer':' seen from Earth ', 'Notes':'Most luminous and massive star known[44]    ', 'plot':'yes'},
# #{'Vmag':    12.89   , 'Name':'moon Deimos',                   'Observer':' seen from Earth ', 'Notes':'Maximum brightness  ', 'plot':'yes'},
# {'Vmag':    12.91   , 'Name':'quasar 3C 273',                 'Observer':' seen from Earth ', 'Notes':'brightest (luminosity distance of 2.4 billion light-years)  ', 'plot':'yes'},
# {#'Vmag':    13.42   , 'Name':'moon Triton',                   'Observer':' seen from Earth ', 'Notes':'Maximum brightness[41]  ', 'plot':'yes'},
# {'Vmag':    13.65   , 'Name':'Pluto',                         'Observer':' seen from Earth ', 'Notes':'maximum brightness,[45] 725 times fainter than magnitude 6.5 naked eye skies    ', 'plot':'yes'},
# #{'Vmag':    13.9    , 'Name':'moon Titania',                  'Observer':' seen from Earth ', 'Notes':'Maximum brightness; brightest moon of Uranus    ', 'plot':'yes'},
# #{'Vmag':    14.1    , 'Name':'star WR 102',                   'Observer':' seen from Earth ', 'Notes':'Hottest known star  ', 'plot':'yes'},
# #{'Vmag':    15.4    , 'Name':'centaur Chiron',                'Observer':' seen from Earth ', 'Notes':'maximum brightness[46]  ', 'plot':'yes'},
# #{'Vmag':    15.55   , 'Name':'moon Charon',                   'Observer':' seen from Earth ', 'Notes':'maximum brightness (the largest moon of Pluto)  ', 'plot':'yes'},
# #{'Vmag':    16.8    , 'Name':'dwarf planet Makemake',         'Observer':' seen from Earth ', 'Notes':'Current opposition brightness[47]   ', 'plot':'yes'},
# #{'Vmag':    17.27   , 'Name':'dwarf planet Haumea',           'Observer':' seen from Earth ', 'Notes':'Current opposition brightness[48]   ', 'plot':'yes'},
# #{'Vmag':    18.7    , 'Name':'dwarf planet Eris',             'Observer':' seen from Earth ', 'Notes':'Current opposition brightness   ', 'plot':'yes'},
# #{'Vmag':    20.7    , 'Name':'moon Callirrhoe',               'Observer':' seen from Earth ', 'Notes':'(small ≈8 km satellite of Jupiter)[41]  ', 'plot':'yes'},
# {'Vmag':    22.     , 'Name':'Faintest objects observable in visible light with a 600 mm (24″) Ritchey-Chrétien telescope with 30 minutes of stacked images (6 subframes at 5 minutes each) using a CCD detector[49]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    22.91   , 'Name':'moon Hydra',                    'Observer':' seen from Earth ', 'Notes':'maximum brightness of Pluto moon  ', 'plot':'yes'},
# #{'Vmag':    23.38   , 'Name':'moon Nix',                      'Observer':' seen from Earth ', 'Notes':'maximum brightness of Pluto moon  ', 'plot':'yes'},
# {'Vmag':    24.     , 'Name':'Faintest objects observable with the Pan-STARRS 1.8-meter telescope using a 60 second exposure[50]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# #{'Vmag':    25.     , 'Name':'moon Fenrir',                   'Observer':' seen from Earth ', 'Notes':'(small ≈4 km satellite of Saturn)[51]   ', 'plot':'yes'},
# {'Vmag':    27.7    , 'Name':'Faintest objects observable with a single 8-meter class ground-based telescope such as the Subaru Telescope in a 10-hour image[52]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    28.2    , 'Name':'Halleys Comet',                 'Observer':' seen from Earth ', 'Notes':'in 2003 when it was 28 AU from the Sun, imaged using 3 of 4 synchronised individual scopes in the ESO Very Large Telescope array using a total exposure time of about 9 hours[53] ', 'plot':'yes'},
# {'Vmag':    28.4    , 'Name':'asteroid 2003 BH91',            'Observer':' seen from Earth orbit   ', 'Notes':'observed magnitude of ≈15-kilometer Kuiper belt object Seen by the Hubble Space Telescope (HST) in 2003, dimmest known directly-observed asteroid.  ', 'plot':'yes'},
# {'Vmag':    31.5    , 'Name':'Faintest objects observable in visible light with Hubble Space Telescope via the EXtreme Deep Field with ~23 days of exposure time collected over 10 years[54]  ', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    34.     , 'Name':'Faintest objects observable in visible light with James Webb Space Telescope[55]', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
# {'Vmag':    35.     , 'Name':'unnamed asteroid',              'Observer':'seen from Earth orbit', 'Notes':'expected magnitude of dimmest known asteroid, a 950-meter Kuiper belt object discovered by the HST passing in front of a star in 2009.[56]  ', 'plot':'yes'},
# {'Vmag':    35.     , 'Name':'star LBV 1806-20',              'Observer':'seen from Earth', 'Notes':'a luminous blue variable star, expected magnitude at visible wavelengths due to interstellar extinction ', 'plot':'yes'}
# ]

#Shortened
Vmags = [{'Vmag':    -26.74  , 'Name':'Sun',                           'Observer':' seen from Earth[17] ', 'Notes':'About 400000 times brighter than mean fullmoon', 'plot':'yes'},
{'Vmag':    -25.00  , 'Name':'Eye pain threshold', 'Observer':'', 'Notes':'', 'plot':'yes'},
{'Vmag':    -12.90  , 'Name':'Full Moon',                     'Observer':' seen from Earth at perihelion   ', 'Notes':'maximum brightness of perigee + perihelion + full moon (mean distance value is -12.74,[20] though values are about 0.18 magnitude brighter when including the opposition effect)    ', 'plot':'yes'},
{'Vmag':    -10.00  , 'Name':'Comet Ikeya–Seki',       'Observer':' seen from Earth ', 'Notes':'which was the brightest Kreutz Sungrazer of modern times[21]    ', 'plot':'yes'},
{'Vmag':    -9.50   , 'Name':'Iridium sat. flare',     'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
{'Vmag':    -7.50   , 'Name':'Supernova of 1006',             'Observer':' seen from Earth ', 'Notes':'the brightest stellar event in recorded history (7200 light-years away)[22] ', 'plot':'yes'},
{'Vmag':    -5.90   , 'Name':'ISS',   'Observer':' seen from Earth ', 'Notes':'when the ISS is at its perigee and fully lit by the Sun[24] ', 'plot':'yes'},
{'Vmag':    -4.92   , 'Name':'Venus',                         'Observer':' seen from Earth ', 'Notes':'maximum brightness[25] when illuminated as a crescent   ', 'plot':'yes'},
{'Vmag':    -4.     , 'Name':'Faintest daytime\nobservable with eye', 'Observer':' ', 'Notes':' ', 'plot':'yes'},
{'Vmag':    -2.94   , 'Name':'Jupiter',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes'},
{'Vmag':    -2.94   , 'Name':'Mars',                          'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes'},
{'Vmag':    -2.48   , 'Name':'Mercury',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness at superior conjunction (unlike Venus, Mercury is at its brightest when on the far side of the Sun, the reason being their different phase curves)[25]   ', 'plot':'yes'},
{'Vmag':    -1.66   , 'Name':'Jupiter',                       'Observer':' seen from Earth ', 'Notes':'minimum brightness[25]  ', 'plot':'yes'},
{'Vmag':    -1.47   , 'Name':'Sirius',                        'Observer':' seen from Earth ', 'Notes':'Brightest star except for the Sun at visible wavelengths[26]    ', 'plot':'yes'},
{'Vmag':    -0.55   , 'Name':'Saturn',                        'Observer':' seen from Earth ', 'Notes':'maximum brightness near opposition and perihelion when the rings are angled toward Earth[25]    ', 'plot':'yes'},
{'Vmag':    -0.3    , 'Name':'Halleys comet',                 'Observer':' seen from Earth ', 'Notes':'Expected apparent magnitude at 2061 passage ', 'plot':'yes'},
{'Vmag':    0.03    , 'Name':'Vega',                          'Observer':' seen from Earth ', 'Notes':'which was originally chosen as a definition of the zero point[29]   ', 'plot':'yes'},
{'Vmag':    3.      , 'Name':'Faintest observable with eye in city', 'Observer':' ', 'Notes':'not city, urban neightborhood', 'plot':'yes', 'lower':3., 'upper':4.},
{'Vmag':    3.44    , 'Name':'Andromeda Galaxy',              'Observer':' seen from Earth ', 'Notes':'M31[30] ', 'plot':'yes'},
{'Vmag':    4.      , 'Name':'Orion Nebula',                  'Observer':' seen from Earth ', 'Notes':'M42 ', 'plot':'yes'},
{'Vmag':    4.5     , 'Name':'M41 open cluster',              'Observer':' seen from Earth ', 'Notes':'an open cluster that may have been seen by Aristotle[32]    ', 'plot':'yes'},
{'Vmag':    4.5     , 'Name':'Sagittarius Dwarf Spheroidal Galaxy', 'Observer':' seen from Earth ', 'Notes':' ', 'plot':'yes'},
{'Vmag':    5.2     , 'Name':'Asteroid Vesta',                'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
{'Vmag':    5.38    , 'Name':'Uranus',                        'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes', 'source':'[33]'},
{'Vmag':    5.72    , 'Name':'M33 spiral galaxy',             'Observer':' seen from Earth ', 'Notes':'which is used as a test for naked eye seeing under dark skies[34][35]   ', 'plot':'yes'},
{'Vmag':    5.8     , 'Name':'gamma-ray burst GRB 080319B',   'Observer':' seen from Earth ', 'Notes':'Peak visual magnitude (the "Clarke Event") seen on Earth on March 19, 2008 from a distance of 7.5 billion light-years.  ', 'plot':'yes'},
{'Vmag':    6.49    , 'Name':'Asteroid Pallas',               'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
{'Vmag':    6.5     , 'Name':'Faintest observable w/eye good conditions', 'Observer':' ', 'Notes':'There are about 9,500 stars visible to mag 6.5.[4] ', 'plot':'yes'},
{'Vmag':    6.64    , 'Name':'Ceres',                         'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
{'Vmag':    6.75    , 'Name':'Asteroid Iris',                 'Observer':' seen from Earth ', 'Notes':'maximum brightness  ', 'plot':'yes'},
{'Vmag':    6.9     , 'Name':'Spiral Galaxy M81',             'Observer':' seen from Earth ', 'Notes':'This is an extreme naked-eyetarget that pushes human eyesight and the Bortle scale to the limit[36] ', 'plot':'yes'},
{'Vmag':    7.      , 'Name':'Extreme eye limit opt. conditions', 'Observer':' ', 'Notes':'Extreme naked-eye limit, Class 1 on Bortle scale, the darkest skies available on Earth[37]', 'plot':'yes', 'lower':7.,'upper':8.},
{'Vmag':    7.67    , 'Name':'Neptune',                       'Observer':' seen from Earth ', 'Notes':'maximum brightness[25]  ', 'plot':'yes', 'source':'[38]'},
{'Vmag':    8.94    , 'Name':'Asteroid 10 Hygiea',            'Observer':' seen from Earth ', 'Notes':'maximum brightness[42]  ', 'plot':'yes'},
{'Vmag':    9.5     , 'Name':'Faintest observable with\n7×50 binoculars', 'Observer':' ', 'Notes':'under typical conditions[43]', 'plot':'yes'},
{'Vmag':    10.7    , 'Name':'Luhman 16',                     'Observer':' seen from Earth ', 'Notes':'Closest brown dwarfs    ', 'plot':'yes'},
{'Vmag':    11.05   , 'Name':'Proxima Centauri',              'Observer':' seen from Earth ', 'Notes':'2nd closest star    ', 'plot':'yes'},
{'Vmag':    12.91   , 'Name':'Quasar 3C 273',                 'Observer':' seen from Earth ', 'Notes':'brightest (luminosity distance of 2.4 billion light-years)  ', 'plot':'yes'},
{'Vmag':    13.65   , 'Name':'Pluto',                         'Observer':' seen from Earth ', 'Notes':'maximum brightness,[45] 725 times fainter than magnitude 6.5 naked eye skies    ', 'plot':'yes'},
{'Vmag':    22.     , 'Name':'Faintest observable with\n0.6m Ritchey-Chrétien telescope', 'Observer':' ', 'Notes':'Faintest objects observable in visible light with a 600 mm (24″) Ritchey-Chrétien telescope with 30 minutes of stacked images (6 subframes at 5 minutes each) using a CCD detector[49]  ', 'plot':'yes'},
{'Vmag':    24.     , 'Name':'Faintest observable with\nPan-STARRS 1.8m telescope', 'Observer':' ', 'Notes':'Faintest objects observable with the Pan-STARRS 1.8-meter telescope using a 60 second exposure[50]', 'plot':'yes'},
{'Vmag':    27.7    , 'Name':'Faintest observable with\n8m ground telescope', 'Observer':' ', 'Notes':'Faintest objects observable with a single 8-meter class ground-based telescope such as the Subaru Telescope in a 10-hour image[52]', 'plot':'yes'},
{'Vmag':    28.4    , 'Name':'15km Asteroid\n2003 BH91',            'Observer':' seen from Earth orbit   ', 'Notes':'observed magnitude of ≈15-kilometer Kuiper belt object Seen by the Hubble Space Telescope (HST) in 2003, dimmest known directly-observed asteroid.  ', 'plot':'yes'},
{'Vmag':    31.5    , 'Name':'Faintest observable w/HST', 'Observer':' ', 'Notes':'Hubble Space Telescope via the EXtreme Deep Field with ~23 days of exposure time collected over 10 years[54]  ', 'plot':'yes'},
{'Vmag':    34.     , 'Name':'Faintest observable w/JWST', 'Observer':' ', 'Notes':'James Webb Space Telescope in Visible spectrum [55]', 'plot':'yes'},
{'Vmag':    35.     , 'Name':'1km Kuiper Asteroid',              'Observer':'seen from Earth orbit', 'Notes':'expected magnitude of dimmest known asteroid, a 950-meter Kuiper belt object discovered by the HST passing in front of a star in 2009.[56]  ', 'plot':'yes'}
]
#100MT nuclear bomb is Vmag =approx. -31

#### Detected Exoplanets by Name
# Fomalhaut b
# bet Pic b
# kap And b
# GJ 504 b
# 51 Eri b
# HIP 79098 AB b
# HN Peg b
# HR 8799 b
# HR 8799 c
# HR 8799 d
# HR 8799 e
# HR 2562 b
# HD 100546 b
# HIP 65426 b
# HIP 78530 b
# HD 95086 b
# HD 106906 b
# HD 203030 b
# AB Pic b
# Ross 458 c
# LkCa 15 b
# LkCa 15 c
# PDS 70 b
# PDS 70 c
# GSC 06214-00210 b
# GQ Lup b
# 2MASS J22362452+4751425 b
# SR 12 AB c
# GU Psc b
# 2MASS J01225093-2439505 b
# ROXs 12 b
# VHS J125601.92-125723.9 b
# 2MASS J12073346-3932539 b

#### plandb planets
#from plandb.sioslab.com with query
#"select pl_name, pl_angsep, completeness,quad_dMag_med_575NM from KnownPlanets where completeness > 0 order by completeness DESC"
simulatedPlanetVmags = [
{'Name':'RR Cae b    ', 'Vmag':  35.8    },
{'Name':'GJ 849 b    ', 'Vmag':  34.55   },
{'Name':'HD 190360 c ', 'Vmag':  26.5    },
{'Name':'55 Cnc d    ', 'Vmag':  28.69   },
{'Name':'HD 160691 c ', 'Vmag':  26.58   },
{'Name':'HD 190360 b ', 'Vmag':  26.41   },
{'Name':'47 UMa c    ', 'Vmag':  25.9    },
{'Name':'GJ 179 b    ', 'Vmag':  36.64   },
{'Name':'HD 219077 b ', 'Vmag':  25.13   },
{'Name':'psi 1 Dra B b   ', 'Vmag':  25.97   },
{'Name':'HD 154345 b ', 'Vmag':  28.75   },
{'Name':'HD 219134 h ', 'Vmag':  28.64   },
{'Name':'GJ 676 A c  ', 'Vmag':  32.2    },
{'Name':'HD 114783 c ', 'Vmag':  30.7    },
{'Name':'HD 134987 c ', 'Vmag':  28.1    },
{'Name':'GJ 317 c    ', 'Vmag':  37.21   },
{'Name':'14 Her b    ', 'Vmag':  26.82   },
{'Name':'HD 39091 b  ', 'Vmag':  24.51   },
{'Name':'ups And d   ', 'Vmag':  23.02   },
{'Name':'HD 217107 c ', 'Vmag':  27.6    },
{'Name':'bet Pic b   ', 'Vmag':  26.18   },
{'Name':'tau Cet f   ', 'Vmag':  27.18   },
{'Name':'HD 142 c    ', 'Vmag':  26.98   },
{'Name':'HD 62509 b  ', 'Vmag':  18.17   },
{'Name':'GJ 328 b    ', 'Vmag':  33.51   },
{'Name':'HD 106515 A b   ', 'Vmag':  28.41   },
{'Name':'HD 150706 b ', 'Vmag':  28.54   },
{'Name':'HD 114613 b ', 'Vmag':  25.2    },
{'Name':'HD 87883 b  ', 'Vmag':  28.88   },
{'Name':'HD 113538 c ', 'Vmag':  31.33   },
{'Name':'HD 100546 b ', 'Vmag':  29.65   },
{'Name':'HD 65216 c  ', 'Vmag':  30.29   },
{'Name':'HD 25015 b  ', 'Vmag':  31.45   },
{'Name':'47 UMa b    ', 'Vmag':  24.65   },
{'Name':'HAT-P-11 c  ', 'Vmag':  30.41   },
{'Name':'gam Cep b   ', 'Vmag':  21.67   },
{'Name':'HD 95872 b  ', 'Vmag':  32.53   },
{'Name':'GJ 832 b    ', 'Vmag':  33.91   },
{'Name':'HD 92788 c  ', 'Vmag':  29.44   },
{'Name':'47 UMa d    ', 'Vmag':  27.97   },
{'Name':'51 Eri b    ', 'Vmag':  29.35   },
{'Name':'eps Eri b   ', 'Vmag':  23.46   },
{'Name':'2MASS J21402931+1625183 A b ', 'Vmag':  46.66   },
{'Name':'HD 196067 b ', 'Vmag':  24.64   },
{'Name':'tau Cet e   ', 'Vmag':  26.23   },
{'Name':'HIP 70849 b ', 'Vmag':  35.41   },
{'Name':'HD 156279 c ', 'Vmag':  29.96   },
{'Name':'HD 192310 c ', 'Vmag':  26.95   },
{'Name':'HD 141399 e ', 'Vmag':  28.29   },
{'Name':'HD 181433 d ', 'Vmag':  30.14   },
{'Name':'HD 30562 b  ', 'Vmag':  23.09   },
{'Name':'HR 2562 b   ', 'Vmag':  30.94   },
{'Name':'HD 92987 b  ', 'Vmag':  28.96   },
{'Name':'HR 8799 d   ', 'Vmag':  30.98   },
{'Name':'HR 8799 e   ', 'Vmag':  30.85   },
{'Name':'HD 27894 d  ', 'Vmag':  31.84   },
{'Name':'HD 13724 b  ', 'Vmag':  31.1    },
{'Name':'HR 8799 c   ', 'Vmag':  31.98   },
{'Name':'GQ Lup b    ', 'Vmag':  38.57   }
]

#https://plandb.sioslab.com/docs/html/index.html
#https://plandb.sioslab.com/index.php
#https://github.com/dsavransky/plandb.sioslab.com
#https://github.com/nasavbailey/DI-flux-ratio-plot

#### Condense into lists
VmagList = list()
NameList = list()
for i in np.arange(len(Vmags)):
    VmagList.append(Vmags[i]['Vmag'])
    NameList.append(Vmags[i]['Name'])

VmagList2 = list()
NameList2 = list()
for i in np.arange(len(simulatedPlanetVmags)):
    VmagList2.append(simulatedPlanetVmags[i]['Vmag'])
    NameList2.append(simulatedPlanetVmags[i]['Name'])

#DELETE
# Vmags = [
# {'Name':'Mercury','Vmag':-2.477},
# {'Name':'Venus','Vmag':-4.916},
# #{'Name':'Earth','Vmag':-6.909},
# {'Name':'Mars','Vmag':-2.862},
# {'Name':'Jupiter','Vmag':-2.934},
# {'Name':'Saturn','Vmag':-0.552},
# {'Name':'Uranus','Vmag':5.381},
# {'Name':'Neptune','Vmag':7.701},]


#### Make Timeline Plot with all WIKIPEDIA Vmag objects
# Choose some nice levels
levels = np.tile([-5./2., 5./2., -3./2., 3./2., -1./2., 1./2.], int(np.ceil(len(Vmags)/6.)))[:len(Vmags)]

# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(figsize=(12., 8.), constrained_layout=True)
#ax.set(title="Astronomical Visual Magnitudes")
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')


tmpmarkerline, tmpstemline, tmpbaseline = ax.stem([0.,0.], [-4.3,4.3], linefmt=" ", basefmt=" ", use_line_collection=False)
plt.setp(tmpmarkerline, visible=False)
markerline, stemline, baseline = ax.stem(VmagList, levels, linefmt="b-", basefmt="k-", use_line_collection=True)
plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(Vmags)))

# annotate lines
kwargs = {'fontsize':'x-small'}
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
for d, l, r, va in zip(VmagList, levels, NameList, vert):
    ax.annotate(r, xy=(d, l), xytext=(+3.5, np.sign(l)*3.), #was -3.
                textcoords="offset points", va=va, ha="right", rotation=90., **kwargs)

# format xaxis with 4 month intervals
#ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=4))
#ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
kwargs2 = {'weight':'bold'}
ax.xaxis.set_ticks([-30.,-20.,-10.,0.,10.,20.,30.,40.])
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", **kwargs2)

# remove y axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_linewidth(2.)

#ax.margins(y=0.1)#,x=0.1)
plt.subplots_adjust(left=0.03, bottom=None, right=0.98, top=None)
ax.set_xlabel('Visual Magnitudes (mag)', weight='bold')
#plt.tight_layout()
plt.show(block=False)
fname = 'SpaceObjectVmagWIKI'
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=200)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=200)



#### Now Plot all Exoplanets from plandb
# Choose some nice levels
#levels2 = np.tile([-5./2., 5./2., -3./2., 3./2., -1./2., 1./2.], int(np.ceil(len(VmagList2)/6.)))[:len(VmagList2)]
levels2 = np.tile([-1./4., 1./4.], int(np.ceil(len(VmagList2))))[:len(VmagList2)]
markerline2, stemline2, baseline2 = ax.stem(VmagList2, levels2, linefmt="k-", basefmt="k-", use_line_collection=True)
plt.setp(markerline2, mec="k", mfc="red", zorder=3)#was "w"
markerline2.set_ydata(np.zeros(len(VmagList2)))
kwargs2 = {'weight':'bold'}
ax.xaxis.set_ticks([-30.,-20.,-10.,0.,10.,20.,30.,40.,50])
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", **kwargs2)


plt.show(block=False)
fname = 'SpaceObjectVmagWIKIwSimulatedEXOPLANET'
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=200)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=200)
