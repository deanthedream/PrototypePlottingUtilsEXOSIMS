"""
Calculate the minimum separation or minimum change in brightness in order for a planet to be of a certain type

Written By: Dean Keithly
Written On: 3/22/2019
"""

#Add in class stuff up here later

"""
This analysis can be separated into 3 sub-cases; Edge on observations, perpendicular to orbital plane, and an inclination in between
"""
def pTypeProbs_EdgeOn(sInd, s, dmag, pType, wavelength=565., D=4., angUncert=3*565*1e-9/4.,\
    photometricUncert=0.01, optimism='pessimistic'):
    """
    Args:
        sInd (integer) - index of the star
        s (float) - separation observed
        dmag (float) - delta magnitude between planet and star observed
        pType (dict?) - a dictionary describing the type of planet you are looking for
        wavelength (float) - wavelength of observation
        D (float) - mirror diameter
        angUncert (float) - uncertainty in angular separation as +/- (astrometric uncertainty), default based on HabEx Interim report 3*lambda/D in rad
        photometricUncert (float) - uncertainty in planet-Star magnitude as +/-% (photometric uncertainty), default based on HabEx interim report
    Return:
        wt (float) - minimum time to wait in days
    """

    return None


def minWaitTime_EdgeOn(sInd, s, dmag, pType, wavelength=565., D=4., angUncert=3*565*1e-9/4.,\
    photometricUncert=0.01, optimism='pessimistic'):
    """
    Args:
        sInd (integer) - index of the star
        s (float) - separation observed
        dmag (float) - delta magnitude between planet and star observed
        pType (dict?) - a dictionary describing the type of planet you are looking for
        wavelength (float) - wavelength of observation
        D (float) - mirror diameter
        angUncert (float) - uncertainty in angular separation as +/- (astrometric uncertainty), default based on HabEx Interim report 3*lambda/D in rad
        photometricUncert (float) - uncertainty in planet-Star magnitude as +/-% (photometric uncertainty), default based on HabEx interim report
        optimism (string) - should we use uncertainty bounds that maximize property 'optimistic', nominal 'nominal', or minimize property 'pessimistic'
    Return:
        wt (float) - minimum time to wait in days
    """
    #Figuring out what to do here
    #wt
    return None