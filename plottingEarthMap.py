
#Attempt #1 using geoplot
# import geopandas as gpd
# import geoplot as gplt
# import geoplot.crs as gcrs

# round_earth = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# round_earth['gdp_pp'] = round_earth['gdp_md_est'] / round_earth['pop_est']

# ax = gplt.polyplot(round_earth, projection=gcrs.Orthographic(), figsize=(10,10))
# ax.set_global()
# ax.outline_patch.set_visible(True)


#BASEMAP example I can't get to work
# import datetime,matplotlib,time
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import numpy as np
# from gifly import gif_maker

# # plt.ion() allows python to update its figures in real-time
# plt.ion()
# fig = plt.figure(figsize=(9,6))

# # set the latitude angle steady, and vary the longitude. You can also reverse this to
# # create a rotating globe latitudinally as well
# lat_viewing_angle = [20.0,20.0]
# lon_viewing_angle = [-180,180]
# rotation_steps = 150
# lat_vec = np.linspace(lat_viewing_angle[0],lat_viewing_angle[0],rotation_steps)
# lon_vec = np.linspace(lon_viewing_angle[0],lon_viewing_angle[1],rotation_steps)

# # for making the gif animation
# gif_indx = 0

# # define color maps for water and land
# ocean_map = (plt.get_cmap('ocean'))(210)
# cmap = plt.get_cmap('gist_earth')

# # loop through the longitude vector above
# for pp in range(0,len(lat_vec)):    
#     plt.cla()
#     m = Basemap(projection='ortho', 
#               lat_0=lat_vec[pp], lon_0=lon_vec[pp])
    
#     # coastlines, map boundary, fill continents/water, fill ocean, draw countries
#     m.drawmapboundary(fill_color=ocean_map)
#     m.fillcontinents(color=cmap(200),lake_color=ocean_map)
#     m.drawcoastlines()
#     m.drawcountries()

#     #show the plot, introduce a small delay to allow matplotlib to catch up
#     plt.show(block=False)
#     plt.pause(0.01)
#     # iterate to create the GIF animation
#     #gif_maker('basemap_rotating_globe.gif','./png_dir/',gif_indx,len(lat_vec)-1,dpi=90)
#     gif_indx+=1

#     

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import matplotlib.path
import matplotlib.ticker
from matplotlib.transforms import BboxTransform, Bbox
import numpy as np


# When drawing the flag, we can either use white filled land, or be a little
# more fancy and use the Natural Earth shaded relief imagery.
filled_land = True


def olive_path():
    """
    Return a Matplotlib path representing a single olive branch from the
    UN Flag. The path coordinates were extracted from the SVG at
    https://commons.wikimedia.org/wiki/File:Flag_of_the_United_Nations.svg.

    """
    olives_verts = np.array(
        [[0,   2,   6,   9,  30,  55,  79,  94, 104, 117, 134, 157, 177,
          188, 199, 207, 191, 167, 149, 129, 109,  87,  53,  22,   0, 663,
          245, 223, 187, 158, 154, 150, 146, 149, 154, 158, 181, 184, 197,
          181, 167, 153, 142, 129, 116, 119, 123, 127, 151, 178, 203, 220,
          237, 245, 663, 280, 267, 232, 209, 205, 201, 196, 196, 201, 207,
          211, 224, 219, 230, 220, 212, 207, 198, 195, 176, 197, 220, 239,
          259, 277, 280, 663, 295, 293, 264, 250, 247, 244, 240, 240, 243,
          244, 249, 251, 250, 248, 242, 245, 233, 236, 230, 228, 224, 222,
          234, 249, 262, 275, 285, 291, 295, 296, 295, 663, 294, 293, 292,
          289, 294, 277, 271, 269, 268, 265, 264, 264, 264, 272, 260, 248,
          245, 243, 242, 240, 243, 245, 247, 252, 256, 259, 258, 257, 258,
          267, 285, 290, 294, 297, 294, 663, 285, 285, 277, 266, 265, 265,
          265, 277, 266, 268, 269, 269, 269, 268, 268, 267, 267, 264, 248,
          235, 232, 229, 228, 229, 232, 236, 246, 266, 269, 271, 285, 285,
          663, 252, 245, 238, 230, 246, 245, 250, 252, 255, 256, 256, 253,
          249, 242, 231, 214, 208, 208, 227, 244, 252, 258, 262, 262, 261,
          262, 264, 265, 252, 663, 185, 197, 206, 215, 223, 233, 242, 237,
          237, 230, 220, 202, 185, 663],
         [8,   5,   3,   0,  22,  46,  46,  46,  35,  27,  16,  10,  18,
          22,  28,  38,  27,  26,  33,  41,  52,  52,  52,  30,   8, 595,
          77,  52,  61,  54,  53,  52,  53,  55,  55,  57,  65,  90, 106,
          96,  81,  68,  58,  54,  51,  50,  51,  50,  44,  34,  43,  48,
          61,  77, 595, 135, 104, 102,  83,  79,  76,  74,  74,  79,  84,
          90, 109, 135, 156, 145, 133, 121, 100,  77,  62,  69,  67,  80,
          92, 113, 135, 595, 198, 171, 156, 134, 129, 124, 120, 123, 126,
          129, 138, 149, 161, 175, 188, 202, 177, 144, 116, 110, 105,  99,
          108, 116, 126, 136, 147, 162, 173, 186, 198, 595, 249, 255, 261,
          267, 241, 222, 200, 192, 183, 175, 175, 175, 175, 199, 221, 240,
          245, 250, 256, 245, 233, 222, 207, 194, 180, 172, 162, 153, 154,
          171, 184, 202, 216, 233, 249, 595, 276, 296, 312, 327, 327, 327,
          327, 308, 284, 262, 240, 240, 239, 239, 242, 244, 247, 265, 277,
          290, 293, 296, 300, 291, 282, 274, 253, 236, 213, 235, 252, 276,
          595, 342, 349, 355, 357, 346, 326, 309, 303, 297, 291, 290, 297,
          304, 310, 321, 327, 343, 321, 305, 292, 286, 278, 270, 276, 281,
          287, 306, 328, 342, 595, 379, 369, 355, 343, 333, 326, 318, 328,
          340, 349, 366, 373, 379, 595]]).T
    olives_codes = np.array([1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 2, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4,
                             4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 4, 79], dtype=np.uint8)

    return matplotlib.path.Path(olives_verts, olives_codes)



blue = '#4b92db'

# We're drawing a flag with a 3:5 aspect ratio.
fig = plt.figure(figsize=[7.5, 4.5], facecolor=blue)

# Set up the Azimuthal Equidistant and Plate Carree projections
# for later use.
az_eq = ccrs.AzimuthalEquidistant(central_latitude=90)
pc = ccrs.PlateCarree()

# Pick a suitable location for the map (which is in an Azimuthal
# Equidistant projection).
ax = fig.add_axes([0.25, 0.24, 0.5, 0.54], projection=az_eq)

# The background patch is not needed in this example.
ax.patch.set_facecolor('none')
# The Axes frame produces the outer meridian line.
for spine in ax.spines.values():
    spine.update({'edgecolor': 'white', 'linewidth': 2})

# We want the map to go down to -60 degrees latitude.
ax.set_extent([-180, 180, -60, 90], ccrs.PlateCarree())

# Importantly, we want the axes to be circular at the -60 latitude
# rather than cartopy's default behaviour of zooming in and becoming
# square.
_, patch_radius = az_eq.transform_point(0, -60, pc)
circular_path = matplotlib.path.Path.circle(0, patch_radius)
ax.set_boundary(circular_path)

if filled_land:
    ax.add_feature(
        cfeature.LAND, facecolor='white', edgecolor='none')
else:
    ax.stock_img()

gl = ax.gridlines(crs=pc, linewidth=2, color='white', linestyle='-')
# Meridians every 45 degrees, and 4 parallels.
#gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(-180, 181, 45))
#parallels = np.arange(-30, 70, 30)
#gl.ylocator = matplotlib.ticker.FixedLocator(parallels)

# Now add the olive branches around the axes. We do this in normalised
# figure coordinates
olive_leaf = olive_path()

olives_bbox = Bbox.null()
olives_bbox.update_from_path(olive_leaf)


plt.show(block=False)



