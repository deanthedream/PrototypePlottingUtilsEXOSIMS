#gifmaker

import glob
from PIL import Image
import numpy as np
import imageio

# filepaths
fp_in = "./EjectaParticleImages/image*.png"
fp_out = "./image.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
myGlob = glob.glob(fp_in)
myInds = list()
for i in np.arange(len(myGlob)):
    myInds.append(int(glob.glob(fp_in)[i].split("/")[2].split(".")[0][5:]))
argSortFiles = np.argsort(np.asarray(myInds))


# imgs = (Image.open(myGlob[i]) for i in argSortFiles)
# img = next(imgs)  # extract first image from iterator
# img.save(fp=fp_out, format='GIF', append_images=imgs,
#          save_all=True, duration=33, loop=0,optimize=True)


#images = []    
#for subdir, dirs, files in os.walk(root):
#    for file in files:
#        images.append(imageio.imread(os.path.join(root,file)))
images = list()
for i in np.arange(len(myGlob)):
    images.append(imageio.imread(myGlob[argSortFiles[i]]))

#savepath = r'path_to_save_folder'
imageio.mimsave('./movie.mp4', images)

