import glob, shutil
import os
os.system('find -L $DIR -maxdepth 1 -type l -delete')


fles = glob.glob('./*.png.png')
for f in fles:
    shutil.move(f, f[:-4])