import os
import shutil
from data_config import *
from parameters import *
from PIL import Image, ImageDraw
import numpy as np

def remove_folder(path):
    """to remove folder if exists"""
    if os.path.exists(path):
        shutil.rmtree(path)


def fidclass(numero,classif):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():     
        if valeur == numero:
            found=True
            return cle
    if not found:
        return 'unknown'


##----------------------------------------
def tagview(fig,label,x,y,classif):
    """write text in image according to label and color"""
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[label]
    labnow=classif[label]
    if label == 'back_ground':
        x=2
        y=0        
        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow-1)%5)
        deltax=80*((labnow-1)//5)
    draw.text((x, y+deltay),label,col,font=font10)
    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)

##----------------------------------------

def normi(img):
    ### normalize image 
    tabi = np.array(img)
    tabi1=tabi-tabi.min()
    maxt=float(tabi1.max())
    if maxt==0:
        maxt=1
    tabi2=tabi1*(imageDepth/maxt)
    if imageDepth<256:
        tabi2=tabi2.astype('uint8')
    else:
        tabi2=tabi2.astype('uint16')
    return tabi2
