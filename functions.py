import glob
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


#---------------------------------------------

def extact_path_windows():
    pathes=[]
    for row in glob.glob('/mnt/c/Users/Mohammad/Desktop/Bsc prj/code/ILD_DB/ILD_DB_lungMasks/**', recursive=True):
        row_splt=row.split('/')
        if os.path.isdir(row):
            if row_splt[1]=='':
                continue
            if not (row_splt[-1] in['lung_mask','lung_mask_bmp','roi_mask','HRCT_pilot','bmp','bgdir','patchfile','scan_bmp','sroi']):
                if not (row_splt[-1] in ['142','154','184','53','57','8','HRCT_pilot']):
                    pathes.append(row)
    return pathes


#----------------------------------------------

def manage_HRCT_pilot(mode):
    master_path=os.getcwd()
    path_hrct_pilot='ILD_DB_lungMasks/HRCT_pilot'
    path=os.path.join(master_path,path_hrct_pilot)
    if mode=="expand":
        for row in os.listdir(path):
            src=os.path.join(path,row)
            shutil.move(src,src.replace('HRCT_pilot/',''))
        os.rmdir(path)
    elif mode=="shrink":
        HRCT_pilot =[200,201,203,204,205,206,207,208,209,210,211]
        try:
            os.mkdir(path)
        except: pass
        for row in HRCT_pilot:
            dst=os.path.join(path,str(row))
            src=dst.replace('HRCT_pilot/','')
            shutil.move(src,dst)
    else:
        print('ERROR : mode not supported!')

#--------------------------------------------------------

def manage_txt_files():
    all_path=extact_path_windows()
    for row in all_path:
        src=row.replace(subdir[1],subdir[3])
        try:
            txt_src=glob.glob(os.path.join(src,'*.txt'))[0]
            txt_dst=txt_src.replace(subdir[3],subdir[1])
            shutil.copy(txt_src,txt_dst)
        except:
            pass
    return

#-----------------------------------------------------------

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 


#------------------------------------------------------------

def cleanup():
    pathes=[]
    for row in glob.glob('/mnt/c/Users/Mohammad/Desktop/Bsc prj/code/ILD_DB/ILD_DB_lungMasks/**', recursive=True):
        row_splt=row.split('/')
        if os.path.isdir(row):
            if row_splt[1]=='':
                continue
            if (row_splt[-1] in ['bmp','bgdir','patchfile','scan_bmp','sroi']):
                pathes.append(row)
    for row in pathes:
        shutil.rmtree(row)
    return

#-----------------------------------------------------------------

def find_dst_names():
    listdirc=[]
    listdirc_tmp=extact_path_windows()
    for row in listdirc_tmp:
        tmp=row.replace('/mnt/c/Users/Mohammad/Desktop/Bsc prj/code/ILD_DB/ILD_DB_lungMasks/','')
        if tmp!='':
            listdirc.append(tmp)
    return listdirc