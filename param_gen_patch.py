
from PIL import ImageFont
import os
import numpy as np
import cv2
from PIL import Image ,ImageDraw
import shutil

#######################################################
namedirHUG = 'Implementation'
subHUG='ILD_DB_lungMasks'

toppatch= 'TOPPATCH4'  
extendir='16_set0_gci'
raw_patch=False

#patch overlapp tolerance
thrpatch = 0.8
# imageDepth=255 #number of bits used on dicom images (2 **n)
imageDepth=65535
avgPixelSpacing=0.734

dimpavx =32
dimpavy = 32
typei='bmp' #can be jpg

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
labelbg='back_ground'
locabg='anywhere'

patchesdirnametop = toppatch+'_'+extendir
patchesdirname = 'patches'
patchesNormdirname = 'patches_norm'
imagedirname='patches_jpeg'
bmpname='scan_bmp'
lungmask='lung_mask'
lungmaskbmp='bmp'
sroi='sroi'
bgdir='bgdir'
bgdirw='bgdirw'
##############################################################################################################
##############################################################################################################

#color of labels
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)

usedclassif = [
    'back_ground',
    'consolidation',
    'HC',
    'fibrosis',
    'ground_glass',
    'healthy',
    'micronodules',
    'reticulation',
    'air_trapping',
    'cysts',
    'bronchiectasis',
    'emphysema',
    'bronchial_wall_thickening',
    'early_fibrosis',
    'increased_attenuation'
    'macronodules'
    'pcp',
    'peripheral_micronodules',
    'tuberculosis']

classif ={
    'back_ground':0,
    'consolidation':1,
    'fibrosis':2,
    'HC':2,
    'ground_glass':3,
    'healthy':4,
    'micronodules':5,
    'reticulation':6,
    'air_trapping':7,
    'cysts':8,
    'bronchiectasis':9,
    'bronchial_wall_thickening':10,
    'early_fibrosis':11,
    'emphysema':12,
    'increased_attenuation':13,
    'macronodules':14,
    'pcp':15,
    'peripheral_micronodules':16,
    'tuberculosis':17}

classifc ={
    'back_ground':darkgreen,
    'consolidation':red,
    'fibrosis':blue,
    'HC':blue,
    'ground_glass':yellow,
    'healthy':green,
    'micronodules':cyan,
    'reticulation':purple,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'bronchial_wall_thickening':white,
    'early_fibrosis':white,
    'emphysema':white,
    'increased_attenuation':white,
    'macronodules':white,
    'pcp':white,
    'peripheral_micronodules':white,
    'tuberculosis':white}
##############################################################################################################
##############################################################################################################

def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]   
    return tabz, imgi

def normi(img):
    tabi = np.array(img)
    tabi1=tabi-tabi.min()
    maxt=float(tabi1.max())
    if maxt==0:
        maxt=1
    tabi2=tabi1*(imageDepth/maxt)
    tabi2=tabi2.astype('uint16')
    return tabi2

def tagview(fig,label,x,y):
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

def contour2(im,l,dimtabx,dimtaby):  
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(imgray,0,255,0)
    contours0, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis



##############################################################################################################
##############################################################################################################


def make_statistics(patchtoppath,patchpath,jpegpath):
    npatcht=0
    for npp in os.listdir(jpegpath):
        if npp.find('.txt')>0 and npp.find('nbp')==0:
            ofilep = open(jpegpath+'/'+npp, 'r')
            tp = ofilep.read()
            ofilep.close()
            numpos2=tp.find('number')
            numposend2=len(tp)
            numposdeb2 = tp.find(':',numpos2)
            nump2=tp[numposdeb2+1:numposend2].strip()
            numpn2=int(nump2)
            npatcht=npatcht+numpn2

    ofilepwt = open(jpegpath+'/totalnbpat.txt', 'w')
    ofilepwt.write('number of patches: '+str(npatcht))
    ofilepwt.close()


    dirlabel=os.walk(patchpath).next()[1]
    eftpt=os.path.join(patchtoppath,'totalnbpat.txt')
    filepwt = open(eftpt, 'w')
    ntot=0
    labellist=[]
    localist=[]
    for dirnam in dirlabel:
        dirloca=os.path.join(patchpath,dirnam)
        listdirloca=os.listdir(dirloca)
        label=dirnam
        loca=''
        if dirnam not in labellist:
            labellist.append(dirnam)
        for dlo in listdirloca:
            loca=dlo
            if dlo not in localist:      
                localist.append(dlo)
            if label=='' or loca =='':
                print('not found:',dirnam)        
            # subdir = os.path.join(dirloca,loca)
            n=0
            listcwd=os.listdir(dirloca)
            for ff in listcwd:
                if ff.find(typei) >0 :
                    n+=1
                    ntot+=1
            filepwt.write('label: '+label+' localisation: '+loca+' number of patches: '+str(n)+'\n') 
    filepwt.close() 

#######################################################



def make_log(patchtoppath,jpegpath,listslice):
    eflabel=os.path.join(patchtoppath,'lislabel.txt')
    mflabel=open(eflabel,"w")
    mflabel.write('label  _  localisation\n')
    mflabel.write('======================\n')
    categ=os.listdir(jpegpath)
    for f in categ:
        if f.find('.txt')>0 and f.find('nb')==0:
            ends=f.find('.txt')
            debs=f.find('_')
            sln=f[debs+1:ends]
            listlabel={}
            for f1 in categ:
                if  f1.find(sln+'_')==0 and f1.find('.txt')>0:
                    debl=f1.find('slice_')
                    debl1=f1.find('_',debl+1)
                    debl2=f1.find('_',debl1+1)
                    endl=f1.find('.txt')
                    j=0
                    while f1.find('_',endl-j)!=-1:
                        j-=1
                    label=f1[debl2+1:endl-j-2]
                    ffle1=os.path.join(jpegpath,f1)
                    fr1=open(ffle1,'r')
                    t1=fr1.read()
                    fr1.close()
                    debsp=t1.find(':')
                    endsp=  t1.find('\n')
                    np=int(t1[debsp+1:endsp])
                    if label in listlabel:
                        listlabel[label]=listlabel[label]+np
                    else:
                        listlabel[label]=np
            listslice.append(sln)
            ffle=os.path.join(jpegpath,f)
            fr=open(ffle,'r')
            t=fr.read()
            fr.close()
            debs=t.find(':')
            ends=len(t)
            nump= t[debs+1:ends]
            mflabel.write(sln+' number of patches: '+nump+'\n')
            for l in listlabel:
                if l !=labelbg+'_'+locabg:
                    mflabel.write(l+' '+str(listlabel[l])+'\n')
            mflabel.write('---------------------'+'\n')
    mflabel.close()