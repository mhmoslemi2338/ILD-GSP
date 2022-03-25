import os
import shutil
import cv2
import numpy as np
from parameters import *
from PIL import Image ,ImageDraw

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

def normi(img,ddepth=8):
    tabi = np.array(img)
    tabi1=tabi-tabi.min()
    maxt=float(tabi1.max())
    if maxt==0:
        maxt=1
    MAX_SIZE=2**ddepth
    tabi2=tabi1*(MAX_SIZE/maxt)
    if ddepth==8:
        tabi2=tabi2.astype('uint8')
    else:
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


##############################################################


def subfile_handler(files,mode):
    target=['142','154','184','53','57','8','HRCT_pilot']
    if mode=='start':
        for row in target:
            master_path=os.getcwd()
            path_now=os.path.join(master_path,'ILD_DB/ILD_DB_lungMasks',row)
            try:
                for idx,row2 in enumerate(os.listdir(path_now),start=1):
                    src=os.path.join(path_now,row2)
                    dst=src.rsplit("/",1)[0]+2*str(0)+str(idx)
                    if row=='HRCT_pilot':
                        dst=dst.replace(row,'200')
                    files.append([src,dst])

                    shutil.move(src,dst)
                    shutil.move(src.replace(subHUG,subHUG_txt),dst.replace(subHUG,subHUG_txt))
                os.rmdir(path_now)
                os.rmdir(path_now.replace(subHUG,subHUG_txt))
            except: pass
        return files
    elif mode=='end':
        for row in files:
            try: 
                os.mkdir(row.rsplit("/",1)[0])
                os.mkdir((row.rsplit("/",1)[0]).replace(subHUG,subHUG_txt))
            except: pass
            try:
                shutil.move(row[1],row[0])
                shutil.move((row[1]).replace(subHUG,subHUG_txt),(row[0]).replace(subHUG,subHUG_txt))
            except: pass
    else:
        print('ERROR : mode not supported!')


##############################################################


def file_stat(pp):
    labels=[]
    for row in os.listdir(pp):
        if row == '.DS_Store':
            continue
        tmp=os.listdir(os.path.join(pp,row))
        labels.append([row,len(tmp)])

    labels.sort(key=lambda x:x[1],reverse=True)
    return labels
