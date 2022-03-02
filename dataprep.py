"""generate patches from DICOM database equalization"""
import os
import numpy as np
import shutil
from pydicom import dcmread 
import cv2
from parameters import *
from data_config import *
from functions import *

#########################################################
#full path names
try:
    shutil.rmtree(toppatch+extendir)
except: pass
manage_txt_files()
manage_HRCT_pilot(mode='expand')

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!') 
### create patch and name of jpeg directory 
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpath=os.path.join(patchtoppath,patchesdirname)
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
jpegpath=os.path.join(patchtoppath,imagedirname)
#### prepare patch
if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)   
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)   
if not os.path.isdir(patchNormpath):
    os.mkdir(patchNormpath)  
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)   
######
eferror=os.path.join(patchtoppath,'genepatcherrortop.txt')
errorfile = open(eferror, 'w')
eflabel=os.path.join(patchtoppath,'lislabel.txt')
mflabel=open(eflabel,"w")
#########################################################


def genebmp(dirName):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    remove_folder(bmp_dir)    
    os.mkdir(bmp_dir)
    bgdirf = os.path.join(dirName, bgdir)
    remove_folder(bgdirf)    
    os.mkdir(bgdirf)
    lung_dir = os.path.join(dirName, lungmask)
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    
    remove_folder(lung_bmp_dir)
    os.mkdir(lung_bmp_dir)
   
    #list dcm files
    fileList = [name for name in os.listdir(dirName) if ".dcm" in name.lower()]
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]
    for filename in fileList:
        #if ".dcm" in filename.lower():  # check whether the file's DICOM
        FilesDCM =(os.path.join(dirName,filename)) 
        ds = dcmread(FilesDCM)
        dsr= ds.pixel_array          
        dsr= dsr-dsr.min()
        c=float(imageDepth)/dsr.max()
        dsr=dsr*c
        if imageDepth <256:
            dsr=dsr.astype('uint8')
        else:
            dsr=dsr.astype('uint16')
            #resize the dicom to have always the same pixel/mm
        fxs=float(ds.PixelSpacing[0])/avgPixelSpacing   
        scanNumber=int(ds.InstanceNumber)
        endnumslice=filename.find('.dcm')                   
        imgcore=filename[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
        bmpfile=os.path.join(bmp_dir,imgcore)
        dim=tuple(np.int32((np.int64(dsr.shape)*fxs)))
        dsrresize1= cv2.resize(dsr,dim,interpolation=cv2.INTER_NEAREST)
        namescan=os.path.join(sroidir,imgcore)                   
        textw='n: '+f+' scan: '+str(scanNumber)
        tablscan=cv2.cvtColor(dsrresize1,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(namescan, tablscan)
        tagviews(namescan,textw,0,20)  
        if globalHist:
            if globalHistInternal:
                dsrresize = normi(dsrresize1) 
            else:
                dsrresize = cv2.equalizeHist(dsrresize1) 
        else:
            dsrresize=dsrresize1
            
        cv2.imwrite(bmpfile,dsrresize)
        dimtabx=dsrresize.shape[0]
        dimtaby=dimtabx

    for lungfile in lunglist:
        #if ".dcm" in lungfile.lower():  # check whether the file's DICOM
        lungDCM =os.path.join(lung_dir,lungfile)  
        dslung = dcmread(lungDCM)
        dsrlung= dslung.pixel_array  

        dsrlung= dsrlung-dsrlung.min()
        if dsrlung.max()>0:
            c=float(imageDepth)/dsrlung.max()
        else:
            c=0
        dsrlung=dsrlung*c
        if imageDepth <256:
            dsrlung=dsrlung.astype('uint8')
        else:
            dsrlung=dsrlung.astype('uint16')
        
        fxslung=float(dslung.PixelSpacing[0])/avgPixelSpacing 
        scanNumber=int(dslung.InstanceNumber)
        endnumslice=lungfile.find('.dcm')                   
        lungcore=lungfile[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
        lungcoref=os.path.join(lung_bmp_dir,lungcore)
        dim=tuple(np.int32((np.int64(dsrlung.shape)*fxslung)))
        lungresize= cv2.resize(dsrlung,dim,interpolation=cv2.INTER_NEAREST)            
        lungresize = cv2.blur(lungresize,(5,5))                 
        np.putmask(lungresize,lungresize>0,100)
        cv2.imwrite(lungcoref,lungresize)
        bgdirflm=os.path.join(bgdirf,lungcore)
        cv2.imwrite(bgdirflm,lungresize)
    return [dimtabx,dimtaby]
    

    
def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]   
    return tabz, imgi
    

def pavbg(namedirtopcf,dx,dy,px,py):
    print('generate back-ground for :',f)
    bgdirf = os.path.join(namedirtopcf, bgdir)
    patchpathc=os.path.join(namedirtopcf,bmpname)
   
    lbmp=os.listdir(patchpathc)
    listbg = os.listdir(bgdirf)
    pxy=float(px*py) 
    for lm in listbg:
        nbp=0
        tabp = np.zeros((dx, dy), dtype='uint8')
        slicenumber=rsliceNum(lm,'_','.bmp')
        nambmp=os.path.join(patchpathc,lm)
        namebg=os.path.join(bgdirf,lm)
        origbg = Image.open(namebg,'r')
        origbl= origbg.convert('L')
       
        for l in lbmp:
          slicen=rsliceNum(l,'_','.bmp')

          if slicen==slicenumber and slicenumber in listsliceok:
              nambmp=os.path.join(patchpathc,l)
              origbmp = Image.open(nambmp,'r')
              origbmpl= origbmp.convert('L')
              tabf=np.array(origbl)
              imagemax=origbl.getbbox()
              min_val=np.min(origbl)
              max_val=np.max(origbl)

              tabfc = np.copy(tabf)
              nz= np.count_nonzero(tabf)
              if nz>0 and min_val!=max_val:
                np.putmask(tabf,tabf>0,1)
                atabf = np.nonzero(tabf)
                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()
              else:
                xmin=0
                xmax=0
                ymin=0
                ymax=0             
              i=xmin
              while i <= xmax:
                        j=ymin
                        while j<=ymax:
                            tabpatch=tabf[j:j+py,i:i+px]
                            area= tabpatch.sum()
                            if float(area)/pxy >thrpatch:
                                 crorig = origbmpl.crop((i, j, i+px, j+py))
                                 imagemax=crorig.getbbox()
                                 nim= np.count_nonzero(crorig)
                                 if nim>0:
                                     min_val=np.min(crorig)
                                     max_val=np.max(crorig)         
                                 else:
                                     min_val=0
                                     max_val=0  
                                 if imagemax!=None and min_val!=max_val:               
                                    nbp+=1
                                    try:
                                        _=int(f)
                                        nampa='/'+labelbg+'/'+locabg+'/'+f+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei 
                                    except:
                                        nampa='/'+labelbg+'/'+locabg+'/'+f.replace('/','_')+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei 
                                    crorig.save(patchpath+nampa)
                                    imgray =np.array(crorig)
                                    if normiInternal:
                                        tabi2 = normi(imgray) 
                                    else:
                                        tabi2 = cv2.equalizeHist(imgray)  
                                      
                                    cv2.imwrite(patchNormpath+nampa, tabi2)
                                
                                    x=0
                                    #we draw the rectange
                                    while x < px:
                                        y=0
                                        while y < py:
                                            tabp[y+j][x+i]=150
                                            if x == 0 or x == px-1 :
                                                y+=1
                                            else:
                                                y+=py-1
                                        x+=1
                                    #we cancel the source
                                    tabf[j:j+py,i:i+px]=0                           
                            j+=1
                        i+=1
                
              tabpw =tabfc+tabp
              try:
                _=int(f)
                cv2.imwrite(jpegpath+'/'+f+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'.jpg', tabpw) 
                mfl=open(jpegpath+'/'+f+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_1.txt',"w")
              except:
                cv2.imwrite(jpegpath+'/'+f.replace('/','_')+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'.jpg', tabpw) 
                mfl=open(jpegpath+'/'+f.replace('/','_')+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_1.txt',"w")

              mfl.write('#number of patches: '+str(nbp)+'\n')
              mfl.close()
              break

def contour2(im,l):  
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(imgray,0,255,0)
    contours0, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis
   
###
  
def pavs (imgi,tab,dx,dy,px,py,namedirtopcf,jpegpath,patchpath,thr,\
    iln,f,label,loca,typei,errorfile):
    """ generate patches from ROI"""
    if label == 'fibrosis':
        label='HC'
    vis=contour2(imgi,label)         
    bgdirf = os.path.join(namedirtopcf, bgdir)    
    patchpathc=os.path.join(namedirtopcf,bmpname)
    # patchpathc path to scan mage bmp   
    contenujpg = os.listdir(patchpathc)
    debnumslice=iln.find('_')+1
    endnumslice=iln.find('_',debnumslice)
    slicenumber=int(iln[debnumslice:endnumslice])

    tabp = np.zeros((dx, dy), dtype='i')

    np.putmask(tab,tab>0,100)
    tabf = np.copy(tab)
    np.putmask(tabf,tabf>0,1)
    pxy=float(px*py)
    nbp=0
    strpac=''
    errorliststring=[]

    lung_dir1 = os.path.join(namedirtopcf, lungmask)
    lung_bmp_dir = os.path.join(lung_dir1,lungmaskbmp)
    lunglist = os.listdir(lung_bmp_dir)
    atabf = np.nonzero(tabf)
    xmin=atabf[1].min()
    xmax=atabf[1].max()
    ymin=atabf[0].min()
    ymax=atabf[0].max()
    found=False
    for  n in contenujpg:   
        
        slicescan=rsliceNum(n,'_','.'+typei)
        if slicescan==slicenumber:
            found=True
            namebmp=os.path.join(patchpathc,n)
            namescan=os.path.join(sroidir,n)   
            orig = Image.open(namebmp)
            orign = Image.open(namescan)
            imscanc= orign.convert('RGB')
           
            tablscan = np.array(imscanc)
            imn=cv2.add(vis,tablscan)
            imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
            cv2.imwrite(namescan,imn)

            for lm in lunglist: # scan lung mask
                slicelung=rsliceNum(n,'_','.'+typei)
                if slicelung==slicenumber:
                    #look in lung maask the name of slice
                    namebg=os.path.join(bgdirf,lm)
                    #find the same name in bgdir directory
                    tabhc=cv2.imread(namebg,0)
                    np.putmask(tabhc,tabhc>0,100)
                    imgray = cv2.cvtColor(imgi,cv2.COLOR_BGR2GRAY)
                    tabf=np.array(imgray)
                    np.putmask(tabf,tabf>0,100)
                    mask=cv2.bitwise_not(tabf)
                    outy=cv2.bitwise_and(tabhc,mask)                                        
                    cv2.imwrite(namebg,outy)
                    break
                    
            tagview(namescan,label,0,100,classif)
            if slicenumber not in listsliceok:
                listsliceok.append(slicenumber )
            i=xmin
            np.putmask(tabf,tabf==1,0)
            np.putmask(tabf,tabf>0,1)
            while i <= xmax:
                j=ymin
                while j<=ymax:
                    tabpatch=tabf[j:j+py,i:i+px]
                    area= tabpatch.sum()  
                    targ=float(area)/pxy

                    if targ >thr:
                    #good patch                                 
                        crorig = orig.crop((i, j, i+px, j+py))
                        #detect black pixels
                        imagemax=crorig.getbbox()
                        min_val=np.min(crorig)
                        max_val=np.max(crorig)

                        if imagemax==None or min_val==max_val:
                            errortext='black or mono level pixel  in: '+ f+' '+ iln+'\n'
                            if errortext not in errorliststring:
                                errorliststring.append(errortext)
                                print(errortext)
                        else:
                            nbp+=1
                            nampa='/'+label+'/'+loca+'/'+f+'_'+iln+'_'+str(nbp)+'.'+typei 

                            try:
                                _=int(f)
                                nampa='/'+labelbg+'/'+locabg+'/'+f+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei 
                            except:
                                nampa='/'+labelbg+'/'+locabg+'/'+f.replace('/','_')+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei 
                            crorig.save(patchpath+nampa) 
                            #normalize patches and put in patches_norm
                            imgray =np.array(crorig)
                            if normiInternal:
                                tabi2 = normi(imgray) 
                            else:
                                tabi2 = cv2.equalizeHist(imgray)  
                            cv2.imwrite(patchNormpath+nampa, tabi2)
                            strpac=strpac+str(i)+' '+str(j)+'\n'
                            x=0
                            #we draw the rectange
                            while x < px:
                                y=0
                                while y < py:
                                    tabp[y+j][x+i]=150
                                    if x == 0 or x == px-1 :
                                        y+=1
                                    else:
                                        y+=py-1
                                x+=1
                            #we cancel the source
                            if label not in labelEnh:
                                tabf[j:j+py,i:i+px]=0
                            else:
                                tabf[j:j+py/2,i:i+px/2]=0                          
                    j+=1
                i+=1
            break
    
    if not found:
        print('ERROR image not found '+namedirtopcf+'/'+bmpname+'/'+str(slicenumber))            
        errorfile.write('ERROR image not found '+namedirtopcf+'/'+bmpname+'/'+str(slicenumber)+'\n')

    tabp =tab+tabp
    try:
        _=int(f)
        f_tmp=f
    except: 
        f_tmp=f.replace('/','_')
    mfl=open(jpegpath+'/'+f_tmp+'_'+iln+'.txt',"w")
    mfl.write('#number of patches: '+str(nbp)+'\n'+strpac)
    mfl.close()
    cv2.imwrite(jpegpath+'/'+f_tmp+'_'+iln+'.jpg', tabp)
    if len(errorliststring) >0:
        for l in errorliststring:
            errorfile.write(l)
    return nbp,tabp


def fileext(namefile,curdir,patchpath):
    listlabel=[labelbg+'_'+locabg]
    plab=os.path.join(patchpath,labelbg)
    ploc=os.path.join(plab,locabg) 
    plabNorm=os.path.join(patchNormpath,labelbg)
    plocNorm=os.path.join(plabNorm,locabg) 
    if not os.path.exists(plab):
        os.mkdir(plab)
    if not os.path.exists(plabNorm):
        os.mkdir(plabNorm)
    if not os.path.exists(ploc):
        os.mkdir(ploc)
    if not os.path.exists(plocNorm):
        os.mkdir(plocNorm)

    ofi = open(namefile, 'r')
    t = ofi.read()
    ofi.close()
    # nslice = t.count('slice')
    # numbercon = t.count('contour')
    nset=0
    spapos=t.find('SpacingX')
    coefposend=t.find('\n',spapos)
    coefposdeb = t.find(' ',spapos)
    coef=t[coefposdeb:coefposend]
    coefi=float(coef)
    labpos=t.find('label')
    while (labpos !=-1):
        labposend=t.find('\n',labpos)
        labposdeb = t.find(' ',labpos)
        
        label=t[labposdeb:labposend].strip()
        if label.find('/')>0:
            label=label.replace('/','_')
    
        locapos=t.find('loca',labpos)
        locaposend=t.find('\n',locapos)
        locaposdeb = t.find(' ',locapos)
        loca=t[locaposdeb:locaposend].strip()
 
        if loca.find('/')>0:
            loca=loca.replace('/','_')
        if label=='fibrosis':
            label='HC'
        if label not in listlabel:
            plab=os.path.join(patchpath,label)
            ploc=os.path.join(plab,loca) 
            plabNorm=os.path.join(patchNormpath,label)
            plocNorm=os.path.join(plabNorm,loca) 
            listlabel.append(label+'_'+loca)     
            listlabeld=os.listdir(patchpath)
            if label not in listlabeld:
                os.mkdir(plab)
                os.mkdir(plabNorm)
            listlocad=os.listdir(plab)
            if loca not in listlocad:
                os.mkdir(ploc)
                os.mkdir(plocNorm)         

        condslap=True
        slapos=t.find('slice',labpos)
        while (condslap==True):
            slaposend=t.find('\n',slapos)
            slaposdeb=t.find(' ',slapos)
            slice=t[slaposdeb:slaposend].strip()
            nbpoint=0
            nbppos=t.find('nb_point',slapos)     
            conend=True
            while (conend):
                nset=nset+1
                nbpoint=nbpoint+1
                nbposend=t.find('\n',nbppos)
                tabposdeb=nbposend+1
                
                slaposnext=t.find('slice',slapos+1)
                nbpposnext=t.find('nb_point',nbppos+1)
                labposnext=t.find('label',labpos+1)
                #last contour in file
                if nbpposnext==-1:
                    tabposend=len(t)-1
                else:
                    tabposend=nbpposnext-1
                #minimum between next contour and next slice
                if (slaposnext >0  and nbpposnext >0):
                    tabposend=min(nbpposnext,slaposnext)-1 
                #minimum between next contour and next label
                if (labposnext>0 and labposnext<nbpposnext):
                    tabposend=labposnext-1

                nametab=curdir+'/patchfile/slice_'+str(slice)+'_'+str(label)+'_'+str(loca)+'_'+str(nbpoint)+'.txt'

                mf=open(nametab,"w")
                mf.write('#label: '+label+'\n')
                mf.write('#localisation: '+loca+'\n')
                mf.write(t[tabposdeb:tabposend])
                mf.close()
                nbppos=nbpposnext 
                #condition of loop contour
                if (slaposnext >1 and slaposnext <nbpposnext) or\
                    (labposnext >1 and labposnext <nbpposnext) or\
                    nbpposnext ==-1:
                    conend=False
            slapos=t.find('slice',slapos+1)
            labposnext=t.find('label',labpos+1)
            #condition of loop slice
            if slapos ==-1 or\
            (labposnext >1 and labposnext < slapos ):
                condslap = False
        labpos=t.find('label',labpos+1)
    return(listlabel,coefi)

# listdirc= (os.listdir(namedirtopc))
listdirc=find_dst_names()
npat=0
for f in listdirc:
    print('work on:',f)
    nbpf=0
    listsliceok=[]
    posp=f.find('.',0)
    posu=f.find('_',0)
    namedirtopcf=namedirtopc+'/'+f
      
    if os.path.isdir(namedirtopcf):    
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)

    remove_folder(namedirtopcf+'/patchfile')
    os.mkdir(namedirtopcf+'/patchfile')
    #namedirtopcf = final/ILD_DB_txtROIs/35
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
        fif=False
        [dimtabx,dimtaby]=genebmp(namedirtopcf)

        for f1 in contenudir:
            if f1.find('.txt') >0 and (f1.find('CT')==0 or f1.find('Tho')==0):
                npat+=1
                fif=True
                fileList =f1
                pathf1=namedirtopcf+'/'+fileList            
                labell,coefi =fileext(pathf1,namedirtopcf,patchpath)
                break

        if not fif:
             print('ERROR: no ROI txt content file', f)
             errorfile.write('ERROR: no ROI txt content file in: '+ f+'\n')
        
        listslice= os.listdir(namedirtopcf+'/patchfile') 
        listcore =[]
        for l in listslice:
            il1=l.find('.',0)
            j=0
            while l.find('_',il1-j)!=-1:
                j-=1
            ilcore=l[0:il1-j-1]
            if ilcore not in listcore:
                listcore.append(ilcore)
        for c in listcore:
            ftab=True
            tabzc = np.zeros((dimtabx, dimtaby), dtype='i')
            imgc = np.zeros((dimtabx,dimtaby,3), np.uint8)
            for l in listslice:
                if l.find(c,0)==0:
                    pathl=namedirtopcf+'/patchfile/'+l
                    tabcff = np.loadtxt(pathl,dtype='f')
                    ofile = open(pathl, 'r')
                    t = ofile.read()
                    ofile.close()
                    labpos=t.find('label')
                    labposend=t.find('\n',labpos)
                    labposdeb = t.find(' ',labpos)
                    label=t[labposdeb:labposend].strip()
                    locapos=t.find('local')
                    locaposend=t.find('\n',locapos)
                    locaposdeb = t.find(' ',locapos)
                    loca=t[locaposdeb:locaposend].strip()
                    tabccfi=tabcff/avgPixelSpacing
                    tabc=tabccfi.astype(int)
                    print('generate tables from:',l,'in:', f)
                    tabz,imgi= reptfulle(tabc,dimtabx,dimtaby)                
                    imgc=imgc+imgi                    
                    tabzc=tabz+tabzc
                    il=l.find('.',0)
                    iln=l[0:il]

            if label in usedclassif:
                print('c :',c, label,loca)
                print('creates patches from:',iln, 'in:', f)
                nbp,tabz1=pavs (imgc,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,jpegpath, patchpath,thrpatch,iln,f,label,loca,typei,errorfile)
                print('end create patches')
                nbpf=nbpf+nbp
                #create patches for back-ground
        pavbg(namedirtopcf,dimtabx,dimtaby,dimpavx,dimpavy)

    try:
        _=int(f)
        ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    except:
        ofilepw = open(jpegpath+'/nbpat_'+f.replace('/','_')+'.txt', 'w')        
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    
    
#################################################################    
#   calculate number of patches
contenupatcht = os.listdir(jpegpath) 
npatcht=0
for npp in contenupatcht:
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
#################################################################

#data statistics on paches
dirlabel=os.walk( patchpath).next()[1]
#file for data pn patches
eftpt=os.path.join(patchtoppath,'totalnbpat.txt')
filepwt = open(eftpt, 'w')
ntot=0;

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
        subdir = os.path.join(dirloca,loca)
        n=0
        listcwd=os.listdir(subdir)
        for ff in listcwd:
            if ff.find(typei) >0 :
                n+=1
                ntot+=1
        filepwt.write('label: '+label+' localisation: '+loca+' number of patches: '+str(n)+'\n')
filepwt.close() 

#write the log file with label list
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

##########################################################
errorfile.write('completed')
errorfile.close()
manage_HRCT_pilot(mode='shrink')
cleanup()
print('completed')