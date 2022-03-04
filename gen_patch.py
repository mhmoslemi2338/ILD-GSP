import glob
import os
import numpy as np
import scipy.misc
import dicom
from PIL import Image
import cv2
from param_gen_patch import *
import tifffile as tiff

#######################################################
remove_folder(patchesdirnametop)

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
path_HUG=os.path.join(cwdtop,namedirHUG)
namedirtopc =os.path.join(path_HUG,subHUG)

patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpath=os.path.join(patchtoppath,patchesdirname)
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
jpegpath=os.path.join(patchtoppath,imagedirname)
SROIpatch=os.path.join(patchtoppath,SROIS)

if not os.path.isdir(patchtoppath): os.mkdir(patchtoppath)   
if raw_patch:
    if not os.path.isdir(patchpath): os.mkdir(patchpath)   
if not os.path.isdir(patchNormpath): os.mkdir(patchNormpath)  
if not os.path.isdir(jpegpath): os.mkdir(jpegpath)   
if not os.path.isdir(SROIpatch): os.mkdir(SROIpatch)   
#######################################################

def genebmp(dirName):
    """generate patches from dicom files and sroi"""
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
        FilesDCM =(os.path.join(dirName,filename))  

        ds = dicom.read_file(FilesDCM)
        dsr= ds.pixel_array          
        dsr= dsr-dsr.min()
        c=float(imageDepth)/dsr.max()
        dsr=dsr*c
        dsr=dsr.astype('uint16')
        #resize the dicom to have always the same pixel/mm
        fxs=float(ds.PixelSpacing[0])/avgPixelSpacing   

        scanNumber=int(ds.InstanceNumber)
        endnumslice=filename.find('.dcm')                   
        imgcore=filename[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
        bmpfile=os.path.join(bmp_dir,imgcore)
        dsrresize1= scipy.misc.imresize(dsr,fxs,interp='bicubic',mode=None) 
        namescan=os.path.join(sroidir,imgcore)                   
        textw='n: '+f+' scan: '+str(scanNumber)

        tablscan=cv2.cvtColor(dsrresize1,cv2.COLOR_GRAY2BGR)
        tiff.imsave(namescan, tablscan)
        tagviews(namescan,textw,0,20)  
        dsrresize=dsrresize1
        tiff.imsave(bmpfile,dsrresize)

        dimtabx=dsrresize.shape[0]
        dimtaby=dimtabx

    for lungfile in lunglist:
        lungDCM =os.path.join(lung_dir,lungfile)  
        dslung = dicom.read_file(lungDCM)
        dsrlung= dslung.pixel_array             
        dsrlung= dsrlung-dsrlung.min()

        c=0
        if dsrlung.max()>0:
            c=float(imageDepth)/dsrlung.max()
        dsrlung=dsrlung*c
        dsrlung=dsrlung.astype('uint16')

        fxslung=float(dslung.PixelSpacing[0])/avgPixelSpacing 
        scanNumber=int(dslung.InstanceNumber)
        endnumslice=lungfile.find('.dcm')                   
        lungcore=lungfile[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
        lungcoref=os.path.join(lung_bmp_dir,lungcore)
        lungresize= scipy.misc.imresize(dsrlung,fxslung,interp='bicubic',mode=None)            
        lungresize = cv2.blur(lungresize,(5,5))                 
        np.putmask(lungresize,lungresize>0,100)

        tiff.imsave(lungcoref,lungresize)
        bgdirflm=os.path.join(bgdirf,lungcore)
        tiff.imsave(bgdirflm,lungresize)

    


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
        #find the same name in bgdir directory
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

                #put all to 1 if>0
                tabfc = np.copy(tabf)
                nz= np.count_nonzero(tabf)
                if nz>0 and min_val!=max_val:
                    np.putmask(tabf,tabf>0,1)
                    atabf = np.nonzero(tabf)
                    #tab[y][x]  convention
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
                            #detect black pixels
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
                                nampa='/'+labelbg+'/'+f+'_'+str(slicenumber)+'_'+str(nbp)+'.'+'tiff' 
                                if raw_patch:
                                    crorig.save(patchpath+nampa)
                                
                                imgray =np.array(crorig,np.float32)
                                tabi2 = np.int16(normi(imgray))
                                tiff.imsave(patchNormpath+nampa, tabi2)
                                x=0
                                #draw the rectange
                                while x < px:
                                    y=0
                                    while y < py:
                                        tabp[y+j][x+i]=150
                                        if x == 0 or x == px-1 :
                                            y+=1
                                        else:
                                            y+=py-1
                                    x+=1
                                tabf[j:j+py,i:i+px]=0  #cancel the source                       
                        j+=1
                    i+=1

                tabpw =tabfc+tabp
                tiff.imsave(jpegpath+'/'+f+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'.jpg', tabpw) 
                mfl=open(jpegpath+'/'+f+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_1.txt',"w")
                mfl.write('#number of patches: '+str(nbp)+'\n')
                mfl.close()
                break


  
def pavs (imgi,tab,dx,dy,px,py,namedirtopcf,jpegpath,patchpath,thr,iln,f,label,loca,typei):
    """ generate patches from ROI"""
    vis=contour2(imgi,label,dimtabx,dimtaby)         
    bgdirf = os.path.join(namedirtopcf, bgdir)    
    patchpathc=os.path.join(namedirtopcf,bmpname)
    
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

            for lm in lunglist: 
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
                    
            tagview(namescan,label,0,100)
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
                            nampa='/'+label+'/'+f+'_'+iln+'_'+str(nbp)+'.'+'tiff' 
                            if raw_patch:
                                crorig.save(patchpath+nampa)
                            #normalize patches and put in patches_norm
                            
                            imgray =np.array(crorig,dtype=np.float32)
                            tabi2 = np.int16(normi(imgray))
                            tiff.imsave(patchNormpath+nampa, tabi2)
                            
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
                            tabf[j:j+py/2,i:i+px/2]=0   #cancel the source                        
                    j+=1
                i+=1
            break
    
    if not found:
        print('***********************  ERROR image not found  ***********************'+namedirtopcf+'/'+bmpname+'/'+str(slicenumber))            
            
    tabp =tab+tabp
    mfl=open(jpegpath+'/'+f+'_'+iln+'.txt',"w")
    mfl.write('#number of patches: '+str(nbp)+'\n'+strpac)
    mfl.close()
    tiff.imsave(jpegpath+'/'+f+'_'+iln+'.jpg', tabp)
    return nbp,tabp


def fileext(namefile,curdir,patchpath):
    listlabel=[labelbg+'_'+locabg]
    plab=os.path.join(patchpath,labelbg)
    plabNorm=os.path.join(patchNormpath,labelbg)

    if raw_patch:
        if not os.path.exists(plab): os.mkdir(plab)
    if not os.path.exists(plabNorm): os.mkdir(plabNorm)

    ofi = open(namefile, 'r')
    t = ofi.read()
    ofi.close()
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
        if label not in listlabel:
            plab=os.path.join(patchpath,label)
            plabNorm=os.path.join(patchNormpath,label)

            listlabel.append(label+'_'+loca)     
            listlabeld=os.listdir(patchNormpath)
            if label not in listlabeld:
                if raw_patch:
                    os.mkdir(plab)
                os.mkdir(plabNorm)

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
                if (slaposnext >1 and slaposnext <nbpposnext) or (labposnext >1 and labposnext <nbpposnext) or(nbpposnext ==-1):
                    conend=False
            slapos=t.find('slice',slapos+1)
            labposnext=t.find('label',labpos+1)
            #condition of loop slice
            if slapos ==-1 or (labposnext >1 and labposnext < slapos ):
                condslap = False
        labpos=t.find('label',labpos+1)

    return(listlabel,coefi)



##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

moslemi=['142','154','184','53','57','8','HRCT_pilot']

listdirc= os.listdir(namedirtopc)
npat=0
for f in listdirc[0:2]:
    if f in moslemi:
        continue

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
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
        fif=False
        genebmp(namedirtopcf)

        for f1 in contenudir:
            if f1.find('.txt') >0 and (f1.find('CT')==0 or (f1.find('Tho')==0)):
                npat+=1
                fif=True
                fileList =f1
                pathf1=namedirtopcf+'/'+fileList            
                labell,coefi =fileext(pathf1,namedirtopcf,patchpath)
                break
        if not fif:
            print('***********   ERROR: no ROI txt content file    ***************', f)

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

                    tabz,imgi= reptfulle(tabc,dimtabx,dimtaby)                
                    imgc=imgc+imgi                    
                    tabzc=tabz+tabzc
                    il=l.find('.',0)
                    iln=l[0:il]

            if label in usedclassif:
                print('c :',c, label,loca)
                print('creates patches from:',iln, 'in:', f)
                nbp,tabz1=pavs (imgc,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,jpegpath, patchpath,thrpatch,iln,f,label,loca,typei)
                print('end create patches')
                nbpf=nbpf+nbp
        #create patches for back-ground
        pavbg(namedirtopcf,dimtabx,dimtaby,dimpavx,dimpavy)
    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    
    
    #### organize and delet extra files
    remove_folder(os.path.join(namedirtopcf, bgdir))
    remove_folder(os.path.join(namedirtopcf, bmpname))
    remove_folder(os.path.join(namedirtopcf, lungmask, typei))
    remove_folder(os.path.join(namedirtopcf, 'patchfile'))
    src_sroi=os.path.join(namedirtopcf, sroi)
    if not os.path.isdir(os.path.join(SROIpatch,f)): os.mkdir(os.path.join(SROIpatch,f))  
    dst_sroi=os.path.join(namedirtopcf, SROIpatch,f)
    for row in os.listdir(src_sroi):
        shutil.move(os.path.join(src_sroi,row),os.path.join(dst_sroi,row))
    remove_folder(src_sroi)
for row in glob.iglob(os.path.join(jpegpath, '*.txt')): os.remove(row) 


#################### data statistics and log on paches ###########################

make_log(patchtoppath,jpegpath,listslice)
print('completed')