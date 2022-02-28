from PIL import ImageFont

#global directory for scan file
namedirHUG = 'ILD_DB'       
subHUG='ILD_DB_lungMasks'

toppatch= 'TOPPATCH2'  
#extension for output dir
extendir='16_set0_gci'

#normalization internal procedure or openCV
normiInternal=True# when True: use internal normi, otherwise opencv equalhist
globalHist=True #use histogram equalization on full image
globalHistInternal=False #use internal for global histogram when True otherwise opencv
#patch overlapp tolerance
thrpatch = 0.8
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
imageDepth=2**16 #number of bits used on dicom images (2 **n)
# average pxixel spacing
avgPixelSpacing=0.734

pset=0

#define the name of directory for patches
patchesdirnametop = toppatch+'_'+extendir
#define the name of directory for patches
patchesdirname = 'patches'
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm'
imagedirname='patches_jpeg'

bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'
#directory name with scan with roi
sroi='sroi'
#directory name with scan with roi
bgdir='bgdir'
bgdirw='bgdirw'


#----------------------------


#image  patch format
typei='bmp' #can be jpg
# typei='jpg'
#patch size


font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
labelbg='back_ground'
locabg='anywhere'

