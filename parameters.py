
from PIL import ImageFont

#######################################################
namedirHUG = 'Implementation'
subHUG='ILD_DB_lungMasks'
subHUG_txt='ILD_DB_txtROIs'

toppatch= 'OUTPUT'  
extendir='PATCHES'
raw_patch=False
make_back_ground=False

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
SROIS='SROIS'
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
