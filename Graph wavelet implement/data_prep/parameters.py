
from PIL import ImageFont
import os
#######################################################

tmp=os.getcwd()
namedirHUG = os.getcwd()
subHUG='ILD_DB/ILD_DB_lungMasks'
subHUG_txt='ILD_DB/ILD_DB_txtROIs'

toppatch= 'data_prep/OUTPUT'  
extendir='PATCH'
raw_patch=False
make_back_ground=False


out_img_size=32
out_img_depth=8

#######################################################

#patch overlapp tolerance
thrpatch = 0.8
# imageDepth=255 #number of bits used on dicom images (2 **n)
imageDepth=255
avgPixelSpacing=0.734

dimpavx =out_img_size
dimpavy = out_img_size
typei='bmp' #can be jpg

font5 = ImageFont.truetype( 'data_prep/arial.ttf', 5)
font10 = ImageFont.truetype( 'data_prep/arial.ttf', 10)
font20 = ImageFont.truetype( 'data_prep/arial.ttf', 20)
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




expanded_files2=[
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/142/CT-INSPIRIUM-2951', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/142001'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/142/CT-INSPIRIUM-2950', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/142002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/154/CT-INSPIRIUM-6409', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/154001'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/154/CT-INSPIRIUM-6410', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/154002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/184/CT-INSPIRIUM-5841', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/184001'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/184/CT-INSPIRIUM-5842', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/184002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/53/CT-INSPIRIUM-7605', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/53001'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/53/CT-INSPIRIUM-1841', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/53002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57/CT-INSPIRIUM-3550', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57001'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57/CT-series-5652', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57/CT--0002', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/57003'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/8/CT-INSPIRIUM-8873', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/8001'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/8/CT-INSPIRIUM-8871', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/8002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/211', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200001'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/210', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200002'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/204', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200003'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/203', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200004'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/205', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200005'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/209', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200006'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/200', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200007'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/207', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200008'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/206', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/200009'],
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/201', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/2000010'], 
['/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/HRCT_pilot/208', '/Users/mohammad/Desktop/Bsc prj/Implementation/ILD_DB/ILD_DB_lungMasks/2000011']]

