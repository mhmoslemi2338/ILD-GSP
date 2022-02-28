from data_config import pset

usedclassif0 = [
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
    'tuberculosis'
    ]
classif0 ={
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
    'tuberculosis':17
    }
##----------------------------------------------------------------
usedclassif1 = [
    'back_ground',
    'consolidation',
    'ground_glass',
    'healthy',
    'cysts'
    ]
    
classif1 ={
    'back_ground':0,
    'consolidation':1,
    'ground_glass':2,
    'healthy':3,
    'cysts':4
    }
##----------------------------------------------------------------
usedclassif2 = [
    'back_ground',
    'fibrosis',
    'healthy',
    'micronodules',
    'reticulation'
    ]
classif2 ={
    'back_ground':0,
    'fibrosis':1,
    'healthy':2,
    'micronodules':3,
    'reticulation':4,
    }
##----------------------------------------------------------------
usedclassif3 = [
    'back_ground',
    'healthy',
    'air_trapping',
    ]
classif3 ={
    'back_ground':0,
    'healthy':1,
    'air_trapping':2,
    }


##----------------------------------------------------------------
##----------------------------------------------------------------


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
'tuberculosis':white
 }


#########################################################
if pset==0:
    #'consolidation', 'HC','ground_glass', 'micronodules', 'reticulation'
    dimpavx =32
    dimpavy = 32
    usedclassif = usedclassif0
    classif =classif0 
elif pset==1:
    #'consolidation', 'ground_glass',
    dimpavx =28 
    dimpavy = 28
    usedclassif = usedclassif1
    classif =classif1   
elif pset==2:
    #picklefile   'HC', 'micronodules'
    dimpavx =16 
    dimpavy = 16
    usedclassif = usedclassif2
    classif =classif2 
elif pset==3: 
    #'air_trapping'
    dimpavx =82 #or 20
    dimpavy = 82
    usedclassif = usedclassif3
    classif =classif3
else:
    print('eRROR :', pset, 'not allowed')

########################################################################

