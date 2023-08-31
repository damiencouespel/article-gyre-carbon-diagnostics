# Adapted from /gpfswork/rech/eee/rdyk004/OLD-MY-PYTHON3-SCRIPT-AND-FIG/MY_PYTHON3/main_anthc_box_budget.py
"""
Compute carbon budget in a box
Save outputs as pckl file
"""
from IPython import embed
import os

import numpy as np
import pickle

from FGYRE import reading, interpolating, averaging


dirout = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET/'
if (not os.path.isdir(dirout)) : os.mkdir(dirout)

##################################################
#                PARAMETETERS                    #
##################################################


fdir  = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'          
res = {'CTL1':'1', 'CTL1_NOGM':'1', 'CTL9':'9',  'CTL27':'27', 
       'CC1':'1', 'CC1_NOGM':'1', 'CC9':'9', 'CC27':'27',
       'CTL1_KL2000':'1', 'CTL1_KL500':'1', 'CTL1_KGM500_KL500':'1', 'CTL1_KGM2000_KL2000':'1',
       'CC1_KL2000':'1', 'CC1_KL500':'1', 'CC1_KGM500_KL500':'1', 'CC1_KGM2000_KL2000':'1'}

#___________________
# define the box where to compute the budget
# lats = [None, 21.9065, 34.2987, 43, 46.6909, None]
# deps = [None, 250, 2000, None]
# lons = [None, None]
# boxes={'box1' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[0], 'ymax':lats[1], 'zmin':deps[0], 'zmax':deps[1]}, \
#        'box2' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[1], 'ymax':lats[2], 'zmin':deps[0], 'zmax':deps[1]}, \
#        'box3' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[2], 'ymax':lats[3], 'zmin':deps[0], 'zmax':deps[1]}, \
#        'box4' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[3], 'ymax':lats[4], 'zmin':deps[0], 'zmax':deps[1]}, \
#        'box5' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[4], 'ymax':lats[5], 'zmin':deps[0], 'zmax':deps[1]}, \
#        'box6' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[0], 'ymax':lats[1], 'zmin':deps[1], 'zmax':deps[2]}, \
#        'box7' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[1], 'ymax':lats[2], 'zmin':deps[1], 'zmax':deps[2]}, \
#        'box8' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[2], 'ymax':lats[3], 'zmin':deps[1], 'zmax':deps[2]}, \
#        'box9' :{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[3], 'ymax':lats[4], 'zmin':deps[1], 'zmax':deps[2]}, \
#        'box10':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[4], 'ymax':lats[5], 'zmin':deps[1], 'zmax':deps[2]}, \
#        'box11':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[0], 'ymax':lats[1], 'zmin':deps[2], 'zmax':deps[3]}, \
#        'box12':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[1], 'ymax':lats[2], 'zmin':deps[2], 'zmax':deps[3]}, \
#        'box13':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[2], 'ymax':lats[3], 'zmin':deps[2], 'zmax':deps[3]}, \
#        'box14':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[3], 'ymax':lats[4], 'zmin':deps[2], 'zmax':deps[3]}, \
#        'box15':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[4], 'ymax':lats[5], 'zmin':deps[2], 'zmax':deps[3]}}
lats = [None, 34.2987, 43, None]
deps = [None, 250, None]
lons = [None, None]
boxes={'box1':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[0], 'ymax':lats[1], 'zmin':deps[0], 'zmax':deps[1]}, \
       'box2':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[1], 'ymax':lats[2], 'zmin':deps[0], 'zmax':deps[1]}, \
       'box3':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[2], 'ymax':lats[3], 'zmin':deps[0], 'zmax':deps[1]}, \
       'box4':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[0], 'ymax':lats[1], 'zmin':deps[1], 'zmax':deps[2]}, \
       'box5':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[1], 'ymax':lats[2], 'zmin':deps[1], 'zmax':deps[2]}, \
       'box6':{'xmin':lons[0], 'xmax':lons[1], 'ymin':lats[2], 'ymax':lats[3], 'zmin':deps[1], 'zmax':deps[2]}}

#___________________
# which simulation
vsim = 'CC9'

#___________________
# set outputs files
fileout = 'main_carbon_box_budget_r9_rad_1ymean_y101-170_6boxes.pckl'

#___________________
# which carbon
carbtra = 'DIC'
carbflx = 'Cflx'
carbtrd = 'DIC'

#___________________
# input files
fsufU   = '_1y_01010101_01701230_grid_U.xml'
fsufV   = '_1y_01010101_01701230_grid_V.xml'
fsufW   = '_1y_01010101_01701230_grid_W.xml'
fsufTr  = '_1y_01010101_01701230_ptrc_T.xml'
fsuftrd = '_1y_01010101_01701230_dictrd.xml'
fsufflx = '_1y_01010101_01701230_diad_T.xml'
sdate  = '0101-01-01'
edate  = '0170-12-31'

time_span=(sdate, edate)
print(time_span)

output = {'fsufU'   : fsufU  ,
          'fsufV'   : fsufV  ,
          'fsufW'   : fsufW  ,
          'fsufTr'  : fsufTr ,
          'fsuftrd' : fsuftrd,
          'sdate' : sdate, 
          'edate' : edate, 
          'boxes' : boxes}

##################################################
##################################################
#                COMPUTE BUDGET                  #
##################################################
##################################################

zwmeshfile = '/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+res[vsim]+'.nc' 
# zwmeshfile = '/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' # TEST
zwmesh = reading.read_mesh(zwmeshfile)
output['mesh'] = zwmesh

def diag_trend(inp, zmesh, zbox) : 
    zw = np.nanmean(inp, axis=0)
    zw = averaging.zmean(zw, zmesh, dim='zyx', grid='T', zmin=zbox['zmin'], zmax=zbox['zmax'], integral = True)
    zw = averaging.ymean(zw, zmesh, dim='yx' , grid='T', ymin=zbox['ymin'], ymax=zbox['ymax'], integral = True)
    zw = averaging.xmean(zw, zmesh, dim='x'  , grid='T', xmin=zbox['xmin'], xmax=zbox['xmax'], integral = True)
    return zw
#

##################################################
# COMPUTE OFFLINE ADVECTIVE FLUXES
##################################################


#-------------------
# read Tr
#-------------------

zwfTr = fdir + vsim + fsufTr
zw   = reading.read_ncdf(carbtra, zwfTr, time=time_span)
Tr = zw['data']

#-------------------
# compute uTr
#-------------------

# read zonal velocity
zwfU = fdir + vsim + fsufU
zw  = reading.read_ncdf('vozocrtx', zwfU, time=time_span)
zwU = zw['data']
# compute zonal tracer transport
uTr = zwU * .5 * (Tr + np.roll(Tr, -1, axis=-1)) * zwmesh['umask'][np.newaxis]
del zwU
# time mean
uTr = np.nanmean(uTr, axis=0)
# loop on boxes
for kbox, vbox in boxes.items() :
    zlon1, zlon2 = vbox['xmin'], vbox['xmax']
    zlat1, zlat2 = vbox['ymin'], vbox['ymax']
    zdep1, zdep2 = vbox['zmin'], vbox['zmax']
    # interpolate on longitude box boundaries
    if zlon1 != None : zuTr1 = interpolating.xinterpol(uTr, zwmesh, zlon1, dim='zyx', grid='U')
    else : zuTr1 = uTr[:, :, 0]
    if zlon2 != None : zuTr2 = interpolating.xinterpol(uTr, zwmesh, zlon2, dim='zyx', grid='U')
    else : zuTr2 = uTr[:, :, -1]
    # spatial integral 
    zuTr1 = averaging.ymean(zuTr1, zwmesh, dim='zy', grid='U', ymin=zlat1, ymax=zlat2, integral = True)
    zuTr2 = averaging.ymean(zuTr2, zwmesh, dim='zy', grid='U', ymin=zlat1, ymax=zlat2, integral = True)
    zuTr1 = averaging.zmean(zuTr1, zwmesh, dim='z' , grid='U', zmin=zdep1, zmax=zdep2, integral = True)
    zuTr2 = averaging.zmean(zuTr2, zwmesh, dim='z' , grid='U', zmin=zdep1, zmax=zdep2, integral = True)
    output['boxes'][kbox]['uTr1'] =  zuTr1 # positive sign if flux entering the box
    output['boxes'][kbox]['uTr2'] = -zuTr2 # positive sign if flux entering the box
#
del uTr, zuTr1, zuTr2

#-------------------
# compute vTr
#-------------------

# read meridional velocity
zwfV = fdir + vsim + fsufV
zw  = reading.read_ncdf('vomecrty', zwfV, time=time_span)
zwV = zw['data']
# compute meridional tracer transport
vTr = zwV * .5 * (Tr + np.roll(Tr, -1, axis=-2)) * zwmesh['vmask'][np.newaxis]
del zwV
# time mean
vTr = np.nanmean(vTr, axis=0)
# loop on boxes
for kbox, vbox in boxes.items() :
    zlon1, zlon2 = vbox['xmin'], vbox['xmax']
    zlat1, zlat2 = vbox['ymin'], vbox['ymax']
    zdep1, zdep2 = vbox['zmin'], vbox['zmax']
    # interpolate on latitude box boundaries
    if zlat1 != None : zvTr1 = interpolating.yinterpol(vTr, zwmesh, zlat1, dim='zyx', grid='V')
    else : zvTr1 = vTr[:, 0, :]
    if zlat2 != None : zvTr2 = interpolating.yinterpol(vTr, zwmesh, zlat2, dim='zyx', grid='V')
    else : zvTr2 = vTr[:, -1, :]
    # spatial integral
    zvTr1 = averaging.xmean(zvTr1, zwmesh, dim='zx', grid='V', xmin=zlon1, xmax=zlon2, integral = True)
    zvTr2 = averaging.xmean(zvTr2, zwmesh, dim='zx', grid='V', xmin=zlon1, xmax=zlon2, integral = True)
    zvTr1 = averaging.zmean(zvTr1, zwmesh, dim='z' , grid='V', zmin=zdep1, zmax=zdep2, integral = True)
    zvTr2 = averaging.zmean(zvTr2, zwmesh, dim='z' , grid='V', zmin=zdep1, zmax=zdep2, integral = True)
    output['boxes'][kbox]['vTr1'] =  zvTr1 # positive sign if flux entering the box
    output['boxes'][kbox]['vTr2'] = -zvTr2 # positive sign if flux entering the box
#
del vTr, zvTr1, zvTr2

#-------------------
# compute wTr
#-------------------

# read vertical velocity
zwfW = fdir + vsim + fsufW
zw  = reading.read_ncdf('vovecrtz', zwfW, time=time_span)
zwW = zw['data']
# compute vertical tracer transport
wTr = np.roll(zwW, -1, axis=-3) * .5 * (Tr + np.roll(Tr, -1, axis=-3)) * zwmesh['wmask'][np.newaxis]
wTr = np.roll(wTr, 1, axis=-3)
# wTr[:, 0, :, :] = zwW[:, 0, :, :] * .5 * Tr[:, 0, :, :]
wTr[:, 0, :, :] = 0. * Tr[:, 0, :, :]
del zwW
# time mean
wTr = np.nanmean(wTr, axis=0)
# loop on boxes
for kbox, vbox in boxes.items() :
    zlon1, zlon2 = vbox['xmin'], vbox['xmax']
    zlat1, zlat2 = vbox['ymin'], vbox['ymax']
    zdep1, zdep2 = vbox['zmin'], vbox['zmax']
    # interpolate on depth box boundaries
    if zdep1 != None : zwTr1 = interpolating.zinterpol(wTr, zwmesh, zdep1, dim='zyx', grid='W')
    else : zwTr1 = wTr[0, :, :]
    if zdep2 != None : zwTr2 = interpolating.zinterpol(wTr, zwmesh, zdep2, dim='zyx', grid='W')
    else : zwTr2 = wTr[-1, :, :]
    # spatial integral
    zwTr1 = averaging.xmean(zwTr1, zwmesh, dim='yx', grid='W', xmin=zlon1, xmax=zlon2, integral = True)
    zwTr2 = averaging.xmean(zwTr2, zwmesh, dim='yx', grid='W', xmin=zlon1, xmax=zlon2, integral = True)
    zwTr1 = averaging.ymean(zwTr1, zwmesh, dim='y' , grid='W', ymin=zlat1, ymax=zlat2, integral = True)
    zwTr2 = averaging.ymean(zwTr2, zwmesh, dim='y' , grid='W', ymin=zlat1, ymax=zlat2, integral = True)
    output['boxes'][kbox]['wTr1'] = -zwTr1 # positive sign if flux entering the box
    output['boxes'][kbox]['wTr2'] =  zwTr2 # positive sign if flux entering the box
#
del wTr, zwTr1, zwTr2

#-------------------
# Compute sum of advective fluxes
#-------------------

for kk, vv in output['boxes'].items() :
    sfx = vv['uTr1'] + vv['uTr2'] + vv['vTr1'] + vv['vTr2'] + vv['wTr1'] + vv['wTr2']
    output['boxes'][kk]['sfx'] = sfx
#


##################################################
# COMPUTE OFFLINE  ADVECTIVE FLUXES GM
##################################################

if vsim in ['CTL1', 'CTL1_KL2000', 'CTL1_KL500', 'CTL1_KGM500_KL500', 'CTL1_KGM2000_KL2000',
            'CC1', 'CC1_KL2000', 'CC1_KL500', 'CC1_KGM500_KL500', 'CC1_KGM2000_KL2000'] :

    #-------------------
    # compute ugmTr
    #-------------------

    # read zonal velocity
    zwfU = fdir + vsim + fsufU
    zw  = reading.read_ncdf('vozoeivx', zwfU, time=time_span)
    zwU = zw['data']
    # compute zonal tracer transport
    ugmTr = zwU * .5 * (Tr + np.roll(Tr, -1, axis=-1)) * zwmesh['umask'][np.newaxis]
    del zwU
    # time mean
    ugmTr = np.nanmean(ugmTr, axis=0)
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zlon1, zlon2 = vbox['xmin'], vbox['xmax']
        zlat1, zlat2 = vbox['ymin'], vbox['ymax']
        zdep1, zdep2 = vbox['zmin'], vbox['zmax']
        # interpolate on longitude box boundaries
        if zlon1 != None : zugmTr1 = interpolating.xinterpol(ugmTr, zwmesh, zlon1, dim='zyx', grid='U')
        else : zugmTr1 = ugmTr[:, :, 0]
        if zlon2 != None : zugmTr2 = interpolating.xinterpol(ugmTr, zwmesh, zlon2, dim='zyx', grid='U')
        else : zugmTr2 = ugmTr[:, :, -1]
        # spatial integral 
        zugmTr1 = averaging.ymean(zugmTr1, zwmesh, dim='zy', grid='U', ymin=zlat1, ymax=zlat2, integral = True)
        zugmTr2 = averaging.ymean(zugmTr2, zwmesh, dim='zy', grid='U', ymin=zlat1, ymax=zlat2, integral = True)
        zugmTr1 = averaging.zmean(zugmTr1, zwmesh, dim='z' , grid='U', zmin=zdep1, zmax=zdep2, integral = True)
        zugmTr2 = averaging.zmean(zugmTr2, zwmesh, dim='z' , grid='U', zmin=zdep1, zmax=zdep2, integral = True)
        output['boxes'][kbox]['ugmTr1'] =  zugmTr1 # positive sign if flux entering the box
        output['boxes'][kbox]['ugmTr2'] = -zugmTr2 # positive sign if flux entering the box
    #
    del ugmTr, zugmTr1, zugmTr2

    #-------------------
    # compute vgmTr
    #-------------------

    # read meridional velocity
    zwfV = fdir + vsim + fsufV
    zw  = reading.read_ncdf('vomeeivy', zwfV, time=time_span)
    zwV = zw['data']
    # compute meridional tracer transport
    vgmTr = zwV * .5 * (Tr + np.roll(Tr, -1, axis=-2)) * zwmesh['vmask'][np.newaxis]
    del zwV
    # time mean
    vgmTr = np.nanmean(vgmTr, axis=0)
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zlon1, zlon2 = vbox['xmin'], vbox['xmax']
        zlat1, zlat2 = vbox['ymin'], vbox['ymax']
        zdep1, zdep2 = vbox['zmin'], vbox['zmax']
        # interpolate on latitude box boundaries
        if zlat1 != None : zvgmTr1 = interpolating.yinterpol(vgmTr, zwmesh, zlat1, dim='zyx', grid='V')
        else : zvgmTr1 = vgmTr[:, 0, :]
        if zlat2 != None : zvgmTr2 = interpolating.yinterpol(vgmTr, zwmesh, zlat2, dim='zyx', grid='V')
        else : zvgmTr2 = vgmTr[:, -1, :]
        # spatial integral
        zvgmTr1 = averaging.xmean(zvgmTr1, zwmesh, dim='zx', grid='V', xmin=zlon1, xmax=zlon2, integral = True)
        zvgmTr2 = averaging.xmean(zvgmTr2, zwmesh, dim='zx', grid='V', xmin=zlon1, xmax=zlon2, integral = True)
        zvgmTr1 = averaging.zmean(zvgmTr1, zwmesh, dim='z' , grid='V', zmin=zdep1, zmax=zdep2, integral = True)
        zvgmTr2 = averaging.zmean(zvgmTr2, zwmesh, dim='z' , grid='V', zmin=zdep1, zmax=zdep2, integral = True)
        output['boxes'][kbox]['vgmTr1'] =  zvgmTr1 # positive sign if flux entering the box
        output['boxes'][kbox]['vgmTr2'] = -zvgmTr2 # positive sign if flux entering the box
    #
    del vgmTr, zvgmTr1, zvgmTr2

    #-------------------
    # compute wgmTr
    #-------------------

    # read vertical velocity
    zwfW = fdir + vsim + fsufW
    zw  = reading.read_ncdf('voveeivz', zwfW, time=time_span)
    zwW = zw['data']
    # compute vertical tracer transport
    wgmTr = np.roll(zwW, -1, axis=-3) * .5 * (Tr + np.roll(Tr, -1, axis=-3)) * zwmesh['wmask'][np.newaxis]
    wgmTr = np.roll(wgmTr, 1, axis=-3)
    # wgmTr[:, 0, :, :] = zwW[:, 0, :, :] * .5 * Tr[:, 0, :, :]
    wgmTr[:, 0, :, :] = 0. * Tr[:, 0, :, :]
    del zwW
    # time mean
    wgmTr = np.nanmean(wgmTr, axis=0)
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zlon1, zlon2 = vbox['xmin'], vbox['xmax']
        zlat1, zlat2 = vbox['ymin'], vbox['ymax']
        zdep1, zdep2 = vbox['zmin'], vbox['zmax']
        # interpolate on depth box boundaries
        if zdep1 != None : zwgmTr1 = interpolating.zinterpol(wgmTr, zwmesh, zdep1, dim='zyx', grid='W')
        else : zwgmTr1 = wgmTr[0, :, :]
        if zdep2 != None : zwgmTr2 = interpolating.zinterpol(wgmTr, zwmesh, zdep2, dim='zyx', grid='W')
        else : zwgmTr2 = wgmTr[-1, :, :]
        # spatial integral
        zwgmTr1 = averaging.xmean(zwgmTr1, zwmesh, dim='yx', grid='W', xmin=zlon1, xmax=zlon2, integral = True)
        zwgmTr2 = averaging.xmean(zwgmTr2, zwmesh, dim='yx', grid='W', xmin=zlon1, xmax=zlon2, integral = True)
        zwgmTr1 = averaging.ymean(zwgmTr1, zwmesh, dim='y' , grid='W', ymin=zlat1, ymax=zlat2, integral = True)
        zwgmTr2 = averaging.ymean(zwgmTr2, zwmesh, dim='y' , grid='W', ymin=zlat1, ymax=zlat2, integral = True)
        output['boxes'][kbox]['wgmTr1'] = -zwgmTr1 # positive sign if flux entering the box
        output['boxes'][kbox]['wgmTr2'] =  zwgmTr2 # positive sign if flux entering the box
    #
    del wgmTr, zwgmTr1, zwgmTr2

    #-------------------
    # Compute sum of gm advective fluxes
    #-------------------

    for kk, vv in output['boxes'].items() :
        sfx = vv['ugmTr1'] + vv['ugmTr2'] + vv['vgmTr1'] + vv['vgmTr2'] + vv['wgmTr1'] + vv['wgmTr2']
        output['boxes'][kk]['sfxgm'] = sfx
    #

#


##################################################
# COMPUTE OFFLINE ADVECTIVE TREND
##################################################

#-------------------
# compute udTr
#-------------------

# read zonal velocity
zwfU = fdir + vsim + fsufU
zw  = reading.read_ncdf('vozocrtx', zwfU, time=time_span)
zwU = zw['data'] * zwmesh['umask'][np.newaxis, :, :, :]
# compute zonal adective trend
zw = zwmesh['e1u'][np.newaxis, np.newaxis, :, :]
# Trd(i) = 0.5 * ( u(i) * [ T(i+1) - T(i) ] + u(i-1) * [ T(i) -T(i-1) ] )
udTr = 0.5 * zwU * ( np.roll(Tr, -1, axis=-1) - Tr ) / zw + \
    0.5 * np.roll(zwU, 1, axis=-1) * ( Tr - np.roll(Tr, 1, axis=-1) ) / np.roll(zw, 1, axis = -1)
udTr = udTr * zwmesh['tmask'][np.newaxis, :, :, :]
del zwU
# loop on boxes
for kbox, vbox in boxes.items() :
    zudTr = diag_trend(udTr, zwmesh, vbox)
    output['boxes'][kbox]['xadoff'] = -zudTr # minus sign so that influx is positive
#
del udTr, zudTr

#-------------------
# compute vdTr
#-------------------

# read meridional velocity
zwfV = fdir + vsim + fsufV
zw  = reading.read_ncdf('vomecrty', zwfV, time=time_span)
zwV = zw['data'] * zwmesh['vmask'][np.newaxis, :, :, :]
# compute meridional advective trend
zw = zwmesh['e2v'][np.newaxis, np.newaxis, :, :]
# Trd(j) = 0.5 * ( v(j) * [ T(j+1) - T(j) ] + v(j-1) * [ T(j) -T(j-1) ] )
vdTr = 0.5 * zwV * ( np.roll(Tr, -1, axis=-2) - Tr ) / zw + \
    0.5 * np.roll(zwV, 1, axis=-2) * ( Tr - np.roll(Tr, 1, axis=-2) ) / np.roll(zw, 1, axis = -2)
vdTr = vdTr * zwmesh['tmask'][np.newaxis, :, :, :]
del zwV
# loop on boxes
for kbox, vbox in boxes.items() :
    zvdTr = diag_trend(vdTr, zwmesh, vbox)
    output['boxes'][kbox]['yadoff'] = -zvdTr # minus sign so that influx is positive
#
del vdTr, zvdTr
    
#-------------------
# compute wdTr
#-------------------

# read vertical velocity
zwfW = fdir + vsim + fsufW
zw  = reading.read_ncdf('vovecrtz', zwfW, time=time_span)
zwW = zw['data'] * zwmesh['wmask'][np.newaxis, :, :, :]
# compute vertical advective trend
zw = zwmesh['e3w'][np.newaxis, :, :, :]
# Trd(k) = 0.5 * ( w(k) * [ T(k-1) - T(k) ] + w(k+1) * [ T(k) -T(k+1) ] )
wdTr = 0.5 * zwW * ( np.roll(Tr, 1, axis=-3) - Tr ) / zw + \
    0.5 * np.roll(zwW, -1, axis=-3) * ( Tr - np.roll(Tr, -1, axis=-3) ) / np.roll(zw, -1, axis = -3)
wdTr[:, -1, :, :] = 0
wdTr = wdTr * zwmesh['tmask'][np.newaxis, :, :, :]
del zwW
# loop on boxes
for kbox, vbox in boxes.items() :
    zwdTr = diag_trend(wdTr, zwmesh, vbox)
    output['boxes'][kbox]['zadoff'] = -zwdTr # minus sign so that influx is positive
#
del wdTr, zwdTr

#-------------------    
# Compute sum of offline adv trd
#-------------------    

for kk, vv in output['boxes'].items() :
    output['boxes'][kk]['hadoff'] = vv['xadoff'] + vv['yadoff']
    output['boxes'][kk]['advoff'] = vv['xadoff'] + vv['yadoff'] + vv['zadoff']
#


##################################################
# COMPUTE OFFLINE ADVECTIVE TREND GM
##################################################

if vsim in ['CTL1', 'CTL1_KL2000', 'CTL1_KL500', 'CTL1_KGM500_KL500', 'CTL1_KGM2000_KL2000',
                'CC1', 'CC1_KL2000', 'CC1_KL500', 'CC1_KGM500_KL500', 'CC1_KGM2000_KL2000'] : 

    #-------------------
    # compute ugmdTr
    #-------------------
    
    # read zonal velocity
    zwfU = fdir + vsim + fsufU
    zw  = reading.read_ncdf('vozoeivx', zwfU, time=time_span)
    zwU = zw['data'] * zwmesh['umask'][np.newaxis, :, :, :]
    # compute zonal adective trend
    zw = zwmesh['e1u'][np.newaxis, np.newaxis, :, :]
    # Trd(i) = 0.5 * ( u(i) * [ T(i+1) - T(i) ] + u(i-1) * [ T(i) -T(i-1) ] )
    ugmdTr = 0.5 * zwU * ( np.roll(Tr, -1, axis=-1) - Tr ) / zw + \
        0.5 * np.roll(zwU, 1, axis=-1) * ( Tr - np.roll(Tr, 1, axis=-1) ) / np.roll(zw, 1, axis = -1)
    ugmdTr = ugmdTr * zwmesh['tmask'][np.newaxis, :, :, :]
    del zwU
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zugmdTr = diag_trend(ugmdTr, zwmesh, vbox)
        output['boxes'][kbox]['xadgmoff'] = -zugmdTr # minus sign so that influx is positive
    #
    del ugmdTr, zugmdTr

    #-------------------
    # compute vgmdTr
    #-------------------
    
    # read meridional velocity
    zwfV = fdir + vsim + fsufV
    zw  = reading.read_ncdf('vomeeivy', zwfV, time=time_span)
    zwV = zw['data'] * zwmesh['vmask'][np.newaxis, :, :, :]
    # compute meridional advective trend
    zw = zwmesh['e2v'][np.newaxis, np.newaxis, :, :]
    # Trd(j) = 0.5 * ( v(j) * [ T(j+1) - T(j) ] + v(j-1) * [ T(j) -T(j-1) ] )
    vgmdTr = 0.5 * zwV * ( np.roll(Tr, -1, axis=-2) - Tr ) / zw + \
        0.5 * np.roll(zwV, 1, axis=-2) * ( Tr - np.roll(Tr, 1, axis=-2) ) / np.roll(zw, 1, axis = -2)
    vgmdTr = vgmdTr * zwmesh['tmask'][np.newaxis, :, :, :]
    del zwV
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zvgmdTr = diag_trend(vgmdTr, zwmesh, vbox)
        output['boxes'][kbox]['yadgmoff'] = -zvgmdTr # minus sign so that influx is positive
    #
    del vgmdTr, zvgmdTr
        
    #-------------------
    # compute wgmdTr
    #-------------------
    
    # read vertical velocity
    zwfW = fdir + vsim + fsufW
    zw  = reading.read_ncdf('voveeivz', zwfW, time=time_span)
    zwW = zw['data'] * zwmesh['wmask'][np.newaxis, :, :, :]
    # compute vertical advective trend
    zw = zwmesh['e3w'][np.newaxis, :, :, :]
    # Trd(k) = 0.5 * ( w(k) * [ T(k-1) - T(k) ] + w(k+1) * [ T(k) -T(k+1) ] )
    wgmdTr = 0.5 * zwW * ( np.roll(Tr, 1, axis=-3) - Tr ) / zw + \
        0.5 * np.roll(zwW, -1, axis=-3) * ( Tr - np.roll(Tr, -1, axis=-3) ) / np.roll(zw, -1, axis = -3)
    wgmdTr[:, -1, :, :] = 0
    wgmdTr = wgmdTr * zwmesh['tmask'][np.newaxis, :, :, :]
    del zwW
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zwgmdTr = diag_trend(wgmdTr, zwmesh, vbox)
        output['boxes'][kbox]['zadgmoff'] = -zwgmdTr # minus sign so that influx is positive
    #
    del wgmdTr, zwgmdTr
    
    #-------------------    
    # Compute sum of offline adv trd
    #-------------------    
    
    for kk, vv in output['boxes'].items() :
        output['boxes'][kk]['hadgmoff'] = vv['xadgmoff'] + vv['yadgmoff']
        output['boxes'][kk]['advgmoff'] = vv['xadgmoff'] + vv['yadgmoff'] + vv['zadgmoff']
    #
#


##################################################
# DIAG ON ONLINE FLUXES/TRENDS
##################################################
    
#-------------------
# Vert diffusive flux
#-------------------

# read trend
zwftrd = fdir + vsim + fsuftrd
zw=reading.read_ncdf('ZDF_'+carbtrd, zwftrd, time = time_span)
zzdf=zw['data']
# loop on boxes
for kbox, vbox in boxes.items() :
    zw = diag_trend(zzdf, zwmesh, vbox)
    output['boxes'][kbox]['zdfonl'] = zw
#
del zzdf

#-------------------
# Lat diffusive flux
#-------------------

# read trend
zwftrd = fdir + vsim + fsuftrd
zw=reading.read_ncdf('LDF_'+carbtrd, zwftrd, time = time_span)
zldf=zw['data']
# loop on boxes
for kbox, vbox in boxes.items() :
    zw = diag_trend(zldf, zwmesh, vbox)
    output['boxes'][kbox]['ldfonl'] = zw
#
del zldf

#-------------------
# Online advective trends
#-------------------

# read trends
zwftrd = fdir + vsim + fsuftrd
zw=reading.read_ncdf('XAD_'+carbtrd, zwftrd, time = time_span)
zxad = zw['data']
zw=reading.read_ncdf('YAD_'+carbtrd, zwftrd, time = time_span)
zyad = zw['data']
zw=reading.read_ncdf('ZAD_'+carbtrd, zwftrd, time = time_span)
zzad = zw['data']
# loop on boxes
for kbox, vbox in boxes.items() :
    # xadonl
    zw = diag_trend(zxad, zwmesh, vbox)
    output['boxes'][kbox]['xadonl'] = zw
    # yadonl
    zw = diag_trend(zyad, zwmesh, vbox)
    output['boxes'][kbox]['yadonl'] = zw
    # zadonl
    zw = diag_trend(zzad, zwmesh, vbox)
    output['boxes'][kbox]['zadonl'] = zw
#
del zxad, zyad, zzad

# Compute sum of online adv trd
for kk, vv in output['boxes'].items() :
    output['boxes'][kk]['hadonl'] = vv['xadonl'] + vv['yadonl']
    output['boxes'][kk]['advonl'] = vv['xadonl'] + vv['yadonl'] + vv['zadonl']
#
    
#-------------------
# AIR-SEA FLUX
#-------------------

# read trend
zwftrd = fdir + vsim + fsufflx
zw = reading.read_ncdf(carbflx, zwftrd, time = time_span)
zflx = zw['data']
# time mean
zflx = np.nanmean(zflx, axis=0) * 10 # mmol/m2/s(factor 10 because the flux saved is divided by the thicknes of the first ocean layer)
# loop on boxes
for kbox, vbox in boxes.items() :
    zlon1, zlon2 = vbox['xmin'], vbox['xmax']
    zlat1, zlat2 = vbox['ymin'], vbox['ymax']
    zdep1, zdep2 = vbox['zmin'], vbox['zmax']
    # spatial integral
    if zdep1 == None : 
        zw = averaging.ymean(zflx, zwmesh, dim='yx', grid='T', ymin=zlat1, ymax=zlat2, integral = True)
        zw = averaging.xmean(zw  , zwmesh, dim='x' , grid='T', xmin=zlon1, xmax=zlon2, integral = True)
    else : zw = 0.
    output['boxes'][kbox]['asflux'] = zw # positive sign if flux entering the box
del zflx

#-------------------
# SMS (air-sea flux removed)
#-------------------

# read trend
zwftrd = fdir + vsim + fsuftrd
zw = reading.read_ncdf('SMS_'+carbtrd, zwftrd, time = time_span)
zsms = zw['data']
# loop on boxes
for kbox, vbox in boxes.items() :
    zw = diag_trend(zsms, zwmesh, vbox)
    output['boxes'][kbox]['smsonl'] = zw - output['boxes'][kbox]['asflux']
#
del zsms

#-------------------
# FOR
#-------------------

# read trend
zwftrd = fdir + vsim + fsuftrd
zw = reading.read_ncdf('FOR_'+carbtrd, zwftrd, time = time_span)
zfor = zw['data']
# loop on boxes
for kbox, vbox in boxes.items() :
    zw = diag_trend(zfor, zwmesh, vbox)
    output['boxes'][kbox]['foronl'] = zw
#
del zfor

##################################################
##################################################
#              END COMPUTE BUDGET                #
##################################################
##################################################

f = open(dirout+fileout, 'wb')
pickle.dump(output, f)
f.close()
print("File saved: "+dirout+fileout)
