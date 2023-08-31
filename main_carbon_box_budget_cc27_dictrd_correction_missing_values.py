# Adapted from /gpfswork/rech/eee/rdyk004/MY-PYTHON3/main_carbon_box_budget.py
"""
Compute carbon budget in a box
Save outputs as pckl file
Add missing years for CC27 dictrd or dic2trd
"""
from IPython import embed
import os

import numpy as np
import pickle
from scipy import interpolate

from FGYRE import reading, interpolating, averaging


dirout = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET-CC27-DICTRD-CORRECTION-MISSING-VALUES/'
if (not os.path.isdir(dirout)) : os.mkdir(dirout)

bcompute_annual_means = False

##################################################
##################################################
#                PARAMETETERS                    #
##################################################
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
vsim = 'CC27'

#___________________
# set outputs files
fileout_pref = 'main_carbon_box_budget_cc27_dictrd_correction_missing_values'
fileout_tmp = fileout_pref + '_r27_rad_1ymean_y101-170_6boxes_annual_values.pckl'
fileout = fileout_pref + '_r27_rad_1ymean_y101-170_6boxes.pckl'

 
#___________________
# which carbon
carbflx = 'Cflx'
carbtrd = 'DIC'

#___________________
# input files
fsuftrd = '_1y_01010101_01701230_dictrd.xml'
fsufflx = '_1y_01010101_01701230_diad_T.xml'
sdate  = '0101-01-01'
edate  = '0170-12-31'

time_span=(sdate, edate)
print(time_span)

output_tmp = {'fsufflx': fsufflx,
              'fsuftrd' : fsuftrd,
              'sdate' : sdate, 
              'edate' : edate, 
              'boxes' : boxes}

##################################################
##################################################
#          COMPUTE BUDGET ANNUAL VALUES          #
##################################################
##################################################

if bcompute_annual_means :

    print('> Compute budget annual values')

    zwmeshfile = '/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+res[vsim]+'.nc' 
    # zwmeshfile = '/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' # TEST
    zwmesh = reading.read_mesh(zwmeshfile)
    output_tmp['mesh'] = zwmesh
    
    def diag_trend(inp, zmesh, zbox) :
        print('func. diag_trend:')
        # box integral
        print('> box integral')
        # zw = np.nanmean(inp, axis=0)
        zw = averaging.zmean(inp, zmesh, dim='tzyx', grid='T', zmin=zbox['zmin'], zmax=zbox['zmax'], integral = True)
        zw = averaging.ymean(zw, zmesh, dim='tyx' , grid='T', ymin=zbox['ymin'], ymax=zbox['ymax'], integral = True)
        zw = averaging.xmean(zw, zmesh, dim='tx'  , grid='T', xmin=zbox['xmin'], xmax=zbox['xmax'], integral = True)
        # interpolation
        print('> interpolation')
        aaa=list(np.arange(65)+1)
        aaa.extend([69, 70])
        t = np.array(aaa)
        fint = interpolate.interp1d(t, zw)
        zw2 = list(zw[:-2])
        zw2.extend([fint(66), fint(67), fint(68)])
        zw2.extend(zw[-2:])
        print('func. end')
        return np.array(zw2)
    #
    
    ##################################################
    # DIAG ON ONLINE FLUXES/TRENDS ANNUAL VALUES
    ##################################################
        
    #-------------------
    # Vert diffusive flux
    #-------------------
    
    print('>> Vert diffusive flux')

    # read trend
    zwftrd = fdir + vsim + fsuftrd
    zw=reading.read_ncdf('ZDF_'+carbtrd, zwftrd, time = time_span)
    zzdf=zw['data']
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zw = diag_trend(zzdf, zwmesh, vbox)
        output_tmp['boxes'][kbox]['zdfonl'] = zw
        #
    del zzdf
    
    #-------------------
    # Lat diffusive flux
    #-------------------
    
    print('>> Lat diffusive flux')

    # read trend
    zwftrd = fdir + vsim + fsuftrd
    zw=reading.read_ncdf('LDF_'+carbtrd, zwftrd, time = time_span)
    zldf=zw['data']
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zw = diag_trend(zldf, zwmesh, vbox)
        output_tmp['boxes'][kbox]['ldfonl'] = zw
        #
    del zldf
    
    #-------------------
    # Online advective trends
    #-------------------
    
    print('>> Online advective trends')

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
        output_tmp['boxes'][kbox]['xadonl'] = zw
        # yadonl
        zw = diag_trend(zyad, zwmesh, vbox)
        output_tmp['boxes'][kbox]['yadonl'] = zw
        # zadonl
        zw = diag_trend(zzad, zwmesh, vbox)
        output_tmp['boxes'][kbox]['zadonl'] = zw
        #
    del zxad, zyad, zzad
    
    # Compute sum of online adv trd
    for kk, vv in output_tmp['boxes'].items() :
        output_tmp['boxes'][kk]['hadonl'] = vv['xadonl'] + vv['yadonl']
        output_tmp['boxes'][kk]['advonl'] = vv['xadonl'] + vv['yadonl'] + vv['zadonl']
        #
        
    #-------------------
    # Air-sea flux
    #-------------------
    
    print('>> Air-sea flux')

    # read trend
    zwftrd = fdir + vsim + fsufflx
    zw = reading.read_ncdf(carbflx, zwftrd, time = time_span)
    zflx = zw['data'] * 10 # mmol/m2/s(factor 10 because the flux saved is divided by the thicknes of the first ocean layer)
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zlon1, zlon2 = vbox['xmin'], vbox['xmax']
        zlat1, zlat2 = vbox['ymin'], vbox['ymax']
        zdep1, zdep2 = vbox['zmin'], vbox['zmax']
        # spatial integral
        if zdep1 == None : 
            zw = averaging.ymean(zflx, zwmesh, dim='tyx', grid='T', ymin=zlat1, ymax=zlat2, integral = True)
            zw = averaging.xmean(zw  , zwmesh, dim='tx' , grid='T', xmin=zlon1, xmax=zlon2, integral = True)
        else : zw = 0.
        output_tmp['boxes'][kbox]['asflux'] = zw # positive sign if flux entering the box
    #
    del zflx
    #
    
    #-------------------
    # SMS (air-sea flux removed)
    #-------------------
    
    print('>> SMS (air-sea flux removed)')

    # read trend
    zwftrd = fdir + vsim + fsuftrd
    zw = reading.read_ncdf('SMS_'+carbtrd, zwftrd, time = time_span)
    zsms = zw['data']
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zw = diag_trend(zsms, zwmesh, vbox)
        output_tmp['boxes'][kbox]['smsonl'] = zw - output_tmp['boxes'][kbox]['asflux']
        #
    del zsms
    
    #-------------------
    # FOR
    #-------------------
    
    print('>> FOR')

    # read trend
    zwftrd = fdir + vsim + fsuftrd
    zw = reading.read_ncdf('FOR_'+carbtrd, zwftrd, time = time_span)
    zfor = zw['data']
    # loop on boxes
    for kbox, vbox in boxes.items() :
        zw = diag_trend(zfor, zwmesh, vbox)
        output_tmp['boxes'][kbox]['foronl'] = zw
        #
    del zfor
    
    ##################################################
    # SAVE OUTPUT_TMP (ANNUAL VALUES)
    ##################################################
    
    print('>> Save output_tmp (annual values)')

    f = open(dirout+fileout_tmp, 'wb')
    pickle.dump(output_tmp, f)
    f.close()
    print("File saved: "+dirout+fileout_tmp)
#
##################################################
##################################################
#        END COMPUTE BUDGET ANNUAL VALUES        #
##################################################
##################################################

    
##################################################
##################################################
#        COMPUTE BUDGET MEAN OVER ALL YEARS      #
##################################################
##################################################

##################################################
# LOAD FILE ANNUAL VALUES
##################################################

if not bcompute_annual_means :
    print('> Load annual values')
    savedfile = dirout + fileout_tmp
    with open(savedfile, 'rb') as f1: output_tmp = pickle.load(f1)
    print("File loaded: "+savedfile)
#

##################################################
# COMPUTE
##################################################


print('> Compute mean over all years')

output = {}
for kkk1, vvv1 in output_tmp.items():
    if kkk1=='boxes':
        output[kkk1] = {}
        for kkk2, vvv2 in vvv1.items(): # kkk2 is box1 or box2...
            output[kkk1][kkk2] = {}
            for kkk3, vvv3 in vvv2.items(): # kkk3 is xmin or xmax or... or hadonl or zdfonl...
                if kkk3 in boxes[kkk2].keys(): output[kkk1][kkk2][kkk3] = vvv3 # xmin, xmax...
                else: output[kkk1][kkk2][kkk3] = np.nanmean(vvv3, axis=0) # hadonl, zadonl...
            #
        #
    else: output[kkk1] = vvv1
#

##################################################
# SAVE OUTPUT (OVERALL MEAN)
##################################################

print('> Save output')
f = open(dirout+fileout, 'wb')
pickle.dump(output, f)
f.close()
print("File saved: "+dirout+fileout)

##################################################
##################################################
#      END COMPUTE BUDGET MEAN OVER ALL YEARS    #
##################################################
##################################################

