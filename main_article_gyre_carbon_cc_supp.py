from IPython import embed
import os
import pickle
import datetime

# import functions_GYRE as fGYRE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('./matplotlibrc_nature_cc.mplstyle')

from FGYRE import reading, interpolating, averaging, \
    plotting, stream_functions

dirout = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-ARTICLE-GYRE-CARBON-CC-SUPP/'
if (not os.path.isdir(dirout)) : os.mkdir(dirout)

r1s  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000']
res  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000', '9', '27']
rplt = ['AVG1', '+STD', '-STD', 'R9', 'R27']
unc = 1. # number of std for uncertainty

plot_beta_gamma_radcou_vs_bgccou    = False
plot_temp_sali_solub   = True


##################################################
# BETA GAMMA RADCOU VS BGCCOU
##################################################

if plot_beta_gamma_radcou_vs_bgccou:

    print('PLOT BETA GAMMA RADCOU VS BGCCOU')
    print(datetime.datetime.now())

    savefig       = dirout + 'beta-gamma-radcou-vs-bgccou.png'
    savedata2plot = dirout + 'data2plot-beta-gamma-radcou-vs-bgccou.pckl'
    force_computation = False

    #====================
    # PREPARE DATA
    #====================

    if (not os.path.isfile(savedata2plot)) | force_computation :
        print('> compute data2plot')
        print(datetime.datetime.now())
        
        fdir = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
        fsuf = '_1y_01010101_01701230_diad_T.xml'
        start_date  = '0101-01-01'
        end_date    = '0170-12-31'   
        
        data2plot={}
        
        zfact=1e-3*3600*24*360*10 # mmolC/m2/s -> mmolC/m2/y (factor 10
                                  # because the flux saved is divided by
                                  # the thicknes of the first ocean layer)
        
        #--------------------
        # 1PCTCO2
        #--------------------
        aco2 = np.loadtxt('/gpfswork/rech/eee/rdyk004/MY-PYTHON3/1pctCO2.txt')
        aco2cc = aco2[1:71, 1]
        
        #--------------------
        # ATMT CHANGE
        #--------------------
        zw = []
        for yyy in range(70) :
            zw.append(yyy*0.04)
        #
        atmT = np.array(zw)
        
        #--------------------
        # COMPUTE CARBON UPTAKE IN CTL, BGC AND COU SIMULATIONS
        #--------------------
        
        tempdict = {}
        for vkR in res :
        
            tempdict['R'+vkR] = {}
            if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
            else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
            #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST
        
            # CTL
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('Cflx', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.cumsum(zw1, axis=0) # time cumulative integral
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            zwctl= np.array(zw1)
            
            # BGC
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('Cflx2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.cumsum(zw1, axis=0) # time cumulative integral
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            zwbgc = np.array(zw1)
            
            # RAD
            fff = fdir + 'CC' + vkR + fsuf
            #fff = fdir + 'CC' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('Cflx', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.cumsum(zw1, axis=0) # time cumulative integral
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            zwrad = np.array(zw1)
            
            # COU
            fff = fdir + 'CC' + vkR + fsuf
            #fff = fdir + 'CC' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('Cflx2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.cumsum(zw1, axis=0) # time cumulative integral
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            zwcou = np.array(zw1)
        
            tempdict['R'+vkR]['COU - CTL'] = zwcou - zwctl
            tempdict['R'+vkR]['COU - BGC'] = zwcou - zwbgc
            tempdict['R'+vkR]['COU - RAD'] = zwcou - zwrad
            tempdict['R'+vkR]['BGC - CTL'] = zwbgc - zwctl
            tempdict['R'+vkR]['RAD - CTL'] = zwrad - zwctl
        
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        tempdict['AVG1'], tempdict['STD1'], tempdict['+STD'], tempdict['-STD'] = {}, {}, {}, {}
        for vkS in tempdict['R'+res[0]].keys() : 
            zw = []
            for vkR in r1s : zw.append(tempdict['R'+vkR][vkS])
            zw = np.array(zw)
            tempdict['AVG1'][vkS] = np.nanmean(zw, axis=0)
            tempdict['STD1'][vkS] = unc*np.nanstd(zw, axis=0)
            tempdict['+STD'][vkS] = tempdict['AVG1'][vkS] + tempdict['STD1'][vkS]
            tempdict['-STD'][vkS] = tempdict['AVG1'][vkS] - tempdict['STD1'][vkS]
        #
            
        
        data2plot = {
            'year' : np.arange(1, 71),
            'ATMCO2': aco2cc,
            'ATMT': atmT
            }
        for vkS in tempdict['R'+res[0]].keys():
            data2plot[vkS] = { 'AVG1': tempdict['AVG1'][vkS],
                               '+STD': tempdict['+STD'][vkS],
                               '-STD': tempdict['-STD'][vkS],
                               'R9'  : tempdict['R9'  ][vkS],
                               'R27' : tempdict['R27' ][vkS]}
            
        print('> save data2plot')
        print(datetime.datetime.now())
        with open(savedata2plot, 'wb') as f1: pickle.dump(data2plot, f1)
        print("File saved: "+savedata2plot)
        print(datetime.datetime.now())
    #
    else :
        print("> load data2plot")
        print(datetime.datetime.now())
        with open(savedata2plot, 'rb') as f1: data2plot = pickle.load(f1)
        print("File loaded: "+savedata2plot)
        print(datetime.datetime.now())
    #
    

    #====================
    # PLOT
    #====================

    print("> plot")
    print(datetime.datetime.now())

    infact  = 1/2.54
    ncol, nrow = 3, 1
    fsize = (ncol*5*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()
    www={'R9':1.5, 'R27':1.5, \
         'AVG1':1.5, '+STD':1, '-STD':1}
    lll={'R9':'-', 'R27':'-', \
         'AVG1':'-'}
    ccc={'AVG1':'black', \
         'R9':'dodgerblue', 'R27':'orange'}
    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg $\pm$ st. dev.', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}


    kwlines = {'R9'  : dict(lw=1.5, color='dodgerblue', ls='-'),
               'R27' : dict(lw=1.5, color='orange'    , ls='--', dashes=[2., 1.]),
               'AVG1': dict(lw=1.5, color='black'     , ls='-'),
               'UNC' : dict(color='black', alpha=.3)}

    #====================
    # PLOT
    #====================

    print("> plot")
    print(datetime.datetime.now())

    infact  = 1/2.54
    ncol, nrow = 2, 2
    fsize = (ncol*5*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey='row', sharex='row') # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()

    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg $\pm$ st. dev.', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}

    kwlines = {'R9'  : dict(lw=1.5, color='dodgerblue', ls='-'),
               'R27' : dict(lw=1.5, color='orange'    , ls='--', dashes=[2., 1.]),
               'AVG1': dict(lw=1.5, color='black'     , ls='-'),
               'UNC' : dict(color='black', alpha=.3)}

    #--------------------
    # BETA : BGC - CTL VS ATMCO2 ACCUMULATION
    #--------------------

    annot = {}

    vdata = data2plot['BGC - CTL']
    zax = ax[0, 0]
    X = data2plot['ATMCO2']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    md, b = np.polyfit(X, Yd, 1)
    mu, b = np.polyfit(X, Yu, 1)
    annot['STD'] = str(np.round(.5*(mu - md), decimals=2))
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        m, b = np.polyfit(X, Y, 1)
        annot[vres]=str(np.round(m, decimals=2))
        ooo+=5
    #

    zax.set_title('('+subnum.pop()+r') $\beta=\Delta C_{BGC}/\Delta C_{atm}$', loc='left')
    zax.set_xlabel('Atm. CO$_2$ accumulation [ppm]')

    #____________________
    # annotations

    vres = 'AVG1'
    idx0 = 40
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0+15, y0-8
    text = annot[vres] + ' $\pm$ ' + annot['STD']
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))


    vres = 'R9'
    idx0 = 50
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-80, y0+2
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    vres = 'R27'
    idx0 = 62
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-80, y0+2
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    
    #--------------------
    # BETA : COU - RAD VS ATMCO2 ACCUMULATION
    #--------------------

    annot = {}

    vdata = data2plot['COU - RAD']
    zax = ax[0, 1]
    X = data2plot['ATMCO2']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    md, b = np.polyfit(X, Yd, 1)
    mu, b = np.polyfit(X, Yu, 1)
    annot['STD'] = str(np.round(.5*(mu - md), decimals=2))
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        m, b = np.polyfit(X, Y, 1)
        annot[vres]=str(np.round(m, decimals=2))
        ooo+=5
    #

    zax.set_title('('+subnum.pop()+r') $\beta = (\Delta C_{COU} - \Delta C_{RAD})/\Delta C_{atm}$', loc='left')
    zax.set_xlabel('Atm. CO$_2$ accumulation [ppm]')

    #____________________
    # annotations

    vres = 'AVG1'
    idx0 = 40
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0+15, y0-8
    text = annot[vres] + ' $\pm$ ' + annot['STD']
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))


    vres = 'R9'
    idx0 = 50
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-80, y0+2
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    vres = 'R27'
    idx0 = 62
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-80, y0+2
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    
    #--------------------
    # GAMMA: COU - BGC VS TEMPERATURE CHANGE
    #--------------------

    annot = {}

    vdata = data2plot['COU - BGC']
    zax = ax[1, 0]
    X = data2plot['ATMT']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    md, b = np.polyfit(X, Yd, 1)
    mu, b = np.polyfit(X, Yu, 1)
    annot['STD'] = str(np.round(.5*(mu - md), decimals=2))
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        m, b = np.polyfit(X, Y, 1)
        annot[vres]=str(np.round(m, decimals=2))
        ooo+=5
    #

    zax.set_title('('+subnum.pop()+r') $\gamma=(\Delta C_{COU}-\Delta C_{BGC})/\Delta T$', loc='left')
    zax.set_xlabel('Change in temperature [°C]')

    #____________________
    # annotations

    vres = 'AVG1'
    idx0 = 62
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-1.5, y0-2
    text = annot[vres] + ' $\pm$ ' + annot['STD']
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))


    vres = 'R9'
    idx0 = 59
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-1, y0+.5
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    vres = 'R27'
    idx0 = 50
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0+0.3, y0+1.5
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    #--------------------
    # GAMMA: RAD - CTL VS TEMPERATURE CHANGE
    #--------------------

    annot = {}

    vdata = data2plot['RAD - CTL']
    zax = ax[1, 1]
    X = data2plot['ATMT']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    md, b = np.polyfit(X, Yd, 1)
    mu, b = np.polyfit(X, Yu, 1)
    annot['STD'] = str(np.round(.5*(mu - md), decimals=2))
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        m, b = np.polyfit(X, Y, 1)
        annot[vres]=str(np.round(m, decimals=2))
        ooo+=5
    #

    zax.set_title('('+subnum.pop()+r') $\gamma=\Delta C_{RAD}/\Delta T$', loc='left')
    zax.set_xlabel('Change in temperature [°C]')

    #____________________
    # annotations

    vres = 'AVG1'
    idx0 = 65
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-1.2, y0-5
    text = annot[vres] + ' $\pm$ ' + annot['STD']
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))


    vres = 'R9'
    idx0 = 55
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0-1, y0-2
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    vres = 'R27'
    idx0 = 50
    x0 , y0  = X[idx0], vdata[vres][idx0]
    x0t, y0t = x0+0.15, y0+1.5
    text = annot[vres]
    zax.annotate(text, xy=(x0, y0), xytext=(x0t, y0t), \
                 bbox=dict(boxstyle="round", fc="1", ec=kwlines[vres]['color']),\
                 arrowprops=dict(arrowstyle="-", shrinkA=0, color=kwlines[vres]['color']))

    for zax in ax.flat:
        zax.set_ylabel('Change in cumul.\nCO$_2$ uptake [molC/m$^2$]')
    #
    for zax in ax[:, 1].flat:
        zax.yaxis.set_ticks_position('right')
        zax.yaxis.set_label_position('right')
    #

    fig.tight_layout()

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    
#

##################################################
# END FIG2
##################################################

##################################################
# PLOT TEMP SALI SOLUB
##################################################
if plot_temp_sali_solub :

    print('PLOT TEMPERATURE SALINITY SOLUBILITY')
    print(datetime.datetime.now())

    savefig       = dirout + 'temp_sali_solub.png'

    zmax = 100
    
    #====================
    # PREPARE DATA
    #====================

    tempdict = {'temp':{}, 'sali':{}, 'solub':{}}

    fdir  = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'          
    fsuf  = '_1y_01010101_01701230_grid_T.xml'
    start_date  = '0101-01-01'
    end_date    = '0170-12-31'
    for vkR in res :
    
        if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
        else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
        #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST

        temp, sali, solub = {}, {}, {}
        for simu in ['CTL', 'CC']:
            fff = fdir + simu + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST

            # TEMP
            zw1  = reading.read_ncdf('votemper', fff, time=(start_date, end_date))
            zw1 = zw1['data']
            zw1 = averaging.zmean(zw1, zwmesh, zmax=zmax, dim='tzyx')
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            temp[simu]= np.array(zw1)

            # SALI
            zw1  = reading.read_ncdf('vosaline', fff, time=(start_date, end_date))
            zw1 = zw1['data']
            zw1 = averaging.zmean(zw1, zwmesh, zmax=zmax, dim='tzyx')
            zw1 = averaging.ymean(zw1, zwmesh, dim='tyx')
            zw1 = averaging.xmean(zw1, zwmesh, dim='tx')
            sali[simu]= np.array(zw1)

            # SOLUB
            # """
            # Compute solubility of CO2 from Weiss (1974)
            # Kh = [CO2]/ p CO2, Weiss (1974), refitted for moist air Weiss and Price (1980)
        
            # inputs: 
            #     ptho    = temperature [degreeC]
            #     psao    = salinity    [psu]
            # outputs: (Kh, Khd) in [mol/l/ppm] and [mol/l/uatm]
            #     Kh: CO2 solubility for use with partial pressure in dry air 
            #         (include correction for accounting for water vapor) in [mol/l/ppm]
            # """
        
            # # Carbon chemistry: Caculate equilibrium constants and solve for [H+] and
            # # carbonate alkalinity (ac)
            tzero   = 273.15
            ttk     = temp[simu] + tzero
            ttk100 = ttk/100.0
            ac1 = -160.7333
            ac2 = 215.4152
            ac3 = 89.8920
            ac4 = -1.47759
            bc1 = 0.029941
            bc2 = -0.027455
            bc3 = 0.0053407
            # Kh = [CO2]/ xCO2 dry mole fraction of [CO2]
            # Weiss (1974), refitted for moist air Weiss and Price (1980) [mol/l/atm]
            nKhwe74 = ac1 + ac2/ttk100 + ac3*np.log(ttk100) + ac4*ttk100**2 + sali[simu]*(bc1 + bc2*ttk100 + bc3*ttk100**2)
            Kh     = np.exp(nKhwe74) # in mol/l/atm
            # Kh: CO2 solubility for use with partial pressure in dry air 
            #     (include correction for accounting for water vapor)
            # solub[simu] = Kh*1e-6 # in  mol/l/uatm
            # solub[simu] = Kh*1e6    # in  mmol/m3/atm
            solub[simu] = Kh*1000    # in  umol/m3/uatm o umol/m3/ppm
        #
        
        tempdict['temp']['R'+vkR]={}
        tempdict['temp']['R'+vkR]['CTL'] = temp['CTL']
        tempdict['temp']['R'+vkR]['COU'] = temp['CC']
        tempdict['temp']['R'+vkR]['COU - CTL'] = temp['CC'] - temp['CTL']

        tempdict['sali']['R'+vkR]={}
        tempdict['sali']['R'+vkR]['CTL'] = sali['CTL']
        tempdict['sali']['R'+vkR]['COU'] = sali['CC']
        tempdict['sali']['R'+vkR]['COU - CTL'] = sali['CC'] - sali['CTL']

        tempdict['solub']['R'+vkR]={}
        tempdict['solub']['R'+vkR]['CTL'] = solub['CTL']
        tempdict['solub']['R'+vkR]['COU'] = solub['CC']
        tempdict['solub']['R'+vkR]['COU - CTL'] = solub['CC'] - solub['CTL']

    #
    

    #--------------------
    # AVG1, STD1
    #--------------------
    for vvv in ['temp', 'sali', 'solub']: 
        tempdict[vvv]['AVG1'], tempdict[vvv]['STD1'] = {}, {}
        tempdict[vvv]['+STD'], tempdict[vvv]['-STD'] = {}, {}
        for vkS in ['CTL', 'COU', 'COU - CTL'] :
            zw = []
            for vkR in r1s : zw.append(tempdict[vvv]['R'+vkR][vkS])
            zw = np.array(zw)
            tempdict[vvv]['AVG1'][vkS] = np.nanmean(zw, axis=0)
            tempdict[vvv]['STD1'][vkS] = unc*np.nanstd(zw, axis=0)
            tempdict[vvv]['+STD'][vkS] = tempdict[vvv]['AVG1'][vkS] + tempdict[vvv]['STD1'][vkS]
            tempdict[vvv]['-STD'][vkS] = tempdict[vvv]['AVG1'][vkS] - tempdict[vvv]['STD1'][vkS]
        #
    #
        
    #--------------------
    # DATA2PLOT
    #--------------------
    data2plot = {
        'year' : np.arange(1, 71),
        'temp': {
            'AVG1': tempdict['temp']['AVG1']['COU - CTL'],
            '+STD': tempdict['temp']['+STD']['COU - CTL'],
            '-STD': tempdict['temp']['-STD']['COU - CTL'],
            'R9'  : tempdict['temp']['R9'  ]['COU - CTL'],
            'R27' : tempdict['temp']['R27' ]['COU - CTL']
        },
        'sali': {
            'AVG1': tempdict['sali']['AVG1']['COU - CTL'],
            '+STD': tempdict['sali']['+STD']['COU - CTL'],
            '-STD': tempdict['sali']['-STD']['COU - CTL'],
            'R9'  : tempdict['sali']['R9'  ]['COU - CTL'],
            'R27' : tempdict['sali']['R27' ]['COU - CTL']
        }, 
        'solub': {
            'AVG1': tempdict['solub']['AVG1']['COU - CTL'],
            '+STD': tempdict['solub']['+STD']['COU - CTL'],
            '-STD': tempdict['solub']['-STD']['COU - CTL'],
            'R9'  : tempdict['solub']['R9'  ]['COU - CTL'],
            'R27' : tempdict['solub']['R27' ]['COU - CTL']
        }
    }

    
    #====================
    # PLOT
    #====================

    
    print("> plot")
    print(datetime.datetime.now())

    infact  = 1/2.54
    ncol, nrow = 3, 1
    fsize = (ncol*6*infact, nrow*4*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey=False, sharex=False) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()
    www={'R9':1.5, 'R27':1.5, \
         'AVG1':1.5, '+STD':1, '-STD':1}
    lll={'R9':'-', 'R27':'-', \
         'AVG1':'-'}
    ccc={'AVG1':'black', \
         'R9':'dodgerblue', 'R27':'orange'}
    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg $\pm$ st. dev.', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}


    kwlines = {'R9'  : dict(lw=1.5, color='dodgerblue', ls='-'),
               'R27' : dict(lw=1.5, color='orange'    , ls='--', dashes=[2., 1.]),
               'AVG1': dict(lw=1.5, color='black'     , ls='-'),
               'UNC' : dict(color='black', alpha=.3)}

    #--------------------
    # TEMP
    #--------------------

    vdata = data2plot['temp']
    zax = ax[0, 0]
    X = data2plot['year']

    lines, labels = [], []
    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        lines.append(zl)
        labels.append(names[vres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') Temperature', loc='left')
    zax.set_ylabel('[°C]')
    zax.legend(lines, labels, loc='best')

    #--------------------
    # SALI
    #--------------------

    vdata = data2plot['sali']
    zax = ax[0, 1]
    X = data2plot['year']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') Salinity', loc='left')
    zax.set_ylabel('[psu]')

    #--------------------
    # Solub
    #--------------------

    vdata = data2plot['solub']
    zax = ax[0, 2]
    X = data2plot['year']

    ooo=5
    # UNCERTAINTY R1
    Yu, Yd = vdata['+STD'], vdata['-STD']
    zax.fill_between(X, Yd, Yu, zorder=ooo, **kwlines['UNC'])
    ooo+=5
    for vres in ['AVG1', 'R9', 'R27'] :
        Y = vdata[vres]
        zl, = zax.plot(X, Y, **kwlines[vres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') Solubility', loc='left')
    zax.set_ylabel('[umol/m3/ppm]')

    #--------------------
    # X AXES
    #--------------------

    from scipy import interpolate
    year   = np.concatenate([np.array([0]), data2plot['year']])

    xticks = [0, 35, 70]

    zax = ax[0, 0]
    zax.xaxis.set_ticks(xticks)
    zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_xlim((0, 70))    

    zax = ax[0, 1]
    zax.xaxis.set_ticks(xticks)
    zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_xlim((0, 70))    
    
    zax = ax[0, 2]
    zax.xaxis.set_ticks(xticks)
    zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_xlim((0, 70))    
    

    for zax in ax.flatten(): zax.axhline(0, lw=1, color='dimgrey')
    
    fig.tight_layout()

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    
#
##################################################
# END PLOT TEMPERATURE SALINITY SOLUBILITY
##################################################

