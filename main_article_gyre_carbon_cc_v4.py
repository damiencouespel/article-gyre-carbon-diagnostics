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

dirout = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-ARTICLE-GYRE-CARBON-CC-V4/'
if (not os.path.isdir(dirout)) : os.mkdir(dirout)

r1s  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000']
res  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000', '9', '27']
rplt = ['AVG1', '+STD', '-STD', 'R9', 'R27']
unc = 1. # number of std for uncertainty

plot_fig1          = True
plot_fig2tst       = False
plot_fig2          = False
plot_fig2bis       = False
plot_fig3          = False
plot_fig3_bis      = False
plot_fig3_ter      = False
savevalues_fig4    = False
savevalues_fig4_1d = False

##################################################
# FIG1
##################################################
if plot_fig1 :

    print('PLOT FIG1')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig1.png'
    savedata2plot = dirout + 'data2plot-fig1.pckl'
    force_computation = False

    #====================
    # PREPARE DATA
    #====================
    
    if (not os.path.isfile(savedata2plot)) | force_computation :

        print('> compute data2plot')
        print(datetime.datetime.now())

        
        data2plot = {}
        #--------------------
        # BSF CTL1
        #--------------------
    
        print('>> compute bsf ctl1')
        print(datetime.datetime.now())

        fdir  = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'          
        fsufV  = '_1y_01010101_01701230_grid_V.xml'
        start_date  = '0166-01-01'
        end_date    = '0170-12-31'
        zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc')
        mapBSF = {}
        for vkR in r1s : 
            fff = fdir + 'CTL' + vkR + fsufV
            zw  = reading.read_ncdf('vomecrty', fff, time=(start_date, end_date))
            zw = zw['data']
            bsf = stream_functions.bsfv(zw, zwmesh)
            mapBSF['CTL'+vkR] = np.mean(bsf, axis = 0)
        #
        # AVG1
        zw = []
        for vkR in r1s : zw.append(mapBSF['CTL'+vkR])
        zw = np.array(zw)
        mapBSF['AVG1'] = np.ma.array(data=np.nanmean(zw, axis=0), mask=mapBSF['CTL1'].mask)
    
        data2plot['mapBSF'] = {'X': zwmesh['lonV'],
                               'Y': zwmesh['latV'],
                               'Z': mapBSF['AVG1']}
        
        #--------------------
        # ATM TEMP
        #--------------------
    
        print('>> compute atm. temp.')
        print(datetime.datetime.now())

        zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
    
        year = np.arange(0, 71)
        zlat = zwmesh['latT'][:, 2]
        zemp_S    = 0.7       # intensity of COS in the South
        zemp_N    = 0.8       # intensity of COS in the North
        zemp_sais = 0.1       #
        zTstar    = 25.6      # intemsity from 25.6 degC at 20 degN !!DC2
        tstarmax = np.mean( (zTstar+25.6/15.2) * np.cos( np.pi*(zlat-20.) / (33.8*(1+8.3/33.8)*2.) ) )
        tstarmin = np.mean( (zTstar-25.6/15.2) * np.cos( np.pi*(zlat-20.) / (33.8*(1-8.3/33.8)*2.) ) )
        tstar = 0.5*(tstarmax+tstarmin)

        tctl = tstar * np.ones_like(year)
        tcc  = tstar + 0.04 * year
        
        data2plot['atmtemp'] = {
            'year': year,
            'CTL': tctl, 
            'CC' : tcc
        }
    
        #--------------------
        # ATM PCO2
        #--------------------

        aco2 = np.loadtxt('/gpfswork/rech/eee/rdyk004/MY-PYTHON3/1pctCO2.txt')
        year = np.arange(0, 71)
        aco2ctl = 280.*np.ones_like(year)
        aco2cc = aco2[:71, 1] 
    
        data2plot['atmco2']= {'year':year, 'CTL':aco2ctl, 'CC':aco2cc}
        
        #--------------------
        # SNAPSHOT
        #--------------------
    
        print('>> compute carbon spinup, ctl, bgc, cou')
        print(datetime.datetime.now())

        data2plot['snapshot'] = {}
        
        fdir = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'          
        fsuf = '_2d_01660101_01701230_diad_T.xml'
    
        tempdict = {}
        for vkR in res :
            if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
            else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
            #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST

            if vkR != '27': 
                start_date  = '0166-03-01'
                end_date    = '0166-03-03'
            else: # ca. 6 months shift in the R27 simulation
                start_date  = '0166-10-01'
                end_date    = '0166-10-03'
            #
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('Cflx2', fff, time=(start_date, end_date))
            zw1 = zw1['data'][0]

            tempdict['R'+vkR] = {'X': zwmesh['lonT'],
                                 'Y': zwmesh['latT'],
                                 'Z': zw1}
        #

        # for vkR in ['1', '9', '27'] :
        for vkR in res:
            data2plot['snapshot']['R'+vkR] = {'X': tempdict['R'+vkR]['X'],
                                              'Y': tempdict['R'+vkR]['Y'],
                                              'Z': tempdict['R'+vkR]['Z']}
        #
                                              
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
    ncol, nrow = 3, 2
    fsize = (ncol*5*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()

    #--------------------
    # ATM CO2
    #--------------------

    #zax = plt.subplot2grid((nrow, ncol), (0, 0), colspan=2, fig=fig)
    zax = ax[0, 0]
    zwdata = data2plot['atmco2']
    ll1, = zax.plot(zwdata['year'], zwdata['CTL'], label='CTL'         , c='royalblue') 
    ll2, = zax.plot(zwdata['year'], zwdata['CC'] , label='BGC and COU' , c='firebrick')
    zax.set_title('('+subnum.pop()+') Forcing: atm. pCO$_2$', loc='left')
    zax.set_xlim(0, 70)
    zax.set_xlabel('Time')
    zax.set_ylabel('[ppm]')
    zax.xaxis.set_ticks([0, 70])
    zax.xaxis.set_ticklabels(['0', '70 years'])
    #zax.yaxis.set_ticks_position('right')
    #zax.yaxis.set_label_position('right')
    zax.locator_params(axis='y', nbins=6)
    leg = zax.legend(handles=[ll1, ll2], loc='best')
    leg._legend_box.align = "left"

    #--------------------
    # ATM TEMP
    #--------------------

    zax = ax[0, 1]
    zwdata = data2plot['atmtemp']
    ll1, = zax.plot(zwdata['year'], zwdata['CTL'], label='CTL and BGC', color='royalblue')
    ll2, = zax.plot(zwdata['year'], zwdata['CC' ], label='COU'        , color='firebrick')
    zax.set_title('('+subnum.pop()+') Forcing: atm. temperature\n     at sea surface', loc='left')
    zax.set_xlim(0, 70)
    zax.set_xlabel('Time')
    zax.set_ylabel('[°C]')
    zax.xaxis.set_ticks([0, 70])
    zax.xaxis.set_ticklabels(['0', '70 years'])
    # zax.yaxis.set_ticks_position('right')
    # zax.yaxis.set_label_position('right')
    zax.locator_params(axis='y', nbins=6)
    leg = zax.legend(handles=[ll1, ll2], loc='best')
    leg._legend_box.align = "left"

    
    #--------------------
    # BSF
    #--------------------

    
    zwdata = data2plot['mapBSF']
    zax = ax[0, 2]    
    X, Y, Z = zwdata['X'], zwdata['Y'], zwdata['Z']
    aa = np.linspace(-30, 0, 7)
    bb = np.linspace(0, 30, 7)
    levBSF = np.concatenate((aa[:-1], bb))
    cmapBSF = 'RdBu'
    cf = zax.contourf(X, Y, Z, levels=levBSF, cmap=cmapBSF, extend='both')
    cl = zax.contour(X, Y, Z, levels=[-30, -20, -10, 10, 20, 30], colors='k', linestyles='solid', linewidths=.5)
    #cl = zax.contour(X, Y, Z, levels=levBSF, colors='k', inline=True, ls='-', lw=.5)
    zax.clabel(cl, inline=True, fontsize=6, fmt='%.0f')
    plotting.make_XY_axis(zax, title='('+subnum.pop()+') Bar. circ., 1° [Sv]')

    # zax.tick_params(bottom=False)
    # zax.label_outer()
    zax.yaxis.set_ticks_position('right')
    zax.yaxis.set_label_position('right')

    #--------------------
    # SNAPSHOT
    #--------------------

    kwmapstick = dict(yticks=[20, 34.2987, 43, 48.5974], yticklabels=['0', '1,590', '2,560', '3,180'])

    zwdata = data2plot['snapshot']

    zlevmax = 15
    # aa = np.linspace(-zlevmax, 0, 12)
    # bb = np.linspace(0, zlevmax, 12)
    # lev = np.concatenate((aa[:-1], bb))
    lev = np.linspace(-zlevmax, zlevmax, 100)
    tickbar = [-15, -10, -5, 0, 5, 10, 15]
    cmap = 'RdBu'

    iplt = 3

    note={'R1'               :'1°', \
          'R1_KL500'         :'1°', \
          'R1_KL2000'        :'1°', \
          'R1_KGM500_KL500'  :'1°', \
          'R1_KGM2000_KL2000':'1°', \
          'R9':'1/9°', 'R27':'1/27°'}
    ttls={'R1'               :'1°, k$_{gm}$=1e$^3$, k$_{iso}$=1e$^3$', \
          'R1_KL500'         :'1°, k$_{gm}$=1e$^3$, k$_{iso}$=500'      , \
          'R1_KL2000'        :'1°, k$_{gm}$=1e$^3$, k$_{iso}$=2e$^3$', \
          'R1_KGM500_KL500'  :'1°, k$_{gm}$=500, k$_{iso}$=500'      , \
          'R1_KGM2000_KL2000':'1°, k$_{gm}$=2e$^3$, k$_{iso}$=2e$^3$', \
          'R9':'1/9°', 'R27':'1/27°'}
    for ires, vres in enumerate(['R1', 'R9', 'R27']):

        zax = ax.flat[iplt]
        X = zwdata[vres]['X']
        Y = zwdata[vres]['Y']
        Z = zwdata[vres]['Z'] * 3600*24*10 # mol/m2/d, nb: factor 10 because the carbon flux is divide by the depth of the first row in the model
        cf = zax.contourf(X, Y, Z, levels=lev, cmap=cmap, extend='both')
        # zax.annotate(note[vres], xy=(-83, 23), \
        #              bbox=dict(boxstyle="round", fc="1"))
        plotting.make_XY_axis(zax, title='('+subnum.pop()+') Air-sea CO$_2$ flux\n     '+ttls[vres]+'', **kwmapstick)
        # zax.axhline(34.2987, lw=1., ls='-', c='k')
        # zax.axhline(43., lw=1., ls='-', c='k')
        iplt+=1
    #

    # kwnote = dict(boxstyle="round", fc="1", linewidth=.5, alpha=.5)
    # ax[1, 0].annotate('STG', xy=(-84, 21.5)   , fontsize=6, c='k', bbox=kwnote)
    # ax[1, 0].annotate('SPG', xy=(-84, 35.7987), fontsize=6, c='k', bbox=kwnote)
    # ax[1, 0].annotate('CZ' , xy=(-84, 44.5)   , fontsize=6, c='k', bbox=kwnote)
    
    # ax[1, 0].tick_params(bottom=False)
    ax[1, 0].label_outer()
    # ax[1, 1].tick_params(bottom=False)
    ax[1, 1].label_outer()
    # ax[1, 1].yaxis.set_ticks_position('right')
    # ax[1, 1].yaxis.set_label_position('right')
    ax[1, 1].tick_params(left=False)
    #ax[1, 2].label_outer()
    ax[1, 2].yaxis.set_ticks_position('right')
    ax[1, 2].yaxis.set_label_position('right')

    # Adjust poistion of snapshots
    for zax in ax[1, :].flatten():
        zw = zax.get_position()
        ny0 = zw.y0 - 0.7*zw.height
        zax.set_position([zw.x0, ny0, zw.width, zw.height])
    #

    # Adjust size of forcings
    zax = ax[0, 0]
    zw = zax.get_position()
    nwidth = 0.9*zw.width
    nx0 = zw.x0+0.05*zw.width
    zax.set_position([nx0, zw.y0, nwidth, zw.height])
    zax = ax[0, 1]
    zw = zax.get_position()
    nwidth = 0.9*zw.width
    nx0 = zw.x0+0.05*zw.width
    zax.set_position([nx0, zw.y0, nwidth, zw.height])
    


    
    # cbtitle = 'CO$_2$ flux on the 3$^{rd}$ of March [mol$\,$C$\,$m$^{-2}$$\,$d$^{-1}$]'
    cbtitle = '[mol$\,$C$\,$m$^{-2}$$\,$d$^{-1}$]'
    zw1 = ax[-1, 0].get_position()
    zw2 = ax[-1, -1].get_position()
    nx0 = zw1.x0 + 0.5*zw1.width
    nw  = zw2.x0 + 0.5*zw2.width - nx0
    nh  = 0.05*nw
    ny0 = zw1.y1 + 0.25*zw1.height
    cbar_ax = fig.add_axes([nx0, ny0, nw, nh])
    fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticklocation='top', label=cbtitle, ticks=tickbar)
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar


    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()
#
##################################################
# END FIG1
##################################################



##################################################
# FIG2 TST
##################################################

if plot_fig2tst :

    print('PLOT FIG2TST')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig2tst.png'
    savedata2plot = dirout + 'data2plot-fig2tst.pckl'
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
            tempdict['R'+vkR]['BGC - CTL'] = zwbgc - zwctl
        
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        tempdict['AVG1'], tempdict['STD1'], tempdict['+STD'], tempdict['-STD'] = {}, {}, {}, {}
        for vkS in ['COU - CTL', 'BGC - CTL', 'COU - BGC'] : 
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
            'ATMT': atmT, 
            'COU - CTL': {
                    'AVG1': tempdict['AVG1']['COU - CTL'],
                    '+STD': tempdict['+STD']['COU - CTL'],
                    '-STD': tempdict['-STD']['COU - CTL'],
                    'R9'  : tempdict['R9'  ]['COU - CTL'],
                    'R27' : tempdict['R27' ]['COU - CTL']
                },
            'COU - BGC': {
                    'AVG1': tempdict['AVG1']['COU - BGC'],
                    '+STD': tempdict['+STD']['COU - BGC'],
                    '-STD': tempdict['-STD']['COU - BGC'],
                    'R9'  : tempdict['R9'  ]['COU - BGC'],
                    'R27' : tempdict['R27' ]['COU - BGC']
                }, 
            'BGC - CTL': {
                    'AVG1': tempdict['AVG1']['BGC - CTL'],
                    '+STD': tempdict['+STD']['BGC - CTL'],
                    '-STD': tempdict['-STD']['BGC - CTL'],
                    'R9'  : tempdict['R9'  ]['BGC - CTL'],
                    'R27' : tempdict['R27' ]['BGC - CTL']
                }
            }
            
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
    ncol, nrow = 1, 2
    fsize = (ncol*7*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey=True, sharex=True) # nrow, ncol
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
    # COU - CTL VS TIME
    #--------------------

    vdata = data2plot['COU - CTL']
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
    zax.set_ylabel(r'mol$\,$C$\,$m$^{-2}$')
    zax.set_title('('+subnum.pop()+r') CO$_2$ uptake in the COU simulation', loc='left')
    zax.legend(lines, labels, loc='best')

    #--------------------
    # BETA : BGC - CTL VS ATMCO2 ACCUMULATION
    # GAMMA: COU - BGC VS TEMPERATURE CHANGE
    #--------------------

    zax = ax[1, 0]
    # X = data2plot['ATMCO2']
    X = data2plot['year']

    vdata = data2plot['BGC - CTL']
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

    vdata = data2plot['COU - BGC']
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

    zax.set_title('('+subnum.pop()+r') CO$_2$ increase VS temperature increase', loc='left')
    bboxdict = dict(boxstyle="round", fc="1", linewidth=.5, alpha=1)
    zax.annotate('BGC simulation\nCarbon-concentration feedback' , xy=(2, 40)   , fontsize=6, c='k', bbox=bboxdict)
    zax.annotate('COU$-$BGC\nCarbon-climate feedback' , xy=(2, -16)   , fontsize=6, c='k', bbox=bboxdict)

    xticks = [0, 35, 70]

    zax.set_ylabel(r'mol$\,$C$\,$m$^{-2}$')
    zax.locator_params(axis='y', nbins=6)
    zax.xaxis.set_ticks(xticks)
    zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_xlim((0, 70))

    #____________________
    # add second xaxis
    
    zax2 = zax.twiny()
    # Move twinned axis ticks and label from top to bottom
    zax2.xaxis.set_ticks_position("bottom")
    zax2.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    zax2.spines["bottom"].set_position(("axes", -0.20))
    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    zax2.set_frame_on(True)
    zax2.patch.set_visible(False)
    # as @ali14 pointed out, for python3, use this
    # for sp in ax2.spines.values():
    # and for python2, use this
    # for sp in zax2.spines.itervalues():
    for sp in zax2.spines.values():
        sp.set_visible(False)
    zax2.spines["bottom"].set_visible(True)

    #____________________
    # add third xaxis

    zax3 = zax.twiny()
    # Move twinned axis ticks and label from top to bottom
    zax3.xaxis.set_ticks_position("bottom")
    zax3.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    zax3.spines["bottom"].set_position(("axes", -0.40))
    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    zax3.set_frame_on(True)
    zax3.patch.set_visible(False)
    # as @ali14 pointed out, for python3, use this
    # for sp in ax2.spines.values():
    # and for python2, use this
    # for sp in zax3.spines.itervalues():
    for sp in zax3.spines.values():
        sp.set_visible(False)
    zax3.spines["bottom"].set_visible(True)

    #____________________
    # ticks and labels for the second and third xaxis

    atmco2 = np.concatenate([np.array([280]), data2plot['ATMCO2']])
    year   = np.concatenate([np.array([0]), data2plot['year']])
    atmt   = np.concatenate([data2plot['ATMT'], np.array([2.8])])
    
    zax2.set_xticks(xticks)
    idx = [np.where(year==yyy)[0][0] for yyy in xticks]
    zax2.set_xticklabels(["%.0f ppm" %v for v in atmco2[idx]])

    zax3.set_xticks(xticks)
    idx = [np.where(year==yyy)[0][0] for yyy in xticks]
    zax3.set_xticklabels(["+%.1f °C" %v for v in atmt[idx]])

    zax.axhline(0, c='dimgrey', lw=.5)
    zax2.xaxis.grid(False)
    zax3.xaxis.grid(False)

    zw = zax.get_position()
    nx0 = zw.x0 - .04*zw.width
    ny0 = zw.y0 + .03*zw.height
    plt.figtext(nx0, ny0, r'Atm. CO$_2$', figure=fig, \
                horizontalalignment='right', verticalalignment='bottom')
    ny0 = ny0 - .2*zw.height
    plt.figtext(nx0, ny0, r'Atm. temp.', figure=fig, \
                horizontalalignment='right', verticalalignment='bottom')
    
    fig.tight_layout()

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    
#


####################
# FIG2
####################

if plot_fig2 :

    print('PLOT FIG2')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig2.png'
    savedata2plot = dirout + 'data2plot-fig2.pckl'
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
            tempdict['R'+vkR]['BGC - CTL'] = zwbgc - zwctl
        
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        tempdict['AVG1'], tempdict['STD1'], tempdict['+STD'], tempdict['-STD'] = {}, {}, {}, {}
        for vkS in ['COU - CTL', 'BGC - CTL', 'COU - BGC'] : 
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
            'ATMT': atmT, 
            'COU - CTL': {
                    'AVG1': tempdict['AVG1']['COU - CTL'],
                    '+STD': tempdict['+STD']['COU - CTL'],
                    '-STD': tempdict['-STD']['COU - CTL'],
                    'R9'  : tempdict['R9'  ]['COU - CTL'],
                    'R27' : tempdict['R27' ]['COU - CTL']
                },
            'COU - BGC': {
                    'AVG1': tempdict['AVG1']['COU - BGC'],
                    '+STD': tempdict['+STD']['COU - BGC'],
                    '-STD': tempdict['-STD']['COU - BGC'],
                    'R9'  : tempdict['R9'  ]['COU - BGC'],
                    'R27' : tempdict['R27' ]['COU - BGC']
                }, 
            'BGC - CTL': {
                    'AVG1': tempdict['AVG1']['BGC - CTL'],
                    '+STD': tempdict['+STD']['BGC - CTL'],
                    '-STD': tempdict['-STD']['BGC - CTL'],
                    'R9'  : tempdict['R9'  ]['BGC - CTL'],
                    'R27' : tempdict['R27' ]['BGC - CTL']
                }
            }
            
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
    fsize = (ncol*5*infact, nrow*7*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey='row', sharex=False) # nrow, ncol
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
    # COU - CTL VS TIME
    #--------------------

    vdata = data2plot['COU - CTL']
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
    zax.set_title('('+subnum.pop()+') COU $-$ CTL\n', loc='left')
    zax.set_ylabel('[mol$\,$C$\,$m$^{-2}$]')
    zax.set_ylim((-20, 60))
    zax.locator_params(axis='y', nbins=6)
    zax.legend(lines, labels, loc='best')

    #--------------------
    # BETA : BGC - CTL VS ATMCO2 ACCUMULATION
    #--------------------

    vdata = data2plot['BGC - CTL']
    zax = ax[0, 1]
    X = data2plot['ATMCO2']

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
    zax.set_title('('+subnum.pop()+') BGC $-$ CTL\n     Carbon-concentration feedback', loc='left')

    #--------------------
    # GAMMA: COU - BGC VS TEMPERATURE CHANGE
    #--------------------

    vdata = data2plot['COU - BGC']
    zax = ax[0, 2]
    X = data2plot['ATMT']

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
    zax.set_title('('+subnum.pop()+') COU$-$BGC\n     Carbon-climate feedback', loc='left')

    #--------------------
    # X AXES
    #--------------------

    from scipy import interpolate
    atmco2 = np.concatenate([np.array([280]), data2plot['ATMCO2']])
    year   = np.concatenate([np.array([0]), data2plot['year']])
    atmt   = np.concatenate([data2plot['ATMT'], np.array([2.8])])

    #____________________
    # ax[0, 0]

    xticks = [0, 35, 70]
    zax = ax[0, 0]
    zax.xaxis.set_ticks(xticks)
    zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_xlim((0, 70))    

    
    #____________________
    # ax[0, 1]

    zax = ax[0, 1]
    tmin, tmax = atmco2[0], atmco2[-1]
    xticks = [tmin, .5*(tmin+tmax), tmax]
    zax.set_xticks(xticks)
    zax.set_xticklabels(["%.0f " %v for v in xticks])
    zax.set_xlabel(r'Atm. CO$_2$ [ppm]')
    zax.set_xlim((xticks[0], xticks[-1]))    

    zax2 = zax.twiny()
    # Move twinned axis ticks and label from top to bottom
    zax2.xaxis.set_ticks_position("bottom")
    zax2.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    zax2.spines["bottom"].set_position(("axes", -0.30))
    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    zax2.set_frame_on(True)
    zax2.patch.set_visible(False)
    # as @ali14 pointed out, for python3, use this
    # for sp in ax2.spines.values():
    # and for python2, use this
    # for sp in zax2.spines.itervalues():
    for sp in zax2.spines.values():
        sp.set_visible(False)
    zax2.spines["bottom"].set_visible(True)

    zax2.set_xticks(xticks)
    zax2.set_xlim(zax.get_xlim())    
    f = interpolate.interp1d(atmco2, year, fill_value="extrapolate")
    zax2.set_xticklabels(["%.0f years" %v for v in f(xticks)])
    zax2.xaxis.grid(False)

    #____________________
    # add second xaxis for ax[0, 2]

    zax = ax[0, 2]
    tmin, tmax = atmt[0], atmt[-1]
    xticks = [tmin, .5*(tmin+tmax), tmax]
    zax.set_xticks(xticks)
    zax.set_xticklabels(["%.1f " %v for v in xticks])
    zax.set_xlabel('Atm. temp. increase [°C]')
    zax.set_xlim((xticks[0], xticks[-1]))    

    zax2 = zax.twiny()
    # Move twinned axis ticks and label from top to bottom
    zax2.xaxis.set_ticks_position("bottom")
    zax2.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    zax2.spines["bottom"].set_position(("axes", -0.30))
    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    zax2.set_frame_on(True)
    zax2.patch.set_visible(False)
    # as @ali14 pointed out, for python3, use this
    # for sp in ax2.spines.values():
    # and for python2, use this
    # for sp in zax2.spines.itervalues():
    for sp in zax2.spines.values():
        sp.set_visible(False)
    zax2.spines["bottom"].set_visible(True)
    
    zax2.set_xticks(xticks)
    zax2.set_xlim(zax.get_xlim())    
    f = interpolate.interp1d(atmt, year, fill_value="extrapolate")
    zax2.set_xticklabels(["%.0f years" %v for v in f(xticks)])
    zax2.xaxis.grid(False)

    for zax in ax.flatten(): zax.axhline(0, lw=1, color='dimgrey')
    
    fig.suptitle(r'Cumulated CO${\bf_2}$ uptake', x=0.06, ha='left', fontweight='bold', y=.98)
    fig.tight_layout()

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    
#

####################
# FIG2BIS
####################

if plot_fig2bis :

    print('PLOT FIG2 BIS')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig2bis.png'
    savedata2plot = dirout + 'data2plot-fig2.pckl'

    #====================
    # LOAD DATA2PLOT
    #====================

    print("> load data2plot")
    with open(savedata2plot, 'rb') as f1: data2plot = pickle.load(f1)
    print("File loaded: "+savedata2plot)

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
    #--------------------
    # COU - CTL VS TIME
    #--------------------

    vdata = data2plot['COU - CTL']
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
    zax.set_title('('+subnum.pop()+') COU simulation', loc='left')
    zax.locator_params(axis='y', nbins=6)
    zax.xaxis.set_ticks([0, 35, 70])
    #zax.xaxis.set_ticklabels(['0', '35 years', '70 years'])
    zax.set_ylabel('Change in cumul.\nCO$_2$ uptake [molC/m$^2$]')
    zax.set_xlabel('Years')
    zax.set_xlim((0, 70))
    zax.legend(lines, labels, loc='best')

    #--------------------
    # BETA : BGC - CTL VS ATMCO2 ACCUMULATION
    #--------------------

    annot = {}

    vdata = data2plot['BGC - CTL']
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

    zax.set_title('('+subnum.pop()+') Carbon-concentration feedback', loc='left')
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
    zax = ax[0, 2]
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

    zax.set_title('('+subnum.pop()+') Carbon-climate feedback', loc='left')
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
# PLOT FIG3
##################################################
if plot_fig3 :

    print('PLOT FIG3')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig3.png'
    savedata2plot = dirout + 'data2plot-fig3.pckl'
    force_computation = False

    lats = [None, 34.2987, 43, None]
    boxes={'box1':{'ymin':lats[0], 'ymax':lats[1]}, \
           'box2':{'ymin':lats[1], 'ymax':lats[2]}, \
           'box3':{'ymin':lats[2], 'ymax':lats[3]}}
    
    #====================
    # PREPARE DATA
    #====================

    if (not os.path.isfile(savedata2plot)) | force_computation :
        print('> compute data2plot')
        print(datetime.datetime.now())
        
        fdir = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
        fsuf = '_1y_01010101_01701230_ptrc_T.xml'
        start_date  = '0161-01-01'
        end_date    = '0170-12-31'   
        
        
        zfact=1
        
        #--------------------
        # LOOP ON RESOLUTION:
        # COMPUTE DIC PROFILE IN
        # CTL, BGC AND COU SIMU
        #--------------------

        tempdict = {}
        for vkR in res :
        
            tempdict['R'+vkR] = {}
            
            if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
            else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
            #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST
            tempdict['R'+vkR]['dep'] = zwmesh['depT']

            #____________________
            # CTL
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0)
            zwctl = averaging.xmean(zw1, zwmesh, dim='zyx')
            
            #____________________
            # BGC
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0) 
            zwbgc = averaging.xmean(zw1, zwmesh, dim='zyx')
            
            #____________________
            # COU
            fff = fdir + 'CC' + vkR + fsuf
            #fff = fdir + 'CC' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0) 
            zwcou = averaging.xmean(zw1, zwmesh, dim='zyx')

            #____________________
            # loop on boxes
            for kbox, vbox in boxes.items() :

                tempdict['R'+vkR][kbox] = {}

                zlat1, zlat2 = vbox['ymin'], vbox['ymax']
                
                zwctl1 = averaging.ymean(zwctl, zwmesh, grid='T', ymin=zlat1, ymax=zlat2, dim='zy')
                zwbgc1 = averaging.ymean(zwbgc, zwmesh, grid='T', ymin=zlat1, ymax=zlat2, dim='zy')
                zwcou1 = averaging.ymean(zwcou, zwmesh, grid='T', ymin=zlat1, ymax=zlat2, dim='zy')
                tempdict['R'+vkR][kbox]['CTL'] = zwctl1
                tempdict['R'+vkR][kbox]['BGC - CTL'] = zwbgc1 - zwctl1
                tempdict['R'+vkR][kbox]['COU - BGC'] = zwcou1 - zwbgc1
            #        
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        
        tempdict['AVG1'] = {'dep': tempdict['R1']['dep']}
        tempdict['STD1'] = {'dep': tempdict['R1']['dep']}
        tempdict['+STD'] = {'dep': tempdict['R1']['dep']}
        tempdict['-STD'] = {'dep': tempdict['R1']['dep']}
        for kbox in boxes.keys() :
            tempdict['AVG1'][kbox], tempdict['STD1'][kbox], tempdict['+STD'][kbox], tempdict['-STD'][kbox] = {}, {}, {}, {}
            for vkS in ['CTL', 'BGC - CTL', 'COU - BGC'] : 
                zw = []
                for vkR in r1s : zw.append(tempdict['R'+vkR][kbox][vkS])
                zw = np.array(zw)
                tempdict['AVG1'][kbox][vkS] = np.nanmean(zw, axis=0)
                tempdict['STD1'][kbox][vkS] = unc*np.nanstd(zw, axis=0)
                tempdict['+STD'][kbox][vkS] = tempdict['AVG1'][kbox][vkS] + tempdict['STD1'][kbox][vkS]
                tempdict['-STD'][kbox][vkS] = tempdict['AVG1'][kbox][vkS] - tempdict['STD1'][kbox][vkS]
            #
        #
        
        #--------------------
        # SAVE DATA2PLOT
        #--------------------

        data2plot = {}
        for ksimu in ['CTL', 'BGC - CTL', 'COU - BGC']:
            data2plot[ksimu] = {}
            for kres in ['AVG1', '+STD', '-STD', 'R9', 'R27']:
                data2plot[ksimu][kres] = {'dep':tempdict[kres]['dep']}
                for kbox in boxes.keys():
                    data2plot[ksimu][kres][kbox] = tempdict[kres][kbox][ksimu]
                #
            #
        #
                                
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
    ncol, nrow = 3, 3
    fsize = (ncol*4*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey=True) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()

    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg $\pm$ st. dev.', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}

    depmax = 2000
    
    kwlines = {'R9'  : dict(lw=1.5, color='dodgerblue', ls='-'),
               'R27' : dict(lw=1.5, color='orange'    , ls='--', dashes=[2., 1.]),
               'AVG1': dict(lw=1.5, color='black'     , ls='-'),
               'UNC' : dict(color='black', alpha=.3)}

    # data2plot[ksimu][kres][kbox]

    #--------------------
    # CTL
    #--------------------

    vdata = data2plot['CTL']
    irow = 0

    icol = 0
    for kbox in boxes.keys() :

        zax = ax[irow, icol]
        lines, labels = [], []
        ooo=5
        
        # UNCERTAINTY R1
        Xu = vdata['+STD'][kbox]
        Xd = vdata['-STD'][kbox]
        Y  = vdata['-STD']['dep']
        zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
        ooo+=5
        for zres in ['AVG1', 'R9', 'R27'] :
            X, Y = vdata[zres][kbox], vdata[zres]['dep']
            zl, = zax.plot(X, Y, **kwlines[zres])
            lines.append(zl)
            labels.append(names[zres])
            ooo+=5
        #
        zax.set_title('('+subnum.pop()+')', loc='left')
        zax.set_xlim((2000, 2200))       
        zax.locator_params(axis='x', nbins=3)

        icol+=1
    #
    leg = ax[0, 0].legend(lines, labels, loc='best')
    leg._legend_box.align = "left"

    #--------------------
    # BGC - CTL
    #--------------------

    vdata = data2plot['BGC - CTL']
    irow = 1

    icol = 0
    for kbox in boxes.keys() :

        zax = ax[irow, icol]
        ooo=5
        
        # UNCERTAINTY R1
        Xu = vdata['+STD'][kbox]
        Xd = vdata['-STD'][kbox]
        Y  = vdata['-STD']['dep']
        zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
        ooo+=5
        for zres in ['AVG1', 'R9', 'R27'] :
            X = vdata[zres][kbox]
            Y = vdata[zres]['dep']
            zl, = zax.plot(X, Y, **kwlines[zres])
            ooo+=5
        #
        zax.set_title('('+subnum.pop()+')', loc='left')
        zax.set_xlim((-20, 100))       
        zax.locator_params(axis='x', nbins=3)

        icol+=1
    #

    #--------------------
    # COU - BGC
    #--------------------

    vdata = data2plot['COU - BGC']
    irow = 2

    icol = 0
    for kbox in boxes.keys() :

        zax = ax[irow, icol]
        ooo=5
        
        # UNCERTAINTY R1
        Xu = vdata['+STD'][kbox]
        Xd = vdata['-STD'][kbox]
        Y = vdata['-STD']['dep']
        zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
        ooo+=5
        for zres in ['AVG1', 'R9', 'R27'] :
            X = vdata[zres][kbox]
            Y = vdata[zres]['dep']
            zl, = zax.plot(X, Y, **kwlines[zres])
            ooo+=5
        #
        zax.set_title('('+subnum.pop()+')', loc='left')
        zax.set_xlim((-30, 5))       
        zax.locator_params(axis='x', nbins=3)

        icol+=1
    #

    for zax in ax[:, 0].flat:
        zax.set_ylabel('Depth [m])')
        zax.set_ylim((depmax, 0))
        zax.yaxis.set_ticks([250, 750, 1250, 1750])
        #zax.locator_params(axis='y', nbins=5)
    for zax in ax[:, 1].flat:
        zax.tick_params(left=False)
    for zax in ax[:, 2].flat:
        zax.set_ylabel('Depth [m]')
        zax.yaxis.set_ticks_position('right')
        zax.yaxis.set_label_position('right')
    for zax in ax[-1, :].flat: 
        zax.set_xlabel('[mmol$\,$C$\,$m$^{-3}$]')
    # ax[1, 1].tick_params(bottom=False)
    # ax[1, 1].label_outer()
    # ax[1, 1].yaxis.set_ticks_position('right')
    # ax[1, 1].yaxis.set_label_position('right')
    
    fig.tight_layout()

    yshift = 0.1
    for zax in ax[0, :]:
        zw = zax.get_position()
        ny0 = zw.y0 + yshift*zw.height
        zax.set_position([zw.x0, ny0, zw.width, zw.height])
    #
    for zax in ax[-1, :]:
        zw = zax.get_position()
        ny0 = zw.y0 - yshift*zw.height
        zax.set_position([zw.x0, ny0, zw.width, zw.height])
    #

    for iii, vvv in enumerate(['Subtropical gyre', 'Subpolar gyre', 'Convection zone']):
        zw = ax[2, iii].get_position()
        plt.figtext(zw.x0+.5*zw.width, zw.y0-.26*zw.height, vvv, figure=fig, \
                    horizontalalignment='center', verticalalignment='top', fontweight='bold')
    #

    for iii, vvv in enumerate(['DIC concentration, CTL', 'DIC conc. change, '+r'$\bf{BGC - CTL}$', 'DIC conc. change, '+r'$\bf{COU - BGC}$']):
        zw = ax[iii, 0].get_position()
        plt.figtext(zw.x0-.5*zw.width, zw.y1+.12*zw.height, vvv, figure=fig, fontweight='bold')
    #

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    

##################################################
# END FIG3
##################################################

##################################################
# PLOT FIG3_BIS
##################################################
if plot_fig3_bis :

    print('PLOT FIG3_BIS')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig3bis.png'
    savedata2plot = dirout + 'data2plot-fig3bis.pckl'
    force_computation = False
    
    #====================
    # PREPARE DATA
    #====================

    if (not os.path.isfile(savedata2plot)) | force_computation :
        print('> compute data2plot')
        print(datetime.datetime.now())
        
        fdir = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
        fsuf = '_1y_01010101_01701230_ptrc_T.xml'
        start_date  = '0161-01-01'
        end_date    = '0170-12-31'   
        
        
        zfact=1
        
        #--------------------
        # LOOP ON RESOLUTION:
        # COMPUTE DIC PROFILE IN
        # CTL, BGC AND COU SIMU
        #--------------------

        tempdict = {}
        for vkR in res :
        
            tempdict['R'+vkR] = {}
            
            if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
            else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
            #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST
            tempdict['R'+vkR]['dep'] = zwmesh['depT']

            #____________________
            # CTL
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0)
            zw1 = averaging.xmean(zw1, zwmesh, dim='zyx')
            zwctl = averaging.ymean(zw1, zwmesh, grid='T', dim='zy')
            
            #____________________
            # BGC
            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0) 
            zw1 = averaging.xmean(zw1, zwmesh, dim='zyx')
            zwbgc = averaging.ymean(zw1, zwmesh, grid='T', dim='zy')
            
            #____________________
            # COU
            fff = fdir + 'CC' + vkR + fsuf
            #fff = fdir + 'CC' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC2', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0) 
            zw1 = averaging.xmean(zw1, zwmesh, dim='zyx')
            zwcou = averaging.ymean(zw1, zwmesh, grid='T', dim='zy')

            tempdict['R'+vkR]['CTL'] = zwctl
            tempdict['R'+vkR]['BGC - CTL'] = zwbgc - zwctl
            tempdict['R'+vkR]['COU - BGC'] = zwcou - zwbgc
            
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        
        tempdict['AVG1'] = {'dep': tempdict['R1']['dep']}
        tempdict['STD1'] = {'dep': tempdict['R1']['dep']}
        tempdict['+STD'] = {'dep': tempdict['R1']['dep']}
        tempdict['-STD'] = {'dep': tempdict['R1']['dep']}
        for vkS in ['CTL', 'BGC - CTL', 'COU - BGC'] : 
            zw = []
            for vkR in r1s : zw.append(tempdict['R'+vkR][vkS])
            zw = np.array(zw)
            tempdict['AVG1'][vkS] = np.nanmean(zw, axis=0)
            tempdict['STD1'][vkS] = unc*np.nanstd(zw, axis=0)
            tempdict['+STD'][vkS] = tempdict['AVG1'][vkS] + tempdict['STD1'][vkS]
            tempdict['-STD'][vkS] = tempdict['AVG1'][vkS] - tempdict['STD1'][vkS]
        #
        
        #--------------------
        # SAVE DATA2PLOT
        #--------------------

        data2plot = {}
        for ksimu in ['CTL', 'BGC - CTL', 'COU - BGC']:
            data2plot[ksimu] = {}
            for kres in ['AVG1', '+STD', '-STD', 'R9', 'R27']:
                data2plot[ksimu][kres] = {'X': tempdict[kres][ksimu],
                                          'Y':tempdict[kres]['dep']}
                #
            #
        #
                                
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
    fsize = (ncol*4*infact, nrow*5*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey=True) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()

    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg $\pm$ std', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}

    depmax = 2000
    
    kwlines = {'R9'  : dict(lw=1.5, color='dodgerblue', ls='-'),
               'R27' : dict(lw=1.5, color='orange'    , ls='--', dashes=[2., 1.]),
               'AVG1': dict(lw=1.5, color='black'     , ls='-'),
               'UNC' : dict(color='black', alpha=.3)}

    irow = 0

    #--------------------
    # CTL
    #--------------------

    vdata = data2plot['CTL']

    icol = 0
    zax = ax[irow, icol]
    lines, labels = [], []
    ooo=5

    # UNCERTAINTY R1
    Xu = vdata['+STD']['X']
    Xd = vdata['-STD']['X']
    Y  = vdata['-STD']['Y']
    zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
    ooo+=5
    for zres in ['AVG1', 'R9', 'R27'] :
        X, Y = vdata[zres]['X'], vdata[zres]['Y']
        zl, = zax.plot(X, Y, **kwlines[zres])
        lines.append(zl)
        labels.append(names[zres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') DIC in CTL', loc='left')
    zax.set_xlim((2000, 2200))       
    zax.locator_params(axis='x', nbins=3)

    leg = ax[0, 0].legend(lines, labels, loc='best')
    leg._legend_box.align = "left"

    #--------------------
    # BGC - CTL
    #--------------------

    vdata = data2plot['BGC - CTL']

    icol = 1
    zax = ax[irow, icol]
    ooo=5
    
    # UNCERTAINTY R1
    Xu = vdata['+STD']['X']
    Xd = vdata['-STD']['X']
    Y  = vdata['-STD']['Y']
    zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
    ooo+=5
    for zres in ['AVG1', 'R9', 'R27'] :
        X = vdata[zres]['X']
        Y = vdata[zres]['Y']
        zl, = zax.plot(X, Y, **kwlines[zres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') DIC change, '+r'BGC – CTL', loc='left')
    zax.set_xlim((-20, 100))       
    zax.locator_params(axis='x', nbins=3)

    #--------------------
    # COU - BGC
    #--------------------

    vdata = data2plot['COU - BGC']

    icol = 2
    zax = ax[irow, icol]
    ooo=5
    
    # UNCERTAINTY R1
    Xu = vdata['+STD']['X']
    Xd = vdata['-STD']['X']
    Y  = vdata['-STD']['Y']
    zax.fill_betweenx(Y, Xu, x2=Xd, **kwlines['UNC'])
    ooo+=5
    for zres in ['AVG1', 'R9', 'R27'] :
        X = vdata[zres]['X']
        Y = vdata[zres]['Y']
        zl, = zax.plot(X, Y, **kwlines[zres])
        ooo+=5
    #
    zax.set_title('('+subnum.pop()+') DIC change, '+r'COU – BGC', loc='left')
    zax.set_xlim((-30, 5))       
    zax.locator_params(axis='x', nbins=3)

    
    zax = ax[0, 0]
    zax.set_ylabel('Depth [m])')
    zax.set_ylim((depmax, 0))
    zax.yaxis.set_ticks([250, 750, 1250, 1750])
    ax[0, 1].tick_params(left=False)
    zax =ax[0, 2]
    zax.set_ylabel('Depth [m]')
    zax.yaxis.set_ticks_position('right')
    zax.yaxis.set_label_position('right')
    for zax in ax[0, :].flat: 
        zax.set_xlabel('[mmol$\,$C$\,$m$^{-3}$]')
    # ax[1, 1].tick_params(bottom=False)
    # ax[1, 1].label_outer()
    # ax[1, 1].yaxis.set_ticks_position('right')
    # ax[1, 1].yaxis.set_label_position('right')
    
    fig.tight_layout()

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    

##################################################
# END FIG3_BIS
##################################################

##################################################
# PLOT FIG3_TER
##################################################
if plot_fig3_ter :

    print('PLOT FIG3_TER')
    print(datetime.datetime.now())

    savefig       = dirout + 'fig3ter.png'
    savedata2plot = dirout + 'data2plot-fig3ter.pckl'
    force_computation = False
    
    #====================
    # PREPARE DATA
    #====================

    if (not os.path.isfile(savedata2plot)) | force_computation :
        print('> compute data2plot')
        print(datetime.datetime.now())
        
        fdir = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
        fsuf = '_1y_01010101_01701230_ptrc_T.xml'
        start_date  = '0161-01-01'
        end_date    = '0170-12-31'   
        
        
        zfact=1
        
        #--------------------
        # LOOP ON RESOLUTION:
        # COMPUTE DIC PROFILE IN
        # CTL, BGC AND COU SIMU
        #--------------------

        tempdict = {}
        for vkR in res :
        
            tempdict['R'+vkR] = {}
            
            if vkR in ['9', '27'] : zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+vkR+'.nc')
            else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )
            #zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' ) # TEST
            tempdict['R'+vkR]['dep'] = zwmesh['depT']
            tempdict['R'+vkR]['lat'] = zwmesh['latT'][:, 0]

            fff = fdir + 'CTL' + vkR + fsuf
            #fff = fdir + 'CTL' + '1' + fsuf # TEST
            zw1  = reading.read_ncdf('DIC', fff, time=(start_date, end_date))
            zw1 = zw1['data']*zfact
            zw1 = np.mean(zw1, axis=0)
            zwctl = averaging.xmean(zw1, zwmesh, dim='zyx')
            
            tempdict['R'+vkR]['CTL'] = zwctl
            
        #
        
        #--------------------
        # AVG1, STD1
        #--------------------
        
        tempdict['AVG1'] = {'dep': tempdict['R1']['dep'],
                            'lat': tempdict['R1']['lat']}
        zw = []
        for vkR in r1s : zw.append(tempdict['R'+vkR]['CTL'])
        zw = np.array(zw)
        tempdict['AVG1']['CTL'] = np.nanmean(zw, axis=0)
       
        #--------------------
        # SAVE DATA2PLOT
        #--------------------

        data2plot = {}
        for kres in ['AVG1', 'R9', 'R27']:
            data2plot[kres] = {'X': tempdict[kres]['lat'],
                                'Y': tempdict[kres]['dep'],
                                'Z': tempdict[kres]['CTL']}
        #
                                
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
    fsize = (ncol*5*infact, nrow*4*infact) #(width, height)
    fig, ax   = plt.subplots(nrow, ncol, figsize=fsize, squeeze=False, sharey=True) # nrow, ncol
    subnum=list('abcdefghijklmnopqrstuvwxyz')
    subnum.reverse()

    names={'R9':'1/9°', 'R27':'1/27°', \
           'AVG1':'1° avg', '+STD':'1°, avg+std', '-STD':'1°, avg-std'}

    irow = 0
    depmax = 1500
    # aa = np.linspace(-zlevmax, 0, 12)
    # bb = np.linspace(0, zlevmax, 12)
    # lev = np.concatenate((aa[:-1], bb))
    lev = np.linspace(2050, 2150, 11)
    tickbar = lev[::2]
    
    kwmapstick = dict(yticks=[20, 34.2987, 43, 48.5974], yticklabels=['0', '1,590', '2,560', '3,180'])

    icol = 0
    for zres in ['AVG1', 'R9', 'R27'] :
                    
        zax = ax[irow, icol]
        X, Y, Z = data2plot[zres]['X'], data2plot[zres]['Y'], data2plot[zres]['Z']
        zcf = zax.contourf(X, Y, Z, levels=lev ,
                           cmap='viridis', extend='both')
        plotting.make_YZ_axis(zax, title='('+subnum.pop()+') '+names[zres],
                              **kwmapstick)
        # zax.axhline(34.2987, lw=1., ls='-', c='k')
        # zax.axhline(43., lw=1., ls='-', c='k')
        icol+=1
    #

    
    zax = ax[0, 0]
    zax.set_ylabel('Depth [m])')
    zax.set_ylim((depmax, 0))
    zax.yaxis.set_ticks([250, 750, 1250])
    ax[0, 1].tick_params(left=False)
    # zax =ax[0, 2]
    # zax.set_ylabel('Depth [m]')
    # zax.yaxis.set_ticks_position('right')
    # zax.yaxis.set_label_position('right')
    
    fig.tight_layout()

    cbtitle = '[mmol$\,$C$\,$m$^{-3}$]'
    zw1 = ax[0, -1].get_position()
    nx0 = zw1.x1 + 0.1*zw1.width
    ny0 = zw1.y0
    nh  = zw1.height
    nw  = 0.03*nh
    cbar_ax = fig.add_axes([nx0, ny0, nw, nh])
    fig.colorbar(zcf, cax=cbar_ax, orientation='vertical', ticklocation='right',
                 label=cbtitle, ticks=tickbar)
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

    for zax in fig.axes : zax.tick_params(color='dimgrey')
    fig.savefig(savefig, bbox_inches='tight')
    print("Figure saved: ", savefig)
    print(datetime.datetime.now())
    plt.close()    

##################################################
# END FIG3_TER
##################################################


##################################################
# SAVEVALUES FIG4
##################################################
if  savevalues_fig4 :


    print('SAVEVALUES FIG4')
    print(datetime.datetime.now())

    savefile = dirout + 'savevalues_fig4_correction_cou27.txt'
    
    fdirtr   = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
    ftrbase = '_1y_01010101_01701230_ptrc_T.xml'
    sdatetr, edatetr = '0166-01-01', '0170-12-31'
    carbtr = 'anthC'

    fdirpckl = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET/'

    bcorrection_cou27 = True
    file_correction_cou27 = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET-CC27-DICTRD-CORRECTION-MISSING-VALUES/'+\
        'main_carbon_box_budget_cc27_dictrd_correction_missing_values_r27_cou_1ymean_y101-170_6boxes.pckl'
    
    bcorrection_rad27 = True
    file_correction_rad27 = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET-CC27-DICTRD-CORRECTION-MISSING-VALUES/'+\
        'main_carbon_box_budget_cc27_dictrd_correction_missing_values_r27_rad_1ymean_y101-170_6boxes.pckl'

    prepckl = 'main_carbon_box_budget_r'
    sufpckl = '_1ymean_y101-170_6boxes.pckl'
    
    res_list  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000', '9', '27']
    simu_list = ['CTL', 'BGC', 'RAD', 'COU']

    ####################
    # PREPARE DATA
    ####################

    print('> compute data2print')
    print(datetime.datetime.now())
        
    tempdict={}
    for kres in res_list :
    #for kres in ['27'] :
    
        tempdict['R'+kres] = {}

        #--------------------
        # READ MESH
        #--------------------

        if kres in ['9', '27'] :
            zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+kres+'.nc')
        else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )

        for ksimu in simu_list :

            tempdict['R'+kres][ksimu] = {}
                
            #--------------------
            # READ PICKLE
            #--------------------

            fff = fdirpckl + prepckl + kres.lower() + '_' + ksimu.lower() + sufpckl
            zf = open(fff, 'rb')
            zw = pickle.load(zf)
            zf.close()
            print("File loaded: "+fff)
            zboxes = zw['boxes']
            boxes_list = zboxes.keys()

            if (ksimu=='COU') & (kres=='27') & (bcorrection_cou27):
                print('!!!! CORRECTION COU27')
                zf = open(file_correction_cou27, 'rb')
                zw_correction_cou27 = pickle.load(zf)
                zf.close()
                print("File loaded: "+file_correction_cou27)
                zboxes_correction_cou27 = zw_correction_cou27['boxes']
            #
            
            if (ksimu=='RAD') & (kres=='27') & (bcorrection_rad27):
                print('!!!! CORRECTION RAD27')
                zf = open(file_correction_rad27, 'rb')
                zw_correction_rad27 = pickle.load(zf)
                zf.close()
                print("File loaded: "+file_correction_rad27)
                zboxes_correction_rad27 = zw_correction_rad27['boxes']
            #

            #--------------------
            # PROCESS DATA IN EACH BOXES
            #--------------------

            for kbox, vbox in zboxes.items():
                tempdict['R'+kres][ksimu][kbox] = {}

                zfact_tr  = 1e-15                    # mmol/m3 -> Tmol/m3
                zfact_trd = 1e-15 * 3600*24*360 * 70 # mmol/s  -> Tmol/70y

                #____________________
                # TREND
                
                tempdict['R'+kres][ksimu][kbox]['SFX']    = vbox['sfx']   *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ADVONL'] = vbox['advonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ADVOFF'] = vbox['advoff']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['UT1']    = vbox['uTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['UT2']    = vbox['uTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['VT1']    = vbox['vTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['VT2']    = vbox['vTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['WT1']    = vbox['wTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['WT2']    = vbox['wTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ZDF']    = vbox['zdfonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['LDF']    = vbox['ldfonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['FLX']    = vbox['asflux']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['SMS']    = vbox['smsonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['FOR']    = vbox['foronl']*zfact_trd
                if (ksimu=='COU') & (kres=='27') & (bcorrection_cou27):
                    print('!!!! CORRECTION COU27')
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] = zboxes_correction_cou27[kbox]['advonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    = zboxes_correction_cou27[kbox]['zdfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['LDF']    = zboxes_correction_cou27[kbox]['ldfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FLX']    = zboxes_correction_cou27[kbox]['asflux']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['SMS']    = zboxes_correction_cou27[kbox]['smsonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FOR']    = zboxes_correction_cou27[kbox]['foronl']*zfact_trd
                if (ksimu=='RAD') & (kres=='27') & (bcorrection_rad27):
                    print('!!!! CORRECTION RAD27')
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] = zboxes_correction_rad27[kbox]['advonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    = zboxes_correction_rad27[kbox]['zdfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['LDF']    = zboxes_correction_rad27[kbox]['ldfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FLX']    = zboxes_correction_rad27[kbox]['asflux']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['SMS']    = zboxes_correction_rad27[kbox]['smsonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FOR']    = zboxes_correction_rad27[kbox]['foronl']*zfact_trd
                #
                if 'sfxgm' in vbox.keys():
                    tempdict['R'+kres][ksimu][kbox]['SFX'] = \
                        tempdict['R'+kres][ksimu][kbox]['SFX'] + \
                                  vbox['sfxgm']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ADVOFF'] = \
                        tempdict['R'+kres][ksimu][kbox]['ADVOFF'] +\
                                  vbox['advgmoff']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['VT1'] = \
                        tempdict['R'+kres][ksimu][kbox]['VT1'] +\
                                  vbox['vgmTr1']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['VT2'] = \
                        tempdict['R'+kres][ksimu][kbox]['VT2'] +\
                                  vbox['vgmTr2']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['WT1'] = \
                        tempdict['R'+kres][ksimu][kbox]['WT1'] +\
                                  vbox['wgmTr1']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['WT2'] = \
                        tempdict['R'+kres][ksimu][kbox]['WT2'] +\
                                  vbox['wgmTr2']*zfact_trd
                #
                tempdict['R'+kres][ksimu][kbox]['TOTONL']     = \
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] + \
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    + \
                    tempdict['R'+kres][ksimu][kbox]['LDF']    + \
                    tempdict['R'+kres][ksimu][kbox]['FOR']    + \
                    tempdict['R'+kres][ksimu][kbox]['FLX']    + \
                    tempdict['R'+kres][ksimu][kbox]['SMS']
                tempdict['R'+kres][ksimu][kbox]['TOTSFX']  = \
                    tempdict['R'+kres][ksimu][kbox]['SFX'] + \
                    tempdict['R'+kres][ksimu][kbox]['ZDF'] + \
                    tempdict['R'+kres][ksimu][kbox]['LDF'] + \
                    tempdict['R'+kres][ksimu][kbox]['FOR'] + \
                    tempdict['R'+kres][ksimu][kbox]['FLX'] + \
                    tempdict['R'+kres][ksimu][kbox]['SMS']
                
                tempdict['R'+kres][ksimu][kbox]['Sum. air-sea'] = \
                    tempdict['R'+kres][ksimu][kbox]['FLX']
                tempdict['R'+kres][ksimu][kbox]['Sum. diffusion'] = \
                    tempdict['R'+kres][ksimu][kbox]['ZDF']        + \
                    tempdict['R'+kres][ksimu][kbox]['LDF']         
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. southern'] = \
                    tempdict['R'+kres][ksimu][kbox]['VT1']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. northern'] = \
                    tempdict['R'+kres][ksimu][kbox]['VT2']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. top'] = \
                    tempdict['R'+kres][ksimu][kbox]['WT1']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. bottom'] = \
                    tempdict['R'+kres][ksimu][kbox]['WT2']        
                tempdict['R'+kres][ksimu][kbox]['Sum. biology'] = \
                    tempdict['R'+kres][ksimu][kbox]['SMS']        
                # The effect of concentration/dillution from evaporation and
                # precipitation (term FOR) is ignored in the summary as it is
                # the same for all resolution
    
                #____________________
                # CARBON

                # read file
                if ksimu == 'CTL' :
                    fff = fdirtr+'CTL'+kres+ftrbase
                    zw = reading.read_ncdf('DIC', fff, time = (sdatetr, edatetr))
                elif ksimu == 'BGC':
                    fff = fdirtr+'CTL'+kres+ftrbase
                    zw = reading.read_ncdf('DIC2', fff, time = (sdatetr, edatetr))
                elif ksimu == 'RAD':
                    fff = fdirtr+'CC'+kres+ftrbase
                    zw = reading.read_ncdf('DIC', fff, time = (sdatetr, edatetr))
                elif ksimu == 'COU':
                    fff = fdirtr+'CC'+kres+ftrbase
                    zw = reading.read_ncdf('DIC2', fff, time = (sdatetr, edatetr))
                #
                zw = zw['data']
                # temporal mean
                zw = averaging.tmean(zw, zwmesh, dim='tzyx', grid='T')
                # spatial integral
                zw = averaging.zmean(zw, zwmesh, dim='zyx', grid='T', zmin=vbox['zmin'], zmax=vbox['zmax'], integral = True)
                zw = averaging.ymean(zw, zwmesh, dim='yx' , grid='T', ymin=vbox['ymin'], ymax=vbox['ymax'], integral = True)
                zw = averaging.xmean(zw, zwmesh, dim='x'  , grid='T', xmin=vbox['xmin'], xmax=vbox['xmax'], integral = True)
                tempdict['R'+kres][ksimu][kbox]['TRA'] = zw*zfact_tr
    
        #
    #

    #--------------------
    # COMPUTE BGC - CTL, COU - BGC, RAD - CTL and COU-BGC-(RAD-CTL)
    #--------------------

    for kres in res_list:
        tempdict['R'+kres]['BGC - CTL'] = {}
        tempdict['R'+kres]['COU - BGC'] = {}
        tempdict['R'+kres]['RAD - CTL'] = {}
        tempdict['R'+kres]['COU - BGC - (RAD - CTL)'] = {}
        for kboxes in boxes_list:
            tempdict['R'+kres]['BGC - CTL'][kboxes] = {}
            tempdict['R'+kres]['COU - BGC'][kboxes] = {}
            tempdict['R'+kres]['RAD - CTL'][kboxes] = {}
            tempdict['R'+kres]['COU - BGC - (RAD - CTL)'][kboxes] = {}
            for kkk, vvv in tempdict['R'+kres]['CTL'][kboxes].items() :
                tempdict['R'+kres]['BGC - CTL'][kboxes][kkk] =\
                    tempdict['R'+kres]['BGC'][kboxes][kkk] - vvv
                tempdict['R'+kres]['COU - BGC'][kboxes][kkk] =\
                    tempdict['R'+kres]['COU'][kboxes][kkk] -\
                    tempdict['R'+kres]['BGC'][kboxes][kkk]
                tempdict['R'+kres]['RAD - CTL'][kboxes][kkk] =\
                    tempdict['R'+kres]['RAD'][kboxes][kkk] - vvv
                tempdict['R'+kres]['COU - BGC - (RAD - CTL)'][kboxes][kkk] =\
                    tempdict['R'+kres]['COU'][kboxes][kkk] -\
                    tempdict['R'+kres]['BGC'][kboxes][kkk] -\
                    (tempdict['R'+kres]['RAD'][kboxes][kkk] - vvv)
            #
        #
    #
                    
    #--------------------
    # AVG1, STD1
    #--------------------
    
    tempdict['AVG1'] = {}
    tempdict['STD1'] = {}
    for ksimu in ['CTL', 'BGC', 'RAD', 'COU', 'BGC - CTL',\
                  'COU - BGC', 'RAD - CTL', 'COU - BGC - (RAD - CTL)']:
        tempdict['AVG1'][ksimu] = {}
        tempdict['STD1'][ksimu] = {}
        for kboxes in boxes_list :
            tempdict['AVG1'][ksimu][kboxes] = {}
            tempdict['STD1'][ksimu][kboxes] = {}
            for kkk in tempdict['R'+res_list[0]][ksimu][kboxes].keys():
                
                zw = []
                for kr1s in r1s : zw.append(tempdict['R'+kr1s][ksimu][kboxes][kkk])
                zw = np.array(zw)
                tempdict['AVG1'][ksimu][kboxes][kkk] =  np.nanmean(zw, axis=0)
                tempdict['STD1'][ksimu][kboxes][kkk] =  unc*np.nanstd(zw, axis=0)
            #
        #
    #
    
    #--------------------
    # SAVE DATA2PRINT
    #--------------------
    
    data2print = {}
    for ksimu in ['CTL', 'BGC - CTL', 'COU - BGC', 'RAD - CTL', \
                  'COU - BGC - (RAD - CTL)', 'BGC', 'RAD', 'COU']:
        data2print[ksimu] = {}
        for kres in ['AVG1', 'STD1', 'R9', 'R27']:
            data2print[ksimu][kres] = {}
            for kboxes in boxes_list: 
                data2print[ksimu][kres][kboxes] = {}
                for kkk, vvv in tempdict[kres][ksimu][kboxes].items():
                    data2print[ksimu][kres][kboxes][kkk] = np.around(vvv, decimals=2)
                #
            #
        #
    #

    #####################
    # SAVE IN TEXT FILE
    #####################
    # data2print[ksimu][kres][kboxes][kkk] = vvv

    print('> save data2print in text file')
    print(datetime.datetime.now())

    fff = open(savefile, "w+")

    fff.write(str(datetime.datetime.now()))
    fff.write("\n")
    fff.write("\n")

    for ksimu, vsimu in data2print.items() :

        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n############")
        fff.write("\n############       " + str(ksimu))
        fff.write("\n############")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n")
        fff.write("\n")
        fff.write("\n")

        for kres, vres in vsimu.items() :

            fff.write("\n==================================================")
            fff.write("\n==================================================")
            fff.write("\n               " + str(kres) + ", "+str(ksimu)     )
            fff.write("\n==================================================")
            fff.write("\n==================================================")
            fff.write("\n")
            fff.write("\n")
            
            for kboxes, vboxes  in vres.items() :

                fff.write("\n--------------------------------------------------")
                fff.write("\n               " + str(kboxes) + ', '+ str(ksimu) + ", "+str(kres))
                fff.write("\n--------------------------------------------------")
                fff.write("\n")
                fff.write("\n")

                # split dict for printing for first the 'Sum...'
                kkk1, kkk2 = [], []
                for kkk in vboxes.keys():
                    if kkk[:3] == 'Sum': kkk1.append(kkk)
                    else: kkk2.append(kkk)
                #
                for kvar in kkk1:
                    fff.write(kvar+":  "+str(vboxes[kvar])+" Tmol-C")
                    fff.write("   ---   ")
                #
                fff.write("\n")
                fff.write("\n")
                for kvar in kkk2:
                    fff.write(kvar+":  "+str(vboxes[kvar])+" Tmol-C")
                    fff.write("   ---   ")
                #
                fff.write("\n")
            #
            fff.write("\n")
            fff.write("\n")
        #
        fff.write("\n")
        fff.write("\n")
        fff.write("\n")
    #
    fff.close()
    print("File saved: ", savefile)


#
##################################################
# END SAVEVALUES FIG4
##################################################

##################################################
# SAVEVALUES FIG4 1D
##################################################
if  savevalues_fig4_1d :


    print('SAVEVALUES FIG4 1D')
    print(datetime.datetime.now())

    savefile = dirout + 'savevalues_fig4_1d_correction_cou27.txt'
    
    fdirtr   = '/gpfswork/rech/eee/rdyk004/GYRE_XML/'
    ftrbase = '_1y_01010101_01701230_ptrc_T.xml'
    sdatetr, edatetr = '0166-01-01', '0170-12-31'
    carbtr = 'anthC'

    fdirpckl = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET/'

    bcorrection_cou27 = True
    file_correction_cou27 = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET-CC27-DICTRD-CORRECTION-MISSING-VALUES/'+\
        'main_carbon_box_budget_cc27_dictrd_correction_missing_values_r27_cou_1ymean_y101-170_6boxes.pckl'

    bcorrection_rad27 = True
    file_correction_rad27 = '/gpfswork/rech/eee/rdyk004/MY-PYTHON3/MAIN-CARBON-BOX-BUDGET-CC27-DICTRD-CORRECTION-MISSING-VALUES/'+\
        'main_carbon_box_budget_cc27_dictrd_correction_missing_values_r27_rad_1ymean_y101-170_6boxes.pckl'

    prepckl = 'main_carbon_box_budget_r'
    sufpckl = '_1ymean_y101-170_6boxes.pckl'
    
    res_list  = ['1', '1_KL2000', '1_KL500', '1_KGM500_KL500', '1_KGM2000_KL2000', '9', '27']
    simu_list = ['CTL', 'BGC', 'RAD', 'COU']
    boxes_1d_list = {'upper': ['box1', 'box2', 'box3'],
                     'lower': ['box4', 'box5', 'box6']}

    ####################
    # PREPARE DATA
    ####################

    print('> compute data2print')
    print(datetime.datetime.now())
        
    tempdict={}
    for kres in res_list :
    
        tempdict['R'+kres] = {}

        #--------------------
        # READ MESH
        #--------------------

        if kres in ['9', '27'] :
            zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R'+kres+'.nc')
        else :  zwmesh = reading.read_mesh('/gpfswork/rech/eee/rdyk004/MESH/mesh_mask_R1.nc' )

        for ksimu in simu_list :

            tempdict['R'+kres][ksimu] = {}
                
            #--------------------
            # READ PICKLE
            #--------------------

            fff = fdirpckl + prepckl + kres.lower() + '_' + ksimu.lower() + sufpckl
            zf = open(fff, 'rb')
            zw = pickle.load(zf)
            zf.close()
            print("File loaded: "+fff)
            zboxes = zw['boxes']
            boxes_list = zboxes.keys()

            if (ksimu=='COU') & (kres=='27') & (bcorrection_cou27):
                print('!!!! CORRECTION COU27')
                zf = open(file_correction_cou27, 'rb')
                zw_correction_cou27 = pickle.load(zf)
                zf.close()
                print("File loaded: "+file_correction_cou27)
                zboxes_correction_cou27 = zw_correction_cou27['boxes']
            #
            
            if (ksimu=='RAD') & (kres=='27') & (bcorrection_rad27):
                print('!!!! CORRECTION RAD27')
                zf = open(file_correction_rad27, 'rb')
                zw_correction_rad27 = pickle.load(zf)
                zf.close()
                print("File loaded: "+file_correction_rad27)
                zboxes_correction_rad27 = zw_correction_rad27['boxes']
            #

            #--------------------
            # PROCESS DATA IN EACH BOXES
            #--------------------

            for kbox, vbox in zboxes.items():
                tempdict['R'+kres][ksimu][kbox] = {}

                zfact_tr  = 1e-15                    # mmol/m3 -> Tmol/m3
                zfact_trd = 1e-15 * 3600*24*360 * 70 # mmol/s  -> Tmol/70y

                #____________________
                # TREND
                
                tempdict['R'+kres][ksimu][kbox]['SFX']    = vbox['sfx']   *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ADVONL'] = vbox['advonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ADVOFF'] = vbox['advoff']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['UT1']    = vbox['uTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['UT2']    = vbox['uTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['VT1']    = vbox['vTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['VT2']    = vbox['vTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['WT1']    = vbox['wTr1']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['WT2']    = vbox['wTr2']  *zfact_trd
                tempdict['R'+kres][ksimu][kbox]['ZDF']    = vbox['zdfonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['LDF']    = vbox['ldfonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['FLX']    = vbox['asflux']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['SMS']    = vbox['smsonl']*zfact_trd
                tempdict['R'+kres][ksimu][kbox]['FOR']    = vbox['foronl']*zfact_trd
                if (ksimu=='COU') & (kres=='27') & (bcorrection_cou27):
                    print('!!!! CORRECTION COU27')
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] = zboxes_correction_cou27[kbox]['advonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    = zboxes_correction_cou27[kbox]['zdfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['LDF']    = zboxes_correction_cou27[kbox]['ldfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FLX']    = zboxes_correction_cou27[kbox]['asflux']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['SMS']    = zboxes_correction_cou27[kbox]['smsonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FOR']    = zboxes_correction_cou27[kbox]['foronl']*zfact_trd
                if (ksimu=='RAD') & (kres=='27') & (bcorrection_rad27):
                    print('!!!! CORRECTION RAD27')
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] = zboxes_correction_rad27[kbox]['advonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    = zboxes_correction_rad27[kbox]['zdfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['LDF']    = zboxes_correction_rad27[kbox]['ldfonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FLX']    = zboxes_correction_rad27[kbox]['asflux']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['SMS']    = zboxes_correction_rad27[kbox]['smsonl']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['FOR']    = zboxes_correction_rad27[kbox]['foronl']*zfact_trd
                #
                if 'sfxgm' in vbox.keys():
                    tempdict['R'+kres][ksimu][kbox]['SFX'] = \
                        tempdict['R'+kres][ksimu][kbox]['SFX'] + \
                                  vbox['sfxgm']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['ADVOFF'] = \
                        tempdict['R'+kres][ksimu][kbox]['ADVOFF'] +\
                                  vbox['advgmoff']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['VT1'] = \
                        tempdict['R'+kres][ksimu][kbox]['VT1'] +\
                                  vbox['vgmTr1']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['VT2'] = \
                        tempdict['R'+kres][ksimu][kbox]['VT2'] +\
                                  vbox['vgmTr2']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['WT1'] = \
                        tempdict['R'+kres][ksimu][kbox]['WT1'] +\
                                  vbox['wgmTr1']*zfact_trd
                    tempdict['R'+kres][ksimu][kbox]['WT2'] = \
                        tempdict['R'+kres][ksimu][kbox]['WT2'] +\
                                  vbox['wgmTr2']*zfact_trd
                #
                tempdict['R'+kres][ksimu][kbox]['TOTONL']     = \
                    tempdict['R'+kres][ksimu][kbox]['ADVONL'] + \
                    tempdict['R'+kres][ksimu][kbox]['ZDF']    + \
                    tempdict['R'+kres][ksimu][kbox]['LDF']    + \
                    tempdict['R'+kres][ksimu][kbox]['FOR']    + \
                    tempdict['R'+kres][ksimu][kbox]['FLX']    + \
                    tempdict['R'+kres][ksimu][kbox]['SMS']
                tempdict['R'+kres][ksimu][kbox]['TOTSFX']  = \
                    tempdict['R'+kres][ksimu][kbox]['SFX'] + \
                    tempdict['R'+kres][ksimu][kbox]['ZDF'] + \
                    tempdict['R'+kres][ksimu][kbox]['LDF'] + \
                    tempdict['R'+kres][ksimu][kbox]['FOR'] + \
                    tempdict['R'+kres][ksimu][kbox]['FLX'] + \
                    tempdict['R'+kres][ksimu][kbox]['SMS']
                
                tempdict['R'+kres][ksimu][kbox]['Sum. air-sea'] = \
                    tempdict['R'+kres][ksimu][kbox]['FLX']
                tempdict['R'+kres][ksimu][kbox]['Sum. diffusion'] = \
                    tempdict['R'+kres][ksimu][kbox]['ZDF']        + \
                    tempdict['R'+kres][ksimu][kbox]['LDF']         
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. southern'] = \
                    tempdict['R'+kres][ksimu][kbox]['VT1']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. northern'] = \
                    tempdict['R'+kres][ksimu][kbox]['VT2']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. top'] = \
                    tempdict['R'+kres][ksimu][kbox]['WT1']        
                tempdict['R'+kres][ksimu][kbox]['Sum. adv. bottom'] = \
                    tempdict['R'+kres][ksimu][kbox]['WT2']        
                tempdict['R'+kres][ksimu][kbox]['Sum. biology'] = \
                    tempdict['R'+kres][ksimu][kbox]['SMS']        
                # The effect of concentration/dillution from evaporation and
                # precipitation (term FOR) is ignored in the summary as it is
                # the same for all resolution
    
                #____________________
                # CARBON

                # read file
                if ksimu == 'CTL' :
                    fff = fdirtr+'CTL'+kres+ftrbase
                    zw = reading.read_ncdf('DIC', fff, time = (sdatetr, edatetr))
                elif ksimu == 'BGC':
                    fff = fdirtr+'CTL'+kres+ftrbase
                    zw = reading.read_ncdf('DIC2', fff, time = (sdatetr, edatetr))
                elif ksimu == 'RAD':
                    fff = fdirtr+'CC'+kres+ftrbase
                    zw = reading.read_ncdf('DIC', fff, time = (sdatetr, edatetr))
                elif ksimu == 'COU':
                    fff = fdirtr+'CC'+kres+ftrbase
                    zw = reading.read_ncdf('DIC2', fff, time = (sdatetr, edatetr))
                #
                zw = zw['data']
                # temporal mean
                zw = averaging.tmean(zw, zwmesh, dim='tzyx', grid='T')
                # spatial integral
                zw = averaging.zmean(zw, zwmesh, dim='zyx', grid='T', zmin=vbox['zmin'], zmax=vbox['zmax'], integral = True)
                zw = averaging.ymean(zw, zwmesh, dim='yx' , grid='T', ymin=vbox['ymin'], ymax=vbox['ymax'], integral = True)
                zw = averaging.xmean(zw, zwmesh, dim='x'  , grid='T', xmin=vbox['xmin'], xmax=vbox['xmax'], integral = True)
                tempdict['R'+kres][ksimu][kbox]['TRA'] = zw*zfact_tr
    
        #
    #

    #--------------------
    # COMPUTE UPPER BOX AND LOWER BOX
    #--------------------

    tempdict_1d = {}
    for kres in res_list:        
        tempdict_1d['R'+kres] = {}
        for ksimu in simu_list:            
            tempdict_1d['R'+kres][ksimu] = {}            
            for kbox, vbox in boxes_1d_list.items():                
                tempdict_1d['R'+kres][ksimu][kbox] = {}
                for kdiag in tempdict['R'+kres][ksimu][vbox[0]].keys():
                    zw = 0.
                    for bbb in vbox: zw+=tempdict['R'+kres][ksimu][bbb][kdiag]
                    tempdict_1d['R'+kres][ksimu][kbox][kdiag] = zw
                #
            #
        #
    #
    
    #--------------------
    # COMPUTE BGC - CTL, COU - BGC, RAD - CTL and COU-BGC-(RAD-CTL)
    #--------------------

    for kres in res_list:
        tempdict_1d['R'+kres]['BGC - CTL'] = {}
        tempdict_1d['R'+kres]['COU - BGC'] = {}
        tempdict_1d['R'+kres]['RAD - CTL'] = {}
        tempdict_1d['R'+kres]['COU - BGC - (RAD - CTL)'] = {}
        for kboxes in boxes_1d_list.keys():
            tempdict_1d['R'+kres]['BGC - CTL'][kboxes] = {}
            tempdict_1d['R'+kres]['COU - BGC'][kboxes] = {}
            tempdict_1d['R'+kres]['RAD - CTL'][kboxes] = {}
            tempdict_1d['R'+kres]['COU - BGC - (RAD - CTL)'][kboxes] = {}
            for kkk, vvv in tempdict_1d['R'+kres]['CTL'][kboxes].items() :
                tempdict_1d['R'+kres]['BGC - CTL'][kboxes][kkk] =\
                    tempdict_1d['R'+kres]['BGC'][kboxes][kkk] - vvv
                tempdict_1d['R'+kres]['COU - BGC'][kboxes][kkk] =\
                    tempdict_1d['R'+kres]['COU'][kboxes][kkk] -\
                    tempdict_1d['R'+kres]['BGC'][kboxes][kkk]
                tempdict_1d['R'+kres]['RAD - CTL'][kboxes][kkk] =\
                    tempdict_1d['R'+kres]['RAD'][kboxes][kkk] - vvv
                tempdict_1d['R'+kres]['COU - BGC - (RAD - CTL)'][kboxes][kkk] =\
                    tempdict_1d['R'+kres]['COU'][kboxes][kkk] -\
                    tempdict_1d['R'+kres]['BGC'][kboxes][kkk] -\
                    (tempdict_1d['R'+kres]['RAD'][kboxes][kkk] - vvv)
            #
        #
    #
                    
    #--------------------
    # AVG1, STD1
    #--------------------
    
    tempdict_1d['AVG1'] = {}
    tempdict_1d['STD1'] = {}
    for ksimu in ['CTL', 'BGC', 'RAD', 'COU', 'BGC - CTL',\
                  'COU - BGC', 'RAD - CTL', 'COU - BGC - (RAD - CTL)']:
        tempdict_1d['AVG1'][ksimu] = {}
        tempdict_1d['STD1'][ksimu] = {}
        for kboxes in boxes_1d_list.keys() :
            tempdict_1d['AVG1'][ksimu][kboxes] = {}
            tempdict_1d['STD1'][ksimu][kboxes] = {}
            for kkk in tempdict_1d['R'+res_list[0]][ksimu][kboxes].keys():
                
                zw = []
                for kr1s in r1s : zw.append(tempdict_1d['R'+kr1s][ksimu][kboxes][kkk])
                zw = np.array(zw)
                tempdict_1d['AVG1'][ksimu][kboxes][kkk] =  np.nanmean(zw, axis=0)
                tempdict_1d['STD1'][ksimu][kboxes][kkk] =  unc*np.nanstd(zw, axis=0)
            #
        #
    #
    
    #--------------------
    # SAVE DATA2PRINT
    #--------------------
    
    data2print = {}
    for ksimu in ['CTL', 'BGC - CTL', 'COU - BGC', 'RAD - CTL', \
                  'COU - BGC - (RAD - CTL)', 'BGC', 'RAD', 'COU']:
        data2print[ksimu] = {}
        for kres in ['AVG1', 'STD1', 'R9', 'R27']:
            data2print[ksimu][kres] = {}
            for kboxes in boxes_1d_list.keys(): 
                data2print[ksimu][kres][kboxes] = {}
                for kkk, vvv in tempdict_1d[kres][ksimu][kboxes].items():
                    data2print[ksimu][kres][kboxes][kkk] = np.around(vvv, decimals=2)
                #
            #
        #
    #

    #####################
    # SAVE IN TEXT FILE
    #####################
    # data2print[ksimu][kres][kboxes][kkk] = vvv

    print('> save data2print in text file')
    print(datetime.datetime.now())

    fff = open(savefile, "w+")

    fff.write(str(datetime.datetime.now()))
    fff.write("\n")
    fff.write("\n")

    for ksimu, vsimu in data2print.items() :

        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n############")
        fff.write("\n############       " + str(ksimu))
        fff.write("\n############")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n##################################################")
        fff.write("\n")
        fff.write("\n")
        fff.write("\n")

        for kres, vres in vsimu.items() :

            fff.write("\n==================================================")
            fff.write("\n==================================================")
            fff.write("\n               " + str(kres) + ", "+str(ksimu)     )
            fff.write("\n==================================================")
            fff.write("\n==================================================")
            fff.write("\n")
            fff.write("\n")
            
            for kboxes, vboxes  in vres.items() :

                fff.write("\n--------------------------------------------------")
                fff.write("\n               " + str(kboxes) + ', '+ str(ksimu) + ", "+str(kres))
                fff.write("\n--------------------------------------------------")
                fff.write("\n")
                fff.write("\n")

                # split dict for printing for first the 'Sum...'
                kkk1, kkk2 = [], []
                for kkk in vboxes.keys():
                    if kkk[:3] == 'Sum': kkk1.append(kkk)
                    else: kkk2.append(kkk)
                #
                for kvar in kkk1:
                    fff.write(kvar+":  "+str(vboxes[kvar])+" Tmol-C")
                    fff.write("   ---   ")
                #
                fff.write("\n")
                fff.write("\n")
                for kvar in kkk2:
                    fff.write(kvar+":  "+str(vboxes[kvar])+" Tmol-C")
                    fff.write("   ---   ")
                #
                fff.write("\n")
            #
            fff.write("\n")
            fff.write("\n")
        #
        fff.write("\n")
        fff.write("\n")
        fff.write("\n")
    #
    fff.close()
    print("File saved: ", savefile)


#
##################################################
# END SAVEVALUES FIG4 1D
##################################################

