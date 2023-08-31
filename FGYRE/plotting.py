"""
Functions to plot fields
"""

from IPython import embed
# import time
import numpy        as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from FGYRE import averaging, extra

####################################################################################################
def map_delta_r1r9r27(dat1, dat9, dat27, lev=None, zfact=1., ext=None, levD=None, extD=None,\
                          cbtitle='', grid='T', CTLcmap = 'viridis', Dcmap = 'RdBu_r', percent=None, percref=None, \
                          savefig=None,\
                          suffig='functions_GYRE_map_delta_r1r9r27_',\
                          dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) :
    """
    Plot 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage: map_delta_r1r9r27(dat1, dat9, dat27, lev=None, zfact=1.,
       ext=None, levD=None, extD='both', cbtitle='', grid='T',
       CTLcmap='viridis', Dcmap = 'RdBu_r', percent=None, percref=None, savefig=None,
       suffig='functions_GYRE_map_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':xyarray, 'CC':xyarray,
       'mesh':mesh dictionary)
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    zfact       : integer
       to change unit of variable.
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    cbtitle     : string
       add title to colorbars
    grid        : 'T' (default), 'U' or 'V'
    CTLcmap     : string defining the cmap
    Dcmap       : string defining the cmap
    percent     : None (default), 'local' or 'global'
       to compute delta in percent or not
    percref     : list
       list of 3 float/int or 3 array used to compute the change in
       percent.  If it is array and percent is local it has to be the
       same shape of dat1, dat 9 and dat27
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_XY_axis : xlim, ylim
    """
    
    print("> plotting_map_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1['mesh']
    mesh9   = dat9['mesh']
    mesh27  = dat27['mesh']
    
    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
    elif percent == 'global' : 
        dzwr1  = (zwcc1  - zwctl1 ) / averaging.ymean(averaging.xmean(ref1 , mesh1 , dim='yx'), mesh1, dim='y') * 100
        dzwr9  = (zwcc9  - zwctl9 ) / averaging.ymean(averaging.xmean(ref9 , mesh9 , dim='yx'), mesh1, dim='y') * 100
        dzwr27 = (zwcc27 - zwctl27) / averaging.ymean(averaging.xmean(ref27, mesh27, dim='yx'), mesh1, dim='y') * 100
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact        = 1/2.54
    figsizeXY_3x3 = (15*infact, 15*infact) #(width, height)
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsizeXY_3x3)

    if isinstance(lev, type(None)) : 
        # std = np.ma.std(zwctl1)
        # med = np.ma.median(zwctl1)
        # zw = med-2*std
        zw = np.quantile(zwctl1, .10)
        if zw != 0 :
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmin = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        # zw = med+2*std
        zw = np.quantile(zwctl1, .90)
        if zw != 0 : 
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        # std = np.ma.std(dzwr1)
        # med = np.ma.median(dzwr1)
        # zw = med-2*std
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #
    if ext==None : 
        if np.min(lev) == 0 : ext='max'
        elif np.max(lev) == 0 : ext='min'
        else : ext='both'
    #
    if extD==None : 
        if np.min(levD) == 0 : extD='max'
        elif np.max(levD) == 0 : extD='min'
        else : extD='both'
    #

    cf1 = ax[0, 0].contourf(mesh1['lon'+grid], mesh1['lat'+grid],  zwctl1, lev,\
                                cmap=CTLcmap, extend=ext)
    ax[1, 0].contourf(mesh1['lon'+grid], mesh1['lat'+grid],   zwcc1, lev,\
                          cmap=CTLcmap, extend=ext)
    cf3 = ax[2, 0].contourf(mesh1['lon'+grid], mesh1['lat'+grid],    dzwr1, levD,\
                                cmap=Dcmap, extend=extD)
    ax[0, 1].contourf(mesh9['lon'+grid], mesh9['lat'+grid],  zwctl9, lev,\
                          cmap=CTLcmap, extend=ext)
    ax[1, 1].contourf(mesh9['lon'+grid], mesh9['lat'+grid],   zwcc9, lev,\
                          cmap=CTLcmap, extend=ext)
    ax[2, 1].contourf(mesh9['lon'+grid], mesh9['lat'+grid],    dzwr9, levD,\
                          cmap=Dcmap, extend=extD)
    ax[0, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid], zwctl27, lev,\
                          cmap=CTLcmap, extend=ext)
    ax[1, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid],  zwcc27, lev,\
                          cmap=CTLcmap, extend=ext)
    ax[2, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid],   dzwr27, levD,\
                          cmap=Dcmap, extend=extD)

    make_XY_axis(ax[0, 0], title='a) CTL1'      , xlab='', **kwargs)
    make_XY_axis(ax[1, 0], title='b) CC1'       , xlab='', **kwargs)
    make_XY_axis(ax[2, 0], title='c) CC1 - CTL1'           , **kwargs)
    make_XY_axis(ax[0, 1], title='d) CTL9'      , xlab='', ylab='', **kwargs)
    make_XY_axis(ax[1, 1], title='e) CC9'       , xlab='', ylab='', **kwargs)
    make_XY_axis(ax[2, 1], title='f) CC9 - CTL9'  , ylab='', **kwargs)
    make_XY_axis(ax[0, 2], title='g) CTL27'     , xlab='', ylab='', **kwargs)
    make_XY_axis(ax[1, 2], title='h) CC27'      , xlab='', ylab='', **kwargs)
    make_XY_axis(ax[2, 2], title='i) CC27 - CTL27', ylab='', **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)
    
    cbar_ax = fig.add_axes([0.17, 0.935, 0.6588, 0.03094])
    fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                     label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, 0.005, 0.6588, 0.03094])

    if (percent=='local' or percent=='global') : fig.colorbar(cf3, cax=cbar_ax, \
                                                                  orientation='horizontal', \
                                                                  label='Change in %', format='%.2e')
    elif percent==None :  fig.colorbar(cf3, cax=cbar_ax, orientation='horizontal',\
                                           label=cbtitle, format='%.2e')
    else : raise ValueError("check percent")
    
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
    #
# END map_delta_r1r9r27
####################################################################################################

####################################################################################################
def meridional_section_delta_r1r9r27(dat1, dat9, dat27, zfact=1., \
                                     lev=None, ext=None, \
                                     levD=None, extD=None,\
                                     norm=None, normD=None,\
                                     percent=None, percref=None,\
                                     cbtitle='', grid='T', CTLcmap = 'viridis', Dcmap = 'RdBu_r',\
                                     addline=None, \
                                     savefig=None, \
                                     suffig='functions_GYRE_meridional_section_delta_r1r9r27_',\
                                     dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/',\
                                     **kwargs) :
    """
    Plot 9 sections :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : meridional_section_delta_r1r9r27(dat1, dat9, dat27,
       zfact=1., lev=None, ext=None, levD=None, extD='both',
       norm=None, normD=None, cbtitle='', grid='T', percent=None,
       percref=None, savefig=None, CTLcmap='viridis',
       suffig='functions_GYRE_meridional_section_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':xyarray, 'CC':xyarray,
       'mesh':mesh dictionary)
    zfact       : integer
       to change unit of variable.
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    norm        : norm instance to set the norm to contourf
       None is default -> linear norm. Can be :
       matplotlib.colors.PowerNorm(.1)
    normD       :
    cbtitle     : string
       add title to colorbars
    grid        : 'T' (default), 'U' or 'V'
    percent     : None (default), 'local' or 'global'
       to compute delta in percent or not
    percref     : list
       list of 3 float/int or 3 array used to compute the change in
       percent.  If it is array and percent is local it has to be the
       same shape of dat1, dat 9 and dat27
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_YZ_axis eg : depmax, depmin, xlim
    """
    
    print("> plotting_meridional_section_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1['mesh']
    mesh9   = dat9['mesh']
    mesh27  = dat27['mesh']
    
    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
    elif percent == 'global' : 
        if isinstance(ref1, (int, float)) & isinstance(ref9, (int, float)) & isinstance(ref27, (int, float)) :
            dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
            dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
            dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        elif isinstance(ref1, type(zwcc1)) & isinstance(ref9, type(zwcc9)) & isinstance(ref27, type(zwcc27)):
            dzwr1  = (zwcc1  - zwctl1 ) / averaging.zmean(averaging.ymean(ref1 , mesh1 , dim='zy'), mesh1, dim='z') * 100
            dzwr9  = (zwcc9  - zwctl9 ) / averaging.zmean(averaging.ymean(ref9 , mesh9 , dim='zy'), mesh9, dim='z') * 100
            dzwr27 = (zwcc27 - zwctl27) / averaging.zmean(averaging.ymean(ref27, mesh27, dim='zy'), mesh27, dim='z') * 100
        else : raise TypeError("check percref")
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact        = 1/2.54
    figsizeYZ_3x3 = (15*infact, 10*infact) # width, height
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsizeYZ_3x3)

    if isinstance(lev, type(None)) : 
        zwq1 = np.quantile(zwctl1, .10)
        zwq9 = np.quantile(zwctl1, .90)
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        if zwq1 != 0 :
            levmin = np.around(zwq1 * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        if zwq9 != 0 : 
            levmax = np.around(zwq9 * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #
    if ext==None : 
        if np.min(lev) == 0 : ext='max'
        elif np.max(lev) == 0 : ext='min'
        else : ext='both'
    #
    if extD==None : 
        if np.min(levD) == 0 : extD='max'
        elif np.max(levD) == 0 : extD='min'
        else : extD='both'
    #

    cf1 = ax[0, 0].contourf(mesh1['lat'+grid][:, 5], mesh1['dep'+grid],  zwctl1, lev,\
                                cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 0].contourf(mesh1['lat'+grid][:, 5], mesh1['dep'+grid],   zwcc1, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    cf3 = ax[2, 0].contourf(mesh1['lat'+grid][:, 5], mesh1['dep'+grid],    dzwr1, levD,\
                                cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 1].contourf(mesh9['lat'+grid][:, 5], mesh9['dep'+grid],  zwctl9, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 1].contourf(mesh9['lat'+grid][:, 5], mesh9['dep'+grid],   zwcc9, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 1].contourf(mesh9['lat'+grid][:, 5], mesh9['dep'+grid],    dzwr9, levD,\
                          cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid], zwctl27, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid],  zwcc27, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid],   dzwr27, levD,\
                          cmap=Dcmap, extend=extD, norm=normD)


    if isinstance(addline, list) : 
        print("addline")
        ax[0, 0].plot(mesh1['lat'+grid] , addline[0]['CTL'], 'k--')
        ax[1, 0].plot(mesh1['lat'+grid] , addline[0]['CC'] , 'k-')
        ax[0, 1].plot(mesh9['lat'+grid] , addline[1]['CTL'], 'k--')
        ax[1, 1].plot(mesh9['lat'+grid] , addline[1]['CC'] , 'k-')
        ax[0, 2].plot(mesh27['lat'+grid], addline[2]['CTL'], 'k--')
        ax[1, 2].plot(mesh27['lat'+grid], addline[2]['CC'] , 'k-')
        ax[2, 0].plot(mesh1['lat'+grid] , addline[0]['CTL'], 'k--')
        ax[2, 0].plot(mesh1['lat'+grid] , addline[0]['CC'] , 'k-')
        ax[2, 1].plot(mesh9['lat'+grid] , addline[1]['CTL'], 'k--')
        ax[2, 1].plot(mesh9['lat'+grid] , addline[1]['CC'] , 'k-')
        ax[2, 2].plot(mesh27['lat'+grid], addline[2]['CTL'], 'k--')
        ax[2, 2].plot(mesh27['lat'+grid], addline[2]['CC'] , 'k-')
    elif isinstance(addline, type(None)) : pass
    else : raise ValueError("Check addline")

    make_YZ_axis(ax[0, 0], title='a) CTL1'      , xlab=''         , **kwargs)
    make_YZ_axis(ax[1, 0], title='b) CC1'       , xlab=''         , **kwargs)
    make_YZ_axis(ax[2, 0], title='c) CC1 - CTL1'                    , **kwargs)
    make_YZ_axis(ax[0, 1], title='d) CTL9'      , xlab='', ylab='', **kwargs)
    make_YZ_axis(ax[1, 1], title='e) CC9'       , xlab='', ylab='', **kwargs)
    make_YZ_axis(ax[2, 1], title='f) CC9 - CTL9'  , ylab=''         , **kwargs)
    make_YZ_axis(ax[0, 2], title='g) CTL27'     , xlab='', ylab='', **kwargs)
    make_YZ_axis(ax[1, 2], title='h) CC27'      , xlab='', ylab='', **kwargs)
    make_YZ_axis(ax[2, 2], title='i) CC27 - CTL27', ylab=''         , **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    # fig.subplots_adjust(wspace = 0.05, hspace = .2)
    cbar_ax = fig.add_axes([0.17, 0.97, 0.6588, 0.0464])
    fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                     label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, -0.06, 0.6588, 0.0464])
    if (percent=='local' or percent=='global') : fig.colorbar(cf3, cax=cbar_ax, \
                                                                  orientation='horizontal', \
                                                                  label='Change in %', format='%.2e')
    elif percent==None :  fig.colorbar(cf3, cax=cbar_ax, orientation='horizontal',\
                                           label=cbtitle, format='%.2e')
    else : raise ValueError("check percent")

    
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
# END meridional_section_delta_r1r9r27
####################################################################################################

####################################################################################################
def zonal_section_delta_r1r9r27(dat1, dat9, dat27, zfact=1., \
                                lev=None, ext=None, \
                                levD=None, extD=None,\
                                norm=None, normD=None,\
                                percent=None, percref=None,\
                                cbtitle='', grid='T', CTLcmap = 'viridis', Dcmap = 'RdBu_r',\
                                addline=None, \
                                savefig=None, \
                                suffig='functions_GYRE_zonal_section_delta_r1r9r27_',\
                                dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/',\
                                **kwargs) :
    """
    Plot 9 sections :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : zonal_section_delta_r1r9r27(dat1, dat9, dat27,
       zfact=1., lev=None, ext=None, levD=None, extD='both',
       norm=None, normD=None, cbtitle='', grid='T', percent=None,
       percref=None, savefig=None, CTLcmap='viridis',
       suffig='functions_GYRE_zonal_section_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':xyarray, 'CC':xyarray,
       'mesh':mesh dictionary)
    zfact       : integer
       to change unit of variable.
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    norm        : norm instance to set the norm to contourf
       None is default -> linear norm. Can be :
       matplotlib.colors.PowerNorm(.1)
    normD       :
    cbtitle     : string
       add title to colorbars
    grid        : 'T' (default), 'U' or 'V'
    percent     : None (default), 'local' or 'global'
       to compute delta in percent or not
    percref     : list
       list of 3 float/int or 3 array used to compute the change in
       percent.  If it is array and percent is local it has to be the
       same shape of dat1, dat 9 and dat27
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_XZ_axis eg : depmax, depmin, xlim
    """
    
    print("> plotting_zonal_section_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1['mesh']
    mesh9   = dat9['mesh']
    mesh27  = dat27['mesh']
    
    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
    elif percent == 'global' : 
        if isinstance(ref1, (int, float)) & isinstance(ref9, (int, float)) & isinstance(ref27, (int, float)) :
            dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
            dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
            dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        elif isinstance(ref1, type(zwcc1)) & isinstance(ref9, type(zwcc9)) & isinstance(ref27, type(zwcc27)):
            dzwr1  = (zwcc1  - zwctl1 ) / averaging.zmean(averaging.xmean(ref1 , mesh1 , dim='zx'), mesh1, dim='z') * 100
            dzwr9  = (zwcc9  - zwctl9 ) / averaging.zmean(averaging.xmean(ref9 , mesh9 , dim='zx'), mesh9, dim='z') * 100
            dzwr27 = (zwcc27 - zwctl27) / averaging.zmean(averaging.xmean(ref27, mesh27, dim='zx'), mesh27, dim='z') * 100
        else : raise TypeError("check percref")
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact        = 1/2.54
    figsizeXZ_3x3 = (15*infact, 10*infact) # width, height
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsizeXZ_3x3)

    if isinstance(lev, type(None)) : 
        zwq1 = np.quantile(zwctl1, .10)
        zwq9 = np.quantile(zwctl1, .90)
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        if zwq1 != 0 :
            levmin = np.around(zwq1 * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        if zwq9 != 0 : 
            levmax = np.around(zwq9 * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #
    if ext==None : 
        if np.min(lev) == 0 : ext='max'
        elif np.max(lev) == 0 : ext='min'
        else : ext='both'
    #
    if extD==None : 
        if np.min(levD) == 0 : extD='max'
        elif np.max(levD) == 0 : extD='min'
        else : extD='both'
    #

    cf1 = ax[0, 0].contourf(mesh1['lon'+grid][5, :], mesh1['dep'+grid],  zwctl1, lev,\
                                cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 0].contourf(mesh1['lon'+grid][5, :], mesh1['dep'+grid],   zwcc1, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    cf3 = ax[2, 0].contourf(mesh1['lon'+grid][5, :], mesh1['dep'+grid],    dzwr1, levD,\
                                cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 1].contourf(mesh9['lon'+grid][5, :], mesh9['dep'+grid],  zwctl9, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 1].contourf(mesh9['lon'+grid][5, :], mesh9['dep'+grid],   zwcc9, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 1].contourf(mesh9['lon'+grid][5, :], mesh9['dep'+grid],    dzwr9, levD,\
                          cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid], zwctl27, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid],  zwcc27, lev,\
                          cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid],   dzwr27, levD,\
                          cmap=Dcmap, extend=extD, norm=normD)

    if isinstance(addline, list) : 
        print("addline")
        ax[0, 0].plot(mesh1['lon'+grid] , addline[0]['CTL'], 'k--')
        ax[1, 0].plot(mesh1['lon'+grid] , addline[0]['CC'] , 'k-')
        ax[0, 1].plot(mesh9['lon'+grid] , addline[1]['CTL'], 'k--')
        ax[1, 1].plot(mesh9['lon'+grid] , addline[1]['CC'] , 'k-')
        ax[0, 2].plot(mesh27['lon'+grid], addline[2]['CTL'], 'k--')
        ax[1, 2].plot(mesh27['lon'+grid], addline[2]['CC'] , 'k-')
        ax[2, 0].plot(mesh1['lon'+grid] , addline[0]['CTL'], 'k--')
        ax[2, 0].plot(mesh1['lon'+grid] , addline[0]['CC'] , 'k-')
        ax[2, 1].plot(mesh9['lon'+grid] , addline[1]['CTL'], 'k--')
        ax[2, 1].plot(mesh9['lon'+grid] , addline[1]['CC'] , 'k-')
        ax[2, 2].plot(mesh27['lon'+grid], addline[2]['CTL'], 'k--')
        ax[2, 2].plot(mesh27['lon'+grid], addline[2]['CC'] , 'k-')
    elif isinstance(addline, type(None)) : pass
    else : raise ValueError("Check addline")

    make_XZ_axis(ax[0, 0], title='a) CTL1'      , xlab=''         , **kwargs)
    make_XZ_axis(ax[1, 0], title='b) CC1'       , xlab=''         , **kwargs)
    make_XZ_axis(ax[2, 0], title='c) CC1 - CTL1'                    , **kwargs)
    make_XZ_axis(ax[0, 1], title='d) CTL9'      , xlab='', ylab='', **kwargs)
    make_XZ_axis(ax[1, 1], title='e) CC9'       , xlab='', ylab='', **kwargs)
    make_XZ_axis(ax[2, 1], title='f) CC9 - CTL9'  , ylab=''         , **kwargs)
    make_XZ_axis(ax[0, 2], title='g) CTL27'     , xlab='', ylab='', **kwargs)
    make_XZ_axis(ax[1, 2], title='h) CC27'      , xlab='', ylab='', **kwargs)
    make_XZ_axis(ax[2, 2], title='i) CC27 - CTL27', ylab=''         , **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    # fig.subplots_adjust(wspace = 0.05, hspace = .2)
    cbar_ax = fig.add_axes([0.17, 0.97, 0.6588, 0.0464])
    fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                     label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, -0.06, 0.6588, 0.0464])
    if (percent=='local' or percent=='global') : fig.colorbar(cf3, cax=cbar_ax, \
                                                                  orientation='horizontal', \
                                                                  label='Change in %', format='%.2e')
    elif percent==None :  fig.colorbar(cf3, cax=cbar_ax, orientation='horizontal',\
                                           label=cbtitle, format='%.2e')
    else : raise ValueError("check percent")

    
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
# END zonal_section_delta_r1r9r27
####################################################################################################

####################################################################################################
def hovmoellerZT_delta_r1r9r27(dat1, dat9, dat27, zfact=1., lev=None, ext=None,\
                               levD=None, extD=None,\
                               norm=None, normD=None,\
                               percent=None, percref=None,\
                               cbtitle='', grid='T', CTLcmap='viridis', Dcmap='RdBu_r', \
                               addline=None, \
                               savefig=None, \
                               suffig='functions_GYRE_hovmoellerZT_delta_r1r9r27_',\
                               dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/',\
                               **kwargs) :

    print("> plotting_hovmoellerZT_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1['mesh']
    mesh9   = dat9['mesh']
    mesh27  = dat27['mesh']
    date1   = extra.cdtime2str(dat1['date'] , **kwargs)
    date9   = extra.cdtime2str(dat9['date'] , **kwargs)
    date27  = extra.cdtime2str(dat27['date'], **kwargs)
    
    # __________________
    # SHIFT MONTH
    date27  = np.roll(date27,6)
    zwctl27 = np.roll(zwctl27,6, axis=0)
    zwcc27  = np.roll(zwcc27,6, axis=0)
    if isinstance(addline, list) : 
        addline[2]['CTL'] = np.roll(addline[2]['CTL'], 6, axis = 0)
        addline[2]['CC']  = np.roll(addline[2]['CC'] , 6, axis = 0)

    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
    elif percent == 'global' : 
        if isinstance(ref1, (int, float)) & isinstance(ref9, (int, float)) & isinstance(ref27, (int, float)) :
            dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
            dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
            dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        elif isinstance(ref1, type(zwcc1)) & isinstance(ref9, type(zwcc9)) & isinstance(ref27, type(zwcc27)):
            dzwr1  = (zwcc1  - zwctl1 ) / averaging.zmean(averaging.tmean(ref1 , mesh1 , dim='tz'), mesh1, dim='z' ) * 100
            dzwr9  = (zwcc9  - zwctl9 ) / averaging.zmean(averaging.tmean(ref9 , mesh9 , dim='tz'), mesh9, dim='z' ) * 100
            dzwr27 = (zwcc27 - zwctl27) / averaging.zmean(averaging.tmean(ref27, mesh27, dim='tz'), mesh27, dim='z') * 100
        else : raise TypeError("check percref")
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact        = 1/2.54
    figsizeTZ_3x3 = (16*infact, 11*infact)
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsizeTZ_3x3)

    if isinstance(lev, type(None)) : 
        zwq1 = np.quantile(zwctl1, .10)
        zwq9 = np.quantile(zwctl1, .90)
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        if zwq1 != 0 :
            levmin = np.around(zwq1 * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        if zwq9 != 0 : 
            levmax = np.around(zwq9 * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zwdq = zwq9 - zwq1
        zw1 = np.floor(np.log10(np.abs(zwdq))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #
    if ext==None : 
        if np.min(lev) == 0 : ext='max'
        elif np.max(lev) == 0 : ext='min'
        else : ext='both'
    #
    if extD==None : 
        if np.min(levD) == 0 : extD='max'
        elif np.max(levD) == 0 : extD='min'
        else : extD='both'
    #
    cf1 = ax[0, 0].contourf(date1, mesh1['dep'+grid],  zwctl1.transpose(), lev,\
                            cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 0].contourf(date1, mesh1['dep'+grid],   zwcc1.transpose(), lev,\
                      cmap=CTLcmap, extend=ext, norm=norm)
    cf3 = ax[2, 0].contourf(date1, mesh1['dep'+grid],    dzwr1.transpose(), levD,\
                            cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 1].contourf(date9, mesh9['dep'+grid],  zwctl9.transpose(), lev,\
                      cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 1].contourf(date9, mesh9['dep'+grid],   zwcc9.transpose(), lev,\
                      cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 1].contourf(date9, mesh9['dep'+grid],    dzwr9.transpose(), levD,\
                      cmap=Dcmap, extend=extD, norm=normD)
    ax[0, 2].contourf(date27, mesh27['dep'+grid], zwctl27.transpose(), lev,\
                      cmap=CTLcmap, extend=ext, norm=norm)
    ax[1, 2].contourf(date27, mesh27['dep'+grid],  zwcc27.transpose(), lev,\
                      cmap=CTLcmap, extend=ext, norm=norm)
    ax[2, 2].contourf(date27, mesh27['dep'+grid],   dzwr27.transpose(), levD,\
                      cmap=Dcmap, extend=extD, norm=normD)

    if isinstance(addline, list) : 
        print("addline")
        ax[0, 0].plot(date1 , addline[0]['CTL'], 'k--')
        ax[1, 0].plot(date1 , addline[0]['CC'] , 'k-')
        ax[0, 1].plot(date9 , addline[1]['CTL'], 'k--')
        ax[1, 1].plot(date9 , addline[1]['CC'] , 'k-')
        ax[0, 2].plot(date27, addline[2]['CTL'], 'k--')
        ax[1, 2].plot(date27, addline[2]['CC'] , 'k-')
        ax[2, 0].plot(date1 , addline[0]['CTL'], 'k--')
        ax[2, 0].plot(date1 , addline[0]['CC'] , 'k-')
        ax[2, 1].plot(date9 , addline[1]['CTL'], 'k--')
        ax[2, 1].plot(date9 , addline[1]['CC'] , 'k-')
        ax[2, 2].plot(date27, addline[2]['CTL'], 'k--')
        ax[2, 2].plot(date27, addline[2]['CC'] , 'k-')
    else : raise ValueError("Check addline")

    make_TZ_axis(ax[0, 0], title='a) CTL1'      , xlab=''         , **kwargs)
    make_TZ_axis(ax[1, 0], title='b) CC1'       , xlab=''         , **kwargs)
    make_TZ_axis(ax[2, 0], title='c) CC1 - CTL1'                    , **kwargs)
    make_TZ_axis(ax[0, 1], title='d) CTL9'      , xlab='', ylab='', **kwargs)
    make_TZ_axis(ax[1, 1], title='e) CC9'       , xlab='', ylab='', **kwargs)
    make_TZ_axis(ax[2, 1], title='f) CC9 - CTL9'  , ylab=''         , **kwargs)
    make_TZ_axis(ax[0, 2], title='g) CTL27'     , xlab='', ylab='', **kwargs)
    make_TZ_axis(ax[1, 2], title='h) CC27'      , xlab='', ylab='', **kwargs)
    make_TZ_axis(ax[2, 2], title='i) CC27 - CTL27', ylab=''         , **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    fig.subplots_adjust(wspace = 0.05, hspace = .2)
    cbar_ax = fig.add_axes([0.17, 0.97, 0.7, 0.0422])
    fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                     label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, -0.06, 0.7, 0.0422])
    if (percent=='local' or percent=='global') : fig.colorbar(cf3, cax=cbar_ax, \
                                                                  orientation='horizontal', \
                                                                  label='Change in %', format='%.2e')
    elif percent==None :  fig.colorbar(cf3, cax=cbar_ax, orientation='horizontal',\
                                           label=cbtitle, format='%.2e')
    else : raise ValueError("check percent")

    
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
# END hovmoellerZT_delta_r1r9r27
####################################################################################################

####################################################################################################
def zonal_line_delta_r1r9r27(dat1, dat9, dat27, zfact=1., title = '', ytitle='', grid='T',\
                             ylim=[None, None], ylimD=[None, None], breakYax=None, breakYDax=None,\
                             percent=None, percref=None, \
                             savefig=None,\
                             suffig='functions_GYRE_zonal_line_delta_r1r9r27_',\
                             dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) :
    """
    Plot zonal lines :
        ----------------
        |  CTL and CC  |
        ----------------
        |    CC-CTL    |
        ----------------
    Usage : zonal_line_delta_r1r9r27(dat1, dat9, dat27, zfact=1.,
       title = '', ytitle='', ylim=[None, None], ylimD=[None, None],
       breakYax=None, breakYDax=None, percent=None, percref=None,
       savefig=None,
       suffig='functions_GYRE_zonla_line_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':yarray, 'CC':yarray,
       'mesh':mesh dictionary)
    zfact             : integer
       to change unit of variable.
    title             : string
       title of the figures
    ytitle            : string
       add title to colorbars
    grid        : string
       'T' (default), 'U', 'V' or 'W'
    ylim        : tupple or list of 2 element 
       set the lower and upper bounds of y axis
    ylimD       : tupple or list of 2 element 
       same as ylim for plotting delta
    breakYax    : list of int or float
       to break the Yaxis. If None (default) no breaking
    breakYDax   : list of int or float
       to break the Yaxis. If None (default) no breaking
    percent     : None (default), 'local' or 'global'
       to compute delta in percent or not
    percref     : list
       list of 3 float/int or 3 array used to compute the change in
       percent.  If it is array and percent is local it has to be the
       same shape of dat1, dat 9 and dat27
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_Yline_axis
    """
    
    print("> plotting_zonal_line_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1 ['mesh']
    mesh9   = dat9 ['mesh']
    mesh27  = dat27['mesh']
    
    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        ytitleb = 'Change in %'
    elif percent == 'global' : 
        dzwr1  = (zwcc1  - zwctl1 ) / averaging.ymean(ref1 , mesh1 , dim='y') * 100
        dzwr9  = (zwcc9  - zwctl9 ) / averaging.ymean(ref9 , mesh9 , dim='y') * 100
        dzwr27 = (zwcc27 - zwctl27) / averaging.ymean(ref27, mesh27, dim='y') * 100
        ytitleb = 'Change in %'
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
        ytitleb = ytitle
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact  = 1/2.54
    fsize   = (8*infact, 8*infact) #(width, height)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex  = 'col', figsize = fsize) 
    if isinstance(breakYax, list) : 
        divider = make_axes_locatable(ax1)
        nsub=len(breakYax)
        ax1b=[]
        for ind in range(nsub) : 
            ax1b.append(divider.append_axes("bottom", size="100%", pad=0.05, sharex=ax1))
        #
    if isinstance(breakYDax, list) : 
        # breakYDax.sort()
        divider = make_axes_locatable(ax2)
        nsubD=len(breakYDax)
        ax2b=[]
        for ind in range(nsubD) : 
            ax2b.append(divider.append_axes("bottom", size="100%", pad=0.05, sharex=ax2))
        #
    #

    lctl1  = ax1.plot(mesh1 ['lat'+grid][:, 5], zwctl1 , 'gray'     , label = 'CTL1' , ls = '--')
    lcc1   = ax1.plot(mesh1 ['lat'+grid][:, 5], zwcc1  , 'gray'     , label = 'CC1'  )
    lctl9  = ax1.plot(mesh9 ['lat'+grid][:, 5], zwctl9 , 'royalblue', label = 'CTL9' , ls = '--')
    lcc9   = ax1.plot(mesh9 ['lat'+grid][:, 5], zwcc9  , 'royalblue', label = 'CC9'  )
    lctl27 = ax1.plot(mesh27['lat'+grid][:, 5], zwctl27, 'firebrick', label = 'CTL27', ls = '--')
    lcc27  = ax1.plot(mesh27['lat'+grid][:, 5], zwcc27 , 'firebrick', label = 'CC27' )
    
    ldr1   = ax2.plot(mesh1 ['lat'+grid][:, 5], dzwr1 , 'gray'     , label = 'R1' )
    ldr9   = ax2.plot(mesh9 ['lat'+grid][:, 5], dzwr9 , 'royalblue', label = 'R9' )
    ldr27  = ax2.plot(mesh27['lat'+grid][:, 5], dzwr27, 'firebrick', label = 'R27')
        
    dash = [2, 2]
    lctl1[0].set_dashes(dash)
    lctl9[0].set_dashes(dash)
    lctl27[0].set_dashes(dash)

    ax1.tick_params(bottom=False)
    ax1.legend( loc = 'upper left', bbox_to_anchor = (1, 1) )    
    if breakYax == None : 
        make_Yline_axis(ax1, title='a) '+title, ylab=ytitle, ylim=ylim, xlab='')
    elif isinstance(breakYax, list) : 
        make_Yline_axis(ax1, title='a) '+title, ylab=ytitle, ylim=(breakYax[-1], ylim[1]), xlab='')
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(bottom=False)
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them

        zwkwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax1.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        for ind in range(nsub) : 
            lbctl1  = ax1b[ind].plot(mesh1 ['lat'+grid][:, 5], zwctl1 , 'gray'     , label = 'CTL1' , ls = '--')
            lbcc1   = ax1b[ind].plot(mesh1 ['lat'+grid][:, 5], zwcc1  , 'gray'     , label = 'CC1'  )
            lbctl9  = ax1b[ind].plot(mesh9 ['lat'+grid][:, 5], zwctl9 , 'royalblue', label = 'CTL9' , ls = '--')
            lbcc9   = ax1b[ind].plot(mesh9 ['lat'+grid][:, 5], zwcc9  , 'royalblue', label = 'CC9'  )
            lbctl27 = ax1b[ind].plot(mesh27['lat'+grid][:, 5], zwctl27, 'firebrick', label = 'CTL27', ls = '--')
            lbcc27  = ax1b[ind].plot(mesh27['lat'+grid][:, 5], zwcc27 , 'firebrick', label = 'CC27' )
            lbctl1[0].set_dashes(dash)
            lbctl9[0].set_dashes(dash)
            lbctl27[0].set_dashes(dash)
            ax1b[ind].spines['top'].set_visible(False)
            if ind+1 != nsub : 
                make_Yline_axis(ax1b[ind], title='', ylab='', ylim=(breakYax[-2-ind], breakYax[-1-ind]), xlab='')
                ax1b[ind].spines['bottom'].set_visible(False)
                ax1b[ind].tick_params(bottom=False)
                ax1b[ind].tick_params(labelbottom=False)
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            else : 
                make_Yline_axis(ax1b[ind], title='', ylab='', ylim=(ylim[0], breakYax[-1-ind]), xlab='')
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
                ax1b[ind].tick_params(bottom=False)
                ax1b[ind].tick_params(labelbottom=False)
            #
        #
    else: raise TypeError("breakYax")

    ax2.legend( loc = 'upper left', bbox_to_anchor = (1, 1) )
    if breakYDax == None : 
        make_Yline_axis(ax2, title='b) '+title+' change', ylab=ytitleb, ylim=ylimD)
    elif isinstance(breakYDax, list) : 
        make_Yline_axis(ax2, title='b) '+title+' change', ylab=ytitle, ylim=(breakYDax[-1], ylimD[1]), xlab='')
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(bottom=False)
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them

        zwkwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax2.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        for ind in range(nsubD) : 
            lbdr1   = ax2b[ind].plot(mesh1 ['lat'+grid][:, 5], dzwr1 , 'gray'     , label = 'R1' )
            lbdr9   = ax2b[ind].plot(mesh9 ['lat'+grid][:, 5], dzwr9 , 'royalblue', label = 'R9' )
            lbdr27  = ax2b[ind].plot(mesh27['lat'+grid][:, 5], dzwr27, 'firebrick', label = 'R27')
            ax2b[ind].spines['top'].set_visible(False)
            if ind+1 != nsubD : 
                make_Yline_axis(ax2b[ind], title='', ylab='', ylim=(breakYDax[-2-ind], breakYDax[-1-ind]), xlab='')
                ax2b[ind].spines['bottom'].set_visible(False)
                ax2b[ind].tick_params(bottom=False)
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
                ax2.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            else : 
                make_Yline_axis(ax2b[ind], title='', ylab='', ylim=(ylimD[0], breakYDax[-1-ind]))
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            #
        #
    else: raise TypeError("breakYDax")
        
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
 
    
# END zonal_line_delta_r1r9r27
####################################################################################################

####################################################################################################
def vertical_profile_delta_r1r9r27(dat1, dat9, dat27, zfact=1., title = '', xtitle='', grid='T',\
                                   xlim=[None, None], xlimD=[None, None], \
                                   breakXax=None, breakXDax=None,\
                                   ylim=None, \
                                   percent=None, percref=None, \
                                   savefig=None,\
                                   suffig='functions_GYRE_vertical_profile_delta_r1r9r27_',\
                                   dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) :
    """
    Plot vertical profiles :
        -----------------------
        |    CTL   |          |  
        |    and   | CC - CTL |                    
        |    CC    |          |
        -----------------------
    Usage : vertical_profile_delta_r1r9r27(dat1, dat9, dat27, zfact=1.,
       title = '', xtitle='', xlim=[None, None], xlimD=[None, None],
       breakXax=None, breakXDax=None, percent=None, percref=None,
       savefig=None,
       suffig='functions_GYRE_vertical_profile_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':1D array, 'CC':1D array,
       'mesh':mesh dictionary)
    zfact             : integer
       to change unit of variable.
    title             : string
       title of the figures
    xtitle            : string
       add title to colorbars
    grid        : string
       'T' (default), 'U', 'V' or 'W'
    xlim        : tupple or list of 2 element 
       set the lower and upper bounds of x axis
    xlimD       : tupple or list of 2 element 
       same as xlim for plotting delta
    breakXax    : list of int or float
       to break the Xaxis. If None (default) no breaking
    breakXDax   : list of int or float
       to break the Xaxis. If None (default) no breaking
    percent     : None (default), 'local' or 'global'
       to compute delta in percent or not
    percref     : list
       list of 3 float/int or 3 array used to compute the change in
       percent.  If it is array and percent is local it has to be the
       same shape of dat1, dat 9 and dat27
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : not passed
    """
    
    print("> plotting_vertical_profile_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    mesh1   = dat1 ['mesh']
    mesh9   = dat9 ['mesh']
    mesh27  = dat27['mesh']
    
    # __________________
    # DELTA CC - CTL
    if percent in ('local', 'global') : 
        if percref == None : 
            ref1=zwctl1
            ref9=zwctl9
            ref27=zwctl27
        else : 
            ref1  = percref[0]
            ref9  = percref[1]
            ref27 = percref[2]
        #
    #
    if percent == 'local' : 
        dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
        dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
        dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        xtitleb = 'Change in %'
    elif percent == 'global' : 
        if isinstance(ref1, (int, float)) & isinstance(ref9, (int, float)) & isinstance(ref27, (int, float)) :
            dzwr1  = (zwcc1  - zwctl1 ) / ref1  * 100
            dzwr9  = (zwcc9  - zwctl9 ) / ref9  * 100
            dzwr27 = (zwcc27 - zwctl27) / ref27 * 100
        elif isinstance(ref1, type(zwcc1)) & isinstance(ref9, type(zwcc9)) & isinstance(ref27, type(zwcc27)):
            dzwr1  = (zwcc1  - zwctl1 ) / averaging.zmean(ref1 , mesh1 , dim='z') * 100
            dzwr9  = (zwcc9  - zwctl9 ) / averaging.zmean(ref9 , mesh9 , dim='z') * 100
            dzwr27 = (zwcc27 - zwctl27) / averaging.zmean(ref27, mesh27, dim='z') * 100
        else : raise TypeError("check percref")
        xtitleb = 'Change in %'
    elif percent == None :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
        xtitleb = xtitle
    else : raise ValueError("check percent")

    # __________________
    # PLOT

    infact  = 1/2.54
    fsize   = (16*infact, 8*infact) #(width, height)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey  = 'row', figsize = fsize) 
    if isinstance(breakXax, list) : 
        divider = make_axes_locatable(ax1)
        nsub=len(breakXax)
        imidle=int(np.ceil(nsub/2))-1
        ax1b=[]
        for ind in range(nsub) : 
            ax1b.append(divider.append_axes("right", size="100%", pad=0.05, sharey=ax1))
        #
    if isinstance(breakXDax, list) : 
        divider = make_axes_locatable(ax2)
        nsubD=len(breakXDax)
        imidleD=int(np.ceil(nsubD/2))-1
        ax2b=[]
        for ind in range(nsubD) : 
            ax2b.append(divider.append_axes("right", size="100%", pad=0.05, sharey=ax2))
        #
    lctl1,  = ax1.plot(zwctl1 , mesh1 ['dep'+grid], 'gray'     , label = 'CTL1' , ls = '--')
    lctl9,  = ax1.plot(zwctl9 , mesh9 ['dep'+grid], 'royalblue', label = 'CTL9' , ls = '--')
    lctl27, = ax1.plot(zwctl27, mesh27['dep'+grid], 'firebrick', label = 'CTL27', ls = '--')
    lcc1,   = ax1.plot(zwcc1  , mesh1 ['dep'+grid], 'gray'     , label = 'CC1'  )
    lcc9,   = ax1.plot(zwcc9  , mesh9 ['dep'+grid], 'royalblue', label = 'CC9'  )
    lcc27,  = ax1.plot(zwcc27 , mesh27['dep'+grid], 'firebrick', label = 'CC27' )
    
    ldr1,   = ax2.plot(dzwr1 , mesh1 ['dep'+grid], 'gray'     , label = 'R1' )
    ldr9,   = ax2.plot(dzwr9 , mesh9 ['dep'+grid], 'royalblue', label = 'R9' )
    ldr27,  = ax2.plot(dzwr27, mesh27['dep'+grid], 'firebrick', label = 'R27')
        
    dash = [2, 2]
    lctl1.set_dashes(dash)
    lctl9.set_dashes(dash)
    lctl27.set_dashes(dash)

    if breakXax == None : 
        make_Zline_axis(ax1, title='a) '+title, xlab=xtitle, xlim=xlim, ylim=ylim)
        ax1.legend( loc = 'upper center', bbox_to_anchor = (0.5, -0.15), ncol=2 )    
    elif isinstance(breakXax, list) : 
        make_Zline_axis(ax1, title='a) '+title, xlab=xtitle, xlim=(xlim[0], breakXax[0]),  ylim=ylim)
        ax1.spines['right'].set_visible(False)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(1))
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        zwkwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax1.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        ax1.plot((1-d, 1+d), (1-d, 1+d), **zwkwargs)        # bottom-right diagonal
        for ind in range(nsub) : 
            lbctl1  = ax1b[ind].plot(zwctl1 , mesh1 ['dep'+grid], 'gray'     , label = 'CTL1' , ls = '--')
            lbctl9  = ax1b[ind].plot(zwctl9 , mesh9 ['dep'+grid], 'royalblue', label = 'CTL9' , ls = '--')
            lbctl27 = ax1b[ind].plot(zwctl27, mesh27['dep'+grid], 'firebrick', label = 'CTL27', ls = '--')
            lbcc1   = ax1b[ind].plot(zwcc1  , mesh1 ['dep'+grid], 'gray'     , label = 'CC1'  )
            lbcc9   = ax1b[ind].plot(zwcc9  , mesh9 ['dep'+grid], 'royalblue', label = 'CC9'  )
            lbcc27  = ax1b[ind].plot(zwcc27 , mesh27['dep'+grid], 'firebrick', label = 'CC27' )
            lbctl1[0].set_dashes(dash)
            lbctl9[0].set_dashes(dash)
            lbctl27[0].set_dashes(dash)
            ax1b[ind].spines['left'].set_visible(False)
            ax1b[ind].xaxis.set_major_locator(plt.MaxNLocator(1))
            ax1b[imidle].legend( loc = 'upper center', bbox_to_anchor = (0.5, -0.15), ncol=2 )    
            if ind+1 != nsub : 
                make_Zline_axis(ax1b[ind], title='', xlab='', xlim=(breakXax[ind], breakXax[1+ind]), ylab='', ylim=ylim)
                ax1b[ind].spines['right'].set_visible(False)
                ax1b[ind].tick_params(left=False)
                ax1b[ind].tick_params(labelleft=False)
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax1b[ind].plot((1-d, 1+d), (1-d, 1+d), **zwkwargs)        # bottom-right diagonal
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((-d, +d), (-d, +d), **zwkwargs)  # top-left diagonal
            else : 
                make_Zline_axis(ax1b[ind], title='', xlab='', xlim=(breakXax[-1], xlim[1]), ylab='', ylim=ylim)
                ax1b[ind].tick_params(left=False)
                ax1b[ind].tick_params(labelleft=False)
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((-d, +d), (-d, +d), **zwkwargs)  # top-left diagonal
            #
        #
    else: raise TypeError("breakXax")

    ax2.tick_params(left=False)
    if breakXDax == None : 
        ax2.legend( loc = 'upper center', bbox_to_anchor = (0.5, -.15))
        make_Zline_axis(ax2, title='b) '+title+' change', xlab=xtitleb, xlim=xlimD, ylab='', ylim=ylim)
        # make_Zline_axis(ax1, title='a) '+title, xlab=xtitle, xlim=xlim, ylim=ylim)
        # ax1.legend( loc = 'upper center', bbox_to_anchor = (0.5, -0.15), ncol=2 )    
    elif isinstance(breakXDax, list) : 
        make_Zline_axis(ax2, title='b) '+title+' change', xlab=xtitleb, xlim=(xlimD[0], breakXDax[0]), ylim=ylim, ylab='')
        ax2.spines['right'].set_visible(False)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(1))
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        zwkwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax2.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        ax2.plot((1-d, 1+d), (1-d, 1+d), **zwkwargs)        # bottom-right diagonal
        for ind in range(nsub) : 
            ldr1,   = ax2b[ind].plot(dzwr1 , mesh1 ['dep'+grid], 'gray'     , label = 'R1' )
            ldr9,   = ax2b[ind].plot(dzwr9 , mesh9 ['dep'+grid], 'royalblue', label = 'R9' )
            ldr27,  = ax2b[ind].plot(dzwr27, mesh27['dep'+grid], 'firebrick', label = 'R27')
            ax2b[ind].spines['left'].set_visible(False)
            ax2b[ind].xaxis.set_major_locator(plt.MaxNLocator(1))
            ax2b[imidle].legend( loc = 'upper center', bbox_to_anchor = (0.5, -0.15))    
            if ind+1 != nsub : 
                make_Zline_axis(ax2b[ind], title='', xlab='', xlim=(breakXDax[ind], breakXDax[1+ind]), ylab='', ylim=ylim)
                ax2b[ind].spines['right'].set_visible(False)
                ax2b[ind].tick_params(left=False)
                ax2b[ind].tick_params(labelleft=False)
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2b[ind].plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax2b[ind].plot((1-d, 1+d), (1-d, 1+d), **zwkwargs)        # bottom-right diagonal
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((-d, +d), (-d, +d), **zwkwargs)  # top-left diagonal
            else : 
                make_Zline_axis(ax2b[ind], title='', xlab='', xlim=(breakXDax[-1], xlimD[1]), ylab='', ylim=ylim)
                ax2b[ind].tick_params(left=False)
                ax2b[ind].tick_params(labelleft=False)
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((-d, +d), (-d, +d), **zwkwargs)  # top-left diagonal
            #
        #
    else: raise TypeError("breakXDax")
           
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
 
    
# END vertical_profile_delta_r1r9r27
####################################################################################################

####################################################################################################
def timeseries_delta_r1r9r27(dat1, dat9, dat27, zfact=1., title = '', ytitle='',\
                             ylim=[None, None], ylimD=[None, None], breakYax=None, breakYDax=None, \
                             percent=False, percref=None, \
                             savefig=None,\
                             suffig='functions_GYRE_timeseries_delta_r1r9r27_',\
                             dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) :
    """
    Plot 2 time series :
        ----------------
        |  CTL and CC  |
        ----------------
        |    CC-CTL    |
        ----------------
    Usage : timeseries_delta_r1r9r27(dat1, dat9, dat27, zfact=1.,
       title = '', ytitle='', ylim=[None, None], breakYax=None,
       breakYDax=None, ylimD=[None, None], percent=False,
       percref=None, savefig=None,
       suffig='functions_GYRE_timeseries_delta_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionaries {'CTL':list, 'CC':list,
       'date':dates of the time series)
    zfact             : integer
       to change unit of variable.
    title             : string
       title of the figures
    ytitle            : string
       add title to y axis
    ylim        : tupple or list of 2 element 
       set the lower and upper bounds of y axis
    ylimD       : tupple or list of 2 element 
       same as ylim for plotting delta
    breakYax    : list of int or float
       to break the Yaxis. If None (default) no breaking
    breakYDax   : list of int or float
       to break the Yaxis. If None (default) no breaking
    percent     : Boolean
       to compute delta in percent (of mean CTL) or not
    percref     : list
       list of 3 float/int used to compute the change in
       percent. 
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : not passed
    """
    
    print("> plotting_timeseries_delta_r1r9r27")
    
    zwctl1  = zfact * dat1['CTL'] 
    zwcc1   = zfact * dat1['CC']
    zwctl9  = zfact * dat9['CTL']
    zwcc9   = zfact * dat9['CC']
    zwctl27 = zfact * dat27['CTL']
    zwcc27  = zfact * dat27['CC']
    date1   = extra.cdtime2str(dat1['date'] , **kwargs)
    date9   = extra.cdtime2str(dat9['date'] , **kwargs)
    date27  = extra.cdtime2str(dat27['date'], **kwargs)
    
    # __________________
    # DELTA CC - CTL
    if percent :
        if percref == None : 
            dzwr1  = (zwcc1  - zwctl1 ) / np.nanmean(zwctl1 ) * 100
            dzwr9  = (zwcc9  - zwctl9 ) / np.nanmean(zwctl9 ) * 100
            dzwr27 = (zwcc27 - zwctl27) / np.nanmean(zwctl27) * 100
        else : 
            dzwr1  = (zwcc1  - zwctl1 ) / percref[0] * 100
            dzwr9  = (zwcc9  - zwctl9 ) / percref[1] * 100
            dzwr27 = (zwcc27 - zwctl27) / percref[2] * 100
        #
        ytitleb = 'Change in %'
    else :
        dzwr1  = zwcc1  - zwctl1
        dzwr9  = zwcc9  - zwctl9
        dzwr27 = zwcc27 - zwctl27
        ytitleb = ytitle
    #

    # __________________
    # PLOT

    infact  = 1/2.54
    fsize   = (8*infact, 8*infact) #(width, height)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex  = 'col', figsize = fsize)

    if isinstance(breakYax, list) : 
        divider = make_axes_locatable(ax1)
        nsub=len(breakYax)
        ax1b=[]
        for ind in range(nsub) : 
            ax1b.append(divider.append_axes("bottom", size="100%", pad=0.05, sharex=ax1))
        #
    if isinstance(breakYDax, list) : 
        # breakYDax.sort()
        divider = make_axes_locatable(ax2)
        nsubD=len(breakYDax)
        ax2b=[]
        for ind in range(nsubD) : 
            ax2b.append(divider.append_axes("bottom", size="100%", pad=0.05, sharex=ax2))
        #
    #


    lctl1  = ax1.plot(date1 , zwctl1 , 'black', lw=3, label = 'CTL1' , ls='--')
    lcc1   = ax1.plot(date1 , zwcc1  , 'black', lw=3, label = 'CC1'  )
    lctl9  = ax1.plot(date9 , zwctl9 , 'black', lw=2, label = 'CTL9' , ls='--')
    lcc9   = ax1.plot(date9 , zwcc9  , 'black', lw=2, label = 'CC9'  )
    lctl27 = ax1.plot(date27, zwctl27, 'black', lw=1, label = 'CTL27', ls='--')
    lcc27  = ax1.plot(date27, zwcc27 , 'black', lw=1, label = 'CC27' )
    
    ldr1   = ax2.plot(date1 , dzwr1 , 'black', lw=3, label = 'R1' )
    ldr9   = ax2.plot(date9 , dzwr9 , 'black', lw=2, label = 'R9' )
    ldr27  = ax2.plot(date27, dzwr27, 'black', lw=1, label = 'R27')
        
    ax1.tick_params(bottom=False)
    ax1.legend( loc = 'upper left', bbox_to_anchor = (1, 1) )    
    if breakYax == None : 
        make_timeseries_axis(ax1, title='a) '+title, ylab=ytitle, ylim=ylim, xlab='')
    elif isinstance(breakYax, list) : 
        make_timeseries_axis(ax1, title='a) '+title, ylab=ytitle, ylim=(breakYax[-1], ylim[1]), xlab='')
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(bottom=False)
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them

        zwkwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax1.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        for ind in range(nsub) : 
            lbctl1  = ax1b[ind].plot(date1 , zwctl1 , 'black', lw=3, label = 'CTL1' , ls='--')
            lbcc1   = ax1b[ind].plot(date1 , zwcc1  , 'black', lw=3, label = 'CC1'  )
            lbctl9  = ax1b[ind].plot(date9 , zwctl9 , 'black', lw=2, label = 'CTL9' , ls='--')
            lbcc9   = ax1b[ind].plot(date9 , zwcc9  , 'black', lw=2, label = 'CC9'  )
            lbctl27 = ax1b[ind].plot(date27, zwctl27, 'black', lw=1, label = 'CTL27', ls='--')
            lbcc27  = ax1b[ind].plot(date27, zwcc27 , 'black', lw=1, label = 'CC27' )
            ax1b[ind].spines['top'].set_visible(False)
            if ind+1 != nsub : 
                make_timeseries_axis(ax1b[ind], title='', ylab='', ylim=(breakYax[-2-ind], breakYax[-1-ind]), xlab='')
                ax1b[ind].spines['bottom'].set_visible(False)
                ax1b[ind].tick_params(bottom=False)
                ax1b[ind].tick_params(labelbottom=False)
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            else : 
                make_timeseries_axis(ax1b[ind], title='', ylab='', ylim=(ylim[0], breakYax[-1-ind]), xlab='')
                zwkwargs.update(transform=ax1b[ind].transAxes)  # switch to the bottom axes
                ax1b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax1b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
                ax1b[ind].tick_params(bottom=False)
                ax1b[ind].tick_params(labelbottom=False)
            #
        #
    else: raise TypeError("breakYax")

    ax2.legend( loc = 'upper left', bbox_to_anchor = (1, 1) )
    if breakYDax == None : 
        make_timeseries_axis(ax2, title='b) '+title+' change', ylab=ytitleb, ylim=ylimD)
    elif isinstance(breakYDax, list) : 
        make_timeseries_axis(ax2, title='b) '+title+' change', ylab=ytitle, ylim=(breakYDax[-1], ylimD[1]), xlab='')
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(bottom=False)
        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them

        zwkwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=0.5)
        ax2.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
        for ind in range(nsubD) : 
            lbdr1   = ax2b[ind].plot(date1 , dzwr1 , 'black', lw=3, label = 'R1' )
            lbdr9   = ax2b[ind].plot(date9 , dzwr9 , 'black', lw=2, label = 'R9' )
            lbdr27  = ax2b[ind].plot(date27, dzwr27, 'black', lw=1, label = 'R27')
            ax2b[ind].spines['top'].set_visible(False)
            if ind+1 != nsubD : 
                make_timeseries_axis(ax2b[ind], title='', ylab='', ylim=(breakYDax[-2-ind], breakYDax[-1-ind]), xlab='')
                ax2b[ind].spines['bottom'].set_visible(False)
                ax2b[ind].tick_params(bottom=False)
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2.plot((-d, +d), (-d, +d), **zwkwargs)        # top-left diagonal
                ax2.plot((1 - d, 1 + d), (-d, +d), **zwkwargs)  # top-right diagonal
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            else : 
                make_timeseries_axis(ax2b[ind], title='', ylab='', ylim=(ylimD[0], breakYDax[-1-ind]))
                zwkwargs.update(transform=ax2b[ind].transAxes)  # switch to the bottom axes
                ax2b[ind].plot((-d, +d), (1 - d, 1 + d), **zwkwargs)  # bottom-left diagonal
                ax2b[ind].plot((1 - d, 1 + d), (1 - d, 1 + d), **zwkwargs)  # bottom-right diagonal
            #
        #
    else: raise TypeError("breakYDax")
        
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
 
    
# END timeseries_delta_r1r9r27
####################################################################################################

####################################################################################################
def make_timeseries_axis(ax, ylab='', xlab='Time', title='', ylim=None, **kwargs) : 
    """
    Add xalbe, ylabel, title to the axes ax for a timeseries plot
    Usage: make_timeseries_axis(ax, ylab='', xlab='Time', title='',
       ylim=None)

    Parameters
    ----------
    ax    : axes 
    xlab  : String to label xaxis
    ylab  : String to label yaxis
    title : String to title of the figure
    ylim  : to set ylimit
    """
    ax.set_title(split_title(title, maxword=8), loc='left')
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
# END make_timeseries_axis
####################################################################################################

####################################################################################################
def make_Yline_axis(ax, xlab='Northward km', ylab='', title = '', \
                    ylim=None, xlim=(20, 48.5974), \
                    xticks=[20, 34.2987, 48.5974], \
                    xticklabels=['0', '1,590', '3,180'], **kwargs) : 
    """
    Add xlabel, ylabel and title to the axes ax 
    Usage : make_Yline_axis(ax, xlab='Northward km', ylab='', title =
            '', ylim=None, xlim=(20, 48.5974), xticks=[20, 34.2987,
            48.5974], xticklabels=['0', '1,590', '3,180'],
            **kwargs)

    Parameters
    ----------
    ax     : axes 
    xlab   : String to label xaxis
    ylab   : String to label yaxis
    title  : String to title of the figure
    xlim   : tupple
    ylim   : tupple
    """
    print("> plotting_make_Yline_axis")
    ax.set_xlabel(xlab)
    ax.set_xlim(xlim)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xticklabels)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylab)
    ax.set_title(split_title(title, maxword=20), loc='left')
# END make_Ylin_axis
####################################################################################################

####################################################################################################
def make_Zline_axis(ax, ylab='Depth [m]', xlab='', title='', ylim=(600, 0), xlim=None, **kwargs) : 
    """
    Add xalbe, ylabel, title to the axes ax for a vertical profile plot
    Usage: make_Zline_axis(ax, ylab='Depth [m]', xlab='', title='',
       ylim=(600, 0), xlim=None, **kwargs)

    Parameters
    ----------
    ax    : axes 
    xlab  : String to label xaxis
    ylab  : String to label yaxis
    title : String to title of the figure
    ylim  : tupple
    xlim  : tupple
    """
    ax.set_title(split_title(title), loc='left')
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
# END make_Zline_axis
####################################################################################################

####################################################################################################
def make_XY_axis(ax, xlab='Eastward km', ylab='Northward km', title = '', \
                 xlim=(-85, -56.4026), ylim=(20, 48.5974), \
                 xticks=[-85, -70.7013, -56.4026], xticklabels=['0', '1,590', '3,180'], \
                 yticks=[20, 34.2987, 48.5974], yticklabels=['0', '1,590', '3,180'], \
                 aspect='equal', \
                 **kwargs) : 
    """
    Add xlabel, ylabel and title to the axes ax and set it equal
    Usage : make_XY_axis(ax, xlab='Eastward km', ylab='Northward km', 
       title = '', xlim=(-85, -56.4026), ylim=(20, 48.5974),
       xticks=[-85, -70.7013, -56.4026], xticklabels=['0', '1,590',
       '3,180'], yticks=[20, 34.2987, 48.5974], yticklabels=['0',
       '1,590', '3,180'], **kwargs)

    Parameters
    ----------
    ax    : axes 
    xlab  : String to label xaxis
    ylab  : String to label yaxis
    xlim  : tupple
    ylim  : tupple
    title : String to title of the figure
    """
    print("> plotting_make_XY_axis")
    ax.set_aspect(aspect)   
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xticklabels)
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)
    ax.set_title(title, loc='left')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(split_title(title, maxword=20), loc='left')
# END make_XY_axis
####################################################################################################

####################################################################################################
def make_YZ_axis(ax, xlab='Northward km', ylab='Depth [m]', title = '', \
                 depmax=600, depmin=0, \
                 xlim=(20, 48.5974), \
                 xticks=[20, 34.2987, 48.5974], xticklabels=['0', '1,590', '3,180'], \
                 **kwargs) : 
    """
    Add xlabel, ylabel and title to the axes ax 
    Usage : make_YZ_axis(ax, xlab='Northward km', ylab='Depth [m]',
                 title = '', \ depmax=600, depmin=0, \ xlim=(20,
                 48.5974), \ xticks=[20, 34.2987, 48.5974],
                 xticklabels=['0', '1,590', '3,180'], \ **kwargs) :


    Parameters
    ----------
    ax     : axes 
    xlab   : String to label xaxis
    ylab   : String to label yaxis
    title  : String to title of the figure
    depmax : float defining depth max of yaxis
    depmin : float defining depth min of yaxis
    xlim   : tupple
    """
    print("> plotting_make_YZ_axis")
    ax.set_xlabel(xlab)
    ax.set_xlim(xlim)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xticklabels)
    ax.set_ylim(depmax, depmin)
    ax.set_ylabel(ylab)
    ax.set_title(split_title(title, maxword=20), loc='left')
# END make_YZ_axis
####################################################################################################

####################################################################################################
def make_XZ_axis(ax, xlab='Eastward km', ylab='Depth [meters]', title = '', \
                 depmax=600, depmin=0, \
                 xlim=(-85, -56.4026), \
                 xticks=[-85, -70.7013, -56.4026], xticklabels=['0', '1,590', '3,180'], \
                 **kwargs) : 
    """
    Add xlabel, ylabel and title to the axes ax 
    Usage : make_XZ_axis(ax, xlab='Eastward km', ylab='Depth
                 [meters]', title = '', \ depmax=600, depmin=0, \
                 xlim=(-85, -56.4026), \ xticks=[-85, -70.7013,
                 -56.4026], xticklabels=['0', '1,590', '3,180'], \
                 **kwargs) :


    Parameters
    ----------
    ax     : axes 
    xlab   : String to label xaxis
    ylab   : String to label yaxis
    title  : String to title of the figure
    depmax : float defining depth max of yaxis
    depmin : float defining depth min of yaxis
    xlim   : tupple
    """
    print("> plotting_make_XZ_axis")
    ax.set_xlabel(xlab)
    ax.set_xlim(xlim)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(xticklabels)
    ax.set_ylabel(ylab)
    ax.set_ylim(depmax, depmin)
    ax.set_title(split_title(title, maxword=20), loc='left')

# END make_XZ_axis
####################################################################################################

####################################################################################################
def make_TZ_axis(ax, xlab='Time', ylab='Depth [meters]', title = '', \
                     depmax=600, depmin=0, **kwargs) : 
    """
    Add xlabel, ylabel and title to the axes ax 
    Usage : make_TZ_axis(ax, xlab='Latitude', ylab='Depth [meters]', title = '', **kwargs)

    Parameters
    ----------
    ax     : axes 
    xlab   : String to label xaxis
    ylab   : String to label yaxis
    title  : String to title of the figure
    depmax : float defining depth max of yaxis
    depmin : float defining depth min of yaxis
    xlim   : tupple
    """
    print("> plotting_make_TZ_axis")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(split_title(title, maxword=20), loc='left')
    #ax.set_title(title)
    ax.set_ylim(depmax, depmin)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

# END make_TZ_axis
####################################################################################################

####################################################################################################
def timeseries_data(dat, lab=None, zfact=1., title = '', ytitle='',\
                    ylim=[None, None], \
                    savefig=None,\
                    suffig='functions_GYRE_timeseries_data_',\
                    dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) :
    """
    Plot time series
    Usage : timeseries_data(dat, lab=None, zfact=1.,
       title = '', ytitle='', ylim=[None, None], savefig=None,
       suffig='functions_GYRE_timeseries_data_r1r9r27_',
       dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs)

    Parameters
    ----------
    dat      : list of dictionaries 
       [{'data':list, 'date':list}, ...]
    lab      : list of string
    zfact    : integer
       to change unit of variable.
    title    : string
       title of the figures
    ytitle   : string
       add title to y axis
    ylim     : tupple or list of 2 element 
       set the lower and upper bounds of y axis
    savefig  : String to save the figure
    suffig   : suffix to saved figure name
    dirfig   : figure folder   
    **kwargs : not passed
    """
    
    print("> plotting_timeseries_data")
    
    nseries = len(dat)
    zwdata = []
    zwdate = []
    for ind, val in np.ndenumerate(dat) : 
        zwdata.append(val['data'] * zfact)
        zwdate.append(extra.cdtime2str(val['date']))
    # endfor

    # __________________
    # PLOT

    infact  = 1/2.54
    fsize   = (8*infact, 4*infact) #(width, height)
    fig, ax = plt.subplots(1, 1, figsize = fsize)
    
    col = ['gray', 'royalblue', 'firebrick', 'green', 'orchid', 'gold']
    if nseries > len(col) : raise IndexError("ERROR, too many series of data")
    if lab == None : 
        lab = []
        for ind in range(nseries) : lab.append('V'+str(ind))
    #
    if nseries > len(lab) : raise IndexError("ERROR, not enough labels")

    for ind in range(nseries) : 
        ldat  = ax.plot(zwdate[ind], zwdata[ind], col[ind], label = lab[ind])
    #
    make_timeseries_axis(ax, title=title, ylab=ytitle, ylim=ylim, xlab='')
    ax.tick_params(bottom=False)
    ax.legend( loc = 'upper left', bbox_to_anchor = (1, 1) )

        
    if savefig != None : 
        print("Figure saved: ", dirfig+suffig+savefig)
        fig.savefig(dirfig+suffig+savefig)
        plt.close()
 
    
# END timeseries_data
####################################################################################################

####################################################################################################
def map_data_xy(field, mesh, *args, grid='T', savefig=None, cbarformat=None, figtitle='', \
                    dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', **kwargs) : 
    """
    Plot 2D horizontal field
    Usage : map_data_xy(field, mesh, *args, grid='T', savefig=None,
         dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/',
         cbarformat=None, figtitle='', **kwargs)

    Parameters
    ----------
    field : 2D array
    mesh  : mesh mask from read_mesh
    grid  : 'T' (default), 'U' or 'V'
    figtitle : to write the title of the figure
    savefig : String to save the figure
    dirfig  : figure folder
    *args can be : a list of level (np.arange(0, 20,1)), an integer...
    **kwargs can be : cmap, norm, extend...
    *args and **kwargs are pass to contourf function. See contourf
     function for details.
    """
    print("> plotting_map_data_xy")
    cf = plt.contourf(mesh['lon'+grid], mesh['lat'+grid], field, *args, **kwargs)
    make_XY_axis(cf.ax, title = figtitle)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(cf.ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cf, format=cbarformat, cax=cax)
    if savefig != None : 
        print("Figure saved: ", dirfig+savefig)
        plt.savefig(dirfig + savefig)
        plt.close()

    #
# END map_data_xy
####################################################################################################

####################################################################################################
def split_title(text, maxword=5):
    print("> plotting_split_title")
    words = text.split(" ")
    total_string = ""
    for counter, word in enumerate(words):
        if counter>0 and counter % maxword == 0:
            total_string +="\n{}".format(word)
        else:
            total_string +=" {}".format(word)
    return total_string.lstrip()
#
####################################################################################################

      
