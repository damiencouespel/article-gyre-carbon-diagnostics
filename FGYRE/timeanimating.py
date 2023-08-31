"""
Functions to plot animated fields
"""

from IPython import embed
# import time
import numpy        as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from FGYRE import averaging, extra, plotting

##################################################
#                MAP DELTA R1R9R27               #
##################################################
def map_delta_r1r9r27(dat1, dat9, dat27, tlab, \
                      grid='T', zfact=1., cbtitle='', \
                      lev=None, ext=None, CTLcmap = 'viridis', \
                      levD=None, extD=None, Dcmap = 'RdBu_r', \
                      savefig=None,\
                      suffig='fgyre_timeanimating_map_delta_r1r9r27_',\
                      dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', \
                      **kwargs) :
    """
    Animate 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage: map_delta_r1r9r27(dat1, dat9, dat27, tlab, \
                grid='T', zfact=1., cbtitle='', \
                lev=None, ext=None, CTLcmap = 'viridis', \
                levD=None, extD=None, Dcmap = 'RdBu_r', \
                savefig=None,\
                suffig='fgyre_timeanimating_map_delta_r1r9r27_',\
                dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', \
                **kwargs)

    Parameters
    ----------
    dat1, dat9, dat27 : dictionary {'CTL':txyarray, 'CC':txyarray,
       'mesh':mesh dictionary). time dimension of same length as tlab 
    tlab        : liste of string giving the time label
    grid        : 'T' (default), 'U' or 'V'
    zfact       : integer
       to change unit of variable.
    cbtitle     : string
       add title to colorbars
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    CTLcmap     : string defining the cmap
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    Dcmap       : string defining the cmap
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_XY_axis : xlim, ylim
    """
    
    print("> timeanimating_map_delta_r1r9r27")
    
    #---------------------------------------
    # 1) INITIALISATION DE LA FIGURE AVEC SUBPLOTS
    #---------------------------------------

    #___________________
    # CREATE FIG AND AXES

    infact        = 1/2.54
    figsize = (15*infact, 17*infact) #(width, height)
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsize)
    fig.subplots_adjust(bottom=0.16, top=0.86)
    
    #___________________
    # SET LEV AND LEVD IF NOT GIVEN
    # from CTL1 and CC1
    
    zwctl1  = np.nanmean(zfact * dat1['CTL'], axis=0) # temp mean on CTL1
    zwcc1  = np.nanmean(zfact * dat1['CC'], axis=0) # temp mean on CC
    dzwr1  = zwcc1  - zwctl1
    if isinstance(lev, type(None)) : 
        zw = np.quantile(zwctl1, .10)
        if zw != 0 :
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmin = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        zw = np.quantile(zwctl1, .90)
        if zw != 0 : 
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #

    #___________________
    # SET EXT AND EXTD
    
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

    #___________________
    # INITIATE PLOT

    X1, Y1, Zct1, Zcc1, Zdd1 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct1 = ax[0, 0].contourf(X1, Y1, Zct1, lev , cmap=CTLcmap, extend=ext)
    cfcc1 = ax[1, 0].contourf(X1, Y1, Zcc1, lev , cmap=CTLcmap, extend=ext)
    cfdd1 = ax[2, 0].contourf(X1, Y1, Zdd1, levD, cmap=Dcmap  , extend=extD)
    X9, Y9, Zct9, Zcc9, Zdd9 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct9 = ax[0, 1].contourf(X9, Y9, Zct9, lev , cmap=CTLcmap, extend=ext)
    cfcc9 = ax[1, 1].contourf(X9, Y9, Zcc9, lev , cmap=CTLcmap, extend=ext)
    cfdd9 = ax[2, 1].contourf(X9, Y9, Zdd9, levD, cmap=Dcmap  , extend=extD)
    X27, Y27, Zct27, Zcc27, Zdd27 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct27 = ax[0, 1].contourf(X27, Y27, Zct27, lev , cmap=CTLcmap, extend=ext)
    cfcc27 = ax[1, 1].contourf(X27, Y27, Zcc27, lev , cmap=CTLcmap, extend=ext)
    cfdd27 = ax[2, 1].contourf(X27, Y27, Zdd27, levD, cmap=Dcmap  , extend=extD)
    
    #___________________
    # SET AXES
    
    plotting.make_XY_axis(ax[0, 0], title='a) CTL1'      , xlab='', **kwargs)
    plotting.make_XY_axis(ax[1, 0], title='b) CC1'       , xlab='', **kwargs)
    plotting.make_XY_axis(ax[2, 0], title='c) CC1 - CTL1'         )
    plotting.make_XY_axis(ax[0, 1], title='d) CTL9'      , xlab='', ylab='', **kwargs)
    plotting.make_XY_axis(ax[1, 1], title='e) CC9'       , xlab='', ylab='', **kwargs)
    plotting.make_XY_axis(ax[2, 1], title='f) CC9 - CTL9'         , ylab='', **kwargs)
    plotting.make_XY_axis(ax[0, 2], title='g) CTL27'     , xlab='', ylab='', **kwargs)
    plotting.make_XY_axis(ax[1, 2], title='h) CC27'      , xlab='', ylab='', **kwargs)
    plotting.make_XY_axis(ax[2, 2], title='i) CC27 - CTL27'       , ylab='', **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    #___________________
    # COLORBARS
    
    cbar_ax = fig.add_axes([0.17, 0.9, 0.6588, 0.0273])
    fig.colorbar(cfct1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                 label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, 0.07, 0.6588, 0.0273])
    fig.colorbar(cfdd1, cax=cbar_ax, orientation='horizontal', label=cbtitle, format='%.2e')

    #___________________
    # TIME COUNTER
    axtxt = fig.add_axes([0.05, 0.95, 0.05, 0.05])
    axtxt.axis('off')
    label = axtxt.text(.5, .5, '0000-00-00', ha='center', va='top')


    #---------------------------------------
    # 2) Function updating the data
    #---------------------------------------

    def updateplot(ttt, tlab, dat1, dat9, dat27) :

        label.set_text(tlab[ttt])

        zwct1  = zfact * dat1['CTL'][ttt] 
        zwcc1  = zfact * dat1['CC'][ttt]
        zwct9  = zfact * dat9['CTL'][ttt]
        zwcc9  = zfact * dat9['CC'][ttt]
        zwct27 = zfact * dat27['CTL'][ttt]
        zwcc27 = zfact * dat27['CC'][ttt]
        dzwr1  = zwcc1  - zwct1
        dzwr9  = zwcc9  - zwct9
        dzwr27 = zwcc27 - zwct27
        mesh1   = dat1['mesh']
        mesh9   = dat9['mesh']
        mesh27  = dat27['mesh']

        for zax in ax.flatten() : zax.collections = []
        
        ax[0, 0].contourf(mesh1['lon'+grid] , mesh1['lat'+grid] ,  zwct1, lev , cmap=CTLcmap, extend=ext )
        ax[1, 0].contourf(mesh1['lon'+grid] , mesh1['lat'+grid] ,  zwcc1, lev , cmap=CTLcmap, extend=ext )
        ax[2, 0].contourf(mesh1['lon'+grid] , mesh1['lat'+grid] ,  dzwr1, levD, cmap=Dcmap  , extend=extD)
        ax[0, 1].contourf(mesh9['lon'+grid] , mesh9['lat'+grid] ,  zwct9, lev , cmap=CTLcmap, extend=ext )
        ax[1, 1].contourf(mesh9['lon'+grid] , mesh9['lat'+grid] ,  zwcc9, lev , cmap=CTLcmap, extend=ext )
        ax[2, 1].contourf(mesh9['lon'+grid] , mesh9['lat'+grid] ,  dzwr9, levD, cmap=Dcmap  , extend=extD)
        ax[0, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid], zwct27, lev , cmap=CTLcmap, extend=ext )
        ax[1, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid], zwcc27, lev , cmap=CTLcmap, extend=ext )
        ax[2, 2].contourf(mesh27['lon'+grid], mesh27['lat'+grid], dzwr27, levD, cmap=Dcmap  , extend=extD)
    #

    #---------------------------------------
    # 3) ANIMATE 
    #---------------------------------------

    nframes = len(tlab)
    anim = animation.FuncAnimation(fig, updateplot, frames=nframes, fargs=(tlab, dat1, dat9, dat27))
    
    #---------------------------------------
    # 4) SAVE
    #---------------------------------------
        

    if savefig != None : 
        anim.save(dirfig+suffig+savefig, writer='imagemagick')
        print("Figure saved: ", dirfig+suffig+savefig)
    #
        
##################################################
#              END MAP DELTA R1R9R27             #
##################################################

##################################################
#       MERIDIONAL SECTION DELTA R1R9R27         #
##################################################
def meridional_section_delta_r1r9r27(dat1, dat9, dat27, tlab, \
                                     grid='T', zfact=1., cbtitle='', \
                                     lev=None, ext=None, CTLcmap = 'viridis', \
                                     levD=None, extD=None, Dcmap = 'RdBu_r', \
                                     savefig=None,\
                                     suffig='fgyre_timeanimating_meridional_section_delta_r1r9r27_',\
                                     dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', \
                                     **kwargs) :
    """
    Plot 9 animeted sections :
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
    dat1, dat9, dat27 : dictionary {'CTL':txyarray, 'CC':txyarray,
       'mesh':mesh dictionary). time dimension of same length as tlab 
    tlab        : liste of string giving the time label
    grid        : 'T' (default), 'U' or 'V'
    zfact       : integer
       to change unit of variable.
    cbtitle     : string
       add title to colorbars
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    CTLcmap     : string defining the cmap
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    Dcmap       : string defining the cmap
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_XY_axis: xlim, ylim
    """ 

    print("> timeanimating_meridional_section_delta_r1r9r27")
    
    #---------------------------------------
    # 1) INITIALISATION DE LA FIGURE AVEC SUBPLOTS
    #---------------------------------------

    #___________________
    # CREATE FIG AND AXES

    infact        = 1/2.54
    figsize = (15*infact, 12*infact) #(width, height)
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsize)
    fig.subplots_adjust(bottom=0.23, top=0.82)

    #___________________
    # SET LEV AND LEVD IF NOT GIVEN
    # from CTL1 and CC1
    
    zwctl1  = np.nanmean(zfact * dat1['CTL'], axis=0) # temp mean on CTL1
    zwcc1  = np.nanmean(zfact * dat1['CC'], axis=0) # temp mean on CC
    dzwr1  = zwcc1  - zwctl1
    if isinstance(lev, type(None)) : 
        zw = np.quantile(zwctl1, .10)
        if zw != 0 :
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmin = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        zw = np.quantile(zwctl1, .90)
        if zw != 0 : 
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #

    #___________________
    # SET EXT AND EXTD
    
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

    #___________________
    # INITIATE PLOT

    X1, Y1, Zct1, Zcc1, Zdd1 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct1 = ax[0, 0].contourf(X1, Y1, Zct1, lev , cmap=CTLcmap, extend=ext)
    cfcc1 = ax[1, 0].contourf(X1, Y1, Zcc1, lev , cmap=CTLcmap, extend=ext)
    cfdd1 = ax[2, 0].contourf(X1, Y1, Zdd1, levD, cmap=Dcmap  , extend=extD)
    X9, Y9, Zct9, Zcc9, Zdd9 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct9 = ax[0, 1].contourf(X9, Y9, Zct9, lev , cmap=CTLcmap, extend=ext)
    cfcc9 = ax[1, 1].contourf(X9, Y9, Zcc9, lev , cmap=CTLcmap, extend=ext)
    cfdd9 = ax[2, 1].contourf(X9, Y9, Zdd9, levD, cmap=Dcmap  , extend=extD)
    X27, Y27, Zct27, Zcc27, Zdd27 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct27 = ax[0, 1].contourf(X27, Y27, Zct27, lev , cmap=CTLcmap, extend=ext)
    cfcc27 = ax[1, 1].contourf(X27, Y27, Zcc27, lev , cmap=CTLcmap, extend=ext)
    cfdd27 = ax[2, 1].contourf(X27, Y27, Zdd27, levD, cmap=Dcmap  , extend=extD)
    
    #___________________
    # SET AXES
    
    plotting.make_YZ_axis(ax[0, 0], title='a) CTL1'        , xlab=''         , **kwargs)
    plotting.make_YZ_axis(ax[1, 0], title='b) CC1'         , xlab=''         , **kwargs)
    plotting.make_YZ_axis(ax[2, 0], title='c) CC1 - CTL1'                    , **kwargs)
    plotting.make_YZ_axis(ax[0, 1], title='d) CTL9'        , xlab='', ylab='', **kwargs)
    plotting.make_YZ_axis(ax[1, 1], title='e) CC9'         , xlab='', ylab='', **kwargs)
    plotting.make_YZ_axis(ax[2, 1], title='f) CC9 - CTL9'           , ylab='', **kwargs)
    plotting.make_YZ_axis(ax[0, 2], title='g) CTL27'       , xlab='', ylab='', **kwargs)
    plotting.make_YZ_axis(ax[1, 2], title='h) CC27'        , xlab='', ylab='', **kwargs)
    plotting.make_YZ_axis(ax[2, 2], title='i) CC27 - CTL27'         , ylab='', **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    #___________________
    # COLORBARS

    cbar_ax = fig.add_axes([0.17, 0.88, 0.6588, 0.0386])
    fig.colorbar(cfct1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                 label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, 0.09, 0.6588, 0.0386])
    fig.colorbar(cfdd1, cax=cbar_ax, orientation='horizontal', label=cbtitle, format='%.2e')

    #___________________
    # TIME COUNTER
    axtxt = fig.add_axes([0.05, 0.95, 0.05, 0.05])
    axtxt.axis('off')
    label = axtxt.text(.5, 0.5, '0000-00-00', ha='center', va='center', fontsize=8)


    #---------------------------------------
    # 2) Function updating the data
    #---------------------------------------

    def updateplot(ttt, tlab, dat1, dat9, dat27) :

        label.set_text(tlab[ttt])

        zwct1  = zfact * dat1['CTL'][ttt] 
        zwcc1  = zfact * dat1['CC'][ttt]
        zwct9  = zfact * dat9['CTL'][ttt]
        zwcc9  = zfact * dat9['CC'][ttt]
        zwct27 = zfact * dat27['CTL'][ttt]
        zwcc27 = zfact * dat27['CC'][ttt]
        dzwr1  = zwcc1  - zwct1
        dzwr9  = zwcc9  - zwct9
        dzwr27 = zwcc27 - zwct27
        mesh1   = dat1['mesh']
        mesh9   = dat9['mesh']
        mesh27  = dat27['mesh']

        for zax in ax.flatten() : zax.collections = []
        
        ax[0, 0].contourf(mesh1['lat'+grid][:, 5] , mesh1['dep'+grid] ,  zwct1, lev , cmap=CTLcmap, extend=ext )
        ax[1, 0].contourf(mesh1['lat'+grid][:, 5] , mesh1['dep'+grid] ,  zwcc1, lev , cmap=CTLcmap, extend=ext )
        ax[2, 0].contourf(mesh1['lat'+grid][:, 5] , mesh1['dep'+grid] ,  dzwr1, levD, cmap=Dcmap  , extend=extD)
        ax[0, 1].contourf(mesh9['lat'+grid][:, 5] , mesh9['dep'+grid] ,  zwct9, lev , cmap=CTLcmap, extend=ext )
        ax[1, 1].contourf(mesh9['lat'+grid][:, 5] , mesh9['dep'+grid] ,  zwcc9, lev , cmap=CTLcmap, extend=ext )
        ax[2, 1].contourf(mesh9['lat'+grid][:, 5] , mesh9['dep'+grid] ,  dzwr9, levD, cmap=Dcmap  , extend=extD)
        ax[0, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid], zwct27, lev , cmap=CTLcmap, extend=ext )
        ax[1, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid], zwcc27, lev , cmap=CTLcmap, extend=ext )
        ax[2, 2].contourf(mesh27['lat'+grid][:, 5], mesh27['dep'+grid], dzwr27, levD, cmap=Dcmap  , extend=extD)
    #

    #---------------------------------------
    # 3) ANIMATE 
    #---------------------------------------

    nframes = len(tlab)
    anim = animation.FuncAnimation(fig, updateplot, frames=nframes, fargs=(tlab, dat1, dat9, dat27))
    
    #---------------------------------------
    # 4) SAVE
    #---------------------------------------
        

    if savefig != None : 
        anim.save(dirfig+suffig+savefig, writer='imagemagick')
        print("Figure saved: ", dirfig+suffig+savefig)
    #

##################################################
#     END MERIDIONAL SECTION DELTA R1R9R27       #
##################################################


##################################################
#       ZONAL SECTION DELTA R1R9R27         #
##################################################
def zonal_section_delta_r1r9r27(dat1, dat9, dat27, tlab, \
                                grid='T', zfact=1., cbtitle='', \
                                lev=None, ext=None, CTLcmap = 'viridis', \
                                levD=None, extD=None, Dcmap = 'RdBu_r', \
                                savefig=None,\
                                suffig='fgyre_timeanimating_zonal_section_delta_r1r9r27_',\
                                dirfig='/gpfswork/rech/dyk/rdyk004/MY_FIG_PYTHON3/', \
                                **kwargs) :
    """
    Plot 9 animeted sections :
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
    dat1, dat9, dat27 : dictionary {'CTL':txyarray, 'CC':txyarray,
       'mesh':mesh dictionary). time dimension of same length as tlab 
    tlab        : liste of string giving the time label
    grid        : 'T' (default), 'U' or 'V'
    zfact       : integer
       to change unit of variable.
    cbtitle     : string
       add title to colorbars
    lev         : list or integer
       pass to contourf to defines levels
       if None (default), it is set automatically from median and
       standard deviation of CTL1
    ext         : None (default) 'both' 'max' or 'min'
       define extend of colorbar. If None, it is define according to
       the levels
    CTLcmap     : string defining the cmap
    levD        : list or integer
       as lev but for the difference between CC and CTL
    extD        : 'both' (default) 'max' or 'min'
       as lev but for the difference between CC and CTL
    Dcmap       : string defining the cmap
    savefig     : String to save the figure
    suffig      : suffix to saved figure name
    dirfig      : figure folder   
    **kwargs    : pass to make_XY_axis: xlim, ylim
    """ 

    print("> timeanimating_zonal_section_delta_r1r9r27")
    
    #---------------------------------------
    # 1) INITIALISATION DE LA FIGURE AVEC SUBPLOTS
    #---------------------------------------

    #___________________
    # CREATE FIG AND AXES

    infact        = 1/2.54
    figsize = (15*infact, 12*infact) #(width, height)
    fig, ax = plt.subplots(3, 3, sharey = 'row', sharex  = 'col', figsize = figsize)
    fig.subplots_adjust(bottom=0.23, top=0.82)

    #___________________
    # SET LEV AND LEVD IF NOT GIVEN
    # from CTL1 and CC1
    
    zwctl1  = np.nanmean(zfact * dat1['CTL'], axis=0) # temp mean on CTL1
    zwcc1  = np.nanmean(zfact * dat1['CC'], axis=0) # temp mean on CC
    dzwr1  = zwcc1  - zwctl1
    if isinstance(lev, type(None)) : 
        zw = np.quantile(zwctl1, .10)
        if zw != 0 :
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmin = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmin = 0
        zw = np.quantile(zwctl1, .90)
        if zw != 0 : 
            zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
            levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        else : levmax = 0
        lev = np.linspace(levmin, levmax, 11)
    #
    if isinstance(levD, type(None)) : 
        zwq1 = np.quantile(dzwr1, .10)
        zwq9 = np.quantile(dzwr1, .90)
        zw = np.max( [np.abs(zwq1), np.abs(zwq9)] )
        zw1 = np.floor(np.log10(np.abs(zw))) # order of magnitude
        levmax = np.around(zw * 10**(-zw1)) * 10**zw1
        aa = np.linspace(-levmax, 0, 6)
        bb = np.linspace(0, levmax, 6)
        levD = np.concatenate((aa[:-1], bb))
    #

    #___________________
    # SET EXT AND EXTD
    
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

    #___________________
    # INITIATE PLOT

    X1, Y1, Zct1, Zcc1, Zdd1 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct1 = ax[0, 0].contourf(X1, Y1, Zct1, lev , cmap=CTLcmap, extend=ext)
    cfcc1 = ax[1, 0].contourf(X1, Y1, Zcc1, lev , cmap=CTLcmap, extend=ext)
    cfdd1 = ax[2, 0].contourf(X1, Y1, Zdd1, levD, cmap=Dcmap  , extend=extD)
    X9, Y9, Zct9, Zcc9, Zdd9 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct9 = ax[0, 1].contourf(X9, Y9, Zct9, lev , cmap=CTLcmap, extend=ext)
    cfcc9 = ax[1, 1].contourf(X9, Y9, Zcc9, lev , cmap=CTLcmap, extend=ext)
    cfdd9 = ax[2, 1].contourf(X9, Y9, Zdd9, levD, cmap=Dcmap  , extend=extD)
    X27, Y27, Zct27, Zcc27, Zdd27 = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    cfct27 = ax[0, 1].contourf(X27, Y27, Zct27, lev , cmap=CTLcmap, extend=ext)
    cfcc27 = ax[1, 1].contourf(X27, Y27, Zcc27, lev , cmap=CTLcmap, extend=ext)
    cfdd27 = ax[2, 1].contourf(X27, Y27, Zdd27, levD, cmap=Dcmap  , extend=extD)
    
    #___________________
    # SET AXES
    
    plotting.make_XZ_axis(ax[0, 0], title='a) CTL1'        , xlab=''         , **kwargs)
    plotting.make_XZ_axis(ax[1, 0], title='b) CC1'         , xlab=''         , **kwargs)
    plotting.make_XZ_axis(ax[2, 0], title='c) CC1 - CTL1'                    , **kwargs)
    plotting.make_XZ_axis(ax[0, 1], title='d) CTL9'        , xlab='', ylab='', **kwargs)
    plotting.make_XZ_axis(ax[1, 1], title='e) CC9'         , xlab='', ylab='', **kwargs)
    plotting.make_XZ_axis(ax[2, 1], title='f) CC9 - CTL9'           , ylab='', **kwargs)
    plotting.make_XZ_axis(ax[0, 2], title='g) CTL27'       , xlab='', ylab='', **kwargs)
    plotting.make_XZ_axis(ax[1, 2], title='h) CC27'        , xlab='', ylab='', **kwargs)
    plotting.make_XZ_axis(ax[2, 2], title='i) CC27 - CTL27'         , ylab='', **kwargs)
    
    ax[0, 0].tick_params(bottom=False)
    ax[1, 0].tick_params(bottom=False)
    ax[0, 1].tick_params(bottom=False, left=False)
    ax[1, 1].tick_params(bottom=False, left=False)
    ax[2, 1].tick_params(left=False)
    ax[0, 2].tick_params(left=False, bottom=False)
    ax[1, 2].tick_params(left=False, bottom=False)
    ax[2, 2].tick_params(left=False)

    #___________________
    # COLORBARS

    cbar_ax = fig.add_axes([0.17, 0.88, 0.6588, 0.0386])
    fig.colorbar(cfct1, cax=cbar_ax, orientation='horizontal', ticklocation='top',\
                 label=cbtitle, format='%.2e')
    cbar_ax = fig.add_axes([0.17, 0.09, 0.6588, 0.0386])
    fig.colorbar(cfdd1, cax=cbar_ax, orientation='horizontal', label=cbtitle, format='%.2e')

    #___________________
    # TIME COUNTER
    axtxt = fig.add_axes([0.05, 0.95, 0.05, 0.05])
    axtxt.axis('off')
    label = axtxt.text(.5, 0.5, '0000-00-00', ha='center', va='center', fontsize=8)


    #---------------------------------------
    # 2) Function updating the data
    #---------------------------------------

    def updateplot(ttt, tlab, dat1, dat9, dat27) :

        label.set_text(tlab[ttt])

        zwct1  = zfact * dat1['CTL'][ttt] 
        zwcc1  = zfact * dat1['CC'][ttt]
        zwct9  = zfact * dat9['CTL'][ttt]
        zwcc9  = zfact * dat9['CC'][ttt]
        zwct27 = zfact * dat27['CTL'][ttt]
        zwcc27 = zfact * dat27['CC'][ttt]
        dzwr1  = zwcc1  - zwct1
        dzwr9  = zwcc9  - zwct9
        dzwr27 = zwcc27 - zwct27
        mesh1   = dat1['mesh']
        mesh9   = dat9['mesh']
        mesh27  = dat27['mesh']

        for zax in ax.flatten() : zax.collections = []
        
        ax[0, 0].contourf(mesh1['lon'+grid][5, :] , mesh1['dep'+grid] ,  zwct1, lev , cmap=CTLcmap, extend=ext )
        ax[1, 0].contourf(mesh1['lon'+grid][5, :] , mesh1['dep'+grid] ,  zwcc1, lev , cmap=CTLcmap, extend=ext )
        ax[2, 0].contourf(mesh1['lon'+grid][5, :] , mesh1['dep'+grid] ,  dzwr1, levD, cmap=Dcmap  , extend=extD)
        ax[0, 1].contourf(mesh9['lon'+grid][5, :] , mesh9['dep'+grid] ,  zwct9, lev , cmap=CTLcmap, extend=ext )
        ax[1, 1].contourf(mesh9['lon'+grid][5, :] , mesh9['dep'+grid] ,  zwcc9, lev , cmap=CTLcmap, extend=ext )
        ax[2, 1].contourf(mesh9['lon'+grid][5, :] , mesh9['dep'+grid] ,  dzwr9, levD, cmap=Dcmap  , extend=extD)
        ax[0, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid], zwct27, lev , cmap=CTLcmap, extend=ext )
        ax[1, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid], zwcc27, lev , cmap=CTLcmap, extend=ext )
        ax[2, 2].contourf(mesh27['lon'+grid][5, :], mesh27['dep'+grid], dzwr27, levD, cmap=Dcmap  , extend=extD)
    #

    #---------------------------------------
    # 3) ANIMATE 
    #---------------------------------------

    nframes = len(tlab)
    anim = animation.FuncAnimation(fig, updateplot, frames=nframes, fargs=(tlab, dat1, dat9, dat27))
    
    #---------------------------------------
    # 4) SAVE
    #---------------------------------------
        

    if savefig != None : 
        anim.save(dirfig+suffig+savefig, writer='imagemagick')
        print("Figure saved: ", dirfig+suffig+savefig)
    #

##################################################
#     END ZONAL SECTION DELTA R1R9R27       #
##################################################

