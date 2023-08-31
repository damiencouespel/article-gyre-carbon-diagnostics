"""
Functions to read and animate fields
"""

from IPython import embed
# import time
import numpy        as np
from FGYRE import reading, averaging, interpolating, timeanimating

##################################################
#      READ AND TIMEANIM MAP DELTA R1R9R27       #
##################################################
def read_and_timeanim_map_delta_r1r9r27(vname, file_suffix, time, \
                                        Zaxis=None, timelab=None,  \
                                        fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                        suffig='fgyre_read_and_timeanim_map_delta_r1r9r27_', \
                                        **kwargs) :

    """
    Read a variable and plot 9 animated maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage: read_and_timeanim_map_delta_r1r9r27(vname, file_suffix, time, \
                Zaxis = None, timelab=None,  \
                fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                suffig='fgyre_read_and_timeanim_map_delta_r1r9r27_', \
                **kwargs) :

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : list of string tupple ultimately passed
       to reading.read_ncdf. Time average between the tupple members. 
       Define the number of frames
       eg. [('0101-01-01', '0110-12-31'), ('0161-01-01', '0170-12-31')]
    Zaxis        : None (default), 'interp' or 'mean' or 'integrate'
       To indicate what we do with Z axis
    timelab     : list of string giving the time label. 
       If None, define from time. Has to be same length as time
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.zmean, interpolating.zinterpol or timeanimating_map_delta_r1r9r27 eg:
       zmin=100, zmax=500, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.gif', integral=True
    """
    
    print("> read_and_timeanim_map_delta_r1r9r27")

    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc'  # TEST
        zwmesh = reading.read_mesh(zwmeshfile)
        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zw3 = []
            tlab = []
            for it, vt in enumerate(time) :
                if timelab==None : tlab.append(vt[0]+'\nto\n'+vt[1])
                else             : tlab.append(timelab[it])
                zwfilename = fdir + vkS + vkR + file_suffix
                # zwfilename = fdir + vkS + '1' + file_suffix # TEST
                zw  = reading.read_ncdf(vname, zwfilename, time=vt)
                #____________________
                # TEMPORAL MEAN
                zw2 = averaging.tmean(zw['data'], zwmesh, keep_ndim=True, **kwargs)
                #____________________
                # VERTICAL AXIS
                if Zaxis=='mean' : 
                    if 'dept' in zw.keys() : zw2 = averaging.zmean(zw2, zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Zaxis should be None maybe ?")
                elif Zaxis=='integrate' : 
                    if 'dept' in zw.keys() : zw2 = averaging.zmean(zw2, zwmesh, keep_ndim=True, integral=True, **kwargs)
                    else : raise ValueError("Zaxis should be None maybe ?")
                elif Zaxis=='interp' : 
                    if 'idep' in kwargs.keys() : 
                        if 'dept' in zw.keys() : zw2  = interpolating.zinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                        else : raise ValueError("Zaxis should be None maybe ?")        
                    else : raise ValueError("idep not defined")        
                elif Zaxis == None : 
                    if 'dept' in  zw.keys() : raise ValueError("Zaxis should be 'mean' or 'interp' maybe ?")        
                else : raise ValueError("Zaxis should be 'mean', 'interp' or None")        
                zw3.append(zw2.squeeze())
            #
            zwouS[vkS] = np.array(zw3)
        #
        zwouR['R'+vkR] = zwouS
    #
    timeanimating.map_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], tlab, suffig=suffig, **kwargs)
##################################################
#    END READ AND TIMEANIM MAP DELTA R1R9R27     #
##################################################

############################################################
#    READ AND TIMEANIM MERIDIONAL SECTION DELTA R1R9R27    #
############################################################
def read_and_timeanim_meridional_section_delta_r1r9r27(vname, file_suffix, time, \
                                        Xaxis=None, timelab=None,  \
                                        fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                        suffig='fgyre_read_and_timeanim_meridional_section_delta_r1r9r27_', \
                                        **kwargs) :

    """
    Read a variable and plot 9 animated meridional sections :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage: read_and_timeanim_meridional_section_delta_r1r9r27(vname, file_suffix, time, \
                Xaxis = None, timelab=None,  \
                fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                suffig='fgyre_read_and_timeanim_meridional_section_delta_r1r9r27_', \
                **kwargs) :

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : list of string tupple ultimately passed
       to reading.read_ncdf. Time average between the tupple members. 
       Define the number of frames
       eg. [('0101-01-01', '0110-12-31'), ('0161-01-01', '0170-12-31')]
    Xaxis        : None (default), 'interp' or 'mean' or 'integrate'
       To indicate what we do with X axis
    timelab     : list of string giving the time label. 
       If None, define from time. Has to be same length as time
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.xmean, interpolating.xinterpol or timeanimating_meridional_section_delta_r1r9r27 eg:
       xmin=100, xmax=500, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.gif', integral=True
    """
    
    print("> read_and_timeanim_meridional_section_delta_r1r9r27")

    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc'  # TEST
        zwmesh = reading.read_mesh(zwmeshfile)
        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zw3 = []
            tlab = []
            for it, vt in enumerate(time) :
                if timelab==None : tlab.append(vt[0]+'\nto\n'+vt[1])
                else             : tlab.append(timelab[it])
                zwfilename = fdir + vkS + vkR + file_suffix
                # zwfilename = fdir + vkS + '1' + file_suffix # TEST
                zw  = reading.read_ncdf(vname, zwfilename, time=vt)
                #____________________
                # TEMPORAL MEAN
                zw2 = averaging.tmean(zw['data'], zwmesh, keep_ndim=True, **kwargs)
                #____________________
                # ZONAL AXIS
                if Xaxis=='mean' : 
                    if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Xaxis should be None maybe ?")
                elif Xaxis=='integrate' : 
                    if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, integral = True, keep_ndim=True, **kwargs)
                    else : raise ValueError("Xaxis should be None maybe ?")
                elif Xaxis=='interp' : 
                    if 'ilon' in kwargs.keys() : 
                        if 'long' in zw.keys() : zw2  = interpolating.xinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                        else : raise ValueError("Xaxis should be None maybe ?")        
                    else : raise ValueError("ilon not defined")        
                elif Xaxis == None : 
                    if 'long' in  zw.keys() : raise ValueError("Xaxis should be 'mean' or 'interp' maybe ?")        
                else : raise ValueError("Xaxis should be 'mean', 'interp' or None")        
                zw3.append(zw2.squeeze())
            #
            zwouS[vkS] = np.array(zw3)
        #
        zwouR['R'+vkR] = zwouS
    #
    timeanimating.meridional_section_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], tlab, suffig=suffig, **kwargs)
############################################################
#  END READ AND TIMEANIM MERIDIONAL_SECTION DELTA R1R9R27  #
############################################################

############################################################
#    READ AND TIMEANIM ZONAL SECTION DELTA R1R9R27    #
############################################################
def read_and_timeanim_zonal_section_delta_r1r9r27(vname, file_suffix, time, \
                                                  Yaxis=None, timelab=None,  \
                                                  fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                                  suffig='fgyre_read_and_timeanim_zonal_section_delta_r1r9r27_', \
                                                  **kwargs) :

    """
    Read a variable and plot 9 animated zonal sections :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage: read_and_timeanim_zonal_section_delta_r1r9r27(vname, file_suffix, time, \
                Yaxis = None, timelab=None,  \
                fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                suffig='fgyre_read_and_timeanim_zonal_section_delta_r1r9r27_', \
                **kwargs) :

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : list of string tupple ultimately passed
       to reading.read_ncdf. Time average between the tupple members. 
       Define the number of frames
       eg. [('0101-01-01', '0110-12-31'), ('0161-01-01', '0170-12-31')]
    Yaxis        : None (default), 'interp' or 'mean' or 'integrate'
       To indicate what we do with Y axis
    timelab     : list of string giving the time label. 
       If None, define from time. Has to be same length as time
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.ymean, interpolating.yinterpol or timeanimating_zonal_section_delta_r1r9r27 eg:
       ymin=100, ymax=500, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.gif', integral=True
    """
    
    print("> read_and_timeanim_zonal_section_delta_r1r9r27")

    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc'  # TEST
        zwmesh = reading.read_mesh(zwmeshfile)
        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zw3 = []
            tlab = []
            for it, vt in enumerate(time) :
                if timelab==None : tlab.append(vt[0]+'\nto\n'+vt[1])
                else             : tlab.append(timelab[it])
                zwfilename = fdir + vkS + vkR + file_suffix
                # zwfilename = fdir + vkS + '1' + file_suffix # TEST
                zw  = reading.read_ncdf(vname, zwfilename, time=vt)
                #____________________
                # TEMPORAL MEAN
                zw2 = averaging.tmean(zw['data'], zwmesh, keep_ndim=True, **kwargs)
                #____________________
                # ZONAL AXIS
                if Yaxis=='mean' : 
                    if 'lati' in zw.keys() : zw2 = averaging.ymean(zw2, zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Yaxis should be None maybe ?")
                elif Yaxis=='integrate' : 
                    if 'lati' in zw.keys() : zw2 = averaging.ymean(zw2, zwmesh, integral = True, keep_ndim=True, **kwargs)
                    else : raise ValueError("Yaxis should be None maybe ?")
                elif Yaxis=='interp' : 
                    if 'ilat' in kwargs.keys() : 
                        if 'lati' in zw.keys() : zw2  = interpolating.yinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                        else : raise ValueError("Yaxis should be None maybe ?")        
                    else : raise ValueError("ilat not defined")        
                elif Yaxis == None : 
                    if 'lati' in  zw.keys() : raise ValueError("Yaxis should be 'mean' or 'interp' maybe ?")        
                else : raise ValueError("Yaxis should be 'mean', 'interp' or None")        
                zw3.append(zw2.squeeze())
            #
            zwouS[vkS] = np.array(zw3)
        #
        zwouR['R'+vkR] = zwouS
    #
    timeanimating.zonal_section_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], tlab, suffig=suffig, **kwargs)
############################################################
#  END READ AND TIMEANIM ZONAL_SECTION DELTA R1R9R27  #
############################################################
