"""
Functions to read and plot fields
"""

from IPython import embed
# import time
import numpy        as np
from FGYRE import reading, averaging, interpolating, plotting

####################################################################################################
def read_and_map_delta_r1r9r27(vname, file_suffix, time=None, Zaxis = None, \
                               fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                               suffig='functions_GYRE_read_and_map_delta_r1r9r27_', **kwargs) :
    """
    Read a variable and plot 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : read_and_map_delta_r1r9r27(vname, file_suffix, time=None,
                               Zaxis = None, \
                               fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                               suffig='functions_GYRE_read_and_map_delta_r1r9r27_',
                               **kwargs)

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Zaxis        : None (default), 'interp' or 'mean'
       To indicate what we do with Z axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.zmean, interpolating.zinterpol or map_delta_r1r9r27 eg:
       zmin=100, zmax=500, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.pdf', integral=True
    """
    
    print("> read_and_map_delta_r1r9r27")

    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
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
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #
    plotting.map_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)

# END read_and_map_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_meridional_section_delta_r1r9r27(vname, file_suffix, time=None, Xaxis = None, \
                                                  fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                                  suffig='functions_GYRE_read_and_meridional_section_delta_r1r9r27_', \
                                                  **kwargs) :
    """
    Read a variable and plot 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : read_and_meridional_section_delta_r1r9r27(vname,
                                                  file_suffix,
                                                  time=None, Xaxis =
                                                  None, \
                                                  fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                                  suffig='functions_GYRE_read_and_meridional_section_delta_r1r9r27_',
                                                  \ **kwargs)
    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Xaxis        : None (default), 'interp' or 'mean' or 'integrate'
       To indicate what we do with X axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.xmean, interpolating.xinterpol or meridional_section_delta_r1r9r27 eg:
       xmin=-80, xmax=-60, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.pdf', integral=True
    """
    
    print("> read_and_meridional_section_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
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
                if 'long' in  zw.keys() : raise ValueError("Xaxis should be 'mean' or 'interp' or 'integrate' maybe ?")        
            else : raise ValueError("Xaxis should be 'mean', 'interp', 'integrate' or None")        
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #
    plotting.meridional_section_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)

# END read_and_meridional_section_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_zonal_section_delta_r1r9r27(vname, file_suffix, time=None, Yaxis = None, \
                                         fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                         suffig='functions_GYRE_read_and_zonal_section_delta_r1r9r27_', \
                                         **kwargs) :
    """
    Read a variable and plot 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : read_and_zonal_section_delta_r1r9r27(vname, file_suffix,
                                         time=None, Yaxis = None, \
                                         fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                         suffig='functions_GYRE_read_and_zonal_section_delta_r1r9r27_',
                                         \ **kwargs) :

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Yaxis        : None (default), 'interp' or 'mean' or 'integrate'
       To indicate what we do with Y axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.xmean, interpolating.xinterpol or zonal_section_delta_r1r9r27 eg:
       xmin=-80, xmax=-60, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.pdf', integral=True
    """
    
    print("> read_and_zonal_section_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
            #____________________
            # TEMPORAL MEAN
            zw2 = averaging.tmean(zw['data'], zwmesh, keep_ndim=True, **kwargs)
            #____________________
            # MERIDIONAL AXIS
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
                if 'lati' in  zw.keys() : raise ValueError("Yaxis should be 'mean', 'interp' or 'integrate' maybe ?")        
            else : raise ValueError("Yaxis should be 'mean', 'interp', 'integrate' or None")        
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #
    plotting.zonal_section_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)

# END read_and_zonal_section_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_hovmoellerZT_delta_r1r9r27(vname, file_suffix, time=None, Xaxis = None, Yaxis=None, fsufmld=None, \
                                        fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                        suffig='functions_GYRE_read_and_hovmoellerZT_delta_r1r9r27_', \
                                        **kwargs) :
    """
    Read a variable and plot 9 maps :
        ----------------------------------------
        |   CTL1    |    CTL9    |    CTL27    |
        ----------------------------------------
        |   CC1     |    CC9     |    CC27     |
        ----------------------------------------
        | CC1-CTL1  |  CC9-CTL9  |  CC27-CTL27 |
        ----------------------------------------
    Usage : 

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Xaxis        : None (default), 'interp' or 'mean' ot 'integrate'
       To indicate what we do with X axis
    Yaxis        : None (default), 'interp' or 'mean' or 'integrate
       To indicate what we do with X axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.xmean, interpolating.xinterpol or hovmoellerZT_delta_r1r9r27 eg:
       xmin=-80, xmax=-60, zfact=10**3, lev=20, lev=np.arange(0, 5,
       .25), savefig='test.pdf', integral=True
    """
    
    print("> read_and_hovmoellerZT_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    zwmlR = {}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R27'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwmlS = {}
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '27' + file_suffix
            if isinstance(fsufmld, str) : zwfmld = fdir + vkS + vkR + fsufmld
            # if isinstance(fsufmld, str) : zwfmld = fdir + vkS + '27' + fsufmld
            #____________________
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
            zw2 = zw['data']
            if vkS == 'CTL' : zwouS['date']=zw['date']
            if isinstance(fsufmld, str) : 
                kwargs2=kwargs.copy()
                kwargs2['dim']='tyx'
                zwml = reading.read_ncdf('somxl010', zwfmld, time=time)
                zwml2 = zwml['data']
            #____________________
            # MERIDIONAL AXIS
            if Yaxis=='mean' : 
                if 'lati' in zw.keys() : zw2 = averaging.ymean(zw2, zwmesh, keep_ndim=True, **kwargs)
                else : raise ValueError("Yaxis should be None maybe ?")
                if isinstance(fsufmld, str) & ('lati' in zwml.keys()) : 
                    zwml2 = averaging.ymean(zwml2, zwmesh, keep_ndim=True, **kwargs2)
            elif Yaxis=='integrate' : 
                if 'lati' in zw.keys() : zw2 = averaging.ymean(zw2, zwmesh, integral = True, keep_ndim=True, **kwargs)
                else : raise ValueError("Yaxis should be None maybe ?")
                if isinstance(fsufmld, str) & ('lati' in zwml.keys()) : 
                    zwml2 = averaging.ymean(zwml2, zwmesh, keep_ndim=True, **kwargs2)
            elif Yaxis=='interp' : 
                if 'ilat' in kwargs.keys() : 
                    if 'lati' in zw.keys() : zw2  = interpolating.yinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Yaxis should be None maybe ?")        
                    if isinstance(fsufmld, str) & ('lati' in zwml.keys()) : 
                        zwml2  = interpolating.yinterpol( zwml2,  zwmesh, keep_ndim=True, **kwargs2)
                else : raise ValueError("ilat not defined")        
            elif Yaxis == None : 
                if 'lati' in  zw.keys() : raise ValueError("Yaxis should be 'mean' or 'interp' maybe ?")        
            else : raise ValueError("Yaxis should be 'mean', 'interp' or None")        
            #____________________
            # ZONAL AXIS
            if Xaxis=='mean' : 
                if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, keep_ndim=True, **kwargs)
                else : raise ValueError("Xaxis should be None maybe ?")
                if isinstance(fsufmld, str) & ('long' in zwml.keys()) : 
                    zwml2 = averaging.xmean(zwml2, zwmesh, keep_ndim=True, **kwargs2)
            elif Xaxis=='integrate' : 
                if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, integral = True, keep_ndim=True, **kwargs)
                else : raise ValueError("Xaxis should be None maybe ?")
                if isinstance(fsufmld, str) & ('long' in zwml.keys()) : 
                    zwml2 = averaging.xmean(zwml2, zwmesh, keep_ndim=True, **kwargs2)
            elif Xaxis=='interp' : 
                if 'ilon' in kwargs.keys() : 
                    if 'long' in zw.keys() : zw2  = interpolating.xinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Xaxis should be None maybe ?")        
                    if isinstance(fsufmld, str) & ('long' in zwml.keys()) : 
                        zwml2  = interpolating.xinterpol( zwml2,  zwmesh, keep_ndim=True, **kwargs2)
                else : raise ValueError("ilon not defined")        
            elif Xaxis == None : 
                if 'long' in  zw.keys() : raise ValueError("Xaxis should be 'mean' or 'interp' maybe ?")        
            else : raise ValueError("Xaxis should be 'mean', 'interp' or None")        
            zwouS[vkS] = zw2.squeeze()
            zwmlS[vkS] = zwml2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
        zwmlR['R'+vkR] = zwmlS
    #
    if not isinstance(fsufmld, str) : 
        plotting.hovmoellerZT_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)
    else : 
        plotting.hovmoellerZT_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, \
                                            addline=[zwmlR['R1'], zwmlR['R9'], zwmlR['R27']], **kwargs)

# END read_and_hovmoellerZT_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_zonal_line_delta_r1r9r27(vname, file_suffix, time=None, Zaxis=None, Xaxis=None,\
                                          fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                          suffig='functions_GYRE_read_and_zonal_line_delta_r1r9r27_',\
                                          **kwargs) :
    """
    Read a variable and plot zonal lines :
        ----------------
        |  CTL and CC  |
        ----------------
        |    CC-CTL    |
        ----------------
    Usage : read_and_zonal_line_delta_r1r9r27(vname, file_suffix,
       time=None, Zaxis=None, Xaxis=None,
       fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',
       suffig='functions_GYRE_read_and_zonal_line_delta_r1r9r27_',
       **kwargs) :

    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Zaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with Z axis
    Xaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with X axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to (zx)mean, (zy)interpol or zonal_line_delta_r1r9r27 eg:
       (zx)min=100, (zx)max=500, zfact=10**3, lev=20, lev=np.arange(0,
       5, .25), savefig='test.pdf', grid='T', ylim=[-5, 2.],
       title='text', ytitle='ylabel', nan=True
    """
    
    print("> read_and_zonal_line_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            #____________________
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
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
            #____________________
            # MERIDIONAL AXIS
            if Xaxis=='mean' : 
                if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, keep_ndim=True, **kwargs)
                else : raise ValueError("Xaxis should be None maybe ?")
            elif Xaxis=='integrate' : 
                if 'long' in zw.keys() : zw2 = averaging.xmean(zw2, zwmesh, keep_ndim=True, integral = True, **kwargs)
                else : raise ValueError("Xaxis should be None maybe ?")
            elif Xaxis=='interp' : 
                if 'ilon' in kwargs.keys() : 
                    if 'long' in zw.keys() : zw2  = interpolating.xinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Xaxis should be None maybe ?")        
                else : raise ValueError("ilon not defined")        
            elif Xaxis == None : 
                if 'long' in  zw.keys() : raise ValueError("Xaxis should be 'mean' or 'interp' maybe ?")        
            else : raise ValueError("Xaxis should be 'mean', 'interp' or None")        
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #
    plotting.zonal_line_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)
# END read_and_zonal_line_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_vertical_profile_delta_r1r9r27(vname, file_suffix, time=None, Yaxis=None, Xaxis=None, \
                                            fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                            suffig='functions_GYRE_read_and_vertical_profile_delta_r1r9r27_',\
                                            **kwargs) :
    """
    Read a variable and plot vertical profiles:
        -----------------------
        |    CTL   |          |  
        |    and   | CC - CTL |                    
        |    CC    |          |
        -----------------------
    Usage: read_and_vertical_profile_delta_r1r9r27(vname, file_suffix,
       time=None, Yaxis=None, Xaxis=None,
       fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',
       suffig='functions_GYRE_read_and_vertical_profile_delta_r1r9r27_',
       **kwargs)

    
    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Yaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with Y axis
    Xaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with X axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to (yx)mean, (yx)interpol or vertical_profile_delta_r1r9r27 eg:
       (yx)min=35, (yx)max=45, zfact=10**3, savefig='test.pdf',
       grid='T', xlim=[-5, 2.], title='text', ytitle='ylabel',
       nan=True
    """
    
    print("> read_and_vertical_profile_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        zwouS = {'mesh':zwmesh} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
            #____________________
            # TEMPORAL MEAN
            zw2 = averaging.tmean(zw['data'], zwmesh, keep_ndim=True, **kwargs)
            #____________________
            # MERIDIONAL AXIS
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
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #

    plotting.vertical_profile_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)
# END read_and_vertical_profile_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_timeseries_delta_r1r9r27(vname, file_suffix, time=None, Zaxis=None, Yaxis=None, Xaxis=None, \
                                      fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                                      suffig='functions_GYRE_read_and_timeseries_delta_r1r9r27_',\
                                      **kwargs) :
    """
    Read a variable and plot 2 time series :
        ----------------
        |  CTL and CC  |
        ----------------
        |    CC-CTL    |
        ----------------
    Usage :read_and_timeseries_delta_r1r9r27(vname, file_suffix,
       time=None, Zaxis=None, Yaxis=None, Xaxis=None,
       fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',
       suffig='functions_GYRE_read_and_timeseries_delta_r1r9r27_',
       **kwargs)
    
    Parameters
    ----------
    vname       : name of the variable to plot
    file_suffix : suffix to append to complete file name
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    Zaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with Z axis
    Yaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with Y axis
    Xaxis        : None (default), 'interp', 'mean' or 'integrate'
       To indicate what we do with X axis
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to averaging.(zyx)mean, interpolating.(zyx)interpol 
       or plotting.timeseries_delta_r1r9r27 eg: 
       (zyx)min=100, (zyx)max=500, idep, ilat, ilon
       zfact=10**3
       savefig='test.pdf', grid='T', ylim=[-5, 2.], title='text',
       ytitle='ylabel', nan=True
    """
    
    print("> read_and_timeseries_delta_r1r9r27")
    kR = ['1', '9', '27']
    kS = ['CTL', 'CC']
    zwouR = {} # local dictionnary to store diag from each (eg zwR['R1'] = {'CTL':...)}
    for vkR in kR : # loop on resolution
        
        zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_R'+vkR+'.nc' 
        # zwmeshfile = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+'R1'+'.nc' 
        zwmesh = reading.read_mesh(zwmeshfile)

        zwouS = {} # local dictionnary to store diag from each simu ('CTL', 'CC', 'mesh')
        for vkS in kS :  # CTL or CC simulation
            zwfilename = fdir + vkS + vkR + file_suffix
            # zwfilename = fdir + vkS + '1' + file_suffix
            #____________________
            zw  = reading.read_ncdf(vname, zwfilename, time=time)
            zw2 = zw['data']
            if vkS == 'CTL' : zwouS['date']=zw['date']
            #____________________
            # VERTICAL AXIS
            if Zaxis=='mean' : 
                if 'dept' in zw.keys() : zw2 = averaging.zmean(zw2, zwmesh, keep_ndim=True, **kwargs)
                else : raise ValueError("Zaxis should be None maybe ?")
            elif Zaxis=='integrate' : 
                if 'dept' in zw.keys() : zw2 = averaging.zmean(zw2, zwmesh, integral=True, keep_ndim=True, **kwargs)
                else : raise ValueError("Zaxis should be None maybe ?")
            elif Zaxis=='interp' : 
                if 'idep' in kwargs.keys() : 
                    if 'dept' in zw.keys() : zw2  = interpolating.zinterpol( zw2,  zwmesh, keep_ndim=True, **kwargs)
                    else : raise ValueError("Zaxis should be None maybe ?")        
                else : raise ValueError("idep not defined")        
            elif Zaxis == None : 
                if 'dept' in  zw.keys() : raise ValueError("Zaxis should be 'mean' or 'interp' maybe ?")        
            else : raise ValueError("Zaxis should be 'mean', 'interp' or None")        
            #____________________
            # MERIDIONAL AXIS
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
            zwouS[vkS] = zw2.squeeze()
        #
        zwouR['R'+vkR] = zwouS
    #


    plotting.timeseries_delta_r1r9r27(zwouR['R1'], zwouR['R9'], zwouR['R27'], suffig=suffig, **kwargs)
# END read_and_timeseries_delta_r1r9r27
####################################################################################################

####################################################################################################
def read_and_timeseries_data(vname, fsuf, fpre, time=None, \
                            fdir='/gpfswork/rech/dyk/rdyk004/GYRE_XML/',\
                             suffig='functions_GYRE_read_and_timeseries_data_',\
                             **kwargs) :
    """
    Read a variable and plot time series :
    Usage :

    Parameters
    ----------
    vname       : name of the variable to plot
    fsuf        : suffix to append to complete file name
    fpre        : list of dictionnaries
       [{'data':string, 'reso':string}, ...]
    time        : tupple of string pass to reading.read_ncdf
       eg. time=('0161-01-01', '0170-12-31')
    fdir        : folder name
    suffig      : suffix to saved figure name
    **kwargs    : pass to timeseries_delta ou averaging.spatial_mean
       (zyx)min, (zyx)max: define the spatial box
       (zyx)int : to integrate along (zyx) axis
       grid: define the grid (only 'T' implemented)
       nan: True or False
       zfact:change unit of variable.
       title: title of the figures
       ytitle: add title to colorbars
       ylim: set the lower and upper bounds of y axis
    """
    
    print("> read_and_timeseries_data")

    dat=[]
    for ind, val in np.ndenumerate(fpre) : 
        fdata  = fdir + val['data'] + fsuf
        fmesh  = '/gpfswork/rech/dyk/rdyk004/MESH/mesh_mask_'+val['reso']+'.nc'
    
        # READ DATA
        zwdata  = reading.read_ncdf(vname, fdata , time=time)
        zwmesh  = reading.read_mesh(fmesh)

        # COMPUTE SPATIAL MEAN
        zw  = averaging.spatial_mean( zwdata['data'], zwmesh, **kwargs )
    
        dat.append({'data':zw, 'date':zwdata['date']})        
    #
    plotting.timeseries_data(dat, suffig=suffig, **kwargs)
# END read_and_timeseries_data
####################################################################################################

