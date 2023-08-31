"""
Functions to average fields
"""

from IPython import embed
# import time
import numpy        as np

####################################################################################################
def spatial_mean(field, mesh, dim='tzyx', xint=False, yint=False, zint=False, **kwargs):
    """
    Compute a spatial mean of a TZYX array. Vertical, meridional and
    zonal limits can be specified. If no spatial limits defined, the
    mean is computed on the whole domain.
    
    Usage: spatial_mean(field, mesh, xint=False, yint=False,
       zint=False, **kwargs)

    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the vertical mean, the field needs a vertical
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    xint: False (default) or True
       to integrate or not along the zonal axis
    yint: False (default) or True
       to integrate or not along the meridional axis
    zint: False (default) or True
       to integrate or not along the vertical axis
    dim : 'tzyx' (default)
       string to specify the dimension of field
    **kwargs: keywords passed to xmean, ymean and zmean
       zmin, zmax, ymin, ymax, xmin, xmax: define the spatial box
       grid: define the grid (only 'T' implemented)
       nan: True or False
    """

    print("> spatial_mean")

    zw=field

    if 'z' in dim : zw  = zmean( zw, mesh, dim=dim, integral=zint, keep_ndim = True, **kwargs)
    if 'y' in dim : zw  = ymean( zw, mesh, dim=dim, integral=yint, keep_ndim = True, **kwargs)
    if 'x' in dim : zw  = xmean( zw, mesh, dim=dim, integral=xint, keep_ndim = True, **kwargs)
    
    output = zw.squeeze()

    return output
# END moyenne
####################################################################################################

####################################################################################################
def tmean(field, mesh, nan=True, dim='tzyx',keep_ndim=False, grid='T', **kwargs) :
    """
    Compute the temporal mean of field 3D array.

    Usage: tmean(field, mesh, nan=True, dim='tzyx',keep_ndim=False,
       grid='T', **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
    mesh : mesh dictionary from read_mesh
        meshmask associated with field, used to get scale factors and
        mask
    nan : True (default) or False
        Exclude (True) or not (False) nan values from the mean. (True) The
        arithmetic mean is the sum of the non-NaN elements along the
        axis divided by the number of non-NaN elements. (False) Return a
        nan values
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
        to specify the dimension of field
    grid : 'T' (default), 'U', 'V' or 'W'
        Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
        Only difference between 'T', 'U' and 'V' is the mask at the
        north and east edges of the domain
    keep_ndim : boolean
        to keep or not the number of dimensions
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> tmean")
    
    # add axis to always have 4 dimensions
    # when we average (axis=1)
    if   dim=='t'    : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, np.newaxis])
    elif dim=='tx'   : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, :])
    elif dim=='ty'   : zwfield = np.ma.array(field.data[np.newaxis, :, :, np.newaxis])
    elif dim=='tz'   : zwfield = np.ma.array(field.data[:, :, np.newaxis, np.newaxis])
    elif dim=='tyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tzx'  : zwfield = np.ma.array(field.data[:, :, np.newaxis, :])
    elif dim=='tzy'  : zwfield = np.ma.array(field.data[:, :, :, np.newaxis])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    # add the mask
    nt, nz, ny, nx = zwfield.shape
    mask = mesh[grid.lower()+'mask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwfield.mask = [x!=1 for x in mask]
    elif (nz==1 and ny>1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0]]
    elif (nz>1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, :]]
    elif (nz>1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, :, 5]]
    elif (nz==1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, :]]
    elif (nz==1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, :, 5]]
    elif (nz>1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, 5]]
    elif (nz==1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, 5]]
    else: raise ValueError("ERROR, zwfield doesn't have the right dimension")

    # The nan values are masked to exclude them from the average
    if nan : zwfield.mask = np.where( zwfield != zwfield, True, zwfield.mask) 

    output = np.ma.average(zwfield, axis = 0)

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='t'    : output = output[np.newaxis, 0, 0, 0]
        elif dim=='tx'   : output = output[np.newaxis, 0, 0, :]
        elif dim=='ty'   : output = output[np.newaxis, 0, :, 0]
        elif dim=='tz'   : output = output[np.newaxis, :, 0, 0]
        elif dim=='tyx'  : output = output[np.newaxis, 0, :, :]
        elif dim=='tzx'  : output = output[np.newaxis, :, 0, :]
        elif dim=='tzy'  : output = output[np.newaxis, :, :, 0]
        elif dim=='tzyx' : output = output[np.newaxis,:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                    ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    else : 
        if   dim=='t'    : output = output[0, 0, 0]
        elif dim=='tx'   : output = output[0, 0, :]
        elif dim=='ty'   : output = output[0, :, 0]
        elif dim=='tz'   : output = output[:, 0, 0]
        elif dim=='tyx'  : output = output[0, :, :]
        elif dim=='tzx'  : output = output[:, 0, :]
        elif dim=='tzy'  : output = output[:, :, 0]
        elif dim=='tzyx' : output = output[:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                    ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    #

    return output
# END tmean
####################################################################################################

####################################################################################################
def zmean(field, mesh, zmin=None, zmax=None, grid='T', nan=True, dim='tzyx',\
              keep_ndim=False, integral=False, **kwargs) :
    """
    Compute the vertical mean of field 3D array between zmin and
    zmax. If no zmin and zmax, the mean is computed over the entire
    water column.

    Usage: tr = zmean(field, mesh, zmin=None, zmax=None, grid='T',
       nan=True, dim='tzyx', keep_ndim=False, integral=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
        field to compute the vertical mean, the field needs a vertical
        dimension
    mesh : mesh dictionary from read_mesh
        meshmask associated with field, used to get scale factors and
        mask
    zmin : None or float
        to give the minimal depth, default is None and in that case
        zmin = 0
    zmax : None or float
        to give the maximal depth, default is None and in that case
        zmax is the bottom
    grid : 'T' (default), 'U', 'V' or 'W' (not implemented yet)
        Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
        Only difference between 'T', 'U' and 'V' is the mask at the
        north and east edges of the domain
    nan : True (default) or False
        Exclude (True) or not (False) nan values from the mean. (True) The
        arithmetic mean is the sum of the non-NaN elements along the
        axis divided by the number of non-NaN elements. (False) Return a
        nan values
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
        to specify the dimension of field
    keep_ndim : boolean
        to keep or not the number of dimensions
    integral: False (default) or True
        to integrate or not 
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> zmean")
    
    # add axis to always have 4 dimensions
    # when we average (axis=1)
    if   dim=='z'    : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, np.newaxis])
    elif dim=='zx'   : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, :])
    elif dim=='zy'   : zwfield = np.ma.array(field.data[np.newaxis, :, :, np.newaxis])
    elif dim=='tz'   : zwfield = np.ma.array(field.data[:, :, np.newaxis, np.newaxis])
    elif dim=='zyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tzx'  : zwfield = np.ma.array(field.data[:, :, np.newaxis, :])
    elif dim=='tzy'  : zwfield = np.ma.array(field.data[:, :, :, np.newaxis])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    # add the mask
    nt, nz, ny, nx = zwfield.shape
    mask = mesh[grid.lower()+'mask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwfield.mask = [x!=1 for x in mask]
    elif (nz==1 and ny>1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0]]
    elif (nz>1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, :]]
    elif (nz>1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, :, 5]]
    elif (nz==1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, :]]
    elif (nz==1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, :, 5]]
    elif (nz>1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, 5]]
    elif (nz==1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, 5]]
    else: raise ValueError("ERROR, zwfield doesn't have the right dimension")

    # The nan values are masked to exclude them from the average
    if nan : zwfield.mask = np.where( zwfield != zwfield, True, zwfield.mask) 

    if grid=='T' or grid=='U' or grid=='V': 
        
        # case 1 full depth mean
        if zmin==None and zmax==None :
            print("case 1: full depth mean")
            output, sumweights = np.ma.average(zwfield, axis = 1,\
                                                   weights = mesh['e3t_0'], returned=True)
        # case 2 from surface to zmax
        elif zmin == None :
            print("case 2: mean from surface to zmax")
            if (0 < zmax < mesh['depW'][-1]) : 
                # last depth index before zmax
                jkmax = np.argwhere(mesh['depW'] <= zmax).max() 
                # weight of last cell before zmax
                wei_jkmax = zmax - mesh['depW'][jkmax]          
                # create a new weight array for averaging 
                new_wei = np.concatenate((mesh['e3t_0'][:jkmax], [wei_jkmax])) 
                output, sumweights = np.ma.average(zwfield[:, :jkmax+1], axis = 1,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(0 < zmax < mesh['depW'][-1])")
            #
        # case 3 from zmin to bottom
        elif zmax == None :
            print("case 3: mean from zmin to bottom")
            if (0 < zmin < mesh['depW'][-1]) :
                # first depth index before zmin
                jkmin = np.argwhere(mesh['depW'] >= zmin).min() 
                # weight of last cell before zmax
                wei_jkmin = mesh['depW'][jkmin] - zmin          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jkmin], mesh['e3t_0'][jkmin:])) 
                output, sumweights = np.ma.average(zwfield[:, jkmin-1:], axis = 1, \
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(0 < zmin < mesh['depW'][-1])")
        # case 4 from zmin to zmax
        else :
            print("case 4: mean from zmin to zmax")
            if (0 < zmin < zmax < mesh['depW'][-1]): 
                # last depth index before zmax
                jkmax = np.argwhere(mesh['depW'] <= zmax).max() 
                # weight of last cell before zmax
                wei_jkmax = zmax - mesh['depW'][jkmax]          
                # first depth index before zmin
                jkmin = np.argwhere(mesh['depW'] >= zmin).min() 
                # weight of last cell before zmax
                wei_jkmin = mesh['depW'][jkmin] - zmin          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jkmin], \
                                              mesh['e3t_0'][jkmin:jkmax], [wei_jkmax])) 
                output, sumweights = np.ma.average(zwfield[:, jkmin-1:jkmax+1], axis = 1,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(0 < zmin < zmax < mesh['depW'][-1])")
        #
    #
    if integral : 
        print("Vertical integral")
        output=output*sumweights
    #

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='z'    : output = output[0, np.newaxis, 0, 0]
        elif dim=='zx'   : output = output[0, np.newaxis, 0, :]
        elif dim=='zy'   : output = output[0, np.newaxis, :, 0]
        elif dim=='tz'   : output = output[:, np.newaxis, 0, 0]
        elif dim=='zyx'  : output = output[0, np.newaxis, :, :]
        elif dim=='tzx'  : output = output[:, np.newaxis, 0, :]
        elif dim=='tzy'  : output = output[:, np.newaxis, :, 0]
        elif dim=='tzyx' : output = output[:, np.newaxis, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                    ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    else : 
        if   dim=='z'    : output = output[0, 0, 0]
        elif dim=='zx'   : output = output[0, 0, :]
        elif dim=='zy'   : output = output[0, :, 0]
        elif dim=='tz'   : output = output[:, 0, 0]
        elif dim=='zyx'  : output = output[0, :, :]
        elif dim=='tzx'  : output = output[:, 0, :]
        elif dim=='tzy'  : output = output[:, :, 0]
        elif dim=='tzyx' : output = output[:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                    ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    #

    return output
# END zmean
####################################################################################################

####################################################################################################
def ymean(field, mesh, ymin=None, ymax=None, grid='T', nan=True, dim='tzyx', keep_ndim=False,\
              integral=False, **kwargs) :
    """
    Compute the meridional mean of field 3D array between ymin and
    xmax. If no ymin and ymax, the meridional mean is computed over
    the entire bassin.

    Usage: ymean(field, mesh, ymin=None, ymax=None, grid='T', nan=True,
       dim='tzyx', keep_ndim=False, integral=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
        field to compute the meridional mean, the field needs a meridional
        dimension
    mesh : mesh dictionary from read_mesh
        meshmask associated with field, used to get scale factors and
        mask
    ymin : None or float
        to give the minimal latitude in degree, default is None and in that case
        ymin = mesh['latV'][0, 5] = 20N
    ymax : None or float
        to give the maximal latitude, default is None and in that case
        ymax = mesh['latV'][-2, 5] = 48.6N
    grid : 'T' (default), 'U', 'W' or 'V' (not implemented yet)
        Field is defined at 'T' (or 'U' or 'W') points or 'V' point.
    nan : True (default) or False
        Exclude (True) or not (False) nan values from the mean. (True) The
        arithmetic mean is the sum of the non-NaN elements along the
        axis divided by the number of non-NaN elements. (False) Return a
        nan values
    dim : 'tzyx' (default), 'zyx', 'tyx', 'tzy', 'ty', 'zy', 'yx', 'y'
        to specify the dimension of field
    keep_ndim : boolean
        to keep or not the number of dimensions
    integral: False (default) or True
        to integrate or not 
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> ymean")

    # add axis to always have 4 dimensions
    # when we average (axis=2)
    if   dim=='y'    : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, np.newaxis])
    elif dim=='yx'   : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, :])
    elif dim=='zy'   : zwfield = np.ma.array(field.data[np.newaxis, :, :, np.newaxis])
    elif dim=='ty'   : zwfield = np.ma.array(field.data[:, np.newaxis, :, np.newaxis])
    elif dim=='zyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tyx'  : zwfield = np.ma.array(field.data[:, np.newaxis, :, :])
    elif dim=='tzy'  : zwfield = np.ma.array(field.data[:, :, :, np.newaxis])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'tzy', 'tyx'", \
                                ", 'zyx', 'ty', 'zy', 'yx', 'y'")

    # add the mask
    nt, nz, ny, nx = zwfield.shape
    mask = mesh[grid.lower()+'mask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwfield.mask = [x!=1 for x in mask]
    elif (nz==1 and ny>1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0]]
    elif (nz>1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, :]]
    elif (nz>1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, :, 5]]
    elif (nz==1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, :]]
    elif (nz==1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, :, 5]]
    elif (nz>1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, 5]]
    elif (nz==1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, 5]]
    else: raise ValueError("ERROR, zwfield doesn't have the right dimension")

    # The nan values are masked to exclude them from the average
    if nan : zwfield.mask = np.where( zwfield != zwfield, True, zwfield.mask) 

    if grid=='T' or grid=='U' or grid=='W': 
        
        # case 1 full meridional mean
        if ymin==None and ymax==None :
            print("case 1: full meridional mean")
            # in GYRE all horizontal aspect ratio are equal
            output, sumweights = np.ma.average(zwfield, axis = 2, \
                                               weights = mesh['e2t'][:, 5], returned=True) 
        # case 2 from mesh['latV'][0, 5]=20 to ymax
        elif ymin == None :
            print("case 2: mean from ", mesh['latV'][0, 5], "to ymax")
            if (mesh['latV'][0, 5] < ymax < mesh['latV'][-2, 5]) : 
                # last latitude index before ymax
                jjmax = np.argwhere(mesh['latV'][:, 5] <= ymax).max() 
                # weight of last cell before ymax
                wei_jjmax = (ymax - mesh['latV'][jjmax, 5])*mesh['e2t'][jjmax+1, 5+1] 
                # create a new weight array for averaging 
                new_wei = np.concatenate((mesh['e2t'][:jjmax+1, 5], [wei_jjmax]))
                output, sumweights = np.ma.average(zwfield[:, :, :jjmax+2, :], axis = 2,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True",\
                                        " (mesh['latV'][0, 5] < ymax < mesh['latV'][-2, 5]): ", \
                                        mesh['latV'][0, 5], " < ", ymax, " < ", mesh['latV'][-2, 5])
        # case 3 from from ymin to mesh['latV'][-2, 5]=48.6
        elif ymax == None :
            print("case 3: mean from ymin to ", mesh['latV'][-2, 5])
            if (mesh['latV'][0, 5] < ymin < mesh['latV'][-2, 5]) : 
                # first latitude index before ymin
                jjmin = np.argwhere(mesh['latV'][:, 5] >= ymin).min() 
                # weight of first cell before ymin
                wei_jjmin = (mesh['latV'][jjmin, 5] - ymin)*mesh['e2t'][jjmin, 5]          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jjmin], mesh['e2t'][jjmin+1:, 5])) 
                output, sumweights = np.ma.average(zwfield[:, :, jjmin:, :], axis = 2,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True",\
                                        " (mesh['latV'][0, 5] < ymin < mesh['latV'][-2, 5]): ", \
                                        mesh['latV'][0, 5], " < ", ymin, " < ", mesh['latV'][-2, 5])
        # case 4 from ymin to ymax
        else :
            print("case 4: mean from ymin to ymax: ", ymin, "to", ymax)
            if (mesh['latV'][0, 5] < ymin < ymax < mesh['latV'][-2, 5]) : 
                # last latitude index before ymax
                jjmax = np.argwhere(mesh['latV'][:, 5] <= ymax).max() 
                # weight of last cell before ymax
                wei_jjmax = (ymax - mesh['latV'][jjmax, 5])*mesh['e2t'][jjmax+1, 5] 
                # first latitude index before ymin
                jjmin = np.argwhere(mesh['latV'][:, 5] >= ymin).min() 
                # weight of first cell before ymin
                wei_jjmin = (mesh['latV'][jjmin, 5] - ymin)*mesh['e2t'][jjmin, 5]          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jjmin], mesh['e2t'][jjmin+1:jjmax+1, 5],\
                                              [wei_jjmax])) 
                output, sumweights = np.ma.average(zwfield[:, :, jjmin:jjmax+2, :], axis = 2,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(mesh['latV'][0, 5] < ymin < ymax < mesh['latV'][-2, 5]): ",\
                                        mesh['latV'][0, 5], "<", ymin, "<",\
                                        ymax, "<", mesh['latV'][-2, 5])
        #
    #
    if integral : 
        print("Meridional integral") 
        output=output*sumweights
    #
    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='y'    : output = output[0, 0, np.newaxis, 0]
        elif dim=='yx'   : output = output[0, 0, np.newaxis, :]
        elif dim=='zy'   : output = output[0, :, np.newaxis, 0]
        elif dim=='ty'   : output = output[:, 0, np.newaxis, 0]
        elif dim=='zyx'  : output = output[0, :, np.newaxis, :]
        elif dim=='tyx'  : output = output[:, 0, np.newaxis, :]
        elif dim=='tzy'  : output = output[:, :, np.newaxis, 0]
        elif dim=='tzyx' : output = output[:, :, np.newaxis, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'tzy', 'tyx'", \
                                    ", 'zyx', 'ty', 'zy', 'yx', 'y'")
    else : 
        if   dim=='y'    : output = output[0, 0, 0]
        elif dim=='yx'   : output = output[0, 0, :]
        elif dim=='zy'   : output = output[0, :, 0]
        elif dim=='ty'   : output = output[:, 0, 0]
        elif dim=='zyx'  : output = output[0, :, :]
        elif dim=='tyx'  : output = output[:, 0, :]
        elif dim=='tzy'  : output = output[:, :, 0]
        elif dim=='tzyx' : output = output[:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'tzy', 'tyx'", \
                                    ", 'zyx', 'ty', 'zy', 'yx', 'y'")
    #
    return output
# END ymean
####################################################################################################

####################################################################################################
def xmean(field, mesh, xmin=None, xmax=None, grid='T', nan=True, dim='tzyx',\
              keep_ndim=False, integral=False, **kwargs) :
    """
    Compute the zonal mean of field 3D array between xmin and
    xmax. If no xmin and xmax, the zonal mean is computed over the entire
    bassin.

    Usage: xmean(field, mesh, xmin=None, xmax=None, grid='T', nan=True,
       dim='tzyx', keep_ndim=False, integral=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
        field to compute the zonal mean, the field needs a zonal
        dimension
    mesh : mesh dictionary from read_mesh
        meshmask associated with field, used to get scale factors and
        mask
    xmin : None or float
        to give the minimal longitude in degree, default is None and in that case
        xmin = mesh['lonU'][5, 0] = -85E
    xmax : None or float
        to give the maximal longitude, default is None and in that case
        xmax = mesh['lonU'][5, -2] = -56.4E
    grid : 'T' (default), 'V', 'W' or 'U' (not implemented yet)
        Field is defined at 'T' (or 'V' or 'W') points or 'U' point.
    nan : True (default) or False
        Exclude (True) or not (False) nan values from the mean. (True) The
        arithmetic mean is the sum of the non-NaN elements along the
        axis divided by the number of non-NaN elements. (False) Return a
        nan values
    dim : 'tzyx' (default), 'zyx', 'tyx', 'tzx', 'tx', 'yx', 'zx', 'x'
        to specify the dimension of field
    keep_ndim : boolean
        to keep or not the number of dimensions
    integral: False (default) or True
        to integrate or not 
    Returns
    -------
    output : masked array same dimension as field
        averaged on the zonal axis, be careful field is masked after this operation
    """

    print("> xmean")
    
    # add axis to always have 4 dimensions
    # when we average (axis=2)
    if   dim=='x'    : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, np.newaxis, :])
    elif dim=='yx'   : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, :])
    elif dim=='zx'   : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, :])
    elif dim=='tx'   : zwfield = np.ma.array(field.data[:, np.newaxis, np.newaxis, :])
    elif dim=='tzx'  : zwfield = np.ma.array(field.data[:, :, np.newaxis, :])
    elif dim=='tyx'  : zwfield = np.ma.array(field.data[:, np.newaxis, :, :])
    elif dim=='zyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                ", 'tzx', 'tx, 'zx', 'yx', 'x'")
    # add the mask
    nt, nz, ny, nx = zwfield.shape
    mask = mesh[grid.lower()+'mask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwfield.mask = [x!=1 for x in mask]
    elif (nz==1 and ny>1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0]]
    elif (nz>1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, :]]
    elif (nz>1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, :, 5]]
    elif (nz==1 and ny==1  and nx>1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, :]]
    elif (nz==1 and ny>1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, :, 5]]
    elif (nz>1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, :, 5, 5]]
    elif (nz==1 and ny==1  and nx==1) : zwfield.mask = [x!=1 for x in mask[:, 0, 5, 5]]
    else: raise ValueError("ERROR, zwfield doesn't have the right dimension")

    # The nan values are masked to exclude them from the average
    if nan : zwfield.mask = np.where( zwfield != zwfield, True, zwfield.mask) 

    if grid=='T' or grid=='V' or grid=='W': 
        
        # case 1 full zonal mean
        if xmin==None and xmax==None :
            print("case 1: full zonal mean")
            # in GYRE all horizontal aspect ratio are equal
            output, sumweights = np.ma.average(zwfield, axis = -1, \
                                               weights = mesh['e1t'][5, :], returned=True) 
        # case 2 from mesh['lonU'][5,0]=-85 to xmax
        elif xmin == None :
            print("case 2: mean from ", mesh['lonU'][5, 0], "to xmax")
            if (mesh['lonU'][5, 0] < xmax < mesh['lonU'][5, -2]) : 
                # last longitude index before xmax
                jimax = np.argwhere(mesh['lonU'][5, :] <= xmax).max() 
                # weight of last cell before xmax
                wei_jimax = (xmax - mesh['lonU'][5, jimax])*mesh['e1t'][5, jimax+1] 
                # create a new weight array for averaging 
                new_wei = np.concatenate((mesh['e1t'][5, :jimax+1], [wei_jimax]))       
                output, sumweights = np.ma.average(zwfield[:, :, :, :jimax+2], axis = -1,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(mesh['lonU'][5, 0] < xmax < mesh['lonU'][5, -2]): ", \
                                        mesh['lonU'][5, 0], " < ", xmax, " < ", mesh['lonU'][5, -2])
        # case 3 from from xmin to mesh['lonU'][5, -2]=-56.4E
        elif xmax == None :
            print("case 3: mean from xmin to ", mesh['lonU'][5, -2])
            if (mesh['lonU'][5, 0] < xmin < mesh['lonU'][5, -2]) : 
                # first longitude index before xmin
                jimin = np.argwhere(mesh['lonU'][5, :] >= xmin).min() 
                # weight of first cell before xmin
                wei_jimin = (mesh['lonU'][5, jimin] - xmin)*mesh['e1t'][5, jimin]          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jimin], mesh['e1t'][5, jimin+1:])) 
                output, sumweights = np.ma.average(zwfield[:, :, :, jimin:], axis = -1,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(mesh['lonU'][5, 0] < xmin < mesh['lonU'][5, -2]): ", \
                                        mesh['lonU'][5, 0], " < ", xmin, " < ", mesh['lonU'][5, -2])
        # case 4 from xmin to xmax
        else :
            print("case 4: mean from xmin to xmax")
            if (mesh['lonU'][5, 0] < xmin < xmax < mesh['lonU'][5, -2]) : 
                # last longitude index before xmax
                jimax = np.argwhere(mesh['lonU'][5, :] <= xmax).max() 
                # weight of last cell before xmax
                wei_jimax = (xmax - mesh['lonU'][5, jimax])*mesh['e1t'][5, jimax+1] 
                # first longitude index before xmin
                jimin = np.argwhere(mesh['lonU'][5, :] >= xmin).min() 
                # weight of first cell before xmin
                wei_jimin = (mesh['lonU'][5, jimin] - xmin)*mesh['e1t'][5, jimin]          
                # create a new weight array for averaging 
                new_wei = np.concatenate(([wei_jimin], mesh['e1t'][5, jimin+1:jimax+1],\
                                              [wei_jimax])) 
                output, sumweights = np.ma.average(zwfield[:, :, :, jimin:jimax+2], axis = -1,\
                                                       weights = new_wei, returned=True)
            else : raise ValueError("ERROR, this condition is not True ",\
                                        "(mesh['lonU'][5, 0] < xmin < xmax < mesh['lonU'][5, -2]): ", \
                                        mesh['lonU'][5, 0], "<", xmin, "<",\
                                        xmax, "<", mesh['lonU'][5, -2])
        #
    #
    if integral :
        print("Zonal integral")
        output=output*sumweights
    #

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='x'    : output = output[0, 0, 0, np.newaxis]
        elif dim=='yx'   : output = output[0, 0, :, np.newaxis]
        elif dim=='zx'   : output = output[0, :, 0, np.newaxis]
        elif dim=='tx'   : output = output[:, 0, 0, np.newaxis]
        elif dim=='tzx'  : output = output[:, :, 0, np.newaxis]
        elif dim=='tyx'  : output = output[:, 0, :, np.newaxis]
        elif dim=='zyx'  : output = output[0, :, :, np.newaxis]
        elif dim=='tzyx' : output = output[:, :, :, np.newaxis]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                    ", 'tzx', 'tx, 'zx', 'yx', 'x'")
    else :
        if   dim=='x'    : output = output[0, 0, 0]
        elif dim=='yx'   : output = output[0, 0, :]
        elif dim=='zx'   : output = output[0, :, 0]
        elif dim=='tx'   : output = output[:, 0, 0]
        elif dim=='tzx'  : output = output[:, :, 0]
        elif dim=='tyx'  : output = output[:, 0, :]
        elif dim=='zyx'  : output = output[0, :, :]
        elif dim=='tzyx' : output = output[:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                    ", 'tzx', 'tx, 'zx', 'yx', 'x'")
    #
    return output
# END xmean
####################################################################################################

