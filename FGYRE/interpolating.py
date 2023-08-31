"""
Functions to interpolate
"""

from IPython import embed
# import time
import numpy        as np
from scipy.interpolate import RegularGridInterpolator, interp1d

####################################################################################################
def zinterpol(field, mesh, idep, grid='T', dim='tzyx', keep_ndim=False, **kwargs) :
    """
    Interpolation of TZYX field at a given depth. Downward gradient is
    positive.

    Usage: zinterpol(field, mesh, idep, grid='T', dim='tzyx',
       keep_ndim=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the vertical mean, the field needs a vertical
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    idep : float or array
       if it is float, field is interpolated at a fixed depth
       if it an array, the array as to fit the t, y and x dimension
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
       to specify the dimension of field
    keep_ndim : boolean
       to keep or not the number of dimensions
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> zinterpol")

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

    if isinstance(idep, (np.ma.core.MaskedArray, np.ndarray)) : 
        # eventually check the dimension fits with field

        XX = mesh['lon'+grid][5]
        YY = mesh['lat'+grid][:, 5]
        ZZ = mesh['dep'+grid]
        TT = np.arange(nt) # time interval are equal
    
        interp_func = RegularGridInterpolator((TT, ZZ, YY, XX), zwfield, bounds_error=False,\
                                                  method="linear") 
        # create the list of points where zwfield is interpolated 
        pts = [] 
        for ind, val in np.ndenumerate(idep) : 
            pts.append([ TT[ind[0]], val, YY[ind[1]], XX[ind[2]] ]) 
        #
        pts = np.array(pts)
        # interpolation
        izwfield = interp_func(pts) #
        # reshape izwfield
        output = np.ma.array(np.reshape(izwfield, (nt, ny, nx)))
    elif isinstance(idep, (float, int)) : 
        ZZ = mesh['dep'+grid]
        interpfn = interp1d(ZZ, zwfield, axis = 1, bounds_error=False)
        output = np.ma.array(interpfn(idep))
        output.mask = zwfield.mask[:, 0, :, :]
    else: raise TypeError("idep")

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='z'    : output = output.squeeze()
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
        if   dim=='z'    : pass
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
# END zinterpol
####################################################################################################

####################################################################################################
def yinterpol(field, mesh, ilat, grid='T', dim='tzyx', keep_ndim=False, **kwargs) :
    """
    Interpolation of TZYX field at a given latitude.

    Usage: yinterpol(field, mesh, ilat, grid='T', dim='tzyx',
       keep_ndim=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to interpolate
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    ilat : float or array
       if it is float, field is interpolated at a fixed latitude
       if it an array, the array as to fit the t, z and x dimension
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
       to specify the dimension of field
    keep_ndim : boolean
       to keep or not the number of dimensions
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
       interpolated field, be careful field is masked after this operation
    """

    print("> yinterpol")
    
    # add axis to always have 4 dimensions
    # when we interpolate (axis=2)
    if   dim=='y'    : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, np.newaxis])
    elif dim=='yx'   : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, :])
    elif dim=='zy'   : zwfield = np.ma.array(field.data[np.newaxis, :, :, np.newaxis])
    elif dim=='ty'   : zwfield = np.ma.array(field.data[:, np.newaxis, :, np.newaxis])
    elif dim=='tzy'  : zwfield = np.ma.array(field.data[:, :, :, np.newaxis])
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

    if   isinstance(ilat, (np.ma.core.MaskedArray, np.ndarray)) : 
        # eventually chech the dimension fits with field

        XX = mesh['lon'+grid][5]
        YY = mesh['lat'+grid][:, 5]
        ZZ = mesh['dep'+grid]
        TT = np.arange(nt) # time interval are equal
    
        interp_func = RegularGridInterpolator((TT, ZZ, YY, XX), zwfield, bounds_error=False,\
                                                  method="linear") 
        # create the list of points where zwfield is interpolated 
        pts = [] 
        for ind, val in np.ndenumerate(ilat) : 
            pts.append([ TT[ind[0]], ZZ[ind[1]], val, XX[ind[2]] ]) 
        #
        pts = np.array(pts)
        # interpolation
        izwfield = interp_func(pts) #
        # reshape izwfield
        output = np.ma.array(np.reshape(izwfield, (nt, nz, nx))) 
    elif isinstance(ilat, (float, int)) : 
        YY = mesh['lat'+grid][:, 5]
        interpfn = interp1d(YY, zwfield, axis = 2, bounds_error=False)
        output = np.ma.array(interpfn(ilat))
        output.mask = zwfield.mask[:, :, 5, :]
    else: raise TypeError("ilat")

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='y'    : output = output.squeeze()
        elif dim=='yx'   : output = output[0, 0, np.newaxis, :]
        elif dim=='zy'   : output = output[0, :, np.newaxis, 0]
        elif dim=='ty'   : output = output[:, 0, np.newaxis, 0]
        elif dim=='tzy'  : output = output[:, :, np.newaxis, 0]
        elif dim=='tyx'  : output = output[:, 0, np.newaxis, :]
        elif dim=='zyx'  : output = output[0, :, np.newaxis, :]
        elif dim=='tzyx' : output = output[:, :, np.newaxis, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                    ", 'tzx', 'tx, 'zx', 'yx', 'x'")
    else :
        if   dim=='y'    : pass
        elif dim=='yx'   : output = output[0, 0, :]
        elif dim=='zy'   : output = output[0, :, 0]
        elif dim=='ty'   : output = output[:, 0, 0]
        elif dim=='tzy'  : output = output[:, :, 0]
        elif dim=='tyx'  : output = output[:, 0, :]
        elif dim=='zyx'  : output = output[0, :, :]
        elif dim=='tzyx' : output = output[:, :, :]
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                    ", 'tzx', 'tx, 'zx', 'yx', 'x'")
    #

    return output
# END yinterpol
####################################################################################################

####################################################################################################
def xinterpol(field, mesh, ilon, grid='T', dim='tzyx', keep_ndim=False, **kwargs) :
    """
    Interpolation of TZYX field at a given longitude.

    Usage: xinterpol(field, mesh, ilon, grid='T', dim='tzyx',
       keep_ndim=False, **kwargs)
    
    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the vertical mean, the field needs a vertical
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    ilon : float or array
       if it is float, field is interpolated at a fixed depth
       if it an array, the array as to fit the t, y and x dimension
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
       to specify the dimension of field
    keep_ndim : boolean
       to keep or not the number of dimensions
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> xinterpol")
    
    # add axis to always have 4 dimensions
    # when we average (axis=3)
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

    if   isinstance(ilon, (np.ma.core.MaskedArray, np.ndarray)) : 
        # eventually chech the dimension fits with field

        XX = mesh['lon'+grid][5]
        YY = mesh['lat'+grid][:, 5]
        ZZ = mesh['dep'+grid]
        TT = np.arange(nt) # time interval are equal
    
        interp_func = RegularGridInterpolator((TT, ZZ, YY, XX), zwfield, bounds_error=False,\
                                                  method="linear") 
        # create the list of points where zwfield is interpolated 
        pts = [] 
        for ind, val in np.ndenumerate(ilon) : 
            pts.append([ TT[ind[0]], ZZ[ind[1]], YY[ind[2]], val ]) 
        #
        pts = np.array(pts)
        # interpolation
        izwfield = interp_func(pts) #
        # reshape izwfield
        output = np.ma.array(np.reshape(izwfield, (nt, nz, ny)))
    elif isinstance(ilon, (float, int)) : 
        XX = mesh['lon'+grid][5, :]
        interpfn = interp1d(XX, zwfield, axis = 3, bounds_error=False)
        output = np.ma.array(interpfn(ilon))
        output.mask = zwfield.mask[:, :, :, 5]
    else: raise TypeError("ilon")

    if keep_ndim : 
        print("keep number of dimensions")
        if   dim=='x'    : output = output.squeeze()
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
        if   dim=='x'    : pass
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
# END xinterpol
####################################################################################################
