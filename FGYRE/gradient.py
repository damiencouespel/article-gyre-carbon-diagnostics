"""
Funcions to compute gradients
"""

# from IPython import embed
# import time
import numpy        as np

####################################################################################################
def gradz(field, mesh, grid='T', dim='tzyx', **kwargs) : 
    """
    Compute vertical gradient
    Usage : gradz(field, mesh, grid='T', dim='tzyx', **kwargs)

    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the vertical gradient, the field needs a vertical
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tzx', 'tz', 'zy', 'zx', 'z'
       to specify the dimension of field
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
        mean on the vertical axis, be careful field is masked after this operation
    """

    print("> gradz")

    # add axis to always have 4 dimensions
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

    # compute gradient
    output = np.zeros_like(zwfield)
    if grid in ['T', 'U', 'V'] : 
        output[:,    0, :, :] = np.float('nan')
        output[:, 1:-1, :, :] = ( zwfield[:, 1:-1, :, :] - zwfield[:, :-2, :, :] ) / \
                                mesh['e3w_0'][np.newaxis, 1:-1, np.newaxis, np.newaxis]
        output[:,   -1, :, :] = np.float('nan')
    elif grid == 'W' : 
        output[:, :-1, :, :] = ( zwfield[:, 1:, :, :] - zwfield[:, :-1, :, :] ) / \
                               mesh['e3t_0'][np.newaxis, :-1, np.newaxis, np.newaxis]
        output[:,  -1, :, :] = np.float('nan')
    else : raise ValueError("ERROR, grid has to be 'T', 'U', 'V' or 'W'")

    if   dim=='z'    : output = output[0, :, 0, 0]
    elif dim=='zx'   : output = output[0, :, 0, :]
    elif dim=='zy'   : output = output[0, :, :, 0]
    elif dim=='tz'   : output = output[:, :, 0, 0]
    elif dim=='zyx'  : output = output[0, :, :, :]
    elif dim=='tzx'  : output = output[:, :, 0, :]
    elif dim=='tzy'  : output = output[:, :, :, 0]
    elif dim=='tzyx' : output = output[:, :, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                                    ", 'tzx', 'tz', 'zy', 'zx', 'z'")
    return output
# END gradz
####################################################################################################

####################################################################################################
def gradx(field, mesh, grid='T', dim='tzyx', **kwargs) : 
    """
    Compute zonal gradient
    Usage : gradx(field, mesh, grid='T', dim='tzyx', **kwargs)

    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the zonal gradient, the field needs a zonal
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tyx', 'tzx', 'tx', 'yx', 'zx', 'x'
       to specify the dimension of field
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
    """

    print("> gradx")

    # add axis to always have 4 dimensions
    if   dim=='x'    : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, np.newaxis, :])
    elif dim=='zx'   : zwfield = np.ma.array(field.data[np.newaxis, :, np.newaxis, :])
    elif dim=='yx'   : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, :])
    elif dim=='tx'   : zwfield = np.ma.array(field.data[:, np.newaxis, np.newaxis, :])
    elif dim=='zyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tzx'  : zwfield = np.ma.array(field.data[:, :, np.newaxis, :])
    elif dim=='tyx'  : zwfield = np.ma.array(field.data[:, np.newaxis, :, :])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                ", 'tzx', 'tx', 'yx', 'zx', 'x'")

    # compute gradient
    output = np.zeros_like(zwfield)
    if grid in ['T', 'V', 'W'] : 
        output[:, :, :,    0] = np.float('nan')
        output[:, :, :, 1:-1] = ( zwfield[:, :, :, 1:-1] - zwfield[:, :, :, :-2] ) / \
                                mesh['e1u'][np.newaxis, np.newaxis, :, 1:-1]
        output[:, :, :,   -1] = np.float('nan')
    elif grid == 'U' : 
        output[:, :, :,  0] = np.float('nan')
        output[:, :, :, 1:] = ( zwfield[:, :, :, 1:] - zwfield[:, :, :, :-1] ) / \
                               mesh['e1t'][np.newaxis, np.newaxis, :, 1:]
    else : raise ValueError("ERROR, grid has to be 'T', 'U', 'V' or 'W'")

    if   dim=='x'    : output = output[0, 0, 0, :]
    elif dim=='zx'   : output = output[0, :, 0, :]
    elif dim=='yx'   : output = output[0, :, 0, :]
    elif dim=='tx'   : output = output[:, 0, 0, :]
    elif dim=='zyx'  : output = output[0, :, :, :]
    elif dim=='tzx'  : output = output[:, :, 0, :]
    elif dim=='tyx'  : output = output[:, 0, :, :]
    elif dim=='tzyx' : output = output[:, :, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tys'", \
                            ", 'tzx', 'tx', 'yx', 'zx', 'x'")
    return output
# END gradx
####################################################################################################

####################################################################################################
def grady(field, mesh, grid='T', dim='tzyx', **kwargs) : 
    """
    Compute meridional gradient
    Usage : grady(field, mesh, grid='T', dim='tzyx', **kwargs)

    Parameters
    ----------
    field : 4D, 3D, 2D or 1D masked array
       field to compute the meridional gradient, the field needs a meridional
       dimension
    mesh : mesh dictionary from read_mesh
       meshmask associated with field, used to get scale factors and
       mask
    grid : 'T' (default), 'U', 'V' or 'W'
       Field is defined at 'T' (or 'U' or 'V') points or 'W' point.
    dim : 'tzyx' (default), 'zyx', 'tzy', 'tyx', 'ty', 'zy', 'yx', 'y'
       to specify the dimension of field
    **kwargs : not passed
    Returns
    -------
    output : masked array same dimension as field
    """

    print("> grady")

    # add axis to always have 4 dimensions
    if   dim=='y'    : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, np.newaxis])
    elif dim=='zy'   : zwfield = np.ma.array(field.data[np.newaxis, :, :, np.newaxis])
    elif dim=='yx'   : zwfield = np.ma.array(field.data[np.newaxis, np.newaxis, :, :])
    elif dim=='ty'   : zwfield = np.ma.array(field.data[:, np.newaxis, :, np.newaxis])
    elif dim=='zyx'  : zwfield = np.ma.array(field.data[np.newaxis, :, :, :])
    elif dim=='tzy'  : zwfield = np.ma.array(field.data[:, :, :, np.newaxis])
    elif dim=='tyx'  : zwfield = np.ma.array(field.data[:, np.newaxis, :, :])
    elif dim=='tzyx' : zwfield = np.ma.array(field.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tyx'", \
                                ", 'tzy', 'ty', 'yx', 'zy', 'y'")

    # compute gradient
    output = np.zeros_like(zwfield)
    if grid in ['T', 'U', 'W'] : 
        output[:, :,    0, :] = np.float('nan')
        output[:, :, 1:-1, :] = ( zwfield[:, :, 1:-1, :] - zwfield[:, :, :-2, :] ) / \
                                mesh['e2v'][np.newaxis, np.newaxis, 1:-1, :]
        output[:, :,   -1, :] = np.float('nan')
    elif grid == 'V' : 
        output[:, :,  0, :] = np.float('nan')
        output[:, :, 1:, :] = ( zwfield[:, :, 1:, :] - zwfield[:, :, :-1, :] ) / \
                               mesh['e2t'][np.newaxis, np.newaxis, 1:, :]
    else : raise ValueError("ERROR, grid has to be 'T', 'U', 'V' or 'W'")

    if   dim=='y'    : output = output[0, 0, :, 0]
    elif dim=='zy'   : output = output[0, :, :, 0]
    elif dim=='yx'   : output = output[0, :, 0, :]
    elif dim=='ty'   : output = output[:, 0, :, 0]
    elif dim=='zyx'  : output = output[0, :, :, :]
    elif dim=='tzy'  : output = output[:, :, :, 0]
    elif dim=='tyx'  : output = output[:, 0, :, :]
    elif dim=='tzyx' : output = output[:, :, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'", \
                            ", 'tyx', 'tx', 'zy', 'yx', 'y'")
    return output
# END grady
####################################################################################################

