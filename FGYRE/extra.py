import numpy as np

####################################################################################################
def cdtime2str(date, year=True, month=False, day=False, hour=False, minute=False, second=False,\
                   **kwargs) : 
    """
    Transform a list of component time (from cdms) to a list of string
    Usage: cdtime2str(date, year=True, month=False, day=False,
       hour=False, minute=False, second=False, **kwargs)
    """
    print("> cdtime2str")
    output = []
    for zw in date : 
        zw1 = ''
        if year   : zw1 = str(zw.year)+'.'
        if month  : zw1 = zw1+str(zw.month) +'.'
        if day    : zw1 = zw1+str(zw.day)   +'.'
        if hour   : zw1 = zw1+str(zw.hour)  +'.'
        if minute : zw1 = zw1+str(zw.minute)+'.'
        if second : zw1 = zw1+str(zw.second)+'.'
        output.append(zw1)
    #
    return output
# END cdtime2str
####################################################################################################

####################################################################################################
def avg_onRX(zin, zoutmesh, zfactor) :
    """
    Average zin on a larger grid (zoutmesh)
    Param : 
       zin: numpy array input
       zoutmesh: mesh dict (reading.read_mesh)
       zfactor: factor between zin grid and zoutmesh grid
    """
    
    zfactor = int(zfactor)
    zw = zin[1:-1, 1:-1] # remove edge
    # only on T grid
    zout = np.zeros(( zw.shape[0]//zfactor + 2, zw.shape[1]//zfactor + 2 ))
    # reshape to get subarray of shape zfactor*zfactor
    zw = np.reshape(zw, (zout.shape[0]-2, zfactor, zout.shape[1]-2, zfactor))
    zw = np.nanmean(zw, axis = (1, 3)) # mean the subarray
    zout[1:-1, 1:-1] = zw # fill output
    # fill corner
    zout[ 0,  0] = zin[ 0,  0]
    zout[ 0, -1] = zin[ 0, -1]
    zout[-1,  0] = zin[-1,  0]
    zout[-1, -1] = zin[-1, -1]
    # fill edge
    zout[1:-1,    0] = zin[zfactor//2+1::zfactor,                   0]
    zout[1:-1,   -1] = zin[zfactor//2+1::zfactor,                  -1]
    zout[   0, 1:-1] = zin[                  0, zfactor//2+1::zfactor]
    zout[  -1, 1:-1] = zin[                 -1, zfactor//2+1::zfactor]
    zout = np.ma.array(zout)
    zout.mask=[x!=1 for x in zoutmesh['tmask']]
    return zout
#
####################################################################################################
