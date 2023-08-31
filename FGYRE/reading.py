"""
Function to read mesh and ncdf files
"""

# from IPython import embed
# import time
import numpy        as np
import cdms2

####################################################################################################
def read_ncdf(vname, filename, ndarray=True, **kwargs) :
    """
    Read a ncdf file with cdms2
    Usage : output = read_ncdf(vname, filename, **kwargs)

    Parameters
    ----------
    vname : string
        name of the variable to extract
    filename : string
        path of the file
    ndarray : boolean
       to return numpy masked array
    **kwargs are pass to cdms2 dataset. It can be time=(0161-01-01,
      0170-12-31)
    Returns
    -------
    a directory with :
       data : 4D array containing the values 
       date : list containing date
       dept : containing depth
       lati : YX array containing latitudes
       long : YX array containing longtitudes
    """
    print('> read_ncdf')
    print(' filename = ', filename)
    print(' vname    = ', vname)
    
    zwftr  = cdms2.open(filename)
    zwtr   = zwftr(vname, **kwargs)
    zwlati = zwtr.getLatitude()
    zwlong = zwtr.getLongitude()
    zwdept = zwtr.getLevel()
    zwdate = zwtr.getTime().asComponentTime()
    zwftr.close()
    
    output={}
    if ndarray: 
        output['data']=np.ma.array(zwtr.getValue())
        output['date']=zwdate
        if zwdept : output['dept']=np.ma.array(zwdept.getValue())
        output['lati']=np.ma.array(zwlati.getValue())
        output['long']=np.ma.array(zwlong.getValue())
        output['vname']=vname
        output['filename']=filename
    else: 
        output['data']=zwtr
        output['date']=zwdate
        if zwdept : output['dept']=zwdept
        output['lati']=zwlati
        output['long']=zwlong
        output['vname']=vname
        output['filename']=filename
    return output
# END read_ncdf
####################################################################################################

####################################################################################################
def read_mesh(meshfile, **kwargs):
    """
    Read and return an dictionnary  with all the value of the mesh mask

    Usage : mesh = read_mesh(filename)

    Parameters
    ----------
    filename: string
       path and the filename of the mesh
    """
    print('> read_mesh')
    print(' meshfile  = ',meshfile)

    output = {}
    # mesh_mask = nc.Dataset(meshfile, 'r', 'float64')
    mesh = cdms2.open(meshfile)
    output['tmask' ] =  mesh('tmask'  , squeeze = 1, **kwargs).getValue()
    output['umask' ] =  mesh('umask'  , squeeze = 1, **kwargs).getValue()
    output['vmask' ] =  mesh('vmask'  , squeeze = 1, **kwargs).getValue()
    output['fmask' ] =  mesh('fmask'  , squeeze = 1, **kwargs).getValue()

    output['latT'  ] =  mesh('gphit'  , squeeze = 1, **kwargs).getValue()
    output['lonT'  ] =  mesh('glamt'  , squeeze = 1, **kwargs).getValue()
    output['latU'  ] =  mesh('gphiu'  , squeeze = 1, **kwargs).getValue()
    output['lonU'  ] =  mesh('glamu'  , squeeze = 1, **kwargs).getValue()
    output['latV'  ] =  mesh('gphiv'  , squeeze = 1, **kwargs).getValue()
    output['lonV'  ] =  mesh('glamv'  , squeeze = 1, **kwargs).getValue()
    output['latF'  ] =  mesh('gphif'  , squeeze = 1, **kwargs).getValue()
    output['lonF'  ] =  mesh('glamf'  , squeeze = 1, **kwargs).getValue()

    output['e1t'   ] =  mesh('e1t'    , squeeze = 1, **kwargs).getValue()
    output['e1u'   ] =  mesh('e1u'    , squeeze = 1, **kwargs).getValue()
    output['e1v'   ] =  mesh('e1v'    , squeeze = 1, **kwargs).getValue()
    output['e1f'   ] =  mesh('e1f'    , squeeze = 1, **kwargs).getValue()

    output['e2t'   ] =  mesh('e2t'    , squeeze = 1, **kwargs).getValue()
    output['e2u'   ] =  mesh('e2u'    , squeeze = 1, **kwargs).getValue()
    output['e2v'   ] =  mesh('e2v'    , squeeze = 1, **kwargs).getValue()
    output['e2f'   ] =  mesh('e2f'    , squeeze = 1, **kwargs).getValue()

    output['e3t'   ] =  mesh('e3t'    , squeeze = 1, **kwargs).getValue()
    output['e3u'   ] =  mesh('e3u'    , squeeze = 1, **kwargs).getValue()
    output['e3v'   ] =  mesh('e3v'    , squeeze = 1, **kwargs).getValue()
    output['e3w'   ] =  mesh('e3w'    , squeeze = 1, **kwargs).getValue()

    output['e3t_0' ] =  mesh('e3t_0'  , squeeze = 1, **kwargs).getValue()
    output['e3w_0' ] =  mesh('e3w_0'  , squeeze = 1, **kwargs).getValue()

    output['depT'  ] =  mesh('gdept_0', squeeze = 1, **kwargs).getValue()
    output['depW'  ] =  mesh('gdepw_0', squeeze = 1, **kwargs).getValue()

    output['ff'    ] =  mesh('ff'     , squeeze = 1, **kwargs).getValue()
    output['mbathy'] =  mesh('mbathy' , squeeze = 1, **kwargs).getValue()

    output['wmask' ] = output['tmask' ]
    output['latW'  ] = output['latT'  ]
    output['lonW'  ] = output['lonT'  ]
    output['e1w'   ] = output['e1t'   ]
    output['e2w'   ] = output['e2t'   ]

    output['e3u_0' ] = output['e3t_0' ]
    output['e3v_0' ] = output['e3t_0' ]
    output['depU'  ] = output['depT'  ]
    output['depV'  ] = output['depT'  ]
    
    zw = mesh('tmask'  , squeeze = 1).getValue()
    output['jpi'] = zw.shape[2]
    output['jpj'] = zw.shape[1]
    output['jpk'] = zw.shape[0]
    
    return output
# END read_mesh
####################################################################################################
