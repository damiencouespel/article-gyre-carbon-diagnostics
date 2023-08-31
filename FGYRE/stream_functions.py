"""
Functions to compute different stream functions
"""
from IPython import embed
import time

import numpy as np
from FGYRE import averaging

def bsfu(uu, mesh, dim = 'tzyx', zmin = None, zmax = None) :
    "Compute the barotropic stream function from the zonal velocity field :  u = - dPSI / dy"

    print("> bsfu")

    # add axis to always have 4 dimensions
    if   dim=='zyx'  : zwu = np.ma.array(uu.data[np.newaxis, :, :, :])
    elif dim=='tzyx' : zwu = np.ma.array(uu.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")

    # add the mask
    nt, nz, ny, nx = zwu.shape
    mask = mesh['umask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwu.mask = [x!=1 for x in mask]
    else: raise ValueError("ERROR, zwu doesn't have the right dimension")

    # integration sur z pour calculer le transport sur la colonne d'eau
    zwtp  = averaging.zmean(zwu * mesh['e2u'] * mesh['umask'], mesh, zmin=zmin, zmax=zmax, integral=True, grid='U')
    # cumul sur y, conversion en Sv
    psi   = - np.cumsum(zwtp, axis = 1) * 1.e-6          

    if   dim=='zyx'  : psi = psi[0, :, :]
    elif dim=='tzyx' : psi = psi[:, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")
    return psi
#

def bsfv(vv, mesh, dim = 'tzyx', zmin = None, zmax = None) :
    "Compute the barotropic stream function from the meridional velocity field :  v = dPSI / dx"

    print("> bsfv")

    # add axis to always have 4 dimensions
    if   dim=='zyx'  : zwv = np.ma.array(vv.data[np.newaxis, :, :, :])
    elif dim=='tzyx' : zwv = np.ma.array(vv.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")

    # add the mask
    nt, nz, ny, nx = zwv.shape
    mask = mesh['vmask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwv.mask = [x!=1 for x in mask]
    else: raise ValueError("ERROR, zwv doesn't have the right dimension")

    # integration sur z pour calculer le transport sur la colonne d'eau
    zwtp  = averaging.zmean(zwv * mesh['e1v'] * mesh['vmask'], mesh, zmin=zmin, zmax=zmax, integral=True, grid='V')
    # cumul sur x, conversion en Sv
    psi   = np.cumsum(zwtp, axis = 2) * 1.e-6          

    if   dim=='zyx'  : psi = psi[0, :, :]
    elif dim=='tzyx' : psi = psi[:, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")
    return psi
#

def msf(vv, mesh, dim='tzyx', sigma=None) : 
    """
    Compute the meridional stream functions from zonal and meridional
    velocities.  If sigma is not None, msf is computed along
    isopycnals. Sigma has to be the isopycnals with the same
    dimensions as vv.
    """
    
    print("> msf")
    
    
    if isinstance(sigma, type(vv)) :
        if (sigma.shape==vv.shape) : 
            GOsigma = True
        else : raise ValueError("Check sigma")
    else : GOsigma = False

    # add axis to always have 4 dimensions
    if   dim=='zyx'  : 
        zwv = np.ma.array(vv.data[np.newaxis, :, :, :])
    elif dim=='tzyx' : 
        zwv = np.ma.array(vv.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")
    
    # add the vmask
    nt, nz, ny, nx = zwv.shape
    mask = mesh['vmask'][np.newaxis]
    if (nz>1 and ny>1 and nx>1) : zwv.mask = [x!=1 for x in mask]
    else: raise ValueError("ERROR, zwv doesn't have the right dimension")

    if GOsigma :
        print("in sigma coordinate")        
        # add axis to always have 4 dimensions
        if   dim=='zyx'  : 
            zws = np.ma.array(sigma.data[np.newaxis, :, :, :])
        elif dim=='tzyx' : 
            zws = np.ma.array(sigma.data[:, :, :, :])
        else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx'")
        # add the tmask
        nt, nz, ny, nx = zws.shape
        mask = mesh['tmask'][np.newaxis]
        if (nz>1 and ny>1 and nx>1) : 
            zws.mask = [x!=1 for x in mask]
        else: raise ValueError("ERROR, zwt doesn't have the right dimension")
        
        smin, smax, sdel = 22., 25.5, .05
        sigv1 = np.arange(smin, smax, sdel)
        smin, smax, sdel = 25.55, 26.5, .005
        sigv2 = np.arange(smin, smax, sdel)
        sigv = np.concatenate((sigv1, sigv2))
        sigv[0], sigv[-1] = 0, 100
        sigv = sigv/1000.

        # # integration zonale
        # zwv = np.nansum(zwv * mesh['e1v'][np.newaxis, np.newaxis, :, :], \
        #                 axis = -1)
        # zws = np.nanmean(zws, axis=-1)
        # # zwv[ik] = transport entre les densite sigv[ik] et sigv[ik+1]
        # # z_sv[ik] = profondeur iso sigv[ik]
        # zwmsfv, z_sv = bin_veloc(zws, sigv, zwv, mesh, 'V', dim='tzy') 

        # zwv[ik] = transport entre les densite sigv[ik] et sigv[ik+1]
        # z_sv[ik] = profondeur iso sigv[ik]
        zwv, z_sv = bin_veloc(zws, sigv, zwv, mesh, 'V') 
        # integration zonale de v
        zwmsfv = np.nansum(zwv * mesh['e1v'][np.newaxis, np.newaxis, :, :], \
                        axis = -1)
        z_sv = np.nanmean(z_sv, axis=-1)
        
        # integration verticale
        res = np.nancumsum(zwmsfv, axis=1) * 1e-6 # res[ik] = transport entre sigv[0] et sigv[ik+1]
        res0 = np.zeros_like(res[:, 0, :])
        res = np.concatenate( (res0[:, np.newaxis, :], res[:, :-1, :]), axis=1) # res[ik] = transport entre sigv[0] et sigv[ik]
        # change edge of the mask, set to 0
        res[:, :, 0] = 0
        res[:, :, -2] = 0
        # # soustraire la valeur de fond (equivalent to upward integration)
        # ref = res[:, -1, :]
        # res = res - ref[:, np.newaxis, :]
        if dim=='zyx' : 
            res  = res[0]
            z_sv = z_sv[0]
        return res, sigv, z_sv
    else : # END fi GOsigma
        print("in z coordinate")
        # integration zonale de v
        zwv = np.nansum(zwv * mesh['e1v'][np.newaxis, np.newaxis, :, :], axis = -1)
        # vertical integral
        res = np.nancumsum(zwv * mesh['e3v'][np.newaxis, :, :, 0], axis=1) * 1e-6 
        # change edge of the mask, set to 0
        res[:, :-1, 0] = 0
        res[:, :-1, -2] = 0
        res.mask[:, :-1, 0] = False
        res.mask[:, :-1, -2] = False
        # soustraire la valeur de fond (equivalent to upward integration)
        # ref = res[:, -2, :]
        # res = res - ref[:, np.newaxis, :]
        if dim=='zyx' : res = res[0]
        return res
    #
#

def bin_veloc(sig, sigv, xxx, mesh, grid, dim='tzyx') : 
    
    print("> bin_veloc")

    from scipy.interpolate import RegularGridInterpolator, interp1d

    # add axis to always have 4 dimensions
    if dim=='tzy' : 
        zwxxx = np.ma.array(xxx.data[:, :, :, np.newaxis])
        zwsig = np.ma.array(sig.data[:, :, :, np.newaxis])
    elif dim=='zyx'  : 
        zwxxx = np.ma.array(xxx.data[np.newaxis, :, :, :])
        zwsig = np.ma.array(sig.data[np.newaxis, :, :, :])
    elif dim=='tzyx' : 
        zwxxx = np.ma.array(xxx.data[:, :, :, :])
        zwsig = np.ma.array(sig.data[:, :, :, :])
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'")

    nt, nz, ny, nx = zwxxx.shape
    
    # set mesh mask to right dimensions
    e3t = np.repeat(mesh['e3t'][np.newaxis], nt, axis=0)
    mask = np.repeat(mesh[grid.lower()+'mask'][np.newaxis], nt, axis=0)
    if dim=='tzy' : 
        e3t = e3t[:,:,:,5]
        e3t = e3t[:,:,:,np.newaxis]
        mask = mask[:,:,:,5]
        mask = mask[:,:,:,np.newaxis]
    elif dim=='zyx' : 
        e3t = e3t[0,:,:,:]
        e3t = e3t[np.newaxis,:,:,:]
        mask = mask[0,:,:,:]
        mask = mask[np.newaxis,:,:,:]
    elif dim=='tzyx' : 
        e3t = e3t[:,:,:,:]
        mask = mask[:,:,:,:]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'")

    # add the mask
    if dim=='tzy' : 
        zwxxx.mask = [x!=1 for x in mask]
        zwsig.mask = [x!=1 for x in mask]
    elif dim=='zyx' : 
        zwxxx.mask = [x!=1 for x in mask]
        zwsig.mask = [x!=1 for x in mask]
    elif dim=='tzyx' : 
        zwxxx.mask = [x!=1 for x in mask]
        zwsig.mask = [x!=1 for x in mask]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'")

    if grid == 'U' : 
        # sigma in U point
        zwsig[:, :, :, :-1] = .5 * (zwsig[:, :, :, :-1] + zwsig[:, :, :, 1:])
    elif grid == 'V' : 
        # sigma in V point
        zwsig[:, :, :-1, :] = .5 * (zwsig[:, :, :-1, :] + zwsig[:, :, 1:, :])
    else : raise ValueError("grid has to be 'U' or 'V'")
    zwsig = zwsig * mask
    
    # initializations
    s_s = sigv         # density range over which we bin
    ns = len(sigv)     # number of density levels
    # Profiles along z level ( suffixe _z)
    c1_z = np.zeros(nz)         # profil du contenu vertical de x
    s_z  = np.zeros(nz)         # profil de la densite
    z_zt = mesh['depT']               # profondeur au point T  (k=0 -> 5m)
    z_zw = np.roll(mesh['depW'], -1)  # profondeur au point W  (k=0 -> 10m)
    z_zw[-1]=mesh['depW'][-1]
    # Profiles along density level (suffixe _s)
    z_s  = np.zeros(ns+1)
    c1_s = np.zeros(ns+1)
    xxx_s = np.zeros(ns+1)
    # xxx binned on density (output array)
    z_sig  = np.zeros((nt, ns, ny, nx))
    xxx_bin = np.zeros((nt, ns, ny, nx))
    # vertical content of xxx (ensure  the integrale conservation)
    xxx_content = np.cumsum( zwxxx * e3t, axis = 1)

    #  Loop over time and horizontal domain 2D
    for tt in range(nt) : 
        for jj in range(ny) : 
            for ii in range(nx) : 

                #  Indices des points U et V dans l ocean
                i_ocean, = np.where( mask[0, :, jj, ii] == 1 )
                
                z_s   = z_s*0.
                c1_s  = c1_s*0.
                xxx_s = xxx_s*0.
                
                if (len(i_ocean) > 0) : # on n'entre que si il y a des points ocean
                    
                    i_bottom = i_ocean[-1]
                    s_z  = zwsig[tt, :, jj, ii]
                    c1_z = xxx_content[tt, :, jj, ii]

                    # critere supplementaire a imposer sur le profil pour eviter les cas
                    # pathologiques en attendant d'ecrire une vraie commande d'extraction
                    # de profils strictement croissant. Il s'agit donc d'un test adhoc.
                    # --------------------
                    # ds = (shift(s_z, -1)-s_z)(i_ocean(0:n_elements(i_ocean)-2))
                    # ds(0) = 0
                    # ind = where(ds < 0.)
                    # croissant =  ind[0] == -1
                    if s_z[0] < s_z[i_bottom] : 

                        z_s[ns] = z_zw[i_bottom]
                        c1_s[ns] = xxx_content[tt, i_bottom, jj, ii]
                        # extraction d'un sous profil strictement croissant
                        mini = np.min( s_z[i_ocean] )
                        maxi = np.max( s_z[i_ocean] )

                        i_min, = np.where( s_z[i_ocean] == mini )
                        i_max, = np.where( s_z[i_ocean] == maxi )
                        #   on prend le plus grand des indices min
                        #   et le plus petit des indices max
                        i_min = i_min[0]
                        i_max = i_max[-1]

                        # Si la valeur du niveau (s_s) est plus faible que la densite de surface,
                        # l isopycne est mise en surface (z_s=0)
                        #
                        ind, = np.where(s_s < s_z[i_min])
                        if len(ind) > 0 :
                            z_s[ind[:-1]]  = 0
                            c1_s[ind[:-1]] = 0
                            z_s[ind[-1]]  = 0      
                            c1_s[ind[-1]] = 0
                            xxx_s[ind[0]-1] = c1_s[ind[0]] - c1_s[ind[0]-1] 
                            xxx_s[ind]      = c1_s[ind+1] - c1_s[ind]
                        #
                        # Si la valeur du niveau (s_s) est plus elevee que la densite du fond,
                        # l isopycne est mise au fond (z_s=z_zw(i_bottom))
                        ind, = np.where(s_s > s_z[i_max])
                        if len(ind) > 0 :
                            z_s[ind[1:]]  = z_s[ns]       
                            c1_s[ind[1:]] = c1_s[ns]
                            z_s[ind[0]]   = z_s[ns]       
                            c1_s[ind[0]]  = c1_s[ns]
                            xxx_s[ind[0]-1] = c1_s[ind[0]] - c1_s[ind[0]-1] 
                            xxx_s[ind]      = c1_s[ind+1] - c1_s[ind]
                        #

                        ds = np.roll(s_z, -1) - s_z
                        ds = ds[i_ocean[0:-2]]
                        ind_c, = np.where(ds < 0.)
                        if ( len(ind_c) == 0) :
                            ind, = np.where( (s_s >= s_z[i_min]) & (s_s < s_z[i_max]) )
                            if ( len(ind) > 0 ) :
                                if ( len(ind) > 1 ) :
                                    i_profil = i_ocean[i_min:i_max]
                                    interpfn = interp1d(s_z[i_profil], z_zt[i_profil], \
                                                        bounds_error=False)
                                    z_s[ind] = interpfn(s_s[ind])
                                    # fonction spline pour interpoler le contenu
                                    interpfn = interp1d(z_zw[i_profil], c1_z[i_profil], \
                                                        bounds_error=False, kind='cubic')
                                    c1_s[ind] = interpfn(z_s[ind])
                                    # l'interpolation lineaire marche aussi. Elle donne 
                                    # des resultats differents. Elle
                                    # me semble moins propre. Je la donne en remarque.
                                    # interpfn = interp1d(z_zw_z[i_profil], c1_z[i_profil], \
                                    #                     bounds_error=False)
                                    # c1_s = interpfn(z_s[ind])
                                    # Remarque : on ne divise pas par l'epaisseur de la couche
                                elif ( len(ind) == 1 ) : 
                                    # cas pathologique (tout dans 1 gamme de densite)
                                    c1_s[ind] = c1_z[-1]
                                #
                                xxx_s[ind[0]-1] = c1_s[ind[0]] - c1_s[ind[0]-1] 
                                xxx_s[ind]      = c1_s[ind+1] - c1_s[ind]
                            #
                        #
                    #
                #
                z_sig[tt, :, jj, ii] = z_s[:-1]
                xxx_bin[tt, :, jj, ii] = xxx_s[:-1]
            #
        #
    #
    if dim=='tzy' : 
        xxx_bin = xxx_bin[:, :, :, 0]
        z_sig = z_sig[:, :, :, 0]
    elif dim=='zyx'  : 
        xxx_bin = xxx_bin[0, :, :, :]
        z_sig   = z_sig[0, :, :, :]
    elif dim=='tzyx' : 
        xxx_bin = xxx_bin[:, :, :, :]
        z_sig   = z_sig[:, :, :, :]
    else : raise ValueError("ERROR, dim has to be 'tzyx', 'zyx', 'tzy'")

    return xxx_bin, z_sig
# end def bin_veloc
