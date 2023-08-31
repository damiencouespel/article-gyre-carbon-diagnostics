"""
Functions to compute density, brunt-vaisala frequency...
"""

from IPython import embed
# import time
import numpy        as np

####################################################################################################
def eos_insitu(temp, sali, alpha=2.0e-4, beta=7.7e-4, **kwargs) :
    """
    Compute the in situ density from potential temperature and
    salinity using a bilinear equation of state : prd = beta * sali -
    alpha * temp
    
    Usage : eos_insitu(temp, sali, alpha=2.0e-4, beta=7.7e-4)
    
    Parameters
    ----------
    temp : 3D or 4D numpy array
       potential temperature in [Celsius] 
    sali  : 3D or 4D numpy array
       salinity in [psu]
    alpha : float
       temperature coefficient in [1/Celsius]
    beta  : float
       salinity coefficient in [1/psu]
    **kwargs : not passed

    Returns
    -------
    prd : 3D or 4D numpy array containin insitu density [no unit]
    """

    print("> eos_instu")
    prd   = beta  * sali - alpha * temp
    return prd
# END eos_insitu
####################################################################################################

####################################################################################################
def eos_insitu_pot(temp, sali, rau0=1035.0, **kwargs) :
    """
    Compute the in situ potential density from potential temperature
    and salinity using a bilinear equation of state :
    prho   = rau0 * ( 1.0 +  prd)
    prd is th in situ density : prd = beta  * sali - alpha * temp

    Usage : eos_insitu_pot(temp, sali, rau0 = 1035.0)

    Parameters
    ----------
    temp  : 3D or 4D numpy array
       potential temperature in [Celsius] 
    sali  : 3D or 4D numpy array
       salinity in [psu]
    rau0  : float
       ref density in [kg/m3]
    **kwargs : passed to eos_insitu
    can be alpha=float or beta=float

    Returns
    -------
    prho : 3D or 4D numpy array containing insitu potential density in
    [kg/m3]
    """

    print("> eos_instu_pot")
    prd  = eos_insitu(temp, sali, **kwargs)
    prho = (1+ prd ) * rau0
    return prho
# END eos_insit_pot
####################################################################################################

####################################################################################################
def eos_bn2(temp, sali, e3w, dim='tzyx', alpha=2.0e-4, beta=7.7e-4) :
    """
    Compute the local Brunt-Vaisala frequency using a linear equation
    of state : 
    grav  = 9.80665     # gravity [m/s2]
    pn2 = grav * (alpha * dk[ temp ] - beta * dk[ sali ] ) / e3w 
    NB : N^2 is set to zero at the first level
    and is never used at this level.

    Usage: eos_bn2(temp, sali, e3w, dim='tzyx', alpha=2.0e-4,
    beta=7.7e-4)

    Parameters
    ----------
    temp : 3D or 4D numpy array
       potential temperature in [Celsius] 
    sal  : 3D or 4D numpy array
       salinity in [psu]
    e3w  : 3D numpy array
       vertical aspect ratio between T point
    dim  : string
       specify the dimension of the fields
    alpha : float
       temperature coefficient in [1/Celsius]
    beta  : float
       salinity coefficient in [1/psu]
    
    Returns
    -------
    pn2 : Brun-Vaisala frequency in [s-2]
    """
    print("> eos_bn2")
    grav = 9.80665     # gravity [m/s2]
    if dim != 'tzyx':
        temp = temp[np.newaxis] 
        sali = sali[np.newaxis] 
    #
    e3w = e3w[np.newaxis]
    pn2 = np.zeros_like(temp) + np.float('nan')
    pn2[:, 1:] = grav * ( alpha * (temp[:, :-1] - temp[:, 1:]) \
                              - beta  * (sali[:, :-1] - sali[:, 1:]) ) / e3w[:, 1:]
    if dim != 'tzyx': pn2 = pn2[0]
    return pn2
# END eos_bn2
####################################################################################################

####################################################################################################
def ldf_slp_mxl(prd, pn2, p_gru, p_grv, p_dzr, nmln, mesh) :
    #   !!----------------------------------------------------------------------
    #   !!                  ***  ROUTINE ldf_slp_mxl  ***
    #   !!
    #   !! ** Purpose :   Compute the slopes of iso-neutral surface just below
    #   !!              the mixed layer.
    #   !!
    #   !! ** Method  :   The slope in the i-direction is computed at u- & w-points
    #   !!              (uslpml, wslpiml) and the slope in the j-direction is computed
    #   !!              at v- and w-points (vslpml, wslpjml) with the same bounds as
    #   !!              in ldf_slp.
    #   !!
    #   !! ** Action  :   uslpml, wslpiml :  i- &  j-slopes of neutral surfaces
    #   !!                vslpml, wslpjml    just below the mixed layer
    #   !!                omlmask         :  mixed layer mask
    #   !!----------------------------------------------------------------------
    #   REAL(wp), DIMENSION(:,:,:), INTENT(in) ::   prd            ! in situ density
    #   REAL(wp), DIMENSION(:,:,:), INTENT(in) ::   pn2            ! Brunt-Vaisala frequency (locally ref.)
    #   REAL(wp), DIMENSION(:,:,:), INTENT(in) ::   p_gru, p_grv   ! i- & j-gradient of density (u- & v-pts)
    #   REAL(wp), DIMENSION(:,:,:), INTENT(in) ::   p_dzr          ! z-gradient of density      (T-point)
    #   !!
    #   INTEGER  ::   ji , jj , jk         ! dummy loop indices
    #   INTEGER  ::   iku, ikv, ik, ikm1   ! local integers
    #   REAL(wp) ::   zeps, zm1_g, zm1_2g            ! local scalars
    #   REAL(wp) ::   zci, zfi, zau, zbu, zai, zbi   !   -      -
    #   REAL(wp) ::   zcj, zfj, zav, zbv, zaj, zbj   !   -      -
    #   REAL(wp) ::   zck, zfk,      zbw             !   -      -
    #   !!----------------------------------------------------------------------

    print('> ldf_slp_mxl Calculation of isoneutral slopes at mixed layer base')

    jpi = mesh['jpi']
    jpj = mesh['jpj']
    jpk = mesh['jpk']
    jpjm1 = jpj-1
    jpim1 = jpi-1
    jpkm1 = jpk-1
    e1u = mesh['e1u']
    e1t = mesh['e1t']
    e2v = mesh['e2v']
    e2t = mesh['e2t']
    e3t = mesh['e3t']
    e3w = mesh['e3w']
    fse3u = e3t
    fse3v = e3t
    fse3w = e3w
    tmask = mesh['tmask']
    umask = mesh['umask']
    vmask = mesh['vmask']

    zeps   =  1.e-20     # ! = =   Local constant initialization   = = !
    grav = 9.80664999999999942     # %gravite  m/2
    zm1_g  = -1.0/ grav
    zm1_2g = -0.5/ grav 

    uslpml  = np.zeros((jpj, jpi))
    vslpml  = np.zeros((jpj, jpi))
    wslpiml = np.zeros((jpj, jpi))
    wslpjml = np.zeros((jpj, jpi))

    uslpml[:, 0]      = 0
    uslpml[:, jpi-1]  = 0
    vslpml[:, 0]      = 0
    vslpml[:, jpi-1]  = 0
    wslpiml[:, 0]     = 0
    wslpiml[:, jpi-1] = 0
    wslpjml[:, 0]     = 0
    wslpjml[:, jpi-1] = 0

    #    ! Slopes of isopycnal surfaces just before bottom of mixed layer
    #    ! --------------------------------------------------------------
    #    ! The slope are computed as in the 3D case.
    #    ! A key point here is the definition of the mixed layer at u- and v-points.
    #    ! It is assumed to be the maximum of the two neighbouring T-point mixed layer depth.
    #    ! Otherwise, a n2 value inside the mixed layer can be involved in the computation
    #    ! of the slope, resulting in a too steep diagnosed slope and thus a spurious eddy
    #    ! induce velocity field near the base of the mixed layer.
    #    !-----------------------------------------------------------------------
    #    !

    for jj in range(1, jpjm1) : 
        for ii in range(1, jpim1) : 
            #  !==   Slope at u- & v-points just below the Mixed Layer   ==!
            #  
            #        !- vertical density gradient for u- and v-slopes (from dzr at T-point)
            iku = np.min( [ np.max( [0, nmln[jj, ii], nmln[jj, ii+1]] ), jpkm1-1 ] )  
            ikv = np.min( [ np.max( [0, nmln[jj, ii], nmln[jj+1, ii]] ), jpkm1-1 ] )             
            zbu = 0.5 * ( p_dzr[iku, jj, ii] + p_dzr[iku, jj, ii+1] )             
            zbv = 0.5 * ( p_dzr[ikv, jj, ii] + p_dzr[ikv, jj+1, ii] )             
            #  !- horizontal density gradient at u- & v-points
            zau = p_gru[iku, jj, ii] / e1u[jj, ii]          
            zav = p_grv[ikv, jj, ii] / e2v[jj, ii]          
            # !- bound the slopes: abs(zw.)<= 1/100 and zb..<0
            #    kxz max= ah slope max =< e1 e3 /(pi**2 2 dt)
            zbu = np.min( [ zbu, -100 * np.abs(zau), -7e+3 / fse3u[iku, jj, ii] * np.abs(zau) ] )
            zbv = np.min( [ zbv, -100 * np.abs(zav), -7e+3 / fse3v[ikv, jj, ii] * np.abs(zav) ] )
            #  !- Slope at u- & v-points (uslpml, vslpml)
            uslpml[jj, ii] = zau / ( zbu - zeps ) * umask[iku, jj, ii]
            vslpml[jj, ii] = zav / ( zbv - zeps ) * vmask[ikv, jj, ii]
            #
            #  !==   i- & j-slopes at w-points just below the Mixed Layer   ==!
            #
            ik   = np.min( [ nmln[jj, ii]+1, jpk-1 ] )
            ikm1 = np.max( [ 0, ik-1 ] )
            # !- vertical density gradient for w-slope (from N^2)
            zbw = zm1_2g * pn2[ik, jj, ii] * ( prd[ik, jj, ii] + prd[ikm1, jj, ii] + 2 )
            # !- horizontal density i- & j-gradient at w-points
            zci = np.max( [ umask[ik, jj, ii-1] + umask[ik, jj, ii] + umask[ikm1, jj, ii-1]
                            + umask[ikm1, jj, ii], zeps ] ) * e1t[jj, ii]
            zcj = np.max( [ vmask[ik, jj-1, ii] + vmask[ik, jj, ii] + vmask[ikm1, jj-1, ii]
                            + vmask[ikm1, jj, ii], zeps ] ) * e2t[jj, ii]
            zai = ( p_gru[ik, jj, ii-1] + p_gru[ik, jj, ii] + p_gru[ikm1, jj, ii-1]
                    + p_gru[ikm1, jj, ii] ) / zci * tmask[ik, jj, ii]
            zaj = ( p_grv[ik, jj-1, ii] + p_grv[ik, jj, ii] + p_grv[ikm1, jj-1, ii]
                    + p_grv[ikm1, jj, ii] ) / zcj * tmask[ik, jj, ii]
            # !- bound the slopes: abs(zw.)<= 1/100 and zb..<0.
            # kxz max= ah slope max =< e1 e3 /(pi**2 2 dt)
            zbi = np.min( [ zbw, -100 * np.abs(zai), -7.e+3 / fse3w[ik, jj, ii] * np.abs(zai) ] )          
            zbj = np.min( [ zbw, -100 * np.abs(zaj), -7.e+3 / fse3w[ik, jj, ii] * np.abs(zaj) ] )          
            # !- i- & j-slope at w-points (wslpiml, wslpjml)
            wslpiml[jj, ii] = zai / ( zbi - zeps ) * tmask[ik, jj, ii]
            wslpjml[jj, ii] = zaj / ( zbj - zeps ) * tmask[ik, jj, ii]
        #
    #
    slpml = np.zeros((4, jpj, jpi))
    slpml[0] = uslpml
    slpml[1] = vslpml
    slpml[2] = wslpiml
    slpml[3] = wslpjml
    
    return slpml
####################################################################################################

####################################################################################################
def ldf_slp( prd, pn2, nmln, mld, omlmask, mesh) : 
    #----------------------------------------------------------------------
    #
    # ** Purpose :   Compute the slopes of neutral surface (slope of isopycnal
    #              surfaces referenced locally) (ln_traldf_iso=T).
    #
    # ** Method  :   The slope in the i-direction is computed at U- and
    #      W-points (uslp, wslpi) and the slope in the j-direction is
    #      computed at V- and W-points (vslp, wslpj).
    #      They are bounded by 1/100 over the whole ocean, and within the
    #      surface layer they are bounded by the distance to the surface
    #      ( slope<= depth/l  where l is the length scale of horizontal
    #      diffusion (here, aht=2000m2/s ==> l=20km with a typical velocity
    #      of 10cm/s)
    #        A horizontal shapiro filter is applied to the slopes
    #        ln_sco=T, s-coordinate, add to the previously computed slopes
    #      the slope of the model level surface.
    #        macro-tasked on horizontal slab (jk-loop)  (2, jpk-1)
    #      [slopes already set to zero at level 1, and to zero or the ocean
    #      bottom slope (ln_sco=T) at level jpk in inildf]
    #
    # ** Action : - uslp, wslpi, and vslp, wslpj, the i- and  j-slopes
    #               of now neutral surfaces at u-, w- and v- w-points, resp.
    #----------------------------------------------------------------------
    # USE wrk_nemo, ONLY: wrk_in_use, wrk_not_released
    # USE oce     , ONLY:   zgru => ua       , zww => va   ! (ua,va) used as workspace
    # USE oce     , ONLY:   zgrv => ta       , zwz => sa   ! (ta,sa) used as workspace
    # USE wrk_nemo, ONLY:   zdzr => wrk_3d_1               ! 3D workspace
    # !!
    # INTEGER , INTENT(in)                   ::   kt    ! ocean time-step index
    # REAL(wp), INTENT(in), DIMENSION(:,:,:) ::   prd   ! in situ density
    # REAL(wp), INTENT(in), DIMENSION(:,:,:) ::   pn2   ! Brunt-Vaisala frequency (locally ref.)
    # !!
    # INTEGER  ::   ji , jj , jk    ! dummy loop indices
    # INTEGER  ::   ii0, ii1, iku   ! temporary integer
    # INTEGER  ::   ij0, ij1, ikv   ! temporary integer
    # REAL(wp) ::   zeps, zm1_g, zm1_2g, z1_16     ! local scalars
    # REAL(wp) ::   zci, zfi, zau, zbu, zai, zbi   !   -      -
    # REAL(wp) ::   zcj, zfj, zav, zbv, zaj, zbj   !   -      -
    # REAL(wp) ::   zck, zfk,      zbw             !   -      -

    print("> ldf_slp")

    fse3u = mesh['e3t']
    fse3v = mesh['e3t']
    fse3w = mesh['e3w']
    e1u = mesh['e1u']
    e2v = mesh['e2v']
    e1t = mesh['e1t']
    e2t = mesh['e2t']
    fsdept = mesh['depT']
    fsdepw = mesh['depW']
    jpi = mesh['jpi']
    jpj = mesh['jpj']
    jpk = mesh['jpk']
    umask = mesh['umask']
    vmask = mesh['vmask']
    tmask = mesh['tmask']

    wslpi = np.zeros((jpk, jpj, jpi))
    wslpj = np.zeros((jpk, jpj, jpi))
    hmlpt = np.zeros((jpj, jpi))

    for ii in range(jpi) : 
        for jj in range(jpj) :
            hmlpt[jj, ii] = fsdept[nmln[jj, ii]]
        #
    #

    hmlp  = mld
    jpjm1 = jpj-1
    jpim1 = jpi-1
    jpkm1 = jpk-1
    grav  = 9.80664999999999942   # gavite m/s
    zeps   =  1.e-20   # !==   Local constant initialization   ==!
    z1_16  =  1./16    
    zm1_g  = -1./grav  
    zm1_2g = -0.5/grav              
    
    zww  = np.zeros((jpk, jpj, jpi))
    zwz  = np.zeros((jpk, jpj, jpi))
    zgru = np.zeros((jpk, jpj, jpi))
    zgrv = np.zeros((jpk, jpj, jpi))
      
    print('Calculation of isoneutral slopes')
    for kk in range(jpk) : #==   i- & j-gradient of density   ==!
        for jj in range(jpjm1) :
            for ii in range(jpim1) :# vector opt.
                zgru[kk, jj, ii] = umask[kk, jj, ii] * (prd[kk, jj, ii+1] - prd[kk, jj, ii]) 
                zgrv[kk, jj, ii] = vmask[kk, jj, ii] * (prd[kk, jj+1, ii] - prd[kk, jj, ii]) 
            #
        #
    #
    zdzr = np.zeros((jpk, jpj, jpi))# Local vertical density gradient at T-point   == !   (evaluated from N^2)      

    for kk in range(1, jpkm1) : 
        # ! zdzr = d/dz(prd)= - ( prd ) / grav * mk(pn2) -- at t point
        # !   trick: tmask(ik  )  = 0   =>   all pn2   = 0   =>   zdzr = 0
        # !    else  tmask(ik+1)  = 0   =>   pn2(ik+1) = 0   =>   zdzr divides by 1
        # !          umask(ik+1) /= 0   =>   all pn2  /= 0   =>   zdzr divides by 2
        # ! NB: 1/(tmask+1) = (1-.5*tmask)  substitute a / by a *  ==> faster
        zdzr[kk] = zm1_g * (prd[kk] + 1) * (pn2[kk] + pn2[kk+1]) * (1 - 0.5 * tmask[kk+1])
    #

    #   ==   Slopes just below the mixed layer   ==!
    slpml   = ldf_slp_mxl( prd, pn2, zgru, zgrv, zdzr, nmln, mesh);        #! output: uslpml, vslpml, wslpiml, wslpjml
    uslpml  = slpml[0]
    vslpml  = slpml[1]
    wslpiml = slpml[2]
    wslpjml = slpml[3]


    #  ! I.  slopes at u and v point      | uslp = d/di( prd ) / d/dz( prd )
    #  ! ===========================      | vslp = d/dj( prd ) / d/dz( prd )
    #  !
    for kk in range(1, jpkm1) : # Slopes at u and v points
        for jj in range(1, jpjm1) :  
            for ii in range(1, jpim1) : # vector opt.
                #     horizontal and vertical density gradient at u- and v-points
                zau = zgru[kk, jj, ii] / e1u[jj, ii]
                zav = zgrv[kk, jj, ii] / e2v[jj, ii]
                zbu = 0.5 * ( zdzr[kk, jj, ii] + zdzr[kk, jj, ii+1] )
                zbv = 0.5 * ( zdzr[kk, jj, ii] + zdzr[kk, jj+1, ii] )
                #    bound the slopes: abs(zw.)<= 1/100 and zb..<0
                #     + kxz max= ah slope max =< e1 e3 /(pi**2 2 dt)
                zbu = np.min([zbu, -100 * np.abs(zau), -7e+3 / fse3u[kk, jj, ii] * np.abs(zau)])         
                zbv = np.min([zbv, -100 * np.abs(zav), -7e+3 / fse3v[kk, jj, ii] * np.abs(zav)])        
                    #    uslp and vslp output in zwz and zww, resp.
                zfi = np.max([omlmask[kk, jj, ii], omlmask[kk, jj, ii+1]])
                zfj = np.max([omlmask[kk, jj, ii], omlmask[kk, jj+1, ii]])
                zwz[kk, jj, ii] = ( (1 - zfi) * zau / (zbu - zeps) + zfi * uslpml[jj, ii] * 0.5 
                                    * ( fsdept[kk] + fsdept[kk] - fse3u[1, jj, ii])
                                    / np.max([hmlpt[jj, ii], hmlpt[jj, ii+1], 5]) ) * umask[kk, jj, ii]
                zww[kk, jj, ii] = ( (1 - zfj) * zav / (zbv - zeps) + zfj * vslpml[jj, ii] * 0.5 
                                    * ( fsdept[kk] + fsdept[kk] - fse3v[1, jj, ii] )
                                    / np.max([hmlpt[jj, ii], hmlpt[jj+1, ii], 5]) ) * vmask[kk, jj, ii]
            #
        #
    #
    #  !* horizontal Shapiro filter
    uslp = np.zeros((jpk, jpj, jpi))
    vslp = np.zeros((jpk, jpj, jpi))
    for kk in range(1, jpkm1) : 
        for ii in range(1, jpim1) :
            #   for  jj = 2:max(1,jpj-3):jpjm1 %                     ! rows jj=2 and =jpjm1 only
            jj = 2-1
            uslp[kk, jj, ii] = z1_16 * ( zwz[kk, jj-1, ii-1] + zwz[kk, jj-1, ii+1] 
                                         + zwz[kk, jj+1, ii-1] + zwz[kk, jj+1, ii+1] 
                                         + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1] 
                                                 + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] ) 
                                         + 4 * zwz[kk, jj, ii])
            vslp[kk, jj, ii] = z1_16 * ( zww[kk, jj-1, ii-1] + zww[kk, jj-1, ii+1]
                                         + zww[kk, jj+1, ii-1] + zww[kk, jj+1, ii+1]
                                         + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                 + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] )
                                         + 4 * zww[kk, jj, ii])
            jj = jpjm1-1
            uslp[kk, jj, ii] = z1_16 * ( zwz[kk, jj-1, ii-1] + zwz[kk, jj-1, ii+1] 
                                         + zwz[kk, jj+1, ii-1] + zwz[kk, jj+1, ii+1] 
                                         + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1] 
                                                 + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] ) 
                                         + 4 * zwz[kk, jj, ii])
            vslp[kk, jj, ii] = z1_16 * ( zww[kk, jj-1, ii-1] + zww[kk, jj-1, ii+1]
                                         + zww[kk, jj+1, ii-1] + zww[kk, jj+1, ii+1]
                                         + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                 + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] )
                                         + 4 * zww[kk, jj, ii])
        #
        for jj in range(2, jpj-2) : #           ! other rows
            for ii in range(2, jpim1) : #           ! vector opt.
                uslp[kk, jj, ii] = z1_16 * ( zwz[kk, jj-1, ii] + zwz[kk, jj-1, ii+1]
                                             + zwz[kk, jj+1, ii] + zwz[kk, jj+1, ii+1]
                                             + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1]
                                                     + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] )
                                             + 4 * zwz[kk, jj, ii] ) 
                vslp[kk, jj, ii] = z1_16 * ( zww[kk, jj-1, ii] + zww[kk, jj-1, ii+1]
                                             + zww[kk, jj+1, ii] + zww[kk, jj+1, ii+1]
                                             + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                     + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] )
                                             + 4 * zww[kk, jj, ii] ) 
            #
        #
        #                                  !* decrease along coastal boundaries
        for jj in range(1, jpjm1) :
            for ii in range(1, jpim1) :  # vector opt.
                uslp[kk, jj, ii] = ( uslp[kk, jj, ii] * ( umask[kk, jj+1, ii] + umask[kk, jj-1, ii] )
                                     * 0.5 * ( umask[kk, jj, ii] + umask[kk+1, jj, ii] ) * 0.5  )
                vslp[kk, jj, ii] = ( vslp[kk, jj, ii] * ( vmask[kk, jj, ii+1] + vmask[kk, jj, ii-1] )
                                     * 0.5 * ( vmask[kk, jj, ii] + vmask[kk+1, jj, ii] ) * 0.5  )
            #
        #
    # 
    #  ! II.  slopes at w point           | wslpi = mij( d/di( prd ) / d/dz( prd )
    #  ! ===========================      | wslpj = mij( d/dj( prd ) / d/dz( prd )
    #  !
    for kk in range(1, jpkm1) :
        for jj in range(1, jpjm1) : 
            for ii in range(1, jpim1) :#! vector opt.
                #     !* Local vertical density gradient evaluated from N2
                zbw = zm1_2g * pn2[kk, jj, ii] * ( prd[kk, jj, ii] + prd[kk-1, jj, ii] + 2 )
                #     !* Slopes at w point
                #     ! i- & j-gradient of density at w-points
                zci = np.max( [ umask[kk, jj, ii-1] + umask[kk, jj, ii] + umask[kk-1, jj, ii-1]
                                + umask[kk-1, jj, ii], zeps ] ) * e1t[jj, ii]
                zcj = np.max( [ vmask[kk, jj-1, ii] + vmask[kk-1, jj, ii] + vmask[kk-1, jj-1, ii]
                                + vmask[kk, jj, ii], zeps ] ) * e2t[jj, ii]
                zai = ( zgru[kk, jj, ii-1] + zgru[kk-1, jj, ii] + zgru[kk-1, jj, ii-1]
                        + zgru[kk, jj, ii] ) / zci * tmask[kk, jj, ii]
                zaj = ( zgrv[kk, jj-1, ii] + zgrv[kk-1, jj, ii] + zgrv[kk-1, jj-1, ii] + 
                        zgrv[kk, jj, ii] ) / zcj * tmask[kk, jj, ii]
                #     ! bound the slopes: abs(zw.)<= 1/100 and zb..<0.
                #     ! + kxz max= ah slope max =< e1 e3 /(pi**2 2 dt)
                zbi = np.min( [ zbw, -100 * np.abs(zai), -7e+3 / fse3w[kk, jj, ii] * np.abs(zai) ] )
                zbj = np.min( [ zbw, -100 * np.abs(zaj), -7e+3 / fse3w[kk, jj, ii] * np.abs(zaj) ] )
                #     ! wslpi and wslpj with ML flattening (output in zwz and zww, resp.)
                zfk = np.max( [ omlmask[kk, jj, ii], omlmask[kk-1, jj, ii] ] ) #   ! zfk=1 in the ML otherwise zfk=0
                zck = fsdepw[kk] / np.max( [hmlp[jj, ii], 10])
                zwz[kk, jj, ii] = ( zai / (zbi - zeps) * (1 - zfk) + zck * wslpiml[jj, ii] * zfk ) * tmask[kk, jj, ii]
                zww[kk, jj, ii] = ( zaj / (zbj - zeps) * (1 - zfk) + zck * wslpjml[jj, ii] * zfk ) * tmask[kk, jj, ii]
            #
        #
    #
    for kk in range(1, jpkm1) : 
        #         FOR  jj=2:max([1 jpj-3]):jpjm1 %                       ! rows jj=2 and =jpjm1 only
        for ii in range(1, jpim1) : 
            jj = 2- 1
            wslpi[kk, jj, ii] = ( zwz[kk, jj-1, ii-1] + zwz[kk, jj-1, ii+1] + zwz[kk, jj+1, ii-1]
                                  + zwz[kk, jj+1, ii+1] + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1]
                                                                + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] )
                                  + 4 * zwz[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
            wslpj[kk, jj, ii] = ( zww[kk, jj-1, ii-1] + zww[kk, jj-1, ii+1] + zww[kk, jj+1, ii-1]
                                  + zww[kk, jj+1, ii+1] + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                                + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] ) 
                                  + 4 * zww[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
            jj = jpjm1- 1
            wslpi[kk, jj, ii] = ( zwz[kk, jj-1, ii-1] + zwz[kk, jj-1, ii+1] + zwz[kk, jj+1, ii-1]
                                  + zwz[kk, jj+1, ii+1] + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1]
                                                                + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] )
                                  + 4 * zwz[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
            wslpj[kk, jj, ii] = ( zww[kk, jj-1, ii-1] + zww[kk, jj-1, ii+1] + zww[kk, jj+1, ii-1]
                                  + zww[kk, jj+1, ii+1] + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                                + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] ) 
                                  + 4 * zww[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
        #
        for jj in range(2, jpj-2) :       # other rows
            for ii in range(1, jpim1) :     # vector opt.
                wslpi[kk, jj, ii] = ( zwz[kk, jj-1, ii-1] + zwz[kk, jj-1, ii+1] + zwz[kk, jj+1, ii-1]
                                      + zwz[kk, jj+1, ii+1] + 2 * ( zwz[kk, jj-1, ii] + zwz[kk, jj, ii-1]
                                                                    + zwz[kk, jj, ii+1] + zwz[kk, jj+1, ii] )
                                      + 4 * zwz[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
                wslpj[kk, jj, ii] = ( zww[kk, jj-1, ii-1] + zww[kk, jj-1, ii+1] + zww[kk, jj+1, ii-1]
                                      + zww[kk, jj+1, ii+1] + 2 * ( zww[kk, jj-1, ii] + zww[kk, jj, ii-1]
                                                                    + zww[kk, jj, ii+1] + zww[kk, jj+1, ii] ) 
                                      + 4 * zww[kk, jj, ii] ) * z1_16 * tmask[kk, jj, ii]
            #
        #
        # !* decrease along coastal boundaries
        for jj in range(1, jpjm1) : 
            for ii in range(1, jpim1) :   #   vector opt.
                zck = ( umask[kk, jj, ii] + umask[kk, jj, ii-1] ) * ( vmask[kk, jj, ii] + vmask[kk, jj-1, ii] ) * 0.25 
                wslpi[kk, jj, ii] = wslpi[kk, jj, ii] * zck                                                  
                wslpj[kk, jj, ii] = wslpj[kk, jj, ii] * zck                                                  
            #
        #
    #
    slp = np.zeros((4, jpk, jpj, jpi))

    slp[0] = wslpi
    slp[1] = wslpj
    slp[2] = uslp
    slp[3] = vslp

    slp = np.where( slp == slp, slp, np.zeros_like(slp) )

    return slp
####################################################################################################
