from os.path import join, dirname
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants as const

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, quadrature


import matplotlib.pyplot as plt

#__pk_file__ = '/users/ycli/code/camb/output/HI_matterpower.dat'
#__pk_file__ = join(dirname(__file__), 'data/input_matterpower.dat')
#__pk_redshift__ = 0.2
__pk_file__ = join(dirname(__file__), 'data/ps_z1.5.dat')
__pk_redshift__ = 1.5

def grouth_rate():

    x = lambda z: ((1.0 / cosmo.Om(0)) - 1.0) / (1.0 + z)**3
    
    dnum = lambda z: 3.0*x(z)*(1.175 + 0.6127*x(z) + 0.01607*x(z)**2)
    dden = lambda z: 3.0*x(z)*(1.857 + 2.042 *x(z) + 0.4590 *x(z)**2)
    
    num = lambda z: 1.0 + 1.175*x(z) + 0.3064*x(z)**2 + 0.005355*x(z)**3
    den = lambda z: 1.0 + 1.857*x(z) + 1.021 *x(z)**2 + 0.1530  *x(z)**3
    
    return lambda z: 1.0 + 1.5 * x(z) / (1.0 + x(z)) + dnum(z) / num(z) - dden(z) / den(z)


def _grouth_factor(z):

    x = ((1.0 / cosmo.Om(0)) - 1.0) / (1.0 + z)**3
    
    num = 1.0 + 1.175*x + 0.3064*x**2 + 0.005355*x**3
    den = 1.0 + 1.857*x + 1.021 *x**2 + 0.1530  *x**3
    
    d = (1.0 + x)**0.5 / (1.0 + z) * num / den
    
    return d

def grouth_factor(z):

    return _grouth_factor(z)/_grouth_factor(__pk_redshift__)

#def grouth_rate():
#    return lambda z: cosmo.Om(z) ** 0.55

#def grouth_factor(z):
#    z = np.array([z,]).flatten()
#    f = grouth_rate()
#    lnD = lambda z: map(lambda z: quadrature(lambda z: - f(z) / (1 + z), 0, z)[0], z)
#    lnD0 = lnD(np.array([__pk_redshift__]))[0]
#    return np.exp(lnD(z) - lnD0)

def matterpowerspectrum(pk_file=__pk_file__):
    pk_data = np.loadtxt(pk_file)
    kh = pk_data[:, 0]
    pk = pk_data[:, 1]
    return interp1d(kh, pk, kind='linear', bounds_error=False, fill_value="extrapolate")

def Pk1D(k, z):

    pk_f = matterpowerspectrum()
    pk = pk_f(k)
    pk = grouth_factor(z)[:, None] ** 2. * pk[None, ...]
    return pk

def Pk2D(k_para, k_perp, z):

    k1 = k_para[:, None]
    k2 = k_perp[None, :]
    pk_f = matterpowerspectrum()

    k = ( k1**2 + k2**2 ) ** 0.5
    k = k.flatten()

    pk = pk_f(k)
    pk.shape = k_para.shape + k_perp.shape
    pk = grouth_factor(z)[:, None, None] ** 2. * pk[None, ...]
    return pk

def Tb_HI(z, OmHI=1.e-3):

    return 0.39 * ( ( cosmo.Om0 + cosmo.Ode0 / ((1 + z) ** 3 ) ) / 0.29) ** (-0.5)\
                    * ((1 + z)/2.5) ** 0.5 * ( OmHI / 1.e-3 )

def Pk1D_HI(k, z, b_HI = 0.67):
    z = np.array([z,]).flatten()
    Tb = Tb_HI(z)[:, None]
    return Pk1D(k, z) * Tb**2 * b_HI ** 2.

def Pk2D_HI(k_para, k_perp, z):
    z = np.array([z,]).flatten()
    b_HI = 0.67
    Tb = Tb_HI(z)[:, None, None]
    k = (k_para[:, None]**2 + k_perp[None, :]**2)**0.5
    mu = k_para[:, None] / k
    sig = 7. * cosmo.h

    mu = mu[None, :, :]
    k  = k[None, :, :]
    f = grouth_rate()(z)[:, None, None]
    rsd = (b_HI + f * mu ** 2) * np.exp(- 0.5 * (k * mu * sig)**2)

    return Pk2D(k_para, k_perp, z)  * (Tb**2) * (rsd ** 2)


def CHI(q, y, z):

    z = np.array([z,]).flatten()
    zi = np.mean(z)

    r   = cosmo.comoving_distance(zi) * cosmo.h
    rnu = const.c.to('km/s') * (1 + zi) ** 2 / cosmo.H(zi) * cosmo.h

    k_para = y / rnu.value
    k_perp = q / r.value

    return Pk2D_HI(k_para, k_perp, zi)[0]

def CN(q, y, freq0, params, bcast=True, return_beam=False, max_beam=False):

    if bcast:
        q = q[None, :]
        y = y[:, None]

    #freq0 = params['freq0']
    dfreq = 2. * np.pi / y * 1420.
    freq = dfreq + freq0

    df = np.min(dfreq)
    Df = np.max(dfreq)

    #T_sys = 25.e3
    T_sys = params['T_sys']

    S_area = params['S_area']
    S_area = S_area * (np.pi / 180.) ** 2.

    pix_size = params['pixel_size']
    S_pixl = (pix_size * np.pi / 180.) ** 2.

    #t_tot = (7. * 1.5 * u.hour).to('s').value
    t_tot = params['t_tot']

    #T_rms = T_sys ** 2./(2. * t_tot * S_pixl / S_area ) / (df * 1.e6)
    T_rms = T_sys ** 2./(2. * t_tot * S_pixl / S_area) / (Df * 1.e6)
    #T_rms = T_sys ** 2./(2. * t_tot) / (1420 * 1.e6) * S_area

    #U_bin = S_area #* ( df / 1420.)

    N_b = params['N_beam']
    N_d = params['N_dish']
    I = 1./ N_b/ N_d

    B_para = np.exp(- ( y * df / 1420. ) ** 2. /  16. / np.log(2.) )

    D_dish = params['D_dish']
    #fwhm1400 = 0.9
    #fwhm = 1.2 * fwhm1400 * 1400./ freq * np.pi / 180.
    fwhm = 1.2 * const.c.to('m/s').value / (freq * 1.e6) / D_dish
    if max_beam:
        print('use max beam %f'%fwhm.max())
        fwhm = np.ones_like(fwhm) * fwhm.max()
    #fwhm = np.ones_like(freq) * np.pi / 180.
    B_perp = np.exp(- ( q * fwhm )**2. / 16. / np.log(2.) )

    #k_area = 2.* np.pi / (S_area)**0.5

    if return_beam:
        return T_rms * I * np.ones_like(B_perp * B_para), B_perp * B_perp * B_para
    else:
        return T_rms * I / B_perp / B_perp / B_para

def Pk2D_N(k_para, k_perp, z, params, bcast=True, return_beam=False, max_beam=False):

    z  = np.array(z).flatten()
    z0 = np.mean(z)
    rr = (cosmo.comoving_distance(z0) * cosmo.h).value
    rv = (const.c.to('km/s') * (1 + z0) ** 2. / cosmo.H(z0) * cosmo.h).value

    q = k_perp * rr
    y = k_para * rv

    freq0 = 1420./(1. + z.max())

    S = params['S_area']
    S = S * (np.pi / 180.)**2.
    r0 = (cosmo.comoving_distance(z0) * cosmo.h).value
    rmin = (cosmo.comoving_distance(z.min()) * cosmo.h).value
    rmax = (cosmo.comoving_distance(z.max()) * cosmo.h).value
    dr = rmax - rmin
    Vbin = S * (r0 **2) * dr

    #k_area = 2.* np.pi / (r0**2 * S)**0.5 # min k_perp

    _r = CN(q, y, freq0, params, bcast=bcast, return_beam=return_beam, 
            max_beam=max_beam)
    if return_beam:
        pkn, B = _r
        #B = B * np.exp(0.5 * k_perp**2/k_area**2)
        pkn = pkn * Vbin
        return pkn, Vbin, B
    else:
        pkn = _r
        pkn = pkn * Vbin
        return pkn, Vbin

def Pk_HI(k, z, mu=None, b_HI = 0.67, RSD=True):
    
    z = np.array([z,]).flatten()
    Tb = Tb_HI(z)[:, None, None]
    k  = k[None, :, None]
    if mu is None:
        mu = np.linspace(0, 1, 101)
    mu = mu[None, None, :]
    sig= 7 * cosmo.h
    f  = grouth_rate()(z)[:, None, None]
    rsd = (b_HI + f * mu ** 2) * np.exp(- 0.5 * (k * mu * sig)**2)
    if not RSD:
        rsd = b_HI * np.ones_like(rsd)
    
    pk_f = matterpowerspectrum()
    pk = pk_f(k) * (grouth_factor(z)[:, None, None] ** 2.)
    return pk * Tb ** 2 * rsd ** 2
    
def Pk_G(k, z, mu=None, b_g=1.0, RSD=True):
    
    z = np.array([z,]).flatten()
    k  = k[None, :, None]
    if mu is None:
        mu = np.linspace(0, 1, 101)
    mu = mu[None, None, :]
    sig= 7 * cosmo.h
    f  = grouth_rate()(z)[:, None, None]
    rsd_g  = (b_g  + f * mu ** 2) * np.exp(- 0.5 * (k * mu * sig)**2)
    if not RSD:
        rsd_g = b_g * np.ones_like(rsd_g)
    
    pk_f = matterpowerspectrum()
    pk = pk_f(k) * (grouth_factor(z)[:, None, None] ** 2.)
    return pk * rsd_g ** 2
    

def Pk_N(k, z, params, mu=None, max_beam=False):
    
    k  = k[:, None]
    if mu is None:
        mu = np.linspace(0, 1, 101)
    mu = mu[None, :]
    
    k_para = k * mu
    k_perp = (k**2 - k_para**2)**0.5
    
    pk, Vbin, B = Pk2D_N(k_para.flatten(), k_perp.flatten(), z, params, 
                   bcast=False, return_beam=True, max_beam=max_beam)
    pk.shape = k.shape[:1] + mu.shape[1:]
    B.shape  = pk.shape
    
    return pk, B, Vbin
    

def est_dP_auto(params, conv_beam=False, RSD=True, max_beam=False):
    
    zz = params['zz']
    
    kh_edgs = params['kh_edgs']
    dkh = kh_edgs[1:] - kh_edgs[:-1]
    if params['logk']:
        kh  = kh_edgs[:-1] * ((kh_edgs[1:] / kh_edgs[:-1]) ** 0.5)
    else:
        kh  = kh_edgs[:-1] + ((kh_edgs[1:] - kh_edgs[:-1]) * 0.5)
    
    mu_edgs = params['mu_edgs']
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)
    
    
    b_HI= params['b_HI']
    pkhi = np.mean(Pk_HI(kh, zz, mu, b_HI=b_HI, RSD=RSD), axis=0)
    pkn, B, Vbin = Pk_N(kh, zz, params, mu, max_beam=max_beam)
    
    if conv_beam:
        pkhi *= B
    else:
        B[B==0] = np.inf
        pkn = pkn / B
    
    S = params['S_area']
    S = S * (np.pi / 180.)**2.
    r0 = (cosmo.comoving_distance(np.mean(zz)) * cosmo.h).value
    k_area = 2.* np.pi / (r0**2 * S)**0.5 # min k_perp
    k_para =  kh[:, None] * mu[None, :]
    k_perp = (kh[:, None]**2 - k_para**2)**0.5
    pkhi[k_perp < k_area] = 0.
    
    rmin = (cosmo.comoving_distance(zz.min()) * cosmo.h).value
    rmax = (cosmo.comoving_distance(zz.max()) * cosmo.h).value
    dr = rmax - rmin
    k_fband = 2.* np.pi / dr #0.03
    #pkhi[k_para < k_fband] = 0.

    if params['k_fgcut'] is not None:
        pkhi[k_para < params['k_fgcut']] = 0.
    
    pkt = (pkn + pkhi)
    pkt[pkt==0] = np.inf
    snr = pkhi / pkt
    
    k2dkdmu = kh[:, None]**2 * dkh[:, None] * dmu[None, :]
    dpk2pk = (0.5 * np.sum( (snr)**2. * k2dkdmu  / (2. * np.pi)**2., axis=1) * Vbin) ** (0.5)
    dpk2pk[dpk2pk==0] = np.inf
    dpk2pk = 1./dpk2pk
    
    pkhi1d = np.sum(pkhi * dmu, axis=1)
    pkn1d = np.sum(pkn * dmu, axis=1)
    
    #pk0 = np.mean(Pk1D_HI(kh, zz), axis=0)
    pk0 = np.sum(np.mean(Pk_HI(kh, zz, mu, b_HI=b_HI, RSD=RSD), axis=0) * dmu, axis=1)
    
    return kh, pkhi1d, pkn1d, dpk2pk, pk0

def est_dP_cross(params, conv_beam=False, RSD=True, max_beam=False):
    
    zz = params['zz']

    kh_edgs = params['kh_edgs']
    dkh = kh_edgs[1:] - kh_edgs[:-1]
    if params['logk']:
        kh  = kh_edgs[:-1] * ((kh_edgs[1:] / kh_edgs[:-1]) ** 0.5)
    else:
        kh  = kh_edgs[:-1] + ((kh_edgs[1:] - kh_edgs[:-1]) * 0.5)
    
    mu_edgs = params['mu_edgs']
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)
    
    b_g = params['b_g']
    b_HI= params['b_HI']
    pkhi = np.mean(Pk_HI(kh, zz, mu, b_HI=b_HI, RSD=RSD), axis=0)
    pkg  = np.mean(Pk_G(kh, zz, mu, b_g=b_g, RSD=RSD),  axis=0)
    pkn, B, Vbin = Pk_N(kh, zz, params, mu, max_beam=max_beam)
    shot = 1. / params['nbar'] * np.ones_like(pkn)
    
    if conv_beam:
        pkhi *= B
    else:
        B[B==0] = np.inf
        pkn = pkn / B
    
    S = params['S_area']
    S = S * (np.pi / 180.)**2.
    r0 = (cosmo.comoving_distance(np.mean(zz)) * cosmo.h).value
    k_area = 2.* np.pi / (r0**2 * S)**0.5 # min k_perp
    k_para =  kh[:, None] * mu[None, :]
    k_perp = (kh[:, None]**2 - k_para**2)**0.5
    pkhi[k_perp < k_area] = 0.
    pkg[k_perp < k_area] = 0.
    
    rmin = (cosmo.comoving_distance(zz.min()) * cosmo.h).value
    rmax = (cosmo.comoving_distance(zz.max()) * cosmo.h).value
    dr = rmax - rmin
    k_fband = 2.* np.pi / dr #0.03
    #pkhi[k_para < k_fband] = 0.
    #pkg[k_para < k_fband] = 0.

    if params['k_fgcut'] is not None:
        pkhi[k_para < params['k_fgcut']] = 0.
        pkg[k_para < params['k_fgcut']] = 0.
    
    
    pkx  = (pkhi * pkg) ** 0.5
    pkt = (pkn + pkhi) * (pkg + shot) + pkx ** 2.
    pkt[pkt==0] = np.inf
    snr = pkhi * pkg / pkt

    k2dkdmu = kh[:, None]**2 * dkh[:, None] * dmu[None, :]
    dpk2pk = (0.5 * np.sum( snr * k2dkdmu  / (2. * np.pi)**2., axis=1) * Vbin) ** (0.5)
    dpk2pk[dpk2pk==0] = np.inf
    dpk2pk = 1./dpk2pk
    
    pkhi1d = np.sum(pkx * dmu,  axis=1)
    pkn1d  = np.sum(pkn * dmu,  axis=1)
    shot1d = np.sum(shot * dmu, axis=1)
    
    #pk0 = np.mean(np.sqrt(Pk1D_HI(kh, zz) * Pk1D(kh, zz)), axis=0)
    pk0 = np.sum(
            ((np.mean(Pk_HI(kh, zz, mu, b_HI=b_HI, RSD=RSD), axis=0) *
              np.mean(Pk_G( kh, zz, mu, b_g=b_g, RSD=RSD), axis=0) ) ** 0.5)\
                      * dmu, axis=1)
    
    return kh, pkhi1d, pkn1d, dpk2pk, pk0, shot1d

def fisher_auto(params, figax=None, label='', dk=1.0, title=''):

    if figax is None:
        fig = plt.figure(figsize=(8, 5))
        ax  = fig.add_subplot(111)
    else:
        fig, ax = figax

    kh, pkhi, pkn, dpk2pk, pk0 = est_dP_auto(params)
    good = pkhi > 0
    #ax.plot(kh, pk0, 'k-', label = 'P(k) HI')
    #ax.plot(kh[good], pkhi[good], 'k--', label = 'P(k) HI with RSD')
    c = ax.errorbar(kh * dk, pkhi, pkhi * dpk2pk, fmt='o--', mfc='w', 
            label='P(k) HI ' + label)[0].get_color()
    ax.plot(kh * dk, pkn, '-', color=c, label='N(k)/B(k) ' + label)
    ax.plot(kh[good] * dk, (pkhi * dpk2pk)[good], '--', color=c,
            label='dP(k) HI ' + label)


    ax.set_ylim(ymax=3.e4, ymin=1.e0)
    ax.set_xlim(xmin=0.03, xmax=1.1)
    ax.loglog()
    ax.set_ylabel(r'$P(k)\,[({\rm Mpc}\, h^{-1})^3 {\rm mK}^2]$')
    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1}\, h]$')
    ax.legend(ncol=2, title=title, loc=2)

    return (fig, ax)

def fisher_cross(params, figax=None, label='', dk=1.0, title=''):

    if figax is None:
        fig = plt.figure(figsize=(8, 5))
        ax  = fig.add_subplot(111)
    else:
        fig, ax = figax

    kh, pkx, pkn, dpk2pk, pk0, shot = est_dP_cross(params)
    good = pkx > 0
    ax.errorbar(kh * dk, pkx, pkx * dpk2pk, fmt='o--', mfc='w', 
            label='P(k) cross ' + label)

    ax.set_ylim(ymax=8.e4, ymin=1.e1)
    ax.set_xlim(xmin=0.03, xmax=1.1)
    ax.loglog()
    ax.set_ylabel(r'$P(k)\,[({\rm Mpc}\, h^{-1})^3 {\rm mK}]$')
    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1}\, h]$')
    ax.legend(ncol=2, title=title, loc=2)

    return (fig, ax)




