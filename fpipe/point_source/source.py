from astroquery.ned import Ned

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from scipy.ndimage import median_filter

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

import numpy as np
import h5py as h5
import healpy as hp
from fpipe.map import algebra as al
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import NearestNDInterpolator

import os
_dir_ = os.path.dirname(__file__)

def get_map_spec(imap_info, ra, dec, beam_size=3./60., mJy=False):

    imap, pixs, nside, nmap = imap_info
    freq = imap.get_axis('freq')

    data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
    f = data[4:, 0] #* 1.e-3
    d = data[4:, 1:]
    fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")(freq)
    fwhm /= 60.

    #beam_size = np.max(fwhm)
    #beam_sig = (fwhm / (2. * np.sqrt(2.*np.log(2.))))[None, :]
    pix_size = hp.nside2resol(nside, arcmin=True) / 60.

    data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
    f = data[4:, 0] #* 1.e-3
    d = data[4:, 1:]
    fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")(freq)
    fwhm /= 60.
    #fwhm = 3./60. * 1420 / freq

    beam_size = np.min(fwhm)
    beam_sig = (fwhm / (2. * np.sqrt(2.*np.log(2.))))[None, :]

    # spec on
    _v = hp.ang2vec(ra, dec, lonlat=True)
    p_ra, p_dec = hp.pix2ang(nside, 
            hp.ang2pix(nside, ra, dec, lonlat=True), lonlat=True)
    #_p = hp.query_disc(nside, _v, np.radians(0.5 * beam_size), inclusive=True)
    _p  = hp.get_all_neighbours(nside, ra, dec, lonlat=True)
    _p = np.append(_p, hp.ang2pix(nside, ra, dec, lonlat=True))
    _w = hp.pix2ang(nside, _p, lonlat=True)
    _w = np.exp( - 0.5*((_w[0] - ra)**2 + (_w[1] - dec)**2 )[:, None] / beam_sig **2)
    _w /= np.max(_w, axis=0)[None, :]

    spec = []
    noise = [] # it is noise inv
    for i in range(_p.shape[0]):
        #spec.append( imap[:, (pixs - _p[i]) == 0][None, :] / _w[i] )
        if np.any((pixs - _p[i]) == 0) :
            spec.append(  imap[:, (pixs - _p[i]) == 0][None, :])
            noise.append( nmap[:, (pixs - _p[i]) == 0][None, :])
        else:
            _w[i] = 0.
    if len(spec) == 0: return None
    spec = np.concatenate(spec, axis=0)
    spec = np.ma.masked_equal(spec, 0)
    #_w[spec.mask[:, :, 0]] = 0.
    spec = np.ma.sum(spec, axis=0) / np.sum(_w, axis=0)[:, None]
    #spec = np.ma.sum(spec, axis=0) / np.sum(_w)

    noise = np.concatenate(noise, axis=0)
    noise = np.ma.masked_equal(noise, 0)
    noise = np.ma.sum(noise, axis=0) / np.sum(_w, axis=0)[:, None]
    # convert to noise error
    noise = np.sqrt(noise)
    noise[noise==0] = np.inf
    noise = 1./noise

    if mJy:
        #spec = spec / mJy2K(freq*1.e-3, eta=0.6,)[:, None] #* 2.
        #print np.mean(mJy2K(freq*1.e-3) )  * 1.e3
        eta = 1.0
        #print mJy2K(freq*1.e-3, eta=eta).min()  * 1.e3,
        #print mJy2K(freq*1.e-3, eta=eta).max()  * 1.e3
        #spec = spec / mJy2K(freq*1.e-3, eta=eta)[:, None] 
        #noise= noise / mJy2K(freq*1.e-3, eta=eta)[:, None] 
        spec  = spec / (25. * 1.e-3)
        noise = noise / (25. * 1.e-3) 


    return freq, spec, p_ra, p_dec, noise

#return None

def get_nvss_flux(nvss_path, nvss_range, threshold=200, eta_b=None, mJy=False):

    nvss_cat = get_nvss_radec(nvss_path, nvss_range)
    freq = 1400.


    for ii in range(nvss_cat.shape[0]):
        _ra = nvss_cat['RA'][ii]
        _dec= nvss_cat['DEC'][ii]
        _flx= nvss_cat['FLUX_20_CM'][ii]
        s_name = nvss_cat['NAME'][ii]

        #_r = ((_ra - ra)**2 + (_dec - dec)**2)**0.5
        #beam_off = _r.min() * 60
        beam_off = 0.
        if nvss_cat['FLUX_20_CM'][ii] > threshold:
            yield freq, _flx, s_name, _ra, _dec, _flx

def project_nvss_to_map(nside, nss_path, nvss_range, threshold=10, fwhm=3.):

    nvss_cat = get_nvss_radec(nvss_path, nvss_range)

    pixs = np.arange(hp.nside2npix(nside) + 1) - 0.5
    nvss_pix = hp.ang2pix(nside, nvss_cat['RA'], nvss_cat['DEC'], lonlat=True)
    nvss_cnt = np.histogram(nvss_pix, pixs)[0] * 1.
    nvss_map = np.histogram(nvss_pix, pixs, weights=nvss_cat['FLUX_20_CM'])[0]

    nvss_cnt[nvss_cnt==0] = np.inf
    nvss_map /= nvss_cnt

    #window = np.zeros(hp.nside2npix(nside), dtype='float32')
    #window[0] = 1.
    #window = hp.smoothing(window, fwhm=np.deg2rad(fwhm/60.))
    #window /= window.max()

    nvss_map[nvss_map<0] = 0
    nvss_map = hp.ma(nvss_map, badval=0.)
    print np.any(nvss_map.mask)
    if fwhm != 0:
        nvss_map = hp.smoothing(nvss_map, fwhm=np.deg2rad(fwhm/60.))
        #sig = np.deg2rad(fwhm/60.) / 2. / np.sqrt(2. * np.log(2.))
        #nvss_map *= 2. * np.pi * sig**2.

    #nvss_map = nvss_map[None, ...]

    return nvss_map, pixs, nside


def project_nvss_to_map_partial(nside, pixs, nvss_path, nvss_range, threshold=500, fwhm=3.):

    sig = fwhm / 60.

    nvss_cat = get_nvss_radec(nvss_path, nvss_range)

    pixs_ra, pixs_dec = hp.pix2ang(nside, pixs, lonlat=True)

    pixs_ra = np.deg2rad(pixs_ra[:, None])
    pixs_dec = np.deg2rad(pixs_dec[:, None])


    nvss_flux = nvss_cat['FLUX_20_CM']
    sel = nvss_flux > threshold
    nvss_ra = np.deg2rad(nvss_cat['RA'][sel][None, :])
    nvss_dec = np.deg2rad(nvss_cat['DEC'][sel][None, :])
    nvss_flux = nvss_flux[sel][None, :]

    nvss_map = np.zeros(pixs.shape, dtype='float64')

    step = 100
    nvss_N = nvss_flux.shape[1]
    for i in range(0, nvss_N, step):

        P = ( np.sin(pixs_dec) * np.sin(nvss_dec[:, i:i+step]) )\
          + ( np.cos(pixs_dec) * np.cos(nvss_dec[:, i:i+step]) )\
          * ( np.cos(pixs_ra - nvss_ra[:, i:i+step]) )

        P = np.rad2deg(np.arccos(P))
        P = np.exp( - 0.5 * ( P / sig ) ** 2. ) * nvss_flux[:, i:i+step]

        P = np.sum(P, axis=1)

        nvss_map = nvss_map + P

        del P


    return nvss_map, pixs, nside

def get_pointsource_spec(nvss_path, nvss_range, threshold=200, eta_b=None, mJy=False):

    freq = np.linspace(1050, 1450, 100)

    nvss_cat = get_nvss_radec(nvss_path, nvss_range)


    for ii in range(nvss_cat.shape[0]):
        _ra = nvss_cat['RA'][ii]
        _dec= nvss_cat['DEC'][ii]

        #_r = ((_ra - ra)**2 + (_dec - dec)**2)**0.5
        #beam_off = _r.min() * 60
        beam_off = 0.
        if nvss_cat['FLUX_20_CM'][ii] > threshold:
            _s = SkyCoord(_ra * u.deg, _dec * u.deg)
            cal_T, s_name, cal_data = get_NED_source(_s, freq * 1.e-3, beam_off, mJy)
            if not cal_T is None:
                #print nvss_cat[ii], "%02dH%02dM%6.4f"%_s.ra.hms, _s.dec.deg
                #_vis = np.ma.mean(vis[np.argmin(_r)-2:np.argmin(_r)+2,], axis=0)
                #_vis_t = vis[np.argmin(_r)-100:np.argmin(_r)+100, ...]
                yield freq, cal_T, s_name, _ra, _dec, cal_data

def get_NED_source(s, freq, beam_off, mJy=False):

    _tb = Ned.query_region(s, radius=0.5 * u.arcmin, equinox='J2000')
    for _name in _tb['Object Name']:
        if _name[:2] == 'B2':
            #_vis = np.ma.mean(vis[np.argmin(_r)-2:np.argmin(_r)+2,], axis=0)
            #_vis = np.ma.masked_equal(_vis, 0)
            #
            #_vis_t = vis[np.argmin(_r)-100:np.argmin(_r)+100, ...]

            _sp = Ned.get_table(_name)
            #print _sp.keys()
            _nu   = _sp['Frequency'].data #* 1.e-9
            _flux = _sp['Flux Density'].data #* 1.e3
            _flux_u = _sp['Upper limit of uncertainty'].data #* 1.e3
            _flux_l = _sp['Lower limit of uncertainty'].data #* 1.e3
            _flux_u[~np.isfinite(_flux_u)] = 0.
            _flux_l[~np.isfinite(_flux_l)] = 0.
            _flux_d = 0.5 * (_flux_u + _flux_l)
            #_flux_d[~np.isfinite(_flux_d)] = 0.

            _sel = _nu < 5.e9
            data = np.concatenate([_nu[_sel, None], _flux[_sel, None],
                                   _flux_d[_sel, None], ], axis=1)

            cal_T, data = get_calibrator_spec(freq, cal_data=data, 
                    beam_off = beam_off, mJy=mJy)
            #if eta_b is not None:
            #    cal_T /= 0.6
            #    cal_T *= eta_b

            return cal_T, _name, data
    return None, None, None

def mJy2K(freq, eta=1., beam_off=0, fwhm=None):

    if fwhm is None:
        data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
        f = data[4:, 0] * 1.e-3
        d = data[4:, 1:]
        fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")

    _lambda = 2.99e8 / (freq * 1.e9)
    #_sigma = 1.02 *  _lambda / 300. / np.pi * 180. * 3600.
    _sigma = fwhm(freq) * 60.

    if beam_off != 0:
        fact = np.exp(-0.5 * beam_off**2 / (_sigma/2./(2.*np.log(2.))**0.5 )**2 )
    else:
        fact = 1.

    return eta * 1.36 * (_lambda*1.e2)**2. / _sigma**2. * fact


def get_calibrator_spec(freq, cal_data=None, ra=None, dec=None, beam_off=0, mJy=False):

    cal_spec_func = np.poly1d(np.polyfit(np.log10(cal_data[:,0]*1.e-9),
        np.log10(cal_data[:,1]*1.e3), deg=2,))
                                             #w = 1./ cal_data[:,2]))
                                             #w = 1./ np.log10(cal_data[:,2])))

    cal_flux = 10. ** cal_spec_func(np.log10(freq)) # in mJy
    if mJy:
        cal_data[:, 1:] *= 1.e3 
        cal_data[:, 0] *= 1.e-6
        return cal_flux, cal_data

    if ra is not None:
        with h5.File('/scratch/users/ycli/fanalysis/etaA_map.h5', 'r') as f:
            etaA = f['eta_A'][:]
            _ra   = f['ra'][:]
            _dec  = f['dec'][:]
        _dec, _ra = np.meshgrid(_dec, _ra)
        _dec = _dec.flatten()[:, None]
        _ra  = _ra.flatten()[:, None]
        etaA = etaA.flatten()
        good = etaA != 0
        etaA = etaA[good]
        coord = np.concatenate([_ra[good, :], _dec[good, :]], axis=1)
        
        etaAf = NearestNDInterpolator(coord, etaA)

        eta = etaAf(ra, dec)
        print '[%f, %f] eta = %f'%(ra, dec, eta)
    else:
        eta   = 1. #0.6 #0.9

    #print 'Jy2K : %f'%(mJy2K[-1] * 1.e3)
    cal_T = cal_flux * mJy2K(freq, 1., beam_off) # in K
    cal_data[:, 1:] *= 1.e3 * mJy2K(cal_data[:, 0].data*1.e-9, 1., beam_off)[:, None]
    cal_data[:, 0] *= 1.e-6
    #cal_T = cal_flux * 16.0e-3

    return cal_T, cal_data #, cal_T * factor

def get_nvss_radec(nvss_path, nvss_range):

    import astropy.io.fits as pyfits

    hdulist = pyfits.open(nvss_path)
    data = hdulist[1].data

    _sel = np.zeros(data['RA'].shape[0], dtype ='bool')
    for _nvss_range in nvss_range:
        #print _nvss_range

        ra_min, ra_max, dec_min, dec_max = _nvss_range

        sel  = data['RA'] > ra_min
        sel *= data['RA'] < ra_max
        sel *= data['DEC'] > dec_min
        sel *= data['DEC'] < dec_max

        #print np.sum(sel)

        _sel = _sel + sel

    hdulist.close()

    return data[_sel]
