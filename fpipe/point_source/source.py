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

def get_map_spec(map_name, map_key, ra, dec, beam_size=3./60.):

    with h5.File(map_name, 'r') as f:
        #print 'Load maps from %s'%map_name
        #print f.keys()
        #imap = f[map_key][indx]
        imap = al.load_h5(f, map_key)
        imap = al.make_vect(imap, axis_names = imap.info['axes'])
        #imap = f['dirty_map'][:]
        pixs = f['map_pix'][:]
        nside = f['nside'][()]

    freq = imap.get_axis('freq')

    if beam_size is None:
        _p = hp.ang2pix(nside, ra, dec, lonlat=True)
        p_ra, p_dec = hp.pix2ang(nside, _p, lonlat=True)
        where = (pixs - _p) == 0
        if np.any(where):
            spec_on = imap[:, where]
            _p = hp.ang2pix(nside, ra+8./60., dec, lonlat=True)
            where = (pixs - _p) == 0
            spec_off = imap[:, where]
            return freq, spec_on - spec_off, p_ra, p_dec

    else:
        beam_sig = beam_size * (2. * np.sqrt(2.*np.log(2.)))
        _v = hp.ang2vec(ra, dec, lonlat=True)
        _p = hp.query_disc(nside, _v, np.radians(0.5 * beam_size))
        _w = hp.pix2ang(nside, _p, lonlat=True)
        _w = np.exp( - 0.5 * ( (_w[0] - ra)**2 + (_w[1] - dec)**2 ) / beam_sig **2 )
        spec = []
        for i in range(_p.shape[0]):
            print _w[i],
            spec.append( imap[:, (pixs - _p[i]) == 0][None, :] / _w[i])
            #spec.append( imap[:, (pixs - _p[i]) == 0][None, :] )
        print 

        spec = np.concatenate(spec, axis=0)
        spec = np.ma.masked_equal(spec, 0)
        #spec = np.ma.sum(spec, axis=0) #/ np.sum(_w)
        spec = np.ma.mean(spec, axis=0)
        #spec = np.ma.sum(spec, axis=0)

        return freq, spec, ra, dec


    return None

def get_pointsource_spec(nvss_path, nvss_range, threshold=200, eta_b=None):

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
            cal_T, s_name, cal_data = get_NED_source(_s, freq * 1.e-3, beam_off)
            if not cal_T is None:
                #print nvss_cat[ii], "%02dH%02dM%6.4f"%_s.ra.hms, _s.dec.deg
                #_vis = np.ma.mean(vis[np.argmin(_r)-2:np.argmin(_r)+2,], axis=0)
                #_vis_t = vis[np.argmin(_r)-100:np.argmin(_r)+100, ...]
                yield freq, cal_T, s_name, _ra, _dec, cal_data

def get_NED_source(s, freq, beam_off):

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

            cal_T, data = get_calibrator_spec(freq, cal_data=data, beam_off = beam_off)
            #if eta_b is not None:
            #    cal_T /= 0.6
            #    cal_T *= eta_b

            return cal_T, _name, data
    return None, None, None

def mJy2K(freq, eta=1., beam_off=0):

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


#def get_calibrator_spec(freq, cal_data_path='', cal_data=None, 
#        cal_param=None, ra=None, dec=None, beam_off=0):
def get_calibrator_spec(freq, cal_data=None, ra=None, dec=None, beam_off=0):

    cal_spec_func = np.poly1d(np.polyfit(np.log10(cal_data[:,0]*1.e-9),
        np.log10(cal_data[:,1]*1.e3), deg=2,))
                                             #w = 1./ cal_data[:,2]))
                                             #w = 1./ np.log10(cal_data[:,2])))

    cal_flux = 10. ** cal_spec_func(np.log10(freq)) # in mJy

    #kBol = 1.38e6 # mJy m^2 /K
    #eta   = 1. #0.6
    #_lambda = 2.99e8 / (freq * 1.e9)
    #_sigma = 1.02 * _lambda / 300. / 2. / (2. * np.log(2.))**0.5
    ##_sigma[:] = _sigma.max()
    ##_sigma = _lambda / 300. / 2. / (2. * np.log(2.))**0.5
    #Aeff   = eta * _lambda ** 2. / (2. * np.pi * _sigma ** 2.)
    #mJy2K  = Aeff / 2. / kBol

    #mJy2K  = np.pi * 300. ** 2. / 8. / kBol  * 0.9
    #print mJy2K

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
