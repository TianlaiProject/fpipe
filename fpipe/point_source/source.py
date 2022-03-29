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
import gc
_dir_ = os.path.dirname(__file__)

def _get_spec(imap, nmap, pixs, ra, dec, nside, beam_sig, ):

    # find pixel RA/Dec where source located
    p_ra, p_dec = hp.pix2ang(nside, hp.ang2pix(nside, ra, dec, lonlat=True), lonlat=True)
    # find pixel index of 8 neighbours
    _p  = hp.get_all_neighbours(nside, ra, dec, lonlat=True)
    # add the center pixel
    _p = np.append(_p, hp.ang2pix(nside, ra, dec, lonlat=True))
    # get pixel RA/Dec
    _ra, _dec = hp.pix2ang(nside, _p, lonlat=True)
    # get the weight
    dra = (_ra - ra) * np.cos(dec*np.pi/180.) # deg
    ddec = _dec - dec # deg
    _w = np.exp( - 0.5 * (dra**2 + ddec**2 )[None, :] / (beam_sig[:, None] **2) )
    #_w /= np.max(_w, axis=0)[None, :]
    #_w /= np.sum(_w, axis=0)[None, :]

    spec = []
    noise = [] # it is noise inv
    for i in range(_p.shape[0]):
        _idx = (pixs - _p[i]) == 0
        if np.any(_idx) :
            noise.append( nmap[:, _idx])
            spec.append( imap[:, _idx] * nmap[:, _idx] )
            _w[(spec[-1] == 0).flat, i] = 0.
            _w[:, i] *= nmap[:, _idx].flat
        else:
            _w[:, i] = 0.
        del _idx
    if len(spec) == 0: return None
    spec = np.concatenate(spec, axis=1)
    spec = np.ma.masked_invalid(spec)
    _w = np.ma.masked_invalid(_w)
    spec_m = np.ma.sum(spec, axis=1) / np.ma.sum(_w, axis=1)
    del spec

    noise = np.concatenate(noise, axis=1)
    noise = np.ma.masked_invalid(noise)
    noise_m = np.ma.sum(noise, axis=1) / np.ma.sum(_w, axis=1)
    del noise
    # convert to noise error
    noise_m = np.ma.filled(noise_m, 0)
    noise_m[noise_m==0] = np.inf
    noise_m = 1./noise_m
    noise_m = np.ma.sqrt(noise_m)

    return p_ra, p_dec, spec_m, noise_m

def get_map_spec(imap_info, ra, dec, freq, mJy=False):

    imap, pixs, nside, nmap = imap_info
    #freq = imap.get_axis('freq')
    #if freq_sel is None:
    #    freq_sel = np.ones(freq.shape, dtype='bool')
    #freq = freq[freq_sel]
    #imap = imap[freq_sel]
    #nmap = nmap[freq_sel]

    # load beam size data
    data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
    f = data[4:, 0] #* 1.e-3
    d = data[4:, 1:]
    fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")(freq)
    fwhm /= 60. # deg
    beam_sig = (fwhm / (2. * np.sqrt(2.*np.log(2.))))

    _r = _get_spec(imap, nmap, pixs, ra, dec, nside, beam_sig)
    if _r is None:
        return None
    p_ra, p_dec, spec_m, noise_m =  _r
    spec_off = _get_spec(imap, nmap, pixs, ra + 5./60., dec, nside, beam_sig)
    if spec_off is None:
        spec_off = _get_spec(imap, nmap, pixs, ra - 5./60., dec, nside, beam_sig)[2]
    else:
        spec_off = spec_off[2]
    spec_m -= spec_off

    if mJy:
        eta = 1.0
        _lambda = 2.99e8 / (freq * 1.e6)
        _Omega = np.pi * (fwhm * np.pi/180.)**2. / 4. / np.log(2.)
        _K2mJy = 2. * 1380. / (_lambda**2.) * 1.e3 * _Omega
        spec_m   = spec_m  * _K2mJy #* 2. #* 0.5
        noise_m  = noise_m * _K2mJy #* 2. #* 0.5

    return spec_m, p_ra, p_dec, noise_m

def get_nvss_flux(nvss_path, nvss_range, threshold=200, eta_b=None, mJy=False, 
        cut=0, fwhm = 3./60.):

    sig = fwhm / 2. / np.sqrt(2. * np.log(2.))
    nvss_cat = get_nvss_radec(nvss_path, nvss_range)
    freq = 1400.

    for ii in range(nvss_cat.shape[0]):
        _ra = nvss_cat['RA'][ii]
        _dec= nvss_cat['DEC'][ii]
        _flx= nvss_cat['FLUX_20_CM'][ii]
        _dflx = nvss_cat['FLUX_20_CM_ERROR'][ii]
        s_name = nvss_cat['NAME'][ii]

        #_r = ((_ra - ra)**2 + (_dec - dec)**2)**0.5
        #beam_off = _r.min() * 60
        beam_off = 0.
        if nvss_cat['FLUX_20_CM'][ii] > threshold:
            if cut != 0:
                sel  = (nvss_cat['RA']  < (_ra + cut))  * (nvss_cat['RA']  > (_ra - cut))
                sel *= (nvss_cat['DEC'] < (_dec + cut)) * (nvss_cat['DEC'] > (_dec - cut))
                
                ra  = nvss_cat['RA'][sel]
                dec = nvss_cat['DEC'][sel]
                flx = nvss_cat['FLUX_20_CM'][sel]
                r = ( np.sin(_dec*np.pi/180.) * np.sin(dec*np.pi/180.) )\
                  + ( np.cos(_dec*np.pi/180.) * np.cos(dec*np.pi/180.) )\
                  * ( np.cos((_ra - ra)*np.pi/180.) )

                r[r>1] = 1.
                r = np.arccos(r) * 180./np.pi
                r = np.exp( - 0.5 * ( r / sig ) ** 2. ) 
                _flx = np.sum(r * flx)
                del flx, sel, ra, dec, r

            yield freq, _flx, s_name, _ra, _dec, _dflx

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
    #print np.any(nvss_map.mask)
    if fwhm != 0:
        nvss_map = hp.smoothing(nvss_map, fwhm=np.deg2rad(fwhm/60.))
        #sig = np.deg2rad(fwhm/60.) / 2. / np.sqrt(2. * np.log(2.))
        #nvss_map *= 2. * np.pi * sig**2.

    #nvss_map = nvss_map[None, ...]

    return nvss_map, pixs, nside


def project_nvss_to_map_partial(nside, pixs, nvss_path, nvss_range, threshold=500, beam=True):

    if beam:
        data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
        f = data[4:, 0] #* 1.e-3
        d = data[4:, 1:]
        fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")(1400)
        fwhm /= 60.
        sig = fwhm / 2. / np.sqrt(2. * np.log(2.))
    pix_size = hp.nside2resol(nside, arcmin=True) / 60.

    nvss_cat = get_nvss_radec(nvss_path, nvss_range)

    pixs_ra, pixs_dec = hp.pix2ang(nside, pixs, lonlat=True)

    pixs_ra  = pixs_ra.astype('float64')
    pixs_dec = pixs_dec.astype('float64')

    pixs_ra = np.deg2rad(pixs_ra[:, None])
    pixs_dec = np.deg2rad(pixs_dec[:, None])


    nvss_flux = nvss_cat['FLUX_20_CM']
    sel = nvss_flux > threshold
    nvss_ra = np.deg2rad(nvss_cat['RA'][sel][None, :].astype('float64'))
    nvss_dec = np.deg2rad(nvss_cat['DEC'][sel][None, :].astype('float64'))
    nvss_flux = nvss_flux[sel][None, :]

    nvss_map = np.zeros(pixs.shape, dtype='float64')

    step = 20
    nvss_N = nvss_flux.shape[1]
    for i in range(0, nvss_N, step):

        P = ( np.sin(pixs_dec) * np.sin(nvss_dec[:, i:i+step]) )\
          + ( np.cos(pixs_dec) * np.cos(nvss_dec[:, i:i+step]) )\
          * ( np.cos(pixs_ra - nvss_ra[:, i:i+step]) )

        P[P>1] = 1.
        P = np.rad2deg(np.arccos(P))
        #P[P<0.5*pix_size] = 0.
        if beam:
            P = np.exp( - 0.5 * ( P / sig ) ** 2. ) 
            #P *= 1. / (2. * np.pi * sig ** 2.)
            P /= np.max(P, axis=0)[None, :]
        else:
            on = P<0.5*pix_size
            P[on] = 1.
            P[~on] = 0.

        P *= nvss_flux[:, i:i+step]
        P = np.sum(P, axis=1)

        nvss_map = nvss_map + P

        del P
        gc.collect()


    return nvss_map, pixs, nside

def load_nvss_flux_from_map(map_name, nvss_path, nvss_range, nvss_map_path=None, threshold=100,
        output_path=None, beam_size=3./60., f0 = 1400, df=25, flag_iter=5):

    with h5.File(map_name, 'r') as f:
        imap = al.load_h5(f, 'clean_map')
        imap = al.make_vect(imap, axis_names = imap.info['axes'])

        nmap = al.load_h5(f, 'noise_diag')
        nmap = al.make_vect(nmap, axis_names = imap.info['axes'])

        # ignore noisy pixels, not really healfull
        #print nmap.max(), nmap[nmap!=0].min()
        #th = np.percentile(nmap[nmap!=0], 40)
        #nmap[nmap<th] = 0.

        #imap = f['dirty_map'][:]
        pixs = f['map_pix'][:]
        nside = f['nside'][()]

    imap_info = [imap, pixs, nside, nmap]

    if nvss_map_path is None:
        nvss_map = project_nvss_to_map_partial(nside, pixs, nvss_path,
                nvss_range, threshold=10)[0]
        pixs_nvss = pixs
        nside_nvss = nside
    else:
        with h5.File(nvss_map_path, 'r') as f: 
            print 'load nvss from %s'%nvss_map_path
            nvss_map = f['clean_map'][:]
            pixs_nvss = f['map_pix'][:]
            nside_nvss = f['nside'][()]

    spec_list = []
    spec_erro = []
    nvss_flux = []
    nvss_dflux = []
    nvss_flux_from_map = []
    nvss_name = []

    freq_imap = imap.get_axis('freq')
    freq_sel = (freq_imap > f0 - df) * (freq_imap < f0 + df)

    for _s in get_nvss_flux(nvss_path, nvss_range, threshold, mJy=True, cut=0.5, fwhm=3./60.):

        _r = get_map_spec(imap_info, _s[3], _s[4], mJy=True, freq_sel=freq_sel)
        if _r is not None:
            freq, spec, p_ra, p_dec, error = _r
        else:
            continue
        spec = np.ma.masked_equal(spec, 0)

        #if beam_size is not None:
        #    beam_sig = beam_size / (2. * np.sqrt(2.*np.log(2.)))
        #    _w = np.exp(-0.5*(((p_ra - _s[3])*np.cos(np.radians(p_dec)))**2 \
        #            + (p_dec - _s[4])**2)/beam_sig **2 )
        #else:
        #    _w = 1.
        _w = 1.

        if not (np.all(spec.mask) or spec.shape[0] == 0):
            spec_list.append(spec/_w)
            spec_erro.append(error/_w)
            nvss_flux.append(_s[1])
            nvss_dflux.append(_s[5])
            nvss_name.append(_s[2])

            _pix = hp.ang2pix(nside_nvss, _s[3], _s[4], lonlat=True)
            _pix = np.where(pixs_nvss == _pix)[0][0]
            nvss_flux_from_map.append(nvss_map[_pix])
        del spec, error, _r, freq
        gc.collect()

    spec = np.ma.array(spec_list)
    spec.mask = spec==0

    error = np.ma.array(spec_erro)
    error.mask = spec==0

    for i in range(flag_iter):

        flux = np.ma.median(spec, axis=1)
        dflux = np.ma.std(spec, axis=1)

        mask = np.ma.abs( spec - flux[:, None] ) - 4 * dflux[:, None] > 0
        if not np.any(mask): break
        #print np.sum(mask.astype('int'))
        spec.mask[mask] = True
        error.mask[mask] = True

    #print np.sum(spec.mask.astype('int'))
    nfreq = np.sum((~spec.mask).astype('int'), axis=1) #spec.shape[1]
    flux = np.ma.median(spec, axis=1)
    dflux = np.ma.std(spec, axis=1) / np.sqrt(nfreq)

    error = error ** 2.
    error[error==0] = np.inf
    #error = 1./error
    #error = 1./np.sqrt(np.ma.mean(error**2, axis=1))
    error = np.ma.sum(1./error, axis=1)
    error[error==0] = np.inf
    error = np.sqrt(1./error)
    #error[error==0] = np.inf

    return nvss_name, nvss_flux, nvss_flux_from_map, spec_list, flux, dflux, error, nvss_dflux




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

    _lambda = 2.99e8 / (freq * 1.e9)
    if fwhm is None:
        data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
        f = data[4:, 0] * 1.e-3
        d = data[4:, 1:]
        fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")

        #_sigma = 1.02 *  _lambda / 300. / np.pi * 180. * 3600.
        _sigma = fwhm(freq) * 60.
    else:
        _sigma = fwhm * 60.

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
