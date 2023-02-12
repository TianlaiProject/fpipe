# modules used for flux checking

import numpy as np

from fpipe.timestream import data_format
from fpipe.timestream.data_base import DATA_BASE as DB
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter as mfilter
from scipy.ndimage import gaussian_filter as gfilter
import os
import gc
import healpy as hp
import h5py


_dir_ = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
b_G = lambda r, sigma: np.exp(-0.5 * r**2/sigma**2)

beam_model = b_G
# ---------------------------------------------------------------------------

def fwhm_func(fwhm, freq):
    if fwhm is None:
        data = np.loadtxt(_dir_ + '/../data/fwhm.dat')
        f = data[4:, 0] * 1.e-3
        d = data[4:, 1:]
        fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")
        return fwhm(freq) #* 60.
    else:
        return fwhm #* 60.

def mJy2K(freq, eta=1., beam_off=0, fwhm=None):
    # freq GHz

    _lambda = 2.99e8 / (freq * 1.e9)
    _sigma = fwhm_func(fwhm, freq) * 60.

    if beam_off != 0:
        fact = np.exp(-0.5 * beam_off**2 / (_sigma/2./(2.*np.log(2.))**0.5 )**2 )
    else:
        fact = 1.

    return eta * 1.36 * (_lambda*1.e2)**2. / _sigma**2. * fact

def get_nvss_radec(nvss_path, nvss_range):

    hdulist = pyfits.open(nvss_path)
    data = hdulist[1].data

    _sel = np.zeros(data['RA'].shape[0], dtype ='bool')
    for _nvss_range in nvss_range:

        ra_min, ra_max, dec_min, dec_max = _nvss_range

        sel  = data['FIRST_RA'] > ra_min
        sel *= data['FIRST_RA'] < ra_max
        sel *= data['FIRST_DEC'] > dec_min
        sel *= data['FIRST_DEC'] < dec_max

        _sel = _sel + sel

    hdulist.close()

    return data[_sel]

def load_catalogue(nvss_path_list, nvss_range, flux_key, name_key, 
        flux_lim=10, threshold=0, max_major_axis = 10000):

    nvss_ra, nvss_dec, nvss_flx, nvss_name = [], [], [], []
    first_major, first_minor,  = [], []
    for nvss_path in nvss_path_list:
        nvss_cat = get_nvss_radec(nvss_path, nvss_range)
        #print nvss_cat['RA'].min(), nvss_cat['RA'].max()
        #_flx = nvss_cat[flux_key]
        #print(_flx[_flx>0].min() )
        #print(nvss_cat['FIRST_RA'].shape)
        #if nvss_cat['FIRST_RA'].shape[0] != 0:
        #    print(nvss_cat['FIRST_RA'][:].min())
        #    print(nvss_cat['NVSS_RA'][:].min())

        nvss_sel = nvss_cat[flux_key] > 0
        first_major.append(nvss_cat['FIRST_MAJOR'][nvss_sel])
        first_minor.append(nvss_cat['FIRST_MINOR'][nvss_sel])
        nvss_ra.append(nvss_cat['FIRST_RA'][nvss_sel])
        nvss_dec.append(nvss_cat['FIRST_DEC'][nvss_sel])
        #nvss_ra.append(nvss_cat['NVSS_RA'][nvss_sel])
        #nvss_dec.append(nvss_cat['NVSS_DEC'][nvss_sel])
        nvss_flx.append(nvss_cat[flux_key][nvss_sel])
        nvss_name.append(nvss_cat[name_key][nvss_sel])

        del nvss_cat
        gc.collect()

    nvss_ra = np.concatenate(nvss_ra, axis=0)
    nvss_dec = np.concatenate(nvss_dec, axis=0)
    nvss_flx = np.concatenate(nvss_flx, axis=0)
    nvss_name = np.concatenate(nvss_name, axis=0)
    first_major = np.concatenate(first_major, axis=0)
    first_minor = np.concatenate(first_minor, axis=0)

    ## remove redundancy according to name
    n_cat = nvss_ra.shape[0]
    nvss_name, redu = np.unique(nvss_name, return_index=True)
    nvss_ra  = nvss_ra[redu]
    nvss_dec = nvss_dec[redu]
    nvss_flx = nvss_flx[redu]
    first_major = first_major[redu]
    first_minor = first_minor[redu]
    print('%d of %d sources left with redundancy removing'%(nvss_ra.shape[0], n_cat))

    if threshold > 0:
        nvss_ra, nvss_dec, nvss_flx, nvss_name, first_major, first_minor = \
                isolate_source(nvss_ra, nvss_dec, nvss_flx, nvss_name, 
                        first_major, first_minor, threshold=threshold)

    nvss_sel = nvss_flx > flux_lim
    print('%d of %d sources left with flux lim cut %f mJy'%(np.sum(nvss_sel.astype('int')), 
                                                           nvss_sel.shape[0], flux_lim))
    #point_source = (first_major - first_minor) / np.sqrt(first_major * first_minor) < 0.2
    #nvss_sel = nvss_sel * point_source
    nvss_sel = nvss_sel * (first_major < max_major_axis)

    return nvss_ra[nvss_sel], nvss_dec[nvss_sel], nvss_flx[nvss_sel], nvss_name[nvss_sel]


def isolate_source(nvss_ra, nvss_dec, nvss_flx, nvss_name, first_major, first_minor, threshold=2.):

    r = np.sin(np.radians(nvss_dec[:, None])) * np.sin(np.radians(nvss_dec[None, :])) \
      + np.cos(np.radians(nvss_dec[:, None])) * np.cos(np.radians(nvss_dec[None, :])) \
      * np.cos(np.radians( nvss_ra[:, None])  - np.radians(nvss_ra[None, :]))
    r[r>1] = 1.
    r = np.rad2deg(np.arccos(r))
    r[r==0] = 1000
    bad = r < threshold/60.

    flux_diff = nvss_flx[:, None] / nvss_flx[None, :]
    #flux_diff = flux_diff > 0.2
    flux_diff = flux_diff > 0.1
    bad *= flux_diff
    bad  = np.any(bad, axis=0)

    n_cat = nvss_ra.shape[0]
    nvss_ra   = nvss_ra[~bad]
    nvss_dec  = nvss_dec[~bad]
    nvss_flx  = nvss_flx[~bad]
    nvss_name = nvss_name[~bad]
    first_major = first_major[~bad]
    first_minor = first_minor[~bad]

    print('%d of %d sources left with threshold cut %f arcmin'%(nvss_ra.shape[0], 
                                                                n_cat, threshold))

    del r
    gc.collect()

    return nvss_ra, nvss_dec, nvss_flx, nvss_name, first_major, first_minor



def iter_nvss_flux_tod(fdata, nvss_cat, fwhm=None, beam_list=None, max_dist=3):

    vis  = fdata.data
    time = fdata.time
    freq = fdata.freq
    mask = fdata.mask
    ns_on= fdata.ns_on
    vis = np.ma.array(vis, mask=mask)
    vis.mask += ns_on[:, None, None, :]
    ra, dec = fdata.ra, fdata.dec

    nvss_ra, nvss_dec, nvss_flx, nvss_name = nvss_cat

    if beam_list is None:
        beam_list = fdata.ants[:, 0]

    fwhm = fwhm_func(fwhm, freq*1.e-3)
    sigma = fwhm / 2. / (2.*np.log(2.))**0.5

    nvss_ra  = nvss_ra[None, :]
    nvss_dec = nvss_dec[None, :]

    nvss_ra  = nvss_ra.astype('float64')
    nvss_dec = nvss_dec.astype('float64')

    for beam in beam_list:

        i = beam - 1

        _ra = ra[:, i][:, None]
        _dec= dec[:, i][:, None]

        _ra  = _ra.astype('float64')
        _dec = _dec.astype('float64')

        # angular distance between source and pointing direction
        r = np.sin(np.radians(_dec)) * np.sin(np.radians(nvss_dec)) \
          + np.cos(np.radians(_dec)) * np.cos(np.radians(nvss_dec)) \
          * np.cos(np.radians(_ra) - np.radians(nvss_ra))
        r[r>1] = 1.
        r = np.arccos(r)

        # parallel angle of the source
        p = (np.sin(r) * np.cos(np.radians(_dec)))
        p[p==0] = np.inf
        p = (np.sin(np.radians(nvss_dec))-np.cos(r)*np.sin(np.radians(_dec)))/p
        p[p>1] = 1.
        p[p<-1] = -1.
        p = np.arccos(p) * np.sign(_ra - nvss_ra)

        r *= 180./np.pi * 60. # arcmin
        p *= 180./np.pi * 60. # arcmin

        t_where, nvss_where = np.where(r<max_dist)
        nvss_on = np.unique(nvss_where)
        for nvss_idx in nvss_on:
            t_sel_c = np.median(t_where[nvss_where==nvss_idx])
            t_sel_c = int(t_sel_c)
            t_sel = np.arange(t_sel_c-50, t_sel_c+50)

            t_sel = t_sel[t_sel>0]
            t_sel = t_sel[t_sel<r.shape[0]]
            t_sel = list(t_sel)

            # all nvss sources
            rr = r[t_sel, :]
            pp = p[t_sel, :]
            rr_min = rr.min()
            flux_model = np.sum( beam_model(rr[:, :, None], sigma[None, None, :])\
                                 * nvss_flx[None, :, None], axis=1)
            flux_model = np.mean(flux_model, axis=1)

            obs_time = time[t_sel]

            flux_measured = vis[t_sel,:,:,i]/mJy2K(freq*1.e-3)[None, :, None]
            #flux_measured = np.ma.mean(flux_measured, axis=1)
            norm = np.sum((~flux_measured.mask).astype('int'), axis=1) * 1.
            flux_measured_rms = np.ma.std(flux_measured, axis=1) / np.sqrt(norm)
            flux_measured = np.ma.median(flux_measured, axis=1)

            yield beam, nvss_name[nvss_idx], nvss_flx[nvss_idx], \
                  nvss_ra[0, nvss_idx], nvss_dec[0, nvss_idx], \
                  flux_model, obs_time, flux_measured, flux_measured_rms, \
                  rr[:, nvss_idx], pp[:, nvss_idx]


def output_source(file_root, file_name_list, output_path, nvss_path_list,
        fwhm=None, flux_key='FLUX_20_CM', name_key='NAME',
        iso_threshold=0, max_major_axis=20, fmin=3000, fmax=4500, f0=1400, df=25, 
        flux_lim=10, debug=False, max_dist=0.5, nvss_range=None):

    nvss_used_total = []
    for file_name in file_name_list:

        fdata = data_format.FASTh5_Spec(file_root + '%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(
            file_name, 1, 1))
        freq = fdata.freq
        fsel = ( freq > f0 - df ) * ( freq < f0 + df )
        fidx = np.arange(freq.shape[0])[fsel]
        fmin, fmax = fidx.min(), fidx.max()
        del fdata
        gc.collect()


        file_list = [file_root + '%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(file_name, i, i)
                     for i in range(1, 8)]
        output_name = file_name.replace('/', '_')

        fdata = data_format.FASTh5_Spec(file_list, fmin=fmin, fmax=fmax + 1)
        print("Frequency range %5.2f - %5.2f MHz"%(fdata.freq.min(), fdata.freq.max()))
        if debug: continue

        ra, dec = fdata.ra, fdata.dec
        if nvss_range is None:
            nvss_range = [[ra.min(), ra.max(), dec.min(), dec.max()],]
        nvss_cat = load_catalogue(nvss_path_list, nvss_range, flux_key, name_key,
                                  flux_lim=flux_lim, threshold=iso_threshold, 
                                  max_major_axis=max_major_axis)

        with h5py.File(output_path + output_name + '.h5', 'w') as f:
            print(output_name)
            for oo in iter_nvss_flux_tod(fdata, nvss_cat, fwhm, max_dist=max_dist):
                beam, nvss_name, nvss_flx, nvss_ra,\
                nvss_dec, flux_model, obs_time, flux_measured, flux_measured_rms, rr, pp = oo
                #print nvss_name, flux_measured, flux_measured_rms
                nvss_used_total.append(nvss_name)

                if isinstance(nvss_name, str):
                    nvss_name = nvss_name.replace(' ', '_')
                elif isinstance(nvss_name, int):
                    nvss_name = 'NVSSID_%d'%nvss_name
                if not np.isfinite(flux_measured[rr.argmin()][0]):
                    continue
                print('Beam %02d %s [%10.5f %10.5f] TOD:%10.5f +- %10.5f'%(
                        beam, nvss_name, nvss_ra, nvss_dec, flux_measured[rr.argmin()][0], 
                        flux_measured_rms[rr.argmin()][0]))
                f['B%02d/%s/NVSS_FLUX'%(beam, nvss_name)] = flux_model
                f['B%02d/%s/TIME_STEP'%(beam, nvss_name)] = obs_time
                f['B%02d/%s/FAST_FLUX'%(beam, nvss_name)] = flux_measured
                f['B%02d/%s/BEAM_DIST'%(beam, nvss_name)] = rr
                f['B%02d/%s/BEAM_ANGL'%(beam, nvss_name)] = pp
                f['B%02d/%s/FLUXRADEC'%(beam, nvss_name)] = (nvss_flx, nvss_ra, nvss_dec)
                f['B%02d/%s/NVSS_NAME'%(beam, nvss_name)] = nvss_name
        print()

    return nvss_used_total
# -------------------------------------------------------------------------------------------------
# functions for plot

def check_flux_raw(file_list, nvss_path_list, beam_list=None,
               fwhm=None, flux_key='FLUX_20_CM', name_key='NAME',
               fmin=3000, fmax=4500,
               sub_mean=True, baseline_file=None):

    fdata = data_format.FASTh5_Spec(file_list, fmin=fmin, fmax=fmax)
    print("Frequency range %5.2f - %5.2f MHz"%(fdata.freq.min(), fdata.freq.max()))

    ra, dec = fdata.ra, fdata.dec
    nvss_range = [[ra.min(), ra.max(), dec.min(), dec.max()],]
    nvss_cat = load_catalogue(nvss_path_list, nvss_range, flux_key, name_key,
                                 flux_lim=10, threshold=6, max_major_axis=20)

    for oo in iter_nvss_flux_tod(fdata,nvss_cat,fwhm,beam_list,max_dist=0.5):
        beam, nvss_name, nvss_flx, nvss_ra,\
        nvss_dec, flux_model, obs_time, flux_measured, flux_measured_rms, rr, pp = oo

        fig = plt.figure(figsize=(6, 2))
        ax  = fig.add_axes([0.1, 0.1, 0.9, 0.9])

        #flux_measured = np.ma.sum(flux_measured, axis=1)/2.
        if sub_mean:
            flux_measured -= np.ma.median(flux_measured, axis=0)
        ax.plot(obs_time - obs_time[0], flux_measured[:, 0], '.-')
        ax.plot(obs_time - obs_time[0], flux_measured[:, 1], '.-')

        rr_min = rr.min()
        label='%6.4f arcmin to Beam %02d'%(rr_min, beam)
        title = '%s\n %6.4f mJy'%(nvss_name, nvss_flx)

        ax.plot(obs_time - obs_time[0], flux_model, 'r', label=label)
        ax.legend(title = title, loc=1)
        #ax.set_ylim(ymin=1.e-1)
        #ax.semilogy()

        plt.show()
        plt.close(fig)

def get_flux_tod_rms(flux_measured, rr):

    mask = flux_measured.mask
    flux_smooth = np.ma.zeros(flux_measured.shape)
    #flux_smooth[~mask[:,0], 0] = gfilter(flux_measured[~mask[:, 0], 0], sigma=2)
    #flux_smooth[~mask[:,1], 1] = gfilter(flux_measured[~mask[:, 1], 1], sigma=2)
    flux_smooth[~mask] = gfilter(flux_measured[~mask], sigma=2)
    flux_smooth.mask = mask
    flux_smooth = flux_smooth - flux_measured
    idx_l = max(0, rr.argmin()-10)
    idx_r = min(rr.shape[0], rr.argmin()+10)
    #bg = 0.5 * (flux_smooth[idx_l]+ flux_smooth[idx_r-1])
    flux_smooth.mask[idx_l:idx_r] = True
    #print np.ma.std(flux_smooth, axis=0)

    return np.ma.std(flux_smooth, axis=0) #, bg

def get_flux_tod(rr, flux_tod, flux_model, flux_nvss, beam=False):

    flux_tod  = np.ma.masked_equal(flux_tod, 0)
    sel_l = slice(rr.argmin() - 20, rr.argmin() - 10)
    sel_r = slice(rr.argmin() + 10, rr.argmin() + 20)
    bg = np.ma.median(flux_tod[sel_l], axis=0) + np.ma.median(flux_tod[sel_r], axis=0)
    bg = bg / 2.
    flux_tod -= bg[None, :]
    #flux_tod -= np.ma.median(flux_tod, axis=0)[None, :]
    flux_tod  = np.ma.mean(flux_tod, axis=1)

    #print np.ma.std(flux_tod[sel_l]), np.ma.std(flux_tod[sel_r])

    flux_tod[flux_tod.mask] = 0
    flux_model[flux_tod.mask] = 0

    flux_rms = get_flux_tod_rms(flux_tod, rr)

    if beam:
        profile = flux_model / flux_nvss
        profile[flux_tod.mask]  = 0
        sel = slice(rr.argmin() - 6, rr.argmin() + 6)
        norm = np.ma.sum(profile[sel])
        return np.ma.sum(flux_tod[sel])/norm, flux_model[rr.argmin()], flux_rms 
        #return np.ma.sum(flux_tod[sel])/norm, np.sum(flux_model[sel])/norm
    else: 
        return flux_tod[rr.argmin()], flux_model[rr.argmin()], flux_rms 

def check_flux(tod_flux_files, beam_list=None,
               fwhm=None, flux_key='FLUX_20_CM', name_key='NAME',
               fmin=3000, fmax=4500,
               sub_mean=True, baseline_file=None):

    for file_name in tod_flux_files:
        data_key = file_name.split('/')[-1]
        data_key = data_key.split('_')[0]
        bad_beam = DB[data_key]['BAD_FEED']
        with h5py.File(file_name, 'r') as f:
            if beam_list is None:
                beam_list = [int(x[1:]) for x in list(f.keys())]

            for b in beam_list:
                if b in bad_beam:
                    print("%02d is bad beam, continue"%b)
                    continue
                #print f['B%02d'%b].keys()
                for s in list(f['B%02d'%b].keys()):
                    #print f['B%02d/%s'%(b,s)].keys()

                    fig = plt.figure(figsize=(6, 2))
                    ax  = fig.add_axes([0.1, 0.1, 0.9, 0.9])

                    obs_time = f['B%02d/%s/TIME_STEP'%(b,s)][:]
                    obs_time -= obs_time[0]

                    rr = f['B%02d/%s/BEAM_DIST'%(b,s)][:]

                    flux_measured = f['B%02d/%s/FAST_FLUX'%(b,s)][:]
                    flux_measured = np.ma.masked_equal(flux_measured, 0)
                    if sub_mean:
                        flux_measured -= np.ma.median(flux_measured, axis=0)
                    ax.plot(obs_time, flux_measured[:, 0], '.-')
                    ax.plot(obs_time, flux_measured[:, 1], '.-')

                    mask = flux_measured.mask
                    flux_smooth = np.ma.zeros(flux_measured.shape)
                    flux_smooth[~mask[:,0], 0] = gfilter(flux_measured[~mask[:, 0], 0], sigma=2)
                    flux_smooth[~mask[:,1], 1] = gfilter(flux_measured[~mask[:, 1], 1], sigma=2)
                    flux_smooth.mask = mask
                    flux_smooth = flux_smooth - flux_measured
                    idx_l = max(0, rr.argmin()-10)
                    idx_r = min(rr.shape[0], rr.argmin()+10)
                    flux_smooth.mask[idx_l:idx_r] = True
                    print(np.ma.std(flux_smooth, axis=0))

                    ax.plot(obs_time, flux_smooth[:, 0], '-')
                    ax.plot(obs_time, flux_smooth[:, 1], '-')

                    flux_model = f['B%02d/%s/NVSS_FLUX'%(b,s)][:]
                    #rr = f['B%02d/%s/BEAM_DIST'%(b,s)][:]
                    nvss_flx = f['B%02d/%s/FLUXRADEC'%(b,s)][0]

                    rr_min = rr.min()
                    label='%6.4f arcmin to Beam %02d'%(rr_min, b)
                    title = '%s: %6.4f mJy'%(s, nvss_flx)

                    ax.plot(obs_time, flux_model, 'r', label=label)
                    ax.legend(title = title, loc=1)
                    #ax.set_ylim(ymin=1.e-1)
                    #ax.semilogy()
                    ax.axvline(obs_time[rr.argmin()],   0, 1, c='k', ls='--')
                    ax.axvline(obs_time[rr.argmin()-6], 0, 1, c='k', ls=':')
                    ax.axvline(obs_time[rr.argmin()+6], 0, 1, c='k', ls=':')

                    plt.show()
                    plt.close(fig)

def iter_flux_beam(beam_list, flux_path_list, tod=True):

    for i in beam_list:
        flux_nvss = []
        flux_fast = []
        flux_rms  = []
        name_nvss = []
        coord_nvss = []
        for flux_path in flux_path_list:
            data_key = flux_path.split('/')[-1]
            data_key = data_key.split('_')[0]
            bad_beam = DB[data_key]['BAD_FEED']
            if i in bad_beam:
                #print("BadBeam%02d"%i)
                continue
            with h5py.File(flux_path, 'r') as f:
                if 'B%02d'%i not in f.keys():
                    #print("BadBeam%02d"%i)
                    continue
                for s in list(f['B%02d'%i].keys()):
                    if tod:
                        rr = f['B%02d/%s/BEAM_DIST'%(i, s)][:]
                        #flux_nvss.append(f['B%02d/%s/NVSS_FLUX'%(i, s)][np.argmin(rr)])
                        #flux_fast.append(f['B%02d/%s/FAST_FLUX'%(i, s)][np.argmin(rr)].sum() * 0.5)

                        _flux_beam = f['B%02d/%s/NVSS_FLUX'%(i, s)]
                        _flux_nvss = f['B%02d/%s/FLUXRADEC'%(i, s)][0]
                        _flux_fast = f['B%02d/%s/FAST_FLUX'%(i, s)]
                        _coord     = f['B%02d/%s/FLUXRADEC'%(i, s)][1:]

                        y, x, e = get_flux_tod(rr, _flux_fast, _flux_beam, _flux_nvss)

                        flux_nvss.append(x)
                        flux_fast.append(y)
                        flux_rms.append(e)
                        name_nvss.append(s)
                        coord_nvss.append(_coord)

                    else:
                        #flux_nvss.append(f['B%02d/%s/MAPS_NVSS'%(i, s)][()])
                        flux_nvss.append(f['B%02d/%s/FLUXRADEC'%(i, s)][0])
                        flux_fast.append(f['B%02d/%s/MAPS_FLUX'%(i, s)][()])
        yield "Feed %02d"%i, np.array(flux_nvss), np.array(flux_fast), np.array(flux_rms),\
                np.array(name_nvss), np.array(coord_nvss)

def iter_flux_days(beam_list, flux_path_list, tod=True):

    for flux_path in flux_path_list:

        data_key = flux_path.split('/')[-1]
        data_key = data_key.split('_')[0]
        bad_beam = DB[data_key]['BAD_FEED']

        flux_nvss = []
        flux_fast = []
        flux_rms = []
        name_nvss = []
        coord_nvss = []
        label = flux_path.split('/')[-1]
        label = label.replace('.h5', '')
        label = label.replace('_', ' ')
        with h5py.File(flux_path, 'r') as f:
            for i in beam_list:

                if i in bad_beam:
                    #print("BadBeam%02d"%i)
                    continue

                if 'B%02d'%i not in f.keys():
                    #print("BadBeam%02d"%i)
                    continue

                for s in list(f['B%02d'%i].keys()):
                    if tod:
                        rr = f['B%02d/%s/BEAM_DIST'%(i, s)][:]
                        #flux_nvss.append(f['B%02d/%s/NVSS_FLUX'%(i, s)][np.argmin(rr)])
                        #flux_fast.append(f['B%02d/%s/FAST_FLUX'%(i, s)][np.argmin(rr)].sum() * 0.5)

                        _flux_beam = f['B%02d/%s/NVSS_FLUX'%(i, s)]
                        _flux_nvss = f['B%02d/%s/FLUXRADEC'%(i, s)][0]
                        _flux_fast = f['B%02d/%s/FAST_FLUX'%(i, s)]
                        _coord     = f['B%02d/%s/FLUXRADEC'%(i, s)][1:]

                        y, x, e = get_flux_tod(rr, _flux_fast, _flux_beam, _flux_nvss)

                        flux_nvss.append(x)
                        flux_fast.append(y)
                        flux_rms.append(e)
                        name_nvss.append(s)
                        coord_nvss.append(_coord)
                    else:
                        #flux_nvss.append(f['B%02d/%s/MAPS_NVSS'%(i, s)][()])
                        flux_nvss.append(f['B%02d/%s/FLUXRADEC'%(i, s)][0])
                        flux_fast.append(f['B%02d/%s/MAPS_FLUX'%(i, s)][()])
        yield label, np.array(flux_nvss), np.array(flux_fast), np.array(flux_rms),\
                np.array(name_nvss), np.array(coord_nvss)


def iter_flux_day_vs_days(beam_list, flux_path_list, tod=True):

    with h5py.File(flux_path_list[0], 'r') as f0:

        data_key = flux_path_list[0].split('/')[-1]
        data_key = data_key.split('_')[0]
        bad_beam0 = DB[data_key]['BAD_FEED']
        #print bad_beam0

        for flux_path in flux_path_list[1:]:

            data_key = flux_path.split('/')[-1]
            data_key = data_key.split('_')[0]
            bad_beam = DB[data_key]['BAD_FEED']

            #print bad_beam

            flux_ref0 = []
            flux_fast = []
            name_nvss = []
            label = flux_path.split('/')[-1]
            label = label.replace('.h5', '')
            label = label.replace('_', ' ')
            with h5py.File(flux_path, 'r') as f:
                for i in beam_list: # iter through ref

                    if i in bad_beam or 'B%02d'%i not in f.keys():
                        #print "Ref:BadBeam%02d"%i
                        continue

                    for s in list(f0['B%02d'%i].keys()): # iter through ref
                        for j in beam_list: # search in individual files

                            if j in bad_beam0:
                                #print "    BadBeam%02d"%j
                                continue

                            try:
                                if tod:
                                    rr = f['B%02d/%s/BEAM_DIST'%(j, s)][:]
                                    #yy = f['B%02d/%s/FAST_FLUX'%(j, s)][np.argmin(rr)].sum()*0.5
                                    _flux_beam = f['B%02d/%s/NVSS_FLUX'%(i, s)]
                                    _flux_nvss = f['B%02d/%s/FLUXRADEC'%(i, s)][0]
                                    _flux_fast = f['B%02d/%s/FAST_FLUX'%(i, s)]
                                    yy = get_flux_tod(rr, _flux_fast, _flux_beam, _flux_nvss)[0]
                                    flux_fast.append(yy)

                                    rr = f0['B%02d/%s/BEAM_DIST'%(i, s)][:]
                                    _flux_beam = f0['B%02d/%s/NVSS_FLUX'%(i, s)]
                                    _flux_nvss = f0['B%02d/%s/FLUXRADEC'%(i, s)][0]
                                    _flux_fast = f0['B%02d/%s/FAST_FLUX'%(i, s)]
                                    yy = get_flux_tod(rr, _flux_fast, _flux_beam, _flux_nvss)[0]
                                    flux_ref0.append(yy)
                                    name_nvss.append(s)
                                else:
                                    flux_fast.append( f['B%02d/%s/MAPS_FLUX'%(j, s)][()])
                                    flux_ref0.append(f0['B%02d/%s/MAPS_FLUX'%(i, s)][()])
                            except KeyError:
                                pass
            yield label, np.array(flux_ref0), np.array(flux_fast), None, np.array(name_nvss), None

def plot_flux_chisq(flux_path_list, beam_list=[1, ], axes=None, rms_sys=None,
        #bins=np.linspace(0, 10, 41), 
        bins=np.logspace(-1, np.log10(30), 31), 
        ymin=1.e-4, ymax=0.60, tod=True, iter_func=iter_flux_beam, label_list=None):

    if axes is None:
        fig = plt.figure(figsize = (6, 4))
        ax = fig.subplots()
    else:
        fig, ax = axes

    #bins = np.linspace(-1.5, 2.5, 41)
    #bins = np.linspace(-1, 1, 31)

    #bins_c = 0.5 * ( bins[:-1] + bins[1:] )
    bins_c = (bins[:-1] * bins[1:]) ** 0.5
    hist_list = []
    sigma_list = []
    sigma_rel_list = []
    legend_list = []

    color=iter(cm.tab20(np.linspace(0,1,12)))

    hist_list = []
    for label, x, y, e, name, coord in iter_func(beam_list, flux_path_list, tod=tod):

        good = ( x > 1.e-1 ) * ( y > 1.e-1 )
        x = x[good]
        y = y[good]
        e = e[good]

        diff = (y - x)
        chisq = ( diff ** 2 ) / ( e ** 2 )

        hist = np.histogram(diff, bins=bins)[0] * 1.
        hist_list.append(hist[None, :])
        hist /= np.sum(hist)

        l = ax.plot(bins_c, hist, '-', color=next(color), drawstyle='steps-mid')[0]
        #l = ax.plot(bins_c, hist, '.-', color=next(color))[0]
        legend_list.append(mpatches.Patch(color=l.get_color(), label=label))

    if len(hist_list) > 1:
        hist = np.concatenate(hist_list, axis=0)
        hist = np.sum(hist, axis=0)
        hist /= np.sum(hist)
        ax.plot(bins_c, hist, 'k-', linewidth=2., drawstyle='steps-mid')

    ax.set_xlim(bins.min(), bins.max())
    #ax.set_ylim(1.e-4, 0.35)
    ax.set_ylim(ymin, ymax)
    #ax.loglog()
    #ax.semilogy()
    #ax.semilogx()
    ncol = 1
    if len(legend_list) > 5: ncol=4
    ax.legend(handles=legend_list, frameon=False, markerfirst=True, loc=1, ncol=ncol)
    #ax.set_aspect('equal')
    if axes is None:
        ax.set_xlabel('NVSS Flux at 1400 MHz (Jy)')
        ax.set_ylabel('Measured Flux at 1400 MHz (Jy)')

    #return fig, ax

def plot_flux_hist(flux_path_list, beam_list=[1, ], axes=None, rms_sys=None,
                        bins=np.linspace(-49, 49,31), ymin=1.e-4, ymax=0.60, tod=True,
                        iter_func=iter_flux_beam, label_list=None, diff_fraction=False):

    print("="*40)
    if axes is None:
        fig = plt.figure(figsize = (6, 6))
        ax = fig.subplots()
    else:
        fig, ax = axes

    #bins = np.linspace(-1.5, 2.5, 41)
    #bins = np.linspace(-1, 1, 31)

    bins_c = 0.5 * ( bins[:-1] + bins[1:] )
    hist_list = []
    sigma_list = []
    sigma_rel_list = []
    legend_list = []

    color=iter(cm.tab20(np.linspace(0,1,12)))

    hist_list = []
    name_list = []
    for label, x, y, e, name, coord in iter_func(beam_list, flux_path_list, tod=tod):

        #good = ( x > 1.e-1 ) * ( y > 1.e-1 )
        good = ( x > 1.e0 ) * ( y > 1.e0 )
        x = x[good]
        y = y[good]
        name = name[good]
        name_list += list(name)

        diff = (y - x)
        if diff_fraction:
            diff = (y - x) / np.sqrt(x * y)

        sigma2 = np.sqrt( np.median( (y - x) ** 2 ) )
        sigma2_rel = np.sqrt( np.median( ( (y - x) / np.sqrt(x * y) )**2. ) )
        #sigma2 = np.sqrt( np.mean( (y - x) ** 2 ) )
        #sigma2_rel = np.sqrt( np.mean( ( (y - x) / np.sqrt(x * y) )**2. ) )

        sigma_rel_list.append( (y - x) / (x * y)**0.5 )
        sigma_list.append( y - x )
        print('%s: rms %f (%f) N =%3d'%(label, sigma2, sigma2_rel, len(y)))

        hist = np.histogram(diff, bins=bins)[0] * 1.
        hist_list.append(hist[None, :])
        hist /= np.sum(hist)

        l = ax.plot(bins_c, hist, '-', color=next(color), drawstyle='steps-mid')[0]
        #l = ax.plot(bins_c, hist, '.-', color=next(color))[0]
        legend_list.append(mpatches.Patch(color=l.get_color(), label=label))

    if len(hist_list) > 1:
        hist = np.concatenate(hist_list, axis=0)
        hist = np.sum(hist, axis=0)
        hist /= np.sum(hist)
        ax.plot(bins_c, hist, 'k-', linewidth=2., drawstyle='steps-mid')

    sigma_list = np.concatenate(sigma_list)
    sigma2 = np.sqrt(np.median(sigma_list ** 2.))
    sigma_rel_list = np.concatenate(sigma_rel_list)
    sigma_rel = np.sqrt(np.median(sigma_rel_list ** 2))
    #print('-'*40)
    print()
    print('Average : rms %f (%f) '%(sigma2, sigma_rel))

    ax.set_xlim(bins.min(), bins.max())
    #ax.set_ylim(1.e-4, 0.35)
    ax.set_ylim(ymin, ymax)
    #ax.loglog()
    #ax.semilogy()
    ncol = 1
    #if len(legend_list) > 5: ncol=2
    ax.legend(handles=legend_list, frameon=False, markerfirst=True, loc=2, ncol=ncol, mode='expand')
    #ax.set_aspect('equal')
    if axes is None:
        ax.set_xlabel('NVSS Flux at 1400 MHz (Jy)')
        ax.set_ylabel('Measured Flux at 1400 MHz (Jy)')

    #print("-"*40)
    print()
    print("Totally there are %4d (%4d) sources"%(len(name_list), len(set(name_list))))
    print()

    #return fig, ax

def plot_flux_hist_ra(flux_path_list, beam_list=[1, ], axes=None, rms_sys=None,
                        bins=np.linspace(-49, 49,31), ymin=1.e-4, ymax=0.60, tod=True,
                        iter_func=iter_flux_beam, label_list=None, diff_fraction=False):

    if axes is None:
        fig = plt.figure(figsize = (6, 6))
        ax = fig.subplots()
    else:
        fig, ax = axes

    legend_list = []

    color=iter(cm.tab20(np.linspace(0,1,12)))

    name_list = []
    for label, x, y, e, name, coord in iter_func(beam_list, flux_path_list, tod=tod):

        #good = ( x > 1.e-1 ) * ( y > 1.e-1 )
        good = ( x > 1.e0 ) * ( y > 1.e0 )
        x = x[good]
        y = y[good]
        name = name[good]
        name_list += list(name)

        diff = (y - x)
        if diff_fraction:
            diff = (y - x) / np.sqrt(x * y)

        ra = coord[:, 0][good]

        argsort = np.argsort(ra)

        ra = ra[argsort]
        diff = diff[argsort]

        l = ax.plot(ra, diff, 'o-', color=next(color), lw=2)[0]
        #l = ax.plot(bins_c, hist, '.-', color=next(color))[0]
        legend_list.append(mpatches.Patch(color=l.get_color(), label=label))

    #ax.set_xlim(bins.min(), bins.max())
    #ax.set_ylim(1.e-4, 0.35)
    #ax.set_ylim(ymin, ymax)
    #ax.loglog()
    #ax.semilogy()
    ncol = 2
    #if len(legend_list) > 5: ncol=2
    ax.legend(handles=legend_list, frameon=False, markerfirst=True, loc=2, ncol=ncol, mode='expand')
    #ax.set_aspect('equal')
    if axes is None:
        ax.set_xlabel('NVSS Flux at 1400 MHz (Jy)')
        ax.set_ylabel('Measured Flux at 1400 MHz (Jy)')

    print("="*40)
    print("Totally there are %4d (%4d) sources"%(len(name_list), len(set(name_list))))
    print("="*40)

def plot_flux_hist_allbeam(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):

    fig = plt.figure(figsize = (9, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, top=0.95, bottom=0.1, left=0.12, right=0.95,
                           wspace=0.01, hspace=0.01)

    ax =fig.add_subplot(gs[0, 0])
    beam_list = [1, ]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    #ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_ylabel(r'$N/N_{\rm total}$')

    ax =fig.add_subplot(gs[0, 1])
    beam_list = [2, 3, 4, 5, 6, 7]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    #ax.set_xticklabels([])
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_yticklabels([])

    ax =fig.add_subplot(gs[0, 2])
    beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    #ax.set_ylabel(r'$N/N_{\rm total}$')
    ax.set_yticklabels([])

    return fig

def plot_fluxfraction_hist_allbeam(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):

    #fig = plt.figure(figsize = (9, 6))
    #gs = gridspec.GridSpec(1, 3, figure=fig, top=0.95, bottom=0.1, left=0.12, right=0.95,
    #                       wspace=0.01, hspace=0.01)

    fig = plt.figure(figsize = (6, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, top=0.98, bottom=0.05, left=0.12, right=0.95,
                           wspace=0.03, hspace=0.03)

    ax =fig.add_subplot(gs[0, 0])
    beam_list = [1, ]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   bins=np.linspace(-0.99, 0.99,31),
                   ymin=ymin,ymax=ymax, tod=tod, diff_fraction=True)
    ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_xlabel(r'$ \delta S $')
    ax.set_ylabel(r'$N/N_{\rm total}$')

    ax =fig.add_subplot(gs[1, 0])
    beam_list = [2, 3, 4, 5, 6, 7]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   bins=np.linspace(-0.99, 0.99,31),
                   ymin=ymin,ymax=ymax, tod=tod, diff_fraction=True)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    ax.set_xticklabels([])
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    #ax.set_xlabel(r'$ \delta S $')
    #ax.set_yticklabels([])
    ax.set_ylabel(r'$N/N_{\rm total}$')

    ax =fig.add_subplot(gs[2, 0])
    beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   bins=np.linspace(-0.99, 0.99,31), 
                   ymin=ymin,ymax=ymax, tod=tod, diff_fraction=True)
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_xlabel(r'$ \delta S $')
    #ax.set_ylabel(r'$N/N_{\rm total}$')
    ax.set_ylabel(r'$N/N_{\rm total}$')
    #ax.set_yticklabels([])

    return fig

def plot_flux_chisq_allbeam(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):

    fig = plt.figure(figsize = (7, 6))
    gs = gridspec.GridSpec(3, 1, figure=fig, top=0.95, bottom=0.1, left=0.12, right=0.95,
                           wspace=0.02, hspace=0.02)

    ax =fig.add_subplot(gs[0, 0])
    beam_list = [1, ]
    plot_flux_chisq(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_ylabel(r'$N/N_{\rm total}$')

    ax =fig.add_subplot(gs[1, 0])
    beam_list = [2, 3, 4, 5, 6, 7]
    plot_flux_chisq(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    ax.set_xticklabels([])
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_ylabel(r'$N/N_{\rm total}$')
    #ax.set_yticklabels([])

    ax =fig.add_subplot(gs[2, 0])
    beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    plot_flux_chisq(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                        ymin=ymin,ymax=ymax, tod=tod)
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    #ax.set_ylabel(r'$N/N_{\rm total}$')
    ax.set_ylabel(r'$N/N_{\rm total}$')
    #ax.set_yticklabels([])

    return fig


#def plot_flux_hist_alldays(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):
#
#    fig = plt.figure(figsize = (9, 6))
#    gs = gridspec.GridSpec(1, 2, figure=fig, top=0.95, bottom=0.1, left=0.12, right=0.95,
#                           wspace=0.01, hspace=0.01)
#
#    ax =fig.add_subplot(gs[0, 0])
#    beam_list = list(range(1, 20))
#    plot_flux_hist(flux_diff_path[:4], beam_list, axes=(fig, ax), rms_sys=rms_sys,
#                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_days)
#    #ax.set_xticklabels([])
#    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
#    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
#    ax.set_ylabel(r'$N/N_{\rm total}$')
#
#    ax =fig.add_subplot(gs[0, 1])
#    beam_list = list(range(1, 20))
#    plot_flux_hist(flux_diff_path[4:], beam_list, axes=(fig, ax), rms_sys=rms_sys,
#                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_days)
#    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
#    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
#    #ax.set_xticklabels([])
#    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
#    ax.set_yticklabels([])
#
#    return fig

def plot_flux_hist_alldays(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_axes([0.12, 0.12, 0.83, 0.85])
    axes = (fig, ax)

    beam_list = list(range(1, 20))
    plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   bins=np.linspace(-0.99, 0.99,31),
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_days, diff_fraction=True)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    #ax.set_xticklabels([])
    ax.set_ylabel(r'$N/N_{\rm total}$')
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_xlabel(r'$ \delta S $')
    #ax.set_yticklabels([])

    return fig

def plot_flux_ra_alldays(flux_diff_path, tod=True, rms_sys=None, ymin=-0.5, ymax=0.5):

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_axes([0.12, 0.12, 0.83, 0.85])
    axes = (fig, ax)

    beam_list = list(range(1, 20))
    plot_flux_hist_ra(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_days, diff_fraction=True)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    #ax.set_xticklabels([])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(135, 195)
    ax.set_ylabel(r'$ \delta S $')
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_xlabel(r'$ R.A. [deg]$')
    #ax.set_yticklabels([])

    return fig


def plot_flux_ra_beams(flux_diff_path, tod=True, rms_sys=None, ymin=-0.5, ymax=0.5):

    fig = plt.figure(figsize = (6, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, top=0.98, bottom=0.05, left=0.12, right=0.95,
                           wspace=0.03, hspace=0.03)

    ax =fig.add_subplot(gs[0, 0])
    beam_list = [1, ]
    plot_flux_hist_ra(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_beam, diff_fraction=True)
    ax.set_xticklabels([])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(135, 195)
    ax.set_ylabel(r'$ \delta S $')

    ax =fig.add_subplot(gs[1, 0])
    beam_list = [2, 3, 4, 5, 6, 7]
    plot_flux_hist_ra(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_beam, diff_fraction=True)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    ax.set_xticklabels([])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(135, 195)
    ax.set_ylabel(r'$ \delta S $')

    ax =fig.add_subplot(gs[2, 0])
    beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    plot_flux_hist_ra(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_beam, diff_fraction=True)
    #ax.set_xticklabels([])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(135, 195)
    ax.set_ylabel(r'$ \delta S $')
    ax.set_xlabel(r'R.A. [deg]')

    return fig


def plot_flux_hist_day_vs_days(flux_diff_path, tod=True, rms_sys=None, ymin=1.e-4, ymax=0.60):
    fig = plt.figure(figsize = (6, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, top=0.95, bottom=0.1, left=0.12, right=0.95,
                           wspace=0.01, hspace=0.01)

    ax =fig.add_subplot(gs[0, 0])
    #beam_list = [1, ]
    beam_list = list(range(1, 20))
    plot_flux_hist(flux_diff_path[:2], beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_day_vs_days)
    #ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_ylabel(r'$N/N_{\rm total}$')

    ax =fig.add_subplot(gs[0, 1])
    beam_list = list(range(1, 20))
    plot_flux_hist(flux_diff_path[2:], beam_list, axes=(fig, ax), rms_sys=rms_sys,
                   ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_day_vs_days)
    ##ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    ##ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    ##ax.set_xticklabels([])
    ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ax.set_yticklabels([])

    #ax =fig.add_subplot(gs[0, 2])
    #beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    #plot_flux_hist(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys,
    #                    ymin=ymin,ymax=ymax, tod=tod, iter_func=iter_flux_day_vs_days)
    #ax.set_xlabel(r'$ S - S_{\rm NVSS} ~ [{\rm mJy}]$')
    ##ax.set_ylabel(r'$N/N_{\rm total}$')
    #ax.set_yticklabels([])

def plot_flux_beam(flux_path_list, beam_list=[1, ], axes=None, rms_sys=None, 
        f_min=None, f_max=None, tod=True):

    if axes is None:
        fig = plt.figure(figsize = (6, 6))
        ax = fig.subplots()
    else:
        fig, ax = axes

    sigma_list = []
    sigma_rel_list = []
    legend_list = []

    #color=iter(cm.tab20(np.linspace(0,1,len(beam_list))))
    color=iter(cm.tab20(np.linspace(0,1,12)))

    for label, x, y, e, name, coord in iter_flux_beam(beam_list, flux_path_list, tod=tod):

        good = ( x > 1.e-1 ) * ( y > 1.e-1 )
        x = x[good]
        y = y[good]
        e = e[good]

        l = ax.plot(x, y, 'o', ms=6, color=next(color))[0]
        #l = ax.errorbar(x, y, e, fmt='o', ms=4, color=next(color))[0]
        legend_list.append(mpatches.Patch(color=l.get_color(), label=label))

    if f_max is None: f_max = max(x.max(), y.max()) * 1.01
    if f_min is None: f_min = min(x.min(), y.min()) / 1.01
    ax.plot([f_min, f_max], [f_min, f_max], 'k-', linewidth=1.5)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(f_min, f_max)
    #ax.loglog()
    #ax.semilogy()
    ax.legend(handles=legend_list, frameon=False, markerfirst=True, loc=2, ncol=1)#, mode='expand')
    #ax.set_aspect('equal')
    if axes is None:
        ax.set_xlabel('NVSS Flux at 1400 MHz (mJy)')
        ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')

    #return fig, ax



def plot_flux_allbeam(flux_diff_path, tod=True, rms_sys=None, f_min=None, f_max=None):

    fig = plt.figure(figsize = (6, 12))
    #gs = gridspec.GridSpec(1, 3, figure=fig, top=0.95, bottom=0.1, left=0.05, right=0.98,
    #                       wspace=0.03, hspace=0.03)
    gs = gridspec.GridSpec(3, 1, figure=fig, top=0.98, bottom=0.05, left=0.12, right=0.95,
                           wspace=0.03, hspace=0.03)

    ax =fig.add_subplot(gs[0, 0])
    beam_list = [1, ]
    plot_flux_beam(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys, 
            f_min=f_min, f_max=f_max, tod=tod)
    ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_xlabel('NVSS Flux at 1400 MHz (mJy)')
    #ax.set_xlabel('NVSS Flux (mJy)')
    ax.set_ylabel('Measured Flux (mJy)')

    ax =fig.add_subplot(gs[1, 0])
    beam_list = [2, 3, 4, 5, 6, 7]
    plot_flux_beam(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys, 
            f_min=f_min, f_max=f_max, tod=tod)
    #ax.set_xlabel('NVSS Flux at 20cm (mJy)')
    #ax.set_ylabel('Measured Flux at 1400 MHz (mJy)')
    #ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux at 1400 MHz (mJy)')
    ax.set_xticklabels([])
    #ax.set_xlabel('NVSS Flux (mJy)')
    #ax.set_yticklabels([])
    ax.set_ylabel('Measured Flux (mJy)')

    ax =fig.add_subplot(gs[2, 0])
    beam_list = [8, 9, 10, 11, 12, 13] + [14, 15, 16, 17, 18, 19]
    plot_flux_beam(flux_diff_path, beam_list, axes=(fig, ax), rms_sys=rms_sys, 
            f_min=f_min, f_max=f_max, tod=tod)
    ax.set_xlabel('NVSS Flux (mJy)')
    #ax.set_ylabel(r'$N/N_{\rm total}$')
    ax.set_ylabel('Measured Flux (mJy)')
    #ax.set_yticklabels([])
    ax.set_xlabel('NVSS Flux (mJy)')

    return fig

# ------------------------------------------------------------------------------------------


def _get_spec(imap, nmap, pixs, ra, dec, nside, beam_sig, r1=None):

    noise_inv = np.ma.filled(nmap.copy(), 0)
    #noise_inv[noise_inv==0] = np.inf
    #noise_inv  = 1./noise_inv

    p_ra, p_dec = hp.pix2ang(nside, hp.ang2pix(nside, ra, dec, lonlat=True), lonlat=True)

    beam_fwhm = beam_sig * 2. * np.sqrt( 2. * np.log(2.) )
    pointing = hp.ang2vec(ra, dec, lonlat=True)

    # get pix index of inner circle
    if r1 is None:
        r1 = np.median( beam_fwhm / 2. )
    _p = hp.query_disc(nside, pointing, r1 * np.pi / 180., inclusive=True)

    # get pix index of outer circle
    r2 = (2**0.5) * r1 #* 2
    _p1 = hp.query_disc(nside, pointing, r1 * np.pi / 180., inclusive=True)
    _p2 = hp.query_disc(nside, pointing, r2 * np.pi / 180., inclusive=True)
    _p2 = np.array([x for x in _p2 if x not in _p1])
    if _p2.shape[0] == 0:
        print('10% enlarge outer circel to include pixels')
        r2 = (2**0.5) * r1 * 1.1
        _p2 = hp.query_disc(nside, pointing, r2 * np.pi / 180., inclusive=True)
        _p2 = np.array([x for x in _p2 if x not in _p])

    # get spec of inner circle
    _ra, _dec = hp.pix2ang(nside, _p, lonlat=True)
    dra = ((_ra - ra) * np.cos(dec*np.pi/180.)).astype('float64') # deg
    ddec = (_dec - dec).astype('float64') # deg
    rr = (dra**2 + ddec**2 ) ** 0.5
    _w = np.exp( - 0.5 * ( rr[None, :] / beam_sig[:, None] ) ** 2. )
    #_w = np.ones( (beam_sig.shape[0], _p.shape[0]))

    spec = []
    noise = [] # it is noise inv
    for i in range(_p.shape[0]):
        _idx = (pixs - _p[i]) == 0
        if np.any(_idx) :
            noise.append( noise_inv[:, _idx])
            spec.append( imap[:, _idx] * noise_inv[:, _idx] )
            _w[(spec[-1] == 0).flat, i] = 0.
            _w[:, i] *= noise_inv[:, _idx].flat
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
    #noise_m = np.ma.sum(noise, axis=1) / np.ma.sum(_w, axis=1)
    noise[noise==0] = np.ma.masked
    noise_m = np.ma.mean(noise, axis=1)
    del noise
    # convert to noise error
    noise_m = np.ma.filled(noise_m, 0)
    noise_m[noise_m==0] = np.inf
    noise_m = 1./noise_m
    noise_m = np.ma.sqrt(noise_m)

    # get spec of outer circle
    _w = np.ones( (beam_sig.shape[0], _p2.shape[0]))
    spec = []
    noise = [] # it is noise inv
    for i in range(_p2.shape[0]):
        _idx = (pixs - _p2[i]) == 0
        if np.any(_idx) :
            noise.append( noise_inv[:, _idx])
            spec.append( imap[:, _idx] * noise_inv[:, _idx] )
            _w[(spec[-1] == 0).flat, i] = 0.
            _w[:, i] *= noise_inv[:, _idx].flat
        else:
            _w[:, i] = 0.
        del _idx
    spec = np.concatenate(spec, axis=1)
    spec = np.ma.masked_invalid(spec)
    _w = np.ma.masked_invalid(_w)
    spec_m_out = np.ma.sum(spec, axis=1) / np.ma.sum(_w, axis=1)
    del spec

    mask = ( spec_m == 0 ) #+ (spec_m_out == 0)
    #spec_m = spec_m - spec_m_out
    spec_m[mask] = 0

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
    data = np.loadtxt('/home/ycli/code/fpipe/fpipe/data/fwhm.dat')
    f = data[4:, 0] #* 1.e-3
    d = data[4:, 1:]
    fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")(freq)
    fwhm /= 60. # deg
    beam_sig = (fwhm / (2. * np.sqrt(2.*np.log(2.))))

    #r1 = hp.nside2resol(nside, arcmin=True) / 60.
    #r1 = np.max( fwhm / 2. ) * 1.0
    r1 = 8./60./2.
    #print(r1)
    #r1 = np.max( fwhm ) * 1.0
    _r = _get_spec(imap, nmap, pixs, ra, dec, nside, beam_sig, r1=r1)
    if _r is None:
        return None
    p_ra, p_dec, spec_m, noise_m =  _r
    #_r = _get_spec(imap, nmap, pixs, ra, dec, nside, beam_sig, factor=2.**0.5)
    #if _r is None:
    #    return None
    #spec_m = 2 * spec_m - _r[2]

    if mJy:
        eta = 1.0
        _lambda = 2.99e8 / (freq * 1.e6)
        _Omega = np.pi * (fwhm * np.pi/180.)**2. / 4. / np.log(2.)
        _K2mJy = 2. * 1380. / (_lambda**2.) * 1.e3 * _Omega
        spec_m   = spec_m  * _K2mJy #* 2. #* 0.5
        noise_m  = noise_m * _K2mJy #* 2. #* 0.5

    spec_m = np.ma.masked_equal(spec_m, 0)
    noise_m = np.ma.masked_equal(noise_m, 0)
    #print np.sum(spec_m.mask),
    for i in range(5):
        flux = np.ma.median(spec_m, axis=0)
        dflux = np.ma.std(spec_m - flux, axis=0)

        mask = np.ma.abs( spec_m - flux ) - 2. * dflux > 0
        if not np.any(mask): break
        #print np.sum(mask.astype('int'))
        spec_m.mask[mask] = True
        noise_m.mask[mask] = True
    #print np.sum(spec_m.mask)

    return spec_m, p_ra, p_dec, noise_m








