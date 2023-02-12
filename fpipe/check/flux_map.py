
from fpipe.timestream.data_base import DATA_BASE as DB
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import astropy.io.fits as pyfits
from reproject import reproject_from_healpix, reproject_to_healpix
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import RectangularFrame, EllipticalFrame

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter as mfilter
from scipy.ndimage import gaussian_filter as gfilter

import numpy as np
import healpy as hp
import h5py
from fpipe.check import flux_tod as cf
from fpipe.plot import plot_map

import gc


from tqdm.autonotebook import tqdm

def flag_spec(spec_m, iteration=5):
    
    spec_m = np.ma.masked_invalid(spec_m)

    spec_m[spec_m==0] = np.ma.masked
    
    for i in range(iteration):
        
        flux = np.ma.median(spec_m, axis=0)
        dflux = np.ma.std(spec_m - flux, axis=0)

        mask = np.ma.abs( spec_m - flux ) - 2. * dflux > 0
        if not np.any(mask): break
        #print np.sum(mask.astype('int'))
        spec_m.mask[mask] = True
        
    return spec_m

def read_flux_pix(imap, pix_id, freq, flag_iteration):

    spec = flag_spec(imap[:, pix_id], flag_iteration)
    flux = np.ma.median((spec/cf.mJy2K(freq*1.e-3))[~spec.mask])
    flux = flux * 2 # 2 pols
    
    return flux

def read_flux_aperture(imap, pix, nside, freq, pointing, aperture_radius, flag_iteration):
    
    ra, dec = pointing
    
    kernel_fwhm = 3./60. * 1420. / freq * 2
    #kernel_fwhm = 3./60. * np.ones_like(freq)
    kernel_sig = kernel_fwhm  / (2. * np.sqrt(2.*np.log(2.)))
    
    _ar = aperture_radius / 60. * np.pi / 180. # convert to radians
    _pt = hp.ang2vec(ra, dec, lonlat=True)
    _pix = hp.query_disc(nside, _pt, _ar, inclusive=True)
    
    if len(_pix) == 0: return None
    
    _flx_from_map = []
    _kernel_list  = []
    for _p in _pix:
        
        pix_id = np.where(pix == _p)[0]
        if len(pix_id) == 0: continue

        _ra, _dec = hp.pix2ang(nside, _p, lonlat=True)
        P = (np.sin(dec*np.pi/180.) * np.sin(_dec*np.pi/180.)) \
          + (np.cos(dec*np.pi/180.) * np.cos(_dec*np.pi/180.)) \
          * (np.cos(ra*np.pi/180. - _ra*np.pi/180.))
        P = np.arccos(P) * 180. / np.pi
        P = np.exp(-0.5 * (P / kernel_sig) ** 2)
        
        P[P<0.01] = 0
        #P /= 2.
        #print(P)

        spec = flag_spec(imap[:, pix_id[0]], flag_iteration)/cf.mJy2K(freq*1.e-3)
        spec *= 2. # 2 pols
        P[spec==0] = 0
        spec.mask[P==0] = np.ma.masked

        _flx_from_map.append((spec)[None, :])
        _kernel_list.append(P[None, :])
        
    if len(_flx_from_map) == 0: return None
    
    _flx_from_map = np.ma.concatenate(_flx_from_map, 0)
    _kernel_list  = np.ma.concatenate(_kernel_list, 0)
    
    r = np.ma.sum(_flx_from_map, 0) / np.ma.sum(_kernel_list, 0)
    r = np.ma.masked_invalid(r)

    e = np.ma.sum((_flx_from_map - r[None, :] * _kernel_list)**2, 0) / np.ma.sum(_kernel_list**2, 0)
    e = ( e * np.ma.sum(_kernel_list**2, 0) / np.ma.sum(_kernel_list, 0) ** 2 )  ** 0.5
    e = np.ma.masked_invalid(e)
    

    r[r==0] = np.ma.masked
    e[r==0] = np.ma.masked
    if np.all(r.mask): return None, None
    
    #print(np.median(r[~r.mask]))
    #print('--'*10)
    
    return np.median(r[~r.mask]), np.median(e[~r.mask])
    

def read_flux(map_info, sim_info, nvss_cat_list, flux_key='NVSS_FLUX', name_key='NVSS_ID',
              flux_lim = 10, iso_threshold = 0, max_major_axis = 100, 
              flag_iteration=0, freq_min = 1400 - 25, freq_max = 1400 + 25,
              aperture_radius=0, nvss_range = None):
    
    '''
    
    aperture_radius: in unit of arcmin
    
    '''
    
    imap_raw, pix, nside = map_info
    freq = imap_raw.get_axis('freq')
    
    imap_sim, pix_sim, nside_sim = sim_info
    freq_sim = imap_sim.get_axis('freq')

    freq_sel = ( freq > freq_min ) * ( freq < freq_max )
    imap = imap_raw[freq_sel]
    freq = freq[freq_sel]
    
    imap_ra, imap_dec = hp.pix2ang(nside, pix, lonlat=True)

    if nvss_range is None:
        nvss_range = [[imap_ra.min(), imap_ra.max(), imap_dec.min(), imap_dec.max()],]
    nvss_ra, nvss_dec, nvss_flx, nvss_name =\
        cf.load_catalogue(nvss_cat_list, nvss_range, flux_key, name_key, 
                          flux_lim, iso_threshold, max_major_axis)
    nvss_numb = nvss_ra.shape[0]

    results = []
    for ii in tqdm(range(nvss_numb), colour='green'):
        
        _ra  = nvss_ra[ii]
        _dec = nvss_dec[ii]
        _flx = nvss_flx[ii]
        
        if aperture_radius == 0:
            _pix     = np.where(pix == hp.ang2pix(nside, _ra, _dec, lonlat=True))[0]
            _pix_sim = np.where(pix_sim == hp.ang2pix(nside_sim, _ra, _dec, lonlat=True))[0]
            if len(_pix) != 0 and len(_pix_sim != 0):
                _flx_from_map = read_flux_pix(imap, _pix[0], freq, flag_iteration)
                _flx_from_sim = read_flux_pix(imap_sim, _pix_sim[0], freq_sim, 0)
                if _flx_from_map > 0 and _flx_from_sim > 0:
                    results.append([_ra, _dec, _flx, _flx_from_map, _flx_from_sim])
        else:
            pointing = [_ra, _dec]
            _flx_from_map, _err_from_map = read_flux_aperture(imap, pix, nside, freq, pointing, 
                                               aperture_radius, flag_iteration)
            _flx_from_sim, _err_from_sim = read_flux_aperture(imap_sim, pix_sim, nside_sim, 
                                               freq_sim, pointing, aperture_radius, 0)
            if not ((_flx_from_map is None) or (_flx_from_sim is None)):
                results.append([_ra, _dec, _flx, _flx_from_map, _flx_from_sim, _err_from_map])
    results = np.array(results)
    print('%d sources used'%results.shape[0])
    print('-'*30)
    return results

def plot_fluxflux(results, label='', axes=None, fmt='r.', do_corr=True, with_err=False, **kwargs):

    if do_corr:
        corr = results[:, 2] - results[:, 4] 
    else:
        corr = 0.

    imap_x = results[:, 2]
    imap_y = results[:, 3] + corr

    if with_err:
        print(results.shape)
        imap_y_err = results[:, 5]
    
    
    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    else:
        fig, ax = axes
        
    if with_err:
        ax.errorbar(imap_x, imap_y, imap_y_err, fmt=fmt, label=label, **kwargs)
    else:
        ax.plot(imap_x, imap_y, fmt, label=label, **kwargs)
    
    xmin = int( imap_x.min() )
    xmax = int( imap_x.max() ) + 10

    xx = np.linspace(xmin, xmax, 100)

    ax.plot(xx, xx, 'k-')
    #ax.fill_between(xx, xx+70, xx-70, color='k', alpha=0.2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    #ax.set_aspect('equal')
    
    return fig, ax

def plot_fluxdiffhist(results, label='', axes=None, fmt='r-', do_corr=True, **kwargs):

    if do_corr:
        corr = results[:, 2] - results[:, 4] 
    else:
        corr = 0

    imap_x = results[:, 2]
    imap_y = results[:, 3] + corr
    
    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    else:
        fig, ax = axes
        
    x = imap_x
    y = imap_y
    diff = ( y - x ) / x #np.sqrt( np.abs( x * y ) )
    #diff = x/y - 1
    
    sig = np.sqrt(np.median(diff**2))
    print(sig)
    
    bins_e = np.linspace(-1, 1, 31)
    hist   = np.histogram(diff, bins=bins_e)[0]
    hist   = hist / np.sum(hist)
    
    bins_c = ( bins_e[1:] + bins_e[:-1] ) * 0.5
    
    ax.plot(bins_c, hist, fmt, label=label, drawstyle='steps-mid', **kwargs) #)


def iter_tod_sources(beam_list, source_path_list):
    
    name_list = []
    for i in beam_list:
        for flux_path in source_path_list:
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
                    
                    if s in name_list: continue
                    else: name_list.append(s)

                    rr = f['B%02d/%s/FLUXRADEC'%(i, s)][:]
                    yield "Feed %02d"%i, s, rr[0], rr[1], rr[2]


def read_flux_tod_sources(map_info, sim_info, source_path_list,  
              flag_iteration=0, freq_min = 1400 - 25, freq_max = 1400 + 25,
              aperture_radius=0):
    
    '''
    
    aperture_radius: in unit of arcmin
    
    '''
    
    imap_raw, pix, nside = map_info
    freq = imap_raw.get_axis('freq')
    
    imap_sim, pix_sim, nside_sim = sim_info
    freq_sim = imap_sim.get_axis('freq')

    freq_sel = ( freq > freq_min ) * ( freq < freq_max )
    imap = imap_raw[freq_sel]
    freq = freq[freq_sel]
    
    imap_ra, imap_dec = hp.pix2ang(nside, pix, lonlat=True)

    #if nvss_range is None:
    #    nvss_range = [[imap_ra.min(), imap_ra.max(), imap_dec.min(), imap_dec.max()],]
    #nvss_ra, nvss_dec, nvss_flx, nvss_name =\
    #    cf.load_catalogue(nvss_cat_list, nvss_range, flux_key, name_key, 
    #                      flux_lim, iso_threshold, max_major_axis)
    #nvss_numb = nvss_ra.shape[0]
    
    #for ii in tqdm(range(nvss_numb), colour='green'):
    results = []
    beam_list = np.arange(1, 20)

    for rr in iter_tod_sources(beam_list, source_path_list):
        
        label, name, _flx, _ra, _dec = rr
        #print(label, name, _flx)

        if aperture_radius == 0:
            _pix     = np.where(pix == hp.ang2pix(nside, _ra, _dec, lonlat=True))[0]
            _pix_sim = np.where(pix_sim == hp.ang2pix(nside_sim, _ra, _dec, lonlat=True))[0]
            if len(_pix) != 0 and len(_pix_sim != 0):
                _flx_from_map = read_flux_pix(imap, _pix[0], freq, flag_iteration)
                _flx_from_sim = read_flux_pix(imap_sim, _pix_sim[0], freq_sim, 0)
                if _flx_from_map > 0 and _flx_from_sim > 0:
                    results.append([_ra, _dec, _flx, _flx_from_map, _flx_from_sim])
        else:
            pointing = [_ra, _dec]
            _flx_from_map, _err_from_map = read_flux_aperture(imap, pix, nside, freq, pointing, 
                                               aperture_radius, flag_iteration)
            _flx_from_sim, _err_from_sim = read_flux_aperture(imap_sim, pix_sim, nside_sim, 
                                               freq_sim, pointing, aperture_radius, 0)
            if not ((_flx_from_map is None) or (_flx_from_sim is None)):
                results.append([_ra, _dec, _flx, _flx_from_map, _flx_from_sim, _err_from_map])
    results = np.array(results)
    print('%d sources used'%results.shape[0])
    print('-'*30)
    return results

def plot_tod(results, tod_path, theshold=0.5, check_cat=False, output_prefix=''):

    for i in tqdm(range(results.shape[0]), colour='green'):
    
        ra   = results[i, 0]
        dec  = results[i, 1]
        
        flx_cat = results[i, 2]
        flx_map = results[i, 3]
        flx_sim = results[i, 4]

        title = 'Map: %5.2f, Sim: %5.2f, Cat: %5.2f'%(flx_map, flx_sim, flx_cat)
        
        if check_cat:
            diff = ( flx_cat - flx_sim ) / flx_sim
            output_name = "./plot/image/difftod_SimCat_%sS%04d_%.6f.png"%(output_prefix, i, diff)
            if np.abs(diff) > theshold:
                plot_tod_partial(ra, dec, tod_path, title)
            
        diff = ( flx_map - flx_sim ) / flx_sim
        output_name = "./plot/image/difftod_SimMap_%sS%04d_%.6f.png"%(output_prefix, i, diff)
        if np.abs(diff) > theshold:
            plot_tod_partial(ra, dec, tod_path, title)
            
def plot_tod_partial(ra_s, dec_s, tod_path, title='', output_name=None):
    DATA_list = ['20210302', '20210305', '20210306', '20210307', 
                 '20210309', '20210313', '20210314' ]
    suffix     = '_1250-1450MHz'
    file_temp = '%s/%s_arcdrift%04d-%04d%s.h5'
    
    x_min = ra_s - 0.2
    x_max = ra_s + 0.2
    y_min = dec_s - 3/60.
    y_max = dec_s + 3/60.

    fig = plt.figure(figsize=(10, 5))
    ax0  = fig.add_axes([0.12, 0.51, 0.83, 0.38])
    ax1  = fig.add_axes([0.12, 0.12, 0.83, 0.38])

    for DATA_Key in DATA_list:
        
        DATA     = DB[DATA_Key]['DATA'    ]
        DATE     = DB[DATA_Key]['DATE'    ]
        bad_feed = DB[DATA_Key]['BAD_FEED']
        
        for file_st in range(1, 8):        
            file_ed = file_st
            tod_file = file_temp%(DATE, DATA, file_st, file_ed, suffix)
            with h5py.File(tod_path + tod_file, 'r') as fp:
                vis = fp['vis'][:]
                msk = fp['vis_mask'][:].astype('bool')
                ra  = fp['ra'][:]
                dec = fp['dec'][:]
                ns_on = fp['ns_on'][:]
                vis_sim = np.ma.array(vis, mask=msk)
                vis_sim.mask += ns_on[:, None, None, :]
            for bi in range(vis.shape[3]):
                if bi + 1 in bad_feed: continue
                if bi == 0: color = 'r'
                elif bi < 7: color = 'g'
                else: color = 'b'
                sel =  ( ra[:, bi]<x_max) * ( ra[:, bi]>x_min)
                sel *= (dec[:, bi]<y_max) * (dec[:, bi]>y_min)
                if np.any(sel):
                    x = ra[sel, bi]
                    #y = np.sum(vis_sim[sel, 0, :, bi], axis=1) / cf.mJy2K(1.42)
                    y = vis_sim[sel, 0, 0, bi] /  cf.mJy2K(1.40)
                    ax0.plot(x, y, '.', color=color)
                    y = vis_sim[sel, 0, 1, bi] /  cf.mJy2K(1.40)
                    ax1.plot(x, y, '.', color=color)
    ax0.set_title(title)
    ax0.set_xlim(xmin=x_max, xmax=x_min)
    ax1.set_xlim(xmin=x_max, xmax=x_min)
    
    if output_name is not None:
        fig.savefig(output_name, dpi=200)
    plt.show()


def plot_coord_partial(imap_real, imap_sim, ra_s, dec_s, tod_path, imap_shp = (100, 50), 
                       pix=3./60., proj='ZEA', figsize=(12, 8), title='', output_name=None,
                       nvss_path = None):
    
    DATA_list = ['20210302', '20210305', '20210306', '20210307', 
                 '20210309', '20210313', '20210314' ]
    suffix     = '_1250-1450MHz'
    file_temp = '%s/%s_arcdrift%04d-%04d%s.h5'

    dra  = imap_shp[0] * 0.5 * pix
    ddec = imap_shp[1] * 0.5 * pix
    
    x_min = ra_s - dra * 0.5
    x_max = ra_s + dra * 0.5
    y_min = dec_s - ddec * 0.5
    y_max = dec_s + ddec * 0.5

    if nvss_path is not None:

        nvss_range = [[x_min, x_max, y_min, y_max], ]

        flux_key='NVSS_FLUX'
        name_key='NVSS_ID'
        flux_lim = 2
        iso_threshold = 0
        max_major_axis = 1000
        
        nvss_ra, nvss_dec, nvss_flx, nvss_name =\
        cf.load_catalogue(nvss_path, nvss_range, flux_key, name_key, 
                          flux_lim, iso_threshold, max_major_axis)
        #print(nvss_ra.min(), nvss_ra.max(), nvss_dec.min(), nvss_dec.max())

    field_center = (ra_s, dec_s)
    #print(field_center)
    #print(nvss_range)
    target_header, projection = plot_map.get_projection(proj, imap_shp, field_center, pix)
    
    fig = plt.figure(figsize=figsize)
    #cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    ax = fig.add_axes([0.1, 0.52, 0.78, 0.4], projection=projection, 
                      frame_class=RectangularFrame)

    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='black', linestyle='--', lw=0.5)

    ax.set_title(title)

    if nvss_path is not None:
        ax.plot(nvss_ra, nvss_dec, 'o', mec='0.5', mfc='none', ms=6, mew=2,
                transform=ax.get_transform('icrs'))
    
    ax.plot(ra_s, dec_s, 'o', mec='r', mfc='none', ms=6, mew=2,
            transform=ax.get_transform('icrs'))

    for DATA_Key in DATA_list:
        
        DATA     = DB[DATA_Key]['DATA'    ]
        DATE     = DB[DATA_Key]['DATE'    ]
        bad_feed = DB[DATA_Key]['BAD_FEED']
        
        for file_st in range(1, 8):        
            file_ed = file_st
            tod_file = file_temp%(DATE, DATA, file_st, file_ed, suffix)
            with h5py.File(tod_path + tod_file, 'r') as fp:
                vis = fp['vis'][:]
                msk = fp['vis_mask'][:].astype('bool')
                ns_on = fp['ns_on'][:]
                vis_sim = np.ma.array(vis, mask=msk)
                vis_sim.mask += ns_on[:, None, None, :]
                
                ra  = np.ma.array(fp['ra'][:],  mask=vis_sim.mask[:, 0, 0, :])
                dec = np.ma.array(fp['dec'][:], mask=vis_sim.mask[:, 0, 0, :])

            for bi in range(vis.shape[3]):
                if bi + 1 in bad_feed: continue
                if bi == 0: color = 'r'
                elif bi < 7: color = 'g'
                else: color = 'b'
                sel =  ( ra[:, bi]<x_max) * ( ra[:, bi]>x_min)
                sel *= (dec[:, bi]<y_max) * (dec[:, bi]>y_min)
                if np.any(sel):
                    x = ra[sel,  bi]
                    y = dec[sel, bi]
                    ax.plot(x, y, '.-', color=color, lw=1, transform=ax.get_transform('icrs'))


def plot_map_partial(imap_real, imap_sim, ra, dec, imap_shp = (100, 50), 
                     pix=3./60., proj='ZEA', figsize=(8, 6), cmap='viridis',
                     sigma=6, vmin=0, vmax=2, title='', output_name=None,
                     nvss_path = None):
    
    if nvss_path is not None:
        dra  = imap_shp[0] * 0.5 * pix
        ddec = imap_shp[1] * 0.5 * pix
        nvss_range = [[ra - dra, ra + dra, dec - ddec, dec + ddec], ]
        
        flux_key='NVSS_FLUX'
        name_key='NVSS_ID'
        flux_lim = 2
        iso_threshold = 0
        max_major_axis = 1000
        
        nvss_ra, nvss_dec, nvss_flx, nvss_name =\
        cf.load_catalogue(nvss_path, nvss_range, flux_key, name_key, 
                          flux_lim, iso_threshold, max_major_axis)
        #print(nvss_ra.min(), nvss_ra.max(), nvss_dec.min(), nvss_dec.max())
        
    
    field_center = (ra, dec)
    #print(field_center)
    #print(nvss_range)
    target_header, projection = plot_map.get_projection(proj, imap_shp, field_center, pix)
    
    fig = plt.figure(figsize=figsize)
    cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    
    # --------------------- sim ------------------------------------------------
    imap_full, pixs, nside = imap_real
    #imap = np.ma.median( np.ma.masked_equal(imap, 0), axis=0)
    
    #imap_full = np.zeros(hp.nside2npix(nside))
    #imap_full[pixs] = imap
    #imap_full = np.ma.masked_equal(imap_full, 0)
    

    array, footprint = reproject_from_healpix((imap_full, 'icrs'), target_header,
                                              nested=False, order='nearest-neighbor')
    
    if sigma is not None:
        mean = np.ma.mean(array)
        std  = np.ma.std(array)
        vmin = mean - sigma * std
        vmax = mean + sigma * std
    else:
        vmin = np.ma.min(array)
        vmax = np.ma.max(array)

    
    ax = fig.add_axes([0.1, 0.52, 0.78, 0.4], projection=projection, 
                      frame_class=RectangularFrame)
    im = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='black', linestyle='--', lw=0.5)

    ax.set_title(title)
    
    if nvss_path is not None:
        ax.plot(nvss_ra, nvss_dec, 'o', mec='0.5', mfc='none', ms=6, mew=2,
                transform=ax.get_transform('icrs'))
    
    ax.plot(ra, dec, 'o', mec='r', mfc='none', ms=6, mew=2,
            transform=ax.get_transform('icrs'))

    ticks = list(np.linspace(vmin, vmax, 5))
    ticks_label = []
    for x in ticks:
        ticks_label.append(r"$%5.2f$"%x)
    
    fig.colorbar(im, cax=cax, ticks=ticks)
    cax.set_yticklabels(ticks_label, rotation=90, va='center')
    cax.minorticks_off()
    c_label = r'$T\,$ K'
    cax.set_ylabel(c_label)
    
    del imap_full
    gc.collect()
    
    
    # --------------------- sim ------------------------------------------------
    imap_full, pixs, nside = imap_sim
    #imap = np.ma.median( np.ma.masked_equal(imap, 0), axis=0)
    
    #imap_full = np.zeros(hp.nside2npix(nside))
    #imap_full[pixs] = imap
    #imap_full = np.ma.masked_equal(imap_full, 0)
    array, footprint = reproject_from_healpix((imap_full, 'icrs'), target_header,
                                              nested=False, order='nearest-neighbor')
    
    ax = fig.add_axes([0.1, 0.10, 0.78, 0.4], projection=projection, 
                      frame_class=RectangularFrame)
    im = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if nvss_path is not None:
        ax.plot(nvss_ra, nvss_dec, 'o', mec='0.5', mfc='none', ms=6, mew=2,
            transform=ax.get_transform('icrs'))
    
    ax.plot(ra, dec, 'o', mec='r', mfc='none', ms=6, mew=2,
            transform=ax.get_transform('icrs'))

    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='black', linestyle='--', lw=0.5)
    
    del imap_full
    gc.collect()
    
    if output_name is not None:
        fig.savefig(output_name, dpi=200)
    
    plt.show()
    plt.clf()
    
    #%reset -f array
    
def plot_source_map(results, map_info, sim_info, output_prefix='', theshold=0.5, 
                    nvss_path=None, check_cat=False):
    
    for i in tqdm(range(results.shape[0]), colour='green'):
    
        ra   = results[i, 0]
        dec  = results[i, 1]
        
        flx_cat = results[i, 2]
        flx_map = results[i, 3]
        flx_sim = results[i, 4]
        
        #diff = ( flx_map - flx_sim ) / np.sqrt(np.abs(flx_map * flx_sim))
        
        if check_cat:
            diff = ( flx_cat - flx_sim ) / flx_sim
            
            if np.abs(diff) > theshold:
                title = 'Map: %5.2f, Sim: %5.2f, Cat: %5.2f'%(flx_map, flx_sim, flx_cat)
                output_name = "./plot/image/diffmap_SimCat_%sS%04d_%.6f.png"%(output_prefix, i, diff)
                plot_map_partial(map_info, sim_info, ra, dec, sigma=None, 
                                 imap_shp = (200, 60), pix=0.3/60., title=title,
                                 output_name = output_name, nvss_path=nvss_path)
            
        diff = ( flx_map - flx_sim ) / flx_sim
        if np.abs(diff) > theshold:
            title = 'Map: %5.2f, Sim: %5.2f, Cat: %5.2f'%(flx_map, flx_sim, flx_cat)
            output_name = "./plot/image/diffmap_SimMap_%sS%04d_%.6f.png"%(output_prefix, i, diff)
            plot_map_partial(map_info, sim_info, ra, dec, sigma=None, 
                             imap_shp = (200, 60), pix=0.3/60., title=title,
                             output_name = output_name, nvss_path=nvss_path)
        
    
    field_center = (ra_s, dec_s)
    #print(field_center)
    #print(nvss_range)
    target_header, projection = plot_map.get_projection(proj, imap_shp, field_center, pix)
    
    fig = plt.figure(figsize=figsize)
    #cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    
    # --------------------- sim ------------------------------------------------
    imap_full, pixs, nside = imap_real

    #array, footprint = reproject_from_healpix((imap_full, 'icrs'), target_header,
    #                                          nested=False, order='nearest-neighbor')

    ax = fig.add_axes([0.1, 0.10, 0.78, 0.82], projection=projection, 
                      frame_class=RectangularFrame)
    #im = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='black', linestyle='--', lw=0.5)

    ax.set_title(title)
    
    for DATA_Key in DATA_list:
        
        DATA     = DB[DATA_Key]['DATA'    ]
        DATE     = DB[DATA_Key]['DATE'    ]
        bad_feed = DB[DATA_Key]['BAD_FEED']
        
        for file_st in range(1, 8):        
            file_ed = file_st
            tod_file = file_temp%(DATE, DATA, file_st, file_ed, suffix)
            with h5py.File(tod_path + tod_file, 'r') as fp:
                vis = fp['vis'][:]
                msk = fp['vis_mask'][:].astype('bool')
                ra  = fp['ra'][:]
                dec = fp['dec'][:]
                ns_on = fp['ns_on'][:]
                vis_sim = np.ma.array(vis, mask=msk)
                vis_sim.mask += ns_on[:, None, None, :]
            for bi in range(vis.shape[3]):
                if bi + 1 in bad_feed: continue
                if bi == 0: color = 'r'
                elif bi < 7: color = 'g'
                else: color = 'b'
                sel =  ( ra[:, bi]<x_max) * ( ra[:, bi]>x_min)
                sel *= (dec[:, bi]<y_max) * (dec[:, bi]>y_min)
                sel *= ~vis_sim.mask[:, 0, 0, bi]
                if np.any(sel):
                    x = ra[sel, bi]
                    y = dec[sel, bi]
                    ax.plot(x, y, '.-', color=color, lw=0.5, ms=1.5, 
                            transform=ax.get_transform('icrs'))
                    
    
    if nvss_path is not None:
        ax.plot(nvss_ra, nvss_dec, 'o', mec='0.5', mfc='none', ms=6, mew=2,
                transform=ax.get_transform('icrs'))
    
    ax.plot(ra_s, dec_s, 'o', mec='r', mfc='none', ms=6, mew=2,
            transform=ax.get_transform('icrs'))
    
    del imap_full
    gc.collect()

    if output_name is not None:
        fig.savefig(output_name, dpi=200)
    
    plt.show()
    plt.clf()
    
    #%reset -f array


def plot_all(results, map_info, sim_info, tod_real, tod_sim, output_prefix='', theshold=0.5, 
             nvss_path=None):
    
    for i in tqdm(range(results.shape[0]), colour='green'):
    
        ra   = results[i, 0]
        dec  = results[i, 1]
        
        flx_cat = results[i, 2]
        flx_map = results[i, 3]
        flx_sim = results[i, 4]

        diff = ( flx_map - flx_sim ) / flx_sim
        if np.abs(diff) > theshold:
            title = 'Map: %5.2f, Sim: %5.2f, Cat: %5.2f'%(flx_map, flx_sim, flx_cat)
            output_name = "./plot/image/S%04d_diffmap_%s%.6f.png"%(i, output_prefix, diff)
            plot_map_partial(map_info, sim_info, ra, dec, sigma=None, 
                             imap_shp = (200, 60), pix=0.3/60., title=title,
                             output_name = output_name, nvss_path=nvss_path)
            
            output_name = "./plot/image/S%04d_difftod_%s%.6f.png"%(i, output_prefix, diff)
            plot_tod_partial(ra, dec, tod_real, title, output_name=output_name)
            
            output_name = "./plot/image/S%04d_diffsim_%s%.6f.png"%(i, output_prefix, diff)
            plot_tod_partial(ra, dec, tod_sim, title, output_name=output_name)
            
            output_name = "./plot/image/S%04d_coodtod_%s%.6f.png"%(i, output_prefix, diff)
            plot_coord_partial(map_info, sim_info, ra, dec, tod_real, imap_shp = (200, 60), 
                       pix=0.3/60., title=title, output_name=output_name, nvss_path = nvss_path)


def plot_sim_gt_cat(results, map_info, sim_info, tod_real, tod_sim, output_prefix='', theshold=0.5, 
             nvss_path=None):
    
    for i in tqdm(range(results.shape[0]), colour='green'):
    
        ra   = results[i, 0]
        dec  = results[i, 1]
        
        flx_cat = results[i, 2]
        flx_map = results[i, 3]
        flx_sim = results[i, 4]

        diff = ( flx_sim - flx_cat ) / flx_cat
        if diff > theshold:
            title = 'Map: %5.2f, Sim: %5.2f, Cat: %5.2f'%(flx_map, flx_sim, flx_cat)
            output_name = "./plot/image/S%04d_diffmap_%s%.6f.png"%(i, output_prefix, diff)
            plot_map_partial(map_info, sim_info, ra, dec, sigma=None, 
                             imap_shp = (200, 60), pix=0.3/60., title=title,
                             output_name = output_name, nvss_path=nvss_path)
            
            output_name = "./plot/image/S%04d_difftod_%s%.6f.png"%(i, output_prefix, diff)
            plot_tod_partial(ra, dec, tod_real, title, output_name=output_name)
            
            output_name = "./plot/image/S%04d_diffsim_%s%.6f.png"%(i, output_prefix, diff)
            plot_tod_partial(ra, dec, tod_sim, title, output_name=output_name)
            
            output_name = "./plot/image/S%04d_coodtod_%s%.6f.png"%(i, output_prefix, diff)
            plot_coord_partial(map_info, sim_info, ra, dec, tod_real, imap_shp = (200, 60), 
                       pix=0.3/60., title=title, output_name=output_name, nvss_path = nvss_path)


def plot_dNdS_multi(results_list, label_list, output=None):

    c_list = ['r', 'k', 'g', 'b']

    fig = plt.figure(figsize=(8, 4))
    #ax  = fig.add_axes([0.11, 0.13, 0.86, 0.85])
    ax  = fig.add_axes([0.1, 0.12, 0.85, 0.82])

    axes = (fig, ax)

    handle_list = []
    for ii, result in enumerate( results_list ):
        plot_dNdS(result, axes=axes, color=c_list[ii])

        handle_list.append(mpatches.Patch(color=c_list[ii], label=label_list[ii]))

    #cfm.plot_dNdS(results_todsource_ap1p5_iso0_flx0, axes=axes, color='r')
    #cfm.plot_dNdS(results_todsource_ap1p5_iso9_flx7, axes=axes, color='0.5')
    #cfm.plot_dNdS(results_todsource_ap1p5_iso9_flx14, axes=axes, color='k')

    ax.axvline(7, 0, 1, color='0.5', ls='-.', lw=3, zorder=-100)
    ax.axvline(14, 0, 1, color='0.5', ls=':', lw=3, zorder=-100)

    nvss_label = mlines.Line2D([], [], color='k', ls='--', label='NVSS', lw=2.5)
    fast_label = mlines.Line2D([], [], color='k', ls='-', marker='o', mfc='w', label='FAST', lw=1.5)
    leg1 = ax.legend(handles=[nvss_label, fast_label], loc='center right')

    leg2 = ax.legend(handles=handle_list, loc='lower right')

    ax.add_artist(leg1)

    if output is not None:
        fig.savefig(output, dpi=200)

def plot_dNdS(results, axes=None, flx_min=2., flx_max=800., color='r'):
    
    if axes is None:
        fig = plt.figure(figsize=(8, 4))
        ax  = fig.subplots()
    else:
        fig, ax = axes

    S_area = 60. * (np.pi / 180.)**2.
    
    flx_bin_e = np.logspace(np.log10(flx_min), np.log10(flx_max), 16)
    dflx_bin = flx_bin_e[1:]/flx_bin_e[:-1]
    flx_bin_c = flx_bin_e[:-1] * ( dflx_bin ** 0.5 )
    dflx = (flx_bin_e[1:] - flx_bin_e[:-1]) * 1.e-3

    dSdN = np.histogram(results[:, 2], bins=flx_bin_e)[0] / dflx / S_area
    corr = results[:, 2] - results[:, 4]
    fast_flx = results[:, 3] + corr
    dSdN_fast = np.histogram(fast_flx, bins=flx_bin_e)[0]/ dflx / S_area
    
    ax.plot(flx_bin_c, dSdN * ((flx_bin_c * 1.e-3) ** 2.5), color=color, ls='--', lw=2.5, drawstyle='steps-mid')
    ax.plot(flx_bin_c, dSdN_fast * ((flx_bin_c * 1.e-3) ** 2.5), 'o-', color=color, mfc='w', mew=1.5, lw=2.)
    
    ax.loglog()
    ax.set_ylabel(r'$S^{5/2}{\rm d}N/{\rm d}S\,[{\rm Jy}^{3/2}{\rm sr}^{-1}] $')
    ax.set_xlabel(r'$S\,[{\rm mJy}]$')
    
    ax.set_xlim(flx_min, flx_max)
    ax.set_ylim(0.1, 1000)

    return fig, ax
    
