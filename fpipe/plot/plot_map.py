#from collections import OrderedDict as dict

from fpipe.map import algebra as al
from fpipe.ps import fgrm
from fpipe.point_source import source
from fpipe.check import tsys

import logging

from astropy.io import fits
from reproject import reproject_from_healpix, reproject_to_healpix
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import RectangularFrame, EllipticalFrame

import h5py as h5
import numpy as np
import scipy as sp
import copy

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.ndimage.filters import gaussian_filter as gf

from astropy.coordinates import SkyCoord
import astropy.units as u

import healpy as hp

logger = logging.getLogger(__name__)

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

def plot_map_hp(map_name, map_key='clean_map', indx = (), imap_shp = (600, 360), 
        pix=3./60., field_center=[(165, 27.8), ], proj='ZEA', figsize=(14, 2),
        sigma=2., vmin=None, vmax=None, title='', cmap='bwr', axes=None,
        nvss_path=None, verbose=True, cbar=True, colors='r', logscale=False,
        freqdiff=False):

    with h5.File(map_name, 'r') as f:
        if verbose:
            print(('Load maps from %s'%map_name))
        #imap = f[map_key][indx]
        imap = al.load_h5(f, map_key)
        imap = al.make_vect(imap, axis_names = imap.info['axes'])
        #imap = f['dirty_map'][:]
        pixs = f['map_pix'][:]
        nside = f['nside'][()]

        print((list(f.keys())))

    freq = imap.get_axis('freq')
    freq = freq[indx[-1]]
    imap = imap[indx]
    imap = np.ma.masked_equal(imap, 0)

    if freqdiff:
        imap, mask = tsys.rms_ABAB_map(imap, imap.mask)
        imap = np.ma.masked_equal(imap, 0)
        freq = freq[:imap.shape[0] * 4]
        freq.shape = (-1, 4)
        freq = np.mean(freq, axis=1)

        std  = np.ma.std(imap)
        print('RMS [dfreq]: %10.8f K [%10.8f MHz]'%(std, (freq[1]-freq[0])/4.))

    if isinstance( indx[-1], slice):
        if map_key == 'noise_diag':
            imap = np.ma.mean(imap, axis=0)
            imap[imap==0] = np.inf
            imap = 1./imap
            imap = np.sqrt(imap)
        else:
            imap = np.ma.mean(imap, axis=0)
        freq_label = 'Frequency %5.2f - %5.2f MHz'%(freq[0], freq[-1])
    else:
        freq_label = 'Frequency %5.2f MHz'%freq

    return _plot_map_hp_multi(imap, pixs, nside, imap_shp = imap_shp, 
        pix=pix, field_center_list=field_center, proj=proj, figsize=figsize,
        sigma=sigma, vmin=vmin, vmax=vmax, cmap=cmap, axes=axes,
        nvss_path=nvss_path, title = title + ' ' + freq_label, 
        logscale=logscale)
    #return _plot_map_hp_multi(imap, pixs, nside, imap_shp = imap_shp, 
    #    pix=pix, field_center_list=field_center, proj=proj, figsize=figsize,
    #    sigma=sigma, vmin=vmin, vmax=vmax, cmap=cmap, axes=axes,
    #    nvss_path=nvss_path, title = title + ' ' + freq_label, 
    #    cbar=cbar, colors=colors)

def get_projection(proj, imap_shp, field_center, pix):

    target_header = "NAXIS   = 2\n"\
                  + "NAXIS1  = %d\n"%imap_shp[0]\
                  + "NAXIS2  = %d\n"%imap_shp[1]\
                  + "CTYPE1  = \'RA---%s\'\n"%proj\
                  + "CRPIX1  = %d\n"%(imap_shp[0]/2)\
                  + "CRVAL1  = %f\n"%field_center[0]\
                  + "CDELT1  = -%f\n"%pix\
                  + "CUNIT1  = \'deg\'\n"\
                  + "CTYPE2  = \'DEC--%s\'\n"%proj\
                  + "CRPIX2  = %d\n"%(imap_shp[1]/2)\
                  + "CRVAL2  = %f\n"%field_center[1]\
                  + "CDELT2  = %f\n"%pix\
                  + "CUNIT2  = \'deg\'\n"\
                  + "COORDSYS= \'icrs\'"

    target_header = fits.Header.fromstring(target_header, sep = '\n')

    return target_header, WCS(target_header)

def _plot_map_hp_multi(imap, pixs, nside, imap_shp = (600, 360), axes=None, 
        pix=3./60., field_center_list=[(165, 27.8), ], proj='ZEA', figsize=(14, 2),
        sigma=2., vmin=None, vmax=None, title='', cmap='bwr', nvss_path=None,
        logscale=False):
    
    imap_full = np.zeros(hp.nside2npix(nside))
    imap_full[pixs] = imap
    imap_full = np.ma.masked_equal(imap_full, 0)

    if sigma is not None:
        mean = np.ma.mean(imap_full)
        std  = np.ma.std(imap_full)
        print('RMS [full freq]: %10.8f K '%(std))
        vmin = mean - sigma * std
        vmax = mean + sigma * std
    #else:
    #    if vmin is None: vmin = imap_full.min()
    #    if vmax is None: vmax = imap_full.max()

    if axes is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(len(field_center_list), 1, right=0.90, 
                figure=fig, wspace=0.0)
        cax = fig.add_axes([0.91, 0.2, 0.1/float(figsize[0]), 0.6])
        ax_list = []
    else:
        fig, ax_list, cax = axes
    for ii, field_center in enumerate(field_center_list):

        target_header, projection = get_projection(proj, imap_shp, field_center, pix)

        array, footprint = reproject_from_healpix((imap_full, 'icrs'), target_header, 
                nested=False, order='nearest-neighbor')

        if axes is None:
            ax = fig.add_subplot(gs[ii, 0], projection=projection, frame_class=RectangularFrame)
            ax_list.append(ax)
        else:
            if isinstance( ax_list[0], float) :
                ax = fig.add_axes(ax_list, projection=WCS(target_header),
                        frame_class=RectangularFrame) 
                ax_list = [ax,]
            else:
                ax = ax_list[ii]

        array = np.ma.masked_invalid(array)
        array[array==0] = np.ma.masked

        if sigma is None:
            if vmin is None: vmin = array.min()
            if vmax is None: vmax = array.max()

        if logscale: 
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        #im = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(array, cmap=cmap, norm=norm)

        ax.minorticks_on()
        ax.set_aspect('equal')
        #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
        ax.coords[0].set_major_formatter('hh:mm')
        ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
        ax.coords[1].set_major_formatter('dd:mm')
        ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
        ax.coords.grid(color='0.5', linestyle='--', lw=0.5)

        if ii == 0:
            ax.set_title( title)

        if nvss_path is not None:
            nvss_range = [
                    field_center[0] - imap_shp[0] * 0.5 * pix * 1.1, 
                    field_center[0] + imap_shp[0] * 0.5 * pix * 1.1, 
                    field_center[1] - imap_shp[1] * 0.5 * pix, 
                    field_center[1] + imap_shp[1] * 0.5 * pix, 
                    ]
            plot_nvss(nvss_path, [nvss_range,], (fig, ax), ms=3, mew=0.2)

    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)

    ticks_label = []
    if not logscale:
        ticks = list(np.linspace(vmin, vmax, 5))
        for x in ticks:
            ticks_label.append(r"$%5.2f$"%x)
        fig.colorbar(im, cax=cax, ticks=ticks)
        cax.set_yticklabels(ticks_label, rotation=90, va='center')
        cax.minorticks_off()
    else:
        fig.colorbar(im, cax=cax)
        cax.minorticks_off()
        ticks_label = cax.get_yticklabels()
        cax.set_yticklabels(ticks_label, rotation=90, va='center')
    c_label = r'$T\,$ K'
    cax.set_ylabel(c_label)


    return fig, ax_list, cax


def _plot_map_hp(imap, pixs, nside, imap_shp = (600, 360), 
        pix=3./60., field_center=(165, 27.8), proj='ZEA', figsize=(14, 2),
        sigma=2., vmin=None, vmax=None, title='', cmap='bwr', axes=None,
        nvss_path=None, cbar=True, colors='r', fill=False):
    
    imap_full = np.zeros(hp.nside2npix(nside))
    imap_full[pixs] = imap
    imap_full = np.ma.masked_equal(imap_full, 0)

    target_header, projection = get_projection(proj, imap_shp, field_center, pix)

    array, footprint = reproject_from_healpix((imap_full, 'icrs'), target_header, 
            nested=False, order='nearest-neighbor')

    if axes is None:
        fig = plt.figure(figsize=figsize)
        if cbar:
            ax = fig.add_axes([0.06, 0.1, 0.88, 0.8], projection=projection,
                              frame_class=RectangularFrame)
            cax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
        else:
            ax = fig.add_axes([0.06, 0.1, 0.92, 0.8], projection=projection,
                              frame_class=RectangularFrame)
    else:
        if len(axes) == 3:
            fig, rect, cax = axes
            ax = fig.add_axes(rect, projection=WCS(target_header),
                              frame_class=RectangularFrame)
        else:
            fig, ax = axes
    #array = np.ma.masked_invalid(footprint)
    array = np.ma.masked_invalid(array)
    array[array==0] = np.ma.masked


    if sigma is not None:
        mean = np.ma.mean(array)
        std  = np.ma.std(array)
        vmin = mean - sigma * std
        vmax = mean + sigma * std
    else:
        if vmin is None: vmin = array.min()
        if vmax is None: vmax = array.max()

    if cbar:
        im = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        array = array.data
        ax.contour(array, levels=[0.1, 1,], colors=colors, 
                linewidths=1.0)
        if fill:
            ax.contourf(array, levels=[0.1, 1,], colors=colors, linewidths=2.5, 
                    alpha=0.3)

    ax.minorticks_on()
    ax.set_aspect('equal')
    #ax.coords[0].set_ticks(spacing=300 * u.arcmin,)
    ax.coords[0].set_major_formatter('hh:mm')
    ax.coords[0].set_separator((r'$^{\rm h}$', r'$^{\rm m}$', r'$^{\rm s}$'))
    ax.coords[0].set_axislabel('R.A. (J2000)', minpad=0.5)
    ax.coords[1].set_major_formatter('dd:mm')
    ax.coords[1].set_axislabel('Dec. (J2000)', minpad=0.5)
    ax.coords.grid(color='black', linestyle='--', lw=0.5)

    ax.set_title( title)

    ticks = list(np.linspace(vmin, vmax, 5))
    ticks_label = []
    for x in ticks:
        ticks_label.append(r"$%5.2f$"%x)
    if cbar:
        fig.colorbar(im, cax=cax, ticks=ticks)
        cax.set_yticklabels(ticks_label, rotation=90, va='center')
        cax.minorticks_off()
        c_label = r'$T\,$ K'
        cax.set_ylabel(c_label)

    if nvss_path is not None:
        nvss_range = [
                field_center[0] - imap_shp[0] * 0.5 * pix, 
                field_center[0] + imap_shp[0] * 0.5 * pix, 
                field_center[1] - imap_shp[1] * 0.5 * pix, 
                field_center[1] + imap_shp[1] * 0.5 * pix, 
                ]
        plot_nvss(nvss_path, [nvss_range,], (fig, ax), ms=10, mew=2.0)

    if cbar:
        return fig, ax, cax
    else:
        return fig, ax

def plot_nvss(nvss_path, nvss_range, axes, threshold=300., ms=10, mew=1,):

    fig, ax = axes

    nvss_cat = source.get_nvss_radec(nvss_path, nvss_range)
    nvss_sel = nvss_cat['FLUX_20_CM'] > threshold
    nvss_ra  = nvss_cat['RA'][nvss_sel]
    nvss_dec = nvss_cat['DEC'][nvss_sel]
    ax.plot(nvss_ra, nvss_dec, 'o', mec='k', mfc='none', ms=ms, mew=mew,
            transform=ax.get_transform('icrs'))


def load_maps_npy(dm_path, dm_file):

    #with h5.File(dm_path+dm_file, 'r') as f:
    #print f.keys()
    imap = al.make_vect(al.load(dm_path+dm_file))
        
    freq= imap.get_axis('freq')
    #print freq[1] - freq[0]
    #print freq[0], freq[-1]
    ra  = imap.get_axis( 'ra')
    dec = imap.get_axis('dec')
    ra_edges  = imap.get_axis_edges( 'ra')
    dec_edges = imap.get_axis_edges('dec')
    #print imap.get_axis('freq')
    return imap, ra, dec, ra_edges, dec_edges, freq

def load_maps(dm_path, dm_file, name='clean_map'):

    with h5.File(dm_path+dm_file, 'r') as f:
        print((list(f.keys())))
        imap = al.load_h5(f, name)
        imap = al.make_vect(imap, axis_names = imap.info['axes'])
        #imap = al.make_vect(al.load_h5(f, name))
        
        freq= imap.get_axis('freq')
        #print freq[1] - freq[0]
        #print freq[0], freq[-1]
        ra  = imap.get_axis( 'ra')
        dec = imap.get_axis('dec')
        ra_edges  = imap.get_axis_edges( 'ra')
        dec_edges = imap.get_axis_edges('dec')
        #print imap.get_axis('freq')

        try:
            mask = f['mask'][:]
        except KeyError:
            mask = None

    return imap, ra, dec, ra_edges, dec_edges, freq, mask

def load_maps_hp(dm_path, dm_file, name='clean_map'):

    with h5.File(dm_path + dm_file, 'r') as f:
        imap = al.load_h5(f, 'clean_map')
        #imap = al.make_vect(imap, axis_names = imap.info['axes'])
        imap = al.make_vect(imap)
        nside = f['nside'][()]
        pixs = f['map_pix'][:]
        freq = imap.get_axis('freq')

        try:
            mask = f['mask'][:]
        except KeyError:
            #print('No mask')
            mask = None

    return imap, pixs, freq, mask

def show_map(map_path, map_type, indx = (), figsize=(10, 4),
            xlim=None, ylim=None, logscale=False,
            vmin=None, vmax=None, sigma=2., inv=False, mK=True,
            title='', c_label=None, factorize=False, 
            nvss_path = None, smoothing=False, 
            opt=False, print_info=False, submean=False):

    ext = os.path.splitext(map_path)[-1]

    if ext == '.h5':
        with h5.File(map_path, 'r') as f:
            keys = tuple(f.keys())
            imap = al.load_h5(f, map_type)
            if print_info:
                logger.info( ('%s '* len(keys))%keys )
                print((imap.info))
            try:
                mask = f['mask'][:].astype('bool')
            except KeyError:
                mask = None
    elif ext == '.npy':
        imap = al.load(map_path)
        mask = None
    else:
        raise IOError('%s not exists'%map_path)

    imap = al.make_vect(imap, axis_names = imap.info['axes'])
    freq = imap.get_axis('freq')
    ra   = imap.get_axis( 'ra')
    dec  = imap.get_axis('dec')
    ra_edges  = imap.get_axis_edges( 'ra')
    dec_edges = imap.get_axis_edges('dec')

    if map_type == 'noise_diag' and factorize:
        imap = fgrm.make_noise_factorizable(imap)

    #imap[np.abs(imap) < imap.max() * 1.e-4] = 0.
    imap = np.ma.masked_equal(imap, 0)
    imap = np.ma.masked_invalid(imap)
    if mask is not None:
        imap[mask] = np.ma.masked

    imap = imap[indx]
    freq = freq[indx[0]]
    if isinstance( indx[0], slice):
        freq = (freq[0], freq[-1])
        #print imap.shape
        imap = np.ma.mean(imap, axis=0)
    else:
        freq = (freq,)

    if not opt:
        if mK: 
            if map_type == 'noise_diag':
                imap = imap * 1.e6
                unit = r'$[\rm mK]^2$'
            else:
                imap = imap * 1.e3
                unit = r'$[\rm mK]$'
        else:
            if map_type == 'noise_diag':
                unit = r'$[\rm K]^2$'
            else:
                unit = r'$[\rm K]$'
    else:
        unit = r'$\delta N$'
        if c_label is None:
            c_label = unit

    if inv:
        imap[imap==0] = np.inf
        imap = 1./imap

    if xlim is None:
        xlim = [ra_edges.min(), ra_edges.max()]
    if ylim is None:
        ylim = [dec_edges.min(), dec_edges.max()]

    #imap -= np.ma.mean(imap)

    if smoothing:
        _sig = 3./(8. * np.log(2.))**0.5 / 1.
        imap = gf(imap, _sig)

    if submean:
        imap -= np.ma.mean(imap)
    
    if logscale:
        imap = np.ma.masked_less(imap, 0)
        if vmin is None: vmin = np.ma.min(imap)
        if vmax is None: vmax = np.ma.max(imap)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if sigma is not None:
            if vmin is None: vmin = np.ma.mean(imap) - sigma * np.ma.std(imap)
            if vmax is None: vmax = np.ma.mean(imap) + sigma * np.ma.std(imap)
        else:
            if vmin is None: vmin = np.ma.min(imap)
            if vmax is None: vmax = np.ma.max(imap)
            #if vmax is None: vmax = np.ma.median(imap)
 
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    print((vmin, vmax))
    
    
    fig = plt.figure(figsize=figsize)
    l = 0.08 * 10. / figsize[0]
    b = 0.08 *  4.  / figsize[1]
    w = 1 - 0.20 * 10.  / figsize[0]
    h = 1 - 0.10 *  4.  / figsize[1]
    ax = fig.add_axes([l, b, w, h])
    l = 1 - 0.11 * 10. / figsize[0]
    b = 0.20 *  4  / figsize[1]
    w = 1 - 0.10 * 10  / figsize[0] - l
    h = 1 - 0.34 *  4  / figsize[1]
    cax = fig.add_axes([l, b, w, h])
    ax.set_aspect('equal')

    #imap = np.sum(imap, axis=1)
    
    #imap = np.array(imap)
    
    cm = ax.pcolormesh(ra_edges, dec_edges, imap.T, norm=norm)
    if len(freq) == 1:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$'%freq)
    else:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$ - $%7.3f\,{\rm MHz}$'%freq)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'${\rm RA}\,[^\circ]$')
    ax.set_ylabel(r'${\rm Dec}\,[^\circ]$')

    nvss_range = [ [ra_edges.min(), ra_edges.max(), 
                    dec_edges.min(), dec_edges.max()],]
    if nvss_path is not None:
        nvss_cat = source.get_nvss_radec(nvss_path, nvss_range)
        nvss_sel = nvss_cat['FLUX_20_CM'] > 10.
        nvss_ra  = nvss_cat['RA'][nvss_sel]
        nvss_dec = nvss_cat['DEC'][nvss_sel]
        ax.plot(nvss_ra, nvss_dec, 'ko', mec='k', mfc='none', ms=8, mew=1.5)

        _sel = nvss_cat['FLUX_20_CM'] > 100.
        _id  = nvss_cat['NAME'][_sel]
        _ra  = nvss_cat['RA'][_sel]
        _dec = nvss_cat['DEC'][_sel]
        _flx =  nvss_cat['FLUX_20_CM'][_sel]
        for i in range(np.sum(_sel)):
            ra_idx = np.digitize(_ra[i], ra_edges) - 1
            dec_idx = np.digitize(_dec[i], dec_edges) - 1
            ax.plot(ra[ra_idx], dec[dec_idx], 'wx', ms=10, mew=2)
            _c = SkyCoord(_ra[i] * u.deg, _dec[i] * u.deg)
            print(('%s [RA,Dec]:'%_id[i] \
                    + '[%7.4fd'%_c.ra.deg \
                    + '(%dh%dm%6.4f) '%_c.ra.hms\
                    + ': %7.4fd], FLUX %7.4f Jy'%(_c.dec.deg, _flx[i]/1000.)))
    if not logscale:
        ticks = list(np.linspace(vmin, vmax, 5))
        ticks_label = []
        for x in ticks:
            ticks_label.append(r"$%5.2f$"%x)
        fig.colorbar(cm, ax=ax, cax=cax, ticks=ticks)
        cax.set_yticklabels(ticks_label)
    else:
        fig.colorbar(cm, ax=ax, cax=cax)
    cax.minorticks_off()
    if c_label is None:
        c_label = r'$T\,$' + unit
    cax.set_ylabel(c_label)

    return xlim, ylim, (vmin, vmax), fig


def plot_map(data, indx = (), figsize=(10, 4),
            xlim=None, ylim=None, logscale=False,
            vmin=None, vmax=None, sigma=2., inv=False, mK=True,
            smoothing=False, nvss_path=None, c_label=None, title=''):

    #imap *= 1.e3
    #imap = data[0][indx + (slice(None),)]
    #if len(indx) == 1: indx = indx + (slice(None),)
    imap = data[0][indx]
    imap = np.ma.masked_equal(imap, 0)
    imap = np.ma.masked_invalid(imap)

    freq = data[5][indx[-1]]
    if isinstance( indx[-1], slice):
        freq = (freq[0], freq[-1])
        print((imap.shape))
        imap = np.ma.mean(imap, axis=0)
    else:
        freq = (freq, )

    if mK:
        imap = imap * 1.e3
        unit = r'$[\rm mK]$'
    else:
        unit = r'$[\rm K]$'
        
    if inv:
        imap[imap==0] = np.inf
        imap = 1./ imap
        #imap = fgrm.noise_diag_2_weight(data[0])[indx]
        #imap = fgrm.make_noise_factorizable(data[0], weight_prior=5.e-5)[indx]


        if mK:
            imap /= 1.e3
            unit = r'$[\rm mK^{-2}]$'
        else:
            unit = r'$[\rm K^{-2}]$'
    ra_edges  = data[3]
    dec_edges = data[4]

    if xlim is None:
        xlim = [ra_edges.min(), ra_edges.max()]
    if ylim is None:
        ylim = [dec_edges.min(), dec_edges.max()]
    
    if smoothing:
        _sig = 0.8/(8. * np.log(2.))**0.5 / 0.4
        imap = gf(imap, _sig)
        
    #imap[np.abs(imap) < imap.max() * 1.e-10] = 0.
    #imap -= np.ma.mean(imap)
    
    if logscale:
        imap = np.ma.masked_less(imap, 0)
        if vmin is None: vmin = np.ma.min(imap)
        if vmax is None: vmax = np.ma.max(imap)
        #print vmin, vmax
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if sigma is not None:
            if vmin is None: vmin = np.ma.mean(imap) - sigma * np.ma.std(imap)
            if vmax is None: vmax = np.ma.mean(imap) + sigma * np.ma.std(imap)
        else:
            if vmin is None: vmin = np.ma.min(imap)
            if vmax is None: vmax = np.ma.max(imap)
            #if vmax is None: vmax = np.ma.median(imap)
 
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_axes([0.07, 0.10, 0.70, 0.8])
    #cax = fig.add_axes([0.78, 0.2, 0.01, 0.6])
    l = 0.08 * 10. / figsize[0]
    b = 0.08 *  4.  / figsize[1]
    w = 1 - 0.20 * 10.  / figsize[0]
    h = 1 - 0.10 *  4.  / figsize[1]
    ax = fig.add_axes([l, b, w, h])
    l = 1 - 0.11 * 10. / figsize[0]
    b = 0.20 *  4  / figsize[1]
    w = 1 - 0.10 * 10  / figsize[0] - l
    h = 1 - 0.34 *  4  / figsize[1]
    cax = fig.add_axes([l, b, w, h])
    ax.set_aspect('equal')

    #imap = np.sum(imap, axis=1)
    
    #imap = np.array(imap)
    
    cm = ax.pcolormesh(ra_edges, dec_edges, imap.T, norm=norm)
    if len(freq) == 1:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$'%freq)
    else:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$ - $%7.3f\,{\rm MHz}$'%freq)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'${\rm RA}\,[^\circ]$')
    ax.set_ylabel(r'${\rm Dec}\,[^\circ]$')

    nvss_range = [ [ra_edges.min(), ra_edges.max(), dec_edges.min(), dec_edges.max()],]
    if nvss_path is not None:
        nvss_cat = source.get_nvss_radec(nvss_path, nvss_range)
        nvss_sel = nvss_cat['FLUX_20_CM'] > 500.
        nvss_ra  = nvss_cat['RA'][nvss_sel]
        nvss_dec = nvss_cat['DEC'][nvss_sel]
        ax.plot(nvss_ra, nvss_dec, 'ko', mec='k', mfc='w', ms=8, mew=1.5)

        _sel = nvss_cat['FLUX_20_CM'] > 1000.
        _id  = nvss_cat['NAME'][_sel]
        _ra  = nvss_cat['RA'][_sel]
        _dec = nvss_cat['DEC'][_sel]
        _flx =  nvss_cat['FLUX_20_CM'][_sel]
        for i in range(np.sum(_sel)):
            print(('%s [RA,Dec]: [%7.4f : %7.4f], FLUX %7.4f Jy'%(
                    _id[i], _ra[i], _dec[i], _flx[i]/1000.)))

    if not logscale:
        ticks = list(np.linspace(vmin, vmax, 5))
        ticks_label = []
        for x in ticks:
            ticks_label.append(r"$%5.2f$"%x)
        fig.colorbar(cm, ax=ax, cax=cax, ticks=ticks)
        cax.set_yticklabels(ticks_label)
    else:
        fig.colorbar(cm, ax=ax, cax=cax)
    cax.minorticks_off()
    if c_label is None:
        c_label = r'$T\,$' + unit
    cax.set_ylabel(c_label)

    return xlim, ylim, (vmin, vmax)
    
def load_ps1d(ps_path, ps_file):

    with h5.File(ps_path + ps_file , 'r') as f:

        #print f.keys()
        ps1d_all = f['binavg_1d'][:]
        ps1d_kbin = f['kbin'][:]
    
    ps1d = np.mean(ps1d_all, axis=0)
    ps1d_error = np.std(ps1d_all, axis=0)
    
    fact  = 1. #ps1d_kbin**3./(2.*np.pi)**2.
    ps1d *= fact
    ps1d_error *= fact
    
    
    return ps1d, ps1d_error, ps1d_kbin 

def load_ps1d_opt(ps_path, ps_file):

    with h5.File(ps_path + ps_file , 'r') as f:

        print((list(f.keys())))
        ps1d_all = f['binavg_1d'][:]
        ps1d_kbin = f['kbin'][:]
    
    ps1d = ps1d_all[0, ...]
    ps1d_error = np.std(ps1d_all[1:,...], axis=0)
    
    fact  = 1. # ps1d_kbin**3./(2.*np.pi)**2.
    ps1d *= fact
    ps1d_error *= fact
    
    
    return ps1d, ps1d_error, ps1d_kbin 

def make_noise_diag(cov_inv_diag, Rot, map_shp):
    bad_modes = cov_inv_diag < 1.e-5 * cov_inv_diag.max()
    noise_diag = 1./cov_inv_diag
    noise_diag[bad_modes] = 0.

    tmp_mat = Rot * noise_diag[None, :]
    for jj in range(np.prod(map_shp)):
        noise_diag[jj] = sp.dot(tmp_mat[jj, :], Rot[jj, :])
    noise_diag.shape = map_shp
    return noise_diag


def plot_1dps_old(ps_list, logk=True, figsize=(8,6), vmin=None, vmax=None):
    
    kh = ps_list[0][2]
    ps_n = len(ps_list)
    shift = np.arange(ps_n) - (float(ps_n) - 1.)/ 2.
    if logk:
        dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))        
        shift = dk ** shift
    else:
        dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
        shift = dk * shift
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.07, 0.10, 0.80, 0.8])
    
    for i in range(ps_n):
        c    = _c_list[i]
        ps, ps_e, kh, label = ps_list[i]

        ax.errorbar(kh * shift[i], ps, ps_e, fmt='.', elinewidth=1, 
                    capsize=2, color=c, label=label)
    
    #psin = np.loadtxt('/users/ycli/code/tlpipe/tlpipe/sim/data/wigglez_halofit_z0.8.dat')
    #psin[:,1 ] = psin[:,1 ] * psin[:,0]**3./(2.*np.pi)**2 * 0.5 # * 0.9e-8
    #ax.plot(psin[:,0], psin[:,1 ], 'k--')
    #plt.plot(psin[:,0], psin[:,1 ]*1e-2, 'k--')
    
    ps_lowz12_path = '/users/ycli/data/SDSS/dr12/GilMarin_boss_data/pre-recon/lowz/'
    ps_lowz12_file = 'GilMarin_2016_LOWZDR12_bestfit_power_spectrum_monopole_pre_recon.txt'
    ps_lowz12_file = 'GilMarin_2016_LOWZDR12_measurement_power_spectrum_monopole_pre_recon.txt'
    psin = np.loadtxt(ps_lowz12_path + ps_lowz12_file)
    #psin[:,1 ] = psin[:,1 ] * psin[:,0]**3./(2.*np.pi)**2 # * 0.9e-8
    ax.plot(psin[:,0], psin[:,1 ], 'k--')



    ns = ( 25. / np.sqrt(1. * 13.e6  * 2 * 16.) )**2.
    print((ns ** 0.5))

    #ns_kh = np.logspace(-3, 2, 100)
    ns_kh = psin[:,0]
    ns *= ns_kh ** 3. / (2. * np.pi)**2
    #ns *= np.ones_like(ns_kh)
    #ax.plot(ns_kh, ns)
    ax.legend()

    ax.set_xlim(5.e-3, 2.)
    ax.set_ylim(vmin, vmax)

    ax.loglog()
    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    ax.set_ylabel(r'P(k)')


def plot_1dps(ps_path, ps_name_list, figsize=(8,6), title='',
              vmin=None, vmax=None, logk=True, ns=25., cube_size=1.):
    
    ps_keys = list(ps_name_list.keys())
    ps_n = len(ps_keys)
    shift = np.arange(ps_n) - (float(ps_n) - 1.)/ 2.
    shift *= 0.5
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.07, 0.10, 0.80, 0.8])   
    
    for ii in range(ps_n):
        c    = _c_list[ii]
        #ps, ps_e, kh, label = ps_list[i]
        label = ps_keys[ii]
        ps, ps_e, kh = load_ps1d(ps_path, ps_name_list[label])
        
        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            kh = kh + ( dk * shift[ii] )
            
        ax.errorbar(kh , ps, ps_e, fmt='o', elinewidth=1.5, 
                    mfc='w', mew='1.5', ms=7,
                    capsize=5, color=c, label=label)
    # plot 25 K noise
    freq = np.linspace(950, 1350, 32)
    dfreq = (freq[1] - freq[0]) * 1.e6
    #ns = ( 25. / np.sqrt(1. * dfreq * 2. * 16.) )**2.
    ns = ns ** 2. * cube_size
    #print

    ns_kh = np.logspace(-3, 2, 100)
    #ns *= ns_kh ** 3. / (2. * np.pi)**2
    ns *= np.ones_like(ns_kh)
    ax.plot(ns_kh, ns, 'k--')
    ax.plot(ns_kh, ns * 2., 'k:')
    
    ax.legend()

    ax.set_xlim(5.e-3, 1.)
    ax.set_ylim(vmin, vmax)

    ax.loglog()
    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    ax.set_ylabel(r'P(k)')
    
def plot_2dps(ps_path, ps_name_list, figsize=(16, 4),title='',
             vmax=None, vmin=None, logk=True):
    
    ps_keys = list(ps_name_list.keys())
    cols = len(ps_keys)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    gs = gridspec.GridSpec(1, cols, right=0.85, figure=fig, wspace=0.1)
    cax = fig.add_axes([0.86, 0.2, 0.1/float(figsize[0]), 0.6])
    ax = []
    
    for ii in range(cols):
        ps_name = ps_name_list[ps_keys[ii]]
        with h5.File(ps_path + ps_name, 'r') as f:
            #print f.keys()
            x = f['kbin_x_edges'][:]
            y = f['kbin_y_edges'][:]
            xlim = [x.min(), x.max()]
            ylim = [y.min(), y.max()]
            
            ps2d = np.ma.masked_equal(f['binavg_2d'][:],0)
            ps2d = np.ma.mean(ps2d, axis=0)
        if vmin is None:
            vmin = np.min(ps2d)
        if vmax is None:
            vmax = np.max(ps2d)
            
        ax = fig.add_subplot(gs[0,ii])
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(x, y, ps2d.T, norm=norm)
        if logk:
            ax.loglog()
        if ii != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$k_\parallel$')
        ax.set_aspect('equal')
        ax.tick_params(direction='out', length=2, width=1)
        ax.tick_params(which='minor', direction='out', length=1.5, width=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$k_\perp$')

        ax.set_title('%s'%ps_keys[ii])
            
        
    fig.colorbar(im, ax=ax, cax=cax)
    cax.minorticks_off()
    cax.set_ylabel(r'$P(k)$')

