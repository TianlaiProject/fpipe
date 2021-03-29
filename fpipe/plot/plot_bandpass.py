import numpy as np
import gc
from fpipe.timestream import timestream_task
from fpipe.utils import axes_utils
import h5py as h5
from astropy.time import Time
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from caput import mpiarray
from scipy.signal import medfilt
from scipy.signal import lombscargle
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy import interpolate

from fpipe.timestream import destripe

import matplotlib.pyplot as plt


def plot_gt(file_name, l=5, fk=0.01, alpha=1.5, title=''):

    #fk = 0.01
    #alpha = 1.5

    with h5.File(file_name, 'r') as f:
        nd    = f['gtgnu'][:]
        time  = f['time'][:]
        freq  = f['freq'][:]

    nd = np.ma.masked_invalid(nd)

    #nd[:, 200:600, ...] = 0
    nd.mask[:, 200:600, ...] = True

    _nd_t = np.ma.mean(nd, axis=1, )
    good = (np.abs(_nd_t - np.ma.mean(_nd_t, axis=0)[None, :,:]) \
            - 3.*np.ma.std(_nd_t, axis=0)[None, :, :])<0
    #nd.mask += ~good[:, None, :, :]

    #time -= time[0]
    #time /= 3600.
    xx = time - time[0]
    xx /= 3600.

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False, title=title)

    for bi in range(19):
        i = bi / 4
        j = bi - i * 4

        ax = axes[bi]
        gt = np.ma.median(nd[:, :, :, bi], axis=1)
        #var = np.ma.var(nd[:, :, :, bi], axis=1)
        var = np.ma.median(nd[:, :, :, bi], axis=1)
        #var[var==0] = np.inf

        ax.plot(xx, gt[:, 0], 'r-', lw=0.2)
        ax.plot(xx, gt[:, 1], 'b-', lw=0.2)

        gt_m = np.ma.mean(gt, axis=0)
        gt -= gt_m[None, :]

        gt[:, 0] = destripe.destriping(l,
                                       gt[good[:, 0, bi], 0],
                                       var[good[:, 0, bi], 0],
                                       time[good[:, 0, bi]],
                                       fk, alpha)(time)
        gt[:, 1] = destripe.destriping(l,
                                       gt[good[:, 0, bi], 1],
                                       var[good[:, 0, bi], 1],
                                       time[good[:, 0, bi]],
                                       fk, alpha)(time)
        #gt = median_filter(gt, [11, 1])

        gt += gt_m[None, :]

        ax.plot(xx, gt[:, 0], 'r-', lw=1)
        ax.plot(xx, gt[:, 1], 'b-', lw=1)

        #ax.legend(title='Feed %02d'%(bi+1), loc=2)
        ax.text(0.04, 0.8, 'Feed %02d'%bi, transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.5, ec='none'))
        ax.set_ylim(0.81, 1.19)
        ax.set_xlim(xx[1], xx[-1])
        #ax.semilogy()
        if i != 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time [hr]')
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$g(t)$')



def plot_gtgnu(file_name, title='', pol=0, norm=False, output=None):
    
    with h5.File(file_name, 'r') as f:
        gtgnu = f['gtgnu'][:]
        time  = f['time'][:]
        freq  = f['freq'][:]
        
    time -= time[0]
    time /= 3600.

    freq /= 1.e3

    fig, axes = axes_utils.setup_axes(5, 4)
    for bi in range(gtgnu.shape[3]):
        i = bi / 4
        j = bi - i * 4
    
        ax = axes[bi]
        
        if norm:
            _n = np.ma.mean(gtgnu[:, :, pol, bi], axis=1)
            _g = gtgnu[:, :, pol, bi].T / _n[None, :]
            vmin = 0.95
            vmax = 1.05
        else:
            _g = gtgnu[:, :, pol, bi].T
            vmin = 0.7
            vmax = 1.3
        
        im = ax.pcolormesh(time, freq, _g, vmin=vmin, vmax=vmax)
        
        ax.set_ylim(freq.min(), freq.max())
        #ax.set_xlim(time.min(), time.max())
        ax.set_xlim(time.min(), time.max())
        
        ax.text(0.04, 0.8, 'Feed %02d'%bi, transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.5, ec='none'))
        #ax.semilogy()
        if i != 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time [hr]')
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$\nu$ [GHz]')

    #cax = fig.add_axes([0.72, 0.18, 0.16, 0.02])
    cax = axes[-1]
    fig.colorbar(im, cax=cax, orientation='horizontal')
    cax.set_xlabel(r'$g(t, \nu)$')
    cax.set_title(title)

    if output is not None:
        fig.savefig(output, formate='png')

def plot_bandpass(bandpass_path, bandpass_name, pol=0,
                  ymin=None, ymax=None, normalize=True, ratio=True, output_path=None):
    
    
    _pol = ['XX', 'YY'][pol]
    
    fig = plt.figure(figsize=[12, 8])
    gs = gridspec.GridSpec(5, 4, left=0.07, bottom=0.07, top=0.97, right=0.97,
                           figure=fig, wspace=0.0, hspace=0.0)
    
    suffix = ''
    if normalize: suffix += '_norm'
    if ratio: suffix += '_ratio'
    bandpass_ref = None
    time_list = []
    
    
    with h5.File(bandpass_path + bandpass_name + '.h5', 'r') as f:
        bandpass_combined = f['bandpass'][:]
        freq     = f['freq'][:]
        time_list = f['time'][:]
    
    #print bandpass_combined.shape
    cnorm = mpl.colors.Normalize(vmin=0, vmax=bandpass_combined.shape[0])
    
    axes = []
    for b in range(19):
        i = b/4
        j = b - i * 4
            
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)
    
    #for block_id in range(blk_st, blk_ed+1):
    for ii in range(bandpass_combined.shape[0]):
    
        #bandpass, freq, time = load_bandpass(bandpass_path, 
        #    bandpass_temp%(bandpass_name, block_id, block_id) + '_%s.h5', tnoise_path)
        #time_list.append(time)
        #bandpass = np.ma.array(bandpass, mask=False)
        #bandpass_smooth = medfilt(bandpass, [1, 201, 1])
        #bandpass_smooth = np.ma.array(bandpass_smooth, mask=False)
        #bandpass_smooth = smooth_bandpass(bandpass.copy(), axis=1)
        bandpass_smooth = bandpass_combined[ii].copy()
        
        if normalize:
            bandpass_smooth /= np.median(bandpass_smooth, axis=1)[:, None, :]
            #print bandpass_smooth.min(), bandpass_smooth.max()
        
        if ratio:
            if bandpass_ref is None:
                bandpass_ref = bandpass_smooth.copy()
            #print bandpass_ref.min(), bandpass_ref.max()
            bandpass_smooth /= bandpass_ref.copy()
            ylabel = r'$g(\nu, t) / g(\nu, t_0)$'
        else:
            ylabel = r'$g(\nu, t)$'
        
        for b in range(19):
            ax = axes[b]
            #ax.plot(freq, bandpass[b, :, pol], '-', color='0.5', lw=0.2)
            ax.plot(freq, bandpass_smooth[b, :, pol], c=cm.jet(cnorm(ii)), 
                    lw=0.8)

            ax.set_ylim(ymin, ymax)
            ax.set_xlim(freq.min(), freq.max())
            
            if ii == 0:
                ax.text(0.70, 0.9, 'Feed%02d %s'%(b, _pol), transform=ax.transAxes)
            
            if i != 4:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Frequency [MHz]')
            
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylabel)

    if not ratio:
        bandpass_combined = np.median(bandpass_combined, axis=0)
        bandpass_combined = medfilt(bandpass_combined, [1, 201, 1])
        if normalize:
            bandpass_combined /= np.median(bandpass_combined, axis=1)[:, None, :]
        for b in range(19):
            axes[b].plot(freq, bandpass_combined[b, :, pol], c='k', lw=0.5)

    #print time_list
    time_list -= time_list[0]
    time_list /= 3600.
    tnorm = mpl.colors.Normalize(vmin=time_list.min(), vmax=time_list.max())

    sm = cm.ScalarMappable(cmap=cm.jet, norm=tnorm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    #ax = fig.add_subplot(gs[-1, -1])
    cax = fig.add_axes([0.76, 0.18, 0.20, 0.01])
    fig.colorbar(sm, cax=cax, orientation='horizontal')
    cax.set_xlabel('Time [hr]')
    if output_path is not None:
        output_name = '%s_%s%s.pdf'%(bandpass_name, _pol, suffix)
        fig.savefig(output_path + output_name)
        
def plot_bandpass_one(bandpass_path, bandpass_name, pol=0, feed=0,
                      ymin=None, ymax=None, normalize=True, ratio=True, 
                      output_path=None):
    
    
    _pol = ['XX', 'YY'][pol]
    b = feed
    
    fig = plt.figure(figsize=[5, 3])
    gs = gridspec.GridSpec(1, 1, left=0.12, bottom=0.15, top=0.95, right=0.95,
                           figure=fig, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    
    suffix = ''
    if normalize: suffix += '_norm'
    if ratio: suffix += '_ratio'
    bandpass_ref = None
    time_list = []

    with h5.File(bandpass_path + bandpass_name + '.h5', 'r') as f:
        bandpass_combined = f['bandpass'][:]
        freq     = f['freq'][:]
        time_list = f['time'][:]
    
    #print bandpass_combined.shape
    cnorm = mpl.colors.Normalize(vmin=0, vmax=bandpass_combined.shape[0])

    #for block_id in range(blk_st, blk_ed+1):
    for ii in range(bandpass_combined.shape[0]):

        bandpass_smooth = bandpass_combined[ii].copy()
        
        if normalize:
            bandpass_smooth /= np.median(bandpass_smooth, axis=1)[:, None, :]
            #print bandpass_smooth.min(), bandpass_smooth.max()
        
        if ratio:
            if bandpass_ref is None:
                bandpass_ref = bandpass_smooth.copy()
            #print bandpass_ref.min(), bandpass_ref.max()
            bandpass_smooth /= bandpass_ref.copy()
            ylabel = r'$g(\nu, t) / g(\nu, t_0)$'
        else:
            ylabel = r'$g(\nu, t)$'

        ax.plot(freq, bandpass_smooth[b, :, pol], c=cm.jet(cnorm(ii)), 
                lw=0.8)

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(freq.min(), freq.max())
        
        if ii == 0:
            ax.text(0.70, 0.9, 'Feed%02d %s'%(b, _pol), transform=ax.transAxes)

            ax.set_xlabel('Frequency [MHz]')
            ax.set_ylabel(ylabel)
            ax.minorticks_on()

    if not ratio:
        bandpass_combined = np.median(bandpass_combined, axis=0)
        #bandpass_combined = medfilt(bandpass_combined, [1, 201, 1])
        if normalize:
            bandpass_combined /= np.median(bandpass_combined, axis=1)[:, None, :]
        ax.plot(freq, bandpass_combined[b, :, pol], c='k', lw=1.0)

    ##print time_list
    #time_list -= time_list[0]
    #time_list /= 3600.
    #tnorm = mpl.colors.Normalize(vmin=time_list.min(), vmax=time_list.max())

    #sm = cm.ScalarMappable(cmap=cm.jet, norm=tnorm)
    ## fake up the array of the scalar mappable. Urgh...
    #sm._A = []
    #
    ##ax = fig.add_subplot(gs[-1, -1])
    #cax = fig.add_axes([0.76, 0.18, 0.20, 0.01])
    #fig.colorbar(sm, cax=cax, orientation='horizontal')
    #cax.set_xlabel('Time [hr]')
    if output_path is not None:
        output_name = '%s_%s%s_F%02d.pdf'%(bandpass_name, _pol, suffix, feed)
        fig.savefig(output_path + output_name)
    
