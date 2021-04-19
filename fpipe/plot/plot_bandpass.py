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

from fpipe.timestream import destripe, bandpass_cal

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

#from meerKAT_analysis.timestream import tod_ps

def gt_ps(file_list, Tnoise_file=None, title='', output=None, ymin=2.e-3, ymax=9.e-1):

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False, title=title)

    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        nd, time, freq = bandpass_cal.est_gtgnu_onefeed(file_list,
                            smooth=(1, 1), gi=bi, Tnoise_file=Tnoise_file)

        nd = np.ma.masked_invalid(nd)
        _nd_t = np.ma.mean(nd, axis=1)[:, None, :]
        mask = np.all(_nd_t == 0, axis=(1, 2))

        ps, bc = est_tcorr_psd1d_fft(_nd_t, time, mask, n_bins = 15, 
                                     f_min=1.e-4, f_max=1./16.)

        ax = axes[bi]
        ax.plot(bc, ps[:, 0, 0], 'ro-')
        ax.plot(bc, ps[:, 0, 1], 'bo-')
        ax.loglog()
        ax.set_xlim(1.9e-4, 1.2/16.)
        ax.set_ylim(ymin, ymax)
        ax.text(0.75, 0.8, 'Feed%02d'%bi, transform=ax.transAxes)
        if ii == 4: ax.set_xlabel(r'f [Hz]')
        else: ax.set_xticklabels([])

        if jj == 0: ax.set_ylabel(r'P(f)')
        else: ax.set_yticklabels([])

    if output is not None:
        fig.savefig(output, formate='pdf')

def est_tcorr_psd1d_fft(data, ax, flag, n_bins=None, inttime=None,
        f_min=None, f_max=None):

    data = data.copy()

    mean = np.mean(data[~flag, ...], axis=0)
    data -= mean[None, :, :]
    data[flag, ...] = 0.

    weight = np.ones_like(data)
    weight[flag, ...] = 0

    windowf_t = np.blackman(data.shape[0])[:, None, None]
    windowf = windowf_t

    #logger.info('apply blackman windowf')
    data   = data   * windowf.copy()
    weight = weight * windowf.copy()

    fftdata = np.fft.fft(data, axis=0) # norm='ortho')
    fftdata /= np.sqrt(np.sum(weight, axis=0))[None, ...]

    n = ax.shape[0]
    if inttime is None:
        d = ax[1] - ax[0]
    else:
        d = inttime
    freq = np.fft.fftfreq(n, d) #* 2 * np.pi

    freq_p    = freq[freq>0]
    fftdata_p = fftdata[freq>0, ...]
    fftdata_p = np.abs(fftdata_p) * np.sqrt(float(d))
    fftdata_p = fftdata_p ** 2.
    fftdata_p = fftdata_p * 2**0.5 # include negative frequency

    if n_bins is not None:

        #if avg:
        #    fftdata_p = np.mean(fftdata_p, axis=1)[:, None, :]

        fftdata_bins = np.zeros((n_bins, ) + fftdata_p.shape[1:])

        if f_min is None: f_min = freq_p.min()
        if f_max is None: f_max = freq_p.max()
        freq_bins_c = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
        freq_bins_d = freq_bins_c[1] / freq_bins_c[0]
        freq_bins_e = freq_bins_c / (freq_bins_d ** 0.5)
        freq_bins_e = np.append(freq_bins_e, freq_bins_e[-1] * freq_bins_d)
        norm = np.histogram(freq_p, bins=freq_bins_e)[0] * 1.
        norm[norm==0] = np.inf

        for i in range(fftdata_p.shape[1]):

            hist_0 = np.histogram(freq_p, bins=freq_bins_e, weights=fftdata_p[:,i,0])[0]
            hist_1 = np.histogram(freq_p, bins=freq_bins_e, weights=fftdata_p[:,i,1])[0]
            hist   = np.concatenate([hist_0[:, None], hist_1[:, None]],axis=1)
            fftdata_bins[:, i, :] = hist / norm[:, None]

        fftdata_bins[freq_bins_c <= freq_p.min()] = 0.
        fftdata_bins[freq_bins_c >= freq_p.max()] = 0.

        return fftdata_bins, freq_bins_c
    else:
        return fftdata_p, freq_p

def plot_gt(file_name, l=5, fk=0.01, alpha=1.5, title='', output=None):

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

        ax.plot(xx, gt[:, 0], 'r-', lw=0.1)
        ax.plot(xx, gt[:, 1], 'b-', lw=0.1)

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

    if output is not None:
        fig.savefig(output, formate='pdf')

def plot_baseline(baseline_file):
    
    with h5.File(baseline_file, 'r') as f:
        baseline = f['baseline'][:]
        time = f['time'][:]
    
    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)
    
    for bi in range(19):
        
        i = bi / 4
        j = bi % 4
        
        xx = time - time[0]
        xx /= 3600.
        
        ax = axes[bi]
        
        ax.plot(xx, baseline[:, 0, bi], 'r-', lw=1)
        ax.plot(xx, baseline[:, 1, bi], 'b-', lw=1)
        
        ax.text(0.04, 0.8, 'Feed %02d'%bi, transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.5, ec='none'))
        ax.set_ylim(14, 26)
        ax.set_xlim(xx[1], xx[-1])
        #ax.semilogy()
        if i != 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time [hr]')
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$T$ K')

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
    
    fig, axes = axes_utils.setup_axes(5, 4)

    suffix = ''
    if normalize: suffix += '_norm'
    if ratio: suffix += '_ratio'
    #bandpass_ref = None
    time_list = []
    
    
    with h5.File(bandpass_path + 'bandpass_' + bandpass_name + '.h5', 'r') as f:
        bandpass_combined = f['bandpass'][:]
        freq     = f['freq'][:]
        time_list = f['time'][:]

    bandpass_ref = np.median(bandpass_combined, axis=0)
    bandpass_ref = medfilt(bandpass_ref, [1, 201, 1])
    if normalize:
        bandpass_ref /= np.ma.median(bandpass_ref, axis=1)[:, None, :]
    
    cnorm = mpl.colors.Normalize(vmin=0, vmax=bandpass_combined.shape[0])
    
    for ii in range(bandpass_combined.shape[0]):
    
        bandpass_smooth = bandpass_combined[ii].copy()
        
        if normalize:
            bandpass_smooth /= np.ma.median(bandpass_smooth, axis=1)[:, None, :]

        if ratio:
            #if bandpass_ref is None:
            #    bandpass_ref = bandpass_smooth.copy()
            bandpass_smooth /= bandpass_ref.copy()
            ylabel = r'$g(\nu, t) / g(\nu, t_0)$'
        else:
            ylabel = r'$g(\nu, t)$'
        
        for b in range(19):
            i = b / 4
            j = b % 4
            ax = axes[b]
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
        #bandpass_combined = np.median(bandpass_combined, axis=0)
        #bandpass_combined = medfilt(bandpass_combined, [1, 201, 1])
        #if normalize:
        #    bandpass_combined /= np.median(bandpass_combined, axis=1)[:, None, :]
        for b in range(19):
            axes[b].plot(freq, bandpass_ref[b, :, pol], c='k', lw=0.5)
    else:
        for b in range(19):
            axes[b].axhline(1, 0, 1, c='k', lw=1.0, ls='--')

    #print time_list
    time_list -= time_list[0]
    time_list /= 3600.
    tnorm = mpl.colors.Normalize(vmin=time_list.min(), vmax=time_list.max())

    sm = cm.ScalarMappable(cmap=cm.jet, norm=tnorm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    #ax = fig.add_subplot(gs[-1, -1])
    #cax = fig.add_axes([0.76, 0.18, 0.20, 0.01])
    cax = axes[-1]
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
    
