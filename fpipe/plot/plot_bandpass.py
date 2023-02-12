import numpy as np
import gc
from fpipe.timestream import timestream_task, rm_baseline
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

from astropy import units as u
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from scipy.optimize import least_squares, curve_fit
from scipy import signal

import pandas as pd
from IPython.display import HTML

def plot_gt_ps(gt_ps_file, Tnoise_file=None, title='', output=None, ymin=5.e-3, ymax=5.e0, fn_only=False):

    with h5.File(gt_ps_file, 'r') as f:
        ps_result  = f['ps_result'][:]
        er_result  = f['er_result'][:]
        bc         = f['f_result'][:]
        paras_list = f['paras'][:]
        ps_fit     = f['ps_fit'][:]
        f_fit      = f['f_fit'][:]

    f_min = bc.min()
    f_max = bc.max()
    fig, axes = axes_utils.setup_axes(4, 5, colorbar=False, title=title)

    for bi in range(19):
        ii = int(bi / 5)
        jj = int(bi % 5)

        ax = axes[bi]

        ps_mean = ps_result[bi]
        ps_err  = er_result[bi]
        g = ps_mean[:, 0] > 0
        ax.errorbar(bc[g], ps_mean[g, 0], ps_err[g, 0], fmt='ro--')
        ax.errorbar(bc[g], ps_mean[g, 1], ps_err[g, 1], fmt='bo--')

        ax.plot(f_fit, ps_fit[bi, 0], 'r-')
        ax.plot(f_fit, ps_fit[bi, 1], 'b-')

        ax.loglog()
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(ymin, ymax)
        ax.text(0.75, 0.8, 'Feed%02d'%(bi+1), transform=ax.transAxes)
        if ii == 3: ax.set_xlabel(r'$f\, {\rm [Hz]}$')
        else: ax.set_xticklabels([])

        if jj == 0: ax.set_ylabel(r'$P(f)\,{\rm K}^2{\rm s}^{-1}$')
        else: ax.set_yticklabels([])

    if output is not None:
        fig.savefig(output, format='png')
        plt.show()
        plt.clf()

    _result = [("F%02d"%(ii+1), ) + tuple(x.flatten()) for ii, x in enumerate(paras_list)]
    if fn_only:
        _result = np.array(_result, dtype = [('Feed', 'S3'),
            ('A XX', 'f4'), ('fk XX', 'f4'), ('alpha XX', 'f4'),
            #('B XX', 'f4'), ('f0 XX', 'f4'), ('w     XX', 'f4'),
            ('A YY', 'f4'), ('fk YY', 'f4'), ('alpha YY', 'f4'),
            #('B YY', 'f4'), ('f0 YY', 'f4'), ('w     YY', 'f4'),
            ])
    else:
        _result = np.array(_result, dtype = [('Feed', 'S3'),
            ('A XX', 'f4'), ('fk XX', 'f4'), ('alpha XX', 'f4'),
            ('B XX', 'f4'), ('f0 XX', 'f4'), ('w     XX', 'f4'),
            ('A YY', 'f4'), ('fk YY', 'f4'), ('alpha YY', 'f4'),
            ('B YY', 'f4'), ('f0 YY', 'f4'), ('w     YY', 'f4'),
            ])
    _result = HTML(pd.DataFrame(_result).to_html(index=False))

    return _result
    #return np.array(paras_list), bc, ps_result, er_result

def plot_gt(file_name, l=5, fk=0.01, alpha=1.5, title='', output=None, 
        do_destripe=False, gtps_file=None):

    #fk = 0.01
    #alpha = 1.5

    #if gtps_file is not None:
    #    with h5.File(gtps_file, 'r') as f:
    #        ps_para = f['paras'][:]

    #with h5.File(file_name, 'r') as f:
    #    nd    = f['gtgnu'][:]
    #    time  = f['time'][:]
    #    freq  = f['freq'][:]
    #    mask  = f['mask'][:]

    #nd = np.ma.array(nd, mask=mask)
    #nd = np.ma.masked_invalid(nd)

    #_nd_t = np.ma.mean(nd, axis=1, )
    #good = (np.abs(_nd_t - np.ma.mean(_nd_t, axis=0)[None, :,:]) \
    #        - 6.*np.ma.std(_nd_t, axis=0)[None, :, :])<0

    ##time -= time[0]
    ##time /= 3600.
    #xx = time - time[0]
    #xx /= 3600.

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False, title=title)

    for bi in range(19):
        i = int(bi / 4)
        j = int(bi - i * 4)

        ax = axes[bi]

        #gt = np.ma.median(nd[:, :, :, bi], axis=1)
        #var = np.ma.var(nd[:, :, :, bi], axis=1)
        #var = np.ma.median(nd[:, :, :, bi], axis=1)
        #var[var==0] = np.inf

        gt, time, gt_smooth = destripe.get_gt(bi, file_name, gtps_file, l)

        xx = time - time[0]
        xx /= 3600.

        ax.plot(xx, gt[:, 0], 'r-', lw=0.1)
        ax.plot(xx, gt[:, 1], 'b-', lw=0.1)

        #if do_destripe:
        #    gt_m = np.ma.median(gt, axis=0)
        #    gt -= gt_m[None, :]

        #    if gtps_file is not None:
        #        #fk = min(ps_para[bi, 0, 1], 2.e-3)
        #        #alpha = min(ps_para[bi, 0, 2], 1.9)
        #        fk    = ps_para[bi, 0, 1]
        #        alpha = ps_para[bi, 0, 2]
        #        #print '%02d: XX [%e, %e]  '%(bi+1, fk, alpha),
        #    gt[:, 0] = destripe.destriping(l,
        #                                   gt[good[:, 0, bi], 0],
        #                                   var[good[:, 0, bi], 0],
        #                                   time[good[:, 0, bi]],
        #                                   fk, alpha)(time)
        #    if gtps_file is not None:
        #        #fk = min(ps_para[bi, 1, 1], 2.e-3)
        #        #alpha = min(ps_para[bi, 1, 2], 1.9)
        #        fk    = ps_para[bi, 1, 1]
        #        alpha = ps_para[bi, 1, 2]
        #        #print 'YY [%e, %e]  '%(fk, alpha)
        #    gt[:, 1] = destripe.destriping(l,
        #                                   gt[good[:, 0, bi], 1],
        #                                   var[good[:, 0, bi], 1],
        #                                   time[good[:, 0, bi]],
        #                                   fk, alpha)(time)
        #    #gt = median_filter(gt, [11, 1])

        #    gt += gt_m[None, :]

        ax.plot(xx, gt_smooth[:, 0], 'r-', lw=1)
        ax.plot(xx, gt_smooth[:, 1], 'b-', lw=1)

        #ax.legend(title='Feed %02d'%(bi+1), loc=2)
        ax.text(0.04, 0.8, 'Feed %02d'%(bi+1), transform=ax.transAxes,
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
        fig.savefig(output, format='png')
        plt.show()
        plt.clf()

def plot_baseline(baseline_file, output_name=None, axes=None, utc=True, tz=8,
        xmin=None, xmax=None):

    with h5.File(baseline_file, 'r') as f:
        baseline = f['baseline'][:]
        time     = f['time'][:]
        baseline_feed = f['baseline_feed'][:]
        #mask     = f['mask'][:]

    baseline_feed = np.ma.masked_equal(baseline_feed, 0)

    #mask_t = np.sum(mask, axis=1) / float(mask.shape[1])
    #mask_t = mask_t > 0.5
    #baseline_feed.mask += mask_t

    if utc:
        _tz = (tz * u.hour).to(u.s).value
        xx = [datetime.utcfromtimestamp(s+_tz) for s in time]
        xlabel = "UTC+%02d %s %s"%(tz, xx[0].date(), xx[0].time())
        xx = mdates.date2num(xx)
    else:
        xx = time - time[0]
        xx /= 3600.

    cnorm = mpl.colors.Normalize(vmin=0, vmax=18)

    if axes is None:
        fig = plt.figure(figsize=(10, 3))
        ax  = fig.add_axes([0.12, 0.16, 0.83, 0.79])
    else:
        fig, ax = axes

    _m = np.ma.mean(baseline_feed, axis=0)
    baseline_feed = baseline_feed - _m[None, ...]
    
    for i in range(19):
        ax.plot(xx, baseline_feed[:, 0, i], c=cm.jet(cnorm(i)), lw=0.5)
        ax.plot(xx, baseline_feed[:, 1, i], c=cm.jet(cnorm(i)), lw=0.5)
    
    ax.plot(xx, baseline, 'k-', lw=1.5)
        
    if axes is None:
        if utc:
            date_format = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('Time [hr]')
        if xmin is None:
            xmin = xx.min()
        if xmax is None:
            xmax = xx.max()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-2, 3)
        #ax.set_xlabel('Time [hr]')
        ax.set_ylabel('Baseline [K]')
        
        if output_name is not None:
            fig.savefig(output_name, format='png', dpi=200)

        plt.show()

def plot_baseline_raw(file_list, output_name=None, axes=None, utc=True,tz=8):

    baseline_list = []
    for fs in file_list:
        _baseline, time = rm_baseline._output_baseline(fs)
        baseline_list.append(_baseline[None, ...])
    baseline_list = np.ma.concatenate(baseline_list, axis=0)
    #baseline = np.ma.median(baseline_list, axis=0)

    #l = 100
    #baseline_pad = np.pad(baseline, ((l, l), (0, 0), (0, 0)), 'symmetric')
    #baseline = signal.medfilt(baseline_pad, [2*l+1, 1, 1])[l:-l]
    
    if utc:
        _tz = (tz * u.hour).to(u.s).value
        xx = [datetime.utcfromtimestamp(s+_tz) for s in time]
        xlabel = "UTC+%02d %s %s"%(tz, xx[0].date(), xx[0].time())
        xx = mdates.date2num(xx)
    else:
        xx = time - time[0]
        xx /= 3600.
    
    cnorm = mpl.colors.Normalize(vmin=0, vmax=18)
    
    if axes is None:
        fig = plt.figure(figsize=(10, 3))
        ax  = fig.add_axes([0.12, 0.16, 0.83, 0.79])
    else:
        fig, ax = axes
    
    for baseline in baseline_list:
        _m = np.ma.mean(baseline, axis=0)
        baseline -= _m[None, ...]
    
        baseline_median = np.ma.median(baseline, axis=(1, 2))
    
        for i in range(19):
            
            ax.plot(xx, baseline[:, 0, i], c=cm.jet(cnorm(i)), lw=0.1)
            ax.plot(xx, baseline[:, 1, i], c=cm.jet(cnorm(i)), lw=0.1)
        
        #plt.plot(baseline_median, 'k-', lw=0.5)
    
        l = 100
        baseline_pad = np.pad(baseline_median, l, 'symmetric')
        baseline = signal.medfilt(baseline_pad, [2*l+1,])[l:-l]
        ax.plot(xx, baseline, 'k-', lw=1.5)
        
    if axes is None:
        if utc:
            date_format = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('Time [hr]')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(-1, 3)
        #ax.set_xlabel('Time [hr]')
        ax.set_ylabel('Baseline [K]')
        
        if output_name is not None:
            fig.savefig(output_name, format='png', dpi=200)

        plt.show()
#def plot_baseline(baseline_file):
#    
#    with h5.File(baseline_file, 'r') as f:
#        baseline = f['baseline'][:]
#        time = f['time'][:]
#
#    print baseline.shape
#    
#    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)
#    
#    for bi in range(19):
#        
#        i = bi / 4
#        j = bi % 4
#        
#        xx = time - time[0]
#        xx /= 3600.
#        
#        ax = axes[bi]
#        
#        ax.plot(xx, baseline[:, 0, bi], 'r-', lw=1)
#        ax.plot(xx, baseline[:, 1, bi], 'b-', lw=1)
#        
#        ax.text(0.04, 0.8, 'Feed %02d'%bi, transform=ax.transAxes,
#                bbox=dict(facecolor='w', alpha=0.5, ec='none'))
#        ax.set_ylim(14, 26)
#        ax.set_xlim(xx[1], xx[-1])
#        #ax.semilogy()
#        if i != 4:
#            ax.set_xticklabels([])
#        else:
#            ax.set_xlabel('Time [hr]')
#        if j != 0:
#            ax.set_yticklabels([])
#        else:
#            ax.set_ylabel(r'$T$ K')

def plot_gtgnu(file_name, title='', pol=0, norm=False, output=None, ymin=None, ymax=None):
    
    with h5.File(file_name, 'r') as f:
        gtgnu = f['gtgnu'][:]
        mask  = f['mask'][:]
        time  = f['time'][:]
        freq  = f['freq'][:]

    gtgnu = np.ma.array(gtgnu, mask = mask)
        
    time -= time[0]
    time /= 3600.

    freq /= 1.e3

    if ymin is None: ymin=freq.min()
    if ymax is None: ymax=freq.max()

    fig, axes = axes_utils.setup_axes(5, 4)
    for bi in range(gtgnu.shape[3]):
        i = int(bi / 4)
        j = int(bi - i * 4)
    
        ax = axes[bi]
        
        if norm:
            _n = np.ma.mean(gtgnu[:, :, pol, bi], axis=1)
            _g = gtgnu[:, :, pol, bi].T / _n[None, :]
            vmin = 0.95
            vmax = 1.05
        else:
            _g = gtgnu[:, :, pol, bi].T
            vmin = 0.9
            vmax = 1.1
        
        im = ax.pcolormesh(time, freq, _g, vmin=vmin, vmax=vmax)
        
        ax.set_ylim(ymin, ymax)
        #ax.set_xlim(time.min(), time.max())
        ax.set_xlim(time.min(), time.max())
        
        ax.text(0.04, 0.8, 'Feed %02d'%(bi+1), transform=ax.transAxes,
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
        fig.savefig(output, format='png')
        plt.show()
        plt.clf()

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
            i = int(b / 4)
            j = int(b % 4)
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

            ax.minorticks_on()

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
    
