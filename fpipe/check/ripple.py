import numpy as np
import healpy as hp
import h5py as h5

from fpipe.timestream import data_format, destripe
from fpipe.timestream import timestream_task as tt

from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

import scipy.special as special

from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import gc

from fpipe.utils import axes_utils


def est_ps_simple(spec, freq, fmin=0.1, fmax=2.5, nbins=80, rm_baseline=False):

    try:
        mask = spec.mask
    except AttributeError:
        mask = np.zeros(spec.shape, dtype='bool')
    
    #spec_smooth = gaussian_filter1d(spec, 201)
    spec_smooth = median_filter(spec, 901)
    spec_smooth = np.ma.array(spec_smooth, mask=mask)

    norm = np.ma.median(spec)
    #print(norm)

    if rm_baseline:
        #print('rm baseline')
        d_spec = (spec - spec_smooth)/baseline
    else:
        d_spec = (spec - norm)/norm 

    d_spec[mask] = 0.

    weight = np.ones_like(d_spec)
    weight[mask] = 0.

    windowf = np.blackman(d_spec.shape[0])

    d_spec = d_spec * windowf
    weight = weight * windowf
    
    n = spec.shape[0]
    d = freq[1]-freq[0]
    ps_f = np.fft.fftfreq(n, d) #* 1.e3

    ps_raw = np.fft.fft(d_spec).real
    ps_raw /= np.sqrt(np.sum(weight))
    ps_raw *= np.sqrt(d * 1.e6)
    ps_raw = ps_raw ** 2.

    #norm = np.sum(np.ones_like(d_spec))
    #ps_raw /= norm
    
    ff_bin_edge = np.linspace(fmin, fmax, nbins + 1)
    ff_bin_cent = ff_bin_edge[:-1] + 0.5 * (ff_bin_edge[1:] - ff_bin_edge[:-1])
    
    norm = np.histogram(ps_f, bins=ff_bin_edge)[0] * 1.0
    ps_bin = np.histogram(ps_f, bins=ff_bin_edge, weights=ps_raw)[0]
    norm[norm==0] = np.inf
    ps_bin /= norm

    ps_er = ps_bin / (norm ** 0.5)
    
    return ps_bin, ff_bin_cent, ps_er

def est_ps_bandpass(file_root, file_name, beam=1, pol=0, st=0, ed=1, rm_baseline=False):
    
    with h5.File(file_root + 'bandpass/%s/bandpass_%s.h5'%tuple(file_name.split('/')), 'r') as fp:
        bandpass = fp['bandpass'][st:ed, beam-1, :, pol]
        bandpass_freq = fp['freq'][:]

        freq_sel = bandpass_freq > 1250
        bandpass_freq = bandpass_freq[freq_sel]
        bandpass = bandpass[:, freq_sel]

    bp = np.mean(bandpass, axis=0)
    ps_bp, ff_bp, ps_er = est_ps_simple(bp, bandpass_freq, rm_baseline=rm_baseline)
    
    return ps_bp, ff_bp, ps_er

def est_ps_ts(file_root, file_name, prefix='bpcal', mask=None, beam=1, pol=0, st=0, ed=1, 
        rm_baseline=False):

    file_list = [file_root + '%s/%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(prefix, file_name, i, i)
                 for i in range(st, ed)]
    fdata_bpcal = tt.FAST_Timestream(file_list) #, fmin=1600*3, fmax=1600*9)
    fdata_bpcal.load_all()
    vis_bpcal = fdata_bpcal.vis.local_data[:, :, pol, beam-1]
    vis_bpcal = vis_bpcal[:]
    vis_bpcal = np.ma.array(vis_bpcal, mask=False)
    if mask is None:
        mask = fdata_bpcal.vis_mask.local_data[:, :, pol, beam-1]
    time_numb = vis_bpcal.shape[0]
    freq_mask = np.sum(mask.astype('int'), axis=0) > 0.6 * time_numb

    freq_bpcal = fdata_bpcal.freq[:]
    del fdata_bpcal
    gc.collect()

    spec_bpcal = np.ma.mean(vis_bpcal,  axis=0)
    spec_bpcal.mask = freq_mask
    ps_bpcal, ff_bpcal, ps_er= est_ps_simple(spec_bpcal, freq_bpcal, rm_baseline=rm_baseline)

    del vis_bpcal, spec_bpcal, freq_bpcal
    gc.collect()

    return ps_bpcal, ff_bpcal, ps_er

def load_mask(file_root, file_name, st, ed, beam=1, pol=0, band='1250-1450MHz'):

    file_list = [file_root + 'rf_sumfeeds/%s_arcdrift%04d-%04d_%s.h5'%(file_name, i, i, band)
                 for i in range(st, ed)]
    fdata_rf = tt.FAST_Timestream(file_list) #, fmin=1600*3, fmax=1600*9)
    fdata_rf.load_all()
    mask = fdata_rf.vis_mask[:, :, pol, beam-1]
    nt = mask.shape[0]

    return mask, nt

#for file_name in file_name_list:
def est_ps_combine(file_root, file_name, st = 1, n_file = 2, beam=1, pol=0):
    
    ed = st + n_file
    print("%s [%02d - %02d]"%(file_name, st, ed-1))

    file_list = [file_root + 'rf_sumfeeds/%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(file_name, i, i)
                 for i in range(st, ed)]
    fdata_rf = tt.FAST_Timestream(file_list) #, fmin=1600*3, fmax=1600*9)
    fdata_rf.load_all()
    mask = fdata_rf.vis_mask[:, :, pol, beam-1]
    nt = mask.shape[0]
    del fdata_rf
    gc.collect()
    

    file_list = [file_root + 'raw/%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(file_name, i, i)
                 for i in range(st, ed)]
    fdata_raw = tt.FAST_Timestream(file_list) #, fmin=1600*3, fmax=1600*9)
    fdata_raw.load_all()
    vis_raw = fdata_raw.vis.local_data[:, :, pol, beam-1]
    vis_raw = np.ma.array(vis_raw, mask=False)
    vis_raw.mask[:nt, :] = mask
    vis_raw.mask[nt:] = True
    freq_raw = fdata_raw.freq[:]
    del fdata_raw
    gc.collect()
    
    spec_raw = np.ma.mean(vis_raw, axis=0)
    ps_raw, ff_raw, ps_raw_er = est_ps_simple(spec_raw, freq_raw)
    
    del vis_raw, spec_raw, freq_raw
    gc.collect()
    

    #file_list = [file_root + 'bpcal/%s_arcdrift%04d-%04d_1250-1450MHz.h5'%(file_name, i, i)
    #             for i in range(st, ed)]
    #fdata_bpcal = tt.FAST_Timestream(file_list) #, fmin=1600*3, fmax=1600*9)
    #fdata_bpcal.load_all()
    #vis_bpcal = fdata_bpcal.vis.local_data[:, :, pol, beam-1]
    #vis_bpcal = np.ma.array(vis_bpcal, mask=False)
    #vis_bpcal.mask[:] = mask
    #freq_bpcal = fdata_bpcal.freq[:]
    #del fdata_bpcal
    #gc.collect()

    ps_bpcal, ff_bpcal = est_ps_ts(file_root, file_name, prefix='bpcal', 
            mask=mask, beam=beam, pol=pol, st=st, ed=ed, rm_baseline=False)
    
    spec_bpcal = np.ma.mean(vis_bpcal,  axis=0)
    ps_bpcal, ff_bpcal, ps_bpcal_er = est_ps_simple(spec_bpcal, freq_bpcal)
    
    del vis_bpcal, spec_bpcal, freq_bpcal
    gc.collect()
    
    
    #with h5.File(file_root + 'bandpass/%s/bandpass_%s.h5'%tuple(file_name.split('/')), 'r') as fp:
    #    bandpass = fp['bandpass'][st:ed, beam-1, :, pol]
    #    bandpass_freq = fp['freq'][:]
    #    
    #    freq_sel = bandpass_freq > 1250
    #    bandpass_freq = bandpass_freq[freq_sel]
    #    bandpass = bandpass[:, freq_sel]

    #bp = np.mean(bandpass, axis=0)
    ps_bp, ff_bp, ps_bp_er = est_ps_bandpass(file_root, file_name, beam, pol, st-1, ed-1)
    
    del bandpass, bp, bandpass_freq
    
    return (ps_raw, ff_raw), (ps_bpcal, ff_bpcal), (ps_bp, ff_bp)

def plot_ps(r, fig_axes=None, vmin=5.e-1, vmax=5.e3, fmin=0, fmax=2.5, title=''):
    
    (ps_raw, ff_raw), (ps_bpcal, ff_bpcal), (ps_bp, ff_bp) = r
    
    if fig_axes is None:
        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(hspace=0.1, wspace=0.08)
        axes = [fig.add_subplot(1,3,1),
                fig.add_subplot(1,3,2),
                fig.add_subplot(1,3,3)]
    else:
        fig, axes = fig_axes
    
    ax = axes[0]
    #ax  = fig.add_subplot(1,3,1)
    norm = ps_raw[ff_raw > 0][0]
    ax.plot(ff_raw[ff_raw > 0], ps_raw[ff_raw > 0]/norm, '.-', lw=2)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(vmin, vmax)
    ax.semilogy()
    #ax.set_xticklabels([])
    #ax.grid(which='both', axis='x')
    ax.axvline(1./1.1 , 0, 1, c='k', ls='--')
    ax.set_ylabel(r'$|\tilde{V}|^2$')
    ax.set_xlabel(r'$\tau ~[{\rm \mu s}]$')
    ax.set_title('Raw TOD')
    
    ax = axes[1]
    #ax  = fig.add_subplot(1,3,2)
    norm = ps_bp[ff_bp > 0][0]
    ax.plot(ff_bp[ff_bp > 0], ps_bp[ff_bp > 0]/norm, '.-', lw=2, label=title)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(vmin, vmax)
    ax.semilogy()
    #ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.grid(which='both', axis='x')
    ax.axvline(1./1.1 , 0, 1, c='k', ls='--')
    #ax.set_ylabel('Bandpass')
    ax.set_xlabel(r'$\tau ~[{\rm \mu s}]$')
    ax.set_title('Bandpass')
    
    ax = axes[2]
    #ax  = fig.add_subplot(1,3,3)
    norm = ps_bpcal[ff_bpcal > 0][0]
    ax.plot(ff_bpcal[ff_bpcal > 0], ps_bpcal[ff_bpcal > 0]/norm, '.-', lw=2)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(vmin, vmax)
    ax.semilogy()
    #ax.set_xticklabels([])
    #ax.grid(which='both', axis='x')
    ax.set_yticklabels([])
    ax.axvline(1./1.1 , 0, 1, c='k', ls='--')
    #ax.set_ylabel('After Bandpass Cal.')
    ax.set_xlabel(r'$\tau ~[{\rm \mu s}]$')
    ax.set_title('After bandpass calibration')
    
    #fig.suptitle(title)
    
    return fig, axes
