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

        ax.legend(title='Feed %02d'%(bi+1), loc=2)
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
