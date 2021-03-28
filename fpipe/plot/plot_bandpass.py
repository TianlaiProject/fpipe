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


import matplotlib.pyplot as plt


def plot_gt(file_name, title=''):

    with h5.File(file_name, 'r') as f:
        nd    = f['gtgnu'][:]
        time  = f['time'][:]
        freq  = f['freq'][:]

    nd = np.ma.masked_invalid(nd)

    time -= time[0]
    time /= 3600.

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    for bi in range(19):
        i = bi / 4
        j = bi - i * 4
        
        ax = axes[bi]
        gt = np.ma.median(nd[:, :, :, bi], axis=1)
        gt = median_filter(gt, [11, 1])
        ax.plot(time, gt[:, 0], 'r-', lw=0.5)
        ax.plot(time, gt[:, 1], 'b-', lw=0.5)
        ax.legend(title='Feed %02d'%(bi+1), loc=2)
        ax.set_ylim(0.81, 1.19)
        ax.set_xlim(time[1], time[-1])
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
