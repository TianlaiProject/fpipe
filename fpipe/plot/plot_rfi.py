from fpipe.container.timestream import FAST_Timestream
import numpy as np
import h5py as h5

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatchs
import matplotlib.lines as mlines

from astropy.time import Time

import gc

def output_rfi_hist(file_list, fbins=[1050, 1140, 1310, 1450], output=None, 
        bad_feed_No = None):
    
    nbin = 100
    bins = np.logspace(-2, 5, nbin+1)
    
    hist = np.zeros((len(fbins)-1, nbin))
    hist_nomask = np.zeros((len(fbins)-1, nbin))
    
    for ii, fblock in enumerate(file_list):
        #hist = np.zeros(nbin)
        for jj, tblock in enumerate(fblock):
            ts = FAST_Timestream(tblock)
            ts.load_all()

            try:
                on = np.sum(ts['ns_on'][:].local_array, axis=1)
                vis = ts['vis'][~on,...].local_array
                vis_mask = ts['vis_mask'][~on,...].local_array
                #time = ts['sec1970'][:].local_array
                freq = ts['freq'][:]
            except AttributeError:
                on = np.sum(ts['ns_on'], axis=1)
                vis = ts['vis'][~on,:,:,:]
                vis_mask = ts['vis_mask'][~on,:,:,:]
                #time = ts['sec1970'][:]
                freq = ts['freq'][:]

            vis_mask = vis_mask.astype('bool')

            if bad_feed_No is not None:
                feed_sel = np.ones(vis.shape[-1], dtype='bool')
                for bf in bad_feed_No:
                    feed_sel[bf-1] = False
                vis = vis[..., feed_sel]
                vis_mask = vis_mask[..., feed_sel]

            freq = freq[1:]

            msk = vis_mask[:, 1:, ...] + vis_mask[:, :-1, ...]
            vis_diff = np.abs(vis[:, 1:, ...] - vis[:, :-1, ...])
            print(np.ma.min(vis_diff, axis=(0, 1)))
            print(np.ma.max(vis_diff, axis=(0, 1)))
            print()
            for jj in range(hist.shape[0]): 
                freq_sel = (freq > fbins[jj]) * (freq < fbins[jj+1])
                _vis_diff = vis_diff[:, freq_sel, ...].flatten()
                hist_nomask[jj] += np.histogram(_vis_diff, bins=bins)[0]
                _msk = msk[:, freq_sel, ...].flatten()
                _vis_diff[_msk] = -1
                hist[jj] += np.histogram(_vis_diff, bins=bins)[0]
                del _vis_diff, _msk

            del vis, ts, vis_diff, msk
            gc.collect()
        
    if output is not None:
        with h5.File(output, 'w') as f:
            f['hist'] = hist
            f['hist_nomask'] = hist_nomask
            f['fbin'] = np.array(fbins)
            f['Tbin'] = bins

def plot_rfi_hist(hist, bins, axes=None, label_list = None, output=None, ls='-',
                  fit=False, dt=1, df=0.00762939453125e6):
    #c_list = list(mcolors.XKCD_COLORS)
    c_list = list(mcolors.TABLEAU_COLORS)
    if axes is None:
        fig = plt.figure(figsize=(6, 4))
        ax  = fig.add_axes([0.12, 0.15, 0.83, 0.80])
    else:
        fig, ax = axes
    
    legend_list = []
    norm_list = []
    for ii in range(hist.shape[0]):

        label = None
        if label_list is not None:
            label = label_list[ii]
        _hist = hist[ii] / (bins[1:] - bins[:-1])
        norm = np.sum(hist[ii])
        #_hist /= norm
        #print norm
        norm_list.append(norm)
        ax.plot(bins[1:], np.log10(_hist), color=c_list[ii], ls=ls)
                #drawstyle='steps-pre')
        legend_list.append(mpatchs.Patch(edgecolor=c_list[ii], facecolor=c_list[ii],
                                         label=label))
        if fit:
            sel = bins[1:] < 1.5
            xx = bins[1:][sel]
            yy = np.log(_hist[sel])
            
            xx_fit = np.logspace(-2, 1, 50)
            fit_params = np.polyfit(xx, yy, 2)
            print(fit_params, end=' ')
            c, b, a = fit_params
            sigma2 = -1./c/2.
            rms = ( sigma2 / 2. ) ** 0.5
            Tsys = rms * (dt * df)**0.5
            print(rms, Tsys)
            yy_fit = np.exp(np.poly1d(fit_params)(xx_fit))
            
            ax.plot(xx_fit, np.log10(yy_fit), 'o-', mfc='w', mec=c_list[ii], mew=0.5, 
                    ms=3, lw=0.8)

    if axes is None:
        ax.semilogx()
        #ax.loglog()
        ax.set_ylabel(r'$\ln ({\rm d}\,N/{\rm d}\,T)$')
        ax.set_xlabel(r'$T\,[{\rm K}]$')
    
        #ax.set_xlim(-5e3, 5e3)
        ax.set_ylim(ymin=0)
        ax.legend()

    if output is not None:
        fig.savefig(output, dpi=200)
        
    return legend_list, norm_list
        
        
def plot_rfi_hist_multi(hist_file, output=None):
    
    ls = ['-', '--']
    
    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_axes([0.12, 0.15, 0.83, 0.80])
    
    with h5.File(hist_file, 'r') as f:
        hist = f['hist'][:]
        hist_nomask = f['hist_nomask'][:]
        fbins = f['fbin'][:]
        bins = f['Tbin'][:]
        
    label_list = ['%d MHz - %d MHz'%(fbins[i], fbins[i+1]) for i in range(len(fbins)-1)]
    leg1, N1 = plot_rfi_hist(hist_nomask, bins, axes=(fig, ax), label_list = label_list)
    leg2, N2 = plot_rfi_hist(hist, bins, axes=(fig, ax), ls='--', fit=True)
    
    N1 = np.array(N1)
    N2 = np.array(N2)
    mask_ratio = 1. - N2 / N1
    print('Mask Ratio: ', mask_ratio)
    
    
    ax.semilogx()
    #ax.loglog()
    ax.set_ylabel(r'$\log ({\rm d}\,N/{\rm d}\,T)$')
    ax.set_xlabel(r'$T\,[{\rm K}]$')
    
    ax.set_xlim(2e-2, 1e4)
    ax.set_ylim(ymin=0, ymax=11)
    #ax.set_ylim(ymin=1.e-10, ymax=10)
    
    leg1 = ax.legend(handles=leg1, frameon=False, markerfirst=True, loc=1)
    leg2 = [mlines.Line2D([], [], color='k', ls='-', label='Before RFI Flagging'),
            mlines.Line2D([], [], color='k', ls='--', label='After RFI Flagging'),
            mlines.Line2D([], [], color='k', ls='-', marker='o', ms=3, mew=0.5, mfc='w',
                         lw=0.8, label='Gaussian Fit')]
    leg2 = ax.legend(handles=leg2, frameon=False, markerfirst=True, loc=3)
    ax.add_artist(leg1)
    
    if output is not None:
        fig.savefig(output, dpi=200)
