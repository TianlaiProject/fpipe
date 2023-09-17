from fpipe.plot import plot_map as pm
from fpipe.plot import plot_spec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h5

import gc

import warnings

from scipy.optimize import leastsq

from tqdm.autonotebook import tqdm


def profile_fit(ax, h, bc, be, freq_diff='ABBA', bc_min=-1, bc_max=1, threshold=0.5):
    '''
    fit histogram with Gaussian profile
    '''
    
    h = np.ma.masked_invalid(h)
    #good = ~h.mask
    h = np.ma.filled(h, 0)

    if freq_diff is not None:
        good = h > threshold * np.ma.max(h)
        #good = (bc<bc_max) * (bc>bc_min) * (h>0)
    else:
        good = (bc < bc_max) * (h>0)

    if not np.any(good):
        return 0, 0

    _h = -2 * np.log(h[good])
    _bc= bc[good]
    param = np.polyfit(_bc, _h, 2)
    h_fit = np.exp( -0.5 * np.poly1d( param )(bc) )
    
    sigma = np.sqrt(1/param[0])
    b = param[1]/2./param[0]
    A = np.exp(-0.5 * (param[2] - (param[1]**2)/4./param[0]))

    if ax is None:
        return sigma, b

    #label = 'sigma=%f[K], b=%f[K], A=%f'%(sigma, b, A)
    if freq_diff == "AB":
        label = r'$\sigma/\sqrt{2}=%4.3f~[{\rm mK}]$'%(sigma/(2.**0.5) * 1.e3) \
              + '\n' + r'$\mu_0=%4.3f~[{\rm mK}]$'%(b * 1.e3)
    else:
        label = r'$\sigma = %4.3f~[{\rm mK}]$'%(sigma * 1.e3) \
              + '\n' + r'$\mu_0 = %4.3f~[{\rm mK}]$'%(b * 1.e3)

    l = ax.plot(bc, h, 'g-', lw=2, drawstyle='steps-mid')[0]
    #plt.plot(bc, h_fit, 'g-')
    ax.plot(bc, A * np.exp(-0.5 * (bc + b)**2/sigma**2), 'r--',
            lw=2, label=label)
    #plt.plot(bc, h, 'r.')
    ax.semilogy()

def check_tsys(ax, m, nfreq, bin_edge=None, freq_diff=True, bc_min=-1, bc_max=1):

    _m = m.copy()
    _m = _m.flatten()
    mask = _m == 0

    if bin_edge is None:
        bs = 0.01
        be = np.arange(-1, 2, bs)
    else:
        be = bin_edge
        bs = be[1] - be[0]
    
    h = np.histogram(_m[~mask], bins=be)[0] #/ nfreq
    bc = be[:-1] + bs*0.5
    
    del _m, mask
    gc.collect()
    
    if ax is None:
        return h, bc, be

    profile_fit(ax, h, bc, be, freq_diff, bc_min, bc_max)


def rms_AB_tod(m, mask):

    freq_l = int(m.shape[1]//2 * 2)
    m      = m[:, :freq_l, :]
    mask   = mask[:, :freq_l, :]
    shp = m.shape
    m.shape = ( shp[0], -1, 2, shp[-1])
    m = m[:, :, 0, :] - m[:, :, 1, :]
    
    mask.shape = ( shp[0], -1, 2, shp[-1] )
    mask = mask[:, :, 0, :] + mask[:, :, 1, :]
    
    #print(m.shape)
    m[mask] = 0
    return m, mask

def rms_ABBA_tod(m, mask):
    
    freq_l = int(m.shape[1]//4 * 4)
    m      = m[:, :freq_l, :]
    mask   = mask[:, :freq_l, :]
    shp = m.shape
    m.shape = (shp[0], -1, 4, shp[-1])
    m = 0.5 * (m[:, :, 1, :] + m[:, :, 2, :]) - 0.5 * (m[:, :, 0, :] + m[:, :, 3, :])
    
    mask.shape = ( shp[0], -1, 4, shp[-1] )
    mask = mask[:, :, 0, :] + mask[:, :, 1, :] + mask[:, :, 2, :] + mask[:, :, 3, :]
    
    #print m.shape
    m[mask] = 0
    return m, mask

def rms_ABAB_tod(m, mask):
    
    freq_l = int(m.shape[1]//4 * 4)
    m      = m[:, :freq_l, :]
    mask   = mask[:, :freq_l, :]
    shp = m.shape
    m.shape = (shp[0], -1, 4, shp[-1])
    #m = 0.5 * (m[:, :, 1, :] + m[:, :, 2, :]) - 0.5 * (m[:, :, 0, :] + m[:, :, 3, :])
    m = 0.5 * (m[:, :, 0, :] + m[:, :, 2, :]) - 0.5 * (m[:, :, 1, :] + m[:, :, 3, :])
    
    mask.shape = ( shp[0], -1, 4, shp[-1] )
    mask = mask[:, :, 0, :] + mask[:, :, 1, :] + mask[:, :, 2, :] + mask[:, :, 3, :]
    
    #print m.shape
    m[mask] = 0
    return m, mask

def rms_ABBA_map(m, mask):

    freq_l = int(m.shape[0]//4 * 4)
    m      = m[:freq_l, :]
    mask   = mask[:freq_l, :]
    shp = m.shape
    m.shape = (-1, 4, shp[-1])
    m = 0.5 * ( m[:, 1, :] + m[:, 2, :] ) - 0.5 * ( m[:, 0, :] + m[:, 3, :] )

    mask.shape = (-1, 4, shp[-1] )
    mask = mask[:, 0, :] + mask[:, 1, :] + mask[:, 2, :] + mask[:, 3, :]
    
    #print m.shape
    m[mask] = 0
    return m, mask

def rms_ABAB_map(m, mask):

    freq_l = int(m.shape[0]//4 * 4)
    m      = m[:freq_l, :]
    mask   = mask[:freq_l, :]
    shp = m.shape
    m.shape = (-1, 4, shp[-1])
    m = 0.5 * ( m[:, 0, :] + m[:, 2, :] ) - 0.5 * ( m[:, 1, :] + m[:, 3, :] )

    mask.shape = (-1, 4, shp[-1] )
    mask = mask[:, 0, :] + mask[:, 1, :] + mask[:, 2, :] + mask[:, 3, :]
    
    #print m.shape
    m[mask] = 0
    return m, mask

def iter_tod_files(tod_path, tod_suffix_list, tod_cent_list, tod_freq, freq_sel):

    tod_tmp = '%s_%s_%s.h5'
    for jj, tod_suffix in enumerate(tod_suffix_list):
        for ii, tod_cent in enumerate(tod_cent_list):
            tod_name = tod_tmp%(tod_cent, tod_suffix, tod_freq)

            with h5.File(tod_path + tod_name, 'r') as f:
                m = f['vis'][:]
                mask = f['vis_mask'][:].astype('bool')
                ns_on = f['ns_on'][:].astype('bool')
                freq = f['freq'][:]

            freq = freq[freq_sel]
            #print(freq[0], freq[-1])

            m    = m[:, freq_sel, :, :]
            mask = mask[:, freq_sel, :, :]

            mask += ns_on[:, None, None, :]

            m[mask] = 0
            m = np.sum(m, axis=2) / 2.
            #mask = m == 0
            mask = mask[:, :, 0, :] + mask[:, :, 1, :]

            yield jj, ii, m, mask

def check_tsys_tod(tod_path, tod_suffix_list, tod_cent_list, tod_freq, freq_diff, 
                     freq_sel = slice(0, None), per_freq=False):
    
    if freq_diff is not None:
        bin_edges = np.linspace(-0.6, 0.6, 100)
    else:
        bin_edges = np.arange(-1, 2, 0.01)
        
    
    N = len(tod_cent_list) * len(tod_suffix_list)
    results = []
    _oo = tqdm(iter_tod_files(tod_path, tod_suffix_list, tod_cent_list, tod_freq, freq_sel),
            colour='green', desc='Main', total=N)
    for jj, ii, m, mask in _oo:

        #freq_mask = np.sum(mask.astype('int'), axis=0) > 0.3 * mask.shape[0]
        #print(np.sum(freq_mask.astype('int'), axis=0) / freq_mask.shape[0])
        #mask += freq_mask[None, ...]

        if freq_diff is not None:
            m, mask = globals()['rms_%s_tod'%freq_diff](m, mask)

        m[mask] = 0
        nfreq = float(m.shape[1])

        #for beam_id in range(19):
        for beam_id in tqdm(range(19), colour='blue', desc='Beam'):
            if per_freq:
                hist = []
                for ff in tqdm(np.arange(m.shape[1]), colour='black', desc='Freq'):
                    h, bc, be = check_tsys(None, m[:, ff, beam_id], 1., bin_edges)
                    hist.append(h[None, :])
                hist = np.concatenate(hist, axis=0)
                results.append(hist[None, :, :])

            else:
                h, bc, be = check_tsys(None, m[:, :, beam_id], nfreq, bin_edges)
                results.append(h[None, :])

    results = np.concatenate(results, axis=0)
    results.shape = (len(tod_suffix_list), len(tod_cent_list), 19) + results.shape[1:]

    return results, bc, be

def check_tsys_map(map_path, map_name_list, freq_diff='ABBA', freq_sel=slice(0, None), 
        freq_mask=None, per_freq=False):

    if freq_diff is not None:
        bin_edges = np.linspace(-0.49, 0.49, 100)
    else:
        bin_edges = np.arange(-1, 2, 0.01)

    results = []
    #for map_name in map_name_list:
    for ii in tqdm(range(len(map_name_list)), colour='green', desc='Main'):
        map_name = map_name_list[ii]
        m, pixs, freq, mask = pm.load_maps_hp(map_path, map_name, 'clean_map')
        m = m[freq_sel, ...]
        freq = freq[freq_sel]
        #print(freq[0], freq[-1])

        if freq_mask is not None:
            m[freq_mask] = 0

        mask = m == 0

        if freq_diff is not None:
            m, mask = globals()['rms_%s_map'%freq_diff](m, mask)

        if per_freq:
            hist = []
            #for ff in np.arange(m.shape[0]):
            for ff in tqdm(np.arange(m.shape[0]), colour='blue', desc='Freq'):
                h, bc, be = check_tsys(None, m[ff], 1., bin_edges)
                hist.append(h[None, :])
            hist = np.concatenate(hist, axis=0)
            results.append(hist[None, :, :])
        else:
            nfreq = float(m.shape[0])
            h, bc, be = check_tsys(None, m, nfreq, bin_edges)
            results.append(h[None, :])

    results = np.concatenate(results, axis=0)
    return results, bc, be

def plot_tsys_tod_blocks(results, bc, be, freq_diff, figsize=(6, 8), ylabel_list=None, 
        title_list=None, beam_id=0):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(results.shape[0], results.shape[1], left=0.055, bottom=0.045,
                           right=0.98, top=0.95, wspace=0.02, hspace=0.02)

    if ylabel_list is None:
        ylabel_list = ['N', ] * len(results.shape[0])
    if title_list is None:
        title_list = ['%02d'%i for i in range(results.shape[1])]
        
    for jj in range(results.shape[0]):
        for ii in range(results.shape[1]):

            ax = fig.add_subplot(gs[jj, ii])

            h = results[jj, ii, beam_id]
            profile_fit(ax, h/np.sum(h), bc, be, freq_diff)

            if ii == 0:
                ax.set_ylabel(ylabel_list[jj])
            else:
                ax.set_yticklabels([])
            if jj == results.shape[0] - 1:
                ax.set_xlabel('T [K]')
            elif jj == 0:
                ax.set_title(title_list[ii])
            else:
                ax.set_xticklabels([])

            ax.set_xlim(-0.6, 0.6)
            ax.legend(loc=1) 
            ax.set_ylim(ymin=2.e-4, ymax=2.e-1)

def plot_tsys_tod_combine(results, bc, be, freq_diff, figsize=(6, 8), ylabel_list=None, bad_beam=None):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_axes([0.12, 0.12, 0.83, 0.83])

    hist_tot = []
    for jj in range(results.shape[0]):
        for ii in range(results.shape[1]):
            for bb in range(results.shape[2]):
                if bad_beam is not None and bb+1 in bad_beam[jj]:
                    continue
                hist = results[jj, ii, bb]
                hist_tot.append(hist)
                ax.plot(bc, hist / np.sum(hist * 1.), color='0.7', lw=0.5, drawstyle='steps-mid')

    hist_tot = np.array(hist_tot)
    hist_tot = np.sum(hist_tot, axis=0)
    ax.plot(bc, hist_tot / np.sum(hist_tot * 1.), 'k-', lw=2, drawstyle='steps-mid')

    profile_fit(ax, hist_tot / np.sum(hist_tot * 1.), bc, be, freq_diff=freq_diff, 
                bc_min=-0.3, bc_max=0.3)
    
    ax.legend()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(2.e-4, 1.e-1)
    #ax.tick_params(axis='y', labelrotation= 90)
    ax.set_xlabel('T [K]')
    ax.set_ylabel('N')
    ax.semilogy()
    #fig.savefig('output/map_rms_hist_combine.png', dpi=200)


def plot_rms_hist_low_and_high_band(hist_l, hist_h, bc_l, bc_h, be_l, be_h, freq_diff='ABBA',
        xlim=(-0.19, 0.19), ylim=(2.e-4, 1.5e-1), output_name='test.png'):
    
    fig = plt.figure(figsize=(6, 4))
    ax_l  = fig.add_axes([0.12, 0.12, 0.41, 0.83])
    ax_h  = fig.add_axes([0.54, 0.12, 0.41, 0.83])
    
    hist_l = np.ma.masked_invalid(hist_l)
    hist_h = np.ma.masked_invalid(hist_h)
    
    r_l_list = []
    for i in range(hist_l.shape[0]):
        if np.ma.sum(hist_l[i] * 1.) == 0:
            continue
        ax_l.plot(bc_l, hist_l[i] / np.ma.sum(hist_l[i] * 1.), color='0.7', lw=0.5, drawstyle='steps-mid')
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                r_l_list.append( profile_fit(None, hist_l[i] / np.ma.sum(hist_l[i] * 1.), bc_l, be_l) )
            except:
                r_l_list.append([0, 0])
            
    hist_tot = np.sum(hist_l, axis=0)
    profile_fit(ax_l, hist_tot / np.ma.sum(hist_tot * 1.), bc_l, be_l, freq_diff=freq_diff, 
                bc_min=-0.04, bc_max=0.04)
    
    r_h_list = []
    for i in range(hist_h.shape[0]):
        if np.ma.sum(hist_h[i] * 1.) == 0:
            continue
        ax_h.plot(bc_h, hist_h[i] / np.ma.sum(hist_h[i] * 1.), color='0.7', lw=0.5, drawstyle='steps-mid')
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                r_h_list.append(profile_fit(None, hist_h[i] / np.ma.sum(hist_h[i] * 1.),  bc_h, be_h))
            except:
                r_h_list.append([0, 0])
                
    hist_tot = np.ma.sum(hist_h, axis=0)
    profile_fit(ax_h, hist_tot / np.ma.sum(hist_tot * 1.), bc_h, be_h, freq_diff=freq_diff, 
                bc_min=-0.04, bc_max=0.04)
    
    ax_l.legend(title='1050-1150 MHz')
    ax_l.set_xlim(*xlim)
    ax_l.set_ylim(*ylim)
    ax_l.set_xlabel(r'$T\,[{\rm K}]$')
    ax_l.set_ylabel('N')
    ax_l.semilogy()
    
    
    ax_h.semilogy()
    ax_h.legend(title='1323-1450 MHz')
    ax_h.set_yticklabels([])
    ax_h.set_xlabel(r'$T\,[{\rm K}]$')
    ax_h.set_xlim(*xlim)
    ax_h.set_ylim(*ylim)
    
    r_l_list = np.array(r_l_list)
    r_h_list = np.array(r_h_list)
    
    if output_name is not None:
        fig.savefig(output_name, dpi=200)
    return r_l_list, r_h_list

def plot_rms_beam(rms_low, rms_hig):

    x= range(1, 20)
    fig = plt.figure(figsize=[8, 4])
    ax  = fig.add_axes([0.1, 0.12, 0.85, 0.82])
    ax.plot(x, rms_low, 'ro--', label='1050-1150 MHz')
    ax.plot(x, rms_hig, 'bo--', label='1323-1450 MHz')
    
    ax.minorticks_off()
    ax.set_xticks(range(1, 20))
    ax.set_xticklabels(['%02d'%i for i in range(1, 20)])
    
    ax.axvline(1.5, 0, 1, color='0.5', ls='-')
    ax.axvline(7.5, 0, 1, color='0.5', ls='-')
    
    ax.hlines(np.mean(rms_low[:1]), 0.5, 1.5, color='r', ls='-')
    ax.hlines(np.mean(rms_low[1:8]), 1.5, 7.5, color='r', ls='-')
    ax.hlines(np.mean(rms_low[8:]), 7.5, 19.5, color='r', ls='-')
    
    ax.hlines(np.mean(rms_hig[:1]), 0.5, 1.5, color='b', ls='-')
    ax.hlines(np.mean(rms_hig[1:8]), 1.5, 7.5, color='b', ls='-')
    ax.hlines(np.mean(rms_hig[8:]), 7.5, 19.5, color='b', ls='-')
    
    ax.set_xlim(0.5, 19.5)
    ax.set_ylim(0, 150)
    
    
    ax.set_ylabel(r'${\rm r.m.s}\, [{\rm mK}]$')
    ax.set_xlabel('Feed \#')
    
    ax.legend()
    
    #fig.savefig('sigma_beams.png', dpi=150)
    fig.savefig('plot/plots_sigma_beams.pdf')


