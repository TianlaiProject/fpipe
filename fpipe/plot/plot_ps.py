'''
functions for plotting ps results

'''
from collections import OrderedDict as dict

import h5py as h5
import numpy as np
import os


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fpipe.sim import fisher

from fpipe.utils import binning
from fpipe.map import algebra as al

from fpipe.ps import pwrspec_estimator as pet

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def iter_ps_list(ps_path, ps_name_list, tr_path=None, cross=True, 
        to1d=False, from2d=True):
    ps_keys = ps_name_list.keys()
    for ii in range(len(ps_keys)):
        ps_name = ps_name_list[ps_keys[ii]]
        if tr_path is not None:
            ps_name, tr_name = ps_name
            if tr_name is not None:
                tr_ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
            else:
                tr_path = None

        with h5.File(ps_path + ps_name, 'r') as f:
            
            x = f['kbin_x_edges'][:]
            y = f['kbin_y_edges'][:]

            xlim = [x.min(), x.max()]
            ylim = [y.min(), y.max()]

            ps2d = np.ma.masked_equal(f['binavg_2d'][:],   0)
            ct2d = np.ma.masked_equal(f['counts_2d'][0, :],0)
            ps1d = np.ma.masked_equal(f['binavg_1d'][:,],  0)
            ct1d = np.ma.masked_equal(f['counts_1d'][:,],  0)

            good = f['kbin_x'][:] < 0.12
            ct2d[~good, ...] = 0.

            if cross:
                ps2d *= 1.e3
                ps1d *= 1.e3
            else:
                ps2d *= 1.e6
                ps1d *= 1.e6

            if tr_path is not None:
                tr = load_2dtr_from3d(ps_path, tr_name, tr_ref_name, 
                        kbin_x_edges=x, kbin_y_edges=y)[0]
                if not cross: tr  **= 2.
                ps2d *= tr

            if to1d:
                ps1d_kbin_edges = f['kbin_edges'][:]
                ps1d_kbin = f['kbin'][:]
                if not from2d:
                    yield ii, ps_keys[ii], ps1d, ct1d, ps1d_kbin
                else:
                    ps1d = []
                    ct1d = []
                    for i in range(ps2d.shape[0]):
                        counts_1d, gaussian_error, binavg_1d = \
                                pet.convert_2d_to_1d_pwrspec(ps2d[i], ct2d, 
                                        x, y, ps1d_kbin_edges)
                        ps1d.append(binavg_1d)
                        ct1d.append(counts_1d)
                    ps1d = np.array(ps1d)
                    ct1d = np.array(ct1d)
                    yield ii, ps_keys[ii], ps1d, ct1d, ps1d_kbin
            else:
                yield ii, ps_keys[ii], ps2d, ct2d, x, y

def _add_axes_1d(figsize, plot_null=False):

    fig = plt.figure(figsize=figsize)
    if plot_null:
        ax = fig.add_axes([0.09, 0.32, 0.88, 0.80])
        ax_null = fig.add_axes([0.09, 0.12, 0.88, 0.2])
    else:
        ax = fig.add_axes([0.09, 0.12, 0.88, 0.80])
        ax_null = None

    return fig, ax, ax_null

def _add_axes_2d(figsize, ps_name_list, logk=True):

    ps_keys = ps_name_list.keys()
    ncol = len(ps_keys)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, ncol, left=0.08, bottom=0.1, top=0.9, right=0.90,
            figure=fig, wspace=0.1)
    cax = fig.add_axes([0.91, 0.2, 0.1/float(figsize[0]), 0.6])
    axes= []
    for ii in range(ncol):
        ax = fig.add_subplot(gs[0, ii])
        if logk:
            ax.loglog()
        ax.set_aspect('equal')
        if ii != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$k_\parallel$')

        ax.tick_params(direction='in', length=2.5, width=1)
        ax.tick_params(which='minor', direction='in', length=1.5, width=1)
        ax.set_xlabel(r'$k_\perp$')
        ax.set_title('%s'%ps_keys[ii])

        axes.append(ax)

    return fig, axes, cax

def load_2dtr_from3d(ps_path, ps_name, ps_ref, kbin_x_edges=None, kbin_y_edges=None):

    with h5.File(ps_path + ps_ref, 'r') as f:
        ps3d_ref = al.make_vect(al.load_h5(f, 'ps3d'))

    with h5.File(ps_path + ps_name, 'r') as f:

        ps3d = al.make_vect(al.load_h5(f, 'ps3d'))
        #ps3d[np.abs(ps3d)<1.e-20] = np.inf
        #ps3d = ps3d * ps3d_ref.copy()

        if kbin_x_edges is None:
            x = f['kbin_x_edges'][:]
        else:
            x = kbin_x_edges

        if kbin_y_edges is None:
            y = f['kbin_y_edges'][:]
        else:
            y = kbin_y_edges

    # find the k_perp by not including k_nu in the distance
    k_perp = binning.radius_array(ps3d, zero_axes=[0])

    # find the k_perp by not including k_RA,Dec in the distance
    k_para = binning.radius_array(ps3d, zero_axes=[1, 2])

    ps2d     = binning.bin_an_array_2d(ps3d,     k_perp, k_para, x, y)[1]
    ps2d_ref = binning.bin_an_array_2d(ps3d_ref, k_perp, k_para, x, y)[1]

    ps2d_ref[ps2d_ref==0] = np.inf
    ps2d /= ps2d_ref

    ps2d = ps2d ** 0.5
    ps2d[ps2d==0] = np.inf
    #ps2d = np.ma.masked_equal(ps2d, 0)
    ps2d = 1./ps2d

    return ps2d, x, y
            
def plot_2dps(ps_path, ps_name_list, tr_path=None, figsize=(16, 4),title='',
              vmax=None, vmin=None, logk=True, cross=False, lognorm=False,
              plot_counts=False, cmap='Blues'):   

    fig, ax, cax = _add_axes_2d(figsize, ps_name_list, logk=logk)
    fig.suptitle(title)

    for ii, ps_name, ps2d, ct2d, kbin_x, kbin_y in \
            iter_ps_list(ps_path, ps_name_list, tr_path, cross=cross):
        
        if lognorm:
            ps2d = np.ma.mean(ps2d, axis=0)
        else:
            ps2d = ps2d[0]

        if vmin is None:
            vmin = np.min(ps2d)
        if vmax is None:
            vmax = np.max(ps2d)
            
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im   = ax[ii].pcolormesh(kbin_x, kbin_y, ps2d.T, norm=norm, cmap=cmap)
        
        ax[ii].set_xlim(kbin_x.min(), kbin_x.max())
        ax[ii].set_ylim(kbin_y.min(), kbin_y.max())

    fig.colorbar(im, ax=ax, cax=cax)
    cax.minorticks_off()
    #cax.set_ylabel(r'$\Delta(k)[K^2]$')
    if cross:
        cax.set_ylabel(r'$P(k)\,[{\rm mK}\, {\rm Mpc}^{3} h^{-3}]$')
    else:
        cax.set_ylabel(r'$P(k)\,[{\rm mK}^2\, {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax

def plot_1dps(ps_path, ps_name_list, tr_path=None, figsize=(8,6), title='',
              vmin=None, vmax=None, logk=True, kmin=None, kmax=None,
              plot_null=False, cross=False, lognorm=False,
              plot_numb=False, shift=0.8,b_HI= 0.62, b_g=1.0,
              max_beam=False, legend_title=None, Tsys=25.,
              nbar = 3.21, fmt='s', from2d=True):

    if lognorm: plot_null = False
    
    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, ax_null = _add_axes_1d(figsize, plot_null)
    
    if plot_numb:
        ax2 = ax.twinx()
    
    for ii, ps_name, ps1d, ct1d, kh in \
            iter_ps_list(ps_path, ps_name_list, tr_path, cross=cross, 
                    to1d=True, from2d=from2d):

        c    = _c_list[ii]
        label = ps_name

        if lognorm:
            ps = np.ma.mean(ps1d, axis=0)
            ps_e = np.ma.std(ps1d, axis=0)
        else:
            ps = ps1d[0]
            ps_e = np.ma.std(ps1d[1:], axis=0)
            null = np.ma.mean(ps1d[1:], axis=0)

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.errorbar(_kh , np.abs(ps), ps_e, fmt=fmt, elinewidth=2.2, 
                    mfc=c, mew=1.5, ms=6,
                    capsize=0, color=c, label=label)

        pp = ps < 0
        ax.errorbar(_kh[pp] , -ps[pp], ps_e[pp], fmt=fmt, elinewidth=2.2, 
                    mfc='w', mew=1.5, ms=6,
                    capsize=0, color=c)

        if plot_null:
            ax_null.errorbar(_kh , null, ps_e, fmt='s', elinewidth=1., 
                        mfc='w', mew='1.', ms=5,
                        capsize=2, color=c, label='NULL',)
        if plot_numb:
            ax2.plot(_kh, numb, '--', color=c, drawstyle='steps-mid')

    plot_th(ax, kh, b_HI=b_HI, b_g=b_g, nbar=nbar, Tsys=Tsys, 
            cross=cross, logk=logk)

    ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    ax.set_xlim(kmin, kmax)
    ax.set_ylim(vmin, vmax)
    if plot_null:
        ax_null.set_xlim(kmin, kmax)
        #ax_null.set_ylim(-vmax, vmax)
        ax_null.axhline(0, 0, 1, color='k', ls='--')
        if logk:
            ax_null.semilogx()

    if logk:
        ax.loglog()
        #ax.semilogx()
    else:
        ax.semilogy()

    if plot_null:
        ax.set_xticklabels([])
        ax_null.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    else:
        ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    #ax.set_ylabel(r'$\Delta(k) [K^2]$')
    if cross:
        ax.set_ylabel(r'$P(k)\, [{\rm mK}\, {\rm Mpc}^{3} h^{-3}]$')
    else:
        ax.set_ylabel(r'$P(k)\, [{\rm mK}^2\,  {\rm Mpc}^{3} h^{-3}]$')
    if plot_numb:
        if logk:
            ax2.loglog()
        else:
            ax2.semilogy()
        ax2.set_ylabel('Number of $k$ modes')

    return fig, ax

def plot_th(ax, kh, b_HI=1., b_g=1., nbar=3., Tsys=16, cross=True, freq=None, 
        logk=True, max_beam=False):

    #----------------------------
    if freq is None:
        freq = np.arange(1050, 1150, 0.5)
    zz = 1420./ freq - 1.

    if logk:
        dkh = kh[1] / kh[0]
        kh_edgs = kh[0] * dkh ** (np.arange(40) - 4.5)
    else:
        dkh = kh[1] - kh[0]
        kh_edgs = kh[0] + dkh * (np.arange(100) - 4.5)
        kh_edgs = kh_edgs[kh_edgs>0.]
        kh_edgs = kh_edgs[kh_edgs<1.]

    #kh_edgs = np.logspace(-2., 0, 101)
    #dkh = kh_edgs[1:] / kh_edgs[:-1]
    #kh = kh_edgs[:-1] * ((kh_edgs[1:] / kh_edgs[:-1]) ** 0.5)
    
    mu_edgs = np.linspace(0, 1, 101)
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)

    #b_HI= 0.62 #* 0.5
    #b_HI= 0.3
    params = {
        'zz' : zz,
        'logk': logk,
        'kh_edgs' : kh_edgs,
        'mu_edgs' : mu_edgs,
        'S_area': 1000.,
        'T_sys' : Tsys * 1.e3,
        'pixel_size' : 1.,
        't_tot' : 100. * 3600.,
        'N_beam': 1.,
        'N_dish': 19.,
        'D_dish': 3./60.,
        'b_HI'  : b_HI,
        'b_g'   : b_g,
        'nbar'  : nbar * 1e-4,
        #'nbar'  : 1.68e-4,
        'k_fgcut' : None, #0.06,
    }

    if cross:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d, shot = \
                fisher.est_dP_cross(params, False, False, max_beam)
    else:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d = \
                fisher.est_dP_auto(params, False, False, max_beam)
    #pkhi1d   = pkhi1d   * kh**3./ (2.*np.pi**2)
    #pkhi1d_c = pkhi1d_c * kh**3./ (2.*np.pi**2)
    #ax.plot(kh, pkhi1d, 'k--', drawstyle='steps-mid')
    label = r'$\Omega_{\rm HI}b_{\rm HI} = %3.2f \times 10^{-3}$'%b_HI
    if cross:
        label += ',' + r'$b_{g}=%2.1f$'%b_g
    ax.plot(kh, pkhi1d_c, 'k-', linewidth=2.0, label = label, drawstyle='steps-mid')
    label = r'$T_{\rm sys}=%3.1f\,{\rm K}$'%Tsys
    if cross:
        label += ',' + r'$\bar{n}_{\rm g} = %3.2f\times10^{-4}$'%nbar
    #ax.fill_between(kh, pkhi1d_c * (1 + dpk2pk), pkhi1d_c * (1 - dpk2pk),
    #        color='0.8', step='mid', 
    #        label=label)

    #----------------------------
