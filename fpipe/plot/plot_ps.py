'''
functions for plotting ps results

'''
from collections import OrderedDict as dict

import h5py as h5
import numpy as np
import os

#from numpy.interpolate import interp2d
from scipy.interpolate import Rbf

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter

from fpipe.sim import fisher
from fpipe.ps import ps_summary

from fpipe.utils import binning
from fpipe.map import algebra as al

from fpipe.ps import pwrspec_estimator as pet

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 'k', '0.5', "#9467bd", 
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

def searching_limit(chisq, lower_cut = False, upper_cut = False):

    L = np.exp(-0.5 * chisq)

    L_argmax = L.argmax()
    L_step = 1. #L[1] - L[0]
    # searching upper limit
    p_upper_arg = L_argmax
    if upper_cut:
        p_upper_total = L[0:L_argmax+1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    else:
        p_upper_total = L[L_argmax:-1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    #print p_upper_total
    p_upper = L[L_argmax] * 0.5 * L_step
    for i in range(L_argmax + 1, len(L)):
        p_upper += L[i] * L_step
        p_upper_arg = i
        if p_upper/p_upper_total > 0.68:
            break
    # searching lower limit
    p_lower_arg = L_argmax
    if lower_cut:
        p_lower_total = L[L_argmax:-1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    else:
        p_lower_total = L[0:L_argmax+1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    #print p_lower_total
    p_lower = L[L_argmax] * 0.5 * L_step
    #print p_lower
    for i in range(L_argmax - 1, -1, -1):
        p_lower += L[i] * L_step
        p_lower_arg = i
        if p_lower/p_lower_total > 0.68:
            break
    return p_upper_arg, p_lower_arg


def _fitting(pk, pk_cov_inv, pk_th, shot=None, sel=None):

    if shot is None: shot = np.zeros_like(pk_th)

    A = np.linspace(0.01, 10.0, 6000)[:, None]

    #A = np.dot(np.dot(pk, pk_cov), pk_th) / np.dot(np.dot(pk_th, pk_cov), pk_th)

    if sel is not None:
        pk = pk[sel]
        shot = shot[sel]
        pk_cov_inv = pk_cov_inv[sel, :][:, sel]
        pk_th = pk_th[sel]

    x     = pk[None, :] - (pk_th[None, :] + shot[None, :]) * A
    chisq = np.sum(x[:, :, None] * pk_cov_inv[None, :, :] * x[:, None, :], axis=(1, 2))

    chisq_min = np.argmin(chisq)

    upper, lower = searching_limit(chisq)

    A = A.flatten()

    #print A[chisq_min], A[upper], A[lower]

    return A[chisq_min], A[upper] - A[chisq_min], A[chisq_min] - A[lower]

def load_3dtr(ps_path, ps_name, ps_ref):

    with h5.File(ps_path + ps_ref, 'r') as f:
        ps3d_ref = np.mean(al.load_h5(f, 'ps3d'), axis=0)

    with h5.File(ps_path + ps_name, 'r') as f:

        ps3d     = np.mean(al.load_h5(f, 'ps3d'), axis=0)
        #ps3d[np.abs(ps3d)<1.e-20] = np.inf
        #ps3d[np.abs(ps3d)<1.e-1] = np.inf
        #ps3d = ps3d * ps3d_ref.copy()

    ps3d = al.make_vect(ps3d)
    k_perp = binning.radius_array(ps3d, zero_axes=[0])
    mask = k_perp > 0.10

    #ps3d_ref[ps3d_ref==0] = np.inf
    #ps3d /= ps3d_ref
    ps3d[ps3d==0] = np.inf
    tr3d = ps3d_ref / ps3d
    #tr3d[np.abs(tr3d)>10] = 0.
    tr3d[mask] = 0.
    #tr3d[tr3d<0] = 0.

    return tr3d

#def iter_3dps_list(ps_path, ps_name_list, tr_path=None, cross=True):
#
#    ps_keys = ps_name_list.keys()
#    for ii in range(len(ps_keys)):
#        ps_name = ps_name_list[ps_keys[ii]]
#        #if tr_path is not None:
#        tr3d = None
#        if isinstance(ps_name, list):
#            if tr_path is None: tr_path = ps_path
#            ps_name, tr_name = ps_name
#            if tr_name is not None:
#                #ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
#                ref_name = os.path.dirname(tr_name) + '/ps_sub00modes.h5'
#                tr3d = ps_summary.load_3dtr(tr_path, tr_name, ref_name)
#
#        with h5.File(ps_path + ps_name, 'r') as f:
#
#            bins_x = f['kbin_x_edges'][:]
#            bins_y = f['kbin_y_edges'][:]
#            xc = f['kbin_x'][:]
#            yc = f['kbin_y'][:]
#            bins = f['kbin_edges'][:]
#            kc = f['kbin'][:]
#
#            ps3d = al.load_h5(f, 'ps3d')
#            if cross: ps3d *= 1.e3
#            else:     ps3d *= 1.e6
#            if tr3d is not None: ps3d *= tr3d.copy()[None, ...]
#
#        yield ii, ps_keys[ii], ps3d, bins_x, bins_y, xc, yc, bins, kc

#def iter_2dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
#        bins_x=None, bins_y=None, bins_xc=None, bins_yc=None):
#    for ps3d_set in iter_3dps_list(ps_path, ps_name_list, tr_path=tr_path, cross=cross):
#        ii, ps_key, ps3d, xe, ye, xc, yc, ke, kc = ps3d_set
#        if bins_x is None: bins_x = xe; bins_xc = xc
#        if bins_y is None: bins_y = ye; bins_yc = yc
#        ps2d = np.zeros((ps3d.shape[0], bins_x.shape[0]-1, bins_y.shape[0]-1))
#        for jj in range(ps3d.shape[0]):
#            ps3d_jj = al.make_vect(ps3d[jj])
#            # find the k_perp by not including k_nu in the distance
#            k_perp = binning.radius_array(ps3d_jj, zero_axes=[0])
#            # find the k_perp by not including k_RA,Dec in the distance
#            k_para = binning.radius_array(ps3d_jj, zero_axes=[1, 2])
#            ct2d, ps2d[jj] = binning.bin_an_array_2d(ps3d_jj, 
#                    k_perp, k_para, bins_x, bins_y)
#
#        yield ii, ps_key, ps2d, ct2d, bins_x, bins_y, bins_xc, bins_yc


#def iter_1dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
#        bins=None, bins_c=None):
#    for ps3d_set in iter_3dps_list(ps_path, ps_name_list, tr_path=tr_path, cross=cross):
#        ii, ps_key, ps3d, xe, ye, xc, yc, ke, kc = ps3d_set
#        if bins is None: bins = ke; bins_c = kc
#        ps1d = np.zeros((ps3d.shape[0], bins.shape[0]-1))
#        for jj in range(ps3d.shape[0]):
#            ps3d_jj = al.make_vect(ps3d[jj])
#            k = binning.radius_array(ps3d_jj)
#            ct1d, ps1d[jj] = binning.bin_an_array(ps3d_jj, bins, radius_arr=k)
#
#        yield ii, ps_key, ps1d, ct1d, bins, bins_c


def plot_2dps(ps_path, ps_name_list, tr_path=None, figsize=(16, 4), title='',
              vmax=None, vmin=None, logk=True, cross=False, 
              cmap='jet'):   

    fig, ax, cax = _add_axes_2d(figsize, ps_name_list, logk=logk)
    fig.suptitle(title)

    for ii, fkey in enumerate( ps_name_list.keys() ):

        c    = _c_list[ii]
        label = fkey


        result = np.loadtxt(ps_path + ps_name_list[fkey])

        shp = (len(set(result[:, 0])), len(set(result[:, 3])))
        khx_l = result[:, 0].reshape(shp)[:, 0]
        khx_c = result[:, 1].reshape(shp)[:, 0]
        khx_h = result[:, 2].reshape(shp)[:, 0]
        khy_l = result[:, 3].reshape(shp)[0, :]
        khy_c = result[:, 4].reshape(shp)[0, :]
        khy_h = result[:, 5].reshape(shp)[0, :]

        ps2d = result[:, 6].reshape(shp)

        kbin_xc = khx_c
        kbin_yc = khy_c

        kbin_x = np.append(khx_l, khx_h[-1:])
        kbin_y = np.append(khy_l, khy_h[-1:])

        ps2d.shape = shp
        ps2d[ps2d < 0] = 0
        ps2d = np.ma.masked_equal(ps2d, 0)

        if vmin is None:
            vmin = np.min(ps2d)
        if vmax is None:
            vmax = np.max(ps2d)

        #if fill_ct0:
        #    ps2d = fill_2dgaps(kbin_xc, kbin_yc, ct2d, ps2d)
            
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im   = ax[ii].pcolormesh(kbin_x, kbin_y, ps2d.T, norm=norm, 
                cmap=plt.cm.get_cmap(cmap, 200))
        
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

def plot_1dps_noise_ratio(ps_path, ps_name_list, tr_path=None, shot_name=None,
        figsize=(8,6), title='', vmin=None, vmax=None, logk=True, bins = None,
        kmin=None, kmax=None, cross=False, lognorm=False,
        from_2d=False, shift=0.8, b_HI= 0.62, b_g=1.0, b_x=1.,
        max_beam=False, legend_title=None, Tsys=25., nbar = 3.21, fmt='s',
        ncol=None, ps_ref='ps_sub00modes', conv_beam=True, fitting_k_max=0.2):


    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, _ = _add_axes_1d(figsize, plot_null=False, logk=logk, logP=False)

    ps_e_list = []
    for ii, fkey in enumerate(ps_name_list.keys()):

        _r = ps_name_list[fkey]
        if isinstance(_r, list):
            ps_name, ps_auto = _r
        else:
            ps_name = _r
            ps_auto = None
        result = np.loadtxt(ps_path + ps_name)
        kh = result[:, 0]
        c    = _c_list[ii]
        label = fkey
        ps   = result[:, 1]
        ps_e = result[:, 2]
        null = result[:, 3]
        shot = result[:, 4]
        cov  = result[:, 5:]
        ps_cov_inv = np.linalg.inv(cov)

        if ps_auto is not None:
            result = np.loadtxt(ps_path + ps_auto)
            kh_auto = result[:, 0]
            ps_auto = result[:, 1]
            if np.any((kh_auto - kh) != 0):
                print(kh_auto)
                print(kh)
                msg = 'kh_auto != kh'
                raise(ValueError(msg))

        kh_th, pk_th, pkerr_th, Tb_HI = est_error_th(kh, b_HI=b_HI,
                b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk,
                same_kh = True)

        ps_e_list.append(ps_e)

        sel = kh < fitting_k_max
        shot = shot * Tb_HI * b_x
        A, A_up, A_lo = _fitting(ps, ps_cov_inv, pk_th, shot)
        #if shot is not None:
        #    ps = ps - shot.copy() * A
        A = 1.

        _kh_th_auto, _pk_th_auto, _pkerr_th_auto, _Tb_HI_auto = est_error_th(kh,
                b_HI=b_HI * A, b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk,
                conv_beam=conv_beam, ps_auto=ps_auto, same_kh=True)

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.plot(_kh, ps_e/_pkerr_th_auto, 's-', linewidth=2, color=c, label=label)

    #ax.plot(_kh_th, _pkerr_th, 'k-', linewidth=2, drawstyle='steps-mid')
    ax.axhline(1, 0, 1, color='k', linestyle='-', linewidth=2)

    #ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    if ncol is None:
        ncol = int(np.ceil((ps_n + 2)/5.))
    ax.legend(loc=0, ncol=ncol, title=legend_title)
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    #ax.set_ylabel(r'$\Delta(k) [K^2]$')
    if cross:
        ax.set_ylabel(r'$\sigma_{P(k)}/\sigma_{P(k)}^{\rm auto}$')
    else:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}^2\,  {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax, ps_e_list

def plot_1dps_noise(ps_path, ps_name_list, tr_path=None, shot_name=None,
        figsize=(8,6), title='', vmin=None, vmax=None, logk=True, bins = None,
        kmin=None, kmax=None, cross=False, lognorm=False,
        from_2d=False, plot_numb=False, shift=0.8, b_HI= 0.62, b_g=1.0, b_x=1.,
        max_beam=False, legend_title=None, Tsys=25., nbar = 3.21, fmt='s',
        ncol=None, ps_ref='ps_sub00modes', conv_beam=True, fitting_k_max=0.2):


    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, _ = _add_axes_1d(figsize, plot_null=False, logk=logk)

    if plot_numb:
        ax2 = ax.twinx()

    ps_e_list = []
    for ii, fkey in enumerate(ps_name_list.keys()):

        _r = ps_name_list[fkey]
        if isinstance(_r, list):
            ps_name, ps_auto = _r
        else:
            ps_name = _r
            ps_auto = None
        result = np.loadtxt(ps_path + ps_name)
        kh = result[:, 0]
        c    = _c_list[ii]
        label = fkey
        ps   = result[:, 1]
        ps_e = result[:, 2]
        null = result[:, 3]
        shot = result[:, 4]
        cov  = result[:, 5:]
        ps_cov_inv = np.linalg.inv(cov)

        if ps_auto is not None:
            result = np.loadtxt(ps_path + ps_auto)
            kh_auto = result[:, 0]
            ps_auto = result[:, 1]
            if np.any((kh_auto - kh) != 0):
                print( kh_auto )
                print( kh )
                msg = 'kh_auto != kh'
                raise(ValueError(msg))
            #shp = (len(set(result[:, 0])), len(set(result[:, 3])))
            #khx_l = result[:, 0].reshape(shp)[:, 0]
            #khx_c = result[:, 1].reshape(shp)[:, 0]
            #khx_h = result[:, 2].reshape(shp)[:, 0]
            #khy_l = result[:, 3].reshape(shp)[0, :]
            #khy_c = result[:, 4].reshape(shp)[0, :]
            #khy_h = result[:, 5].reshape(shp)[0, :]

            #ps2d = result[:, 6].reshape(shp)

            #kbin_xc = khx_c
            #kbin_yc = khy_c

            #kbin_x = np.append(khx_l, khx_h[-1:])
            #kbin_y = np.append(khy_l, khy_h[-1:])

            #ps2d.shape = shp
            #ps2d[ps2d < 0] = 0
            #ps2d = np.ma.masked_equal(ps2d, 0)


        kh_th, pk_th, pkerr_th, Tb_HI = est_error_th(kh, b_HI=b_HI,
                b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk,
                same_kh = True)

        ps_e_list.append(ps_e)

        sel = kh < fitting_k_max
        shot = shot * Tb_HI * b_x
        A, A_up, A_lo = _fitting(ps, ps_cov_inv, pk_th, shot)
        #if shot is not None:
        #    ps = ps - shot.copy() * A
        A = 1.

        _kh_th_auto, _pk_th_auto, _pkerr_th_auto, _Tb_HI_auto = est_error_th(kh,
                b_HI=b_HI * A, b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk,
                conv_beam=conv_beam, ps_auto=ps_auto, same_kh=True)

        _kh_th, _pk_th, _pkerr_th, _Tb_HI = est_error_th(kh, b_HI=b_HI * A,
                b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk,
                conv_beam=conv_beam, ps_auto=None, same_kh=True)
        #print b_HI * A, b_HI * A_up

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.plot(_kh, ps_e, 's-', linewidth=2, color=c, label=label)
        ax.plot(_kh_th_auto, _pkerr_th_auto, '--', color=c, linewidth=3,
            drawstyle='steps-mid')

    ax.plot(_kh_th, _pkerr_th, 'k-', linewidth=2, drawstyle='steps-mid')

    #ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    if ncol is None:
        ncol = int(np.ceil((ps_n + 2)/5.))
    ax.legend(loc=0, ncol=ncol, title=legend_title)
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    #ax.set_ylabel(r'$\Delta(k) [K^2]$')
    if cross:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}\, {\rm Mpc}^{3} h^{-3}]$')
    else:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}^2\,  {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax, ps_e_list

def plot_1dps_noise_from_raw_ps_output(ps_path, ps_name_list, tr_path=None, 
        shot_name=None, figsize=(8,6), title='', vmin=None, vmax=None, logk=True,
        bins = None, kmin=None, kmax=None, cross=False, lognorm=False, 
        from_2d=False, plot_numb=False, shift=0.8, b_HI= 0.62, b_g=1.0,
        max_beam=False, legend_title=None, Tsys=25., nbar = 3.21, fmt='s',
        ncol=None, ps_ref='ps_sub00modes', conv_beam=True, fitting_k_max=0.2 ):

    
    ps_keys = list(ps_name_list.keys())
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, _ = _add_axes_1d(figsize, plot_null=False, logk=logk)

    if bins is not None:
        kh = bins[:-1]
        if logk:
            dk = bins[1] / bins[0]
            kh = kh * (dk ** 0.5)
        else:
            dk = bins[1] - bins[0]
            kh = kh + (dk * 0.5)
    else:
        kh = None
    
    if plot_numb:
        ax2 = ax.twinx()

    kh_th, pk_th, pkerr_th, Tb_HI = load_pkth(kh, b_HI=b_HI,
            b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk, 
            same_kh = True)
    #print pkerr_th

    if shot_name is not None:
        for xx in ps_summary.iter_1dps_list(ps_path, {'shot': shot_name}, 
                cross=False, bins=bins, bins_c=kh):
            shot = np.ma.mean(xx[2][1:] * 1.e-3, axis=0)
            #shot = np.mean(shot)
            shot = shot * Tb_HI
    else:
        shot = None
    
    ps_e_list = []
    for ii, ps_name, ps1d, ct1d, bins, kh in\
            ps_summary.iter_1dps_list(ps_path, ps_name_list, tr_path, cross=cross, 
                    bins=bins, bins_c=kh, ps_ref=ps_ref):

        c    = _c_list[ii]
        label = ps_name

        if lognorm:
            ps = np.ma.mean(ps1d, axis=0)
            ps_e = np.ma.std(ps1d, axis=0)
        else:
            ps = ps1d[0]
            ps_e = np.ma.std(ps1d[1:], axis=0)
            ps_cov_inv = np.linalg.inv(np.cov(ps1d[1:], rowvar=False))
            null = np.ma.mean(ps1d[1:], axis=0)

            #ps = ps - shot.copy()


        #ps = ps/alias_corr.copy()
        ps_e_list.append(ps_e)

        A, A_up, A_lo = _fitting(ps, ps_cov_inv, pk_th, shot)
        if shot is not None:
            ps = ps - shot.copy() * A

        _kh_th, _pk_th, _pkerr_th, _Tb_HI = load_pkth(kh, b_HI=b_HI * A,
                b_g=b_g, nbar=nbar, Tsys=Tsys, cross=cross, logk=logk, conv_beam=conv_beam)
        #print b_HI * A, b_HI * A_up

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.plot(_kh, ps_e, 's-', linewidth=2, color=c, label=label)

    ax.plot(_kh_th, _pkerr_th, 'k-', linewidth=2, drawstyle='steps-mid')

    #ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    if ncol is None:
        ncol = int(np.ceil((ps_n + 2)/5.))
    ax.legend(loc=0, ncol=ncol, title=legend_title)
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    #ax.set_ylabel(r'$\Delta(k) [K^2]$')
    if cross:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}\, {\rm Mpc}^{3} h^{-3}]$')
    else:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}^2\,  {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax, ps_e_list

def plot_noise_diffmodes(ps_path, ps_name_list, ps_e_list = None,
        figsize=(8,6), title='', 
        vmin=None, vmax=None, logk=True,
        kmin=None, kmax=None, cross=False, lognorm=False, 
        legend_title=None):

    
    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.12, 0.13, 0.85, 0.80])

    #bins = np.logspace(np.log10(0.045), np.log10(0.3), 3)
    bins = np.array([0.04, 0.1, 0.3])

    kh = bins[:-1]
    if logk:
        dk = bins[1] / bins[0]
        kh = kh * (dk ** 0.5)
    else:
        dk = bins[1] - bins[0]
        kh = kh + (dk * 0.5)
    
    if ps_e_list is None:
        ps_e_list = []
        for ii, ps_name, ps1d, ct1d, bins, kh in\
                ps_summary.iter_1dps_list(ps_path, ps_name_list, tr_path=None, cross=cross, 
                        bins=bins, bins_c=kh):

            if lognorm:
                ps = np.ma.mean(ps1d, axis=0)
                ps_e = np.ma.std(ps1d, axis=0)
            else:
                ps = ps1d[0]
                ps_e = np.ma.std(ps1d[1:], axis=0)

            #ps = ps/alias_corr.copy()
            ps_e_list.append(ps_e)

        ps_e_list = np.array(ps_e_list)

    xx = np.arange(ps_e_list.shape[0])

    for i in range(ps_e_list.shape[1]):
        l = ax.plot(xx, ps_e_list[:, i], 'o-')
        _text = r'$k \in [%4.2f, %4.2f]\,{\rm Mpc}^{-1}h$'%(bins[i], bins[i+1])
        fig.text(0.9, 0.8 - i * 0.08, _text, fontsize=15, ha='right', 
                color=l[0].get_color())

    ax.set_xticks(xx)
    ax.set_xticklabels(ps_keys)

    ax.semilogy()


    #ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'Number of SVD modes subtracted')
    if cross:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}\, {\rm Mpc}^{3} h^{-3}]$')
    else:
        ax.set_ylabel(r'$\sigma_{P(k)}\, [{\rm mK}^2\,  {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax, ps_e_list

def plot_1dps_ratio(ps_path, ps_name_list, figsize=(8,6), title='', fmt='s', 
        vmin=None, vmax=None, kmin=None, kmax=None, cross=True, logk=True,
        legend_title=None, shift=0.5, b_HI= 0.62, b_g=1.0, b_x=1.0,
        Tsys=25., nbar = 3.21, k2Pk=False, fitting_k_max=0.2,
        ref_name=None):

    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    if ref_name is None:
        ref_name = ps_keys[0]
    ref_ps = np.loadtxt(ps_path + ps_name_list[ref_name])[:, 1]
    ref_ps[ref_ps==0] == np.inf
    
    if not isinstance(b_g, list): b_g = [b_g, ] * ps_n

    fig, ax, ax_null = _add_axes_1d(figsize, plot_null=False, logk=logk, logP= False)

    for ii, fkey in enumerate( ps_name_list.keys() ):

        if fkey == ref_name: continue

        result = np.loadtxt(ps_path + ps_name_list[fkey])
        kh = result[:, 0]

        c    = _c_list[ii]
        label = fkey

        ps   = (result[:, 1] - ref_ps) / ref_ps
        ps_e = (result[:, 2] - ref_ps) / ref_ps
        null = result[:, 3]
        shot = result[:, 4]
        cov  = result[:, 5:]
        ps_cov_inv = np.linalg.inv(cov)


        kh_th, pk_th, pkerr_th, Tb_HI = load_pkth(kh, b_HI=b_HI,
                b_g=b_g[ii], nbar=nbar, Tsys=Tsys, cross=cross, logk=logk, 
                same_kh = True)

        sel = kh < fitting_k_max
        shot = shot * Tb_HI * b_x
        A, A_up, A_lo = _fitting(ps, ps_cov_inv, pk_th, shot, sel=sel)
        #ps = ps - shot * A
        #print shot*A

        _kh_th, _pk_th, _pkerr_th, _Tb_HI = load_pkth(kh, b_HI=b_HI * A,
                b_g=b_g[ii], nbar=nbar, Tsys=Tsys, cross=cross, logk=logk)
        #plot_th(ax, _kh_th, _pk_th, _pkerr_th, b_HI * A, b_g[ii], Tsys, nbar,
        #        b_HI_err = b_HI * A_up, cross=cross, color=c, k2Pk=k2Pk)

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        if k2Pk:
            ax.errorbar(_kh , ps * _kh**2, ps_e * _kh**2, fmt=fmt, elinewidth=2.2, 
                    mfc=c, mew=1.5, ms=6, capsize=0, color=c, label=label)
            ax.fill_between(_kh, ps * _kh**2 + ps_e * _kh**2, ps * _kh**2 - ps_e * _kh**2, 
                    color=c, alpha=0.3)
        else:
            #ps = ps * _kh ** 3. / 2. / np.pi / np.pi
            #ps_e = ps_e * _kh ** 3. / 2. / np.pi / np.pi
            ax.errorbar(_kh , np.abs(ps), ps_e, fmt=fmt, elinewidth=2.2, 
                    mfc=c, mew=1.5, ms=6, capsize=0, color=c, label=label)

            pp = ps < 0
            ax.errorbar(_kh[pp] , -ps[pp], ps_e[pp], fmt=fmt[0], elinewidth=2.2, 
                    mfc='w', mew=1.5, ms=6, capsize=0, color=c)

    ax.legend(loc=0, ncol=1, title=legend_title) #, mode='expand')

    if k2Pk:
        ax.axhline(0, 0, 1, linewidth=1, linestyle='--', color='k')

    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    ax.set_ylabel(r'$P(k)/P_{\rm ref}(k) - 1$')
    #if cross:
    #    if k2Pk:
    #        ax.set_ylabel(r'$k^2 P(k)\, [{\rm mK}\, {\rm Mpc}\,  h^{-1}]$')
    #    else:
    #        ax.set_ylabel(r'$P(k)\, [{\rm mK}\, {\rm Mpc}^{3}\,  h^{-3}]$')
    #else:
    #    if k2Pk:
    #        ax.set_ylabel(r'$k^2 P(k)\, [{\rm mK}^2\, {\rm Mpc}\,  h^{-1}]$')
    #    else:
    #        ax.set_ylabel(r'$P(k)\, [{\rm mK}^2\,  {\rm Mpc}^{3}\,  h^{-3}]$')

    return fig, ax

def plot_1dps(ps_path, ps_name_list, figsize=(8,6), title='', fmt='s', 
        vmin=None, vmax=None, kmin=None, kmax=None, cross=True, logk=True,
        legend_title=None, shift=0.5, b_HI= 0.62, b_g=1.0, b_x=1.0,
        Tsys=25., nbar = 3.21, k2Pk=False, fitting_k_max=0.2, add_th=False,
        ncol=2):

    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift
    
    if not isinstance(b_g, list): b_g = [b_g, ] * ps_n

    fig, ax, ax_null = _add_axes_1d(figsize, plot_null=False, logk=logk, logP= not k2Pk)

    for ii, fkey in enumerate( ps_name_list.keys() ):

        result = np.loadtxt(ps_path + ps_name_list[fkey])
        kh = result[:, 0]

        c    = _c_list[ii]
        label = fkey

        ps   = result[:, 1]
        ps_e = result[:, 2] 
        null = result[:, 3]
        shot = result[:, 4]
        cov  = result[:, 5:]
        ps_cov_inv = np.linalg.inv(cov)


        kh_th, pk_th, pkerr_th, Tb_HI, pkn1d = load_pkth(kh, b_HI=b_HI,
                b_g=b_g[ii], nbar=nbar, Tsys=Tsys, cross=cross, logk=logk, 
                same_kh = True, return_pkn=True)

        sel = kh < fitting_k_max
        shot = shot * Tb_HI * b_x
        A, A_up, A_lo = _fitting(ps, ps_cov_inv, pk_th, shot, sel=sel)
        #ps = ps - shot * A
        #print shot*A

        _kh_th, _pk_th, _pkerr_th, _Tb_HI = load_pkth(kh, b_HI=b_HI * A,
                b_g=b_g[ii], nbar=nbar, Tsys=Tsys, cross=cross, logk=logk)
        #if add_th:
        #    plot_th(ax, _kh_th, _pk_th, _pkerr_th, b_HI * A, b_g[ii], Tsys, nbar,
        #        b_HI_err = b_HI * A_up, cross=cross, color=c, k2Pk=k2Pk)

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        if k2Pk:
            ax.errorbar(_kh , ps * _kh**2, ps_e * _kh**2, fmt=fmt, elinewidth=2.2, 
                    mfc=c, mew=1.5, ms=6, capsize=0, color=c, label=label)
            ax.fill_between(_kh, ps * _kh**2 + ps_e * _kh**2, ps * _kh**2 - ps_e * _kh**2, 
                    color=c, alpha=0.3)
        else:
            #ps = ps * _kh ** 3. / 2. / np.pi / np.pi
            #ps_e = ps_e * _kh ** 3. / 2. / np.pi / np.pi
            #ax.errorbar(_kh , np.abs(ps), ps_e, fmt=fmt, elinewidth=2.2, 
            #        mfc=c, mew=1.5, ms=6, capsize=0, color=c, label=label)
            ax.errorbar(_kh , np.abs(ps), ps_e, fmt=fmt, elinewidth=1.5, 
                    mfc=c, mew=1., ms=5, capsize=0, color=c, label=label)
            #ax.fill_between(_kh, ps  + ps_e , ps  - ps_e , 
            #        color=c, alpha=0.3)

            pp = ps < 0
            ax.errorbar(_kh[pp] , -ps[pp], ps_e[pp], fmt=fmt[0], elinewidth=1.5, 
                    mfc='w', mew=1., ms=5, capsize=0, color=c)

    if add_th:
        ax.errorbar(kh_th , pk_th, pkerr_th, fmt=fmt, elinewidth=2.2, 
                mfc='none', mew=1.5, ms=6, capsize=0, color='k')
        ax.plot(kh_th, pkn1d, 'k--')


    leg = ax.legend(loc=2, ncol=ncol, title=legend_title) #, mode='expand')
    leg._legend_box.align = "left"

    if k2Pk:
        ax.axhline(0, 0, 1, linewidth=1, linestyle='--', color='k')

    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    if cross:
        if k2Pk:
            ax.set_ylabel(r'$k^2 P(k)\, [{\rm mK}\, {\rm Mpc}\,  h^{-1}]$')
        else:
            ax.set_ylabel(r'$P(k)\, [{\rm mK}\, {\rm Mpc}^{3}\,  h^{-3}]$')
    else:
        if k2Pk:
            ax.set_ylabel(r'$k^2 P(k)\, [{\rm mK}^2\, {\rm Mpc}\,  h^{-1}]$')
        else:
            ax.set_ylabel(r'$P(k)\, [{\rm mK}^2\,  {\rm Mpc}^{3}\,  h^{-3}]$')

    return fig, ax
            
def plot_1dps_null(ps_path, ps_name_list, tr_path=None, 
        figsize=(8,6), title='', vmin=None, vmax=None, logk=True, bins = None,
        kmin=None, kmax=None, cross=False, lognorm=False, 
        shift=0.8, max_beam=False, legend_title=None):

    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, ax_null = _add_axes_1d(figsize, False, logk=logk)

    if bins is not None:
        kh = bins[:-1]
        if logk:
            dk = bins[1] / bins[0]
            kh = kh * (dk ** 0.5)
        else:
            dk = bins[1] - bins[0]
            kh = kh + (dk * 0.5)
    else:
        kh = None
    
    for ii, ps_name, ps1d, ct1d, bins, kh in\
            ps_summary.iter_1dps_list(ps_path, ps_name_list, tr_path, cross=cross, 
                    bins=bins, bins_c=kh):

        c    = _c_list[ii]
        label = ps_name

        if cross: ps1d *= 1.e-3
        else: ps1d *= 1.e-6

        ps_e = np.ma.std(ps1d[1:], axis=0)
        null = np.ma.mean(ps1d[1:], axis=0)

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.errorbar(_kh , null, ps_e, fmt='s', elinewidth=2.2, 
                    mfc='w', mew='1.5', ms=6, capsize=2.5, color=c, label=label,)

    ax.legend(loc=0, ncol=int(np.ceil((ps_n + 2)/5.)), title=legend_title)
    ax.set_yscale('linear')
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)
    ax.axhline(0, 0, 1, color='k', ls='--')

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    if cross:
        ax.set_ylabel(r'$P(k)\, [{\rm K}\, {\rm Mpc}^{3} h^{-3}]$', labelpad=0)
    else:
        ax.set_ylabel(r'$P(k)\, [{\rm K}^2\,  {\rm Mpc}^{3} h^{-3}]$')

    return fig, ax

def plot_1dps_opt(ps_path, ps_name_list, figsize=(8,6), title='', fmt='s', 
        vmin=None, vmax=None, kmin=None, kmax=None, logk=True,
        legend_title=None, shift=0.8, b_g=1.0, nbar = 3.21):

    _c = ['b', 'r']

    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)
    shift = (np.arange(ps_n) - (float(ps_n) - 1.)/ 2.) * shift

    fig, ax, ax_null = _add_axes_1d(figsize, plot_null=False, logk=logk)

    #map_path = '/scratch/users/ycli/analysis_meerkat/wigglez11hr/freq550-1050_l6mask/'\
    #         + 'opt_f000-250/WiggleZ11hr_Combined_cube/'
    #map_name = 'binning_realmap.h5'
    #with h5.File(map_path + map_name, 'r') as f:
    #    imap = al.make_vect(al.load_h5(f, 'delta'))
    #alias_corr = ca._alias_effect_correction_1D(imap, kh, n=2)
    alias_corr = 1.
    
    for ii, fkey in enumerate( ps_name_list.keys() ):

        result = np.loadtxt(ps_path + ps_name_list[fkey])
        kh = result[:, 0]

        #c    = _c_list[ii]
        c    = _c[ii]
        label = fkey

        ps   = result[:, 1] * 1.e-3
        ps_e = result[:, 2] * 1.e-3
        null = result[:, 3] * 1.e-3
        shot = result[:, 4]
        cov  = result[:, 5:] * 1.e-6
        ps_cov_inv = np.linalg.inv(cov)

        print(shot)
        ps = ps - shot#.copy()

        if logk:
            dk = (( kh[1] / kh[0] ) ** 0.5) ** (1./float(ps_n))
            _kh = kh * ( dk ** shift[ii] )
        else:        
            dk = (( kh[1] - kh[0] ) * 0.5) * (1./float(ps_n))
            _kh = kh + ( dk * shift[ii] )

        ax.errorbar(_kh , np.abs(ps)/alias_corr, ps_e, fmt=fmt, elinewidth=2.2, 
                    mfc=c, mew=1.5, ms=6, capsize=0, color=c, label=label)

        ax.plot(_kh, shot/alias_corr, '--', color=c, linewidth=2, 
                label=r'%s $P_{\rm shot}(k)$'%label)
        ax.plot(_kh, np.ones_like(_kh) * 1./nbar * 1.e4, 'k--', linewidth=2,
                label = r'$1/\bar{n}$')
                #label=r'$1/\bar{n}\,,\bar{n} = %3.2f$'%nbar\
                #        + r'$\times 10^{-4}\,{\rm Mpc}^{-3}h^3$')

        pp = ps < 0
        ax.errorbar(_kh[pp] , -ps[pp], ps_e[pp], fmt=fmt, elinewidth=2.2, 
                    mfc='w', mew=1.5, ms=6, capsize=0, color=c)

        #plot_th_opt(ax, kh, b_g=b_g, nbar=nbar, logk=logk)
        kh_th, pk_th = load_pkth(kh, b_HI=1., b_g=b_g, 
                nbar=nbar, Tsys=16, opt=True, logk=logk, same_kh=True,
                RSD=False, )
        b, b_up, b_lo = _fitting(ps, ps_cov_inv, pk_th)
        print(b_g, b, (b**0.5) * b_g, (b_up**0.5) * b_g, (b_lo**0.5) * b_g)
        _kh_th, _pk_th = load_pkth(kh, b_HI=1., b_g=b_g * (b**0.5), 
                nbar=nbar, Tsys=16, opt=True, logk=logk, RSD=False, )
        ax.plot(_kh_th, _pk_th, '-', linewidth=2.5, color=c,
                label=r'%s $b_{\rm g}^2 P(k)$'%label)

    ax.legend(loc=0, ncol=3, title=legend_title, mode='expand')
    if kmin is not None:
        ax.set_xlim(kmin, kmax)
    if vmin is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1} h]$')
    ax.set_ylabel(r'$P(k)\, [{\rm Mpc}^{3} h^{-3}]$')

    return fig, ax

def plot_corr(ps_path, ps_name_list, tr_path=None,
        figsize=(8,6), title='', vmin=None, vmax=None, logk=True, bins = None,
        kmin=None, kmax=None, cross=False, lognorm=False,
        legend_title=None,):

    ps_keys = ps_name_list.keys()
    ps_n = len(ps_keys)

    fig, axes, cax = _add_axes_2d(figsize, ps_name_list, logk=logk,
                                 left=0.12, right=0.87)

    if bins is not None:
        kh = bins[:-1]
        if logk:
            dk = bins[1] / bins[0]
            kh = kh * (dk ** 0.5)
        else:
            dk = bins[1] - bins[0]
            kh = kh + (dk * 0.5)
    else:
        kh = None

    for ii, ps_name, ps1d, ct1d, bins, kh in\
            ps_summary.iter_1dps_list(ps_path, ps_name_list, tr_path, cross=cross,
                    bins=bins, bins_c=kh):

        label = ps_name

        if lognorm:
            ps_e = np.ma.std(ps1d, axis=0)
        else:
            #ps_e = np.ma.std(ps1d[1:], axis=0)
            ps_cov = np.cov(ps1d[1:], rowvar=False)
            ps_diag = np.diag(ps_cov) ** 0.5
            ps_diag = ps_diag[None, :] * ps_diag[:, None]
            ps_corr = ps_cov / ps_diag


        im = axes[ii].pcolormesh(bins, bins, ps_corr, vmin=vmin, vmax=vmax)


        axes[ii].set_xticks([0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5])
        axes[ii].set_xticklabels([r'$0.02$', r'$0.04$', r'$0.06$', r'$0.1$', r'$0.2$',
            r'$0.3$', r'$0.4$', r'$0.5$', ])
        axes[ii].set_yticks([0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5])
        axes[ii].set_xlim(xmin=bins.min(), xmax=bins.max())
        axes[ii].set_ylim(ymin=bins.min(), ymax=bins.max())
        if ii == 0:
            axes[ii].set_yticklabels([r'$0.02$', r'$0.04$', r'$0.06$', r'$0.1$', r'$0.2$',
            r'$0.3$', r'$0.4$', r'$0.5$', ])

    fig.colorbar(im, ax=axes[-1], cax=cax)
    cax.set_ylabel('Correlation Matrix')


    return fig, axes

def plot_1dtr(ps_path, ps_name_list, tr_path=None, figsize=(16, 4), logk=True,
            vmin=None, vmax=None, ps_ref='ps_sub00modes', cmap='seismic',
            logscal=False, legend_title=''):

    fig, ax, _ = _add_axes_1d(figsize, plot_null=False, logk=logk, logP=True)

    for ii, ps_name in enumerate(ps_name_list.keys()):

        with h5.File(ps_path + ps_name_list[ps_name][1], 'r') as f:
            ps1d = f['tr1d'][:]
            kbin_c = f['kbin_c'][:]

        ax.plot(kbin_c, ps1d, 'o-', label=ps_name)

    ax.set_xlim(kbin_c.min() * 0.9, kbin_c.max() * 1.1)
    ax.set_ylim(vmin, vmax)
    ax.legend(loc=0, ncol=3, title=legend_title)
    ax.set_xlabel(r'$k~[h{\rm Mpc}^{-1}]$')

    ax.set_ylabel(r'$T(k) = P_{\rm c}(k) / P(k)$')

    return fig, ax


def plot_2dtr(ps_path, ps_name_list, tr_path=None, figsize=(16, 4), logk=True,
            vmin=None, vmax=None, ps_ref='ps_sub00modes', cmap='seismic',
            logscal=False):

    fig, ax, cax = _add_axes_2d(figsize, ps_name_list, logk=logk)

    for ii, ps_name in enumerate(ps_name_list.keys()):

        with h5.File(ps_path + ps_name_list[ps_name][1], 'r') as f:
            ps2d = f['tr2d'][:]
            kbin_x = f['kbin_x'][:]
            kbin_y = f['kbin_y'][:]

        if vmin is None:
            vmin = np.min(ps2d)
        if vmax is None:
            vmax = np.max(ps2d)

        if logscal:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        im   = ax[ii].pcolormesh(kbin_x, kbin_y, ps2d.T, norm=norm, cmap=cmap)

        ax[ii].set_xlim(kbin_x.min(), kbin_x.max())
        ax[ii].set_ylim(kbin_y.min(), kbin_y.max())

    fig.colorbar(im, ax=ax, cax=cax)
    cax.minorticks_off()
    #cax.set_ylabel(r'$\Delta(k)[K^2]$')

    cax.set_ylabel(r'$T(k) = P_{\rm c}(k) / P(k)$')

    return fig, ax

def plot_2dtr_ratio(ps_path, ps_name_list, ps_name_list2, figsize=(16, 4), logk=True,
            vmin=None, vmax=None, ps_ref='ps_sub00modes', cmap='seismic',
            logscal=False):

    fig, ax, cax = _add_axes_2d(figsize, ps_name_list, logk=logk)

    for ii, ps_name in enumerate(ps_name_list.keys()):

        with h5.File(ps_path + ps_name_list[ps_name][1], 'r') as f:
            ps2d = f['tr2d'][:]
            kbin_x = f['kbin_x'][:]
            kbin_y = f['kbin_y'][:]

        with h5.File(ps_path + ps_name_list2[ps_name][1], 'r') as f:
            ps2d2 = f['tr2d'][:]

        ps2d2[ps2d2==0] = np.inf
        ps2d = ps2d / ps2d2

        if vmin is None:
            vmin = np.min(ps2d)
        if vmax is None:
            vmax = np.max(ps2d)

        if logscal:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        im   = ax[ii].pcolormesh(kbin_x, kbin_y, ps2d.T, norm=norm, cmap=cmap)

        ax[ii].set_xlim(kbin_x.min(), kbin_x.max())
        ax[ii].set_ylim(kbin_y.min(), kbin_y.max())

    fig.colorbar(im, ax=ax, cax=cax)
    cax.minorticks_off()
    #cax.set_ylabel(r'$\Delta(k)[K^2]$')

    cax.set_ylabel(r'$ T(k) / T(k)_{\rm avg1st} $')

    return fig, ax


def fill_2dgaps(kbin_xc, kbin_yc, ct2d, ps2d):
    yy, xx = np.meshgrid(kbin_yc, kbin_xc)
    msk = ct2d.mask
    msk = msk.flatten()
    xx_flat = xx.flatten()[~msk]
    yy_flat = yy.flatten()[~msk]
    zz_flat = (ps2d).flatten()[~msk]
    rbf = Rbf(yy_flat, xx_flat, zz_flat, function='linear')
    return rbf(yy, xx)

def _add_axes_1d(figsize, plot_null=False, logk=True, logP=True):

    fig = plt.figure(figsize=figsize)
    if plot_null:
        ax = fig.add_axes([0.12, 0.32, 0.88, 0.80])
        ax_null = fig.add_axes([0.09, 0.12, 0.88, 0.2])
    else:
        #ax = fig.add_axes([0.09, 0.13, 0.88, 0.80])
        ax = fig.add_axes([0.12, 0.14, 0.82, 0.80])
        ax_null = None

    if logk and logP:
        ax.loglog()
        if plot_null:
            ax_null.semilogx()
    elif not logk and logP:
        ax.semilogy()
    elif logk and not logP:
        ax.semilogx()

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_minor_formatter(NullFormatter())
    if plot_null:
        ax_null.xaxis.set_major_formatter(NullFormatter())
        ax_null.xaxis.set_minor_formatter(NullFormatter())
        ax_null.yaxis.set_major_formatter(NullFormatter())
        ax_null.yaxis.set_minor_formatter(NullFormatter())

    if plot_null:
        ax.set_xticks([])
        ax_null.set_xticks([0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax_null.set_xticklabels(
                [r'$0.02$', r'$0.04$', r'$0.06$', r'$0.1$', r'$0.2$', r'$0.3$',
                    r'$0.4$', r'$0.5$', ])
    else:
        ax.set_xticks([0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels([r'$0.02$', r'$0.04$', r'$0.06$', r'$0.1$', r'$0.2$', 
            r'$0.3$', r'$0.4$', r'$0.5$', ])

    return fig, ax, ax_null

def _add_axes_2d(figsize, ps_name_list, logk=True, left=0.07, bottom=0.16, top=0.9, right=0.92):

    ps_keys = ps_name_list.keys()
    ncol = len(ps_keys)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, ncol, left=left, bottom=bottom, top=top, right=right,
            figure=fig, wspace=0.1)
    cax = fig.add_axes([right+0.01, 0.2, 0.1/float(figsize[0]), 0.66])
    axes= []
    for ii in range(ncol):
        ax = fig.add_subplot(gs[0, ii])
        if logk:
            #ax.loglog()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('equal')

            #ax.xaxis.set_major_formatter(NullFormatter())
            #ax.xaxis.set_minor_formatter(NullFormatter())
            #ax.yaxis.set_major_formatter(NullFormatter())
            #ax.yaxis.set_minor_formatter(NullFormatter())

            #ax.set_yticks([0.1, 0.2, 0.4, 0.6, 0.8])
            #if ii == 0:
            #    ax.set_yticklabels([r'$0.1$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'])

            #ax.set_xticks([0.02, 0.04, 0.08])
            #ax.set_xticklabels([r'$0.02$', r'$0.04$', r'$0.08$'])
        if ii != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$k_\parallel$')

        ax.tick_params(direction='in', length=2.5, width=1)
        ax.tick_params(which='minor', direction='in', length=1.5, width=1)
        ax.set_xlabel(r'$k_\perp$')
        title = ps_keys[ii]
        title = title.replace(', ', '\n')
        ax.set_title('%s'%title)
        #ax.set(xticks=[], yticks=[])

        ticklabels = ax.xaxis.get_ticklabels()
        #ticklabels[0].set_ha('right')
        ticklabels[0].set_ha('right')


        axes.append(ax)

    return fig, axes, cax

def plot_th_opt(ax, kh, b_g=1., nbar=3., freq=None, logk=True):

    if freq is None:
        freq = np.arange(971, 1023, 0.2)
    zz = 1420./ freq - 1.

    if logk:
        dkh = kh[1] / kh[0]
        kh_edgs = kh[0] * dkh ** (np.arange(40) - 4.5)
    else:
        dkh = kh[1] - kh[0]
        kh_edgs = kh[0] + dkh * (np.arange(100) - 4.5)
        kh_edgs = kh_edgs[kh_edgs>0.]
        kh_edgs = kh_edgs[kh_edgs<1.]

    mu_edgs = np.linspace(0, 1, 101)
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)

    pk = np.sum( np.mean(fisher.Pk_G(kh, zz, mu, b_g=b_g, RSD=True), axis=0) * dmu, axis=1)
    ax.plot(kh, pk, 'k-', linewidth=2.0, ) #drawstyle='steps-mid')

def load_pkth(kh, b_HI=1., b_g=1., nbar=3., Tsys=16, cross=True, opt=False,
        freq=None, 
        logk=True, max_beam=False, same_kh=False, 
        conv_beam=True, RSD=True, return_pkn=False):

    if freq is None:
        freq = np.arange(971, 1023, 0.2)
    zz = 1420./ freq - 1.

    if logk:
        dkh = kh[1] / kh[0]
        if same_kh:
            kh_edgs = kh[0] * dkh ** (np.arange(kh.shape[0] + 1) - 0.5)
        else:
            kh_edgs = kh[0] * dkh ** (np.arange(40) - 4.5)
    else:
        dkh = kh[1] - kh[0]
        if same_kh:
            kh_edgs = kh[0] + dkh * (np.arange(kh.shape[0] + 1) - 0.5)
        else:
            kh_edgs = kh[0] + dkh * (np.arange(100) - 4.5)
            kh_edgs = kh_edgs[kh_edgs>0.]
            kh_edgs = kh_edgs[kh_edgs<1.]
    
    mu_edgs = np.linspace(0, 1, 101)
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)

    if opt:

        pk1d_c = fisher.Pk_G(kh, zz, mu, b_g=b_g, RSD=True)
        pk1d_c = np.sum(np.mean(pk1d_c, axis=0) * dmu, axis=1)

        return kh, pk1d_c

    params = {
        'zz' : zz,
        'logk': logk,
        'kh_edgs' : kh_edgs,
        'mu_edgs' : mu_edgs,
        'S_area': 170., #260., #170.,
        'T_sys' : Tsys * 1.e3,
        'pixel_size' : 0.25, #1.,
        't_tot' : 7. * 1.5 * 3600.,
        'N_beam': 1.,
        'N_dish': 60.,
        'D_dish': 13.5,
        'b_HI'  : b_HI,
        'b_g'   : b_g,
        'nbar'  : nbar * 1e-4,
        #'nbar'  : 1.68e-4,
        'k_fgcut' : None #0.03,
    }

    if cross:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d, shot = \
                fisher.est_dP_cross(params, conv_beam=conv_beam, RSD=RSD, max_beam=max_beam)
    else:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d = \
                fisher.est_dP_auto(params, conv_beam=conv_beam, RSD=RSD, max_beam=max_beam)
        #pkhi1d_c = pkhi1d_c + pkn1d

    Tb = np.mean(fisher.Tb_HI(zz, OmHI = b_HI * 1.e-3))

    pkhi1d_err = pkhi1d_c * dpk2pk

    if return_pkn:
        return kh, pkhi1d_c + pkn1d, pkhi1d_err, Tb, pkn1d
    else:
        return kh, pkhi1d_c, pkhi1d_err, Tb

def est_error_th(kh, b_HI=1., b_g=1., nbar=3., Tsys=16, cross=True, opt=False,
        freq=None, logk=True, max_beam=False, same_kh=False, conv_beam=True, RSD=True, ps_auto=None):
    
    freq = np.arange(971, 1023, 0.2)
    zz = 1420./ freq - 1.
    
    if logk:
        dkh = kh[1] / kh[0]
        if same_kh:
            kh_edgs = kh[0] * dkh ** (np.arange(kh.shape[0] + 1) - 0.5)
        else:
            kh_edgs = kh[0] * dkh ** (np.arange(40) - 4.5)
    else:
        dkh = kh[1] - kh[0]
        if same_kh:
            kh_edgs = kh[0] + dkh * (np.arange(kh.shape[0] + 1) - 0.5)
        else:
            kh_edgs = kh[0] + dkh * (np.arange(100) - 4.5)
            kh_edgs = kh_edgs[kh_edgs>0.]
            kh_edgs = kh_edgs[kh_edgs<1.]

    mu_edgs = np.linspace(0, 1, 101)
    dmu = mu_edgs[1:] - mu_edgs[:-1]
    mu = mu_edgs[:-1] + (dmu * 0.5)
    
    params = {
        'zz' : zz,
        'logk': logk,
        'kh_edgs' : kh_edgs,
        'mu_edgs' : mu_edgs,
        'S_area': 170., #260., #170.,
        'T_sys' : Tsys * 1.e3,
        'pixel_size' : 0.25, #1.0,
        't_tot' : 7. * 1.5 * 3600.,
        'N_beam': 1.,
        'N_dish': 60.,
        'D_dish': 13.5,
        'b_HI'  : b_HI,
        'b_g'   : b_g,
        'nbar'  : nbar * 1e-4,
        #'nbar'  : 1.68e-4,
        'k_fgcut' : None #0.03,
    }

    if cross:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d, shot = \
                fisher.est_dP_cross_with_psauto(params, ps_auto, conv_beam=conv_beam, RSD=RSD, max_beam=max_beam)
    else:
        kh, pkhi1d_c, pkn1d, dpk2pk, pkhi1d = \
                fisher.est_dP_auto(params,  conv_beam=conv_beam, RSD=RSD, max_beam=max_beam)

    Tb = np.mean(fisher.Tb_HI(zz, OmHI = b_HI * 1.e-3))

    pkhi1d_err = pkhi1d_c * dpk2pk
    
    return kh, pkhi1d_c, pkhi1d_err, Tb

def plot_th(ax, kh, pkhi1d_c, pkhi1d_err, b_HI, b_g, Tsys, nbar, color='k', 
        b_HI_err=None, cross=True, k2Pk=False):
    #pkhi1d   = pkhi1d   * kh**3./ (2.*np.pi**2)
    #pkhi1d_c = pkhi1d_c * kh**3./ (2.*np.pi**2)
    #ax.plot(kh, pkhi1d, 'k--', drawstyle='steps-mid')
    if b_HI_err is not None:
        label = r'$\Omega_{\rm HI}b_{\rm HI} = %3.2f \pm %3.2f \times 10^{-3}$'%(
                b_HI, b_HI_err)
    else:
        label = r'$\Omega_{\rm HI}b_{\rm HI} = %3.2f \times 10^{-3}$'%b_HI
    if cross:
        label += ',' + r'$b_{g}=%3.2f$'%b_g
    if k2Pk:
        ax.plot(kh, pkhi1d_c * kh**2, '-', color=color, linewidth=1.5, 
                label = label, )
    else:
        ax.plot(kh, pkhi1d_c, '-', color=color, linewidth=2.0, 
                label = label, drawstyle='steps-mid')
    label = r'$T_{\rm sys}=%3.1f\,{\rm K}$'%Tsys
    if cross:
        label += ',' + r'$\bar{n}_{\rm g} = %3.2f\times10^{-4}$'%nbar
    #ax.fill_between(kh, pkhi1d_c + pkhi1d_err, pkhi1d_c - pkhi1d_err,
    #        color=color, step='mid', alpha=0.3)

