from fpipe.point_source import abs_cal
import h5py as h5
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from astropy.coordinates.angles import Angle

from scipy.interpolate import interp1d
from fpipe.utils import axes_utils
_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", 'w']


def iter_file_list(result_path, key_list, beam_list,
                   band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz']):

    for kk, key in enumerate(key_list):
        for beam in beam_list[kk]:
            for suffix in band_list:
                result_name = '%s%s_F%02d.h5'%(key, suffix, beam)

                yield beam, result_path + result_name

def plot_Tnoise(result_path, key_list, beam_list,
                band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                tnoise_model=None):

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    xmin = 1.e10
    xmax =-1.e10

    if tnoise_model is not None:
        with h5.File(tnoise_model, 'r') as f:
            #print f.keys()
            tnoise_md = f['Tnoise'][:]
            tnoise_md_freq = f['freq'][:]


    for beam, path in iter_file_list(result_path, key_list, beam_list, band_list):

        A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

        ax = axes[beam-1]

        ax.plot(freq/1.e3, A[:, 0]*0.7, c='r', lw=0.1)
        ax.plot(freq/1.e3, A[:, 1]*0.7, c='b', lw=0.1)

        if freq.min()/1.e3 < xmin: xmin=freq.min()/1.e3
        if freq.max()/1.e3 > xmax: xmax=freq.max()/1.e3

    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.61, 1.49)

        if tnoise_model is not None:
            ax.plot(tnoise_md_freq/1.e3, tnoise_md[:, 0, bi], 'k--',
                    linewidth=0.1, label='Tnoise Lab.')
            ax.plot(tnoise_md_freq/1.e3, tnoise_md[:, 1, bi], 'k-',
                    linewidth=0.1, label='Tnoise Lab.')

        if ii == 4:
            ax.set_xlabel('Frequency [GHz]')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel('ND [K]')
        else:
            ax.set_yticklabels([])


def plot_eta(result_path, key_list, beam_list,
                band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                tnoise_model=None):

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    xmin = 1.e10
    xmax =-1.e10

    with h5.File(tnoise_model, 'r') as f:
        #print f.keys()
        tnoise_md = f['Tnoise'][:]
        tnoise_md_freq = f['freq'][:]


    for beam, path in iter_file_list(result_path, key_list, beam_list, band_list):

        A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

        ax = axes[beam-1]


        eta = interp1d(tnoise_md_freq, tnoise_md[:, 0, beam-1])(freq) / A[:, 0]
        ax.plot(freq/1.e3, eta, c='r', lw=0.1)

        eta = interp1d(tnoise_md_freq, tnoise_md[:, 1, beam-1])(freq) / A[:, 1]
        ax.plot(freq/1.e3, eta, c='b', lw=0.1)

        if freq.min()/1.e3 < xmin: xmin=freq.min()/1.e3
        if freq.max()/1.e3 > xmax: xmax=freq.max()/1.e3

    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.45, 0.95)

        if ii == 4:
            ax.set_xlabel('Frequency [GHz]')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel(r'$\eta$')
        else:
            ax.set_yticklabels([])

    legend_list = []
    legend_list.append(mpatches.Patch(color='r', label='XX Polarization'))
    legend_list.append(mpatches.Patch(color='b', label='YY Polarization'))

    fig.legend(handles=legend_list, frameon=False, loc=4,
               bbox_to_anchor=(0.94, 0.10), ncol=2)


def plot_eta_days(result_path, key_list_dict, beam_list,
                  band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                  tnoise_model=None, pol=0, output=None):

    _p = ['XX', 'YY']

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    xmin = 1.e10
    xmax =-1.e10

    with h5.File(tnoise_model, 'r') as f:
        #print f.keys()
        tnoise_md = f['Tnoise'][:]
        tnoise_md_freq = f['freq'][:]

    eta_list = []
    for bi, eta_f, eta in abs_cal.iter_avg_eta(result_path, key_list_dict, beam_list, 
            band_list, tnoise_model, pol):
        eta_list.append(eta)

    legend_list = []
    ci = 0
    for date in key_list_dict.keys():

        key_list = key_list_dict[date]
        for beam, path in iter_file_list(result_path + '/%s/'%date, key_list,
                                         beam_list, band_list):

            A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

            ax = axes[beam-1]

            eta = interp1d(tnoise_md_freq, tnoise_md[:, pol, beam-1])(freq) / A[:, pol]
            ax.plot(freq/1.e3, eta, color=_c_list[ci], lw=0.05)

            #_eta = interp1d(eta_f, eta_list[beam-1])(freq)
            #xx = np.linspace(0, 1, freq.shape[0]) #freq[~eta.mask]
            #msk = eta.mask
            #yy = (eta - _eta)[~msk]
            #eta_poly = np.poly1d(np.polyfit(xx[~msk], yy, 2))
            #yy = eta_poly(xx)
            #ax.plot(freq/1.e3, yy + _eta[beam-1], color=_c_list[ci], lw=0.5)

            if freq.min()/1.e3 < xmin: xmin=freq.min()/1.e3
            if freq.max()/1.e3 > xmax: xmax=freq.max()/1.e3

        legend_list.append(mpatches.Patch(color=_c_list[ci],
                                          label=r'%s'%date.replace('_', ' ')))
        ci += 1

    for ii, key, bi, f, eta in abs_cal.iter_fit_eta_days(result_path, key_list_dict, beam_list, 
            band_list, tnoise_model, pol):

        ax = axes[bi]
        ax.plot(f/1.e3, eta, color=_c_list[ii], lw=1.0)

    #for bi, f, eta in abs_cal.iter_avg_eta(result_path, key_list_dict, beam_list, 
    #       band_list, tnoise_model, pol):
    for bi in range(19):

        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        ax.plot(eta_f/1.e3, eta_list[bi], color='k', lw=1.0)
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.51, 0.99)

        if ii == 4:
            ax.set_xlabel('Frequency [GHz]')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel(r'$\eta$')
        else:
            ax.set_yticklabels([])

    fig.legend(handles=legend_list, title = '%s Polarization'%_p[pol],
               frameon=False, loc=4, bbox_to_anchor=(0.94, 0.10), ncol=2)

    if output is not None:
        fig.savefig(output, formate='png', dpi=200)


def plot_fwhm_days(result_path, key_list_dict, beam_list,
                   band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                   fwhm_model=None, pol=0, fmin=None, fmax=None, output=None):

    _p = ['XX', 'YY']

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    if fmin is None: xmin = 1.e10
    if fmax is None: xmax =-1.e10


    legend_list = []
    ci = 0
    for date in key_list_dict.keys():

        key_list = key_list_dict[date]
        for beam, path in iter_file_list(result_path + '/%s/'%date, key_list,
                                         beam_list, band_list):

            A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

            ax = axes[beam-1]
            ax.plot(freq/1.e3, fwhm[:, pol], color=_c_list[ci], lw=0.1)

            if fmin is None:
                if freq.min()/1.e3 < xmin: xmin=freq.min()/1.e3
            if fmax is None:
                if freq.max()/1.e3 > xmax: xmax=freq.max()/1.e3

        legend_list.append(mpatches.Patch(color=_c_list[ci],
                                          label=r'%s'%date.replace('_', ' ')))
        ci += 1

    if fwhm_model is not None:
        data = np.loadtxt(fwhm_model)
        f = data[:, 0] * 1.e-3
        d = data[:, 1:]

    if fmin is None: fmin=xmin
    if fmax is None: fmax=xmax
    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        if fwhm_model is not None:
            ax.plot(f, d[:, bi], 'k.-', lw=0.2)
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)

        ax.set_xlim(fmin, fmax)
        ax.set_ylim(2.1, 3.9)

        if ii == 4:
            ax.set_xlabel('Frequency [GHz]')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel(r'$\theta_{\rm FWHM}$')
        else:
            ax.set_yticklabels([])

    fig.legend(handles=legend_list, title = '%s Polarization'%_p[pol],
               frameon=False, loc=4, bbox_to_anchor=(0.94, 0.10), ncol=2)

    if output is not None:
        fig.savefig(output, formate='png', dpi=200)


def plot_Tnoise_days(result_path, key_list_dict, beam_list,
                     band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                     tnoise_model=None, pol=0, output=None):

    _p = ['XX', 'YY']

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    xmin = 1.e10
    xmax =-1.e10

    with h5.File(tnoise_model, 'r') as f:
        tnoise_md = f['Tnoise'][:]
        tnoise_md_freq = f['freq'][:]

    eta_list = [] #[[], ] * len(key_list_dict.keys())
    #for bi, eta_f, eta in abs_cal.iter_avg_eta(result_path, key_list_dict, beam_list, 
    #        band_list, tnoise_model, pol):
    for ii, key, bi, eta_f, eta in abs_cal.iter_fit_eta_days(result_path, key_list_dict, beam_list, 
            band_list, tnoise_model, pol):
        eta_list.append(eta)

    eta_list = np.array(eta_list)
    eta_list.shape = (len(key_list_dict.keys()), 19, -1)


    legend_list = []
    ci = 0
    for date in key_list_dict.keys():

        key_list = key_list_dict[date]
        for beam, path in iter_file_list(result_path + '/%s/'%date, key_list,
                                         beam_list, band_list):

            A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

            ax = axes[beam-1]


            #eta = interp1d(tnoise_md_freq, tnoise_md[:, pol, beam-1])(freq) / A[:, pol]
            #ax.plot(freq/1.e3, eta, color=_c_list[ci], lw=0.1)
            _eta = interp1d(eta_f, eta_list[ci, beam-1])(freq)
            ax.plot(freq/1.e3, A[:, pol]*_eta, color=_c_list[ci], lw=0.1)


            if freq.min()/1.e3 < xmin: xmin=freq.min()/1.e3
            if freq.max()/1.e3 > xmax: xmax=freq.max()/1.e3

        legend_list.append(mpatches.Patch(color=_c_list[ci],
                                          label=r'%s'%date.replace('_', ' ')))
        ci += 1

    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        ax.plot(tnoise_md_freq/1.e3, tnoise_md[:, pol, bi], color='k', lw=0.1)
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0.7, 1.4)

        if ii == 4:
            ax.set_xlabel('Frequency [GHz]')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel(r'$T_{\rm ND}\,[\rm K]$')
        else:
            ax.set_yticklabels([])

    fig.legend(handles=legend_list, title = '%s Polarization'%_p[pol],
               frameon=False, loc=4, bbox_to_anchor=(0.94, 0.10), ncol=2)

    if output is not None:
        fig.savefig(output, formate='png', dpi=200)

def plot_Tnoise_diff_days(result_path, key_list_dict, beam_list,
                     band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                     tnoise_model=None, pol=0, output=None, eta_path=None):

    _p = ['XX', 'YY']

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=False)

    with h5.File(tnoise_model, 'r') as f:
        tnoise_md = f['Tnoise'][:]
        tnoise_md_freq = f['freq'][:]

    if eta_path is None:
        eta_list = []
        #for bi, eta_f, eta in abs_cal.iter_avg_eta(result_path, key_list_dict, beam_list,
        #        band_list, tnoise_model, pol):
        #    eta_list.append(eta)
        for ii, key, bi, eta_f, eta in abs_cal.iter_fit_eta_days(result_path, 
                key_list_dict, beam_list, band_list, tnoise_model, pol):
            eta_list.append(eta)

        eta_list = np.array(eta_list)
        eta_list.shape = (len(key_list_dict.keys()), 19, -1)


    legend_list = []
    ci = 0
    peak = []
    for date in key_list_dict.keys():

        if eta_path is not None:
            with h5.File(eta_path + 'eta_%s.h5'%date, 'r') as f:
                _eta_list = f['eta'][:, pol, :]
                eta_f = f['eta_f'][:]
        else:
            _eta_list = eta_list[ci]

        _peak = []
        key_list = key_list_dict[date]
        for beam, path_list in abs_cal.iter_beam_list(result_path + '/%s/'%date,
                                                       key_list, beam_list, band_list):

            Tnd_diff = []
            for path in path_list:
                A, mu, fwhm, sigma2, freq = abs_cal.load_fitting_params(path)

                Tnd_th = interp1d(tnoise_md_freq, tnoise_md[:, pol, beam-1])(freq)
                _eta = interp1d(eta_f, _eta_list[beam-1])(freq)
                Tnd = A[:, pol]*_eta

                Tnd_diff.append(Tnd / Tnd_th - 1.)

            ax = axes[beam-1]
            bins = np.linspace(-0.1, 0.1, 101)
            Tnd_diff = np.ma.concatenate(Tnd_diff)
            hist, bins = np.histogram(Tnd_diff[~Tnd_diff.mask], bins=bins)
            ax.plot(bins[:-1], hist/1.e3, color=_c_list[ci], drawstyle='steps-post')
            _peak.append(bins[np.argmax(hist)])


            #ax.plot(freq/1.e3, A[:, pol]*_eta, color=_c_list[ci], lw=0.1)

        _peak = np.array(_peak)[np.argsort(sum(beam_list, []))]
        peak.append(_peak)

        legend_list.append(mpatches.Patch(color=_c_list[ci],
                                          label=r'%s'%date.replace('_', ' ')))
        ci += 1

    for bi in range(19):
        ii = bi / 4
        jj = bi % 4

        ax = axes[bi]
        #ax.plot(tnoise_md_freq/1.e3, tnoise_md[:, pol, bi], color='k', lw=0.1)
        ax.text(0.1, 0.85, 'Feed%02d'%bi, transform=ax.transAxes)
        ax.set_xlim(-0.12, 0.12)
        #ax.semilogy()
        ax.set_ylim(0, 1.5)

        if ii == 4:
            #ax.set_xlabel(r'$\delta T_{\rm ND} [{\rm K}]$')
            ax.set_xlabel(r'$\delta T_{\rm ND}/T_{\rm ND} $')
        else:
            ax.set_xticklabels([])

        if jj == 0:
            ax.set_ylabel(r'$N \times 10^3$')
        else:
            ax.set_yticklabels([])

    fig.legend(handles=legend_list, title = '%s Polarization'%_p[pol],
               frameon=False, loc=4, bbox_to_anchor=(0.94, 0.10), ncol=2)

    if output is not None:
        fig.savefig(output, formate='png', dpi=200)

    return np.array(peak)
