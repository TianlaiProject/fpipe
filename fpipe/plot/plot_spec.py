import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.signal import medfilt
from scipy.signal import convolve2d

from fpipe.plot import plot_map as pm

from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import NearestNDInterpolator

import h5py as h5

def mJy2K(freq, eta=1., beam_off=0):

    data = np.loadtxt('/users/ycli/code/fpipe/fpipe/data/fwhm.dat')
    f = data[4:, 0] * 1.e-3
    d = data[4:, 1:]
    fwhm = interp1d(f, np.mean(d, axis=1), fill_value="extrapolate")

    _lambda = 2.99e8 / (freq * 1.e9)
    #_sigma = 1.02 *  _lambda / 300. / np.pi * 180. * 3600.
    _sigma = fwhm(freq) * 60.

    if beam_off != 0:
        fact = np.exp(-0.5 * beam_off**2 / (_sigma/2./(2.*np.log(2.))**0.5 )**2 )
    else:
        fact = 1.

    return eta * 1.36 * (_lambda*1.e2)**2. / _sigma**2. * fact


def get_calibrator_spec(freq, cal_data_path='', cal_data=None, 
        cal_param=None, ra=None, dec=None, beam_off=0):

    if cal_param is None:
        if cal_data is None:
            cal_data = np.loadtxt(cal_data_path)
        cal_spec_func = np.poly1d(np.polyfit(np.log10(cal_data[:,0]*1.e-9),
                                             np.log10(cal_data[:,1]*1.e3),
                                             deg=2,))
                                             #w = 1./ cal_data[:,2]))
                                             #w = 1./ np.log10(cal_data[:,2])))

        cal_flux = 10. ** cal_spec_func(np.log10(freq)) # in mJy
    else:
        a, nu, idx = cal_param
        cal_flux = (10 ** a) * ((freq / nu) ** (idx) )

    #kBol = 1.38e6 # mJy m^2 /K
    #eta   = 1. #0.6
    #_lambda = 2.99e8 / (freq * 1.e9)
    #_sigma = 1.02 * _lambda / 300. / 2. / (2. * np.log(2.))**0.5
    ##_sigma[:] = _sigma.max()
    ##_sigma = _lambda / 300. / 2. / (2. * np.log(2.))**0.5
    #Aeff   = eta * _lambda ** 2. / (2. * np.pi * _sigma ** 2.)
    #mJy2K  = Aeff / 2. / kBol

    #mJy2K  = np.pi * 300. ** 2. / 8. / kBol  * 0.9
    #print mJy2K

    if ra is not None:
        with h5.File('/scratch/users/ycli/fanalysis/etaA_map.h5', 'r') as f:
            etaA = f['eta_A'][:]
            _ra   = f['ra'][:]
            _dec  = f['dec'][:]
        _dec, _ra = np.meshgrid(_dec, _ra)
        _dec = _dec.flatten()[:, None]
        _ra  = _ra.flatten()[:, None]
        etaA = etaA.flatten()
        good = etaA != 0
        etaA = etaA[good]
        coord = np.concatenate([_ra[good, :], _dec[good, :]], axis=1)
        
        etaAf = NearestNDInterpolator(coord, etaA)

        eta = etaAf(ra, dec)
        print '[%f, %f] eta = %f'%(ra, dec, eta)
    else:
        eta   = 0.6 #0.9

    #print 'Jy2K : %f'%(mJy2K[-1] * 1.e3)
    cal_T = cal_flux * mJy2K(freq, eta, beam_off) # in K

    return cal_T #, cal_T * factor

def get_source_spec_group(source, map_path='', map_name_group=['',], 
        smoothing=False, label_list=None, n_rebin=1, output=None,
        offidx=(5, 6)):

    if label_list is None:
        label_list = ['%d'%ii for ii in range(len(map_name_group))]

    fig = plt.figure(figsize=[8, 3])
    ax  = fig.add_axes([0.1, 0.18, 0.85, 0.78])

    for ii, map_list in enumerate(map_name_group):
        label = [label_list[ii], ] + [None, ] * (len(map_list) - 1)
        c_list = [pm._c_list[ii], ] * len(map_list)
        freq = get_source_spec(source, map_path=map_path, map_name_list=map_list,
                smoothing=smoothing, c_list = c_list, 
                label_list=label, n_rebin=n_rebin, figaxes=[fig, ax],
                output=output + '_%s'%label_list[ii], offidx=offidx)

    s_path = source['path']
    s_ra = source['ra']
    s_dec = source['dec']
    cal_T = get_calibrator_spec(freq, s_path, ra=s_ra, dec=s_dec)
    ax.plot(freq, cal_T, 'k--', label=source['name'])

    ax.set_ylim(ymin=-0.5, ymax=cal_T.max() * 1.5)

    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('T [K]')

    ax.legend(loc=1)
    if output is not None:
        fig.savefig(output + '_spec.pdf', formate='pdf')

def get_source_spec(source, map_path='', map_name_list=['',], smoothing=True,
        c_list = None, label_list=None, n_rebin=64, figaxes=None, output=None,
        offidx = (5, 6)):

    if c_list is None:
        c_list = ['r',]  * len(map_name_list)
    if label_list is None:
        label_list = [None, ] * len(map_name_list)

    s_ra = source['ra']
    s_dec = source['dec']

    if figaxes is None:
        fig = plt.figure(figsize=[8, 3])
        ax  = fig.add_axes([0.1, 0.18, 0.85, 0.78])
    else:
        fig, ax = figaxes
    freq_min = 1.e9
    freq_max = -1.e9
    df = 0.001

    fig_map = plt.figure(figsize=[8, 3])
    gs = gridspec.GridSpec(1, len(map_name_list), left=0.1, bottom=0.1, 
            right=0.95, top=0.95, wspace=0.05, hspace=0.0)

    for mm, map_name in enumerate( map_name_list ):
        _c = c_list[mm]
        for _map_name in map_name:
            try:
                imap, ra, dec, ra_edges, dec_edges, freq, mask\
                    = pm.load_maps(map_path, _map_name, 'clean_map')
                nmap, ra, dec, ra_edges, dec_edges, freq, mask\
                    = pm.load_maps(map_path, _map_name, 'noise_diag')

                ra_idx = np.digitize(s_ra, ra_edges) - 1
                dec_idx = np.digitize(s_dec, dec_edges) - 1

                #if ra_idx == -1 or ra_idx == imap.shape[1]:
                #    print "source %s out side of map ra range"%source['name']
                #    continue

                #if dec_idx == -1 or dec_idx == imap.shape[2]:
                #    print "source %s out side of map dec range"%source['name']
                #    continue

                if (not ( ra_idx == -1 or  ra_idx == imap.shape[1])) and \
                   (not (dec_idx == -1 or dec_idx == imap.shape[2])):

                    break

            except IOError:
                print 'Map not found %s'%map_name
                continue

        if ra_idx == -1 or ra_idx == imap.shape[1]:
            print "source %s out side of map ra range"%source['name']
            continue

        if dec_idx == -1 or dec_idx == imap.shape[2]:
            print "source %s out side of map dec range"%source['name']
            continue

        if smoothing:
            map_mask = imap == 0.
            _pix = np.abs(dec[1] - dec[0]) * 60.
            pm.smooth_map(imap, _pix, freq)
            imap[map_mask] = 0.

        freq = freq / 1.e3
        #print 'Freq. range %f - %f'%( freq.min(), freq.max())

        if freq.min() < freq_min: freq_min = freq.min()
        if freq.max() > freq_max: freq_max = freq.max()

        #_sig = 3./(8. * np.log(2.))**0.5 / 1.
        #imap = gf(imap, [1, _sig, _sig])
        #imap = gf(imap, [2, 1, 1])


        spec = imap[:, ra_idx, dec_idx]
        nois = nmap[:, ra_idx, dec_idx]

        ra_cent  = ra[ra_idx]
        dec_cent = dec[dec_idx]
        ra_off   = ra[ra_idx-offidx[0]]
        dec_off  = dec[dec_idx-offidx[1]]
        ra_delta = np.abs(ra[1] - ra[0])
        dec_delta= np.abs(dec[1]- dec[0])
        ra_st = max(ra_idx-10, 0)
        ra_ed = min(ra_idx+11, imap.shape[1]-1)
        dec_st = max(dec_idx - 10, 0)
        dec_ed = min(dec_idx + 11, imap.shape[2]-1)
        _imap = np.ma.masked_equal(imap[:, ra_st:ra_ed, dec_st:dec_ed], 0)
        ax_map = fig_map.add_subplot(gs[0, mm])
        ax_map.pcolormesh(ra_edges[ra_st:ra_ed+1], dec_edges[dec_st:dec_ed+1],
                np.ma.mean(_imap, axis=0).T, cmap='Blues')
        ax_map.plot(s_ra, s_dec, 'gx', mfc='none', mew=1.5)
        ax_map.plot(ra_cent, dec_cent, 'r+', mfc='none', mew=1.5)
        ax_map.plot(ra_off, dec_off,  'k+', mfc='none', mew=1.5)
        ax_map.set_xlim(ra_cent - ra_delta*10, ra_cent + ra_delta*10)
        ax_map.set_ylim(dec_cent - dec_delta*10, dec_cent + dec_delta*10)
        ax_map.ticklabel_format(axis='x', style='plain', useOffset=0)
        ax_map.set_aspect('equal')
        ax_map.set_title('%3.2f-%3.2fGHz'%(freq.min(), freq.max()))
        ax_map.set_xlabel('R.A.(J2000)')
        if mm !=0:
            ax_map.set_yticklabels([])
        else:
            ax_map.set_ylabel('Dec(J2000)')

        #nois[nois<0.1] = 0
        spec[nois==0] = 0

        spec = np.ma.masked_equal(spec, 0)
        nois = np.ma.masked_equal(nois, 0)

        #spec[~spec.mask] = medfilt(spec[~spec.mask], 25)
        #ax.plot(freq, spec, '.', color='0.5', zorder=-1000)

        #n_rebin = 64
        if n_rebin != 1:
            freq_rebin = np.mean(freq.reshape(-1, n_rebin), axis=1)
            spec_rebin = spec.reshape(-1, n_rebin)
            nois_rebin = nois.reshape(-1, n_rebin)

            freq_rebin += df * mm/2

            spec_error = np.std(spec_rebin, axis=1)

            nois_rebin[nois_rebin==0] = np.inf
            nois_rebin = 1./nois_rebin

            spec_rebin = np.sum(spec_rebin * nois_rebin, axis=1)

            norm = np.sum(nois_rebin, axis=1)
            norm[norm==0] = np.inf
            spec_rebin /= norm
        else:
            freq_rebin = freq
            spec_rebin = spec
            spec_error = np.zeros_like(spec)

        ax.errorbar(freq_rebin, spec_rebin, yerr=spec_error,
                     fmt= 'o', color=_c, mfc='w', mec=_c, ms=3,
                     label=label_list[mm])

        # plot off
        spec = imap[:, ra_idx - offidx[0], dec_idx - offidx[1]]
        nois = nmap[:, ra_idx - offidx[0], dec_idx - offidx[1]]

        #nois[nois<0.1] = 0
        spec[nois==0] = 0

        spec = np.ma.masked_equal(spec, 0)
        nois = np.ma.masked_equal(nois, 0)

        #spec[~spec.mask] = medfilt(spec[~spec.mask], 25)
        #ax.plot(freq, spec, '.', color='0.5', zorder=-1000)

        #n_rebin = 64
        freq_rebin = np.mean(freq.reshape(-1, n_rebin), axis=1)
        spec_rebin = spec.reshape(-1, n_rebin)
        nois_rebin = nois.reshape(-1, n_rebin)

        spec_error = np.std(spec_rebin, axis=1)

        nois_rebin[nois_rebin==0] = np.inf
        nois_rebin = 1./nois_rebin

        spec_rebin = np.sum(spec_rebin * nois_rebin, axis=1)

        norm = np.sum(nois_rebin, axis=1)
        norm[norm==0] = np.inf
        spec_rebin /= norm

        ax.errorbar(freq_rebin, spec_rebin, yerr=spec_error,
                     fmt='o', mec='0.5', mfc='none', ms=3)

    if output is not None:
        #fig_map.suptitle(label_list[0])
        fig_map.savefig(output + '_submaps.pdf', formate='pdf')

    if figaxes is None:
        s_path = source['path']
        freq = np.linspace(freq_min, freq_max, 1000)
        cal_T = get_calibrator_spec(freq, s_path, ra=s_ra, dec=s_dec)
        ax.plot(freq, cal_T, 'k--', label=source['name'])

        ax.set_ylim(ymin=-0.5)

        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('T [K]')

        ax.legend(loc=1)
    else:
        return np.linspace(freq_min, freq_max, 1000)

def check_spec(source_list, output=None):

    fig = plt.figure(figsize=[8, 3])
    ax  = fig.add_axes([0.1, 0.18, 0.85, 0.78])

    x = np.logspace(np.log10(0.05), np.log10(10), 100)

    for source in source_list:
        cal_path = source['path']
        cal_name = source['name']


        cal_data = np.loadtxt(cal_path)
        cal_spec_func = np.poly1d(np.polyfit(np.log10(cal_data[:,0]*1.e-9),
                                             np.log10(cal_data[:,1]*1.e3),
                                             deg=2,))
        cal_flux = 10. ** cal_spec_func(np.log10(x)) # in mJy
        _l = ax.plot(x, cal_flux, '-', label=cal_name)
        _c = _l[0].get_color()
        ax.errorbar(cal_data[:, 0]*1.e-9, cal_data[:, 1]*1.e3, cal_data[:,2]*1.e3,
                fmt='o', ecolor=_c, mec=_c, mfc=_c)

    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Flux [mJy]')
    ax.legend()
    ax.loglog()

    if output is not None:
        fig.savefig(output + '.pdf', formate='pdf')

