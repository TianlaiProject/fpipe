import multiprocessing as mp
#from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import h5py as h5
from scipy.interpolate import interp1d

from fpipe.timestream import bandpass_cal
import matplotlib.pyplot as plt
from fpipe.point_source import source

from astropy.coordinates.angles import Angle
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u


_Lon = (106. + 51./60. + 24.0/3600.) * u.deg
_Lat = (25. + 39./60. + 10.6/3600.) * u.deg
_Location = EarthLocation.from_geodetic(_Lon, _Lat)


def cali(data_path, data_name, cal_ra, cal_dec, cal_model,
         off_threshold=6./60., on_threshold=3./60., plot=True,
         output=None):

    for o in iter_beams(data_path, data_name):
        ii, bi, vis, msk, ns, ra, dec, freq = o

        print('Beam %02d Freq %6.2f - %6.2f'%(bi, freq[0], freq[-1]))

        Tnd, mask, r, fit_params, vis = cali_onefeed(vis, msk, ns, ra, dec, freq,
                                                     cal_ra, cal_dec, cal_model,
                                                     off_threshold, on_threshold)
        Tnd = np.array(Tnd)
        if output is not None:
            with h5.File(output + '_F%02d.h5'%bi, 'w') as f:
                f['Tnd'] = Tnd
                f['freq'] = freq
                f['freq_mask'] = mask
                f['fit_params'] = np.array(fit_params)
        if plot:
            fig = plt.figure(figsize=(8, 4))
            ax  = fig.add_axes([0.12, 0.12, 0.83, 0.83])

            ra_factor = np.cos(cal_dec.radian)
            cal_on_idx = np.argmin(r)

            ax.axvline(ra_factor * (ra[~ns][cal_on_idx] - cal_ra.deg),
                       0, 1, color='k', ls='-')
            ax.axvline(ra_factor * (ra[~ns][r<=off_threshold][0] - cal_ra.deg),
                       0, 1, color='k', ls='--')
            ax.axvline(ra_factor * (ra[~ns][r<=off_threshold][-1] - cal_ra.deg),
                       0, 1, color='k', ls='--')
            ax.axvline(ra_factor * (ra[~ns][r<=on_threshold][0]  - cal_ra.deg),
                       0, 1, color='g', ls='--')
            ax.axvline(ra_factor * (ra[~ns][r<=on_threshold][-1] - cal_ra.deg),
                       0, 1, color='g', ls='--')

            xx = (ra[~ns] - cal_ra.deg) * ra_factor
            #yy = np.ma.mean(vis[~ns, :, :], axis=1)[:, 0]
            yy = vis[~ns, 0, 0]
            ee = np.ma.std(vis[~ns, :, :], axis=1)[:, 0]
            l = ax.plot(xx, yy, '.-', lw=1)
            ax.fill_between(xx, yy-ee, yy+ee, alpha=0.5, color=l[0].get_color())
            yy_fit = np.exp(np.poly1d(fit_params[0][0])(r))
            ax.plot(xx, yy_fit, color='b', lw=2.5, ls='--')


            yy = vis[~ns, 0, 1]
            ee = np.ma.std(vis[~ns, :, :], axis=1)[:, 1]
            l = ax.plot(xx, yy, '.-', lw=1)
            ax.fill_between(xx, yy-ee, yy+ee, alpha=0.5, color=l[0].get_color())
            yy_fit = np.exp(np.poly1d(fit_params[0][1])(r))
            ax.plot(xx, yy_fit, color='r', lw=2.5, ls='--')


            ax.set_ylim(ymax=800, ymin=1.e-2)
            ax.set_xlim(-0.4, 0.4)

            ax.semilogy()

            if output is not None:
                plt.savefig(output+'_F%02d.png'%bi)

            #plt.show()


            #fig = plt.figure(figsize=(10, 3))
            #ax  = fig.add_axes([0.12, 0.15, 0.83, 0.80])
            #Tnd = np.ma.masked_equal(Tnd, 0)
            #ax.plot(freq, Tnd[:, 0], lw=0.1)
            #ax.plot(freq, Tnd[:, 1], lw=0.1)
            #ax.set_ylim(0.5, 2.0)
            #plt.show()



def cal_3C286(fwhm=None):

    a0 =  1.2481
    a1 = -0.4507
    a2 = -0.1798
    a3 =  0.0357

    #J2K = 25.6
    #J2K = 22.6
    J2K = source.mJy2K

    Jy2mJy = 1.e3

    return lambda nu: 10**(a0 + a1 * np.log10(nu) + a2 * np.log10(nu)**2
                           + a3 * np.log10(nu)**2) * J2K(nu, fwhm=fwhm) * Jy2mJy


def iter_beams(data_path, data_name):

    with h5.File(data_path + data_name, 'r') as f:

        vis_all = f['vis'][...,0:2, :]
        msk_all = f['vis_mask'][...,0:2, :]
        ns_all  = f['ns_on'][:, :].astype('bool')
        #print f.keys()
        beam_id = f['feedno'][:]
        #print f.attrs.keys()

        freq = np.arange(f.attrs['nfreq']) * f.attrs['freqstep'] + f.attrs['freqstart']

        ra_all  = f['ra'][:, :]
        dec_all = f['dec'][:, :]

    for bi, beam in enumerate(beam_id):

        yield bi, beam, vis_all[..., bi], msk_all[..., bi], ns_all[..., bi], \
              ra_all[..., bi], dec_all[..., bi], freq


def cali_onefeed(vis, msk, ns, ra, dec, freq, cal_ra, cal_dec, cal_model,
                 off_threshold=6./60., on_threshold=3./60., return_vis_only=False,
                 nworker=32):

    vis = np.ma.array(vis, mask=msk)

    # calibrate to noise diode power
    nd, ns= bandpass_cal.get_Ncal(vis, msk, ns, on_t=1, off_t=2)
    vis.mask += ns[:, None, None]
    nd = np.median(nd, axis=0)
    nd_smooth = bandpass_cal.smooth_bandpass(nd, axis=0)
    vis /= nd_smooth[None, :, :]

    # find the peak
    ra = ra.astype('float64')[~ns]
    dec = dec.astype('float64')[~ns]
    r = np.sin(np.radians(dec)) * np.sin(cal_dec.radian)\
      + np.cos(np.radians(dec)) * np.cos(cal_dec.radian)\
      * np.cos(np.radians(ra) - cal_ra.radian)
    r[r>1] = 1.
    r = np.arccos(r) * 180./np.pi
    cal_on_idx = np.argmin(r)

    # subtract source off
    vis_source_off = np.ma.median(vis[~ns, ...][r>off_threshold], axis=0)
    vis -= vis_source_off[None, ...]

    # flag RFI in freq
    _vis = vis.copy()
    _vis.mask[~ns] += (r<=off_threshold)[:, None, None]
    mask = flag_freq(_vis)
    vis.mask += mask[None, :, None]

    if r[cal_on_idx] > 1.5/60.:
        msg = 'Cal Source off pointing'
        raise ValueError(msg)

    if return_vis_only:
        return vis, mask

    # calibrate freq-by-freq
    #print('calibrate freq-by-freq')

    Tnd = []
    fit_params = []
    args = (vis[~ns, ...], freq, r, on_threshold)
    #with ThreadPoolExecutor(max_workers=nworker) as p:
    with mp.Pool(16) as p:
        result = p.map(partial(_cal_func, args=args), [fid for fid in range(len(freq))])
    for _r in result:
        Tnd_xx, Tnd_yy, _fit_params_xx, _fit_params_yy = _r
        Tnd.append([Tnd_xx, Tnd_yy])
        fit_params.append([_fit_params_xx, _fit_params_yy])

    #Tnd = []
    #fit_params = []
    #for fid, f in enumerate(freq):
    #    if np.all(vis.mask[~ns, :, :][:, fid, :]):
    #        Tnd.append([0, 0])
    #        fit_params.append([[0, 0, 0], [0, 0, 0]])
    #        continue

    #    _on_threshold = on_threshold * 1000./f
    #    cal_T = np.mean(cal_3C286()(f * 1.e-3))
    #    _yy = vis[~ns, :, :][:, fid, 0]
    #    _fit_params_xx = fit_beam(r[r<_on_threshold], _yy[r<_on_threshold])
    #    Tnd_xx = cal_T / np.exp(_fit_params_xx[-1])

    #    _yy = vis[~ns, :, :][:, fid, 1]
    #    _fit_params_yy = fit_beam(r[r<_on_threshold], _yy[r<_on_threshold])
    #    Tnd_yy = cal_T / np.exp(_fit_params_yy[-1])

    #    Tnd.append([Tnd_xx, Tnd_yy])
    #    fit_params.append([_fit_params_xx, _fit_params_yy])

    return Tnd, mask, r, fit_params, vis

def _cal_func(fid, args):

    vis, freq, r, on_threshold = args

    f = freq[fid]
    _on_threshold = on_threshold #* 1000./f
    cal_T = np.mean(cal_3C286()(f * 1.e-3))

    if np.all(vis[:, fid, :].mask):
        return 0, 0, [0, 0, 0], [0, 0, 0]

    _yy = vis[:, fid, 0][r<_on_threshold]
    if np.all(_yy<=0):
        _fit_params_xx = [0, 0, 0]
        Tnd_xx = 0
    else:
        _fit_params_xx = fit_beam(r[r<_on_threshold], _yy)
        Tnd_xx = cal_T / np.exp(_fit_params_xx[-1])

    _yy = vis[:, fid, 1][r<_on_threshold]
    if np.all(_yy<=0):
        _fit_params_yy = [0, 0, 0]
        Tnd_yy = 0
    else:
        _fit_params_yy = fit_beam(r[r<_on_threshold], _yy)
        Tnd_yy = cal_T / np.exp(_fit_params_yy[-1])

    return Tnd_xx, Tnd_yy, _fit_params_xx, _fit_params_yy

def fit_beam(xx, yy):

    good = yy > 0
    amp = np.polyfit(xx[good], np.log(yy[good]), deg=2)
    #print amp

    return amp

def flag_freq(vis):

    spec = np.ma.mean(vis, axis=(0, 2))
    mask = np.zeros(spec.shape, dtype='bool')
    for i in range(20):
        nmask = np.sum(mask.astype('int'))
        sig = np.ma.std(spec)
        mean = np.ma.mean(spec)
        bad = np.abs(spec) > 5*sig
        for j in np.arange(spec.shape[0])[bad]:
            mask[j-100:j+100] = True
        spec.mask += mask
        new_mask = np.sum(mask.astype('int')) - nmask
        if new_mask  == 0:
            #print("mask loop %02d"%i)
            return mask

    print(new_mask)
    return mask

def load_fitting_params(path):

    with h5.File(path, 'r') as f:
        Tnd = f['Tnd'][:]
        freq = f['freq'][:]
        freq_mask = f['freq_mask'][:]
        fit_params = f['fit_params'][:]

    Tnd = np.ma.array(Tnd, mask=False)
    Tnd.mask += freq_mask[:, None]

    a = fit_params[:, :, 2]
    b = fit_params[:, :, 1]
    c = fit_params[:, :, 0]

    mask = c==0
    c[mask] = np.inf
    sigma2 = np.ma.array(-1./c/2., mask=mask)
    mu = np.ma.array(- b/c/2., mask=mask)
    A = np.ma.array(np.exp(a - b**2/4./c), mask=mask)

    fwhm = (sigma2**0.5) * 2. * ( 2. * np.log(2.) ) **0.5
    fwhm = fwhm * 60.

    #fwhm_xx = interp1d(freq * 1.e-3, fwhm[:, 0], fill_value="extrapolate", axis=0)
    #fwhm_yy = interp1d(freq * 1.e-3, fwhm[:, 1], fill_value="extrapolate", axis=0)

    fwhm_xx = None
    fwhm_yy = None

    cal_T_xx = cal_3C286(fwhm_xx)(freq * 1.e-3)
    cal_T_yy = cal_3C286(fwhm_yy)(freq * 1.e-3)

    A[:, 0] = cal_T_xx / A[:, 0]
    A[:, 1] = cal_T_yy / A[:, 1]

    A = np.ma.array(A, mask=Tnd.mask)
    mu = np.ma.array(mu, mask=Tnd.mask) * 60.
    sigma2 = np.ma.array(sigma2, mask=Tnd.mask)

    return A, mu, fwhm, sigma2, freq

def iter_beam_list(result_path, key_list, beam_list,
                   band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz']):

    for kk, key in enumerate(key_list):
        for beam in beam_list[kk]:
            result_name = []
            for suffix in band_list:
                result_name.append(result_path + '%s%s_F%02d.h5'%(key, suffix, beam))

            yield beam, result_name


def load_eta(result_path, key_list_dict, beam_list,
                band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                tnoise_model=None):

    with h5.File(tnoise_model, 'r') as f:
        tnoise_md = f['Tnoise'][:]
        tnoise_md_freq = f['freq'][:]

    r = []
    f = None
    for date in list(key_list_dict.keys()):
        key_list = key_list_dict[date]

        #beam_list = []
        eta_list = []

        for beam, path_list in iter_beam_list(result_path + '/%s/'%date, key_list,
                                              beam_list, band_list):

            _eta = []
            _f = []
            for path in path_list:
                A, mu, fwhm, sigma2, freq = load_fitting_params(path)
                eta = interp1d(tnoise_md_freq, tnoise_md[..., beam-1], axis=0)(freq) / A
                _eta.append(eta)
                _f.append(freq)
                #print eta.shape
            _eta = np.ma.concatenate(_eta, axis=0)
            _f = np.concatenate(_f)
            if f is None: f = _f

            eta_list.append(_eta[None, ...])

        eta_list = np.ma.concatenate(eta_list, axis=0)
        argsort = np.argsort(sum(beam_list, []))
        eta_list = eta_list[argsort]
        r.append(eta_list[None, ...])

    r = np.ma.concatenate(r, axis=0)
    return r, f

def iter_avg_eta(result_path, key_list_dict, beam_list,
                band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
                tnoise_model=None, pol=0):

    eta, f = load_eta(result_path, key_list_dict, beam_list, band_list, tnoise_model)
    eta_avg = np.ma.median(eta, axis=0)
    for bi in range(19):
        xx = np.linspace(0, 1, f.shape[0]) #freq[~eta.mask]
        msk = eta_avg[bi, :, pol].mask
        yy = eta_avg[bi, :, pol][~msk]
        eta_poly = np.poly1d(np.polyfit(xx[~msk], yy, 15))
        #yy = np.ma.array(eta_poly(freq), mask=eta.mask)
        yy = np.ma.array(eta_poly(xx), mask=False)
        yield bi, f, yy

def iter_fit_eta_days(result_path, key_list_dict, beam_list,
        band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
        tnoise_model=None, pol=0):

    eta, f = load_eta(result_path, key_list_dict, beam_list, band_list, tnoise_model)
    eta_avg = np.ma.median(eta, axis=0)
    eta_model = []
    xx = np.linspace(0, 1, f.shape[0]) #freq[~eta.mask]
    for bi in range(19):
        #xx = np.linspace(0, 1, f.shape[0]) #freq[~eta.mask]
        msk = eta_avg[bi, :, pol].mask
        yy = eta_avg[bi, :, pol][~msk]
        eta_poly = np.poly1d(np.polyfit(xx[~msk], yy, 15))
        #yy = np.ma.array(eta_poly(freq), mask=eta.mask)
        yy = np.ma.array(eta_poly(xx), mask=False)
        eta_model.append(yy)
        #yield bi, f, yy

    #for ii in range(eta.shape[0]):
    for ii, key in enumerate(key_list_dict.keys()):
        for bi in range(19):
            _eta = eta[ii, bi, :, pol]
            _msk = eta[ii, bi, :, pol].mask

            yy = (_eta - eta_model[bi])[~_msk]
            eta_poly = np.poly1d(np.polyfit(xx[~_msk], yy, 3))
            yy = np.ma.array(eta_poly(xx), mask=False)

            yield ii, key, bi, f, yy + eta_model[bi]

def output_fit_eta(result_path, key_list_dict, output_path, beam_list,
        band_list=['_1050-1150MHz', '_1150-1250MHz', '_1250-1450MHz'],
        tnoise_model=None):

    eta_xx = []
    for ii, key, bi, eta_f, eta in iter_fit_eta_days(result_path, key_list_dict, beam_list, 
            band_list, tnoise_model, 0):
        eta_xx.append(eta)
    eta_xx = np.array(eta_xx)
    eta_xx.shape = (len(list(key_list_dict.keys())), 19, -1)
            

    eta_yy = []
    for ii, key, bi, eta_f, eta in iter_fit_eta_days(result_path, key_list_dict, beam_list, 
            band_list, tnoise_model, 1):
        eta_yy.append(eta)
    eta_yy = np.array(eta_yy)
    eta_yy.shape = (len(list(key_list_dict.keys())), 19, -1)


    eta = np.concatenate([eta_xx[:, :, None, :], eta_yy[:, :, None, :]], axis=2)

    for ii, key in enumerate(key_list_dict):
        with  h5.File(output_path + 'eta_%s.h5'%key, 'w') as f:
            f['eta'] = eta[ii]
            f['eta_f'] = eta_f












