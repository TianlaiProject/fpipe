import numpy as np
import h5py as h5

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
            ax.plot(xx, yy_fit, color='b', lw=2.5)


            yy = vis[~ns, 0, 1]
            ee = np.ma.std(vis[~ns, :, :], axis=1)[:, 1]
            l = ax.plot(xx, yy, '.-', lw=1)
            ax.fill_between(xx, yy-ee, yy+ee, alpha=0.5, color=l[0].get_color())
            yy_fit = np.exp(np.poly1d(fit_params[0][1])(r))
            ax.plot(xx, yy_fit, color='r', lw=2.5)


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



def cal_3C286():

    a0 =  1.2481
    a1 = -0.4507
    a2 = -0.1798
    a3 =  0.0357

    #J2K = 25.6
    #J2K = 22.6
    J2K = source.mJy2K

    Jy2mJy = 1.e3

    return lambda nu: 10**(a0 + a1 * np.log10(nu) + a2 * np.log10(nu)**2
                           + a3 * np.log10(nu)**2) * J2K(nu) * Jy2mJy


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
                 off_threshold=6./60., on_threshold=3./60.):

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

    # calibrate freq-by-freq
    Tnd = []
    fit_params = []
    for fid, f in enumerate(freq):
        if np.all(vis.mask[~ns, :, :][:, fid, :]):
            Tnd.append([0, 0])
            fit_params.append([[0, 0, 0], [0, 0, 0]])
            continue
        cal_T = np.mean(cal_3C286()(f * 1.e-3))
        _yy = vis[~ns, :, :][:, fid, 0]
        _fit_params_xx = fit_beam(r[r<on_threshold], _yy[r<on_threshold])
        Tnd_xx = cal_T / np.exp(_fit_params_xx[-1])

        _yy = vis[~ns, :, :][:, fid, 1]
        _fit_params_yy = fit_beam(r[r<on_threshold], _yy[r<on_threshold])
        Tnd_yy = cal_T / np.exp(_fit_params_yy[-1])

        Tnd.append([Tnd_xx, Tnd_yy])
        fit_params.append([_fit_params_xx, _fit_params_yy])

    return Tnd, mask, r, fit_params, vis

def fit_beam(xx, yy):

    amp = np.polyfit(xx, np.log(yy), deg=2)
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
            mask[j-60:j+60] = True
        spec.mask += mask
        new_mask = np.sum(mask.astype('int')) - nmask
        if new_mask  == 0:
            return mask

    print new_mask
    return mask
