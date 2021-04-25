import numpy as np
import h5py

from fpipe.timestream import timestream_task
from scipy import signal
from scipy.interpolate import interp1d

from fpipe.container.timestream import FAST_Timestream

from fpipe.timestream import bandpass_cal

import matplotlib.pyplot as plt


class TimeVar_Cal(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'gt_file' : None,
            'fk' : 0.01,
            'alpha' : 1.5,
            'l' : 5,
            }

    prefix = 'gtcal_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(TimeVar_Cal, self).process(ts)

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        fk    = self.params['fk']
        alpha = self.params['alpha']
        l     = self.params['l']

        with h5py.File(self.params['gt_file'], 'r') as f:
            nd= f['gtgnu'][:, :, :, bl[0]-1]
            freq  = f['freq'][:]
            nd_time  = f['time'][:]
            print 'Feed%02d '%(bl[0])

            #bandpass_interp = interpolate.interp1d(_bandfreq, _bandpass, axis=0)
            #bandpass = bandpass_interp(ts['freq'][:])

        time = ts['sec1970'][:]
        nd = np.ma.masked_invalid(nd)

        # ignore freq between 1200-1300 due to strong RFI residule
        nd.mask[:, 200:600, :] = True
        gt  = np.ma.median(nd, axis=1)
        var = np.ma.median(nd, axis=1)

        gt_m = np.ma.mean(gt, axis=0)
        gt_s = np.ma.std(gt, axis=0)
        # there are still some RFI, flag peaks ove 3 sigma
        good = np.abs(gt - gt_m[None, :]) - 3 * gt_s[None, :] < 0.

        # remove mean before destriping
        gt = gt - gt_m[None, :]

        gt_xx = destriping(l, gt[good[:, 0], 0], var[good[:, 0], 0], 
                nd_time[good[:, 0]], fk, alpha)(time).flatten()
        gt_yy = destriping(l, gt[good[:, 1], 1], var[good[:, 1], 1], 
                nd_time[good[:, 1]], fk, alpha)(time).flatten()

        #plt.plot(gt_xx, label='Feed%02d'%bl[0])
        #plt.plot(gt_yy)
        #plt.legend()
        #plt.show()

        # add mean back after destriping
        gt_xx = gt_xx + gt_m[0]
        gt_yy = gt_yy + gt_m[1]

        gt_xx[gt_xx==0] = np.inf
        gt_yy[gt_yy==0] = np.inf

        vis[:, :, 0] /= gt_xx[:, None]
        vis[:, :, 1] /= gt_yy[:, None]


def Ctt(fk, alpha, time):
    '''
    Keihanen et, al. 2005, appendix A1.
    '''

    sigma2 = 1.
    fs = 1.

    fmin = 0.00001
    fmax = 1000.0
    gamma = -alpha
    gk = np.logspace(np.log10(fmin), np.log10(fmax), 1000)
    D  = np.log(gk[1] / gk[0])
    bk = sigma2 / fs / np.pi * (2.*np.pi*fk)**(-gamma)\
            * np.sin((gamma+2)*np.pi/2.) * gk**(gamma+1) * D
    gk = gk[:, None]
    bk = bk[:, None]
    ct = lambda t: np.sum(bk * np.exp(-gk * t[None, :]), axis=0)

    tt = time - time[0]

    c = ct(tt)
    #c[0] += 1

    C_tt = np.zeros(c.shape * 2)

    for i in range(C_tt.shape[0]):
        C_tt[i] = np.roll(c, i)

    C_tt = np.triu(C_tt, 0) + np.triu(C_tt, 1).T

    return C_tt

def evaluate_cov_with_fn_realization(alpha, fk, time, N=1000):

    ntime = time.shape[0]
    dtime = time[1] - time[0]

    gamma = -alpha
    P = lambda f: (f/fk)**gamma

    f_axis = np.fft.fftfreq( ntime, d=dtime)
    f_abs = np.abs(f_axis)
    f_abs[f_abs==0] = np.inf

    f0 = fk

    fn_psd = P(f_abs)

    fn_psd[0] = 0.
    shp = (N, ) + fn_psd.shape

    fn_psd /= 2.0
    fn_psd = np.sqrt(fn_psd / dtime) #  * ntime * _n * nfreq)

    fn_k = np.random.standard_normal(shp) + 1.J * np.random.standard_normal(shp)
    fn_k *= fn_psd[None, :]

    fn_r = np.fft.irfft(fn_k, axis=1, norm='ortho')

    fsel = slice(ntime)
    fn_r = fn_r[:, fsel]

    return  np.cov(fn_r, rowvar=False)


def destriping(l, vis, var, time, fk, alpha):

    n_time = time.shape[0]
    I = np.mat(np.eye(n_time))

    N = 1./var
    N = np.mat(N * np.eye(n_time))

    L = int(np.ceil(n_time / float(l)))
    F = np.zeros((n_time, L))
    for ii, st in enumerate(range(0, n_time, l)):
        F[st:st+l, ii] = 1.
    F = np.mat(F)

    y = np.mat(vis[:, None])
    #y -= np.mean(y)
    
    Zy = y

    if fk is not None:
        #C_tt = Ctt(fk, alpha, time) * var
        C_tt = evaluate_cov_with_fn_realization(alpha, fk, time) * var
        C_tt = np.mat(C_tt)
        Ca = ((F.T * F).I)**2 * F.T * C_tt * F
        #Ca = F.T * C_tt * F
        a = (F.T * N * F + Ca.I).I * F.T * N * Zy
    else:
        a = (F.T * N * F).I * F.T * N * Zy
        
    return interp1d(time, (F*a).flatten(), bounds_error=False, fill_value='extrapolate')

def est_gain(input_files, beam=0, freq_idx=slice(0, None)):

    ts = FAST_Timestream(input_files)
    ts.load_all()

    i = beam
    vis = ts['vis'].local_data[..., i]
    msk = ts['vis_mask'].local_data[..., i]
    ns_on = ts['ns_on'].local_data[..., i]

    time = ts['sec1970'][:]
    time0 = ts.attrs['sec1970']
    #print ts.attrs['sec1970']
    #print ts.attrs['obstime']

    vis = np.ma.array(vis, mask=msk)

    on_t = 1

    off_l = np.sum((~ns_on).astype('bool'))
    gain = np.zeros([off_l, 2])

    on_l = np.sum((ns_on).astype('bool'))
    gTnd = np.ma.zeros([on_l, 2])

    # get T_nd_on - T_nd_off = g * T_nd
    _gTnd, ns = bandpass_cal.get_Ncal(vis, msk, ns_on, on_t, off_t=1, cut_end=False)

    #print _gTnd.shape
    # average across freq

    gTnd       = np.ma.mean(_gTnd[:, freq_idx, ...], axis=1)
    gTnd.mask  = np.all(_gTnd[:, freq_idx, ...].mask, axis=1)
    #print gTnd

    gTnd_mean = np.ma.mean(gTnd, axis=0)
    gTnd -= gTnd_mean[None, :]

    var = np.ones_like(gTnd) * np.var(gTnd, axis=0)[None, :]

    fk    = 0.01
    alpha = 2.0

    gain[:, 0] = destriping(10, gTnd[:, 0], var[:, 0], time[ns_on], fk, alpha)(time[~ns_on])
    gain[:, 1] = destriping(10, gTnd[:, 1], var[:, 1], time[ns_on], fk, alpha)(time[~ns_on])

    gTnd += gTnd_mean[None, :]
    gain += gTnd_mean[None, :]

    return time + time0, gain, gTnd, ns_on, vis

