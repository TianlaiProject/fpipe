'''
'''

import numpy as np 
from scipy.optimize import least_squares, curve_fit

from fpipe.timestream import bandpass_cal

import h5py as h5

#def fit_fn(f, ps, ps_err, pars = [1.e-3, 1.e-3, 1.5]):
def fit_fn(f, ps, ps_err, pars = [0, -3, 1.5, 100., -3.5, -4], 
        bounds=([-4, -5, 0.1, 0, -3.6, -5], [0, -1, 6.99, 150, -3.4, -3])):

    if np.all(ps_err==0):
        ps_err = np.ones_like(ps) * 10.
        ps_err[ps<=0] = 0.

    good  = ps_err > 0
    good[:2] = False

    _f = np.logspace(np.log10(f.min()), np.log10(f.max()), 100)

    func = lambda p, f: (10.**p[0]) * ( 1. + ((10.**p[1])/f)**p[2])
    func2= lambda p, f: (10.**p[0])*p[3] / (1. + ((f - 10.**p[4])/(10.**p[5]))**2.)
    residuals = lambda p, f, y, yerr: (np.log10(y) \
            - np.log10(func(p, f) + func2(p, f)))/np.log10(yerr)
    r = least_squares(residuals, pars, 
                      args=(f[good], ps[good], ps_err[good]), 
                      bounds = bounds,
                      loss = 'cauchy', max_nfev=100000)
                      #loss = 'linear', method='lm')
    #r.x[0] = 10. ** r.x[0]
    #r.x[1] = 10. ** r.x[1]
    return func(r.x, _f) + func2(r.x, _f), _f, r.x

def fit_fn_only(f, ps, ps_err, pars = [0, -3, 1.5], bounds=([-4, -5, 0.1], [0, -1, 6.99])):

    if np.all(ps_err==0):
        ps_err = np.ones_like(ps) * 10.
        ps_err[ps<=0] = 0.

    good  = ps_err > 0
    #good[:2] = False

    _f = np.logspace(np.log10(f.min()), np.log10(f.max()), 100)

    func = lambda p, f: (10.**p[0]) * ( 1. + ((10.**p[1])/f)**p[2])
    residuals = lambda p, f, y, yerr: (np.log10(y) - np.log10(func(p, f)))/np.log10(yerr)
    r = least_squares(residuals, pars, 
                      args=(f[good], ps[good], ps_err[good]), 
                      bounds = bounds,
                      loss = 'cauchy', max_nfev=100000)
                      #loss = 'linear', method='lm')
    #r.x[0] = 10. ** r.x[0]
    #r.x[1] = 10. ** r.x[1]
    return func(r.x, _f), _f, r.x

def est_gt_ps(file_list, Tnoise_file=None, avg_len=21, output='./gt_ps.h5', fn_only=False):

    if fn_only:
        _fit_fn = fit_fn_only
    else:
        _fit_fn = fit_fn

    paras_list = []

    ps_result = []
    er_result = []

    ps_fit_result = []

    for bi in range(19):
        nd, time, freq = bandpass_cal.est_gtgnu_onefeed(file_list,
                            smooth=(1, 1), gi=bi, Tnoise_file=Tnoise_file,
                            interp_mask=False)

        nd = np.ma.masked_invalid(nd)
        #_nd_t = np.ma.mean(nd, axis=1)[:, None, :]
        #mask = np.all(_nd_t == 0, axis=(1, 2))
        mask = nd.mask
        _nd_t, mask, freq = avg_freq(nd, mask, freq, avg_len=avg_len)
        if _nd_t.shape[1] > 1:
            #print 'mask bad freq'
            bad_freq = np.sum(mask, axis=(0, 2)) > 0.8*(2 * _nd_t.shape[1])
            mask = mask[:, ~bad_freq, ...]
            _nd_t = _nd_t[:, ~bad_freq, ...]
        n_freq = _nd_t.shape[1]
        #print "%d unmasked freq channels"%n_freq
        #print _nd_t.max(), _nd_t.min()

        gt_m = np.ma.median(_nd_t, axis=0)
        gt_s = np.ma.std(_nd_t, axis=0)
        good = np.abs(_nd_t - gt_m[None, :]) - 6.*gt_s[None, :] < 0.
        mask += ~good

        dtime = time[1] - time[0]
        f_max = 1./ dtime / 2.
        f_min = 1./ dtime / float(time.shape[0]) #* 4.
        ps, bc = est_tcorr_psd1d_fft(_nd_t, time, mask, n_bins = 20, 
                                     f_min=f_min, f_max=f_max)

        ps_mean = np.mean(ps, axis=1)
        if ps.shape[1] == 1:
            ps_err = np.zeros_like(ps_mean)
            #ps_err[ps_mean<=0] = 0.
        else:
            ps_err  = np.std(ps,  axis=1) / np.sqrt(n_freq)

        ps_result.append(ps_mean)
        er_result.append(ps_err)

        ps_fit_xx, f_fit, paras_xx = _fit_fn(bc, ps_mean[:, 0], ps_err[:, 0])
        ps_fit_yy, f_fit, paras_yy = _fit_fn(bc, ps_mean[:, 1], ps_err[:, 1])
        paras_xx[0] = 10. ** paras_xx[0]
        paras_xx[1] = 10. ** paras_xx[1]
        paras_yy[0] = 10. ** paras_yy[0]
        paras_yy[1] = 10. ** paras_yy[1]
        paras_list.append([paras_xx, paras_yy])
        ps_fit_result.append([ps_fit_xx, ps_fit_yy])

    with h5.File(output, 'w') as f:
        f['ps_result'] = np.array(ps_result)
        f['er_result'] = np.array(er_result)
        f['f_result'] = bc
        f['paras'] = np.array(paras_list)
        f['ps_fit'] = np.array(ps_fit_result)
        f['f_fit'] = f_fit

    #return np.array(paras_list), bc, ps_result, er_result

def est_tcorr_psd1d_fft(data, ax, flag, n_bins=None, inttime=None,
        f_min=None, f_max=None):

    data = data.copy()

    #mean = np.mean(data[~flag, ...], axis=0)
    mean = np.ma.median(data, axis=0)
    std  = np.ma.std(data, axis=0)
    data -= mean[None, :, :]

    fill = np.random.standard_normal(data.shape) * std[None, ...] * 0.1
    fill[~flag] = 0.
    data += fill
    #data[flag, ...] = 0.

    weight = np.ones_like(data)
    weight[flag, ...] = 0

    windowf_t = np.blackman(data.shape[0])[:, None, None]
    windowf = windowf_t

    #logger.info('apply blackman windowf')
    data   = data   * windowf.copy()
    weight = weight * windowf.copy()

    fftdata = np.fft.fft(data, axis=0) # norm='ortho')
    fftdata /= np.sqrt(np.sum(weight, axis=0))[None, ...]

    n = ax.shape[0]
    if inttime is None:
        d = ax[1] - ax[0]
    else:
        d = inttime
    freq = np.fft.fftfreq(n, d) #* 2 * np.pi

    freq_p    = freq[freq>0]
    fftdata_p = fftdata[freq>0, ...]
    fftdata_p = np.abs(fftdata_p) * np.sqrt(float(d))
    fftdata_p = fftdata_p ** 2.
    fftdata_p = fftdata_p * 2**0.5 # include negative frequency

    if n_bins is not None:

        #if avg:
        #    fftdata_p = np.mean(fftdata_p, axis=1)[:, None, :]

        fftdata_bins = np.zeros((n_bins, ) + fftdata_p.shape[1:])

        if f_min is None: f_min = freq_p.min()
        if f_max is None: f_max = freq_p.max()
        freq_bins_c = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
        freq_bins_d = freq_bins_c[1] / freq_bins_c[0]
        freq_bins_e = freq_bins_c / (freq_bins_d ** 0.5)
        freq_bins_e = np.append(freq_bins_e, freq_bins_e[-1] * freq_bins_d)
        norm = np.histogram(freq_p, bins=freq_bins_e)[0] * 1.
        norm[norm==0] = np.inf

        for i in range(fftdata_p.shape[1]):

            hist_0 = np.histogram(freq_p,bins=freq_bins_e,weights=fftdata_p[:,i,0])[0]
            hist_1 = np.histogram(freq_p,bins=freq_bins_e,weights=fftdata_p[:,i,1])[0]
            hist   = np.concatenate([hist_0[:, None], hist_1[:, None]],axis=1)
            fftdata_bins[:, i, :] = hist / norm[:, None]

        fftdata_bins[freq_bins_c < freq_p.min()] = 0.
        fftdata_bins[freq_bins_c > freq_p.max()] = 0.

        return fftdata_bins, freq_bins_c
    else:
        return fftdata_p, freq_p

def avg_freq(vis, mask, freq, avg_len=20):

    #print "average every %d frequencis before ps. est."%avg_len
    time_n, freq_n, pol_n = vis.shape
    if avg_len is None:
        avg_len = freq_n
    split_n = int(freq_n / avg_len)
    if split_n == 0:
        msg = "Only %d freq channels, avg_len should be less than it"%freq_n
        raise ValueError(msg)
    #print "%d/%d freq channels are using"%(split_n * avg_len, freq_n)
    vis = vis[:, :split_n * avg_len, :]
    mask = mask[:, :split_n * avg_len, :]
    vis.shape = (time_n, split_n, avg_len, pol_n)
    mask.shape = (time_n, split_n, avg_len, pol_n)
    #vis = np.ma.mean(vis, axis=2)
    vis = np.ma.median(vis, axis=2)
    if split_n != 1:
        mask = np.sum(mask, axis=2) > 0.6 * avg_len
    else:
        mask = np.sum(mask, axis=2) == avg_len
    if np.all(mask):
        raise ValueError('All masked after freq avg')

    freq = freq[:split_n * avg_len]
    freq.shape = (split_n, avg_len)
    freq = np.median(freq, axis=1)

    return vis, mask, freq

def _CK(delta_nu, alpha, beta, w_min=1.e-3, w_max=1.):
    '''
    get C and K
    '''

    N_nu = np.float64(w_max / (w_min))
    d_nu = 1./w_max

    N_max = 1 * N_nu
    w_0 = 0.5 * w_min
    # I = w/w_min
    I = np.arange(1, N_max, 0.5).astype('float64')
    gamma = (beta - 1.)/beta

    C = 2 * np.sum( I ** gamma)
    C = 1.e6 / (C / float(len(I)-1.))
    C = 1.

    # np.sinc(x) = np.sin(np.pi * x) / (np.pi * x)
    K = 2. * np.sum((np.sinc( I * w_min * delta_nu ) ** 2.) * (I ** gamma))
    K = K * w_0/1.e6

    K0 = 1./ (delta_nu * 1.e6)

    #print
    #print d_nu, delta_nu, 1./K*1.e-6, K/K0

    return C, K/K0

def _fn_model_2d_pow(alpha, f0, beta, w_min=1.0e-3, w_max=1., dalpha=0):

    #F = lambda f: (f0/f)**(alpha + dalpha * np.log10(f0/f) )
    F = lambda f: (f0/f)**alpha

    C = _CK(1., alpha, beta, w_min, w_max)[0]

    H = lambda w: C * (w_min/np.abs(w)) ** ((1. - beta)/beta)

    return lambda f, w: F(f) * H(w)

def _fk_to_f0(delta_nu, alpha, fk, beta, w_min=1.0e-3, w_max=1.):

    '''
    delta_nu in MHz
    '''
    
    C, K = _CK(delta_nu, alpha, beta, w_min, w_max)

    lgf0 = np.log10(fk) - np.log10(C * K) / alpha

    return 10. ** lgf0

def _f0_to_fk(delta_nu, alpha, f0, beta, w_min=1.0e-3, w_max=1.):

    '''
    delta_nu in MHz
    '''

    C, K = _CK(delta_nu, alpha, beta, w_min, w_max)

    lgfk = np.log10(f0) + np.log10(C * K) / alpha

    return 10. ** lgfk

class FNoise(object):

    def __init__(self, dtime, dfreq, alpha, f0, beta):

        self.dtime = dtime
        self.dfreq = dfreq

        self.alpha = alpha
        self.f0    = f0
        self.beta  = beta

        self.fn_psd_f = _fn_model_2d_pow

    def fk_to_f0(self, delta_nu, w_min, w_max):

        return _fk_to_f0(delta_nu, self.alpha, self.f0, self.beta, w_min, w_max)


    def realisation(self, nfreq, ntime, delta_nu=None, wn=True):

        _n = 10

        dtime = self.dtime
        dfreq = self.dfreq
                     
        alpha = self.alpha
        #f0    = self.f0
        beta  = self.beta


        f_axis = np.fft.fftfreq( ntime, d=dtime)
        w_axis = np.fft.rfftfreq( _n * nfreq, d=dfreq/2.)

        f_abs = np.abs(f_axis)
        f_abs[f_abs==0] = np.inf

        w_min = w_axis[w_axis>0].min()
        w_max = w_axis[w_axis>0].max()
        w_axis[w_axis==0] = np.inf

        if delta_nu is None:
            f0 = self.f0
        else:
            f0 = self.fk_to_f0(delta_nu, w_min, w_max)

        fn_psd = self.fn_psd_f(alpha, f0, beta, w_min, w_max)(
                 f_abs[:, None], w_axis[None, :])
        fn_psd[:, w_axis==np.inf] = 1.
        fn_psd[0, 0] = 0.
        shp = fn_psd.shape

        if wn:
            fn_psd += 1.
        fn_psd = np.sqrt(fn_psd / dtime / (dfreq * 1.e6)/2.  * ntime * (_n * nfreq))

        fn_k = np.random.standard_normal(shp) + 1.J * np.random.standard_normal(shp)
        fn_k *= fn_psd

        fn_r = np.fft.irfftn(fn_k)

        fn_r = fn_r[:, :nfreq]
        return fn_r


