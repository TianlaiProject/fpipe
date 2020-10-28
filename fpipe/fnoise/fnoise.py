'''
'''

import numpy as np 

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


