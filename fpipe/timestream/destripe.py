import numpy as np

from scipy import signal
from scipy.interpolate import interp1d

from fpipe.container.timestream import FAST_Timestream

from fpipe.timestream import bandpass_cal


#def Ctt_fft(fk, alpha, time, ext=1):
#
#    dtime = (time[1] - time[0])
#
#    f = np.fft.fftfreq(time.shape[0]*ext, dtime)
#    f[f==0] = np.inf
#    P  = (0. + (fk / np.abs(f))**alpha )
#    xi = np.fft.ifft(P, norm='ortho').real * (1./(time.shape[0]*ext * dtime))
#    x = np.fft.fftfreq(xi.shape[0], 1./(dtime*P.shape[0]))
#
#    x = np.fft.fftshift(x)
#    xi = np.fft.fftshift(xi)
#
#    c_interp = interp1d(x, xi, bounds_error=False, fill_value='extrapolate')
#
#    tt = time - time[0]
#
#    c = c_interp(tt)
#
#    C_tt = np.zeros(c.shape * 2)
#
#    for i in range(C_tt.shape[0]):
#        C_tt[i] = np.roll(c, i)
#
#    C_tt = np.triu(C_tt, 0) + np.triu(C_tt, 1).T
#
#    return C_tt

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

    C_tt = np.zeros(c.shape * 2)

    for i in range(C_tt.shape[0]):
        C_tt[i] = np.roll(c, i)

    C_tt = np.triu(C_tt, 0) + np.triu(C_tt, 1).T

    return C_tt


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
        C_tt = Ctt(fk, alpha, time) * var
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

