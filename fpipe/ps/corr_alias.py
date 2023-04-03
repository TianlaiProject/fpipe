"""
Estimate the correction of the FFT alias effect for NGP case
"""

import numpy as np
from meerKAT_utils import algebra as al
from meerKAT_sim.ps import pwrspec_estimator as pse
from meerKAT_sim.sim import fisher
from meerKAT_utils import binning

__ps_func__ = fisher.matterpowerspectrum()

def alias_effect_window_1point(cube):

    info, k_axes, width = pse.make_k_axes(cube)

    W = al.make_vect(np.zeros_like(cube), axis_names=k_axes)
    W.info = info

    kN = np.pi / width # Nyquist frequency
    #print kN

    ps_func = __ps_func__

    C2 = []
    k0 = np.abs(W.get_axis('k_freq'))[:, None, None]
    k1 = np.abs(W.get_axis('k_ra')  )[None, :, None]
    k2 = np.abs(W.get_axis('k_dec') )[None, None, :]

    W0 = np.sinc(k0 / 2. / kN[0])
    W1 = np.sinc(k1 / 2. / kN[1])
    W2 = np.sinc(k2 / 2. / kN[2])

    k  = np.sqrt(k0**2 + k1**2 + k2**2)


    W[:]  = (( W0 * W1 * W2 ) ** 2.) * ps_func(k)

    return W, k0.flatten(), k1.flatten(), k2.flatten()

def alias_effect_window(cube, n=4):

    info, k_axes, width = pse.make_k_axes(cube)

    W = al.make_vect(np.zeros_like(cube), axis_names=k_axes)
    W.info = info

    kN = np.pi / width # Nyquist frequency

    ps_func = __ps_func__

    C2 = []
    k0 = np.abs(W.get_axis('k_freq'))
    k1 = np.abs(W.get_axis('k_ra'))
    k2 = np.abs(W.get_axis('k_dec'))

    for i, ki in enumerate(k0):
        _k0 = ki + 2 * kN[0] * np.arange(0, n+1, 1)[:, None, None]
        #W0 = np.sin(np.pi * _k0 / 2. / kN[0])/(np.pi * _k0 / 2. / kN[0])
        W0 = np.sinc(_k0 / 2. / kN[0])
        for j, kj in enumerate(k1):
            _k1 = kj + 2 * kN[1] * np.arange(0, n+1, 1)[None, :, None]
            #W1 = np.sin(np.pi * _k1 / 2. / kN[1])/(np.pi * _k1 / 2. / kN[1])
            W1 = np.sinc(_k1 / 2. / kN[1])
            for k, kk in enumerate(k2):
                _k2 = kk + 2 * kN[2] * np.arange(0, n+1, 1)[None, None, :]
                #W2 = np.sin(np.pi * _k2 / 2. / kN[2])/(np.pi * _k2 / 2. / kN[2])
                W2 = np.sinc(_k2 / 2. / kN[2])

                #if ki == 0 or kj == 0 or kk == 0: continue

                _k = np.sqrt(_k0 ** 2 + _k1 ** 2 + _k2 ** 2)
                W[i, j, k] = np.sum( (W0 * W1 * W2)**2 * ps_func(_k) )

    return W, k0, k1, k2

def _alias_effect_correction_3D(cube, n=4):

    ps_func = __ps_func__

    if n == 0:
        W, k0, k1, k2 = alias_effect_window_1point(cube)
    else:
        W, k0, k1, k2 = alias_effect_window(cube, n)

    _k_ref  = np.sqrt(k0[:, None, None]**2 + k1[None, :, None]**2 + k2[None, None, :]**2)
    _k_ref  = np.abs(_k_ref)
    power_ref = ps_func(_k_ref)

    W[:] /= power_ref

    return W

def _alias_effect_correction_2D(cube,  kbin_x, kbin_y, n=4, logk=True):

    ps_func = __ps_func__

    if n == 0:
        W, k0, k1, k2 = alias_effect_window_1point(cube)
    else:
        W, k0, k1, k2 = alias_effect_window(cube, n)

    kbin_x_edges = binning.find_edges(kbin_x, logk=logk)
    kbin_y_edges = binning.find_edges(kbin_y, logk=logk)

    k_perp = binning.radius_array(W, zero_axes=[0])
    k_para = binning.radius_array(W, zero_axes=[1, 2])

    count_2d, w_2d = binning.bin_an_array_2d(W, k_perp, k_para,
                                                 kbin_x_edges, kbin_y_edges)

    _k_ref = np.sqrt((kbin_x[:, None])**2 + (kbin_y[None, :])**2)
    power_ref = ps_func(_k_ref)

    return w_2d / power_ref

def _alias_effect_correction_1D(cube, kbin, n=4, logk=True):

    ps_func = __ps_func__

    if n == 0:
        W, k0, k1, k2 = alias_effect_window_1point(cube)
    else:
        W, k0, k1, k2 = alias_effect_window(cube, n)

    kbin_edges = binning.find_edges(kbin, logk=True)

    k = binning.radius_array(W)
    count_1d, w_1d = binning.bin_an_array(W, kbin_edges, k)

    _k_ref = kbin
    power_ref = ps_func(_k_ref)

    return w_1d / power_ref


