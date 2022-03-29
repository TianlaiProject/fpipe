import logging

import matplotlib.pyplot as plt

from caput import mpiutil
from caput import mpiarray
from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.utils.path_util import output_path

from fpipe.timestream import timestream_task
from fpipe.map import algebra as al
from fpipe.map import mapbase, dirtymap
#from meerKAT_sim.fnoise import fnoise

import healpy as hp
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
#from scipy import linalg
from numpy.linalg import multi_dot
from numpy import linalg
from scipy import special
import h5py
import sys
import gc

logger = logging.getLogger(__name__)

from meerKAT_utils.constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from meerKAT_utils.constants import f_medium, f_large

__dtype__ = 'float32'


class SubtractSky(timestream_task.TimestreamTask, mapbase.MultiMapBase):
    params_init = {
        'naive_map' : '',
        'beam_fwhm_at21cm': 1.,
        'debug' : False,
        'tblock_len' : None,
    }
    prefix = 'subsky_'
    
    def process(self, ts):
        
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        
        self.load_naive_map()

        func = ts.freq_data_operate
        
        ts.redistribute('frequency')
        func(self.sub_sky, full_data=False, copy_data=False,
                show_progress=show_progress,
                progress_step=progress_step, keep_dist_axis=False)

        mpiutil.barrier()

        return super(SubtractSky, self).process(ts)
    
    def load_naive_map(self):
        
        msg = 'load naive map from \n%s'%self.params['naive_map']
        logger.info(msg)
        naive_path = self.params['naive_map']
        self.open(naive_path)
        self.naive_map = al.make_vect(al.load_h5(self.df_in[0], 'clean_map'))
        self.noise_map = al.make_vect(al.load_h5(self.df_in[0], 'noise_diag'))

    def sub_sky(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        
        if not isinstance(gi, tuple): gi = (gi, )
        if not isinstance(li, tuple): li = (li, )
        #dfreq = np.abs(ts.freq[1] - ts.freq[0])
        freq = ts.freq[gi[0]] * 1.e-3
        time = ts['sec1970'][:]
        time = time - time[0]
        beam_fwhm = self.params['beam_fwhm_at21cm'] * 1.42 / freq
        msg = "RANK%03d:"%mpiutil.rank + \
                " Local  (" + ("%04d, "*len(li))%li + ")," +\
                " Global (" + ("%04d, "*len(gi))%gi + ")"
        logger.info(msg)
        
        n_time, n_pol, n_bl = vis.shape
        vis_idx = ((bb, pp, gi[0]) for bb in range(n_bl) for pp in range(n_pol))
        vis_axis_names = ('bl', 'pol', 'freq')
        #vis_idx = ((bb, gi[0]) for bb in range(n_bl))
        #vis_axis_names = ('bl', 'freq')
        
        ra_axis = self.naive_map.get_axis('ra')
        dec_axis = self.naive_map.get_axis('dec')
        map_axis_names = self.naive_map.info['axes']
        ra_centr  = ra_axis  * np.pi / 180.
        dec_centr = dec_axis * np.pi / 180.

        for _vis_idx in vis_idx:
            
            b_idx, p_idx, f_idx = _vis_idx
            #b_idx, f_idx = _vis_idx
            map_idx = [_vis_idx[ii] for ii, name in enumerate(vis_axis_names)
                    if name in map_axis_names]
            map_idx = tuple(map_idx)

            msg = "RANK%03d:"%mpiutil.rank + \
                    " VIS (" + ("%03d, "*len(_vis_idx))%_vis_idx + ")" +\
                    " Map (" + ("%03d, "*len(map_idx)%map_idx) + ")"
            logger.debug(msg)
            
            msk= vis_mask[:, p_idx, b_idx]
            var= ts['vis_var'][:, f_idx, p_idx, b_idx]
            ra = ts['ra'][:, b_idx]
            dec= ts['dec'][:, b_idx]

            if np.all(msk):
                msg = " VIS (" + ("%03d, "*len(_vis_idx))%_vis_idx + ")" +\
                        " All masked, continue"
                logger.info(msg)
                #self.df['mask'][map_idx[:-1]] = 1
                continue
                
            _good  = ( ra  < max(ra_axis))
            _good *= ( ra  > min(ra_axis))
            _good *= ( dec < max(dec_axis))
            _good *= ( dec > min(dec_axis))
            _good *= ~msk
            if np.sum(_good) == 0: continue
            #n_good = np.sum(_good)
            
            ra = ra[_good] * np.pi / 180.
            dec= dec[_good]* np.pi / 180.
            
            P = est_pointing_matrix(ra, dec, ra_centr, dec_centr, beam_size=beam_fwhm)
            P = np.mat(P)

            Fa = subsky_func( var[_good], P, 
                            self.noise_map[map_idx], 
                            self.naive_map[map_idx], 
                            time[_good], )

            Fa = np.array(Fa.flat)
            vis[_good, p_idx, b_idx] -= Fa
            #ts['vis_var'][_good, f_idx, p_idx, b_idx] += Fa**2
        
    def init_vis(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        time = ts['sec1970'][:]
        n_time = time.shape[0]
        tblock_len = self.params['tblock_len']
        if tblock_len is None:
            tblock_len = n_time
        logger.info('break into block length of %d'%tblock_len)

        n_poly = 1

        vis_var = ts['vis_var'].local_data

        logger.debug('est. var %d %d'%(n_time, tblock_len))
        for st in range(0, n_time, tblock_len):
            et = st + tblock_len
            _time = time[st:et] #ts['sec1970'][st:et]
            _vis_mask = (vis_mask[st:et,...]).astype('bool')

            _fit = dirtymap.sub_ortho_poly(vis[st:et,...], _time, ~_vis_mask, n_poly)
            vis[st:et,...] -= _fit
            _vis = vis[st:et,...] - _fit
            _vis[_vis_mask] = 0.

            _vars = sp.sum(_vis ** 2., axis=0)
            _cont = sp.sum(~_vis_mask, axis=0) * 1.
            _bad = _cont == 0
            _cont[_bad] = np.inf
            _vars /= _cont
            _vars[_vars<T_small**2] = T_small**2
            #_vis_mask += (_vis - 3 * np.sqrt(_vars[None, ...])) > 0.
            #_vis[_vis_mask] = 0.
            msg = 'min vars = %f, max vars = %f'%(_vars.min(), _vars.max())
            logger.debug(msg)
            vis_var[st:et, li, ...] += _vars[None, :] * _fit**2

#def Ctt(fk, alpha, time, ext=1):
#
#    dtime = (time[1] - time[0])
#
#    f = np.fft.fftfreq(time.shape[0]*ext, dtime)
#    f[f==0] = np.inf
#    P  = (1. + (fk / np.abs(f))**alpha )
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
#    #C_tt[C_tt<0] = 0.
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
    #print c[:5]
    c[0] += 1.

    C_tt = np.zeros(c.shape * 2)

    for i in range(C_tt.shape[0]):
        C_tt[i] = np.roll(c, i)

    C_tt = np.triu(C_tt, 0) + np.triu(C_tt, 1).T

    return C_tt

def est_pointing_matrix(ra, dec, ra_centr, dec_centr, beam_size=1., beam_cut=None):

    beam_sig = beam_size  / (2. * np.sqrt(2.*np.log(2.)))

    P = (np.sin(dec[:, None]) * np.sin(dec_centr[None, :]))[:, None, :]\
      + (np.cos(dec[:, None]) * np.cos(dec_centr[None, :]))[:, None, :]\
      * (np.cos(ra[:, None] - ra_centr[None, :]))[:, :, None]

    P  = np.arccos(P) * 180. / np.pi
    P  = np.exp(- 0.5 * (P / beam_sig) ** 2)


    P.shape = (ra.shape[0], -1)
    if beam_cut is None:
        # split the power into nearby 4 pixels
        P_max = np.argmax(P, axis=1)
        P *= 0.
        P[tuple(range(ra.shape[0])), tuple(P_max)] = 1.
    else:
        P[P < beam_cut] *= 0.
        P_norm = np.sum(P, axis=1)
        #P_norm = np.max(P, axis=1)
        P_norm[P_norm==0] = np.inf
        P /= P_norm[:, None]

    return P

def subsky_func(var, P, noise_map, naive_map, time):

    n_time = time.shape[0]
    I = np.mat(np.eye(n_time))

    N = 1./var
    N = np.mat(N * np.eye(n_time))

    PNP = noise_map.flatten()
    PNP = np.mat(PNP * np.eye(PNP.shape[0]))
    Z = I - P * PNP * P.T * N

    return Z

    #y = np.mat(vis[:, None])

    #P = np.mat(P)
    #M = np.mat((naive_map.flatten())[:, None])
    #Zy = y - P * M

    #return P * M
