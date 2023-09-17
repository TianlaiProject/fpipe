from collections import OrderedDict as dict

import h5py as h5
import numpy as np
import os

from scipy.interpolate import interp2d
from scipy.interpolate import Rbf

from fpipe.utils import binning
from fpipe.map import algebra as al

import logging
logger = logging.getLogger(__name__)

def load_1dtr(ps_path, ps_name, ps_ref, kbin):

    with h5.File(ps_path + ps_ref, 'r') as f:
        # correct for taking the average later after taking the ratio
        ref = al.load_h5(f, 'ps3d')


    with h5.File(ps_path + ps_name, 'r') as f:

        # correct for taking the average later after taking the ratio
        ps3d     = al.load_h5(f, 'ps3d')

    tr1d = []
    for ii in range(ps3d.shape[0]):
        _ps3d = al.make_vect(ps3d[ii])
        _ref  = al.make_vect(ref[ii])
        ref1d = binning.bin_an_array(_ref, kbin)[1]
        ct1d, ps1d  = binning.bin_an_array(_ps3d, kbin)
        ref1d[ref1d==0] = np.inf
        tr1d.append( (ps1d / ref1d)[None, :] )
    tr1d = np.mean(np.concatenate(tr1d, axis=0), axis=0)

    tr1d[tr1d==0] = np.inf
    tr1d = 1./tr1d

    return tr1d, ct1d

def load_1dtr_average_first(ps_path, ps_name, ps_ref, kbin):

    with h5.File(ps_path + ps_ref, 'r') as f:
        ref = np.mean(al.load_h5(f, 'ps3d'), axis=0)

    with h5.File(ps_path + ps_name, 'r') as f:
        ps3d = np.mean(al.load_h5(f, 'ps3d'), axis=0)

    ps3d = al.make_vect(ps3d)
    ref  = al.make_vect(ref)
    ref1d = binning.bin_an_array(ref, kbin)[1]
    ct1d, ps1d  = binning.bin_an_array(ps3d, kbin)
    ps1d[ps1d==0] = np.inf
    tr1d = ref1d / ps1d

    return tr1d, ct1d

def load_2dtr(ps_path, ps_name, ps_ref, kbin_x, kbin_y):

    with h5.File(ps_path + ps_ref, 'r') as f:
        # correct for taking the average later after taking the ratio
        ref = al.load_h5(f, 'ps3d')


    with h5.File(ps_path + ps_name, 'r') as f:

        # correct for taking the average later after taking the ratio
        ps3d     = al.load_h5(f, 'ps3d')

    _ps3d = al.make_vect(ps3d[0])
    k_perp = binning.radius_array(_ps3d, zero_axes=[0])
    k_para = binning.radius_array(_ps3d, zero_axes=[1, 2])
    #kbin_x = np.arange(0, k_perp.max(), 2*k_perp[k_perp!=0].min())
    #kbin_y = np.arange(0, k_para.max(), 2*k_para[k_para!=0].min())
    tr2d = []
    for ii in range(ps3d.shape[0]):
        ref2d = binning.bin_an_array_2d(ref[ii], k_perp, k_para, kbin_x, kbin_y)[1]
        ct2d, ps2d  = binning.bin_an_array_2d(ps3d[ii],k_perp, k_para, kbin_x, kbin_y)
        ref2d[ref2d==0] = np.inf
        tr2d.append( (ps2d / ref2d)[None, :, :] )
    tr2d = np.mean(np.concatenate(tr2d, axis=0), axis=0)

    tr2d[tr2d==0] = np.inf
    tr2d = 1/tr2d

    return tr2d, ct2d

def load_3dtr(ps_path, ps_name, ps_ref, avg2d=True):

    with h5.File(ps_path + ps_ref, 'r') as f:
        # correct for taking the average later after taking the ratio
        ref = al.load_h5(f, 'ps3d')


    with h5.File(ps_path + ps_name, 'r') as f:

        # correct for taking the average later after taking the ratio
        ps3d     = al.load_h5(f, 'ps3d')

    if avg2d:
        _ps3d = al.make_vect(ps3d[0])
        k_perp = binning.radius_array(_ps3d, zero_axes=[0])
        k_para = binning.radius_array(_ps3d, zero_axes=[1, 2])
        kbin_x = np.arange(0, k_perp.max(), 2*k_perp[k_perp!=0].min())
        kbin_y = np.arange(0, k_para.max(), 2*k_para[k_para!=0].min())
        tr2d = []
        for ii in range(ps3d.shape[0]):
            ref2d = binning.bin_an_array_2d(ref[ii], k_perp, k_para, kbin_x, kbin_y)[1]
            ps2d  = binning.bin_an_array_2d(ps3d[ii],k_perp, k_para, kbin_x, kbin_y)[1]
            ref2d[ref2d==0] = np.inf
            tr2d.append( (ps2d / ref2d)[None, :, :] )
        tr2d = np.mean(np.concatenate(tr2d, axis=0), axis=0)

        kbin_x = (kbin_x[:-1] + kbin_x[1:]) * 0.5
        kbin_y = (kbin_y[:-1] + kbin_y[1:]) * 0.5
        tr2d_interp = interp2d(kbin_x, kbin_y, tr2d.T, bounds_error=False, fill_value=0)
        xx = k_perp[0, :, :].flat
        yy = k_para[:, 0, 0].flat
        unsorted_idxs_xx = np.argsort(np.argsort(xx))
        unsorted_idxs_yy = np.argsort(np.argsort(yy))
        tr3d = tr2d_interp(xx, yy)[unsorted_idxs_yy, :][:, unsorted_idxs_xx]
        tr3d.shape = _ps3d.shape

        tr3d = al.make_vect(tr3d)
        tr3d.info = _ps3d.info

    else:
        ref[ref==0] = np.inf
        tr3d = np.mean(ps3d/ref, axis=0)

    tr3d[tr3d==0] = np.inf
    tr3d = 1/tr3d

    return tr3d

def load_3dtr_average_first(ps_path, ps_name, ps_ref, avg2d=True):

    with h5.File(ps_path + ps_ref, 'r') as f:
        ps3d_ref = np.mean(al.load_h5(f, 'ps3d'), axis=0)


    with h5.File(ps_path + ps_name, 'r') as f:

        ps3d     = np.mean(al.load_h5(f, 'ps3d'), axis=0)
        #ps3d[np.abs(ps3d)<1.e-20] = np.inf
        #ps3d[np.abs(ps3d)<1.e-1] = np.inf
        #ps3d = ps3d * ps3d_ref.copy()

    ps3d = al.make_vect(ps3d)

    k_perp = binning.radius_array(ps3d, zero_axes=[0])
    if avg2d:
        k_para = binning.radius_array(ps3d, zero_axes=[1, 2])

        kbin_x = np.arange(0, k_perp.max(), 2*k_perp[k_perp!=0].min())
        kbin_y = np.arange(0, k_para.max(), 2*k_para[k_para!=0].min())
        ps2d_ref = binning.bin_an_array_2d(ps3d_ref, k_perp, k_para, kbin_x, kbin_y)[1]
        ps2d     = binning.bin_an_array_2d(ps3d, k_perp, k_para, kbin_x, kbin_y)[1]
        ps2d[ps2d==0] = np.inf
        tr2d = ps2d_ref / ps2d

        kbin_x = (kbin_x[:-1] + kbin_x[1:]) * 0.5
        kbin_y = (kbin_y[:-1] + kbin_y[1:]) * 0.5

        tr2d_interp = interp2d(kbin_x, kbin_y, tr2d.T, bounds_error=False, fill_value=0)
        xx = k_perp[0, :, :].flat
        yy = k_para[:, 0, 0].flat
        unsorted_idxs_xx = np.argsort(np.argsort(xx))
        unsorted_idxs_yy = np.argsort(np.argsort(yy))
        tr3d = tr2d_interp(xx, yy)[unsorted_idxs_yy, :][:, unsorted_idxs_xx]
        tr3d.shape = ps3d.shape

        tr3d = al.make_vect(tr3d)
        tr3d.info = ps3d.info

    else:
        ps3d[ps3d==0] = np.inf
        tr3d = ps3d_ref / ps3d

    #mask = k_perp > 0.10
    #mask += tr3d < 0
    #mask += tr3d > 10

    #tr3d[mask] = 0.
    #tr3d[tr3d<0] = 0.

    return tr3d

def iter_1dtr_list(ps_path, ps_name_list, tr_path=None, cross=True, 
        ps_ref='ps_sub00modes', average_first=True, kbin=None, kbin_c=None):

    if average_first:
        _load_1dtr = load_1dtr_average_first
    else:
        _load_1dtr = load_1dtr

    for ii, ps_key in enumerate(ps_name_list.keys()):
        ps_name = ps_name_list[ps_key]
        tr1d = None
        if isinstance(ps_name, list):
            if tr_path is None: tr_path = ps_path
            ps_name, tr_name = ps_name
            if tr_name is not None:
                #ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
                ref_name = os.path.dirname(tr_name) + '/%s.h5'%ps_ref
                tr1d, ct1d = _load_1dtr(tr_path, tr_name, ref_name, kbin)

        yield ii, ps_key, tr1d, ct1d, kbin, kbin_c


def iter_3dtr_list(ps_path, ps_name_list, tr_path=None, cross=True, 
        ps_ref='ps_sub00modes', average_first=True):

    if average_first:
        _load_3dtr = load_3dtr_average_first
    else:
        _load_3dtr = load_3dtr

    for ii, ps_key in enumerate(ps_name_list.keys()):
        ps_name = ps_name_list[ps_key]
        #if tr_path is not None:
        tr3d = None
        if isinstance(ps_name, list):
            if tr_path is None: tr_path = ps_path
            ps_name, tr_name = ps_name
            if tr_name is not None:
                #ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
                ref_name = os.path.dirname(tr_name) + '/%s.h5'%ps_ref
                tr3d = _load_3dtr(tr_path, tr_name, ref_name, avg2d=True)

        yield ii, ps_key, tr3d

def iter_3dto2dtr_list(ps_path, ps_name_list, tr_path=None, cross=True,
        bins_x=None, bins_y=None, bins_xc=None, bins_yc=None, ps_ref='ps_sub00modes',
        average_first=True):

    for ps3d_set in iter_3dtr_list(ps_path, ps_name_list, tr_path=None, 
            cross=True, ps_ref=ps_ref, average_first=average_first):
        ii, ps_key, ps3d = ps3d_set
        ps2d = np.zeros((bins_x.shape[0]-1, bins_y.shape[0]-1))

        #ps3d_jj = al.make_vect(ps3d)
        ps3d_jj = ps3d
        # find the k_perp by not including k_nu in the distance
        k_perp = binning.radius_array(ps3d_jj, zero_axes=[0])
        # find the k_perp by not including k_RA,Dec in the distance
        k_para = binning.radius_array(ps3d_jj, zero_axes=[1, 2])
        ct2d, ps2d[:] = binning.bin_an_array_2d(ps3d_jj, k_perp, k_para, bins_x, bins_y)

        yield ii, ps_key, ps2d, ct2d, bins_x, bins_y, bins_xc, bins_yc

def iter_2dtr_list(ps_path, ps_name_list, tr_path=None, cross=True,
        bins_x=None, bins_y=None, bins_xc=None, bins_yc=None, ps_ref='ps_sub00modes'):

    for ii, ps_key in enumerate(ps_name_list.keys()):
        ps_name = ps_name_list[ps_key]
        #if tr_path is not None:
        tr2d = None
        if isinstance(ps_name, list):
            if tr_path is None: tr_path = ps_path
            ps_name, tr_name = ps_name
            if tr_name is not None:
                #ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
                ref_name = os.path.dirname(tr_name) + '/%s.h5'%ps_ref
                tr2d, ct2d = load_2dtr(tr_path, tr_name, ref_name, bins_x, bins_y)

        yield ii, ps_key, tr2d, ct2d, bins_x, bins_y, bins_xc, bins_yc


def iter_3dps_list(ps_path, ps_name_list, tr_path=None, cross=True, ps_ref='ps_sub00modes'):

    for ii, ps_key in enumerate(ps_name_list.keys()):
        ps_name = ps_name_list[ps_key]
        #if tr_path is not None:
        tr3d = None
        if isinstance(ps_name, list):
            if tr_path is None: tr_path = ps_path
            ps_name, tr_name = ps_name
            if tr_name is not None:
                #ref_name = os.path.dirname(tr_name) + '/ps_raw.h5'
                with h5.File(tr_path + tr_name, 'r') as f:
                    tr3d = f['tr3d'][:]
                #ref_name = os.path.dirname(tr_name) + '/%s.h5'%ps_ref
                #tr3d = load_3dtr(tr_path, tr_name, ref_name)
                if not cross:
                    tr3d **= 2

        logger.debug( ps_path + ps_name )
        with h5.File(ps_path + ps_name, 'r') as f:

            bins_x = f['kbin_x_edges'][:]
            bins_y = f['kbin_y_edges'][:]
            xc = f['kbin_x'][:]
            yc = f['kbin_y'][:]
            bins = f['kbin_edges'][:]
            kc = f['kbin'][:]

            ps3d = al.load_h5(f, 'ps3d')
            if cross: ps3d *= 1.e3
            else:     ps3d *= 1.e6
            if tr3d is not None: ps3d *= tr3d.copy()[None, ...]

        yield ii, ps_key, ps3d, bins_x, bins_y, xc, yc, bins, kc

def iter_2dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
        bins_x=None, bins_y=None, bins_xc=None, bins_yc=None, ps_ref='ps_sub00modes'):
    for ps3d_set in iter_3dps_list(ps_path, ps_name_list, tr_path=tr_path, cross=cross, ps_ref=ps_ref):
        ii, ps_key, ps3d, xe, ye, xc, yc, ke, kc = ps3d_set
        if bins_x is None: bins_x = xe; bins_xc = xc
        if bins_y is None: bins_y = ye; bins_yc = yc
        ps2d = np.zeros((ps3d.shape[0], bins_x.shape[0]-1, bins_y.shape[0]-1))
        for jj in range(ps3d.shape[0]):
            ps3d_jj = al.make_vect(ps3d[jj])
            # find the k_perp by not including k_nu in the distance
            k_perp = binning.radius_array(ps3d_jj, zero_axes=[0])
            # find the k_perp by not including k_RA,Dec in the distance
            k_para = binning.radius_array(ps3d_jj, zero_axes=[1, 2])
            ct2d, ps2d[jj] = binning.bin_an_array_2d(ps3d_jj, 
                    k_perp, k_para, bins_x, bins_y)

        yield ii, ps_key, ps2d, ct2d, bins_x, bins_y, bins_xc, bins_yc


def iter_1dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
        bins=None, bins_c=None, ps_ref='ps_sub00modes'):
    for ps3d_set in iter_3dps_list(ps_path, ps_name_list, tr_path=tr_path, cross=cross, ps_ref=ps_ref):
        ii, ps_key, ps3d, xe, ye, xc, yc, ke, kc = ps3d_set
        if bins is None: bins = ke; bins_c = kc
        ps1d = np.zeros((ps3d.shape[0], bins.shape[0]-1))
        for jj in range(ps3d.shape[0]):
            ps3d_jj = al.make_vect(ps3d[jj])
            k = binning.radius_array(ps3d_jj)
            ct1d, ps1d[jj] = binning.bin_an_array(ps3d_jj, bins, radius_arr=k)

        yield ii, ps_key, ps1d, ct1d, bins, bins_c

def summary_3dtr(ps_path, ps_name_list, tr_path=None, cross=True,
        ps_ref='ps_sub00modes', average_first=True, suffix=''):

    for ps3d_set in iter_3dtr_list(ps_path, ps_name_list, tr_path=tr_path, 
            cross=cross, ps_ref=ps_ref, average_first=average_first):
        ii, ps_key, ps3d = ps3d_set

        fname = ps_name_list[ps_key]
        if isinstance( fname, list): fname = fname[1]
        fname = ps_path + fname.replace('.h5', '_3dtr%s.h5'%suffix)
        logger.debug( fname )
        with h5.File(fname, 'w') as f:
            f['tr3d']    = ps3d

def summary_2dtr(ps_path, ps_name_list, tr_path=None, 
            kbin_x=None, kbin_y=None, logk=True, cross=False,
            ps_ref='ps_sub00modes', average_first=True, suffix=''):

    if kbin_x is None: raise ValueError('kbin_x needed')
    if kbin_y is None: raise ValueError('kbin_y needed')

    kbin_xc = kbin_x[:-1]
    if logk:
        dkbin_x = kbin_x[1] / kbin_x[0]
        kbin_xc = kbin_xc * (dkbin_x ** 0.5)
    else:
        dkbin_x = kbin_x[1] - kbin_x[0]
        kbin_xc = kbin_xc + (dkbin_x * 0.5)

    kbin_yc = kbin_y[:-1]
    if logk:
        dkbin_y = kbin_y[1] / kbin_y[0]
        kbin_yc = kbin_yc * (dkbin_y ** 0.5)
    else:
        dkbin_y = kbin_y[1] - kbin_y[0]
        kbin_yc = kbin_yc + (dkbin_y * 0.5)

    for ii, ps_name, ps2d, ct2d, kbin_x, kbin_y, kbin_xc, kbin_yc, in \
            iter_3dto2dtr_list(ps_path, ps_name_list, tr_path=tr_path, 
                    bins_x=kbin_x, bins_y=kbin_y, bins_xc=kbin_xc, bins_yc=kbin_yc, 
                    cross=cross, ps_ref=ps_ref, average_first=average_first):

        logger.info(ps_name)
        ps2d[ps2d==0] = np.inf
        ps2d = 1./ps2d

        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[1]
        fname = ps_path + fname.replace('.h5', '_2dtr%s.h5'%suffix)
        logger.debug( fname )
        with h5.File(fname, 'w') as f:
            f['tr2d']    = ps2d
            f['ct2d']    = ct2d
            f['kbin_x']  = kbin_x
            f['kbin_y']  = kbin_y
            f['kbin_xc'] = kbin_xc
            f['kbin_yc'] = kbin_yc

def summary_1dtr(ps_path, ps_name_list, tr_path=None, 
            kbin=None, logk=True, cross=False,
            ps_ref='ps_sub00modes', average_first=True, suffix=''):

    if kbin is None: raise ValueError('kbin_x needed')

    kbin_c = kbin[:-1]
    if logk:
        dkbin = kbin[1] / kbin[0]
        kbin_c = kbin_c * (dkbin ** 0.5)
    else:
        dkbin = kbin[1] - kbin[0]
        kbin_c = kbin_c + (dkbin * 0.5)

    for ii, ps_name, ps1d, ct1d, kbin, kbin_c, in \
            iter_1dtr_list(ps_path, ps_name_list, tr_path=tr_path, 
                    kbin=kbin, kbin_c=kbin_c,
                    cross=cross, ps_ref=ps_ref, average_first=average_first):

        logger.info(ps_name)
        ps1d[ps1d==0] = np.inf
        ps1d = 1./ps1d

        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[1]
        fname = ps_path + fname.replace('.h5', '_1dtr%s.h5'%suffix)
        logger.debug( fname )
        with h5.File(fname, 'w') as f:
            f['tr1d']   = ps1d
            f['ct1d']   = ct1d
            f['kbin']   = kbin
            f['kbin_c'] = kbin_c

def summary_1dxps(ps_path, ps_name_list, shot_name=None, logk=True, bins=None, sim=False,
        suffix='', ps_ref='ps_sub00modes', tr1d_name_list=None):

    logger.critical('summary 1d ps estimation')

    if bins is not None:
        kh = bins[:-1]
        if logk:
            dk = bins[1] / bins[0]
            kh = kh * (dk ** 0.5)
        else:
            dk = bins[1] - bins[0]
            kh = kh + (dk * 0.5)
    else:
        kh = None

    if shot_name is not None:
        for xx in iter_1dps_list(ps_path, {'shot': shot_name},
                cross=True, bins=bins, bins_c=kh):
            shot = np.ma.mean(xx[2][1:] * 1.e-3, axis=0)
            #shot = np.mean(shot)
            shot = shot #* Tb_HI #* 0.8
    else:
        shot = np.zeros(kh.shape[0])

    for ii, ps_name, ps1d, ct1d, bins, kh in\
            iter_1dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
                    bins=bins, bins_c=kh, ps_ref=ps_ref):

        logger.info(ps_name)
        if not sim:
            ps = ps1d[0]
            ps_e = np.ma.std(ps1d[1:], axis=0)
            null = np.ma.mean(ps1d[1:], axis=0)
            ps_cov = np.cov(ps1d[1:], rowvar=False)
        else:
            ps = np.mean(ps1d, axis=0)
            ps_e = np.ma.std(ps1d, axis=0) #/ np.sqrt(ps1d.shape[0] * 1.)
            null = np.zeros(ps.shape[0])
            #ps_cov_inv = np.linalg.inv(np.cov(ps1d[1:], rowvar=False))
            ps_cov = np.cov(ps1d, rowvar=False)

        if tr1d_name_list is not None:
            with h5.File(ps_path + tr1d_name_list[ps_name], 'r') as f:
                tr1d = f['tr1d'][:]
                kbin_c = f['kbin_c'][:]
            if np.any(kbin_c - kh):
                raise ValueError 

            tr1d[tr1d==0] = np.inf
            ps = ps/tr1d
            ps_e = ps_e/tr1d
            ps_cov = ps_cov / (tr1d[:, None] * tr1d[None, :])

        results = [kh[:, None], ps[:, None], ps_e[:, None], null[:, None],
                   shot[:, None], ps_cov]
        results = np.concatenate(results, axis=1)
        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[0]
        fname = ps_path + fname.replace('.h5', '_1d%s.txt'%suffix)
        header = '%23s %24s %24s %24s %24s %24s'%(
                'kh', 'pk', 'pk_err', 'null', 'shot', 'cov')
        logger.debug( fname )
        np.savetxt(fname, results, fmt='%20.18e', header=header)

def summary_1daps(ps_path, ps_name_list, shot_name=None, logk=True, bins=None, sim=False,
        suffix='', ps_ref='ps_sub00modes'):

    logger.critical( 'summary 1d ps estimation' )

    if bins is not None:
        kh = bins[:-1]
        if logk:
            dk = bins[1] / bins[0]
            kh = kh * (dk ** 0.5)
        else:
            dk = bins[1] - bins[0]
            kh = kh + (dk * 0.5)
    else:
        kh = None

    if shot_name is not None:
        for xx in iter_1dps_list(ps_path, {'shot': shot_name},
                cross=True, bins=bins, bins_c=kh):
            shot = np.ma.mean(xx[2][1:] * 1.e-3, axis=0)
            #shot = np.mean(shot)
            shot = shot #* Tb_HI #* 0.8
    else:
        shot = np.zeros(kh.shape[0])


    for ii, ps_name, ps1d, ct1d, bins, kh in\
            iter_1dps_list(ps_path, ps_name_list, tr_path=None, cross=False,
                    bins=bins, bins_c=kh, ps_ref=ps_ref):

        logger.info( ps_name )
        if not sim:
            ps = ps1d[0]
            ps_e = np.zeros(ps.shape) #np.ma.std(ps1d[1:], axis=0)
            null = np.zeros(ps.shape) #np.ma.mean(ps1d[1:], axis=0)
            ps_cov = np.eye(ps.shape[0]) #np.cov(ps1d[1:], rowvar=False)
        else:
            ps = np.mean(ps1d, axis=0)
            ps_e = np.ma.std(ps1d, axis=0)
            null = np.zeros(ps.shape[0])
            #ps_cov_inv = np.linalg.inv(np.cov(ps1d[1:], rowvar=False))
            ps_cov = np.cov(ps1d, rowvar=False)

        count = ct1d

        results = [kh[:, None], ps[:, None], ps_e[:, None], count[:, None],
                   shot[:, None], ps_cov]
        results = np.concatenate(results, axis=1)
        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[0]
        fname = ps_path + fname.replace('.h5', '_1d%s.txt'%suffix)
        header = '%23s %24s %24s %24s %24s %24s'%(
                'kh', 'pk', 'pk_err', 'count', 'shot', 'cov')
        logger.debug( fname )
        np.savetxt(fname, results, fmt='%20.18e', header=header)

def summary_2dxps(ps_path, ps_name_list, logk=True, bins_x=None, bins_y=None, sim=False, 
        suffix='', ps_ref='ps_sub00modes'):

    logger.critical( 'summary 2d cross ps estimation' )

    if bins_x is not None:
        khx = bins_x[:-1]
        if logk:
            dkx = bins_x[1] / bins_x[0]
            khx = khx * (dkx ** 0.5)
        else:
            dkx = bins_x[1] - bins_x[0]
            khx = khx + (dkx * 0.5)
    else:
        khx = None

    if bins_y is not None:
        khy = bins_y[:-1]
        if logk:
            dky = bins_y[1] / bins_y[0]
            khy = khy * (dky ** 0.5)
        else:
            dky = bins_y[1] - bins_y[0]
            khy = khy + (dky * 0.5)
    else:
        khy = None

    for ii, ps_name, ps2d, ct2d, bins_x, bins_y, khx, khy in\
            iter_2dps_list(ps_path, ps_name_list, tr_path=None, cross=True,
                    bins_xc=khx, bins_yc=khy, bins_x=bins_x, bins_y=bins_y, ps_ref=ps_ref):

        logger.info( ps_name )
        if not sim:
            ps = ps2d[0]
            ps_e = np.ma.std(ps2d[1:], axis=0)
            null = np.ma.mean(ps2d[1:], axis=0)
        else:
            ps = np.mean(ps2d, axis=0)
            ps_e = np.ma.std(ps2d, axis=0)
            null = np.zeros(ps.shape)

        khy_c, khx_c = np.meshgrid(khy, khx)
        khy_l, khx_l = np.meshgrid(bins_y[:-1], bins_x[:-1])
        khy_h, khx_h = np.meshgrid(bins_y[1:], bins_x[1:])

        results = [
                khx_l.flatten()[:, None], 
                khx_c.flatten()[:, None], 
                khx_h.flatten()[:, None], 
                khy_l.flatten()[:, None], 
                khy_c.flatten()[:, None], 
                khy_h.flatten()[:, None], 
                ps.flatten()[:, None],
                ps_e.flatten()[:, None],
                ]

        #results = [kh[:, None], ps[:, None], ps_e[:, None], null[:, None],
        #           shot[:, None], ps_cov]
        results = np.concatenate(results, axis=1)
        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[0]
        fname = ps_path + fname.replace('.h5', '_2d%s.txt'%suffix)
        header = '%23s %24s %24s %24s %24s %24s %24s %24s'%(
                'khx_l', 'khx_c', 'khx_h', 'khy_l', 'khy_c', 'khy_h', 'ps', 'ps_e')
        logger.debug( fname )
        np.savetxt(fname, results, fmt='%20.18e', header=header)

def summary_2daps(ps_path, ps_name_list, logk=True, bins_x=None, bins_y=None, sim=False, 
        suffix='', ps_ref='ps_sub00modes'):

    logger.critical( 'summary 2d ps estimation' )

    if bins_x is not None:
        khx = bins_x[:-1]
        if logk:
            dkx = bins_x[1] / bins_x[0]
            khx = khx * (dkx ** 0.5)
        else:
            dkx = bins_x[1] - bins_x[0]
            khx = khx + (dkx * 0.5)
    else:
        khx = None

    if bins_y is not None:
        khy = bins_y[:-1]
        if logk:
            dky = bins_y[1] / bins_y[0]
            khy = khy * (dky ** 0.5)
        else:
            dky = bins_y[1] - bins_y[0]
            khy = khy + (dky * 0.5)
    else:
        khy = None

    for ii, ps_name, ps2d, ct2d, bins_x, bins_y, khx, khy in\
            iter_2dps_list(ps_path, ps_name_list, tr_path=None, cross=False,
                    bins_xc=khx, bins_yc=khy, bins_x=bins_x, bins_y=bins_y, ps_ref=ps_ref):

        logger.info( ps_name )
        if not sim:
            ps = ps2d[0]
            ps_e = np.ma.std(ps2d[1:], axis=0)
            null = np.ma.mean(ps2d[1:], axis=0)
        else:
            ps = np.mean(ps2d, axis=0)
            ps_e = np.ma.std(ps2d, axis=0)
            null = np.zeros(ps.shape)

        khy_c, khx_c = np.meshgrid(khy, khx)
        khy_l, khx_l = np.meshgrid(bins_y[:-1], bins_x[:-1])
        khy_h, khx_h = np.meshgrid(bins_y[1:], bins_x[1:])

        results = [
                khx_l.flatten()[:, None], 
                khx_c.flatten()[:, None], 
                khx_h.flatten()[:, None], 
                khy_l.flatten()[:, None], 
                khy_c.flatten()[:, None], 
                khy_h.flatten()[:, None], 
                ps.flatten()[:, None],
                ps_e.flatten()[:, None],
                ]

        #results = [kh[:, None], ps[:, None], ps_e[:, None], null[:, None],
        #           shot[:, None], ps_cov]
        results = np.concatenate(results, axis=1)
        fname = ps_name_list[ps_name]
        if isinstance( fname, list): fname = fname[0]
        fname = ps_path + fname.replace('.h5', '_2d%s.txt'%suffix)
        header = '%23s %24s %24s %24s %24s %24s %24s %24s'%(
                'khx_l', 'khx_c', 'khx_h', 'khy_l', 'khy_c', 'khy_h', 'ps', 'ps_e')
        logger.debug( fname )
        np.savetxt(fname, results, fmt='%20.18e', header=header)
