"""Module to do the map-making."""

import matplotlib.pyplot as plt
import logging

from caput import mpiutil
from caput import mpiarray
from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.utils.path_util import output_path

from fpipe.timestream import timestream_task
from fpipe.map import algebra as al
from fpipe.map import mapbase

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

from tlpipe.rfi import interpolate
from tlpipe.rfi import gaussian_filter


logger = logging.getLogger(__name__)

from meerKAT_utils.constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from meerKAT_utils.constants import f_medium, f_large

__dtype__ = 'float32'



class DirtyMap(timestream_task.TimestreamTask, mapbase.MapBase):

    params_init = {
            #'ra_range' :  [0., 25.],
            #'ra_delta' :  0.5,
            #'dec_range' : [-4.0, 5.0],
            #'dec_delta' : 0.5,

            'healpix_map' : True,
            'nside' : 1024,

            'field_centre' : (12., 0.,),
            'pixel_spacing' : 0.5,
            'map_shape'     : (10, 10),

            'noise_weight' : True,

            'beam_data_path' : None,
            'beam_fwhm_at21cm' : 3.0/60.,
            'beam_cut'  : 0.01,

            'interpolation' : 'linear',
            'tblock_len' : 100,
            'data_sets'  : 'vis',
            'corr' : 'auto',
            'deweight_time_slope' : False,
            'deweight_time_mean'  : True,
            'pol_select': (0, 2), # only useful for ts
            'freq_select' : (0, 4), 

            'save_cov' : False,
            'diag_cov' : False,

            'save_localHI' : False,

            'baseline_file' : None,
            }

    prefix = 'dm_'

    def __init__(self, *args, **kwargs):

        super(DirtyMap, self).__init__(*args, **kwargs)
        #mapbase.MapBase.__init__(self)


    def setup(self):

        pass

    def init_ps_datasets(self, ts):

        pass

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if self.params['save_localHI']:
            if mpiutil.rank0:
                logger.info('save local HI')
            freq = ts['freq'][:] - 1420.
            local_hi = np.abs(freq) < 1
            ts.local_vis_mask[:, local_hi, ...] = False

        self.init_output()
        if 'ns_on' in ts.iterkeys():
            ns = ts['ns_on'].local_data
            ts.local_vis_mask[:] += ns[:, None, None, :]
            #ns = ts['ns_on'][:]
            #ts.vis_mask[:] += ns[:, None, None, :]

        func = self.init_ps_datasets(ts)

        ts.redistribute('frequency')

        vis_var = mpiarray.MPIArray.wrap(np.zeros(ts.vis.local_shape), 1)
        axis_order = tuple(xrange(len(ts.vis.shape)))
        ts.create_time_ordered_dataset('vis_var', data=vis_var, axis_order=axis_order)
        #var  = ts['vis_var'][:]
        #print mpiutil.rank, var.shape

        ts.freq_data_operate(self.init_vis, full_data=True, copy_data=False, 
                show_progress=show_progress, progress_step=progress_step, 
                keep_dist_axis=False)
        mpiutil.barrier()
        #vis_var = ts['vis_var'].local_data
        #vis_var = mpiutil.allreduce(vis_var)
        #ts['vis_var'][:] = vis_var

        #print ts['vis_var'].shape

        if not func is None:

            #ts.redistribute('time')
            ts.redistribute('frequency')
            func(self.make_map, full_data=False, copy_data=True, 
                    show_progress=show_progress, 
                    progress_step=progress_step, keep_dist_axis=False)

        mpiutil.barrier()

        self.df.close()

    def init_vis(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        time = ts['sec1970'][:]
        n_time = time.shape[0]
        tblock_len = self.params['tblock_len']
        if tblock_len is None:
            tblock_len = n_time

        if self.params['deweight_time_slope']:
            n_poly = 2
        else:
            n_poly = 1

        vis_var = ts['vis_var'].local_data

        baseline_file = self.params['baseline_file'] 
        if baseline_file is not None:
            bsl = fit_baseline(vis, vis_mask, time, baseline_file)
            vis -= bsl
        else:
            bsl = np.ones_like(vis)

        logger.debug('est. var %d %d'%(n_time, tblock_len))
        #_vis = np.ma.array(vis.copy())
        #_vis.mask = vis_mask
        #median = np.ma.median(_vis, axis=0)
        #vis -= median[None, ...]
        if self.params['noise_weight']:
            for st in range(0, n_time, tblock_len):
                et = st + tblock_len
                _time = time[st:et] #ts['sec1970'][st:et]
                _vis_mask = (vis_mask[st:et,...]).astype('bool')
                _bsl = bsl[st:et]

                #_fit = sub_ortho_poly(vis[st:et,...], _time, ~_vis_mask, n_poly)
                #vis[st:et,...] -= _fit

                _vis = vis[st:et,...]
                _vis = np.ma.array(_vis, mask = _vis_mask)
                #vis[st:et,...] -= np.ma.median(_vis, axis=0)
                _vis[_vis_mask] = 0.

                # rm bright sources for var est.
                _vis.shape = _time.shape + (-1, )
                _vis_mask.shape = _vis.shape
                bg  = gaussian_filter.GaussianFilter(
                        interpolate.Interpolate(_vis, _vis_mask).fit(), 
                        time_kernal_size=0.5, freq_kernal_size=1, 
                        filter_direction = ('time', )).fit()
                _vis = _vis - bg
                _vis.shape = (-1, ) + vis.shape[1:]
                _vis_mask.shape = _vis.shape
                _vis[_vis_mask] = 0.

                #for i in range(5):
                _vars = sp.sum(_vis ** 2., axis=0)
                _cont = sp.sum(~_vis_mask, axis=0) * 1.
                _bad = _cont == 0
                _cont[_bad] = np.inf
                _vars /= _cont
                #_vis_mask += (_vis - 3 * np.sqrt(_vars[None, ...])) > 0.
                #_vis[_vis_mask] = 0.
                msg = 'min vars = %f, max vars = %f'%(_vars.min(), _vars.max())
                logger.debug(msg)
                #logger.info(msg)
                #_vars[_bad] = T_infinity ** 2.
                #_bad = _vars < T_small ** 2
                #_vars[_bad] = T_small ** 2
                #vis_var[st:et, li, ...] += _vars[None, :] * vis_fit[st:et, ...]
                #vis_var[st:et, li, ...] += _vars[None, :] * median[None, ...]
                vis_var[st:et, li, ...] += _vars[None, :] #* _bsl
                #vis_var[st:et, li, ...][_vis_mask, ...] = T_infinity ** 2.
                #vis_var[st:et, li, ...] += _vars[None, :] #* _fit**2
        else:
            vis_var[:] = 1.

    def init_output(self):

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))
        #if mpiutil.rank0:
        #    self.df = h5py.File(output_file, mode='w')
        self.allocate_output(output_file, 'w')

    def map_map(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        pass

    def finish(self):

        if mpiutil.rank0:
            print 'Finishing MapMaking.'

        mpiutil.barrier()

class MakeMap_FlatSky(DirtyMap):

    def setup(self):

        params = self.params
        self.n_ra, self.n_dec = params['map_shape']
        self.map_shp = (self.n_ra, self.n_dec)
        self.spacing = params['pixel_spacing']
        self.dec_spacing = self.spacing
        # Negative sign because RA increases from right to left.
        self.ra_spacing = -self.spacing/sp.cos(params['field_centre'][1]*sp.pi/180.)

        axis_names = ('ra', 'dec')
        map_tmp = np.zeros(self.map_shp, dtype=__dtype__)
        map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        map_tmp.set_axis_info('ra',   params['field_centre'][0], self.ra_spacing)
        map_tmp.set_axis_info('dec',  params['field_centre'][1], self.dec_spacing)
        self.map_tmp = map_tmp


    def process(self, ts):

        ra_axis  = self.map_tmp.get_axis('ra')
        dec_axis = self.map_tmp.get_axis('dec')
        if mpiutil.rank0:
            msg = 'RANK %03d:  RA  Range [%5.2f, %5.2f] deg'%(
                    mpiutil.rank, ra_axis.min(), ra_axis.max())
            logger.info(msg)
            msg = 'RANK %03d:  Dec Range [%5.2f, %5.2f] deg\n'%(
                    mpiutil.rank, dec_axis.min(), dec_axis.max())
            logger.info(msg)

        super(MakeMap_FlagSky, self).process(ts)

    def init_ps_datasets(self, ts):

        ts.main_data_name = self.params['data_sets']
        n_time, n_freq, n_pol, n_bl = ts.main_data.shape
        tblock_len = self.params['tblock_len']

        freq = ts['freq']
        freq_c = freq[n_freq//2]
        freq_d = freq[1] - freq[0]

        field_centre = self.params['field_centre']

        self.pol = ts['pol'][:]
        self.bl  = ts['blorder'][:]

        # for now, we assume no frequency corr, and thermal noise only.

        ra_spacing = self.ra_spacing
        dec_spacing = self.dec_spacing


        axis_names = ('bl', 'pol', 'freq', 'ra', 'dec')
        dirty_map_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        dirty_map_tmp = al.make_vect(dirty_map_tmp, axis_names=axis_names)
        dirty_map_tmp.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
        dirty_map_tmp.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
        dirty_map_tmp.set_axis_info('freq', freq_c, freq_d)
        dirty_map_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        dirty_map_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        self.map_axis_names = axis_names
        #self.dirty_map = dirty_map_tmp

        self.create_dataset_like('dirty_map',  dirty_map_tmp)
        self.create_dataset_like('clean_map',  dirty_map_tmp)
        self.create_dataset_like('noise_diag', dirty_map_tmp)
        self.df['mask'] = np.zeros([n_bl, n_pol, n_freq])
        #self.mask = np.zeros([n_bl, n_pol, n_freq])

        if self.params['diag_cov']:
            axis_names = ('bl', 'pol', 'freq', 'ra', 'dec')
            cov_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        else:
            axis_names = ('bl', 'pol', 'freq', 'ra', 'dec', 'ra', 'dec')
            cov_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp + self.map_shp)
        cov_tmp = al.make_vect(cov_tmp, axis_names=axis_names)
        cov_tmp.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
        cov_tmp.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
        cov_tmp.set_axis_info('freq', freq_c, freq_d)
        cov_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        cov_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        #self.cov = cov_tmp
        
        self.create_dataset_like('cov_inv', cov_tmp)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        #func = ts.freq_pol_and_bl_data_operate
        func = ts.freq_data_operate

        return func

    def make_map(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #print "make map vis shape = ", vis.shape
        if not isinstance(gi, tuple): gi = (gi, )
        if not isinstance(li, tuple): li = (li, )
        freq = ts.freq[gi[0]] * 1.e-3
        beam_fwhm = self.params['beam_fwhm_at21cm'] * 1.42 / freq
        print "RANK%03d:"%mpiutil.rank + \
                " Local  (" + ("%04d, "*len(li))%li + ")," +\
                " Global (" + ("%04d, "*len(gi))%gi + ")"  +\
                " at %5.4fGHz (fwhm = %4.3f deg)"%(freq, beam_fwhm)
        if vis.dtype == np.complex:
            vis = np.abs(vis)

        time = ts['sec1970'][:]
        tblock_len = 6000
        n_time, n_pol, n_bl = vis.shape
        ff = gi[0]
        vis_idx = ((bb, pp, ff) for bb in range(n_bl) for pp in range(n_pol))
        vis_axis_names = ('bl', 'pol', 'freq')
        map_axis_names = self.map_axis_names

        ra_axis  = self.map_tmp.get_axis('ra')
        dec_axis = self.map_tmp.get_axis('dec')

        map_shp = ra_axis.shape + dec_axis.shape
        if self.params['diag_cov']:
            _ci = np.zeros((np.product(map_shp),), dtype=__dtype__)
        else:
            _ci = np.zeros((np.product(map_shp), np.product(map_shp)), dtype=__dtype__)
        _dm = np.zeros((np.product(map_shp),), dtype=__dtype__)

        for _vis_idx in vis_idx:

            b_idx, p_idx, f_idx = _vis_idx

            map_idx = [_vis_idx[ii] for ii, name in enumerate(vis_axis_names) 
                    if name in map_axis_names]
            map_idx = tuple(map_idx)

            msg = "RANK%03d:"%mpiutil.rank + \
                    " VIS (" + ("%03d, "*len(_vis_idx))%_vis_idx + ")" +\
                    " Map (" + ("%03d, "*len(map_idx)%map_idx) + ")"
            if mpiutil.rank0:
                logger.info(msg)
            else:
                logger.debug(msg)

            _vis = vis[:, p_idx, b_idx]
            _vis_mask = vis_mask[:, p_idx, b_idx]

            if np.all(_vis_mask):
                print " VIS (" + ("%03d, "*len(_vis_idx))%_vis_idx + ")" +\
                        " All masked, continue"
                #self.df['mask'][map_idx[:-1]] = 1
                continue

            ra   = ts['ra'][:,  b_idx]
            dec  = ts['dec'][:, b_idx]
            vis_var = ts['vis_var'][:, f_idx, p_idx, b_idx]

            vis_shp = _vis.shape
            ra_shp  = ra.shape
            var_shp = vis_var.shape
            logger.debug('RANK %02d: vis shape'%mpiutil.rank+' %d'*len(vis_shp)%vis_shp)
            logger.debug('RANK %02d: ra  shape'%mpiutil.rank+' %d'*len(ra_shp)%ra_shp)
            logger.debug('RANK %02d: var shape'%mpiutil.rank+' %d'*len(var_shp)%var_shp)

            for st in range(0, n_time, tblock_len):
                et = st + tblock_len

                timestream2map(_vis[st:et, ...], 
                               _vis_mask[st:et, ...], 
                               vis_var[st:et, ...], 
                               time[st:et], 
                               ra[st:et, ...], 
                               dec[st:et, ...], 
                               ra_axis, dec_axis, 
                               _ci, _dm,
                               diag_cov = self.params['diag_cov'],
                               beam_size= beam_fwhm,
                               beam_cut = self.params['beam_cut'])

        logger.debug('write to disk')
        _dm.shape = map_shp
        self.write_block_to_dset('dirty_map', map_idx, _dm)
        if self.params['diag_cov']:
            _ci.shape = map_shp
        else:
            _ci.shape = map_shp * 2
        self.write_block_to_dset('cov_inv', map_idx, _ci)
        del _ci, _dm
        gc.collect()

class MakeMap_Ionly(MakeMap_FlatSky):

    def init_ps_datasets(self, ts):

        ts.lin2I()

        func = super(MakeMap_Ionly, self).init_ps_datasets(ts)
        return func

class MakeMap_CombineAll(MakeMap_FlatSky):

    def init_ps_datasets(self, ts):

        ts.lin2I()

        ts.main_data_name = self.params['data_sets']
        n_time, n_freq, n_pol, n_bl = ts.main_data.shape
        tblock_len = self.params['tblock_len']

        freq = ts['freq']
        freq_c = freq[n_freq//2]
        freq_d = freq[1] - freq[0]

        field_centre = self.params['field_centre']

        self.pol = ts['pol'][:]
        self.bl  = ts['blorder'][:]

        # for now, we assume no frequency corr, and thermal noise only.

        ra_spacing = self.ra_spacing
        dec_spacing = self.dec_spacing

        axis_names = ('freq', 'ra', 'dec')

        msg = 'init dirty map dsets'
        logger.debug(msg)
        dirty_map_shp  = (n_freq, ) +  self.map_shp
        dirty_map_info = {
                'ra_delta'    : self.ra_spacing,
                'ra_centre'   : field_centre[0],
                'dec_delta'   : self.dec_spacing,
                'dec_centre'  : field_centre[1],
                'freq_delta'  : freq_d,
                'freq_centre' : freq_c,
                'axes'        : axis_names,
                }
        self.map_axis_names = axis_names
        #self.map_tmp = map_tmp

        self.create_dataset('dirty_map',  dirty_map_shp, dirty_map_info, __dtype__)
        self.create_dataset('clean_map',  dirty_map_shp, dirty_map_info, __dtype__)
        self.create_dataset('noise_diag', dirty_map_shp, dirty_map_info, __dtype__)

        self.df['mask'] = np.zeros(n_freq)

        msg = 'init cov dsets'
        logger.debug(msg)
        if self.params['diag_cov']:
            axis_names = ('freq', 'ra', 'dec')
            cov_shp = (n_freq, ) +  self.map_shp 
        else:
            axis_names = ('freq', 'ra', 'dec', 'ra', 'dec')
            cov_shp = (n_freq, ) +  self.map_shp + self.map_shp
        cov_info = {
                'ra_delta'    : self.ra_spacing,
                'ra_centre'   : field_centre[0],
                'dec_delta'   : self.dec_spacing,
                'dec_centre'  : field_centre[1],
                'freq_delta'  : freq_d,
                'freq_centre' : freq_c,
                'axes'        : axis_names,
                }
        self.create_dataset('cov_inv', cov_shp, cov_info, __dtype__)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        #func = ts.freq_pol_and_bl_data_operate
        func = ts.freq_data_operate

        msg = 'RANK %03d: everything init done'%mpiutil.rank
        logger.debug(msg)
        mpiutil.barrier()
        return func

def timestream2map(vis_one, vis_mask, vis_var, time, ra, dec, ra_axis, dec_axis, 
        cov_inv_block, dirty_map, diag_cov=False, beam_size=3./60.,  beam_cut = 0.01,):

    map_shp = ra_axis.shape + dec_axis.shape

    #cov_inv_block = np.zeros((np.product(map_shp), np.product(map_shp)),
    #        dtype=__dtype__)
    beam_sig = beam_size  / (2. * np.sqrt(2.*np.log(2.)))

    vis_mask = (vis_mask.copy()).astype('bool')
    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.
    
    _good  = ( ra  < max(ra_axis))
    _good *= ( ra  > min(ra_axis))
    _good *= ( dec < max(dec_axis))
    _good *= ( dec > min(dec_axis))
    _good *= ~vis_mask
    if np.sum(_good) == 0: return

    ra   = ra[_good] * np.pi / 180.
    dec  = dec[_good]* np.pi / 180.
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time = time[_good]
    vis_var = vis_var[_good]

    ra_centr  = ra_axis  * np.pi / 180.
    dec_centr = dec_axis * np.pi / 180.

    logger.debug('est. pointing')
    P = (np.sin(dec[:, None]) * np.sin(dec_centr[None, :]))[:, :, None]\
      + (np.cos(dec[:, None]) * np.cos(dec_centr[None, :]))[:, :, None]\
      * (np.cos(ra[:, None] - ra_centr[None, :]))[:, None, :]
    #P = (np.sin(ra[:, None]) * np.sin(ra_centr[None, :]))[:, :, None]\
    #  + (np.cos(ra[:, None]) * np.cos(ra_centr[None, :]))[:, :, None]\
    #  * (np.cos(dec[:, None] - dec_centr[None, :]))[:, None, :]

    P  = np.arccos(P) * 180. / np.pi
    P  = np.exp(- 0.5 * (P / beam_sig) ** 2)
    

    P.shape = (ra.shape[0], -1)
    if beam_cut is None:
        P_max = np.argmax(P, axis=1)
        P *= 0.
        P[tuple(range(ra.shape[0])), tuple(P_max)] = 1.
    else:
        if mpiutil.rank0:
            logger.info('beam cut %f'%(beam_cut))
        P[P < beam_cut] *= 0.
        #P_norm = np.sum(P, axis=1)
        P_norm = np.max(P, axis=1)
        P_norm[P_norm==0] = np.inf
        P /= P_norm[:, None]
        #P /=  2. * np.pi * (beam_sig * np.pi / 180.) ** 2.


    vis_var[vis_var==0] = np.inf #T_infinity ** 2.
    noise_inv_weight = 1. /vis_var

    weight = noise_inv_weight

    logger.debug('est. dirty map')
    dirty_map += np.dot(P.T, vis_one * weight)
    #dirty_map = dirty_map.astype(__dtype__)
    #dirty_map.shape = map_shp

    logger.debug('est. noise inv')
    weight = np.eye(vis_one.shape[0]) * weight
    #cov_inv_block += np.dot(np.dot(P.T, weight) , P)
    #P[P!=0] = 1.
    if diag_cov:
        cov_inv_block += np.diag(multi_dot([P.T, weight, P]))
    else:
        cov_inv_block += multi_dot([P.T, weight, P])
    #cov_inv_block.shape = map_shp * 2
    #cov_inv_block[cov_inv_block<1.e-10] = 0.

    del weight, P
    gc.collect()

    #return dirty_map, cov_inv_block

def timestream2map_GBT(vis_one, vis_mask, time, ra, dec, map_tmp, n_poly = 1, 
        interpolation = 'linear'):

    vis_mask = (vis_mask.copy()).astype('bool')

    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.

    cov_inv_block = np.zeros(map_tmp.shape * 2)
    
    polys = ortho_poly(time, n_poly, ~vis_mask, 0)
    amps = np.sum(polys * vis_one[None, :], -1)
    vis_fit = np.sum(amps[:, None] * polys, 0)
    vis_one -= vis_fit

    _good  = ( ra  < max(map_tmp.get_axis('ra') ))
    _good *= ( ra  > min(map_tmp.get_axis('ra') ))
    _good *= ( dec < max(map_tmp.get_axis('dec')))
    _good *= ( dec > min(map_tmp.get_axis('dec')))
    _good *= ~vis_mask
    if np.sum(_good) < 5: 
        print 'bad block < 5'
        return al.zeros_like(map_tmp), cov_inv_block

    ra   = ra[_good]
    dec  = dec[_good]
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time = time[_good]

    P = Pointing(('ra', 'dec'), (ra, dec), map_tmp, interpolation)

    _vars = sp.sum(vis_one ** 2.)
    _cont = sp.sum(~vis_mask)
    if _cont != 0:
        _vars /= _cont
    else:
        _vars = T_infinity ** 2.
    if _vars < T_small ** 2:
        print "vars too small"
        _vars = T_small ** 2
    #thermal_noise = np.var(vis_one)
    thermal_noise = _vars
    vis_one = al.make_vect(vis_one[None, :], axis_names=['freq', 'time'])
    N = Noise(vis_one, time)
    N.add_thermal(thermal_noise)
    if n_poly == 1:
        N.deweight_time_mean(T_huge ** 2.)
    elif n_poly == 2:
        N.deweight_time_slope(T_huge ** 2.)
    N.finalize(frequency_correlations=False, preserve_matrices=False)
    vis_weighted = N.weight_time_stream(vis_one)
    dirty_map = P.apply_to_time_axis(vis_weighted)[0,...]

    P.noise_channel_to_map(N, 0, cov_inv_block)

    return dirty_map, cov_inv_block


def make_cleanmap_GBT(dirty_map, cov_inv_block, threshold=1.e-5):
    
    map_shp = dirty_map.shape
    dirty_map.shape = (np.prod(map_shp), )
    cov_inv_block.shape = (np.prod(map_shp), np.prod(map_shp))
    #cov_inv_block += np.eye(np.prod(map_shp)) * 1.e-1
    cov_inv_diag, Rot = linalg.eigh(cov_inv_block, overwrite_a=True)
    map_rotated = sp.dot(Rot.T, dirty_map)
    bad_modes = cov_inv_diag <= threshold * cov_inv_diag.max()
    print "cov_inv_diag max = %4.1f"%cov_inv_diag.max()
    print "discarded: %4.1f" % (100.0 * sp.sum(bad_modes) / bad_modes.size) +\
                "% of modes" + " (tho=%f)"%threshold
    map_rotated[bad_modes] = 0.
    cov_inv_diag[bad_modes] = 1.
    print "cov_inv_diag min = %5.4e"%cov_inv_diag.min()
    #cov_inv_diag[cov_inv_diag == 0] = 1.
    #print "cov_inv_diag min = %5.4e"%cov_inv_diag.min()
    map_rotated /= cov_inv_diag
    clean_map = sp.dot(Rot, map_rotated)
    clean_map.shape = map_shp

    noise_diag = 1./cov_inv_diag
    noise_diag[bad_modes] = 0.

    tmp_mat = Rot * noise_diag
    for jj in range(np.prod(map_shp)):
        noise_diag[jj] = sp.dot(tmp_mat[jj, :], Rot[jj, :])
    noise_diag.shape = map_shp
    noise_diag[noise_diag<1.e-20] = 0.

    del cov_inv_diag, Rot, tmp_mat, map_rotated
    gc.collect()

    return clean_map, noise_diag

def fit_baseline(vis, mask, time, baseline_file):

    logger.info('Load Baseline from: %s'%baseline_file)

    with h5py.File(baseline_file, 'r') as f:
        bsl  = f['baseline'][:]
        bsl_time = f['time'][:]
        
    bsl_xx = interp1d(bsl_time, bsl, bounds_error=False,
                      fill_value='extrapolate')(time) + 20.
    
    bsl = np.zeros(vis.shape)

    for i in range(19):
        #window = (~mask[:, 0, i]).astype('int')
        _msk = mask[:, 0, i]
        _vis_xx = np.matrix(vis[:, 0, i][:, None])
    
        _bsl_xx  = np.matrix(bsl_xx[:, None])
        a_xx = (_bsl_xx.T * _bsl_xx)**(-1) * _bsl_xx.T * _vis_xx
        bsl[:, 0, i] = (np.array(a_xx) * np.array(_bsl_xx)).flat
    
    return bsl

def sub_ortho_poly(vis, time, mask, n):

    logger.debug('sub. mean')
    
    window = mask
    x = time
    
    upbroad = (slice(None), slice(None)) + (None, ) * (window.ndim - 1)
    window = window[None, ...]
    
    x_mid = (x.max() + x.min())/2.
    x_range = (x.max() - x.min()) /2.
    x = (x - x_mid) / x_range
    
    n = np.arange(n)[:, None]
    x = x[None, :]
    polys = special.eval_legendre(n, x, out=None)
    polys = polys[upbroad] * window

    for ii in range(n.shape[0]):
        for jj in range(ii):
            amp = np.sum(polys[ii, ...] * polys[jj, ...], axis=0)
            polys[ii, ...] -= amp[None, ...] * polys[jj, ...]
            
        norm  = np.sqrt(np.sum(polys[ii] ** 2, axis=0))
        norm[norm==0] = np.inf
        polys[ii] /= norm[None, ...]
    
    amp = np.sum(polys * vis[None, ...], 1)
    vis_fit = np.sum(amp[:, None, ...] * polys, 0)
    vis -= vis_fit
    #return vis
    return vis_fit

def ortho_poly(x, n, window=1., axis=-1):
    """Generate orthonormal basis polynomials.

    Generate the first `n` orthonormal basis polynomials over the given domain
    and for the given window using the Gram-Schmidt process.
    
    Parameters
    ----------
    x : 1D array length m
        Functional domain.
    n : integer
        number of polynomials to generate. `n` - 1 is the maximum order of the
        polynomials.
    window : 1D array length m
        Window (weight) function for which the polynomials are orthogonal.

    Returns
    -------
    polys : n by m array
        The n polynomial basis functions. Normalization is such that
        np.sum(polys[i,:] * window * polys[j,:]) = delta_{ij}
    """
    
    if np.any(window < 0):
        raise ValueError("Window function must never be negative.")
    # Check scipy versions. If there is a stable polynomial package, use it.
    s_ver = sp.__version__.split('.')
    major = int(s_ver[0])
    minor = int(s_ver[1])
    if major <= 0 and minor < 8:
        new_sp = False
        if n > 20:
            raise NotImplementedError("High order polynomials unstable.")
    else:
        new_sp = True
    # Get the broadcasted shape of `x` and `window`.
    # The following is the only way I know how to get the broadcast shape of
    # x and window.
    # Turns out I could use np.broadcast here.  Fix this later.
    print x.shape, window.shape
    shape = np.broadcast(x, window).shape
    m = shape[axis]
    # Construct a slice tuple for up broadcasting arrays.
    upbroad = [slice(sys.maxsize)] * len(shape)
    upbroad[axis] = None
    upbroad = tuple(upbroad)
    # Allocate memory for output.
    polys = np.empty((n,) + shape, dtype=__dtype__)
    # For stability, rescale the domain.
    x_range = np.amax(x, axis) - np.amin(x, axis)
    x_mid = (np.amax(x, axis) + np.amin(x, axis)) / 2.
    x = (x - x_mid[upbroad]) / x_range[upbroad] * 2
    # Reshape x to be the final shape.
    x = np.zeros(shape, dtype=__dtype__) + x
    # Now loop through the polynomials and construct them.
    # This array will be the starting polynomial, before orthogonalization
    # (only used for earlier versions of scipy).
    if not new_sp:
        basic_poly = np.ones(shape, dtype=__dtype__) / np.sqrt(m)
    for ii in range(n):
        # Start with the basic polynomial.
        # If we have an up-to-date scipy, start with nearly orthogonal
        # functions.  Otherwise, just start with the next polynomial.
        if not new_sp:
            new_poly = basic_poly.copy()
        else:
            new_poly = special.eval_legendre(ii, x)
        # Orthogonalize against all lower order polynomials.
        for jj in range(ii):
            new_poly -= (np.sum(new_poly * window * polys[jj,:], axis)[upbroad]
                         * polys[jj,:])
        # Normalize, accounting for possibility that all data is masked. 
        norm = np.array(np.sqrt(np.sum(new_poly**2 * window, axis)))
        if norm.shape == ():
            if norm == 0:
                bad_inds = np.array(True)
                norm = np.array(1.)
            else:
                bad_inds = np.array(False)
        else:
            bad_inds = norm == 0
            norm[bad_inds] = 1.
        new_poly /= norm[upbroad]
        #new_poly[bad_inds[None,]] = 0
        new_poly *= ~bad_inds[upbroad]
        # Copy into output.
        polys[ii,:] = new_poly
        # Increment the base polynomial with another power of the domain for
        # the next iteration.
        if not new_sp:
            basic_poly *= x
    return polys

