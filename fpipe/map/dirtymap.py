"""Module to do the map-making."""

import matplotlib.pyplot as plt
import logging

from caput import mpiutil
from caput import mpiarray
from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.utils.path_util import output_path

from fpipe.timestream import timestream_task
from fpipe.map import algebra as al
from fpipe.map.pointing import Pointing
from fpipe.map.noise_model import Noise
from fpipe.map import mapbase

import healpy as hp
import numpy as np
import scipy as sp
#from scipy import linalg
from numpy import linalg
from scipy import special
from scipy.ndimage import gaussian_filter
import h5py
import sys
import gc

logger = logging.getLogger(__name__)

from constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from constants import f_medium, f_large



class DirtyMap_GBT(timestream_task.TimestreamTask, mapbase.MapBase):

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
            'ra_block_olp' : 20,
            'ra_block_len' : 10,

            'beam_fwhm' : 3./60.,
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
            }

    prefix = 'dm_'

    def __init__(self, *args, **kwargs):

        super(DirtyMap_GBT, self).__init__(*args, **kwargs)
        mapbase.MapBase.__init__(self)

    def setup(self):


        params = self.params
        self.n_ra, self.n_dec = params['map_shape']
        self.map_shp = (self.n_ra, self.n_dec)
        self.spacing = params['pixel_spacing']
        self.dec_spacing = self.spacing
        # Negative sign because RA increases from right to left.
        self.ra_spacing = -self.spacing/sp.cos(params['field_centre'][1]*sp.pi/180.)

        axis_names = ('ra', 'dec')
        map_tmp = np.zeros(self.map_shp)
        map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        map_tmp.set_axis_info('ra',   params['field_centre'][0], self.ra_spacing)
        map_tmp.set_axis_info('dec',  params['field_centre'][1], self.dec_spacing)
        self.map_tmp = map_tmp

        ## split ra
        #ra_total_len = self.n_ra
        #ra_block_olp = self.params['ra_block_olp']
        #ra_block_len = self.params['ra_block_len']
        #ra_block_num = int(np.ceil(ra_total_len / float(ra_block_len)))
        #if mpiutil.rank0:
        #    logger.debug('split RA into %d x %d'%(ra_block_num, ra_block_len))
        #    logger.debug('RA overlap set to %d'%ra_block_olp)

        #map_global_slice = []
        #map_local_slice = []
        #for i in xrange(ra_block_num):
        #    ra_block_st = min( i * ra_block_len,  ra_total_len - 1)
        #    ra_block_ed = min((i + 1) * ra_block_len, ra_total_len)
        #    map_global_slice.append(slice(ra_block_st, ra_block_ed))

        #    ra_ovlap_st = max(ra_block_st - ra_block_olp, 0)
        #    ra_ovlap_ed = min(ra_block_ed + ra_block_olp, ra_total_len)

        #    #ra_local_len = ra_ovlap_ed - ra_ovlap_st
        #    #ra_ovlap_st = ra_block_st - ra_ovlap_st
        #    #ra_ovlap_ed = ra_ovlap_st + ra_block_len

        #    map_local_slice.append(slice(ra_ovlap_st, ra_ovlap_ed))

        #self.map_global_slice = map_global_slice 
        #self.map_local_slice = map_local_slice 

            #map_local_shp = (ra_local_len, self.n_dec)
            #logger.debug('RANK %02d: local map shape %3d x %3d'%\
            #        ((mpiutil.rank, ) + self.map_local_shp))

        #map_local_tmp = np.zeros(self.map_local_shp)
        #map_local_tmp = al.make_vect(map_local_tmp, axis_names=axis_names)
        #ra_axis_local = map_tmp.get_axis('ra')[ra_ovlap_st : ra_ovlap_ed]
        #ra_local_centre = ra_axis_local[ra_local_len//2]
        #map_local_tmp.set_axis_info('ra',   ra_local_centre, self.ra_spacing)
        #map_local_tmp.set_axis_info('dec',  params['field_centre'][1], self.dec_spacing)
        #self.map_local_tmp = map_local_tmp


    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ra_axis  = self.map_tmp.get_axis('ra')
        dec_axis = self.map_tmp.get_axis('dec')
        msg = 'RANK %03d:  RA  Range [%5.2f, %5.2f] deg'%(
                mpiutil.rank, ra_axis.min(), ra_axis.max())
        logger.info(msg)
        msg = 'RANK %03d:  Dec Range [%5.2f, %5.2f] deg\n'%(
                mpiutil.rank, dec_axis.min(), dec_axis.max())
        logger.info(msg)


        self.init_output()
        ns = ts['ns_on'].local_data
        ts.local_vis_mask[:] += ns[:, None, None, :]

        func = self.init_ps_datasets(ts)

        vis_var = mpiarray.MPIArray.wrap(np.zeros(ts.vis.local_shape), 0)
        #vis_var = mpiarray.MPIArray.wrap(np.ones(ts.vis.local_shape), 0)
        axis_order = tuple(xrange(len(ts.vis.shape)))
        ts.create_time_ordered_dataset('vis_var', data=vis_var, axis_order=axis_order)
        var  = ts['vis_var'][:]
        #print mpiutil.rank, var.shape

        ts.bl_data_operate(self.init_vis, full_data=True, copy_data=False, 
                show_progress=show_progress, progress_step=progress_step, 
                keep_dist_axis=False)
        mpiutil.barrier()
        vis_var = ts['vis_var'].local_data
        vis_var = mpiutil.allreduce(vis_var)
        ts['vis_var'][:] = vis_var

        if not func is None:

            ts.redistribute('time')
            func(self.make_map, full_data=False, copy_data=True, 
                    show_progress=show_progress, 
                    progress_step=progress_step, keep_dist_axis=False)

            self.df.close()

        mpiutil.barrier()

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

        for st in range(0, n_time, tblock_len):
            et = st + tblock_len
            _time = time[st:et] #ts['sec1970'][st:et]

            _vis_mask = (vis_mask[st:et,...]).astype('bool')
            _vis = vis[st:et,...]
            _vis[_vis_mask] = 0.

            sub_ortho_poly(_vis, _time, ~_vis_mask, n_poly)

            vis[st:et,...] = _vis

            _vars = sp.sum(_vis ** 2., axis=0)
            _cont = sp.sum(~_vis_mask, axis=0) * 1.
            _bad = _cont == 0
            _cont[_bad] = np.inf
            _vars /= _cont
            #_vars[_bad] = T_infinity ** 2.
            _bad = _vars < T_small ** 2
            _vars[_bad] = T_small ** 2
            vis_var[st:et, ..., gi] += _vars[None, :]


    def init_output(self):

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))
        self.allocate_output(output_file, 'w')

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

        self.create_dataset_like('dirty_map',  dirty_map_tmp)

        self.create_dataset_like('clean_map',  dirty_map_tmp)

        self.create_dataset_like('noise_diag', dirty_map_tmp)

        self.df['mask'] = np.zeros([n_bl, n_pol, n_freq])

        #if self.params['save_cov']:
        axis_names = ('bl', 'pol', 'freq', 'ra', 'dec', 'ra', 'dec')
        cov_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp + self.map_shp)
        cov_tmp = al.make_vect(cov_tmp, axis_names=axis_names)
        cov_tmp.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
        cov_tmp.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
        cov_tmp.set_axis_info('freq', freq_c, freq_d)
        cov_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        cov_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        self.create_dataset_like('cov_inv', cov_tmp)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        func = ts.freq_pol_and_bl_data_operate

        return func

    def make_map(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        print "make map vis shape = ", vis.shape

        vis_idx = gi[::-1]
        vis_axis_names = ('bl', 'pol', 'freq')
        map_axis_names = self.map_axis_names
        map_idx = [vis_idx[ii] for ii, name in enumerate(vis_axis_names) 
                if name in map_axis_names]
        map_idx = tuple(map_idx)
        #map_slice       = (slice(self.ra_block_st, self.ra_block_ed), slice(None))
        #map_local_slice = (slice(self.ra_ovlap_st, self.ra_ovlap_ed), slice(None))

        print "RANK%03d:"%mpiutil.rank + \
                " Local  (" + ("%03d, "*len(li))%li + ")," +\
                " Global (" + ("%03d, "*len(gi))%gi + ")"

        if np.all(vis_mask):
            print "\t All masked, continue"
            self.df['mask'][map_idx[:-1]] = 1
            return

        vis_shp = vis.shape
        ra   = ts['ra'][:, gi[-1]]
        dec  = ts['dec'][:, gi[-1]]
        vis_var = ts['vis_var'][ (slice(None), ) + gi ]

        ra_shp = ra.shape
        var_shp = vis_var.shape
        logger.debug('RANK %02d: vis shape'%mpiutil.rank + ' %d'*len(vis_shp)%vis_shp)
        logger.debug('RANK %02d: ra  shape'%mpiutil.rank + ' %d'*len(ra_shp)%ra_shp)
        logger.debug('RANK %02d: var shape'%mpiutil.rank + ' %d'*len(var_shp)%var_shp)

        time = ts['sec1970'][:]
        if vis.dtype == np.complex:
            vis = np.abs(vis)

        ra_axis  = self.map_tmp.get_axis('ra')
        dec_axis = self.map_tmp.get_axis('dec')

        ra_axis_edges = self.map_tmp.get_axis_edges('ra')
        ra_num = ra_axis.shape[0]
        ra_min = np.digitize(ra.max(), ra_axis_edges) - 1
        ra_max = np.digitize(ra.min(), ra_axis_edges) - 1
        ra_min = max(ra_min, 0)
        ra_min = min(ra_min, ra_num)
        ra_max = min(ra_max, ra_num)
        ra_max = max(ra_max, 0)
        ra_block_len = ra_max - ra_min
        if ra_block_len == 0: return
        gslice = ( slice(ra_min, ra_max), slice(None) ) 
        logger.debug('RANK %02d: ra [%3d - %3d] %3d'%(
            mpiutil.rank, ra_min, ra_max, ra_block_len))

        ra_block_olp = self.params['ra_block_olp']
        ra_ovlap_st = max(ra_min - ra_block_olp, 0)
        ra_ovlap_ed = min(ra_max + ra_block_olp, ra_num)
        lslice = ( slice(ra_ovlap_st, ra_ovlap_ed), slice(None) )
        logger.debug('RANK %02d: ra ovlap [%3d - %3d]'%(
            mpiutil.rank, ra_ovlap_st, ra_ovlap_ed))

        ra_map_st = ra_min - ra_ovlap_st
        ra_map_ed = ra_map_st + ra_block_len
        mslice = ( slice(ra_map_st, ra_map_ed), slice(None) )
        logger.debug('RANK %02d: ra map [%3d - %3d]'%(
            mpiutil.rank, ra_map_st, ra_map_ed))

        _dm, _ci = timestream2map(vis, vis_mask, vis_var, time, ra, dec, 
                ra_axis[lslice[0]], dec_axis, beam_size=self.params['beam_fwhm'],
                beam_cut = self.params['beam_cut'])

        if _dm is not None:
            self.df['dirty_map' ][map_idx + gslice] += _dm[mslice]
            self.df['cov_inv'][map_idx + gslice + lslice] \
                    += _ci[mslice + (slice(None), slice(None))]

        #gslice = self.map_global_slice
        #lslice = self.map_local_slice
        #ra_block_len = self.params['ra_block_len']
        #for ii, (_gslice, _lslice) in enumerate(zip(gslice, lslice)):

        #    _dm, _ci = timestream2map(vis, vis_mask, vis_var, time, ra, dec, 
        #            ra_axis[_lslice], dec_axis, beam_size=self.params['beam_fwhm'],
        #            beam_cut = self.params['beam_cut'])

        #    ra_olp_st = _gslice.start - _lslice.start
        #    ra_olp_ed = ra_olp_st + ra_block_len
        #    _mslice = (slice(ra_olp_st, ra_olp_ed), slice(None))
        #    _gslice = (_gslice, slice(None))
        #    _lslice = (_lslice, slice(None))

        #    self.df['dirty_map' ][map_idx + _gslice] += _dm[_mslice]
        #    self.df['cov_inv'][map_idx + _gslice + _lslice]\
        #            += _ci[_mslice + (slice(None), slice(None))]

        #    del _ci, _dm
        #    gc.collect()

    def finish(self):

        if mpiutil.rank0:
            print 'Finishing MapMaking.'

        mpiutil.barrier()

def timestream2map(vis_one, vis_mask, vis_var, time, ra, dec, ra_axis, dec_axis, 
        beam_size=3./60.,  beam_cut = 0.01):

    map_shp = ra_axis.shape + dec_axis.shape

    cov_inv_block = np.zeros((np.product(map_shp), np.product(map_shp)),)
    beam_sig = beam_size  / (2. * np.sqrt(2.*np.log(2.)))

    vis_mask = (vis_mask.copy()).astype('bool')
    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.
    
    _good  = ( ra  < max(ra_axis))
    _good *= ( ra  > min(ra_axis))
    _good *= ( dec < max(dec_axis))
    _good *= ( dec > min(dec_axis))
    _good *= ~vis_mask
    if np.sum(_good) == 0: return None, None
    #if np.sum(_good) < 5: 
    #    #print 'bad block < 5'
    #    cov_inv_block.shape = map_shp * 2
    #    return np.zeros(map_shp), cov_inv_block

    ra   = ra[_good] * np.pi / 180.
    dec  = dec[_good]* np.pi / 180.
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time = time[_good]
    vis_var = vis_var[_good]

    ra_centr  = ra_axis  * np.pi / 180.
    dec_centr = dec_axis * np.pi / 180.

    P = (np.sin(ra[:, None]) * np.sin(ra_centr[None, :]))[:, :, None]\
      + (np.cos(ra[:, None]) * np.cos(ra_centr[None, :]))[:, :, None]\
      * (np.cos(dec[:, None] - dec_centr[None, :]))[:, None, :]

    P = np.arccos(P) * 180. / np.pi
    P = np.exp(- 0.5 * (P / beam_sig) ** 2)

    P.shape = (ra.shape[0], -1)
    if beam_cut is None:
        P_max = np.argmax(P, axis=1)
        P *= 0.
        P[tuple(range(ra.shape[0])), tuple(P_max)] = 1.
    else:
        logger.debug('beam cut %f'%beam_cut)
        P[P<beam_cut] = 0.
        norm = np.sum(P, axis=1) 
        norm[norm == 0] = np.inf
        P /= norm[:, None]


    vis_var[vis_var==0] = np.inf #T_infinity ** 2.
    noise_inv_weight = 1./vis_var

    weight = noise_inv_weight

    dirty_map = np.dot(P.T, vis_one * weight)
    dirty_map.shape = map_shp

    weight = np.eye(vis_one.shape[0]) * weight
    cov_inv_block += np.dot(np.dot(P.T, weight) , P)
    cov_inv_block.shape = map_shp * 2

    return dirty_map, cov_inv_block

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



class MakeMap_Ionly(DirtyMap_GBT):

    def init_ps_datasets(self, ts):

        ts.lin2I()

        func = super(MakeMap_Ionly, self).init_ps_datasets(ts)
        return func

class MakeMap_CombineAll(DirtyMap_GBT):

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
        #dirty_map_tmp = np.zeros((n_freq, ) +  self.map_shp)
        #dirty_map_tmp = al.make_vect(dirty_map_tmp, axis_names=axis_names)
        #dirty_map_tmp.set_axis_info('freq', freq_c, freq_d)
        #dirty_map_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        #dirty_map_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        #self.create_dataset_like('dirty_map',  dirty_map_tmp)
        #self.create_dataset_like('clean_map',  dirty_map_tmp)
        #self.create_dataset_like('noise_diag', dirty_map_tmp)

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

        self.create_dataset('dirty_map',  dirty_map_shp, dirty_map_info, np.float32)
        self.create_dataset('clean_map',  dirty_map_shp, dirty_map_info, np.float32)
        self.create_dataset('noise_diag', dirty_map_shp, dirty_map_info, np.float32)

        self.df['mask'] = np.zeros(n_freq)

        #if self.params['save_cov']:
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
        self.create_dataset('cov_inv', cov_shp, cov_info)
        #cov_tmp = np.zeros((n_freq, ) +  self.map_shp + self.map_shp)
        #cov_tmp = al.make_vect(cov_tmp, axis_names=axis_names)
        #cov_tmp.set_axis_info('freq', freq_c, freq_d)
        #cov_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        #cov_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        #self.create_dataset_like('cov_inv', cov_tmp)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        func = ts.freq_pol_and_bl_data_operate

        return func


def sub_ortho_poly(vis, time, mask, n):
    
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
        polys[ii] /= norm[None, ...]
    
    amp = np.sum(polys * vis[None, ...], 1)
    vis_fit = np.sum(amp[:, None, ...] * polys, 0)
    vis -= vis_fit
    #return vis

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
    polys = np.empty((n,) + shape, dtype=float)
    # For stability, rescale the domain.
    x_range = np.amax(x, axis) - np.amin(x, axis)
    x_mid = (np.amax(x, axis) + np.amin(x, axis)) / 2.
    x = (x - x_mid[upbroad]) / x_range[upbroad] * 2
    # Reshape x to be the final shape.
    x = np.zeros(shape, dtype=float) + x
    # Now loop through the polynomials and construct them.
    # This array will be the starting polynomial, before orthogonalization
    # (only used for earlier versions of scipy).
    if not new_sp:
        basic_poly = np.ones(shape, dtype=float) / np.sqrt(m)
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

