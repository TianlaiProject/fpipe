from fpipe.map import dirtymap, cleanmap
from fpipe.map import algebra as al
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from numpy.linalg import multi_dot
import gc


import healpy as hp
import numpy as np


import logging

__dtype__ = 'float32'
logger = logging.getLogger(__name__)


class DirtyMap_healpix(dirtymap.DirtyMap):
    
    params_init = {
        'nside' : 1024,
        'ra_range' : [130, 135],
        'dec_range' : [26.0, 27.0],
        'center_only' : False, 
    }
    
    prefix = 'dmh_'
    
    def setup(self):

        params = self.params
        
        self.nside = self.params['nside']
        self.ra_range = self.params['ra_range']
        self.dec_range = self.params['dec_range']

        ra_min  = np.min(self.ra_range)
        ra_max  = np.max(self.ra_range)
        dec_min = np.min(self.dec_range)
        dec_max = np.max(self.dec_range)

        v1 = hp.ang2vec(np.pi/2 - np.radians(dec_min), np.radians(ra_min))
        v2 = hp.ang2vec(np.pi/2 - np.radians(dec_min), np.radians(ra_max))
        v3 = hp.ang2vec(np.pi/2 - np.radians(dec_max), np.radians(ra_max))
        v4 = hp.ang2vec(np.pi/2 - np.radians(dec_max), np.radians(ra_min))
        self.map_pix = hp.query_polygon(self.nside, np.array([v1, v2, v3, v4]))

        ra_centr, dec_centr = hp.pix2ang(self.nside, self.map_pix, lonlat=True)
        self.ra_centr  = ra_centr  * np.pi / 180.
        self.dec_centr = dec_centr * np.pi / 180.

        self.map_shp = (len(self.map_pix),)
        #axis_names = ('pix', )
        #map_tmp = np.zeros(self.map_shp, dtype=__dtype__)
        #map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        #map_tmp.set_axis_info('pix', self.map_shp[0]//2, 1)
        #self.map_tmp = map_tmp
    
        
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
        
        pix_spacing = 1.
        
        axis_names = ('freq', 'pix')
        msg = 'init dirty map dsets'

        logger.debug(msg)
        dirty_map_shp  = (n_freq, ) +  self.map_shp
        dirty_map_info = {
                #'pix_delta'   : 1.,
                #'pix_centre'  : self.map_shp[0]//2,
                'freq_delta'  : freq_d,
                'freq_centre' : freq_c,
                'axes'        : axis_names,
                }
        self.map_axis_names = axis_names
        
        self.create_dataset('dirty_map',  dirty_map_shp, dirty_map_info, __dtype__)
        self.create_dataset('clean_map',  dirty_map_shp, dirty_map_info, __dtype__)
        self.create_dataset('noise_diag', dirty_map_shp, dirty_map_info, __dtype__)

        self.df['mask'] = np.zeros(n_freq)
        
        msg = 'init cov dsets'
        logger.debug(msg)
        if self.params['diag_cov']:
            axis_names = ('freq', 'pix')
            cov_shp = (n_freq, ) +  self.map_shp
        else:
            axis_names = ('freq', 'pix', 'pix')
            cov_shp = (n_freq, ) +  self.map_shp + self.map_shp
        cov_info = {
                #'pix_delta'   : 1.,
                #'pix_centre'  : self.map_shp[0]//2,
                'freq_delta'  : freq_d,
                'freq_centre' : freq_c,
                'axes'        : axis_names,
                }
        self.create_dataset('cov_inv', cov_shp, cov_info, __dtype__)

        self.df['map_pix'] = self.map_pix
        self.df['pol'] = self.pol
        self.df['bl']  = self.bl
        self.df['nside'] = self.nside

        #func = ts.freq_pol_and_bl_data_operate
        func = ts.freq_data_operate

        msg = 'RANK %03d: everything init done'%mpiutil.rank
        logger.debug(msg)
        mpiutil.barrier()
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
        #tblock_len = 200
        n_time, n_pol, n_bl = vis.shape
        tblock_len = n_time
        ff = gi[0]
        vis_idx = ((bb, pp, ff) for bb in range(n_bl) for pp in range(n_pol))
        vis_axis_names = ('bl', 'pol', 'freq')
        map_axis_names = self.map_axis_names

        #ra_axis  = self.map_tmp.get_axis('ra')
        #dec_axis = self.map_tmp.get_axis('dec')
        pix_axis = self.map_pix
        ra_centr = self.ra_centr
        dec_centr = self.dec_centr

        map_shp = self.map_shp
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

            _vis_mask = vis_mask[:, p_idx, b_idx]
            
            if np.all(_vis_mask):
                print " VIS (" + ("%03d, "*len(_vis_idx))%_vis_idx + ")" +\
                        " All masked, continue"
                #self.df['mask'][map_idx[:-1]] = 1
                del _vis_mask
                continue

            _vis = vis[:, p_idx, b_idx]

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
                               pix_axis, 
                               #ra_centr, dec_centr,
                               self.nside,
                               self.ra_range, self.dec_range,
                               _ci, _dm,
                               diag_cov = self.params['diag_cov'],
                               beam_size= beam_fwhm,
                               #beam_cut = self.params['beam_cut'],
                               center_only = self.params['center_only'],
                               )
            del _vis, _vis_mask
            gc.collect()

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
        
def timestream2map(vis_one, vis_mask, vis_var, time, ra, dec, pix_axis,
        nside, ra_range, dec_range, cov_inv_block, dirty_map, diag_cov=False,
        beam_size=3./60.,  center_only=False): 
    
    map_shp = pix_axis.shape
    
    beam_sig = beam_size  / (2. * np.sqrt(2.*np.log(2.)))

    vis_mask = (vis_mask.copy()).astype('bool')
    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.

    _good  = ( ra  < max(ra_range))
    _good *= ( ra  > min(ra_range))
    _good *= ( dec < max(dec_range))
    _good *= ( dec > min(dec_range))
    _good *= ~vis_mask
    if np.sum(_good) == 0: return

    ra       = ra[_good]  #* np.pi / 180.
    dec      = dec[_good] #* np.pi / 180.
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time     = time[_good]
    vis_var  = vis_var[_good]
    
    logger.debug('est. pointing')
    pix = hp.ang2pix(nside, ra, dec, nest=False, lonlat=True)
    
    if center_only:
        P  = (pix_axis - pix[:, None])
        on = P==0
        P[on]  = 1.
        P[~on] = 0.
    else:
        pix_size = hp.nside2resol(nside, arcmin=True) / 60.
        pix = [pix[None, :], hp.get_all_neighbours(nside, ra, dec, lonlat=True)]
        pix = np.concatenate(pix, axis=0)
        pix_ra, pix_dec = hp.pix2ang(nside, pix, lonlat=True)

        # GBT weighting
        #w = np.sqrt((pix_ra - ra[None, :]) ** 2 + (pix_dec - dec[None, :]) ** 2)
        #w = 1. - w / pix_size
        #w[w<0] = 0.
        #w = w ** 2.
        #w = w / np.sum(w, axis=0)[None, :]

        w = np.sqrt((pix_ra - ra[None, :]) ** 2 + (pix_dec - dec[None, :]) ** 2)
        w = np.exp(-0.5 * (w ** 2) / beam_sig**2 )
        w[w<0.001] = 0.
        w = w / np.max(w, axis=0)[None, :]

        P = np.zeros(ra.shape + pix_axis.shape, dtype=__dtype__ )
        for ii in range(9):
            on = (pix_axis - pix[ii][:, None]) == 0
            P[on] = w[ii][np.any(on, axis=1)]
        del pix, w, pix_ra, pix_dec, on


    vis_var[vis_var==0] = np.inf #T_infinity ** 2.
    noise_inv_weight = 1. /vis_var

    weight = noise_inv_weight

    logger.debug('est. dirty map')
    dirty_map += np.dot(P.T, vis_one * weight)

    logger.debug('est. noise inv')
    if diag_cov:
        #cov_inv_block += np.diag(multi_dot([P.T, weight, P]))
        cov_inv_block += np.dot(P.T**2., weight)
    else:
        weight = np.eye(vis_one.shape[0]) * weight
        cov_inv_block += multi_dot([P.T, weight, P])

    del weight, P
    gc.collect()

def timestream2map_deconv(vis_one, vis_mask, vis_var, time, ra, dec,
        ra_centr, dec_centr, nside, ra_range, dec_range, cov_inv_block, dirty_map, 
        diag_cov=False, beam_size=3./60.,  center_only=False): 
    
    map_shp = ra_centr.shape
    
    beam_sig = beam_size  / (2. * np.sqrt(2.*np.log(2.)))

    vis_mask = (vis_mask.copy()).astype('bool')
    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.

    _good  = ( ra  < max(ra_range))
    _good *= ( ra  > min(ra_range))
    _good *= ( dec < max(dec_range))
    _good *= ( dec > min(dec_range))
    _good *= ~vis_mask
    if np.sum(_good) == 0: return

    ra       = ra[_good]  #* np.pi / 180.
    dec      = dec[_good] #* np.pi / 180.
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time     = time[_good]
    vis_var  = vis_var[_good]
    
    logger.debug('est. pointing')

    ra  = ra  * np.pi / 180.
    dec = dec * np.pi / 180.

    P = (np.sin(ra[:, None]) * np.sin(ra_centr[None, :]))\
      + (np.cos(ra[:, None]) * np.cos(ra_centr[None, :]))\
      * (np.cos(dec[:, None] - dec_centr[None, :]))

    P  = np.arccos(P) * 180. / np.pi
    P  = np.exp(- 0.5 * (P / beam_sig) ** 2)
    

    P.shape = (ra.shape[0], -1)

    P[P < 0.1] *= 0.
    P_norm = np.sum(P, axis=1)
    #P_norm = np.max(P, axis=1)
    P_norm[P_norm==0] = np.inf
    P /= P_norm[:, None]

    vis_var[vis_var==0] = np.inf #T_infinity ** 2.
    noise_inv_weight = 1. /vis_var

    weight = noise_inv_weight

    logger.debug('est. dirty map')
    dirty_map += np.dot(P.T, vis_one * weight)

    logger.debug('est. noise inv')
    if diag_cov:
        #cov_inv_block += np.diag(multi_dot([P.T, weight, P]))
        cov_inv_block += np.dot(P.T**2, weight)
    else:
        weight = np.eye(vis_one.shape[0]) * weight
        cov_inv_block += multi_dot([P.T, weight, P])

    del weight, P
    gc.collect()

