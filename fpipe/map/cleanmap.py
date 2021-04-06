"""Module to do the clean map"""

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
#from scipy import linalg
from numpy import linalg
import h5py
import sys
import gc

logger = logging.getLogger(__name__)

from meerKAT_utils.constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from meerKAT_utils.constants import f_medium, f_large

__dtype__ = 'float32'


class CleanMap(OneAndOne, mapbase.MultiMapBase):

    params_init = {

            'save_cov' : False,
            'diag_cov' : True,
            'threshold' : 1.e-3,
            'healpix' : False,
            }

    prefix = 'cm_'

    def __init__(self, *args, **kwargs):

        super(CleanMap, self).__init__(*args, **kwargs)

        #mapbase.MultiMapBase.__init__(self)

    def read_input(self):

        for input_file in self.input_files:

            if mpiutil.rank0:
                logger.info('%s'%input_file)
            self.open(input_file)

        map_tmp = al.load_h5(self.df_in[0], 'dirty_map')
        self.map_tmp = al.make_vect(map_tmp, axis_names = map_tmp.info['axes'])
        self.map_shp = self.map_tmp.shape
        for output_file in self.output_files:
            output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
            self.allocate_output(output_file, 'w')
            self.create_dataset_like(-1, 'clean_map',  self.map_tmp)
            self.create_dataset_like(-1, 'noise_diag', self.map_tmp)
            self.create_dataset_like(-1, 'dirty_map',  self.map_tmp)
            if self.params['healpix']:
                self.df_out[-1]['map_pix'] = self.df_in[0]['map_pix'][:]
                self.df_out[-1]['nside']   = self.df_in[0]['nside'][()]

        return 1

    def process(self, input):

        def _indx_f(x, shp): 
            if x >= np.prod(shp): return 
            _i = [int(x / np.prod(shp[1:])), ]
            for i in range(1, len(shp)): 
                x -= _i[i-1] * np.prod(shp[i:])
                _i += [int(x / np.prod(shp[i+1:])),]
            return tuple(_i)

        diag_cov  = self.params['diag_cov']
        threshold = self.params['threshold']
        if self.params['healpix']:
            task_n = np.prod(self.map_shp[:-1])
        else:
            task_n = np.prod(self.map_shp[:-2])

        for task_ind in mpiutil.mpirange(task_n):


            if self.params['healpix']:
                map_shp = self.map_shp[-1:]
                indx = _indx_f(task_ind, self.map_shp[:-1])
            else:
                map_shp = self.map_shp[-2:]
                indx = _indx_f(task_ind, self.map_shp[:-2])

            #print mpiutil.rank,  indx
            print "RANK%03d: ("%mpiutil.rank + ("%04d, "*len(indx))%indx + ")"

            _dirty_map = np.zeros(map_shp, dtype=__dtype__)
            if diag_cov:
                _cov_inv = np.zeros(map_shp, dtype=__dtype__)
            else:
                _cov_inv = np.zeros(map_shp * 2, dtype=__dtype__)
            for ii, df in enumerate(self.df_in):
                _dirty_map += df['dirty_map'][indx + (slice(None), )]
                self.read_block_from_dset(ii, 'cov_inv', indx, _cov_inv)
                #_cov_inv   += df['cov_inv'][indx + (slice(None), )]

            self.df_out[-1]['dirty_map' ][indx + (slice(None), )] = _dirty_map
            clean_map, noise_diag = make_cleanmap(_dirty_map,_cov_inv,diag_cov,threshold)
            self.df_out[-1]['clean_map' ][indx + (slice(None), )] = clean_map
            self.df_out[-1]['noise_diag'][indx + (slice(None), )] = noise_diag
            del _cov_inv
            gc.collect()

        mpiutil.barrier()

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing CleanMapMaking.'


        mpiutil.barrier()
        super(CleanMap, self).finish()

def make_cleanmap(dirty_map, cov_inv_block, diag_cov=False, threshold=1.e-5):

    map_shp = dirty_map.shape
    dirty_map.shape = (np.prod(map_shp), )
    if diag_cov:
        cov_inv_block.shape = (np.prod(map_shp), )
        cov_inv_block[cov_inv_block<threshold] = np.inf
        noise_diag = 1./cov_inv_block
        clean_map = dirty_map.copy() / cov_inv_block
    else:
        cov_inv_block.shape = (np.prod(map_shp), np.prod(map_shp))

        cov_inv_diag = np.diag(cov_inv_block).copy()
        cov_inv_bad = cov_inv_diag == 0
        cov_inv_diag_max = cov_inv_diag.max()
        if np.all(cov_inv_diag == 0):
            logger.error('Singular Noise Matrix, ignore')
            noise = np.zeros_like(cov_inv_block)
        else:
            cov_inv_diag_min = cov_inv_diag[cov_inv_diag!=0].min()
            logger.info('cov inv diag max %e, min %e'%(cov_inv_diag_max, cov_inv_diag_min))
            #cov_inv_diag[cov_inv_diag!=0] = cov_inv_diag_max * 0.1
            cov_inv_diag[:] = cov_inv_diag_max * threshold
            #cov_inv_diag[:] = threshold

            cov_inv_block += np.eye(np.prod(map_shp)) * cov_inv_diag[:, None]
            #noise = linalg.pinv(cov_inv_block, rcond=threshold)
            noise = linalg.inv(cov_inv_block)
            noise[cov_inv_bad] = 0.

        clean_map = np.dot(noise, dirty_map)
        noise_diag = np.diag(noise)

        del noise
        gc.collect()

    clean_map.shape = map_shp
    noise_diag.shape = map_shp


    return clean_map, noise_diag

def make_cleanmap_old(dirty_map, cov_inv_block, threshold=1.e-5):

    map_shp = dirty_map.shape
    dirty_map.shape = (np.prod(map_shp), )

    cov_inv_block.shape = (np.prod(map_shp), np.prod(map_shp))

    #cov_inv_block[cov_inv_block<1.e-5] = 0.

    #cov_inv_diag = np.diag(cov_inv_block).copy()
    #cov_inv_bad = cov_inv_diag == 0
    #cov_inv_diag[cov_inv_bad] += 1./threshold
    #cov_inv_diag[~cov_inv_bad] = 0
    #cov_inv_diag += 1./threshold
    #cov_inv_block += np.eye(np.prod(map_shp)) * cov_inv_diag
    #noise = linalg.inv(cov_inv_block)

    cov_inv_diag = np.diag(cov_inv_block).copy()
    cov_inv_bad = cov_inv_diag == 0
    cov_inv_diag_max = cov_inv_diag.max()
    if np.all(cov_inv_diag == 0):
        logger.error('Singular Noise Matrix, ignore')
        noise = np.zeros_like(cov_inv_block)
    else:
        cov_inv_diag_min = cov_inv_diag[cov_inv_diag!=0].min()
        logger.info('cov inv diag max %e, min %e'%(cov_inv_diag_max, cov_inv_diag_min))
        #cov_inv_diag[cov_inv_diag!=0] = cov_inv_diag_max * 0.1
        #cov_inv_diag[:] = cov_inv_diag_max * threshold
        cov_inv_diag[:] = threshold

        cov_inv_block += np.eye(np.prod(map_shp)) * cov_inv_diag[:, None]
        #noise = linalg.pinv(cov_inv_block, rcond=threshold)
        noise = linalg.inv(cov_inv_block)
        noise[cov_inv_bad] = 0.

    clean_map = np.dot(noise, dirty_map)
    noise_diag = np.diag(noise)

    clean_map.shape = map_shp
    noise_diag.shape = map_shp

    del noise
    gc.collect()

    return clean_map, noise_diag

class CleanMap_SplitRA(OneAndOne, mapbase.MultiMapBase):

    params_init = {

            'save_cov' : False,
            'threshold' : 1.e-3,
            'block_length': 80, 
            'block_overlap': 10,
            }

    prefix = 'cmsplitra_'

    def __init__(self, *args, **kwargs):

        super(CleanMap_SplitRA, self).__init__(*args, **kwargs)
        mapbase.MultiMapBase.__init__(self)

    def read_input(self):

        for input_file in self.input_files:

            print input_file
            self.open(input_file)

        self.map_tmp = al.make_vect(al.load_h5(self.df_in[0], 'dirty_map'))
        self.map_shp = self.map_tmp.shape
        for output_file in self.output_files:
            output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
            self.allocate_output(output_file, 'w')
            self.create_dataset_like(-1, 'clean_map',  self.map_tmp)
            self.create_dataset_like(-1, 'noise_diag', self.map_tmp)

        return 1

    def process(self, input):

        def _indx_f(x, shp): 
            if x >= np.prod(shp): return 
            _i = [int(x / np.prod(shp[1:])), ]
            for i in range(1, len(shp)): 
                x -= _i[i-1] * np.prod(shp[i:])
                _i += [int(x / np.prod(shp[i+1:])),]
            return tuple(_i)

        threshold = self.params['threshold']

        loop_n = np.prod(self.map_shp[:-2])

        block_length = self.params['block_length']
        block_olap = self.params['block_overlap']
        ra_length_tot = self.map_shp[-2]
        dec_length_tot = self.map_shp[-1]
        task_n = int( ra_length_tot / block_length) + 1
        if mpiutil.rank0:
            logger.debug('RA split into %4d task with block length %4d'%(
                task_n, block_length))


        for task_ind in mpiutil.mpirange(task_n):

            ra_st = task_ind * block_length
            ra_ed = min((task_ind + 1) * block_length, ra_length_tot)

            ra_st_olap     = max(ra_st - block_olap, 0)
            ra_ed_olap     = min(ra_ed + block_olap, ra_length_tot)
            ra_length      = ra_ed      - ra_st
            ra_length_olap = ra_ed_olap - ra_st_olap
            olap_lower     = ra_st - ra_st_olap

            logger.debug('RANK %03d: RA %4d - %4d'%(mpiutil.rank, ra_st, ra_ed))
            logger.debug('RANK %03d: RA olap %4d - %4d'%(mpiutil.rank, ra_st_olap, ra_ed_olap))

            radec_slice = (slice(ra_st, ra_ed), slice(None))
            radec_slice_olap = (slice(ra_st_olap, ra_ed_olap), slice(None))

            for loop_ind in xrange(loop_n):

                indx = _indx_f(loop_ind, self.map_shp[:-2])
                logger.debug('RANK %03d: Loop idx '%mpiutil.rank\
                        + '%3d'*len(indx)%indx)

                map_shp = (ra_length_olap, dec_length_tot)
                _dirty_map = np.zeros(map_shp)
                _cov_inv = np.zeros(map_shp * 2, dtype=float)
                for df in self.df_in:
                    _dirty_map += df['dirty_map'][indx + radec_slice_olap]
                    _cov_inv   += df['cov_inv'][indx + radec_slice_olap * 2]

                clean_map, noise_diag = make_cleanmap(_dirty_map, _cov_inv, threshold)
                clean_map  =  clean_map[ olap_lower : olap_lower+ra_length ]
                noise_diag = noise_diag[ olap_lower : olap_lower+ra_length ]
                self.df_out[-1]['clean_map' ][indx + radec_slice] = clean_map
                self.df_out[-1]['noise_diag'][indx + radec_slice] = noise_diag

        mpiutil.barrier()

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing CleanMapMaking.'


        mpiutil.barrier()
        super(CleanMap_SplitRA, self).finish()
