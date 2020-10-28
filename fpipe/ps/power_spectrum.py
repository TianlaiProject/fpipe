"""Module to estimate the power spectrum."""
import numpy as np
import h5py as h5
import gc
import copy
import logging

from caput import mpiutil
from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.utils.path_util import output_path

from fpipe.map import mapbase
from fpipe.map import algebra as al
from fpipe.utils import physical_gridding as gridding
from fpipe.utils import binning

from fpipe.ps import pwrspec_estimator as pse, fgrm

logger = logging.getLogger(__name__)

physical_grid = gridding.physical_grid_lf
#physical_grid = gridding.physical_grid


class PowerSpectrum(OneAndOne, mapbase.MapBase):
    """Module to estimate the power spectrum."""

    params_init = {
            'prefix'   : 'MeerKAT3',
            'kmin'     : 1.e-2,
            'kmax'     : 1.,
            'knum'     : 10,
            'kbin_x'   : None,
            'kbin_y'   : None,
            'logk'     : True,
            'logk_2d'  : True,
            'unitless' : False, 
            'map_key'     : ['clean_map', 'clean_map'],
            'weight_key'  : ['noise_diag', 'noise_diag'],
            'nonorm'      : True, 
            'transfer_func' : None,
            'FKP' : 0,
            'sim' : False,
            'cube_input' : [False, False],
            'freq_mask' : [],
            'prewhite' : True,
            'refinement' : 1.,
            }

    prefix = 'ps_'

    #def __init__(self, *arg, **kwargs):

    #    super(PowerSpectrum, self).__init__(*arg, **kwargs)
    #    mapbase.MapBase.__init__(self)

    def __init__(self, parameter_file_or_dict=None, feedback=0):
        super(PowerSpectrum, self).__init__(parameter_file_or_dict, feedback)
        mapbase.MapBase.__init__(self)
        self.feedback = feedback

    def setup(self):

        super(PowerSpectrum, self).setup()

        input_files = self.input_files
        input_files_num = len(input_files)
        self.input_files_num = input_files_num

        self.init_kbins()

        self.init_task_list()

        self.init_output()

    def init_output(self):

        output_file = self.output_files[0]
        output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
        self.allocate_output(output_file, 'w')
        # load one file to get the ant_n and pol_n
        self.create_dataset('binavg_1d', self.dset_shp + (self.knum,))
        self.create_dataset('counts_1d', self.dset_shp + (self.knum,))

        self.create_dataset('binavg_2d', self.dset_shp + (self.knum_x, self.knum_y))
        self.create_dataset('counts_2d', self.dset_shp + (self.knum_x, self.knum_y))

        self.df['kbin'] = self.kbin
        self.df['kbin_x'] = self.kbin_x
        self.df['kbin_y'] = self.kbin_y
        self.df['kbin_edges'] = self.kbin_edges
        self.df['kbin_x_edges'] = self.kbin_x_edges
        self.df['kbin_y_edges'] = self.kbin_y_edges
        self.df['sim'] = self.params['sim']

    def init_kbins(self):

        logk = self.params['logk']
        logk_2d = self.params['logk_2d']
        kmin = self.params['kmin']
        kmax = self.params['kmax']
        knum = self.params['knum']
        if logk:
            kbin = np.logspace(np.log10(kmin), np.log10(kmax), knum)
            #dk = kbin[1] / kbin[0]
            #kbin_edges = kbin / (dk ** 0.5)
            #kbin_edges = np.append(kbin_edges, kbin_edges[-1]*dk)
        else:
            kbin = np.linspace(kmin, kmax, knum)
            #dk = kbin[1] - kbin[0]
            #kbin_edges = kbin - (dk * 0.5)
            #kbin_edges = np.append(kbin_edges, kbin_edges[-1] + dk)
        kbin_edges = binning.find_edges(kbin, logk=logk)

        self.knum = knum
        self.kbin = kbin
        self.kbin_edges = kbin_edges

        kbin_x = self.params['kbin_x']
        if kbin_x is None:
            self.kbin_x = kbin 
            self.kbin_x_edges = kbin_edges 
        else:
            self.kbin_x = kbin_x
            self.kbin_x_edges = binning.find_edges(kbin_x, logk=logk_2d)
        self.knum_x = len(self.kbin_x)

        kbin_y = self.params['kbin_y']
        if kbin_y is None:
            self.kbin_y = kbin 
            self.kbin_y_edges = kbin_edges 
        else:
            self.kbin_y = kbin_y
            self.kbin_y_edges = binning.find_edges(kbin_y, logk=logk_2d)
        self.knum_y = len(self.kbin_y)



    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        task_list = []
        for ii in range(self.input_files_num):
            for jj in range(ant_n):
                for kk in range(pol_n):
                    tind_l = (ii, jj, kk)
                    tind_r = tind_l
                    tind_o = tind_l
                    task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (self.input_files_num, ant_n, pol_n)

    def read_input(self):

        input = []
        for ii in range(self.input_files_num):
            input.append(h5.File(self.input_files[ii], 'r', 
                                 driver='mpio', comm=mpiutil._comm))
            #input = al.make_vect(al.load_h5(f, 'clean_map'))

        return input

    def iterpstasks(self, input):

        refinement = self.params['refinement']

        task_list = self.task_list
        for task_ind in mpiutil.mpirange(len(task_list)):
            tind_l, tind_r, tind_o = task_list[task_ind]
            tind_l = tuple(tind_l)
            tind_r = tuple(tind_r)
            tind_o = tuple(tind_o)
            msg = ("RANK %03d est. ps.(" + "%03d,"*len(tind_l) + ") x ("\
                    + "%03d,"*len(tind_r) + ")")%((mpiutil.rank, ) + tind_l + tind_r)
            logger.info(msg)

            cube = []
            cube_w = []
            tind_list = [tind_l, tind_r]
            for i in range(2):
                tind = tind_list[i]

                map_key = self.params['map_key'][i]
                input_map = input[tind[0]][map_key][tind[1:] + (slice(None), )]
                input_map_mask  = ~np.isfinite(input_map)
                if (map_key is not 'delta') and (len(self.params['freq_mask']) != 0):
                    # ignore freqency mask for optical data
                    logger.info('apply freq_mask')
                    input_map_mask[self.params['freq_mask'], ...] = True
                #input_map_mask += input_map == 0.
                input_map[input_map_mask] = 0.
                if self.params['prewhite']:
                    input_map_mask = input_map == 0.
                    _mean = np.ma.mean(np.ma.masked_equal(input_map, 0), axis=(1, 2))
                    input_map -= _mean[:, None, None]
                    input_map[input_map_mask] = 0.
                input_map = al.make_vect(input_map, axis_names = ['freq', 'ra', 'dec'])
                for key in input_map.info['axes']:
                    input_map.set_axis_info(key,
                                            self.map_info[key+'_centre'],
                                            self.map_info[key+'_delta'])

                weight_key = self.params['weight_key'][i]
                if weight_key is not None:
                    weight = input[tind[0]][weight_key][tind[1:] + (slice(None), )]
                    weight[input_map_mask] = 0.
                    if weight_key == 'noise_diag':
                        weight = fgrm.make_noise_factorizable(weight)
                    if weight_key == 'separable':
                        logger.debug('apply FKP weight')
                        weight = weight / (1. + weight * self.params['FKP'])
                    weight = al.make_vect(weight, axis_names = ['freq', 'ra', 'dec'])
                    weight.info = input_map.info

                if not self.params['cube_input'][i]:
                    c, c_info = physical_grid(input_map, refinement=refinement,order=0)
                else:
                    logger.debug('cube input')
                    c = input_map
                    c_info = None

                cube.append(c)

                if weight_key is not None:
                    if not self.params['cube_input'][i]:
                        cw, cw_info = physical_grid(weight, refinement=refinement,order=0)
                    else:
                        cw = weight
                        cw_info = None
                    #cw[c==0] = 0.
                    cube_w.append(cw)
                    del weight
                else:
                    cw = al.ones_like(c)
                    cw[c==0] = 0.
                    cube_w.append(cw)

                del c, c_info, cw, input_map

                if tind_l == tind_r:
                    cube.append(cube[0])
                    cube_w.append(cube_w[0])
                    break
            yield tind_o, cube, cube_w

    def load_transfer_func(self, transfer_func_path):

        if transfer_func_path is not None:
            with h5.File(transfer_func_path, 'r') as f:
                tf = al.make_vec(al.load_h5(f, 'ps3d'))
                tf[tf==0] = np.inf
                tf = 1./tf
        else:
            tf = None

        return tf

    def process(self, input):

        tf = self.load_transfer_func(self.params['transfer_func'])

        for tind_o, cube, cube_w in self.iterpstasks(input):

            ps2d, ps1d = pse.calculate_xspec(
                    cube[0], cube[1], cube_w[0], cube_w[1],
                    window= 'blackman', #None, 
                    #window='hamming', #None, 
                    #window='hanning',
                    #window='kaiser',
                    bins=self.kbin_edges, bins_x = self.kbin_x_edges, 
                    bins_y = self.kbin_y_edges,
                    logbins = self.params['logk'],
                    logbins_2d = self.params['logk_2d'],
                    unitless=self.params['unitless'],
                    nonorm = self.params['nonorm'],
                    feedback = self.feedback,
                    transfer_func = tf, )

            self.df['binavg_1d'][tind_o + (slice(None), )] = ps1d['binavg']
            self.df['counts_1d'][tind_o + (slice(None), )] = ps1d['counts_histo']

            self.df['binavg_2d'][tind_o + (slice(None), )] = ps2d['binavg']
            self.df['counts_2d'][tind_o + (slice(None), )] = ps2d['counts_histo']

            del ps2d, ps1d, cube, cube_w
            gc.collect()

        for ii in range(self.input_files_num):
            input[ii].close()


    def finish(self):
        #if mpiutil.rank0:
        logger.info('RANK %03d Finishing Ps.'%(mpiutil.rank))

        mpiutil.barrier()
        self.df.close()

class AutoPS_CubeFile(PowerSpectrum):

    prefix = 'apscube_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            #ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        aps_num = map_tmp.shape[0]

        task_list = []
        for ii in range(aps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (0, ii)
            tind_r = tind_l
            tind_o = (ii, )
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (aps_num, )

class AutoPS_OneByOne(PowerSpectrum):

    prefix = 'aps1b1_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            #ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        aps_num = self.input_files_num

        task_list = []
        for ii in range(aps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (ii, )
            tind_r = tind_l
            tind_o = tind_l
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (aps_num, )

class CrossPS_OneByOne(PowerSpectrum):

    '''
    Est. the Cross power spectrum between input_files and input_files2

    input_files[0][indx] x input_files2[0][indx]
    input_files[1][indx] x input_files2[1][indx]
    ...
    input_files[N][indx] x input_files2[N][indx]

    '''

    params_init = {
            'input_files2' : [],
            }

    prefix = 'xps1b1_'

    def __init__(self, parameter_file_or_dict=None, feedback=0):

        super(CrossPS_OneByOne, self).__init__(parameter_file_or_dict, feedback)

        input_files  = self.params['input_files']
        input_files2 = self.params['input_files2']
        if len(input_files) != len(input_files2):
            msg = "input_files and intput_files2 should have the same length."
            raise ValueError(msg)
        self.params['input_files'] = input_files + input_files2
        super(CrossPS_OneByOne, self)._init_input_files()


    def load_transfer_func(self, transfer_func_path):

        if transfer_func_path is not None:
            with h5.File(transfer_func_path, 'r') as f:
                tf = f['ps3d'][:]
                tf[tf==0] = np.inf
                tf = 1./tf
                tf = tf ** 0.5
        else:
            tf = None

        return tf


    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        xps_num = self.input_files_num / 2

        task_list = []
        for ii in range(xps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (ii,           )
            tind_r = (ii + xps_num, )
            tind_o = tind_l
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (xps_num, )

class AutoPS_Opt(PowerSpectrum):

    params_init = {
            'map_key'     : ['delta',     'delta'],
            'weight_key'  : ['separable', 'separable'],
            }

    prefix = 'apsopt_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'delta'))
            self.map_info = map_tmp.info

        xps_num = self.input_files_num

        task_list = []
        for ii in range(xps_num):
            tind_l = (ii, )
            tind_r = (ii,)
            tind_o = (ii,)
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (xps_num, )




