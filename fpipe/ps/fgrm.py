"""Module to remove fg with SVD"""
import logging
import numpy as np
import h5py as h5
import gc, os
import copy
import numpy.ma as ma

from caput import mpiutil
from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path

from fpipe.map import mapbase
from fpipe.map import algebra as al
from fpipe.sim import beam
from fpipe.ps import find_modes

logger = logging.getLogger(__name__)


class FGRM_SVD(pipeline.OneAndOne, mapbase.MultiMapBase):
    """Module to estimate the power spectrum."""

    params_init = {
            'mode_list': [0, 1],
            'output_combined' : None,
            'freq_mask' : [],
            'weight_prior' : 1.e3,
            'map_key' : 'clean_map',
            'weight_key' : 'noise_diag',
            'svd_path' : None,
            'svd_key'  : None,
            'conv_factor' : 0,
            'fwhm1400' : 0.9,
            'beam_file': None,
            'add_map' : None,
            }

    prefix = 'fg_'

    def __init__(self, *arg, **kwargs):

        super(FGRM_SVD, self).__init__(*arg, **kwargs)
        mapbase.MultiMapBase.__init__(self)

    def setup(self):

        super(FGRM_SVD, self).setup()

        input_files = self.input_files
        input_files_num = len(input_files)
        self.input_files_num = input_files_num
        self.svd_info  = None
        self.mode_list = None

        self.init_output()
        self.init_svd_info()
        self.init_task_list()


    def init_output(self):

        for output_file in self.output_files:
            output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
            self.allocate_output(output_file, 'w')
            #self.create_dataset('binavg_1d', self.dset_shp + (self.knum,))

            #self.df_out[-1]['mode_list'] = self.mode_list

    def init_svd_info(self):

        svd_path = self.params['svd_path']
        svd_key  = self.params['svd_key']
        if svd_path is not None:
            if svd_key is None:
                logger.error('Need to specify svd key')
            svd_info = []
            with h5.File(svd_path[0], 'r') as f:
                logger.info('Load sigvalu from %s'%svd_key[0])
                svd_info.append(f[svd_key[0] + '_sigvalu'][:])
                logger.info('Load left  sigvect from %s'%svd_key[0])
                svd_info.append(f[svd_key[0] + '_sigvect'][:])
                mode_list = f['mode_list'][:]
            with h5.File(svd_path[1], 'r') as f:
                logger.info('Load right sigvect from %s'%svd_key[1])
                svd_info.append(f[svd_key[1] + '_sigvect'][:])
            self.svd_info = svd_info
            self.mode_list = mode_list

    def combine_results(self):

        output_combined = self.params['output_combined']
        output_combined  = output_path(output_combined, 
                relative= not output_combined.startswith('/'))
        self.allocate_output(output_combined, 'w')
        self.df_out[-1]['mode_list'] = self.mode_list

        for _m in self.mode_list:
            mode_key = 'cleaned_%02dmode'%_m
            self.create_dataset(-1, mode_key, dset_shp = self.dset_shp,
                                dset_info = self.map_info)
            self.create_dataset(-1, mode_key + '_weight', dset_shp = self.dset_shp, 
                                dset_info = self.map_info)
            mask = np.ones(self.dset_shp[:1]).astype('bool')
            for ii in range(len(self.output_files)):
                for key in self.df_out[ii][mode_key].keys():
                    self.df_out[-1][mode_key][:]\
                            += self.df_out[ii]['weight'][:]\
                            *  self.df_out[ii]['%s/%s'%(mode_key, key)][:]
                    self.df_out[-1][mode_key + '_weight'][:]\
                            += self.df_out[ii]['weight'][:]
                mask *= ~(self.df_out[ii]['mask'][:].astype('bool'))
            weight = self.df_out[-1][mode_key + '_weight'][:]
            weight[weight==0] = np.inf
            self.df_out[-1][mode_key][:] /= weight
            self.df_out[-1][mode_key + '_mask'] = (~mask).astype('int')

    def init_task_list(self):

        '''
        init task list
        '''

        pass

    def read_input(self):

        input = []
        for ii in range(self.input_files_num):
            input.append(h5.File(self.input_files[ii], 'r', 
                                 driver='mpio', comm=mpiutil._comm))

        return input

    def process(self, input):

        task_list = self.task_list
        for task_ind in mpiutil.mpirange(len(task_list)):
            tind_l, tind_r, tind_o = task_list[task_ind]
            tind_l = tuple(tind_l)
            tind_r = tuple(tind_r)
            tind_o = tind_o
            print ("RANK %03d fgrm.\n(" + "%03d,"*len(tind_l) + ") x ("\
                    + "%03d,"*len(tind_r) + ")\n")%((mpiutil.rank, ) + tind_l + tind_r)

            tind_list = [tind_l, tind_r]
            maps    = []
            weights = []
            freq_good = np.ones(self.dset_shp[0]).astype('bool')
            if len(self.params['freq_mask']) != 0:
                freq_good[self.params['freq_mask']] = False
            for i in range(2):
                tind = tind_list[i]

                map_key = self.params['map_key'] #'clean_map'
                input_map = al.load_h5(input[tind[0]], map_key)
                input_map = al.make_vect(input_map, axis_names = ['freq', 'ra', 'dec'])
                maps.append(input_map)

                weight_key = self.params['weight_key'] #'noise_diag'
                if weight_key is not None:
                    weight = al.load_h5(input[tind[0]], weight_key)
                    if weight_key is 'noise_diag':
                        weight_prior = self.params['weight_prior']
                        logger.info('using wp %e'%weight_prior)
                        weight = make_noise_factorizable(weight, weight_prior)
                else:
                    weight = np.ones_like(input_map)
                    weight[input_map==0] = 0.

                weight = al.make_vect(weight, axis_names = ['freq', 'ra', 'dec'])
                weight.info = input_map.info

                try:
                    freq_good *= ~(input[tind[0]]['mask'][:]).astype('bool')
                except KeyError:
                    logger.info('mask doesn\' exist')
                    pass

                weights.append(weight)

            maps[0][~freq_good] = 0.
            maps[1][~freq_good] = 0.
            weights[0][~freq_good] = 0.
            weights[1][~freq_good] = 0.

            if self.params['conv_factor'] !=0:
                maps, weights = degrade_resolution(maps, weights, 
                        conv_factor=self.params['conv_factor'], mode='constant',
                        beam_file = self.params['beam_file'],
                        fwhm1400  = self.params['fwhm1400'])
            else:
                logger.info('common reso. conv. ignored')

            if self.params['add_map'] is not None:
                _maps = self.params['add_map']
                _map_A_path, _map_A_name = os.path.split(os.path.splitext(_maps[0])[0])
                _map_B_path, _map_B_name = os.path.split(os.path.splitext(_maps[1])[0])
                logger.info('add real map pair (%s %s)'%(_map_A_name, _map_B_name))
                with h5.File(os.path.join(_map_A_path,_map_A_name+'.h5'), 'r') as f:
                    _map0 = al.load_h5(f, 'cleaned_00mode/%s'%_map_B_name)
                    maps[0][:] +=  _map0
                with h5.File(os.path.join(_map_B_path,_map_B_name+'.h5'), 'r') as f:
                    _map1 = al.load_h5(f, 'cleaned_00mode/%s'%_map_A_name)
                    maps[1][:] += _map1

            svd_info = self.svd_info
            if svd_info is None:
                freq_cov, counts = find_modes.freq_covariance(maps[0], maps[1], 
                    weights[0], weights[1], freq_good, freq_good)
                svd_info = find_modes.get_freq_svd_modes(freq_cov, np.sum(freq_good))

            if self.params['add_map'] is not None:
                maps[0][:] -=  _map0
                maps[1][:] -=  _map1


            mode_list = self.mode_list

            mode_list_ed = copy.deepcopy(mode_list)
            mode_list_st = copy.deepcopy(mode_list)
            mode_list_st[1:] = mode_list_st[:-1]

            dset_key = tind_o[0] + '_sigvalu'
            self.df_out[tind_l[0]][dset_key] = svd_info[0]
            dset_key = tind_o[0] + '_sigvect'
            self.df_out[tind_l[0]][dset_key] = svd_info[1]
            self.df_out[tind_l[0]]['weight'][:] = weights[0]
            self.df_out[tind_l[0]]['mask'][:]   = (~freq_good).astype('int')

            if tind_o[1] != tind_o[0]:
                dset_key = tind_o[1] + '_sigvalu'
                self.df_out[tind_r[0]][dset_key] = svd_info[0]
                dset_key = tind_o[1] + '_sigvect'
                self.df_out[tind_r[0]][dset_key] = svd_info[2]
                self.df_out[tind_r[0]]['weight'][:] = weights[1]
                self.df_out[tind_r[0]]['mask'][:]   = (~freq_good).astype('int')

            for (n_modes_st, n_modes_ed) in zip(mode_list_st, mode_list_ed):
                svd_modes = svd_info[1][n_modes_st:n_modes_ed]
                group_name ='cleaned_%02dmode/'%n_modes_ed 
                maps[0], amp = find_modes.subtract_frequency_modes(
                        maps[0], svd_modes, weights[0], freq_good)
                dset_key = group_name + tind_o[0]
                self.df_out[tind_l[0]][dset_key][:] = copy.deepcopy(maps[0])

                if tind_o[0] != tind_o[1]:
                    svd_modes = svd_info[2][n_modes_st:n_modes_ed]
                    maps[1], amp = find_modes.subtract_frequency_modes(
                            maps[1], svd_modes, weights[1], freq_good)
                    dset_key = group_name + tind_o[1]
                    self.df_out[tind_r[0]][dset_key][:] = copy.deepcopy(maps[1])

                # for the case of auto with different svd svd modes
                if 'Combined' in self.df_out[tind_r[0]][group_name].keys():
                    dset_key = group_name + 'Combined'
                    _map = maps[0].copy() * weights[0].copy()\
                         + maps[1].copy() * weights[1].copy()
                    _wet = weights[0].copy() + weights[1].copy() 
                    _wet[_wet==0] = np.inf
                    _map /= _wet
                    self.df_out[tind_r[0]][dset_key][:] = copy.deepcopy(_map)


        if self.params['output_combined'] is not None:
            self.combine_results()

        for ii in range(self.input_files_num):
            input[ii].close()


    def finish(self):
        #if mpiutil.rank0:
        print 'RANK %03d Finishing FGRM'%(mpiutil.rank)

        mpiutil.barrier()
        for df in self.df_out:
            df.close()

class FGRM_SVD_FullCross(FGRM_SVD):

    prefix = 'fgcross_'

    def init_task_list(self):

        '''
        init task list [A1, A2, A3, ... An]
        A1   x A2
        A1   x A3
        ...
        An-1 x An
        '''
        if self.mode_list is None:
            self.mode_list = self.params['mode_list']

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            self.map_info = map_tmp.info

        task_list = []
        for ii in range(self.input_files_num):
            input_file_name_ii = self.input_files[ii].split('/')[-1]
            input_file_name_ii = input_file_name_ii.replace('.h5', '')
            for jj in range(ii + 1, self.input_files_num):
                input_file_name_jj = self.input_files[jj].split('/')[-1]
                input_file_name_jj = input_file_name_jj.replace('.h5', '')
                tind_l = (ii, )
                tind_r = (jj, )
                tind_o = [input_file_name_jj, input_file_name_ii]
                task_list.append([tind_l, tind_r, tind_o])

                for kk in self.mode_list:
                    self.create_dataset(ii, 'cleaned_%02dmode/'%kk + input_file_name_jj, 
                            dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    self.create_dataset(jj, 'cleaned_%02dmode/'%kk + input_file_name_ii, 
                            dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    #self.create_dataset(ii, 'cleaned_%02dmode/Combined'%kk, 
                    #        dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    #self.create_dataset(jj, 'cleaned_%02dmode/Combined'%kk, 
                    #        dset_shp = map_tmp.shape, dset_info = map_tmp.info)

            self.create_dataset(ii, 'weight', dset_shp = map_tmp.shape,
                                dset_info = map_tmp.info)
            self.create_dataset(ii, 'mask', dset_shp = map_tmp.shape[:1])
            self.df_out[ii]['mode_list'] = self.mode_list

        self.task_list = task_list
        self.dset_shp  = map_tmp.shape

class FGRM_SVD_Auto(FGRM_SVD):

    prefix = 'fgauto_'

    def init_task_list(self):

        '''
        init task list [A1, A2, A3, ... An]
        A1   x A1
        A2   x A2
        ...
        An   x An
        '''
        if self.mode_list is None:
            self.mode_list = self.params['mode_list']

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            self.map_info = map_tmp.info

        task_list = []
        for ii in range(self.input_files_num):
            input_file_name_ii = self.input_files[ii].split('/')[-1]
            input_file_name_ii = input_file_name_ii.replace('.h5', '')
            input_file_name_jj = input_file_name_ii
            if self.params['svd_key'] is not None:
                input_file_name_ii = self.params['svd_key'][0]
                input_file_name_jj = self.params['svd_key'][1]
            tind_l = (ii, )
            tind_r = (ii, )

            for kk in self.mode_list:
                if input_file_name_jj != input_file_name_ii:
                    tind_o = [input_file_name_ii, input_file_name_jj]
                    self.create_dataset(ii, 'cleaned_%02dmode/'%kk + input_file_name_ii, 
                        dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    self.create_dataset(ii, 'cleaned_%02dmode/'%kk + input_file_name_jj, 
                        dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    #print 'cleaned_%02dmode/Combined'%kk
                    self.create_dataset(ii, 'cleaned_%02dmode/Combined'%kk, 
                        dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                else:
                    tind_o = ['cmap', 'cmap']
                    self.create_dataset(ii, 'cleaned_%02dmode/cmap'%kk, 
                        dset_shp = map_tmp.shape, dset_info = map_tmp.info)

            task_list.append([tind_l, tind_r, tind_o])

            self.create_dataset(ii, 'weight', dset_shp = map_tmp.shape,
                                dset_info = map_tmp.info)
            self.create_dataset(ii, 'mask', dset_shp = map_tmp.shape[:1])
            self.df_out[ii]['mode_list'] = self.mode_list

        self.task_list = task_list
        self.dset_shp  = map_tmp.shape


def noise_diag_2_weight(weight):

    weight[weight==0] = np.inf
    weight = 1./weight
    
    mask = (weight != 0).astype('int')
    spat_w = np.sum(weight, axis=0)
    norm = np.sum(mask, axis=0) * 1.
    norm[norm==0] = np.inf
    spat_w /= norm
    
    freq_w = np.sum(weight, axis=(1, 2))
    norm = np.sum(mask, axis=(1, 2)) * 1.
    norm[norm==0] = np.inf
    freq_w /= norm
    
    weight = freq_w[:, None, None] * spat_w[None, :, :]
    weight = weight ** 0.5
    cut  = np.percentile(weight, 10)
    cut2 = np.percentile(weight, 80)
    #weight[weight>cut] = cut
    weight[weight<cut] = 0.

    weight[weight>cut2] = cut2

    return weight

def make_noise_factorizable(noise, weight_prior=1.e3):
    r"""Convert noise diag such that the factor into a function a
    frequency times a function of pixel by taking means over the original
    weights.
    
    input noise_diag;
    output weight
    
    weight_prior used to be 10^-30 before prior applied
    """
    print "making the noise factorizable"
    
    #noise[noise < weight_prior] = 1.e-30
    #noise = 1. / noise
    #noise[noise < 5.e-5] = 0.
    if noise[noise!=0].min() > 1./weight_prior:
        logger.warning('Noise Too High, ignore weight_prior %3.2e'%weight_prior)
    else:
        noise[noise > 1./weight_prior] = 1.e30
    noise = ma.array(noise)
    # Get the freqency averaged noise per pixel.  Propagate mask in any
    # frequency to all frequencies.
    for noise_index in range(ma.shape(noise)[0]):
        if np.all(noise[noise_index, ...] > 1.e20):
            noise[noise_index, ...] = ma.masked
    noise_fmean = ma.mean(noise, 0)
    noise_fmean[noise_fmean > 1.e20] = ma.masked
    # Get the pixel averaged noise in each frequency.
    noise[noise > 1.e20] = ma.masked
    noise /= noise_fmean
    noise_pmean = ma.mean(ma.mean(noise, 1), 1)
    # Combine.
    noise = noise_pmean[:, None, None] * noise_fmean[None, :, :]
    #noise[noise == 0] = np.inf
    noise[noise==0] = ma.masked
    weight = (1. / noise)
    weight = weight.filled(0)

    cut_l  = np.percentile(weight, 10)
    cut_h = np.percentile(weight, 80)
    weight[weight<cut_l] = cut_l
    weight[weight>cut_h] = cut_h

    return weight

def degrade_resolution(maps, noises, conv_factor=1.2, mode="constant", 
        beam_file=None, fwhm1400=0.9):
    r"""Convolves the maps down to the lowest resolution.

    Also convolves the noise, making sure to deweight pixels near the edge
    as well.  Converts noise to factorizable form by averaging.

    mode is the ndimage.convolve flag for behavior at the edge
    """
    print "degrading the resolution to a common beam: ", conv_factor
    noise1, noise2 = noises
    map1, map2 = maps

    # Get the beam data.
    if beam_file is not None:
        logger.info('load beam from %s'%beam_file)
        _bd = np.loadtxt(beam_file)
        freq_data = _bd[:, 0]
        beam_data = _bd[:, 1]
    else:
        freq_data = map1.get_axis_edges('freq')
        #freq_data = np.linspace(800., 1600., 100).astype('float')
        beam_data = 1.2 * fwhm1400 * 1400. / freq_data

    beam_diff = np.sqrt(max(conv_factor * beam_data) ** 2 - (beam_data) ** 2)
    common_resolution = beam.GaussianBeam(beam_diff, freq_data)
    # Convolve to a common resolution.
    map2 = common_resolution.apply(map2)
    map1 = common_resolution.apply(map1)

    # This block of code needs to be split off into a function and applied
    # twice (so we are sure to do the same thing to each).
    #good = np.isfinite(noise1)
    #noise1[~good] = 0.
    #noise1[noise1 == 0] = np.inf # 1.e-30
    #noise1 = 1. / noise1
    noise1 = common_resolution.apply(noise1, mode=mode, cval=0)
    noise1 = common_resolution.apply(noise1, mode=mode, cval=0)
    #noise1[noise1 == 0] = np.inf
    #noise1[noise1 < 1.e-5] = np.inf
    #noise1 = 1. / noise1
    #noise1[noise1 < 1.e-20] = 0.

    #good = np.isfinite(noise2)
    #noise2[~good] = 0.
    #noise2[noise2 == 0] = np.inf # 1.e-30
    #noise2 = 1 / noise2
    noise2 = common_resolution.apply(noise2, mode=mode, cval=0)
    noise2 = common_resolution.apply(noise2, mode=mode, cval=0)
    #noise2[noise2 == 0] = np.inf # 1.e-30
    #noise2[noise2 < 1.e-5] = np.inf
    #noise2 = 1. / noise2
    #noise2[noise2 < 1.e-20] = 0.

    #noise_inv1 = algebra.as_alg_like(noise1, self.noise_inv1)
    #noise_inv2 = algebra.as_alg_like(noise2, self.noise_inv2)

    return [map1, map2], [noise1, noise2]
