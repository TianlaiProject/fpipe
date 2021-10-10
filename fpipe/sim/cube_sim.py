#! 
import numpy as np
import healpy as hp
import h5py
import copy
from numpy import random
from scipy.interpolate import interp2d
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from caput import mpiutil

from fpipe.map import algebra as al

from cora.signal import corr21cm
from fpipe.sim import lognorm
from fpipe.map import mapbase

from . import beam

#from pipeline.Observatory.Receivers import Noise
__nu21__ = 1420.40575177

class CubeSim(pipeline.TaskBase, mapbase.MapBase):

    params_init = {
            'prefix'        : 'MeerKAT3',


            'freq' : np.linspace(950, 1350, 32), 

            'mock_n' : 100,
            'scenario': 'str',
            'refinement': 1,

            'map_tmp' : None,
            'map_tmp_key' : 'clean_map',
            'map_tmp_weight' : 'noise_diag',
            'selection'      : None,

            'field_centre' : (12., 0.,),
            'pixel_spacing' : 0.5,
            'map_shape'     : (10, 10),
            'map_pad'       : 5,

            #'outfile_raw'     : True,
            #'outfile_physical': True,
            'outfiles' : ['raw', 'delta', 'withbeam'],
            'outfiles_split' : [],

            'lognorm' : False,
            'beam_file' : None,

            }

    prefix = 'csim_'

    def __init__(self, *args, **kwargs):

        super(CubeSim, self).__init__(*args, **kwargs)
        mapbase.MapBase.__init__(self)

        if self.params['lognorm']:
            self.corr = lognorm.LogNormal
        else:
            self.corr = corr21cm.Corr21cm

    def setup(self):

        self.refinement = self.params['refinement']
        self.scenario = self.params['scenario']

        map_pad = self.params['map_pad']

        if self.params['map_tmp'] is None:

            freq  = self.params['freq'] #* 1.e6
            freq_d = freq[1] - freq[0]
            freq_n = freq.shape[0]
            freq_c = freq[freq_n//2]

            field_centre = self.params['field_centre']
            spacing = self.params['pixel_spacing']
            dec_spacing = spacing
            ra_spacing  = - spacing / np.cos(field_centre[1] * np.pi / 180.)

            axis_names = ['freq', 'ra', 'dec']
            map_shp = [x + map_pad for x in self.params['map_shape']]
            map_tmp = np.zeros([freq_n, ] + map_shp)
            map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
            map_tmp.set_axis_info('freq', freq_c, freq_d)
            map_tmp.set_axis_info('ra',   field_centre[0], ra_spacing)
            map_tmp.set_axis_info('dec',  field_centre[1], dec_spacing)
            self.map_tmp = map_tmp

        else:

            pad_shp = ((0, 0),(map_pad,map_pad),(map_pad,map_pad))
            with h5py.File(self.params['map_tmp'], 'r') as f:
                _map_tmp = al.load_h5(f, self.params['map_tmp_key'])
                _axis_names = _map_tmp.info['axes']
                _info = _map_tmp.info
                _map_tmp = np.pad(_map_tmp, pad_shp, 'constant')
                _map_tmp = al.make_vect(_map_tmp, axis_names = _axis_names)
                _map_tmp.info.update(_info)
                _weight = al.load_h5(f, self.params['map_tmp_weight'])
                _weight = np.pad(_weight, pad_shp, 'constant')
                _weight = al.make_vect(_weight, axis_names = _axis_names)
            #self.map_tmp = al.zeros_like(_map_tmp)
            self.map_tmp = _map_tmp
            self.weight = _weight

        # here we use 300 h km/s from WiggleZ for streaming dispersion
        self.streaming_dispersion = 300.*0.72
        self.map_pad = map_pad

        #self.beam_data = np.array([1., 1., 1.])
        #self.beam_freq = np.array([900, 1100, 1400]) #* 1.e6
        if self.params['beam_file'] is not None:
            _bd = np.loadtxt(self.params['beam_file'])
            self.beam_freq = _bd[:, 0] #* 1.e6
            self.beam_data = _bd[:, 1]
        else:
            fwhm1400=0.9
            self.beam_freq = np.linspace(800., 1600., 500).astype('float')
            self.beam_data = 1.2 * fwhm1400 * 1400. / self.beam_freq
            #self.beam_freq *= 1.e6

        random.seed(3936650408)
        seeds = random.random_integers(100000000, 1000000000, mpiutil.size)
        self.seed = seeds[mpiutil.rank]
        print("RANK: %02d with random seed [%d]"%(mpiutil.rank, self.seed))
        random.seed(self.seed)


        self.outfiles = self.params['outfiles']
        self.outfiles_split = self.params['outfiles_split']
        self.open_outputfiles()

        self.iter_list = mpiutil.mpirange(self.params['mock_n'])
        self.iter      = 0
        self.iter_num  = len(self.iter_list)

    def __next__(self):

        if self.iter == self.iter_num:
            mpiutil.barrier()
            #self.close_outputfiles()
            next(super(CubeSim, self))

        print("rank %03d, %03d"%(mpiutil.rank, self.iter_list[self.iter]))

        self.realize_simulation()
        if 'delta' in self.outfiles:
            self.make_delta_sim()
        if 'optsim' in self.outfiles:
            self.make_optical_sim()
        if 'withbeam' in self.outfiles:
            self.convolve_by_beam()

        self.write_to_file()
        self.write_to_file_splitmock()

        self.iter += 1


    def realize_simulation(self):
        """do basic handling to call Richard's simulation code
        this produces self.sim_map and self.sim_map_phys
        """
        # corr.like_kiyo_map requires frequency in Hz
        self.map_tmp.info['freq_delta']  *= 1.e6
        self.map_tmp.info['freq_centre'] *= 1.e6

        if self.scenario == "nostr":
            print("running dd+vv and no streaming case")
            #simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp)
            simobj = self.corr.like_kiyo_map(self.map_tmp)
            maps = simobj.get_kiyo_field_physical(refinement=self.refinement)

        else:
            if self.scenario == "str":
                print("running dd+vv and streaming simulation")
                #simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp,
                simobj = self.corr.like_kiyo_map(self.map_tmp,
                                           sigma_v=self.streaming_dispersion)

                maps = simobj.get_kiyo_field_physical(refinement=self.refinement)

            if self.scenario == "ideal":
                print("running dd-only and no mean simulation")
                #simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp)
                simobj = self.corr.like_kiyo_map(self.map_tmp)
                maps = simobj.get_kiyo_field_physical(
                                            refinement=self.refinement,
                                            density_only=True,
                                            no_mean=True,
                                            no_evolution=True)

        # corr.like_kiyo_map requires frequency in Hz
        # chenge back to MHz for the rest
        self.map_tmp.info['freq_delta']  /= 1.e6
        self.map_tmp.info['freq_centre'] /= 1.e6

        self.simobj = simobj
        self.kk_input = np.logspace(-2, 0, 200)
        self.pk_input = simobj.get_pwrspec(self.kk_input)

        (gbtsim, gbtphys, physdim) = maps

        # process the physical-space map
        self.sim_map_phys = al.make_vect(gbtphys, axis_names=('freq', 'ra', 'dec'))
        pshp = self.sim_map_phys.shape

        # define the axes of the physical map; several alternatives are commented
        info = {}
        info['axes'] = ('freq', 'ra', 'dec')
        info['type'] = 'vect'
        info['freq_delta'] = abs(physdim[0] - physdim[1]) / float(pshp[0] - 1)
        info['freq_centre'] = physdim[0] + info['freq_delta'] * float(pshp[0] // 2)
        #        'freq_centre': abs(physdim[0] + physdim[1]) / 2.,

        info['ra_delta'] = abs(physdim[2]) / float(pshp[1] - 1)
        #info['ra_centre'] = info['ra_delta'] * float(pshp[1] // 2)
        #        'ra_centre': abs(physdim[2]) / 2.,
        info['ra_centre'] = 0.

        info['dec_delta'] = abs(physdim[3]) / float(pshp[2] - 1)
        #info['dec_centre'] = info['dec_delta'] * float(pshp[2] // 2)
        #        'dec_centre': abs(physdim[3]) / 2.,
        info['dec_centre'] = 0.

        self.sim_map_phys.info = info

        # process the map in observation coordinates
        self.sim_map = al.make_vect(gbtsim, axis_names=('freq', 'ra', 'dec'))
        self.sim_map.copy_axis_info(self.map_tmp)
        self.sim_map_raw = self.sim_map

    def make_delta_sim(self):
        r"""this produces self.sim_map_delta"""
        print("making sim in units of overdensity")
        freq_axis = self.sim_map.get_axis('freq')  #/ 1.e6
        z_axis = __nu21__ / freq_axis - 1.0

        #simobj = corr21cm.Corr21cm()
        #simobj = self.corr()
        simobj = self.simobj
        T_b = simobj.T_b(z_axis) #* 1e-3

        self.sim_map_delta = copy.deepcopy(self.sim_map)
        self.sim_map_delta /= T_b[:, np.newaxis, np.newaxis]

    def make_optical_sim(self):

        if self.params['selection'] is None:
            print('optical sim need selection function, pass')
            return
        else:
            with h5py.File(self.params['selection'], 'r') as f:
                _sel = al.load_h5(f, 'separable')
                _axis_names = _sel.info['axes']
                _sel = al.make_vect(_sel, axis_names = _axis_names)
                _sel_ra  = _sel.get_axis('ra')
                _sel_dec = _sel.get_axis('dec')
                _sel = np.ma.masked_equal(_sel, 0)

                # the simulated cube may have different shape than the 
                # original shape of selection function, we 2d interpolate 
                # to the correct shape
                _sel_mean = np.ma.mean(_sel, axis=0)
                _sel_freq = np.ma.sum(_sel, axis=(1, 2))
                _sel_freq /= np.ma.sum(_sel_freq)
                #_cut = np.percentile(_sel_mean[~_sel_mean.mask], 60)
                #_sel_mean[_sel_mean>_cut] = _cut
                _sel_2dintp = interp2d(_sel_dec, _sel_ra, _sel_mean, 
                        bounds_error=False, fill_value=0)

                _ra  = self.map_tmp.get_axis('ra')
                _dec = self.map_tmp.get_axis('dec')
                _ra_s = np.argsort(_ra)
                _sel = _sel_2dintp(_dec, _ra[_ra_s])[_ra_s, ...]
                #_sel = _sel * al.ones_like(self.map_tmp)
                _sel = _sel * _sel_freq[:, None, None]

            if not hasattr(self, 'sim_map_delta'):
                self.make_delta_sim()

            poisson_vect = np.vectorize(np.random.poisson)
            mean_num_gal = (self.sim_map_delta + 1.) * _sel
            self.sim_map_optsim = poisson_vect(mean_num_gal)
            self.sim_map_optsim = mean_num_gal
            _sel[_sel==0] = np.inf
            self.sim_map_optsim = self.sim_map_optsim/_sel - 1.
            _sel[_sel==np.inf] = 0.
            self.sel = _sel


    def convolve_by_beam(self):
        r"""this produces self.sim_map_withbeam"""
        print("convolving simulation by beam")
        beamobj = beam.GaussianBeam(self.beam_data, self.beam_freq)
        self.sim_map_withbeam = beamobj.apply(self.sim_map)


    def open_outputfiles(self):

        output_prefix = '/%s_cube_%s'%(self.params['prefix'], self.params['scenario'])
        output_file = output_prefix + '.h5'
        output_file = output_path(output_file, relative=True)
        self.allocate_output(output_file, 'w')

        dset_shp  = (self.params['mock_n'], ) + self.map_tmp.shape
        dset_info = {}
        dset_info['axes'] = ('mock', 'freq', 'ra', 'dec')
        dset_info['type'] = 'vect'
        dset_info['mock_delta']  = 1
        dset_info['mock_centre'] = self.params['mock_n']//2
        dset_info['freq_delta']  = self.map_tmp.info['freq_delta'] #/ 1.e6
        dset_info['freq_centre'] = self.map_tmp.info['freq_centre'] #/ 1.e6
        dset_info['ra_delta']    = self.map_tmp.info['ra_delta']
        dset_info['ra_centre']   = self.map_tmp.info['ra_centre']
        dset_info['dec_delta']   = self.map_tmp.info['dec_delta']
        dset_info['dec_centre']  = self.map_tmp.info['dec_centre']
        #if self.params['outfile_raw']:
        for outfile in self.outfiles:
            self.create_dataset(outfile, dset_shp, dset_info)

    def write_to_file(self):

        df = self.df

        for outfile in self.outfiles:
            df[outfile][self.iter_list[self.iter], ...] = getattr(self, 'sim_map_'+outfile)

        if self.iter==0:
            df['kk_input'] = self.kk_input
            df['pk_input'] = self.pk_input

        #if 'optsim' in self.outfiles:
        #    self.write_optsim()

    def write_to_file_splitmock(self):

        output_prefix = '/%s_cube_%s'%(self.params['prefix'], self.params['scenario'])

        for outfile in self.outfiles_split:
            mock_idx = self.iter_list[self.iter]
            output_file = output_prefix + '_%03d_%s.h5'%(mock_idx, outfile)
            output_file = output_path(output_file, relative=True)
            #self.allocate_output(output_file, 'w')
            if outfile is 'optsim':
                map_key = 'delta'
                weight_key = 'separable'
                map_name = 'sim_map_optsim'
                weight_name = 'sel'

            else:
                map_key = self.params['map_tmp_key']
                weight_key = self.params['map_tmp_weight']
                map_name = 'sim_map_'+outfile
                weight_name = 'weight'


            map_pad = self.map_pad
            map_shp = self.map_tmp.shape
            _map_shp = (map_shp[0], map_shp[1] - 2*map_pad, map_shp[2]-2*map_pad)
            map_slice = (slice(0,None),
                         slice(map_pad,map_shp[1]-map_pad),
                         slice(map_pad,map_shp[2]-map_pad))
            with h5py.File(output_file, 'w') as f:
                d = f.create_dataset(map_key, _map_shp, dtype=self.map_tmp.dtype)
                _d = getattr(self, map_name)
                d[:] = _d[map_slice]
                for key, value in self.map_tmp.info.items():
                    d.attrs[key] = repr(value)
                d.attrs['freq_delta']  = repr(self.map_tmp.info['freq_delta'] )
                d.attrs['freq_centre'] = repr(self.map_tmp.info['freq_centre'] )
                #d.attrs['ra_delta']    = repr(self.map_tmp.info['ra_delta']) 
                #d.attrs['ra_centre']   = repr(self.map_tmp.info['ra_centre'])
                #d.attrs['dec_delta']   = repr(self.map_tmp.info['dec_delta'])
                #d.attrs['dec_centre']  = repr(self.map_tmp.info['dec_centre'])

                if hasattr(self, weight_name):
                    #mask = 
                    #mask[mask!=0] = 1.
                    #d[:] *= mask
                    n = f.create_dataset(weight_key, _map_shp, dtype=self.map_tmp.dtype)
                    n[:] = getattr(self, weight_name)[map_slice]
                    for key, value in self.map_tmp.info.items():
                        n.attrs[key] = repr(value)
                    n.attrs['freq_delta']  = repr(self.map_tmp.info['freq_delta'] )
                    n.attrs['freq_centre'] = repr(self.map_tmp.info['freq_centre'])
                    #n.attrs['ra_delta']    = repr(self.map_tmp.info['ra_delta'])
                    #n.attrs['ra_centre']   = repr(self.map_tmp.info['ra_centre'])
                    #n.attrs['dec_delta']   = repr(self.map_tmp.info['dec_delta'])
                    #n.attrs['dec_centre']  = repr(self.map_tmp.info['dec_centre'])


    #def write_optsim(self):

    #    output_prefix = '/%s_cube_%s'%(self.params['prefix'], self.params['scenario'])
    #    mock_idx = self.iter_list[self.iter]
    #    output_file = output_prefix + '_%03d_%s.h5'%(mock_idx, 'optsim')
    #    output_file = output_path(output_file, relative=True)
    #    map_key = 'delta'
    #    weight_key = 'separable'
    #    map_name = 'sim_map_optsim'
    #    weight_name = 'sel'
    #    with h5py.File(output_file, 'w') as f:
    #        d = f.create_dataset(map_key, 
    #                             self.map_tmp.shape, 
    #                             dtype=self.map_tmp.dtype)
    #        d[:] = getattr(self, map_name)
    #        for key, value in self.map_tmp.info.iteritems():
    #            d.attrs[key] = repr(value)

    #        n = f.create_dataset(weight_key, 
    #                             self.map_tmp.shape, 
    #                             dtype=self.map_tmp.dtype)
    #        n[:] = getattr(self, weight_name)
    #        for key, value in self.map_tmp.info.iteritems():
    #            n.attrs[key] = repr(value)


    #def close_outputfiles(self):

    #    self.df.close()

class TransferSim(CubeSim):

    prefix = 'tsim_'

    def open_outputfiles(self):

        pass

    def write_to_file(self):

        output_prefix = '/%s_cube_%s'%(self.params['prefix'], self.params['scenario'])

        for outfile in self.outfiles:
            mock_idx = self.iter_list[self.iter]
            output_file = output_prefix + '_%03d_%s.h5'%(mock_idx, outfile)
            output_file = output_path(output_file, relative=True)
            #self.allocate_output(output_file, 'w')
            if outfile is 'optsim':
                map_key = 'delta'
                weight_key = 'separable'
                map_name = 'sim_map_optsim'
                weight_name = 'sel'

            else:
                map_key = self.params['map_tmp_key']
                weight_key = self.params['map_tmp_weight']
                map_name = 'sim_map_'+outfile
                weight_name = 'weight'


            with h5py.File(output_file, 'w') as f:
                d = f.create_dataset(map_key, 
                                     self.map_tmp.shape, 
                                     dtype=self.map_tmp.dtype)
                d[:] = getattr(self, map_name)
                for key, value in self.map_tmp.info.items():
                    d.attrs[key] = repr(value)

                if hasattr(self, weight_name):
                    #mask = 
                    #mask[mask!=0] = 1.
                    #d[:] *= mask
                    n = f.create_dataset(weight_key, 
                                         self.map_tmp.shape, 
                                         dtype=self.map_tmp.dtype)
                    n[:] = getattr(self, weight_name)
                    for key, value in self.map_tmp.info.items():
                        n.attrs[key] = repr(value)

class ScanMode(object):

    def __init__(self, site_lon, site_lat, params=None):

        self.location = EarthLocation.from_geodetic(site_lon, site_lat)
        self.alt_list = None
        self.az_list  = None
        self.t_list   = None
        self.ra_list  = None
        self.dec_list = None

        self.params = params

    def generate_altaz(self):

        pass

    def radec_list(self):

        _alt = self.alt_list
        _az  = self.az_list
        _t_list = self.t_list
        _obs_len = len(_alt)

        #radec_list = np.zeros([int(_obs_len), 2])
        #for i in mpiutil.mpirange(_obs_len): #[rank::size]:
            #print mpiutil.rank, i
        pp = SkyCoord(alt=_alt, az=_az,  frame='altaz', 
                location=self.location, obstime=_t_list)
        pp = pp.transform_to('icrs')
        #radec_list[:, 0] = pp.ra.deg
        #radec_list[:, 1] = pp.dec.deg

        #radec_list =  mpiutil.allreduce(radec_list)
        self.ra_list  = pp.ra.deg
        self.dec_list = pp.dec.deg

    @property
    def ra(self):
        return self.ra_list

    @property
    def dec(self):
        return self.dec_list

class AzDrift(ScanMode):

    def generate_altaz(self):

        obs_speed = self.params['obs_speed']
        obs_int   = self.params['obs_int']
        obs_tot   = self.params['obs_tot']
        obs_len   = int((obs_tot / obs_int).decompose().value) 

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = self.params['starttime']
        startpointing_list = self.params['startpointing']
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start, az_start  = startpointing_list[ii]
            alt_start *= u.deg
            az_start  *= u.deg

            _alt_list = ((np.ones(obs_len) * alt_start)/u.deg).value
            _az_list  = (((np.arange(obs_len) * obs_speed * obs_int)\
                    + az_start)/u.deg).value

            alt_list.append(_alt_list)
            az_list.append(_az_list)
            t_list.append(np.arange(obs_len) * obs_int + starttime)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

class HorizonRasterDrift(ScanMode):

    def generate_altaz(self):

        obs_speed = self.params['obs_speed']
        obs_int   = self.params['obs_int']
        #obs_tot   = self.params['obs_tot']
        #obs_len   = int((obs_tot / obs_int).decompose().value) 
        block_time = self.params['block_time']
        obs_len   = int((block_time / obs_int).decompose().value) 
        obs_az_range = self.params['obs_az_range']

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = self.params['starttime']
        startpointing_list = self.params['startpointing']
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start, az_start  = startpointing_list[ii]
            alt_start *= u.deg
            az_start  *= u.deg

            _alt_list = ((np.ones(obs_len) * alt_start)/u.deg).value
            alt_list.append(_alt_list)

            t_list.append((np.arange(obs_len) - 0.5 * obs_len) * obs_int + starttime)

            _az_space = obs_speed * obs_int
            _one_way_npoints = (obs_az_range / obs_speed / obs_int).decompose()
            _az_list = np.arange(_one_way_npoints) - 0.5 * _one_way_npoints
            _az_list = np.append(_az_list, -_az_list)
            _az_list *= _az_space
            _az_list += az_start
            _az_list = (_az_list / u.deg).value
            _az_list = [_az_list[i%int(2.*_one_way_npoints)] for i in range(obs_len)]
            az_list.append(_az_list)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

