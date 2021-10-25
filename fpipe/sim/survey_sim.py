#! 
import logging
import numpy as np
import scipy as sp
import healpy as hp
import h5py
from scipy import interpolate
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz 

from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from fpipe.map import algebra as al
from fpipe.sim import multibeam as mb
from fpipe.map import mapbase

import pygsm

from caput import mpiutil
#from mpi4py import MPI


import gc

import pandas as pd

from fpipe.fnoise import fnoise

__dtype__ = 'float32'

logger = logging.getLogger(__name__)

class SurveySim(pipeline.TaskBase):

    params_init = {
            'prefix'        : 'MeerKAT3',
            'survey_mode'   : 'AzDrift',
            'schedule_file' : None,
            'ant_file'      : '/users/ycli/code/tlpipe/tlpipe/sim/data/meerKAT.dat',
            'ant_Lon'       : (21. + 26./60. + 37.69/3600.) * u.deg,
            'ant_Lat'       :-(30. + 42./60. + 46.53/3600.) * u.deg,
            'multibeam'     : False, 
            #'starttime'     : ['2018-09-15 21:06:00.000',], #UTC
            #'startpointing' : [[55.0, 180.], ],#[Alt, Az]

            #'obs_speed' : 2. * u.deg / u.second,
            #'obs_int'   : 100. * u.second,
            #'obs_tot'   : 5. * u.hour,
            #'obs_az_range' : 15. * u.deg, # for HorizonRasterDrift

            ##'beam_size' : 1. * u.deg, # FWHM
            'beam_file' : None,

            #'block_time' : 1. * u.hour,

            'T_rec' : 25., # K

            'freq' : np.linspace(950, 1350, 32), 

            'FG' : None,
            'HI_model' : None,
            'HI_bias'  : 1.0,
            'HI_model_type' : 'delta', # withbeam, raw, delta
            'HI_scenario'   : 'ideal',
            'HI_mock_ids'   : None,

            'mock_n'   : 10,
            'mock_id'  : None,

            'fnoise'   : True,
            'f0'       : 0.1, 
            'alpha'      : 1.,
            'beta'       : 0.5,
            'delta_nu'   : 1., #MHz
            }

    prefix = 'ssim_'

    def setup(self):

        ant_file = self.params['ant_file']
        #ant_dat = np.genfromtxt(ant_file, 
        #        dtype=[('name', 'S4'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])
        ant_dat = pd.read_fwf(ant_file, header=None, 
                names=['name', 'X', 'Y', 'Z', 'px', 'py'])
        
        self.ants = np.array(ant_dat['name'], dtype='str')
        
        ants_pos = [np.array(ant_dat['X'])[:, None], 
                    np.array(ant_dat['Y'])[:, None], 
                    np.array(ant_dat['Z'])[:, None]]
        self.ants_pos = np.concatenate(ants_pos, axis=1)


        self.SM = globals()[self.params['survey_mode']](self.params['schedule_file'])
        #self.SM.generate_altaz(startalt, startaz, starttime, obs_len, obs_speed, obs_int)
        self.SM.generate_altaz()
        self.SM.radec_list([ant_dat['px'], ant_dat['py']])

        obs_int   = self.SM.obs_int #self.params['obs_int']
        self.obs_int = obs_int
        samplerate = ((1./obs_int).to(u.Hz)).value

        self.block_time = np.array(self.SM.sche['block_time'])
        block_num  = self.block_time.shape[0]

        if self.params['HI_model'] is not None:
            with h5py.File(self.params['HI_model'], 'r') as fhi:
                #_HI_model = al.make_vect(
                _HI_model = al.load_h5(fhi, self.params['HI_model_type'])
                logger.info('HI bias %3.2f'%self.params['HI_bias'])
                _HI_model *= self.params['HI_bias']
                if self.params['HI_mock_ids'] is not None:
                    self.HI_mock_ids = list(self.params['HI_mock_ids'])
                    _HI_model = _HI_model[self.HI_mock_ids]
                else:
                    self.HI_mock_ids = range(_HI_model.shape[0])

                self.mock_n = _HI_model.shape[0]
                self.HI_model = al.make_vect(_HI_model)
                freq = self.HI_model.get_axis('freq')
                logger.info('Using freq from HI model, freq [st=%f, ed=%f, N=%d]'%\
                        (freq[0], freq[-1], freq.shape[0]))
        else:
            self.mock_n = self.params['mock_n']
            freq  = self.params['freq']
            self.HI_mock_ids = range(self.mock_n)

        dfreq = freq[1] - freq[0]
        freq_n = freq.shape[0]

        _obs_int = (obs_int.to(u.second)).value
        self._RMS = self.params['T_rec'] / np.sqrt(_obs_int * dfreq * 1.e6)


        if self.params['FG']:
            GSM = pygsm.GlobalSkyModel(freq_unit='MHz',basemap='haslam')
            # global sky model in healpix format, NSIDE=512, in galactic coord,
            # and in unit of K
            self.syn_model = GSM.generate(freq).T

            if self.params['beam_file'] is not None:
                _bd = np.loadtxt(self.params['beam_file'])
                beam_freq = _bd[:, 0]
                beam_data = _bd[:, 1]
            else:
                fwhm1400=3./60.
                beam_freq = np.linspace(800., 1600., 500).astype('float')
                beam_data = 1.2 * fwhm1400 * 1400. / beam_freq

            fwhm = interpolate.interp1d(beam_freq, beam_data)(freq)
            for i in range(self.syn_model.shape[1]):
                self.syn_model[:, i] = hp.smoothing(self.syn_model[:, i], 
                        fwhm = fwhm[i] * np.pi / 180., verbose=i==0)

        if self.params['fnoise']:
            self.FN = fnoise.FNoise(dtime=obs_int.value, 
                                    dfreq=dfreq, 
                                    alpha = self.params['alpha'],
                                    f0 = self.params['f0'], 
                                    beta = self.params['beta'])

        
        self.get_blorder()

        #self.iter_list =  mpiutil.mpirange(0, obs_len, self.block_len)
        self.iter_list =  mpiutil.mpirange(0, block_num)
        self.iter = 0
        self.iter_num = len(self.iter_list)

    def next(self):

        if self.iter == self.iter_num:
            mpiutil.barrier()
            super(SurveySim, self).next()

        mock_n = self.mock_n

        freq     = self.params['freq']
        delta_nu = self.params['delta_nu']
        #dfreq  = freq[1] - freq[0]
        freq_n = freq.shape[0]

        block_time = self.block_time[:self.iter+1]

        block_n  = int(block_time[-1] / self.obs_int.to(u.s).value)
        idx_st   = int(np.sum(block_time[:-1]) / self.obs_int.to(u.s).value)
        idx_ed   = idx_st + block_n
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed, :]
        dec_list = self.SM.dec[idx_st:idx_ed, :]
        radec_shp = ra_list.shape

        _sky = np.zeros((mock_n, block_n, freq_n, len(self.ants))) + self.params['T_rec']

        if self.params['FG'] :
            logger.info( "add syn")
            syn_model_nside = hp.npix2nside(self.syn_model.shape[0])
            for bb in range(radec_shp[1]):
                c = SkyCoord(ra  = ra_list[:, bb].flatten()  * u.deg, 
                             dec = dec_list[:, bb].flatten() * u.deg, 
                             frame='icrs')
                l = c.galactic.l.radian
                b = c.galactic.b.radian
                _idx_pix = hp.ang2pix(syn_model_nside, np.pi/2.0 - b, l, nest=False)
                _sky[..., bb] += self.syn_model[_idx_pix, :][None, ...] #/ 2.

        if self.params['HI_model'] is not None:
            logger.info( "add HI")
            #HI_model_nside = hp.npix2nside(self.HI_model.shape[0])
            #_idx_pix = hp.ang2pix(HI_model_nside, ra_list, dec_list, lonlat=True)
            #_HI = self.HI_model[_idx_pix, :]
            HI_model_ra  = self.HI_model.get_axis('ra')
            HI_model_dec = self.HI_model.get_axis('dec')
            _HI = np.rollaxis(self.HI_model, 1, 4)
            if not self.params['multibeam']:
                _ra_list = ra_list[:, 0]
                _dec_list = dec_list[:, 0]
                ra_idx  = np.argmin(np.abs(_ra_list[:,None]  - HI_model_ra[None, :]), axis=1)
                dec_idx = np.argmin(np.abs(_dec_list[:,None] - HI_model_dec[None, :]), axis=1)
                _sky = _sky + _HI[:, list(ra_idx), list(dec_idx), :, None] #/ 2.
            else:
                for bb in range(radec_shp[1]):
                    _ra_list = ra_list[:, bb]
                    _dec_list = dec_list[:, bb]
                    ra_idx  = np.argmin(np.abs(_ra_list[:,None]  - HI_model_ra[None, :]), axis=1)
                    dec_idx = np.argmin(np.abs(_dec_list[:,None] - HI_model_dec[None, :]), axis=1)
                    _sky[..., bb] = _sky[..., bb] + _HI[:, list(ra_idx), list(dec_idx), :] #/ 2.

        rvis = np.empty([mock_n, block_n, freq_n, 2, len(self.ants)])
        if self.params['fnoise']:
            logger.info( "add 1/f" )
            for i in range(len(self.ants)):
                for j in range(mock_n):
                    rvis[j,...,0,i] = _sky[j, ..., i]\
                            * (self.FN.realisation(freq_n, block_n, delta_nu)+ 1.)
                    rvis[j,...,1,i] = _sky[j, ..., i]\
                            * (self.FN.realisation(freq_n, block_n, delta_nu)+ 1.)
        else:
            logger.info( "no 1/f" )
            rvis = _sky[..., None, :]
            WN = self._RMS * np.random.randn(mock_n, block_n, freq_n, 2, len(self.ants))
            logger.info( "    %f K(%f K^2)" % (np.std(WN), np.var(WN)) )
            rvis = rvis + WN
            del WN
            gc.collect()


        del _sky
        gc.collect()



        #shp = (mock_n, block_n, freq_n, 4, len(self.blorder))
        #vis = np.empty(shp, dtype=np.complex)
        #vis[:, :, :, 0, self.auto_idx] = rvis[:, :, :, 0, :] + 0. * 1j
        #vis[:, :, :, 1, self.auto_idx] = rvis[:, :, :, 1, :] + 0. * 1j

        for ii in range(mock_n):
            #for ii in self.HI_mock_ids:

            #shp = (block_n, freq_n, 4, len(self.blorder))
            #vis = np.empty(shp, dtype=np.complex)
            #vis[:, :, 0, self.auto_idx] = rvis[ii, :, :, 0, :] + 0. * 1j
            #vis[:, :, 1, self.auto_idx] = rvis[ii, :, :, 1, :] + 0. * 1j

            #output_prefix = '/sim_mock%03d/%s_%s_%s_%s'%(
            #        ii, self.params['prefix'], self.params['survey_mode'], 
            #        self.params['HI_scenario'], self.params['HI_model_type'])
            #self.write_to_file(rvis[ii, ...] + 0. * 1j, output_prefix=output_prefix)
            #self.write_to_file(rvis[ii, ...], output_prefix=output_prefix)
            self.write_to_file(rvis[ii, ...], mock=ii)
            #del vis
            #gc.collect()

        del rvis
        gc.collect()


        self.iter += 1

    def get_blorder(self):

        feedno = []
        channo = []
        feedpos = []
        for ii in range(len(self.ants)):
            ant = self.ants[ii]
            antno = int(ant[1:]) + 1
            feedno.append(antno)
            channo.append([2 * antno - 1, 2 * antno])
            feedpos.append(self.ants_pos[ii])

        feedno = np.array(feedno)
        channo = np.array(channo)
        feedpos = np.array(feedpos)

        antn   = len(feedno)

        #blorder = [[feedno[i], feedno[j]] for i in range(antn) for j in range(i, antn)]
        blorder = [[feedno[i], feedno[i]] for i in range(antn)]
        auto_idx = [blorder.index([feedno[i], feedno[i]]) for i in range(antn)]

        self.blorder = blorder
        self.auto_idx = auto_idx
        self.feedno = feedno
        self.channo = channo
        self.feedpos = feedpos


    def write_to_file(self, vis=None, mock=0):

        output_prefix = '/sim_mock%03d/%s_%s_%s_%s'%(
                self.HI_mock_ids[mock], 
                self.params['prefix'], 
                self.params['survey_mode'], 
                self.params['HI_scenario'], 
                self.params['HI_model_type'])

        block_time = self.block_time[:self.iter+1]

        block_n = int(block_time[-1] / self.obs_int.to(u.s).value)
        idx_st   = int(np.sum(block_time[:-1]) / self.obs_int.to(u.s).value)
        idx_ed   = idx_st + block_n

        #block_n  = self.block_len
        #idx_st   = self.iter_list[self.iter]
        #idx_ed   = idx_st + block_n
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed, :]
        dec_list = self.SM.dec[idx_st:idx_ed, :]


        #output_prefix = '/sim/%s_%s'%(self.params['prefix'], self.params['survey_mode'])
        output_file = output_prefix + '_%s.h5'%t_list[0].datetime.strftime('%Y%m%d%H%M%S')
        output_file = output_path(output_file, relative=True)
        msg = 'RANK %02d %s %s'%(mpiutil.rank, t_list[0], output_file)
        logger.info(msg)
        with h5py.File(output_file, 'w') as df:
            df.attrs['nickname'] = output_prefix
            df.attrs['comment'] = 'just a simulation'
            df.attrs['observer'] = 'Robot'
            history = 'Here is the beginning of the history'
            df.attrs['history'] = history
            df.attrs['keywordver'] = '0.0' # Keyword version.

            # Type B Keywords
            df.attrs['sitename'] = 'MeerKAT'
            df.attrs['sitelat'] = self.SM.location.lat.deg #-(30. + 42./60. + 47.41/3600.)
            df.attrs['sitelon'] = self.SM.location.lon.deg #  21. + 26./60. + 38.00/3600. 
            df.attrs['siteelev'] = 1000.0    # Not precise
            df.attrs['timezone'] = 'UTC+02'  # 
            df.attrs['epoch'] = 2000.0  # year

            df.attrs['telescope'] = 'MeerKAT-Dish-I' # 
            df.attrs['dishdiam'] = 13.5
            df.attrs['nants'] = vis.shape[-1]
            df.attrs['npols'] = vis.shape[-2]
            df.attrs['cylen'] = -1 # For dish: -1
            df.attrs['cywid'] = -1 # For dish: -1

            df.attrs['recvver'] = '0.0'    # Receiver version.
            df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.

            df.attrs['corrver'] = '0.0'    # Correlator version.
            df.attrs['samplingbits'] = 8 # ADC sampling bits.
            df.attrs['corrmode'] = 1 # 2, 3

            obstime = '%s'%t_list[0].isot
            obstime = obstime.replace('-', '/')
            obs_int = self.obs_int
            inttime = obs_int.to(u.second).value
            df.attrs['inttime'] = inttime
            df.attrs['obstime'] = obstime
            #df.attrs['sec1970'] = _t_list[0].unix

            #df.attrs['sec1970'] = t_list.unix[0]
            df['sec1970'] = t_list.unix
            df['sec1970'].attrs['dimname'] = 'Time,'
            df['jul_date'] = t_list.jd
            df['jul_date'].attrs['dimname'] = 'Time,'

            df['ra']  = ra_list
            df['ra'].attrs['dimname'] = 'Time, BaseLines'

            df['dec'] = dec_list
            df['dec'].attrs['dimname'] = 'Time, BaseLines'

            freq = self.params['freq']
            df.attrs['nfreq'] = freq.shape[0] # Number of Frequency Points
            df.attrs['freqstart'] = freq[0] # MHz; Frequency starts.
            df.attrs['freqstep'] = freq[1] - freq[0] # MHz; Frequency step.

            # Data Array
            #df.create_dataset('vis', chunks = (10, 1024, 1, 4), data=vis,
            df.create_dataset('vis', data=vis, dtype = vis.dtype, shape = vis.shape)
            df['vis'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

            # df['pol'] = np.array(['hh', 'vv', 'hv', 'vh'])
            df['pol'] = np.string_(['hh', 'vv', 'hv', 'vh']) # np.string_ for python 3
            df['pol'].attrs['pol_type'] = 'linear'
            
            df['feedno'] = self.feedno
            df['channo'] = self.channo
            df['channo'].attrs['dimname'] = 'Feed No., (HPolarization VPolarization)'
            
            df['blorder'] = self.blorder
            df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'

            
            df['feedpos'] = self.feedpos
            df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
            df['feedpos'].attrs['unit'] = 'm'
            
            #df['antpointing'] = antpointing(16)
            #df['antpointing'].attrs['dimname'] = 'Feed No., (Az,Alt,AzErr,AltErr)'
            #df['antpointing'].attrs['unit'] = 'degree'

class SurveySimToMap(SurveySim, mapbase.MultiMapBase):

    params_init = {
        'field_centre' : (12., 0.,),
        'pixel_spacing' : 0.5,
        'map_shape'     : (10, 10),
    }

    prefix = 'ssimm_'

    def __init__(self, *args, **kwargs):

        super(SurveySimToMap, self).__init__(*args, **kwargs)
        mapbase.MultiMapBase.__init__(self)

    def setup(self):

        super(SurveySimToMap, self).setup()

        params = self.params
        freq  = params['freq']
        self.n_freq = freq.shape[0]
        self.freq_spacing = freq[1] - freq[0]
        self.n_ra, self.n_dec = params['map_shape']
        self.map_shp = (self.n_freq, self.n_ra, self.n_dec)
        self.spacing = params['pixel_spacing']
        self.dec_spacing = self.spacing
        self.ra_spacing  = self.spacing/sp.cos(params['field_centre'][1]*sp.pi/180.)

        axis_names = ('freq', 'ra', 'dec')
        map_tmp = np.zeros(self.map_shp, dtype=__dtype__)
        map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        map_tmp.set_axis_info('freq', freq[self.n_freq//2],      self.freq_spacing)
        map_tmp.set_axis_info('ra',   params['field_centre'][0], self.ra_spacing)
        map_tmp.set_axis_info('dec',  params['field_centre'][1], self.dec_spacing)
        self.map_tmp = map_tmp

        mock_n = self.mock_n

        #for ii in range(mock_n):
        for ii in self.HI_mock_ids:
            output_file = 'sim_mock%03d_%s_%s_%s_%s.h5'%(
                ii, self.params['prefix'], self.params['survey_mode'],
                self.params['HI_scenario'], self.params['HI_model_type'])
            output_file = output_path(output_file, relative=True)
            self.allocate_output(output_file, 'w')
            self.create_dataset_like(-1, 'dirty_map', map_tmp)
            self.create_dataset_like(-1, 'clean_map', map_tmp)
            self.create_dataset_like(-1, 'count_map', map_tmp)

    def write_to_file(self, vis, mock=0):

        super(SurveySimToMap, self).write_to_file(vis, mock)

        block_time = self.block_time[:self.iter+1]

        block_n  = int(block_time[-1] / self.obs_int.to(u.s).value)
        idx_st   = int(np.sum(block_time[:-1]) / self.obs_int.to(u.s).value)
        idx_ed   = idx_st + block_n

        ra  = self.SM.ra[idx_st:idx_ed, :]
        dec = self.SM.dec[idx_st:idx_ed, :]


        ra_bin_edges  = self.map_tmp.get_axis_edges('ra')
        dec_bin_edges = self.map_tmp.get_axis_edges('dec')

        norm = np.histogram2d(ra.flatten(), dec.flatten(), 
                bins=[ra_bin_edges, dec_bin_edges])[0] * 1.
        #norm[norm==0] = np.inf
        for i in range(vis.shape[1]):
            _vis = vis[:, i, ...]
            _vis = np.sum(_vis, axis=1) * 0.5
            hist = np.histogram2d(ra.flatten(), dec.flatten(), 
                    bins=[ra_bin_edges, dec_bin_edges],
                    weights=_vis.flatten())[0]
            #hist = hist / norm.copy()
            self.df_out[mock]['dirty_map'][i, ...] += hist
            self.df_out[mock]['count_map'][i, ...] += norm

    def finish(self):
        mpiutil.barrier()

        if mpiutil.rank0:
            #for ii in self.HI_mock_ids:
            for ii in range(self.mock_n):
                norm = self.df_out[ii]['count_map'][:] 
                hist = self.df_out[ii]['dirty_map'][:] 
                norm[norm==0] = np.inf
                self.df_out[ii]['clean_map'][:] = hist/norm
            print('Finishing CleanMapMaking.')

        mpiutil.barrier()
        super(SurveySimToMap, self).finish()

class ScanMode(object):

    def __init__(self, schedule_file):

        self.read_schedule(schedule_file)

        self.location = EarthLocation.from_geodetic(self.site_lon, self.site_lat)
        self.alt_list = None
        self.az_list  = None
        self.t_list   = None
        self.ra_list  = None
        self.dec_list = None

        #self.params = params

    def read_schedule(self, schedule_file):

        with open(schedule_file) as f:
            for l in f.readlines():
                l = l.split()
                if l[0] != '#': continue
                if l[1] == 'Log.Lat.':
                    self.site_lon = float(l[2]) * u.deg
                    self.site_lat = float(l[3]) * u.deg
                if l[1] == 'AZRange':
                    self.obs_az_range = float(l[2]) * getattr(u, l[3])
                if l[1] == 'ScanSpeed':
                    self.obs_speed = float(l[2]) * getattr(u, l[3]) / getattr(u, l[5])
                if l[1] == 'Int.Time':
                    self.obs_int = float(l[2]) * getattr(u, l[3])
                if l[1] == 'SlewTime':
                    self.obs_slow = float(l[2]) * getattr(u, l[3])

        #self.sche = np.genfromtxt(schedule_file,
        #             names = ['scan', 'UTC', 'LST', 'AZ', 'Alt', 'block_time'],
        #             dtype = ['S1', 'S23', 'f8', 'f8', 'f8', 'f8'],
        #             delimiter=', ')
        self.sche = pd.read_fwf(schedule_file,
                     names = ['scan', 'UTC', 'LST', 'AZ', 'Alt', 'block_time'],
                     delimiter=',', comment='#', skipinitialspace=True)


    def generate_altaz(self):

        pass

    def radec_list(self, beampointing=None):

        _alt = self.alt_list
        _az  = self.az_list
        _t_list = self.t_list
        _obs_len = len(_alt)

        px, py = beampointing

        logger.info('Generating ra dec form multibeam')
        self.ra_list, self.dec_list = mb.get_pointing_meridian_scan(
                _t_list, _alt, _az, self.location, px, py)

    @property
    def ra(self):
        return self.ra_list

    @property
    def dec(self):
        return self.dec_list

class AzDrift(ScanMode):

    def generate_altaz(self):

        obs_speed = self.obs_speed
        obs_int   = self.obs_int
        obs_tot   = self.obs_tot
        obs_len   = int((obs_tot / obs_int).decompose().value) 

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = np.array(self.sche['UTC'], dtype='str')
        start_az_list  = np.array(self.sche['AZ']) * u.deg
        start_alt_list = np.array(self.sche['Alt']) * u.deg
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start = start_alt_list[ii]
            az_start  = start_az_list[ii]

            _alt_list = ((np.ones(obs_len) * alt_start)/u.deg).value
            _az_list  = (((np.arange(obs_len) * obs_speed * obs_int)\
                    + az_start)/u.deg).value

            alt_list.append(_alt_list)
            az_list.append(_az_list)
            t_list.append(np.arange(obs_len) * obs_int + starttime)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

class DriftScan(ScanMode):

    def generate_altaz(self):

        obs_int      = self.obs_int #self.params['obs_int']
        block_time   = np.array(self.sche['block_time']) * u.s

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = np.array(self.sche['UTC'], dtype='str')
        start_az_list  = np.array(self.sche['AZ']) * u.deg
        start_alt_list = np.array(self.sche['Alt']) * u.deg
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii], scale='utc')
            alt_start = start_alt_list[ii]
            az_start  = start_az_list[ii]
            obs_len   = int((block_time[ii] / obs_int).decompose().value) 

            _alt_list = (np.ones(obs_len) * alt_start).value
            alt_list.append(_alt_list)

            t_list.append(np.arange(obs_len) * obs_int + starttime)

            _az_list = (np.ones(obs_len) * az_start).value
            az_list.append(_az_list)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

class HorizonRasterDrift(ScanMode):

    def generate_altaz(self):

        obs_speed    = self.obs_speed #self.params['obs_speed']
        obs_int      = self.obs_int #self.params['obs_int']
        block_time   = np.array(self.sche['block_time']) * u.s 
        obs_az_range = self.obs_az_range #self.params['obs_az_range']

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = np.array(self.sche['UTC'], dtype='str')
        #startpointing_list = self.params['startpointing']
        start_az_list  = np.array(self.sche['AZ']) * u.deg
        start_alt_list = np.array(self.sche['Alt']) * u.deg
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start = start_alt_list[ii]
            az_start  = start_az_list[ii]

            obs_len   = int((block_time[ii] / obs_int).decompose().value) 

            _alt_list = (np.ones(obs_len) * alt_start).value
            alt_list.append(_alt_list)

            #t_list.append((np.arange(obs_len) - 0.5 * obs_len) * obs_int + starttime)
            t_list.append(np.arange(obs_len) * obs_int + starttime)

            _az_space = (obs_speed * obs_int).to(u.deg)
            _one_way_npoints = (obs_az_range / obs_speed / obs_int).decompose()
            #_az_list = np.arange(_one_way_npoints) - 0.5 * _one_way_npoints
            #_az_list = np.append(_az_list, -_az_list)
            _az_list = np.arange(_one_way_npoints)
            _az_list = np.append(_az_list, _az_list[::-1])
            _az_list = _az_list * _az_space
            _az_list += az_start
            _az_list = _az_list.value
            _az_list = [_az_list[i%int(2.*_one_way_npoints)] for i in range(obs_len)]
            az_list.append(_az_list)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))


class RealOBS(ScanMode):

    def read_schedule(self, schedule_file):

        self.obsfiles = []

        with open(schedule_file) as f:
            for l in f.readlines():
                l = l.split()
                if l[0] != '#': continue
                if l[1] == 'Log.Lat.':
                    self.site_lon = float(l[2]) * u.deg
                    self.site_lat = float(l[3]) * u.deg

        self.obsfiles = np.genfromtxt(schedule_file, dtype='S')

    def generate_altaz(self):

        import pdb; pdb.set_trace()
        block_time = []

        self.t_list   = []
        self.az_list  = []
        self.alt_list = []

        for fname in self.obsfiles:

            with h5py.File(fname, 'r') as df:

                inttime = df.attrs['inttime']
                obstime = df.attrs['obstime']
                ra_list = df['ra'][:, 0] * u.deg
                dec_list= df['dec'][:, 0] * u.deg

            self.obs_int = inttime * u.s

            block_n = ra_list.shape[0]

            obstime = obstime.replace('(UTC)', '')
            obstime = obstime.replace('/', '-')
            t_list  = Time(obstime)
            t_list += np.arange(block_n) * inttime * u.s

            _c = SkyCoord(ra=ra_list, dec=dec_list, frame='icrs')
            _c = _c.transform_to(AltAz(obstime=t_list, location=self.location))

            az_list  = _c.altaz.az.deg 
            alt_list = _c.altaz.alt.deg

            self.t_list.append(list(t_list.fits))
            self.az_list.append(list(az_list))
            self.alt_list.append(list(alt_list))
            block_time.append(block_n * inttime)

        self.t_list   = Time(np.concatenate(self.t_list), format='fits')
        self.az_list  = np.concatenate(self.az_list) * u.deg
        self.alt_list = np.concatenate(self.alt_list) * u.deg

        self.sche = np.array(block_time, dtype=[('block_time', 'f8'), ])

class RealOBSc(RealOBS):

    def generate_altaz(self):

        block_time = []

        #self.t_list   = []
        #self.az_list  = []
        #self.alt_list = []

        obstime = None
        ra_list = []
        dec_list = []
        block_n = 0

        for fname in self.obsfiles:

            with h5py.File(fname, 'r') as df:

                inttime = df.attrs['inttime']
                _obstime = df.attrs['obstime']
                ra = df['ra'][:, 0] #* u.deg
                dec= df['dec'][:, 0] #* u.deg

            self.obs_int = inttime * u.s

            block_n += ra.shape[0]

            _obstime = _obstime.replace('(UTC)', '')
            _obstime = _obstime.replace('/', '-')
            if obstime is None:
                obstime = _obstime
            elif obstime != _obstime:
                raise ValueError('Obstime not the same')

            ra_list  += list(ra)
            dec_list += list(dec)

        self.t_list  = Time(obstime)
        self.t_list += np.arange(block_n) * self.obs_int

        ra_list  = np.array(ra_list)  * u.deg
        dec_list = np.array(dec_list) * u.deg


        _c = SkyCoord(ra=ra_list, dec=dec_list, frame='icrs')
        _c = _c.transform_to(AltAz(obstime=self.t_list, location=self.location))

        self.az_list  = _c.altaz.az.deg  * u.deg
        self.alt_list = _c.altaz.alt.deg * u.deg

        self.sche = np.array([block_n * self.obs_int], dtype=[('block_time', 'f8'), ])
