
import numpy as np
from fpipe.timestream import timestream_task
from fpipe.check import flux_tod as cf


import logging
logger = logging.getLogger(__name__)


class SimNVSS(timestream_task.TimestreamTask):
    
    params_init = {
        'nvss_cat_list' : None,
        
        'flux_key' : 'NVSS_FLUX',
        'name_key' : 'NVSS_ID',
        'flux_lim' : 1,
        'iso_threshold': 0,
        'max_major_axis' : 100,
        'nvss_range' : None,
    }
    
    prefix='simnvss_'
    
    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        logger.info('simulate tod using nvss catalogue')

        func = ts.bl_data_operate
        func(self.sim_tod, full_data=True, copy_data=False,
             show_progress=show_progress,
             progress_step=progress_step, keep_dist_axis=False)

        ts.redistribute('time')

        #return super(SimNVSS, self).process(ts)

    def sim_tod(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        
        bi = bl[0] - 1
        
        freq = ts['freq'][:]
        fwhm = cf.fwhm_func(None, freq*1.e-3)
        sigma= fwhm / 2. / (2.*np.log(2.))**0.5
        
        ra = ts['ra'][:][:, bi][:, None].astype('float64')
        dec= ts['dec'][:][:, bi][:, None].astype('float64')
        
        nvss_range = self.params['nvss_range']
        if nvss_range is not None:
            _ra_min, _ra_max, _dec_min, _dec_max = nvss_range
        else:
            _ra_min, _ra_max, _dec_min, _dec_max = 0, 180, -90, 90
            
        ext = 10./60. # deg
        _ra_min = max(_ra_min, ra.min()-ext)
        _ra_max = min(_ra_max, ra.max()+ext)
        _dec_min = max(_dec_min, dec.min()-ext)
        _dec_max = min(_dec_max, dec.max()+ext)
        nvss_range = [[_ra_min, _ra_max, _dec_min, _dec_max],]
        #print(nvss_range)
        
        
        nvss_cat_list = self.params['nvss_cat_list']
        flux_key      = self.params['flux_key']
        name_key      = self.params['name_key']
        flux_lim      = self.params['flux_lim']
        threshold     = self.params['iso_threshold']
        max_major_axis= self.params['max_major_axis']
        nvss_ra, nvss_dec, nvss_flx, nvss_name = cf.load_catalogue(nvss_cat_list, 
                                                                   nvss_range,
                                                                   flux_key,
                                                                   name_key, 
                                                                   flux_lim, 
                                                                   threshold,
                                                                   max_major_axis)
        
        nvss_ra  = nvss_ra[None,  :]
        nvss_dec = nvss_dec[None, :]
        nvss_flx = nvss_flx[None, :]
        
        # angular distance between source and pointing direction
        r = np.sin(np.radians(dec)) * np.sin(np.radians(nvss_dec)) \
          + np.cos(np.radians(dec)) * np.cos(np.radians(nvss_dec)) \
          * np.cos(np.radians(ra) - np.radians(nvss_ra))
        r[r>1] = 1.
        r = np.arccos(r) * 180./np.pi * 60. # arcmin
        
        
        sigma = sigma[None, None, :]
        r     = r[:, :, None]
        nvss_flx = nvss_flx[:, :, None]
        flux_model = np.sum( cf.beam_model(r, sigma) * nvss_flx, axis=1) * cf.mJy2K(freq*1.e-3)

        vis[:] = flux_model[:, :, None]
        
        msg = 'Sim Tod for B%02d, %4d sources used'%(bi, nvss_name.shape[0])
        logger.info(msg)
        

