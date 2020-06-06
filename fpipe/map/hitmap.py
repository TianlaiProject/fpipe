import logging
from fpipe.timestream import timestream_task
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

logger = logging.getLogger(__name__)

class MakeAvgMap(timestream_task.TimestreamTask):

    params_init = {
            'main_data' : 'vis',
            'corr' : 'auto',
            'freq_idx' : 0,
            'nside' : 512,
            }
    prefix = 'mkavgm_'

    def process(self, ts):

        nside = self.params['nside']
        self.pixls = np.arange(hp.nside2npix(nside) + 1) - 0.5
        self.hitmap = np.zeros(hp.nside2npix(nside), dtype='float32')
        self.avgmap = np.zeros(hp.nside2npix(nside), dtype='float32')
        self.freq = ts['freq'][self.params['freq_idx']]

        ts.main_data_name = self.params['main_data']

        ts.redistribute('baseline')

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        func(self.makemap, full_data=True, show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        self.write_output(None)

    def makemap(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        nside = self.params['nside']

        ra  = ts['ra'][:, gi]
        dec = ts['dec'][:, gi]


        logger.info('%03d'%gi)

        _vis = np.sum(vis[:, self.params['freq_idx'], :], axis=-1)

        pixidx = hp.ang2pix(nside, ra, dec, lonlat=True)
        self.hitmap += np.histogram(pixidx, self.pixls)[0]
        self.avgmap += np.histogram(pixidx, self.pixls, weights=_vis.flat)[0]

    def write_output(self, output):

        norm = self.hitmap
        norm[norm==0] = np.inf
        self.avgmap /= norm

        nside = self.params['nside']
        map_name = 'avgmap_nside%d_f%6.2fMHz.fits'%(nside, self.freq)
        map_name = output_path(map_name)

        hp.write_map(map_name, self.avgmap, coord='C', overwrite=True)

        self.avgmap[self.avgmap==0] = hp.UNSEEN
        hp.mollview(self.avgmap, title='Map @ %6.2f MHz'%self.freq, badcolor='none')
        hp.graticule(coord='C', dpar=30, dmer=30)

        plt.show()
