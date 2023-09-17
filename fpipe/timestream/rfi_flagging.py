"""RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
import h5py as h5
import gc
from . import timestream_task
from fpipe.timestream import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.rfi import interpolate
from tlpipe.rfi import gaussian_filter
from tlpipe.rfi import sum_threshold

import logging
logger = logging.getLogger(__name__)

class Flag(timestream_task.TimestreamTask):
    """RFI flagging.

    RFI flagging by using the SumThreshold method, applied to sum of 19 feeds.

    """

    params_init = {
                    'first_threshold': 6.0,
                    'exp_factor': 1.5,
                    'distribution': 'Rayleigh',
                    'max_threshold_len': 1024,
                    'sensitivity': 1.0,
                    'min_connected': 1,
                    'flag_direction': ('time', 'freq'),
                    'tk_size': 1.0, # 128.0 for dish
                    'fk_size': 3.0, # 2.0 for dish
                    'threshold_num': 2, # number of threshold

                    'bad_freqs' : [],
                    'rfi_hist' : None,
                  }

    prefix = 'rf_'

    def process(self, ts):

        ts.redistribute('baseline')

        #func = ts.pol_and_bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        #func(self.flag, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False)
        self.flag(ts)
        if self.params['rfi_hist'] is not None:
            output_rfi_hist(ts, output=self.params['rfi_hist'])

        return super(Flag, self).process(ts)

    #def flag(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
    def flag(self, ts, **kwargs):
        """Function that does the actual flag."""

        vis = ts.vis

        freq = ts['freq'][:]

        # if all have been masked, no need to flag again
        if ts.vis_mask[:].all():
            return

        first_threshold = self.params['first_threshold']
        exp_factor = self.params['exp_factor']
        distribution = self.params['distribution']
        max_threshold_len = self.params['max_threshold_len']
        sensitivity = self.params['sensitivity']
        min_connected = self.params['min_connected']
        flag_direction = self.params['flag_direction']
        tk_size = self.params['tk_size']
        fk_size = self.params['fk_size']
        threshold_num = max(0, int(self.params['threshold_num']))

        if 'ns_on' in iter(ts.keys()):
            has_ns = True
            if len(ts['ns_on'].shape) == 1:
                on = ts['ns_on']
            elif len(ts['ns_on'].shape) == 2:
                on = np.sum(ts['ns_on'], axis=1) # assume noise diode align beteen feeds
            else:
                raise RuntimeError('ns_on must be a 1d or 2d array')
        else:
            has_ns = False

        #vis_abs = np.abs(vis) # operate only on the amplitude
        vis_abs = np.sum(np.abs(vis), axis=(2, 3)) # sum over pol and feeds
        vis_mask = np.sum(ts.vis_mask[:], axis=(2, 3))
        if has_ns:
            vis_mask[on] = True # temporarily make ns_on masked

        # remove GNSS contaminated freqs
        for _bf in self.params['bad_freqs']:
            logger.info('mask freq %f - %f MHz'%(_bf[0], _bf[1]))
            vis_mask[:, (freq > _bf[0]) * (freq < _bf[1])] = True

        # first round
        # first complete masked vals due to ns by interpolate
        itp = interpolate.Interpolate(vis_abs, vis_mask)
        background = itp.fit()
        # Gaussian fileter
        gf = gaussian_filter.GaussianFilter(background, time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
        background = gf.fit()
        # sum-threshold
        vis_diff = vis_abs - background
        # an initial run of N = 1 only to remove extremely high amplitude RFI
        st = sum_threshold.SumThreshold(vis_diff, vis_mask, first_threshold, exp_factor, distribution, 1, min_connected)
        st.execute(sensitivity, flag_direction)

        # if all have been masked, no need to flag again
        if st.vis_mask.all():
            ts.vis_mask[:] += st.vis_mask[:, :, None, None]
            if has_ns:
                ts.vis_mask[on] = False # undo ns_on mask
            return

        # next rounds
        for i in range(threshold_num):
            # Gaussian fileter
            gf = gaussian_filter.GaussianFilter(vis_diff, np.array(st.vis_mask), time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
            background = gf.fit()
            # sum-threshold
            vis_diff = vis_diff - background
            st = sum_threshold.SumThreshold(vis_diff, np.array(st.vis_mask), first_threshold, exp_factor, distribution, max_threshold_len, min_connected)
            st.execute(sensitivity, flag_direction)

            # if all have been masked, no need to flag again
            if st.vis_mask.all():
                break

        # replace vis_mask with the flagged mask
        ts.vis_mask[:] += st.vis_mask[:, :, None, None]
        if has_ns:
            #ts.vis_mask[(on.astype('bool'), Ellipsis)] = False # undo ns_on mask
            #ts.vis_mask[on[:, None, None, None]] = False # undo ns_on mask
            ts.vis_mask[:] *= ~(on[:, None, None, None].astype('bool'))

def output_rfi_hist(ts, tbins=None, fbins=[1050, 1140, 1310, 1450], output=None):
    
    if tbins is None:
        nbin = 100
        bins = np.logspace(-2, 5, nbin+1)
    else:
        bins = tbins
        nbin = len(tbins) - 1
    
    hist = np.zeros((len(fbins)-1, nbin))
    hist_nomask = np.zeros((len(fbins)-1, nbin))
    
    vis = ts['vis'][:]
    vis_mask = ts['vis_mask'][:]
    on = np.sum(ts['ns_on'], axis=1)
    freq = ts['freq'][:]

    #on = on.astype('bool')
    # rm nd on
    vis = vis[~on]
    vis_mask = vis_mask[~on]
    vis_mask = vis_mask.astype('bool')

    freq = freq[1:]

    msk = vis_mask[:, 1:, ...] + vis_mask[:, :-1, ...]
    vis_diff = np.abs(vis[:, 1:, ...] - vis[:, :-1, ...])
    #vis_diff = np.ma.array(vis_diff, mask=msk)

    for jj in range(hist.shape[0]): 
        freq_sel = (freq > fbins[jj]) * (freq < fbins[jj+1])
        _vis_diff = vis_diff[:, freq_sel, ...].flatten()
        hist_nomask[jj] += np.histogram(_vis_diff, bins=bins)[0]
        _msk = msk[:, freq_sel, ...].flatten()
        _vis_diff[_msk] = -1
        hist[jj] += np.histogram(_vis_diff, bins=bins)[0]
        del _vis_diff, _msk

    del vis, ts, vis_diff, msk
    gc.collect()
        
    if output is not None:
        with h5.File(output, 'w') as f:
            f['hist'] = hist
            f['hist_nomask'] = hist_nomask
            f['fbin'] = np.array(fbins)
            f['Tbin'] = bins
