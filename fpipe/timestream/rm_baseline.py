from fpipe.timestream import timestream_task
from scipy import signal
from scipy.interpolate import interp1d

from fpipe.container.timestream import FAST_Timestream

from fpipe.timestream import bandpass_cal

import matplotlib.pyplot as plt

import numpy as np
import h5py as h5

from scipy import signal

import warnings
warnings.simplefilter('ignore', np.RankWarning)

class Remove_Baseline(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'baseline_file' : None,
            }

    prefix = 'rmbsl_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(Remove_Baseline, self).process(ts)

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        on = ts['ns_on'][:, gi]
        on = on.astype('bool')

        time = ts['sec1970'][:]
        bsl = fit_baseline(vis[~on], time[~on], vis_mask[~on], 
                self.params['baseline_file'])

        vis[~on] -= bsl

        _vis = np.ma.array(vis, mask=vis_mask)
        _vis.mask += on[:, None, None]

        vis -= np.ma.median(_vis, axis=0)

def fit_baseline(vis, time, vis_mask, baseline_file):

    with h5.File(baseline_file, 'r') as f:
        bsl  = f['baseline'][:]
        bsl_time = f['time'][:]

    bsl_xx = interp1d(bsl_time, bsl, bounds_error=False,
                      fill_value='extrapolate')(time) + 20.
    #bsl_yy = interp1d(bsl_time, bsl, bounds_error=False,
    #                  fill_value='extrapolate')(time) + 20.

    bsl = np.zeros(vis.shape)

    #_vis_xx = np.matrix(vis[:, :, 0])
    for pi in range(2):
        #a_xx = np.zeros(vis.shape[1])
        p = []
        _xx = np.linspace(-1, 1, time.shape[0])
        for ii in range(vis.shape[1]):
            _vis_xx = vis[:, ii, pi]
            _msk_xx = vis_mask[:, ii, pi].astype('bool')
            _yy = (_vis_xx / bsl_xx)
            #_msk_xx += ~np.isfinite(_yy)
            _msk_xx += _yy == 0.

            if np.all(_msk_xx):
                p.append(np.zeros(4)[None, :])
            else: 
                p.append(np.polyfit(_xx[~_msk_xx], _yy[~_msk_xx], 3)[None, :])

            #_vis_xx = np.matrix(_vis_xx[~_msk_xx][:, None])
            #_bsl_xx = np.matrix( bsl_xx[~_msk_xx][:, None])
            #a_xx[ii] = np.array((_bsl_xx.T * _bsl_xx)**(-1) * _bsl_xx.T * _vis_xx )[0]

        p = np.concatenate(p, axis=0)

        fmask = np.sum(vis_mask[:, :, pi].astype('int'), axis=0) > 0.1 * vis.shape[0]
        fmask += np.all(p == 0, axis=1)
        xx = np.arange(vis.shape[1])
        #a_xx = interp1d(xx[~fmask], a_xx[~fmask],  axis=0,
        #        bounds_error=False, fill_value='extrapolate')(xx)
        if np.all(fmask):
            bsl[:, :, pi] = 0.
        else:
            p = interp1d(xx[~fmask], p[~fmask, :], axis=0,
                    bounds_error=False, fill_value='extrapolate' )(xx)
            for ii in range(vis.shape[1]):
                bsl[:, ii, pi] = np.poly1d(p[ii])(_xx) * bsl_xx
            
        #bsl[:, :, pi] =   a_xx[None, :] * bsl_xx[:, None]

        #a_xx = a_xx[None, ...]
        #x = x[:, None]
        ##a_poly = a_xx[..., 0] * x**2 + a_xx[..., 1] * x + a_xx[..., 2]
        #a_poly = a_xx[..., 0] * x + a_xx[..., 1]
        #bsl[:, :, pi] =   a_poly * bsl_xx[:, None]

    ##_vis_yy = np.matrix(vis[:, :, 1])
    #_vis_yy = vis[:, :, 1]
    #_vis_yy[vis_mask[:, :, 1]] = 0.
    #_vis_yy = np.matrix(_vis_yy)
    #bsl_yy  = np.matrix(bsl_yy[:, None])
    #a_yy = (bsl_yy.T * bsl_yy)**(-1) * bsl_yy.T * _vis_yy
    #bsl[:, :, 1] = np.array(a_yy) * np.array(bsl_yy)

    return bsl

def _output_baseline(file_list):

    ts = FAST_Timestream(file_list)
    ts.load_all()

    vis = ts['vis'][:].local_array
    vis_mask = ts['vis_mask'][:].local_array

    on = ts['ns_on'][:].local_array
    on = on.astype('bool')
    on = on[:, None, None, :]

    vis = np.ma.array(vis, mask=vis_mask)

    vis.mask += on

    baseline = np.ma.median(vis, axis=1)

    time = ts['sec1970'][:]

    return baseline, time

def output_baseline(file_list, output_name):

    baseline = []
    for fs in file_list:
        _baseline, time = _output_baseline(fs)
        baseline.append(_baseline[None, ...])
    baseline = np.ma.concatenate(baseline, axis=0)
    baseline = np.ma.median(baseline, axis=0)

    _m = np.ma.mean(baseline, axis=0)
    baseline -= _m[None, ...]
    baseline_median = np.ma.median(baseline, axis=(1, 2))

    l = 100
    baseline_pad = np.pad(baseline_median, l, 'symmetric')
    baseline = signal.medfilt(baseline_pad, [2*l+1,])[l:-l]

    with h5.File(output_name, 'w') as f:
        f['baseline'] = baseline
        f['time'] = time

