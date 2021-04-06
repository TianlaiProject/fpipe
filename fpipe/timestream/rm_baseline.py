from fpipe.timestream import timestream_task
from scipy import signal
from scipy.interpolate import interp1d

from fpipe.container.timestream import FAST_Timestream

from fpipe.timestream import bandpass_cal

import matplotlib.pyplot as plt

import numpy as np
import h5py as h5

from scipy import signal

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
        bsl = fit_baseline(vis[~on], time[~on], self.params['baseline_file'])

        vis[~on] -= bsl

def fit_baseline(vis, time, baseline_file):

    with h5.File(baseline_file, 'r') as f:
        bsl  = f['baseline'][:]
        bsl_time = f['time'][:]

    bsl_xx = interp1d(bsl_time, bsl, bounds_error=False,
                      fill_value='extrapolate')(time) + 20.
    bsl_yy = interp1d(bsl_time, bsl, bounds_error=False,
                      fill_value='extrapolate')(time) + 20.

    bsl = np.zeros(vis.shape)

    _vis_xx = np.matrix(vis[:, :, 0])
    bsl_xx  = np.matrix(bsl_xx[:, None])
    a_xx = (bsl_xx.T * bsl_xx)**(-1) * bsl_xx.T * _vis_xx
    bsl[:, :, 0] = np.array(a_xx) * np.array(bsl_xx)

    _vis_yy = np.matrix(vis[:, :, 1])
    bsl_yy  = np.matrix(bsl_yy[:, None])
    a_yy = (bsl_yy.T * bsl_yy)**(-1) * bsl_yy.T * _vis_yy
    bsl[:, :, 1] = np.array(a_yy) * np.array(bsl_yy)

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

    l = 50
    baseline_pad = np.pad(baseline_median, l, 'symmetric')
    baseline = signal.medfilt(baseline_pad, [2*l+1,])[l:-l]

    with h5.File(output_name, 'w') as f:
        f['baseline'] = baseline
        f['time'] = time

