import numpy as np
import h5py as h5
import gc
from fpipe.timestream import timestream_task
from fpipe.container.timestream import FAST_Timestream
import h5py
from astropy.time import Time
from tlpipe.utils.path_util import output_path as op_func
from caput import mpiutil
from caput import mpiarray
from scipy.signal import medfilt
from scipy.signal import lombscargle
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

class Bandpass_Cal(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'noise_on_time': 2,
            'bandpass_file' : None,
            #'bandpass_smooth' : 51,
            #'timevars_poly' : 4,
            #'Tnoise_file'   : None,
            #'T_sys' : None,
            #'plot_spec' : False,
            }

    prefix = 'bpcal_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        logger.info('Load bandpass from %s'%self.params['bandpass_file'])

        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        ts.redistribute('time')

        return super(Bandpass_Cal, self).process(ts)

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        with h5py.File(self.params['bandpass_file'], 'r') as f:
            _bandpass = f['bandpass'][:, bl[0]-1, :, :]
            _bandfreq = f['freq'][:]
            _bandpass = np.median(_bandpass, axis=0)
            msg = 'Apply bandpass to Feed %02d'%bl[0]
            logger.info(msg)

            bandpass_interp = interpolate.interp1d(_bandfreq, _bandpass, axis=0)
            bandpass = bandpass_interp(ts['freq'][:])

        # smooth the bandpass to remove some RFI
        #bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=kernel_size)
        #bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=kernel_size)

        bandpass[bandpass==0] = np.inf

        vis /= bandpass[None, ...]

class OutputBandpass(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'noise_on_time': 2,
            'bandpass_smooth' : 51,
            'timevars_poly' : 4,
            'Tnoise_file'   : None,
            'T_sys' : None,
            'bandpass_output' : '',
            }

    prefix = 'bpcal_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        self.bandpass = np.zeros([19, ts.freq.shape[0], 2])

        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        #print(self.output_files)
        logger.info('Output bandpass to file %s'%self.output_files)
        op_func(self.output_files, relative=False)
        with h5py.File(self.output_files[0], 'w') as f:
            f['bandpass'] = self.bandpass
            f['freq'] = ts.freq
            f['time'] = ts['sec1970'][:] #+ ts.attrs['sec1970']

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        on_t        = self.params['noise_on_time']
        kernel_size = self.params['bandpass_smooth']
        poly_order  = self.params['timevars_poly']
        Tnoise_file = self.params['Tnoise_file']
        if Tnoise_file is not None:
            freq      = ts['freq'][:]
            Tnoise_xx, Tnoise_yy =  get_ndmodel(Tnoise_file, freq, bl[0] - 1)

        vis1 = np.ma.array(vis.copy())
        vis1.mask = vis_mask.copy()

        if 'ns_on' in iter(ts.keys()):
            logger.info('Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1))
            if len(ts['ns_on'].shape) == 2:
                on = ts['ns_on'][:, gi].astype('bool')
            else:
                on = ts['ns_on'][:]
            #on = ts['ns_on'][:]
            vis1.mask[~on, ...] = True
            vis1.mask[on, ...] = False
        else:
            logger.info("No Noise Diode Mask info")

        vis1, on = get_Ncal(vis1, vis_mask, on, on_t)

        # take the median value of each channel as the bandpass
        bandpass = np.ma.median(vis1, axis=0)
        self.bandpass[bl[0] - 1, :, :] = bandpass


class Apply_EtaA(timestream_task.TimestreamTask):

    params_init = {
            'eta_A' : None,
            }

    prefix = 'etaA_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        with h5.File(self.params['eta_A'], 'r') as f:
            self.eta = f['eta'][:]
            self.eta_f = f['eta_f'][:]


        func = ts.bl_data_operate
        func(self.cal_etaA, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(Apply_EtaA, self).process(ts)

    def cal_etaA(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #eta_A = self.params['eta_A']
        #if eta_A is not None:
        #    print 'eta A cal'
        #    #factor = np.pi ** 2. / 4. / np.log(2.)
        #    vis /= eta_A[gi] #* factor

        bi = bl[0] - 1
        _eta = interp1d(self.eta_f, self.eta[bi], axis=-1)(ts.freq)
        _eta = _eta.T
        vis /= _eta[None, :, :]


def output_smoothed_bandpass(bandpass_path, bandpass_name, tnoise_path, output_path,
                    blk_st=1, blk_ed=7, bandpass_temp='%s_arcdrift%04d-%04d'):

    #bandpass_temp = '%s_arcdrift%04d-%04d'

    time_list = []
    bandpass_combined = []
    for block_id in range(blk_st, blk_ed+1):

        print(bandpass_temp%(bandpass_name, block_id, block_id))
        bandpass, freq, time = load_bandpass(bandpass_path,
            bandpass_temp%(bandpass_name, block_id, block_id) + '_%s.h5', tnoise_path)
        time_list.append(time)
        bandpass = np.ma.array(bandpass, mask=False)
        #bandpass_smooth = medfilt(bandpass, [1, 201, 1])
        #bandpass_smooth = np.ma.array(bandpass_smooth, mask=False)
        bandpass_smooth = smooth_bandpass(bandpass.copy(), axis=1)
        bandpass_combined.append(bandpass_smooth.copy())

    bandpass_combined = np.array(bandpass_combined)

    time_list = np.concatenate(time_list)

    output_name = 'bandpass_%s.h5'%(bandpass_name)
    with h5.File(output_path + output_name, 'w') as f:
        f['bandpass'] = bandpass_combined
        f['freq'] = freq
        f['time'] = time_list

def load_bandpass(bandpass_path, bandpass_temp, tnoise_path=None,
                  freq_bands=['1050-1150MHz', '1150-1250MHz', '1250-1450MHz']):
    freq = []
    bandpass = []
    for i, f in enumerate(freq_bands):
        with h5.File(bandpass_path + bandpass_temp%f, 'r') as f:
            bandpass.append(f['bandpass'][:])
            freq.append(f['freq'][:])
            time = f['time'][:]
    bandpass = np.concatenate(bandpass, axis=1)
    freq = np.concatenate(freq)

    if tnoise_path is not None:

        #with h5.File(tnoise_path + 'Tnosie_M_low.h5', 'r') as f:
        with h5.File(tnoise_path, 'r') as f:
            #print f.keys()
            tnoise_md = f['Tnoise'][:]
            tnoise_md_freq = f['freq'][:]
            tnoise_interp = interp1d(tnoise_md_freq, tnoise_md, axis=0)

    norm = tnoise_interp(freq)
    norm = np.rollaxis(norm, -1)

    bandpass /= norm

    #print time
    return bandpass, freq, time

def smooth_bandpass(bandpass, axis=0):

    kernel = [1,] * bandpass.ndim
    kernel[axis] = 501

    _bandpass_smooth = medfilt(bandpass, kernel)
    _bandpass_flat = bandpass - _bandpass_smooth

    _bandpass_flat = np.ma.array(_bandpass_flat, mask=False)
    _s = [slice(None), ] * bandpass.ndim
    _s[axis] = None
    _s = tuple(_s)
    for i in range(5):
        std = np.ma.std(_bandpass_flat, axis=axis)
        msk = np.abs(_bandpass_flat) - 3 * std[_s] > 0
        _bandpass_flat.mask += msk

    msk = _bandpass_flat.mask
    bandpass[msk] = 0
    _bandpass_smooth[~msk] = 0

    bandpass += _bandpass_smooth

    b, a = signal.butter( 3, 0.05 * 0.209 )

    return signal.filtfilt(b, a, bandpass, axis=axis)

def ND_statics(file_list, gi=0, on_t=1):

    ts = FAST_Timestream(file_list)
    ts.load_all()

    vis = ts['vis'][..., gi].local_array
    vis_mask = ts['vis_mask'][..., gi].local_array

    on = ts['ns_on'][:, gi].local_array
    on = on.astype('bool')

    vis = np.ma.array(vis, mask=vis_mask.copy())

    vis.mask[~on, ...] = True
    vis.mask[on, ...] = False

    vis, on = get_Ncal(vis, vis_mask, on, on_t)

    time = ts['sec1970'][:]
    #time -= time[0]

    freq = ts['freq'][:]

    return vis, time[on], freq

def est_gtgnu_onefeed(file_list, smooth=(3, 51), gi=0, Tnoise_file=None,
        interp_mask = True):

    nd = []
    nd_mask = []
    freq = []
    for ii, _file_list in enumerate(file_list):
        #print "Freq Band %d; "%ii,
        _nd, time, _freq = ND_statics(_file_list, gi=gi)
        nd.append(_nd.data)
        nd_mask.append(_nd.mask)
        freq.append(_freq)
    #print

    nd = np.ma.concatenate(nd, axis=1)
    nd.mask = np.ma.concatenate(nd_mask, axis=1)
    freq = np.concatenate(freq)

    if Tnoise_file is not None:
        Tnoise_xx, Tnoise_yy = get_ndmodel(Tnoise_file, freq, gi)
        nd[:, :, 0] /= Tnoise_xx[None, :]
        nd[:, :, 1] /= Tnoise_yy[None, :]

    _m = np.ma.mean(nd)
    _s = np.ma.std(nd)

    #nd.mask += np.abs(nd - _m) - 2 * _s > 0

    bad_freq = np.sum(nd.mask, axis=(0, 2))
    bad_freq = bad_freq > 0.7 * freq.shape[0]
    nd.mask[:, bad_freq, :] = True

    if interp_mask:

        shp = nd.shape[:2]
        grid_x, grid_y = np.mgrid[0:shp[0], 0:shp[1]]
        xi = [grid_x.flatten()[:, None], grid_y.flatten()[:, None]]
        xi = np.concatenate(xi, axis=1)

        # xx
        msk = nd.mask[:, :, 0].flatten()
        values = nd[:, :, 0].flatten()[~msk]
        points = xi[~msk, :]
        if len(points) == 0:
            logger.info('all mask')
            nd_xx = np.ones(shp)
        else:
            nd_xx = griddata(points, values, xi, method='nearest')
            nd_xx.shape = shp
            nd_xx = median_filter(nd_xx, smooth)

        # yy
        msk = nd.mask[:, :, 1].flatten()
        values = nd[:, :, 1].flatten()[~msk]
        points = xi[~msk, :]
        if len(points) == 0:
            logger.info('all mask')
            nd_yy = np.ones(shp)
        else:
            nd_yy = griddata(points, values, xi, method='nearest')
            nd_yy.shape = shp
            nd_yy = median_filter(nd_yy, smooth)

        nd =  np.concatenate([nd_xx[:, :, None], nd_yy[:, :, None]], axis=-1)

    return nd, time, freq

#def output_gtgnu(file_list, output_name, smooth=(7, 51), Tnoise_file='', 
def output_gtgnu(file_list, output_name, smooth=(1, 1), Tnoise_file='', 
        interp_mask=True):

    nd_list = []
    for gi in range(19):
        #print "Feed %02d"%gi
        nd, time, freq = est_gtgnu_onefeed(file_list, smooth=smooth, gi=gi,
                                           Tnoise_file = Tnoise_file,
                                           interp_mask = interp_mask)
        nd_list.append(nd[..., None])

    nd = np.ma.concatenate(nd_list, axis=-1)

    #print nd.shape

    with h5.File(output_name, 'w') as f:
        f['gtgnu'] = nd
        f['mask']  = nd.mask
        f['freq']  = freq
        f['time']  = time

def get_Ncal(vis, vis_mask, on, on_t, off_t=1, cut_end=True):

    if cut_end:
        # remove the cal at the beginning/ending
        on[ :on_t] = False
        on[-on_t:] = False
    if on_t == 2:
        # noise cal may have half missing, because of the RFI flagging
        # remove them
        on  = (np.roll(on, 1) * on) + (np.roll(on, -1) * on)
        # use one time stamp before, one after as cal off
        off = (np.roll(on, 1) + np.roll(on, -1)) ^ on
        vis1_on  = vis[on, ...].data
        vis1_off = vis[off, ...].data
        mask     = vis_mask[off, ...]
    elif on_t == 1:
        off = np.zeros_like(on, dtype='bool')
        for i in range(off_t):
            off += np.roll(on, i+1) + np.roll(on, -i-1)
        vis1_on  = vis[on, ...].data
        vis1_off = vis[off, ...].data
        # because the nearby time are masked, we use futher ones.
        #mask_off = np.zeros_like(on, dtype='bool')
        #for i in range(off_t):
        #    mask_off += np.roll(on, i+1) + np.roll(on, -i-1)
        #mask     = vis_mask[mask_off, ...]
        mask_off = vis_mask[off, ...]
        mask_on  = vis_mask[on,  ...]

        vis_shp = vis1_off.shape
        vis1_off = vis1_off.reshape( (-1, 2 * off_t) + vis_shp[1:] )
        vis1_off = np.ma.mean(vis1_off, axis=1)

        mask_off  = mask_off.reshape((-1, 2 * off_t) + vis_shp[1:])
        mask_off  = np.sum(mask_off, axis=1).astype('bool')
        mask = mask_on + mask_off
    else:
        raise
    
    vis1 = vis1_on - vis1_off
    vis1 = np.ma.array(vis1)
    vis1.mask = mask

    if on_t > 1:
        vis_shp = vis1.shape
        vis1 = vis1.reshape((-1, on_t) + vis_shp[1:])
        vis1 = vis1 + vis1[:, ::-1, ...]
        vis1.shape = vis_shp

    return vis1, on


def get_ndmodel(Tnoise_file, freq, bi):

    with h5.File(Tnoise_file, 'r') as f:
        Tnoise_xx = f['Tnoise'][:, 0, bi]
        Tnoise_yy = f['Tnoise'][:, 1, bi]
        Tnoise_f = f['freq'][:]
    Tnoise_xx = gaussian_filter1d( Tnoise_xx, sigma=10 )
    Tnoise_yy = gaussian_filter1d( Tnoise_yy, sigma=10 )

    #freq      = ts['freq'][:]
    Tnoise_xx = interpolate.interp1d(Tnoise_f, Tnoise_xx,
            bounds_error=False, fill_value=0)(freq)
    Tnoise_yy = interpolate.interp1d(Tnoise_f, Tnoise_yy,
            bounds_error=False, fill_value=0)(freq)

    return Tnoise_xx, Tnoise_yy

# =========================================================================
# some old code
# =========================================================================
# |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 
# V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V  V 

class Normal_Tsys(timestream_task.TimestreamTask):

    params_init = {
            'T_sys' : 20. ,
            'relative_gain' : None,
            'eta_A' : None,
            'timevars_poly' : 6,
            'noise_on_time': 1,
            'sub_mean' : True,
            }

    prefix = 'tsyscal_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        func = ts.bl_data_operate
        func(self.cal_tsys, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(Normal_Tsys, self).process(ts)

    def cal_tsys(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        poly_order  = self.params['timevars_poly']
        on_t = self.params['noise_on_time']
        if 'ns_on' in iter(ts.keys()):
            print('Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1))
            if len(ts['ns_on'].shape) == 2:
                on = ts['ns_on'][:, gi].astype('bool')
            else:
                on = ts['ns_on'][:]
        else:
            print("No Noise Diode Mask info")
            on = np.zeros(vis.shape[0]).astype('bool')

        vis1 = vis.copy()
        vis1 = np.ma.array(vis1)
        vis1.mask  = vis_mask.copy()
        vis1[vis1==0] = np.ma.masked
        vis1.mask[~on, ...] = True
        vis1.mask[on, ...] = False

        vis1, on = get_Ncal(vis1, vis_mask, on, on_t)
        bandpass = np.ma.median(vis1, axis=0)
        # smooth the bandpass to remove some RFI
        bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=101)
        bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=101)
        bandpass = np.ma.filled(bandpass, 0)
        bandpass[bandpass==0] = np.inf

        time  = ts['sec1970'][:]
        #time -= time[0]
        #time /= time.max()
        #vis1 /= np.ma.median(vis1, axis=(0,1))[None, None, :]
        vis1 /= bandpass[None, ...]
        vis1[vis1 == 0.] = np.ma.masked
        vis1 = np.ma.median(vis1, axis=1)
        #poly_xx, poly_yy = polyfit_timedrift(vis1, time, on, poly_order)
        poly_xx, poly_yy = medfilt_timedrift(vis1, time, on)
        vis[..., 0] /= poly_xx[:, None]
        vis[..., 1] /= poly_yy[:, None]

        del vis1

        vis1 = vis.copy()
        vis1 = np.ma.array(vis1)
        vis1.mask  = vis_mask.copy()
        vis1[vis1==0] = np.ma.masked

        T_sys = self.params['T_sys']
        if T_sys is not None:
            print("Norm. T_sys to %f K"%T_sys)
            vis /= np.ma.median(vis1[~on, ...], axis=(0, 1))[None, None, :]
            vis *= T_sys
            if self.params['sub_mean']:
                vis -= T_sys

        relative_gain = self.params['relative_gain']
        if relative_gain is not None:
            print("relative gain cal %d (%f %f)"%((gi,) + tuple(relative_gain[gi])))
            vis *= relative_gain[gi, :][..., :]

        eta_A = self.params['eta_A']
        if eta_A is not None:
            print('eta A cal')
            #factor = np.pi ** 2. / 4. / np.log(2.)
            vis /= eta_A[gi] #* factor

        del vis1

def medfilt_timedrift(vis1, time, on, kernel_size=31, fill_value = 'extrapolate'):

    good_xx = ~vis1.mask[:, 0]
    good_yy = ~vis1.mask[:, 1]

    nd_xx = medfilt(vis1[:, 0][good_xx], kernel_size=(kernel_size))
    nd_yy = medfilt(vis1[:, 1][good_yy], kernel_size=(kernel_size))

    medfilt_xx = interpolate.interp1d(time[on][good_xx], nd_xx, kind='linear', 
            bounds_error=False, fill_value=fill_value)(time)
    medfilt_yy = interpolate.interp1d(time[on][good_yy], nd_yy, kind='linear', 
            bounds_error=False, fill_value=fill_value)(time)

    return medfilt_xx, medfilt_yy

def polyfit_timedrift(vis1, time, on, poly_order, poly_len=2048):

    vis_st = 0
    poly_xx = []
    poly_yy = []
    for st in np.arange(0, time.shape[0], poly_len):
        ed  = st + poly_len
        _time = time[st:ed].copy()
        _time -= _time[0]
        _time /= _time[-1]
        _time_on = _time[on[st:ed]]
        vis_ed = vis_st + _time_on.shape[0]
        _vis1 = vis1[vis_st:vis_ed, ...]
        vis_st = vis_ed
        _good = ~_vis1.mask
        vis1_poly_xx = np.poly1d(np.polyfit(_time_on[_good[:,0]], 
                                            _vis1[:, 0][_good[:,0]],
                                            poly_order))
        vis1_poly_yy = np.poly1d(np.polyfit(_time_on[_good[:,1]], 
                                            _vis1[:, 1][_good[:,1]], 
                                            poly_order))
    
        poly_xx.append(vis1_poly_xx(_time))
        poly_yy.append(vis1_poly_yy(_time))

    poly_xx = np.concatenate(poly_xx)
    poly_yy = np.concatenate(poly_yy)
    return poly_xx, poly_yy

class Bandpass_Cal_old(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'noise_on_time': 2,
            'bandpass_smooth' : 51,
            'timevars_poly' : 4,
            'Tnoise_file'   : None,
            'T_sys' : None,
            'plot_spec' : False,
            }

    prefix = 'bpcalold_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(Bandpass_Cal, self).process(ts)

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        on_t        = self.params['noise_on_time']
        kernel_size = self.params['bandpass_smooth']
        poly_order  = self.params['timevars_poly']
        Tnoise_file = self.params['Tnoise_file']
        plot_spec   = self.params['plot_spec']
        if Tnoise_file is not None:
            with h5py.File(Tnoise_file, 'r') as f:
                Tnoise_xx = f['Tnoise'][:, 0, bl[0] - 1]
                Tnoise_yy = f['Tnoise'][:, 1, bl[0] - 1]
                Tnoise_f = f['freq'][:]
            Tnoise_xx = gaussian_filter1d( Tnoise_xx, sigma=10 )
            Tnoise_yy = gaussian_filter1d( Tnoise_yy, sigma=10 )

            freq      = ts['freq'][:]
            Tnoise_xx = interpolate.interp1d(Tnoise_f, Tnoise_xx, 
                    bounds_error=False, fill_value=0)(freq)
            Tnoise_yy = interpolate.interp1d(Tnoise_f, Tnoise_yy, 
                    bounds_error=False, fill_value=0)(freq)

        vis1 = np.ma.array(vis.copy())
        vis1.mask = vis_mask.copy()

        if 'ns_on' in iter(ts.keys()):
            print('Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1))
            if len(ts['ns_on'].shape) == 2:
                on = ts['ns_on'][:, gi].astype('bool')
            else:
                on = ts['ns_on'][:]
            #on = ts['ns_on'][:]
            vis1.mask[~on, ...] = True
            vis1.mask[on, ...] = False
            #vis1.mask[:, bad_freq] = True
        else:
            print("No Noise Diode Mask info")

        vis1, on = get_Ncal(vis1, vis_mask, on, on_t)

        # take the median value of each channel as the bandpass
        bandpass = np.ma.median(vis1, axis=0)
        if plot_spec:
            fig = plt.figure(figsize=(6, 4))
            ax  = fig.add_axes([0.06, 0.1, 0.90, 0.8])
            ax.plot(ts['freq'][:], bandpass[:, 0], 'r', label='bandpass X')
            ax.plot(ts['freq'][:], bandpass[:, 1], 'b', label='bandpass Y')

        # smooth the bandpass to remove some RFI
        bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=kernel_size)
        bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=kernel_size)

        if plot_spec:
            ax.plot(ts['freq'][:], bandpass[:, 0], 'w')
            ax.plot(ts['freq'][:], bandpass[:, 1], 'w')
            ax.legend()
            ax.set_ylim(ymin=2, ymax=12)
            ax.set_xlim(xmin=ts['freq'][:].min(),xmax=ts['freq'][:].max())

        bandpass = np.ma.filled(bandpass, 0)
        bandpass[bandpass==0] = np.inf

        vis /= bandpass[None, ...]

        if plot_spec:
            fig = plt.figure(figsize=(12, 4))
            ax  = fig.add_axes([0.06, 0.1, 0.90, 0.8])
            ax.plot(ts['freq'][:], np.ma.median(vis, axis=0)[:, 0], 'r')
            ax.plot(ts['freq'][:], np.ma.median(vis, axis=0)[:, 1], 'b')

        # get the time var
        time  = ts['sec1970'][:]
        time -= time[0]
        time_on = time[on]
        #vis1 /= np.ma.median(vis1, axis=(0,1))[None, None, :]
        vis1 /= bandpass[None, ...]
        vis1 = np.ma.median(vis1, axis=1)
        good = ~vis1.mask
        vis1_poly_xx = np.poly1d(np.polyfit(time_on[good[:,0]], 
                                            vis1[:, 0][good[:,0]],
                                            poly_order))
        vis1_poly_yy = np.poly1d(np.polyfit(time_on[good[:,1]], 
                                            vis1[:, 1][good[:,1]], 
                                            poly_order))

        if Tnoise_file is not None:
            vis[..., 0] *= Tnoise_xx[None, :]
            vis[..., 1] *= Tnoise_yy[None, :]

        if plot_spec:
            ax.plot(ts['freq'][:], np.ma.median(vis, axis=0)[:, 0], 'm')
            ax.plot(ts['freq'][:], np.ma.median(vis, axis=0)[:, 1], 'g')
            ax.set_xlim(xmin=ts['freq'][:].min(),xmax=ts['freq'][:].max())
            ax.set_ylim(ymin=15, ymax=25)
            #ax.set_ylim(ymin=15, ymax=80)

        if self.params['T_sys'] is not None:
            T_sys = self.params['T_sys']
            print("Norm. T_sys to %f K"%T_sys)
            vis /= np.ma.median(vis[~on, ...], axis=(0, 1))[None, None, :]
            vis *= T_sys
