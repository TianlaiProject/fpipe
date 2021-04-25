"""Plot waterfall images.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

# import pytz
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from fpipe.timestream import timestream_task
from fpipe.timestream import bandpass_cal as bp
#from tlpipe.container.raw_timestream import RawTimestream
#from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
from tlpipe.utils import hist_eq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator

from astropy.time import Time

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy import special

from tlpipe.rfi import interpolate
from tlpipe.rfi import gaussian_filter

from fpipe.container.timestream import FAST_Timestream
from fpipe.utils import axes_utils

import logging

logger = logging.getLogger(__name__)

# tz = pytz.timezone('Asia/Shanghai')

def load_ts(file_list):

    ts = FAST_Timestream(file_list)
    ts.load_all()

    vis = ts['vis'][:].local_array
    vis_mask = ts['vis_mask'][:].local_array

    on = ts['ns_on'][:].local_array
    on = on.astype('bool')

    vis = np.ma.array(vis, mask=vis_mask)

    vis.mask += on[:, None, None, :]

    time = ts['sec1970'][:].local_array

    freq = ts['freq'][:]

    return vis, time, freq


def plot_wf(file_list, title='', pol=0, vmax=None, vmin=None, output=None):

    vis  = []
    mask = []
    freq = []
    for ii, _file_list in enumerate(file_list):
        _vis, time, _freq = load_ts(_file_list)
        vis.append(_vis)
        mask.append(_vis.mask)
        freq.append(_freq)

    vis = np.concatenate(vis, axis=1)
    mask= np.concatenate(mask, axis=1)
    freq= np.concatenate(freq)

    time -= time[0]
    time /= 3600.
    freq /= 1.e3

    _m = np.ma.mean(vis)
    _s = np.ma.std(vis)
    if vmax is None: vmax= _m + 2 * _s
    if vmin is None: vmin= _m - 2 * _s

    vis = np.ma.array(vis, mask=mask)

    fig, axes = axes_utils.setup_axes(5, 4, colorbar=True)

    for bi in range(19):

        print "Feed%02d "%bi,

        i = bi/4
        j = bi%4

        ax = axes[bi]
        im = ax.pcolormesh(time, freq, vis[:, :, pol, bi].T, vmax=vmax, vmin=vmin)

        if i != 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time [hr]')
        if j != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$\nu$ [GHz]')


    print
    cax = axes[-1]
    fig.colorbar(im, cax=cax, orientation='horizontal')
    cax.set_xlabel(r'$T$ [K]')
    cax.set_title(title)

    print 'output image'

    if output is not None:
        fig.savefig(output, formate='png', dpi=300)

    fig.clf()

def plot_wf_onefeed(file_list, bi=0, pi=0, vmin=None, vmax=None, output=None):

    fig = plt.figure(figsize=(12, 4))
    ax  = fig.add_axes([0.07, 0.15, 0.85, 0.80])
    cax = fig.add_axes([0.925, 0.15, 0.018, 0.80])

    xlabel = None
    xmin =  1.e10
    xmax = -1.e10
    ymin =  1.e10
    ymax = -1.e10
    for tblock in file_list:
        for fblock in tblock:
            ts = FAST_Timestream(fblock)
            ts.load_all()

            vis = ts['vis'][:, :, pi, bi].local_array
            vis_mask = ts['vis_mask'][:, :, pi, bi].local_array
            on = ts['ns_on'][:, bi].local_array
            on = on.astype('bool')

            vis = np.ma.array(vis, mask=vis_mask)
            vis.mask += on[:, None]

            time = ts['sec1970'][:].local_array
            freq = ts['freq'][:] * 1.e-3

            x_axis = [ datetime.utcfromtimestamp(s) for s in time]
            if xlabel is None:
                x_label = 'UTC %s' % x_axis[0].date()
            x_axis = mdates.date2num(x_axis)

            im = ax.pcolormesh(x_axis, freq, vis.T, vmin=vmin, vmax=vmax)
            if x_axis.min() < xmin: xmin=x_axis.min()
            if x_axis.max() > xmax: xmax=x_axis.max()
            if freq.min() < ymin: ymin=freq.min()
            if freq.max() > ymax: ymax=freq.max()

            del vis, ts
            gc.collect()

    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency [GHz]')
    fig.autofmt_xdate()
    fig.colorbar(im, ax=ax, cax=cax)
    cax.set_ylabel('T[K]')

    if output is not None:
        fig.savefig(output, dpi=200)
    fig.clf()

class PlotMeerKAT(timestream_task.TimestreamTask):

    params_init = {
            'main_data' : 'vis',
            'corr' : 'auto',
            'flag_mask' : True,
            'flag_ns'   : False,
            'flag_raw'  : True,
            're_scale'  : None,
            'vmin'      : None,
            'vmax'      : None,
            'xmin'      : None,
            'xmax'      : None,
            'ymin'      : None,
            'ymax'      : None,
            'fig_name'  : 'wf/',
            'bad_freq_list' : None,
            'bad_time_list' : None,
            'show'          : None,
            'plot_index'    : False,
            'plot_ra'       : False,
            'unit' : r'${\rm T}\,[{\rm K}]$', 
            }
    prefix = 'pkat_'

    def __init__(self, parameter_file_or_dict=None, feedback=0):
        super(PlotMeerKAT, self).__init__(parameter_file_or_dict, feedback)
        self.feedback = feedback

    def process(self, ts):

        ts.main_data_name = self.params['main_data']

        #ts.redistribute('baseline')

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        if bad_time_list is not None:
            print "Mask bad time"
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        if 'flags' in ts.iterkeys() and self.params['flag_raw']:
            print 'apply raw flags'
            ts.vis_mask[:] += ts['flags'][:]

        func(self.plot, full_data=True, show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(PlotMeerKAT, self).process(ts)


    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']
        fig_prefix = self.params['fig_name']
        main_data = self.params['main_data']

        vis1 = np.ma.array(vis)
        if flag_mask:
            vis1.mask  = vis_mask
            logger.info('share mask with pols')
            vis1.mask += np.sum(vis1.mask, axis=-1).astype('bool')[:, :, None]
            freq = ts['freq'][:] - 1420.
            local_hi = np.abs(freq) < 5
            vis1.mask[:, local_hi, ...] = False
        else:
            vis1.mask = np.zeros(vis1.shape, dtype='bool')


        if flag_ns:
            if 'ns_on' in ts.iterkeys():
                print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
                #vis1 = vis.copy()
                #on = np.where(ts['ns_on'][:])[0]
                #vis1[on] = complex(np.nan, np.nan)
                on = ts['ns_on'][:, gi]
                vis1.mask[on, ...] = True
            else:
                print "No Noise Diode Mask info"

        if self.params['plot_index']:
            y_axis = np.arange(ts.freq.shape[0])
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            x_label = 'time index'
        else:
            y_axis = ts.freq[:] * 1.e-3
            y_label = r'$\nu$ / GHz'
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                x_label = 'R.A.' 
                print 'RA range [%6.2f, %6.2f]'%( x_axis.min(), x_axis.max())
            else:
                time = ts['sec1970'] #+ 
                try:
                    time += ts.attrs['sec1970']
                except KeyError:
                    pass
                x_axis = [ datetime.utcfromtimestamp(s) for s in time]
                x_label = 'UTC %s' % x_axis[0].date()
                # convert datetime objects to the correct format for 
                # matplotlib to work with
                x_axis = mdates.date2num(x_axis)
            if xmin is not None:
                xmin = x_axis[xmin]
            if xmax is not None:
                xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        #if np.any(bad_time):
        #    good_time_st = np.argwhere(~bad_time)[ 0, 0]
        #    good_time_ed = np.argwhere(~bad_time)[-1, 0]
        #    vis1 = vis1[good_time_st:good_time_ed, ...]
        #    x_axis = x_axis[good_time_st:good_time_ed]

        #if np.any(bad_freq):
        #    good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
        #    good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
        #    vis1 = vis1[:, good_freq_st:good_freq_ed, ...]
        #    y_axis = y_axis[good_freq_st:good_freq_ed]


        if re_scale is not None:
            mean = np.ma.mean(vis1)
            std  = np.ma.std(vis1)
            print mean, std
            vmax = mean + re_scale * std
            vmin = mean - re_scale * std
        else:
            vmax = self.params['vmax']
            vmin = self.params['vmin']

        fig  = plt.figure(figsize=(10, 6))
        axhh = fig.add_axes([0.10, 0.52, 0.75, 0.40])
        axvv = fig.add_axes([0.10, 0.10, 0.75, 0.40])
        cax  = fig.add_axes([0.86, 0.20, 0.02, 0.60])

        im = axhh.pcolormesh(x_axis, y_axis, vis1[:,:,0].T, vmax=vmax, vmin=vmin)
        im = axvv.pcolormesh(x_axis, y_axis, vis1[:,:,1].T, vmax=vmax, vmin=vmin)

        fig.colorbar(im, cax=cax, ax=axvv)

        axhh.set_title('Antenna M%03d'%(bl[0] - 1))

        # format datetime string
        # date_format = mdates.DateFormatter('%y/%m/%d %H:%M')
        date_format = mdates.DateFormatter('%H:%M')
        # date_format = mdates.DateFormatter('%H:%M', tz=pytz.timezone('Asia/Shanghai'))

        ## reduce the number of tick locators
        #locator = MaxNLocator(nbins=6)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axhh.xaxis.set_major_formatter(date_format)
            #axhh.xaxis.set_major_locator(MultipleLocator(0.5))
        axhh.set_xticklabels([])
        axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        if ymin is None: ymin = y_axis[0]
        if ymax is None: ymax = y_axis[-1]
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axvv.xaxis.set_major_formatter(date_format)
        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            fig.autofmt_xdate()

        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm time median}$')
        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm noise\, cal}$')
        cax.set_ylabel(self.params['unit'])

        if fig_prefix is not None:
            fig_name = '%s_%s_m%03d_x_m%03d.png' % (fig_prefix, main_data,
                                                    bl[0]-1,    bl[1]-1)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)
        if self.params['show'] is not None:
            if self.params['show'] == 'all':
                plt.show()
            elif self.params['show'] == bl[0]-1:
                plt.show()
        plt.close()

class PlotTimeStream(timestream_task.TimestreamTask):

    params_init = {
            'main_data' : 'vis',
            'corr' : 'auto',
            'flag_mask' : True,
            'flag_ns'   : False,
            're_scale'  : None,
            'vmin'      : None,
            'vmax'      : None,
            'xmin'      : None,
            'xmax'      : None,
            'ymin'      : None,
            'ymax'      : None,
            'fig_name'  : 'wf/',
            'bad_freq_list' : None,
            'bad_time_list' : None,
            'show'          : None,
            'plot_index'    : False,
            'plot_ra'       : False,
            'legend_title' : '', 
            'nvss_cat' : None,
            }
    prefix = 'ptsbase_'

    def __init__(self, parameter_file_or_dict=None, feedback=0):
        super(PlotTimeStream, self).__init__(parameter_file_or_dict, feedback)
        self.feedback = feedback

    def process(self, ts):

        fig  = plt.figure(figsize=(8, 6))
        self.axhh = fig.add_axes([0.11, 0.52, 0.83, 0.40])
        self.axvv = fig.add_axes([0.11, 0.10, 0.83, 0.40])
        self.fig  = fig
        self.xmin =  1.e19
        self.xmax = -1.e19

        ts.main_data_name = self.params['main_data']

        ts.redistribute('baseline')

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        if bad_time_list is not None:
            print "Mask bad time"
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        if self.params['nvss_cat'] is not None:
            self.nvss_range = []


        func(self.plot, full_data=True, show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        self.write_output(None)
        #return super(PlotTimeStream, self).process(ts)

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        pass

    def write_output(self, output):

        fig_prefix = self.params['output_files'][0]
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        if self.params['nvss_cat'] is not None:
            nvss_cat = get_nvss_radec(self.params['nvss_cat'] , self.nvss_range)
            _ymin, _ymax = axhh.get_ylim()
            for ii in range(nvss_cat.shape[0]):
                axhh.axvline(nvss_cat['RA'][ii], 0, 1, 
                        color='k', linestyle='--', linewidth=0.8)
                axvv.axvline(nvss_cat['RA'][ii], 0, 1, 
                        color='k', linestyle='--', linewidth=0.8)

        x_label = self.x_label

        date_format = mdates.DateFormatter('%H:%M')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axhh.xaxis.set_major_formatter(date_format)
            #axhh.xaxis.set_major_locator(MultipleLocator(1./6.))
            axhh.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axhh.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        axhh.set_xticklabels([])
        #axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        #axhh.set_ylabel('HH Polarization')
        #if xmin is None: xmin = x_axis[0]
        #if xmax is None: xmax = x_axis[-1]
        xmin = self.xmin
        xmax = self.xmax
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')
        axhh.legend(title=self.params['legend_title'], ncol=6)

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axvv.xaxis.set_major_formatter(date_format)
            #axvv.xaxis.set_major_locator(MultipleLocator(1./6.))
            axvv.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axvv.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        axhh.set_xticklabels([])
        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        #axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        #axvv.set_ylabel('VV Polarization')
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            fig.autofmt_xdate()

        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm time median}$')
        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm noise\, cal}$')
        #cax.set_ylabel(self.params['unit'])

def get_nvss_radec(nvss_path, nvss_range):

    import astropy.io.fits as pyfits

    hdulist = pyfits.open(nvss_path)
    data = hdulist[1].data

    _sel = np.zeros(data['RA'].shape[0], dtype ='bool')
    for _nvss_range in nvss_range:
        #print _nvss_range

        ra_min, ra_max, dec_min, dec_max = _nvss_range

        sel  = data['RA'] > ra_min
        sel *= data['RA'] < ra_max
        sel *= data['DEC'] > dec_min
        sel *= data['DEC'] < dec_max

        #print np.sum(sel)

        _sel = _sel + sel

    hdulist.close()

    return data[_sel]

def sub_ortho_poly(vis, time, mask, n):

    logger.debug('sub. mean')
    
    window = mask
    x = time
    
    upbroad = (slice(None), slice(None)) + (None, ) * (window.ndim - 1)
    window = window[None, ...]
    
    x_mid = (x.max() + x.min())/2.
    x_range = (x.max() - x.min()) /2.
    x = (x - x_mid) / x_range
    
    n = np.arange(n)[:, None]
    x = x[None, :]
    polys = special.eval_legendre(n, x, out=None)
    polys = polys[upbroad] * window

    for ii in range(n.shape[0]):
        for jj in range(ii):
            amp = np.sum(polys[ii, ...] * polys[jj, ...], axis=0)
            polys[ii, ...] -= amp[None, ...] * polys[jj, ...]
            
        norm  = np.sqrt(np.sum(polys[ii] ** 2, axis=0))
        norm[norm==0] = np.inf
        polys[ii] /= norm[None, ...]
    
    amp = np.sum(polys * vis[None, ...], 1)
    vis_fit = np.sum(amp[:, None, ...] * polys, 0)
    #vis -= vis_fit
    return vis_fit

class PlotVvsTime(PlotTimeStream):

    prefix = 'pts_'

    params_init = {
            'rm_mean' : False,
            'rm_slop' : False,
            'rm_point_sources' : False,
            't_block' : None
            }

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #vis = np.abs(vis)
        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        vis1 = np.ma.array(vis)
        if flag_mask:
            vis1.mask = vis_mask
        else:
            vis1.mask = np.zeros(vis1.shape, dtype='bool')

        if flag_ns:
            if 'ns_on' in ts.iterkeys():
                print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
                #vis1 = vis.copy()
                #on = np.where(ts['ns_on'][:])[0]
                #vis1[on] = complex(np.nan, np.nan)
                if len(ts['ns_on'].shape) == 2:
                    on = ts['ns_on'][:, gi].astype('bool')
                else:
                    on = ts['ns_on'][:]
                #on = ts['ns_on'][:]
                vis1.mask[on, ...] = True
            else:
                print "No Noise Diode Mask info"

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            #x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
            #self.x_label = '%s' % x_axis[0].date()
            #x_axis = mdates.date2num(x_axis)
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                self.x_label = 'R.A.' 
                print 'RA range %f - %f'%(x_axis.min(), x_axis.max())
            else:
                time = ts['sec1970'] + ts.attrs['sec1970']
                x_axis = [ datetime.fromtimestamp(s) for s in time]
                self.x_label = '%s UTC' % x_axis[0].date()
                x_axis = mdates.date2num(x_axis)
            #if xmin is not None:
            #    xmin = x_axis[xmin]
            #if xmax is not None:
            #    xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        vis1 = vis1[good_time_st:good_time_ed, ...]
        x_axis = x_axis[good_time_st:good_time_ed]

        good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
        good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
        vis1 = vis1[:, good_freq_st:good_freq_ed, ...]

        self._plot(vis1, x_axis, xmin, xmax, gi, bl, ts)

    def _plot(self, vis1, x_axis, xmin, xmax, gi, bl, ts):

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        #axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '.', label = label)
        #axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '.')
        if len(vis1.shape) == 3:
            vis1 = np.ma.mean(vis1, axis=1)

        time = np.arange(x_axis.shape[0])
        _n = 0
        t_block = self.params['t_block']
        if   self.params['rm_slop']: _n = 2
        elif self.params['rm_mean']: _n = 1
        if t_block is None: t_block = len(time)
        for ii in range(0, len(time), t_block):
            st = ii
            ed = ii + t_block
            _vis1 = vis1[st:ed, ...]
            _time = time[st:ed]
            bg  = gaussian_filter.GaussianFilter(
                        interpolate.Interpolate(_vis1, _vis1.mask).fit(), 
                        time_kernal_size=0.5, freq_kernal_size=1, 
                        filter_direction = ('time', )).fit()
            if _n != 0:
                _vis1 -= sub_ortho_poly(bg, _time, ~_vis1.mask, _n)
            if self.params['rm_point_sources']:
                bg  = gaussian_filter.GaussianFilter(
                        interpolate.Interpolate(_vis1, _vis1.mask).fit(), 
                        time_kernal_size=0.5, freq_kernal_size=1, 
                        filter_direction = ('time', )).fit()
                _vis1 -= bg
            vis1[st:ed, ...] = _vis1
        

        _l = axhh.plot(x_axis, vis1[:,0], '-', drawstyle='steps-mid',
                label = label)
        #_l = axhh.plot(x_axis, vis1[:,0], '.', c=_l[0].get_color())

        _l = axvv.plot(x_axis, vis1[:,1], '-', drawstyle='steps-mid')
        #_l = axvv.plot(x_axis, vis1[:,1], '.', c=_l[0].get_color())


        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

        if self.params['nvss_cat'] is not None:
            dec_min = np.min(ts['dec'][:, gi])
            dec_max = np.max(ts['dec'][:, gi])
            #dec_min = 25.65294 #15.15294
            #dec_max = 25.65294 #15.15294
            #if dec_min == dec_max:
            dec_min -= 2./60.
            dec_max += 2./60.
            #dec_min = 25.541339365641274
            #dec_max = 25.61497357686361
            #dec_min = 25.458541361490884
            #dec_max = 25.532173665364585
            #dec_min = 25.789624659220376
            #dec_max = 25.863256963094077
            self.nvss_range.append([xmin, xmax, dec_min, dec_max])

    def write_output(self, output):

        super(PlotVvsTime, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        axhh.set_ylabel('HH Polarization')
        axvv.set_ylabel('VV Polarization')

        if fig_prefix is not None:
            fig_name = '%s_%s_TS.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)
        #if self.params['show'] is not None:
        #    if self.params['show'] == bl[0]-1:
        #        plt.show()
        #plt.close()

class PlotNcalVSTime(PlotVvsTime):


    params_init = {
            'noise_on_time' : 2,
            'timevars_poly' : 4,
            'kernel_size' : 11,
            }

    prefix = 'pnt_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #vis = np.abs(vis)
        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        on_t = self.params['noise_on_time']

        vis1 = np.ma.array(vis)
        if flag_mask:
            vis1.mask = vis_mask

        if 'ns_on' in ts.iterkeys():
            print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
            if len(ts['ns_on'].shape) == 2:
                on = ts['ns_on'][:, gi].astype('bool')
            else:
                on = ts['ns_on'][:]
            #on = ts['ns_on'][:]
            vis1.mask[~on, ...] = True
        else:
            print "No Noise Diode Mask info"

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            #x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
            #self.x_label = '%s' % x_axis[0].date()
            #x_axis = mdates.date2num(x_axis)
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                self.x_label = 'R.A.' 
            else:
                time = ts['sec1970'] + ts.attrs['sec1970']
                x_axis = [ datetime.fromtimestamp(s) for s in time]
                self.x_label = '%s UTC' % x_axis[0].date()
                x_axis = mdates.date2num(x_axis)
            if xmin is not None:
                xmin = x_axis[xmin]
            if xmax is not None:
                xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        vis1 = vis1[good_time_st:good_time_ed, ...]
        vis_mask = vis_mask[good_time_st:good_time_ed, ...]
        time = ts['sec1970'][good_time_st:good_time_ed]
        x_axis = x_axis[good_time_st:good_time_ed]

        on  = on[good_time_st:good_time_ed]

        good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
        good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
        vis1 = vis1[:, good_freq_st:good_freq_ed, ...]
        vis_mask = vis_mask[:, good_freq_st:good_freq_ed, ...]


        kernel_size = self.params['kernel_size']
        vis1, on = bp.get_Ncal(vis1, vis_mask, on, on_t)
        bandpass = np.ma.median(vis1, axis=0)
        bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=kernel_size)
        bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=kernel_size)
        bandpass = np.ma.filled(bandpass, 0)
        bandpass[bandpass==0] = np.inf
        vis1 /= bandpass[None, ...]
        #vis1 /= np.ma.mean(vis1, axis=(0,1))[None, None, :]

        #x_axis   = x_axis[on]
        #x_axis_norm = x_axis - x_axis[0]
        #x_axis_norm /= x_axis_norm.max()

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        #axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '.', label = label)
        #axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '.')

        vis1[vis1 == 0] = np.ma.masked
        vis1 = np.ma.median(vis1, axis=1)
        poly_order = self.params['timevars_poly']
        good = ~vis1.mask

        #vis1[:, 0][good[:, 0]] = medfilt(vis1[:, 0][good[:, 0]], kernel_size=(31,))
        #vis1[:, 1][good[:, 1]] = medfilt(vis1[:, 1][good[:, 1]], kernel_size=(31,))
        #vis1_poly_xx, vis1_poly_yy = bp.polyfit_timedrift(vis1, time, on, poly_order)
        vis1_poly_xx, vis1_poly_yy = bp.medfilt_timedrift(vis1, time, on,
                kernel_size=51)

        #vis1_poly_xx = np.poly1d(
        #        np.polyfit(x_axis_norm[good[:,0]], vis1[:, 0][good[:,0]], poly_order))
        #vis1_poly_yy = np.poly1d(
        #        np.polyfit(x_axis_norm[good[:,1]], vis1[:, 1][good[:,1]], poly_order))

        #vis1 = np.ma.array(medfilt(vis1.data, kernel_size=(11, 1)), mask = vis1.mask)

        _l = axhh.plot(x_axis[on], vis1[:,0], '-',lw=0.1) #,drawstyle='steps-mid')
        axhh.plot(x_axis, vis1_poly_xx, '-', c=_l[0].get_color(), lw=2.0,
                zorder=1000,label=label)
        _l = axvv.plot(x_axis[on], vis1[:,1], '-',lw=0.1) #,drawstyle='steps-mid')
        axvv.plot(x_axis, vis1_poly_yy, '-', c=_l[0].get_color(), lw=2.0,
                zorder=1000)

        axhh.axhline(1, 0, 1, c='k', lw=1.5, ls='--')
        axvv.axhline(1, 0, 1, c='k', lw=1.5, ls='--')

        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

def get_Ncal(vis, vis_mask, on, on_t):

    # remove the cal at the beginning/ending
    on[ :on_t] = False
    on[-on_t:] = False
    if on_t == 2:
        # noise cal may have half missing, because of the RFI flagging
        # remove them
        on  = (np.roll(on, 1) * on) + (np.roll(on, -1) * on)
        # use one time stamp before, one after as cal off
        off = (np.roll(on, 1) + np.roll(on, -1)) ^ on
        vis1_on  = vis[on, ...]
        vis1_off = vis[off, ...].data
    elif on_t == 1:
        off = np.roll(on, 1) + np.roll(on, -1)
        vis1_on  = vis[on, ...]
        vis1_off = vis[off, ...].data
        vis_shp = vis1_off.shape
        vis1_off = vis1_off.reshape( (-1, 2) + vis_shp[1:] )
        vis1_off = np.mean(vis1_off, axis=1)
    else:
        raise
    
    #vis1 = vis1.data
    vis1 = vis1_on - vis1_off

    if on_t > 1:
        vis_shp = vis1.shape
        vis1 = vis1.reshape((-1, on_t) + vis_shp[1:])
        vis1 = vis1 + vis1[:, ::-1, ...]
        vis1.shape = vis_shp

    return vis1, on


class PlotNoiseCal(PlotVvsTime):

    prefix = 'pcal_'

    params_init = {
            'noise_cal_init_time' : None,
            'noise_cal_period' : 19.9915424299,
            'noise_cal_length' : 1.8,
            'noise_cal_delayed_ant' : [],
            }

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        super(PlotNoiseCal, self).plot(vis, vis_mask, li, gi, bl, ts, **kwargs)

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        if self.params['noise_cal_init_time'] is not None:
            print "plot noise diode"
            if self.params['plot_index']:
                msg = 'noise diode need to plot in time, not index'
            else:
                t0 = Time(self.params['noise_cal_init_time']).unix
                t1 = ts['sec1970'][-1]
                tl = self.params['noise_cal_length']
                tp = self.params['noise_cal_period']
                #tp = 19.9915424299

                if bl[0] - 1 in self.params['noise_cal_delayed_ant']:
                    print "Cal delayed, t0 plus 1 s"
                    t0 += 1

                noise_st = np.arange(t0,      t1, tp)
                noise_ed = np.arange(t0 + tl, t1, tp)

                for _st in noise_st:
                    _st = datetime.fromtimestamp(_st) 
                    _st = mdates.date2num(_st)
                    axhh.axvline(_st, color='k', linestyle='-', linewidth=1)
                    axvv.axvline(_st, color='k', linestyle='-', linewidth=1)

                for _ed in noise_ed:
                    _ed = datetime.fromtimestamp(_ed) 
                    _ed = mdates.date2num(_ed)
                    axhh.axvline(_ed, color='k', linestyle='--', linewidth=1)
                    axvv.axvline(_ed, color='k', linestyle='--', linewidth=1)

class PlotPointingvsTime(PlotTimeStream):

    prefix = 'ppt_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        az  = ts['az'][:, gi]
        el = ts['el'][:, gi]

        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            time = ts['sec1970'] + ts.attrs['sec1970']
            x_axis = [ datetime.fromtimestamp(s) for s in time]
            self.x_label = '%s' % x_axis[0].date()
            x_axis = mdates.date2num(x_axis)

        bad_time = np.all(vis_mask, axis=(1, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        x_axis = x_axis[good_time_st:good_time_ed]
        az = az[good_time_st:good_time_ed]
        el = el[good_time_st:good_time_ed]

        az[az < 0] = az[az < 0] + 360.
        az = (az - 180.) * 60.
        el = (el - np.mean(el)) * 60

        az_slope = np.poly1d(np.polyfit(x_axis, az, 2))
        az -= az_slope(x_axis)

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        axhh.plot(x_axis, az, label = label)
        axvv.plot(x_axis, el)

        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

    def write_output(self, output):

        super(PlotPointingvsTime, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        axhh.set_ylabel('Azimuth [arcmin]')
        axvv.set_ylabel('Elevation [arcmin]')

        if fig_prefix is not None:
            fig_name = '%s_%s_AzEl.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)


class PlotSpectrum(PlotTimeStream):

    prefix = 'psp_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        print "global index %2d [m%03d]"%(gi, bl[0]-1)
        freq_indx = np.arange(vis.shape[1])
        freq = ts['freq'][:] * 1.e-3
        self.x_label = r'$\nu$ / GHz'
        #self.x_label = 'f [GHz]'

        bad_freq = np.all(vis_mask, axis=(0, 2))
        bad_time = np.all(vis_mask, axis=(1, 2))

        if vis.dtype == 'complex' or vis.dtype == 'complex64':
            print "Convert complex to float by abs"
            vis = np.abs(vis.copy())

        vis = np.ma.array(vis)
        vis.mask = vis_mask.copy()
        
        if 'ns_on' in ts.iterkeys():
            ns_on = ts['ns_on'][:, gi]
        else:
            ns_on = np.zeros(bad_time.shape).astype('bool')

        vis.mask[ns_on, ...] = True
        spec_HH = np.ma.median(vis[..., 0], axis=0)
        spec_VV = np.ma.median(vis[..., 1], axis=0)
        
        #bp_HH = np.ma.array(medfilt(spec_HH, 11))
        #bp_HH.mask = bad_freq
        #spec_HH /= bp_HH
        
        #bp_VV = np.ma.array(medfilt(spec_VV, 11))
        #bp_VV.mask = bad_freq
        #spec_VV /= bp_VV

        axhh = self.axhh
        axvv = self.axvv
        
        label = 'M%03d'%(bl[0] - 1)
        axhh.plot(freq, spec_HH , label=label, linewidth=1.5)
        #ax.plot(freq, bp_HH , 'y-', linewidth=1.0)
        
        axvv.plot(freq, spec_VV , linewidth=1.5)
        #ax.plot(freq, bp_VV , 'y-', linewidth=1.0)

        #if 'ns_on' in ts.iterkeys():

        #    vis.mask = vis_mask.copy()
        #    vis.mask[~ns_on] = True
        #    ns_HH = np.ma.median(vis[..., 0], axis=0)
        #    ns_VV = np.ma.median(vis[..., 1], axis=0)

        #    ns_HH = np.ma.array(medfilt(ns_HH, 11))
        #    ns_HH.mask = bad_freq

        #    ns_VV = np.ma.array(medfilt(ns_VV, 11))
        #    ns_VV.mask = bad_freq

        #    axhh.plot(freq, ns_HH , 'g--', label='noise HH', linewidth=1.5)
        #    axvv.plot(freq, ns_VV , 'r--', label='noise VV', linewidth=1.5)

        if xmin is None: xmin = freq[0]
        if xmax is None: xmax = freq[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

    def write_output(self, output):

        #super(PlotSpectrum, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        x_label = self.x_label

        axhh.set_xticklabels([])
        #axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        #axhh.set_ylabel('HH Polarization')
        #if xmin is None: xmin = x_axis[0]
        #if xmax is None: xmax = x_axis[-1]
        xmin = self.xmin
        xmax = self.xmax
        axhh.set_xlim(xmin=self.xmin, xmax=self.xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')
        axhh.legend(title=self.params['legend_title'], ncol=6)
        axhh.set_ylabel('HH Polarization')

        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        #axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        #axvv.set_ylabel('VV Polarization')
        axvv.set_xlim(xmin=self.xmin, xmax=self.xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')
        axvv.set_ylabel('VV Polarization')

        if fig_prefix is not None:
            fig_name = '%s_%s_Spec.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)

class CheckSpec(timestream_task.TimestreamTask):
    
    prefix = 'csp_'
    
    params_init = {
        'corr' : 'auto',
        'bad_freq_list' : [],
        'bad_time_list' : [],
        'legend_title' : '',
        'show' : None,
        'ymin' : None,
        'ymax' : None,
        'xmin' : None,
        'xmax' : None,
    }
    
    
    def process(self, ts):
        
        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']
        
        if bad_time_list is not None:
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        ts.redistribute('baseline')
        

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        
        func(self.plot, full_data=False, show_progress=show_progress, 
             progress_step=progress_step, keep_dist_axis=False)

        return super(CheckSpec, self).process(ts)

    
    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        ymin = self.params['ymin']
        ymax = self.params['ymax']
        xmin = self.params['xmin']
        xmax = self.params['xmax']
        
        
        bad_freq = np.all(vis_mask, axis=(0, 2))
        bad_time = np.all(vis_mask, axis=(1, 2))
        
        print "global index %2d [m%03d]"%(gi, bl[0]-1)
        freq_indx = np.arange(vis.shape[1])
        freq = ts['freq'][:]
        
        fig = plt.figure(figsize=(8, 4))
        ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        
        if vis.dtype == 'complex' or vis.dtype == 'complex64':
            vis = np.abs(vis)
        vis = np.ma.array(vis)
        vis.mask = vis_mask.copy()
        
        #spec_HH = np.ma.mean(vis[:, :, 0], axis=0)
        #spec_VV = np.ma.mean(vis[:, :, 1], axis=0)

        if 'ns_on' in ts.iterkeys():
            ns_on = ts['ns_on'][:, gi]
        else:
            ns_on = np.zeros(bad_time.shape).astype('bool')

        vis.mask[ns_on, ...] = True
        #spec_HH = np.ma.median(vis[..., 0], axis=0)
        #spec_VV = np.ma.median(vis[..., 1], axis=0)
        spec_HH = np.ma.mean(vis[..., 0], axis=0)
        spec_VV = np.ma.mean(vis[..., 1], axis=0)
        
        #bp_HH = np.ma.array(medfilt(spec_HH, 5))
        bp_HH = np.ma.array(gaussian_filter1d(spec_HH, 1.))
        bp_HH.mask = bad_freq
        spec_HH /= bp_HH.copy()
        
        #bp_VV = np.ma.array(medfilt(spec_VV, 11))
        #bp_VV.mask = bad_freq
        #spec_VV /= bp_VV

        
        ax.plot(freq, spec_HH/np.median(spec_HH) , 'r-', label='HH', linewidth=1.5, )
                #drawstyle='steps-mid')
        #ax.plot(freq, bp_HH/np.median(spec_HH) , 'k-', linewidth=1.0)
        
        #ax.plot(freq, spec_VV , 'r-', label='VV', linewidth=1.5)
        #ax.plot(freq, bp_VV , 'y-', linewidth=1.0)

        if 'ns_on' in ts.iterkeys():

            vis.mask = vis_mask.copy()
            vis.mask[~ns_on] = True
            ns_HH = np.ma.median(vis[..., 0], axis=0)
            ns_VV = np.ma.median(vis[..., 1], axis=0)

            ns_HH = np.ma.array(medfilt(ns_HH, 1))
            ns_HH.mask = bad_freq

            ns_VV = np.ma.array(medfilt(ns_VV, 1))
            ns_VV.mask = bad_freq

            ns_HH /= bp_HH.copy()

            dfreq =  freq[1] - freq[0]
            fs = float(ns_HH.shape[0])
            fc = 1./dfreq  # Cut-off frequency of the filter
            print fc
            w = fc / (fs / 2.) # Normalize the frequency
            b, a = signal.butter(5, w, 'low')
            ns_HH_smooth = signal.filtfilt(b, a, ns_HH)

            #ax.plot(freq, spec_HH/ns_HH , 'g-', label='noise HH', linewidth=1.5)
            ax.plot(freq, ns_HH/np.median(ns_HH) , 'g-', label='noise HH', linewidth=1.5)
            ax.plot(freq, ns_HH_smooth/np.median(ns_HH) , 'k-', 
                    label='noise HH smooth', linewidth=1.5)
            #ax.plot(freq, ns_VV , 'r--', label='noise VV', linewidth=1.5)
        


        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Power')
        #ax.semilogy()
        ax.legend(title=self.params['legend_title'])
        ax.tick_params(length=4, width=0.8, direction='in')
        ax.tick_params(which='minor', length=2, width=0.8, direction='in')
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        if self.params['show'] is not None:
            if self.params['show'] == bl[0]-1:
                plt.show()
        plt.close()
        #plt.show()
