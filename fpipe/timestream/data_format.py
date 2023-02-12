import astropy.io.fits as pyfits
import numpy as np
import h5py
from astropy import units as u
from astropy.time import Time

import gc

class FASTfits_Spec(object):
    '''
    Load the raw FAST fits file

    attrs:
        data : the main data, [time, freq, pol]
        mask : mask for data.
        time : the timestamps, in mjd
        freq : the frequency chennals, in MHz

    '''

    history = ''
    
    def __init__(self, file_name_list, fmin=None, fmax=None):

        if not isinstance( file_name_list, list):
            file_name_list = [file_name_list, ]
        
        data = []
        time = []
        mask = []
        freq = None
        for file_name in file_name_list:
            _data, _mask, _time, _freq = self.load_one_file(file_name, fmin, fmax)
            data.append(_data)
            time.append(_time)
            mask.append(_mask)
            if freq is None:
                freq = _freq
            else:
                if np.any(_freq != freq):
                    raise

        self.freq = freq
        self.data = np.concatenate(data, axis=0)
        
        time = np.concatenate(time, axis=0)
        time = Time(time, format='mjd', scale='utc')
        self.time = time.unix
        self.date_obs = time[0]
        self.mask = np.concatenate(mask, axis=0)

        #print self.data.shape
        #print self.time.shape
        #print self.freq.shape

    def load_one_file(self, file_name, fmin, fmax):

        self.history += '%s\n'%file_name
        hdulist = pyfits.open(file_name, memmap=False)
        
        data_sets = hdulist[1].data
        freq_0 = data_sets.field('FREQ')[0]
        freq_n = data_sets.field('NCHAN')[0]
        freq_w = data_sets.field('CHAN_BW')[0]

        freq = np.arange(freq_n) * freq_w + freq_0

        if not ((fmin is None) and (fmax is None)):
            f_st, f_ed = self.freq_truncate(freq, fmin, fmax)
        else:
            f_st = 0
            f_ed = None
        freq = freq[f_st:f_ed]
        #print "freq resol %20.16f MHz"%(freq[1] - freq[0])

        data = data_sets.field('DATA')[:, f_st: f_ed, :]
        #data = np.swapaxes(data, -1,-2)
        data *= 1.e-10 # raw data have huge value, reacaled by 10^-10.
        #data = data.astype('float32')
        mask = np.zeros(data.shape).astype('bool')
        
        time_bins = float(data.shape[1])
        time = data_sets.field('UTOBS')

        hdulist.close()

        return data, mask, time, freq

    def freq_truncate(self, freq, fmin, fmax):

        if fmin is None: fmin = -1
        if fmax is None: fmax = 1.e20
        
        _good = (freq >= fmin) * (freq < fmax)
        _good_st = np.argwhere(_good)[ 0, 0]
        _good_ed = np.argwhere(_good)[-1, 0]
        #self.freq = self.freq[_good_st:_good_ed]
        #self.data = self.data[:, :, :, _good_st:_good_ed]
        return _good_st, _good_ed

    def flag_cal(self, p=8, l=1, d=0):

        msg = "Flag NCal %d of every %d time stamps with delay of %d time stamps\n"\
                %(l, p, d)
        self.history += msg
        
        cal_on = np.zeros(self.data.shape[0]).astype('bool')
        #cal_on = cal_on.reshape(-1, p)
        #cal_on[:, d:l] = True
        #cal_on = cal_on.flatten()
        for i in range(l):
            cal_on[slice(d+i, None, p)] = True
        self.cal_on = cal_on
        self.cal_off = np.roll(cal_on, 1)

    def rebin_freq(self, n=16):

        time_n, freq_n, pol_n = self.data.shape

        freq = self.freq
        freq_reso = freq[1] - freq[0]
        freq_n = freq_n / n
        freq = freq[:freq_n*n].reshape(freq_n, n)
        msg  = "Degrade frequency resolution from %16.12f kHz to %16.12f kHz\n"%(
                freq_reso * 1.e3, freq_reso * n * 1.e3)
        msg += "By averaging avery %d frequency bins\n"%n

        data_shp = (time_n, freq_n, n, pol_n)

        data = self.data[:,:freq_n*n,:].reshape(data_shp)
        mask = self.mask[:,:freq_n*n,:].reshape(data_shp)
        data[mask] = 0.
        mask = (~mask).astype('int')
        
        freq = np.mean(freq, axis=1)
        msg += "Freq(0) = %16.12f MHz, dFreq = %16.12f kHz\n"%(
                freq[0], (freq[1] - freq[0] ) * 1.e3)
        data = np.sum(data, axis=2)
        norm = np.sum(mask, axis=2) * 1.
        mask = norm < n * 0.8
        norm[mask] = np.inf
        data = data / norm
        
        self.data = data
        self.mask = mask
        self.freq = freq

        self.history += msg

    def rebin_time(self, n=10):

        time_n, freq_n, pol_n = self.data.shape

        time = self.time
        time_reso = time[1] - time[0]
        time_n = time_n / n
        time = time[:time_n*n].reshape(time_n, n)
        msg  = "Degrade time interval from %16.12f s to %16.12f s\n"%(
                time_reso, time_reso * n)
        msg += "By averaging avery %d time stampes\n"%n
        msg += "Abandon last %d time stamps\n"%(self.data.shape[0] - time_n*n)

        data_shp = (time_n, n, freq_n, pol_n)

        data = self.data[:time_n*n, ...].reshape(data_shp)
        mask = self.mask[:time_n*n, ...].reshape(data_shp)
        data[mask] = 0.
        mask = (~mask).astype('int')
        
        time = np.mean(time, axis=1)
        msg += "time(0) = %16.12f s, dtime = %16.12f s\n"%(
                time[0], (time[1] - time[0] ))
        data = np.sum(data, axis=1)
        norm = np.sum(mask, axis=1) * 1.
        mask = norm < n * 0.8
        norm[mask] = np.inf
        data = data / norm
        
        self.data = data
        self.mask = mask
        self.time = time

        self.history += msg

class FASTh5_Spec(object):
    '''
    Load the FAST hdf5 file

    attrs:
        data : the main data, [time, freq, pol]
        mask : mask for data.
        time : the timestamps, in mjd
        freq : the frequency chennals, in MHz

    '''

    def __init__(self, file_name_list, fmin=None, fmax=None):

        if not isinstance( file_name_list, list):
            file_name_list = [file_name_list, ]

        freq_sel = [fmin, fmax]
        
        data = None
        for file_name in file_name_list:
            data = self.load_data(file_name, data=data, freq_sel = freq_sel)
        self.freq = data['freq']
        self.time = data['time']
        self.date_obs = self.time[0]
        self.data = data['vis'].data
        if not hasattr(data['vis'].mask, '__iter__'):
            self.mask = np.zeros(self.data.shape, dtype='bool')
        else:
            self.mask = data['vis'].mask
        self.ra   = data['ra']
        self.dec  = data['dec']
        self.ants = data['ants']
        self.ns_on= data['ns_on']

        del data
        gc.collect()


    def load_data(self, data_file, data=None, freq_sel = [None, None]):

        with h5py.File(data_file, 'r') as fh: 
            #freqstart = fh.attrs['freqstart']
            #freqstep  = fh.attrs['freqstep']
            #freqn     = fh.attrs['nfreq']
            #freq = np.arange(freqn) * freqstep + freqstart
            freq = fh['freq'][:]
            
            ants = fh['blorder'][:]
            
            freq = freq[slice(*freq_sel)]
        
            #vis = np.abs(fh['vis'][:, slice(*freq_sel), ...])
            vis = fh['vis'][:, slice(*freq_sel), ...]
        
            #timestart = fh.attrs['sec1970']
            #timestep  = fh.attrs['inttime']
            #timen     = fh['vis'].shape[0]
            #time = np.arange(timen) * timestep + timestart
            time = fh['sec1970'][:]
            
            ra = fh['ra'][:]
            dec= fh['dec'][:]

            ns_on = fh['ns_on'][:].astype('bool')
            mask = fh['vis_mask'][:, slice(*freq_sel), ...].astype('bool')
        
        vis = np.ma.array(vis, mask=mask)
        #vis.mask = np.zeros(vis.shape).astype('bool')

        if data is None:
            data = {}
            data['vis']  = vis
            data['freq'] = freq
            data['time'] = time
            data['ants'] = ants
            data['ra'] = ra
            data['dec'] = dec
            data['ns_on'] = ns_on
        else:
            data['vis']  = np.ma.concatenate([data['vis'], vis],    axis=0)
            data['time'] = np.ma.concatenate([data['time'], time],  axis=0)
            data['ra']   = np.ma.concatenate([data['ra'], ra],      axis=0)
            data['dec']  = np.ma.concatenate([data['dec'], dec],    axis=0)
            data['ns_on']= np.ma.concatenate([data['ns_on'], ns_on],axis=0)

        del vis, mask, ra, dec, ns_on, freq, ants
        gc.collect()

        return data

