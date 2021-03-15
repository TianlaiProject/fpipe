"""RFI flagging by using iterated thresholding.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
import scipy.signal as sig
from caput import mpiutil
from fpipe.timestream import timestream_task
from tlpipe.container.timestream import Timestream


class Flag(timestream_task.TimestreamTask):
    """RFI flagging by using iterated thresholding."""

    params_init = {
                    'time_sigma_thres': 3.0,
                    'freq_sigma_thres': 6.0,
                    'badness_thres': 0.1,
                    'time_cut': 40,
                    'freq_cut': 10,
                    'max_itr': 20,
                    'n_bands': 20,
                    'time_bins_smooth': 10.0,
                    'block_length' : 600,
                  }

    prefix = 'fgbt_'

    def process(self, ts):

        assert mpiutil.size == 1, 'This task works only for a single process run'

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        badness_thres = self.params['badness_thres']
        max_itr = self.params['max_itr']
        block_length = self.params['block_length']
        total_length = ts.local_vis.shape[0]

        for ii in range(0, total_length, block_length):

            sel = slice(ii, ii + block_length)
            print sel

            data1 = np.ma.array(ts.local_vis[sel, ...].copy(), 
                    mask=ts.local_vis_mask[sel, ...].copy())
            itr = 0
            bad_freqs = []
            amount_masked = -1 # For recursion
            while not (amount_masked == 0) and itr < max_itr:
                amount_masked = self.rfi_flagging_freq(data1, bad_freqs)
                itr += 1
            bad_freqs.sort()
            nchan1 = data1.shape[1]
            percent_masked1 = float(len(bad_freqs)) / nchan1
            badness = (percent_masked1 > badness_thres)
            print("percent masked = ", percent_masked1)
            if badness:
                data2 = np.ma.array(ts.local_vis[sel, ...].copy(), 
                        mask=ts.local_vis_mask[sel, ...].copy())
                self.filter_foregrounds(data2)
                # Mask the bad times
                itr = 0
                bad_times = []
                amount_masked = -1 # For recursion
                while not (amount_masked == 0) and itr < max_itr:
                    amount_masked = self.rfi_flagging_time(data2, bad_times)
                    itr += 1
                itr = 0
                bad_freqs = []
                amount_masked = -1
                while not (amount_masked == 0) and itr < max_itr:
                    amount_masked = self.rfi_flagging_freq(data2, bad_freqs)
                    itr += 1
                bad_freqs.sort()
                nchan2 = data2.shape[1]
                percent_masked2 = float(len(bad_freqs)) / nchan2
                badness = (percent_masked1 - percent_masked2) < 0.05
                if not badness:
                    itr = 0
                    while not (amount_masked == 0) and itr < max_itr:
                        amount_masked = self.rfi_flagging_freq(data2, bad_freqs)
                        itr += 1
                    data1 = data2
                    print("time flagging applied")
            # print("data1 = ", data1)
            self.filter_foregrounds(data1)

            itr = 0
            while not (amount_masked ==0) and itr < max_itr:
                amount_masked = self.rfi_flagging_freq(data1, bad_freqs)
                itr += 1

            ts.local_vis_mask[sel, ...] =\
                    np.logical_or(ts.local_vis_mask[sel, ...], data1.mask)


        return super(Flag, self).process(ts)

    def rfi_flagging_freq(self, data, bad_freq_list):

        freq_sigma_thres = self.params['freq_sigma_thres']
        freq_cut = self.params['freq_cut']

        flag_size = freq_cut

        spec_time_ava = np.ma.mean(data, axis=0)
        sig = np.ma.std(spec_time_ava, axis=0)
        max_sig = freq_sigma_thres * sig
        max_accepted = np.ma.mean(spec_time_ava, axis=0) + max_sig
        amount_masked = 0
        nfreq = np.int(data.shape[1])
        fmask = np.zeros(nfreq).astype('bool')
        for freq in range(0, nfreq):
            if np.any(spec_time_ava[freq, :, :] > max_accepted[:, :]):
                amount_masked += 1
                st = max(freq-flag_size, 0)
                ed = min(freq+flag_size, nfreq)
                fmask[st:ed] = True
        data.mask += fmask[None, :, None, None]
        bad_freq_list += list(np.where(fmask)[0])
        num_mask = np.ma.count_masked(data)
        print("mask freq number = ", amount_masked, num_mask)

        return amount_masked

    def rfi_flagging_time(self, data, bad_time_list):

        time_sigma_thres = self.params['time_sigma_thres']
        time_cut = self.params['time_cut']

        flag_size = time_cut

        spec_freq_ava = np.ma.mean(data, axis=1)
        sig = np.ma.std(spec_freq_ava, axis=0)
        max_accepted = np.ma.mean(spec_freq_ava, axis=0) + time_sigma_thres*sig
        bad_time_list = []
        amount_masked = 0
        for time in range(0, data.shape[0]):
            if np.any(spec_freq_ava[time, :, :] > max_accepted[:, :]):
                amount_masked += 1
                bad_time_list.append(time)
        for time in bad_time_list:
            st = max(time-flag_size, 0)
            ed = min(time+flag_size, data.shape[0])
            data[st:ed, :, :, :] = np.ma.masked
        num_mask = np.ma.count_masked(data)
        print("mask time number = ", amount_masked, num_mask)

        return amount_masked

    def filter_foregrounds(self, data):
        n_bands = self.params['n_bands']
        time_bins_smooth = self.params['time_bins_smooth']

        n_chan = data.shape[1]
        sub_band_width = float(n_chan) / n_bands
        width = time_bins_smooth / 2.355
        # 2.355 means the relation between FWHM and sigma, FWHM=sigma*2*sqrt(2*ln2)=2.355*sigma
        nk = np.int(round(4*width) + 1)
        smoothing_kernal = sig.gaussian(nk, width)
        smoothing_kernal /= np.sum(smoothing_kernal)
        smoothing_kernal.shape = (nk, 1, 1)
        # Now loop through the sub-bands. Foregrounds are assumed to be identical
        # within a sub-band.
        for subband_ii in range(n_bands):
            # Figure out what data is in this subband.
            band_start = np.int(round(subband_ii * sub_band_width))
            band_end = np.int(round((subband_ii + 1) * sub_band_width))
            data0 = data[:, band_start:band_end, :, :]
            # Estimate the forgrounds.
            # Take the band mean.
            foregrounds = np.ma.mean(data, 1)
            # Now low pass filter.
            fore_weights = (np.ones(foregrounds.shape, dtype=float) - np.ma.getmaskarray(foregrounds))
            foregrounds -= np.ma.mean(foregrounds, 0)
            foregrounds = foregrounds.filled(0)
            foregrounds = sig.convolve(foregrounds, smoothing_kernal, mode='same')
            fore_weights = sig.convolve(fore_weights, smoothing_kernal, mode='same')
            fore_weights[fore_weights==0] = np.inf
            foregrounds /= fore_weights
            # Subtract out the foregrounds.
            data0[...] -= foregrounds[:, None, :, :]
            data = data0
