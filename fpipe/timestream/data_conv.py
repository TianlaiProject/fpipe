import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator,ScalarFormatter

import data_format
import h5py

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from scipy.signal import lombscargle
import copy
import gc


_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "0.2", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", 'w']
_l_list = ['s-', 'o--', '.--', 's--', 'o-']

_Lon = (106. + 51./60. + 24.0/3600.) * u.deg
_Lat = (25. + 39./60. + 10.6/3600.) * u.deg
_Location = EarthLocation.from_geodetic(_Lon, _Lat)



def convert_to_tl(data_path, data_file, output_path, alt, az, feed_rotation=0,
                  beam_list = [0, ], block_list = [0, 1],
                  fmin=None, fmax=None, degrade_freq_resol=None,
                  noise_cal = [8, 1, 0]):
    
    data_file_list = [[data_path + data_file%(_beam, _block)
                       for _block in block_list] for _beam in beam_list]
    
    history = 'Convert from:\n'
    with h5py.File(output_path, 'w') as df:
        
        fill_info(df) # some infomation
        
        data_shp = None
        beam_n = len(beam_list)
        for ii in range(beam_n):
        
            fdata = data_format.FASTfits_Spec(data_file_list[ii], fmin, fmax)
            fdata.flag_cal(*noise_cal)
            if degrade_freq_resol is not None:
                fdata.rebin_freq(degrade_freq_resol)
            print fdata.history
            if ii == 0: history += fdata.history
        
            if data_shp is None:
                data_shp = fdata.data.shape + (beam_n, )
                df.create_dataset('vis', dtype='float32', shape=data_shp)
                df['vis'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'
            
                df.create_dataset('vis_mask', dtype='uint8', shape=data_shp)
                df['vis_mask'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'
                
                obstime = fdata.date_obs.datetime.strftime('%Y/%m/%d %H:%M:%S')
                inttime = fdata.time[1] - fdata.time[0]
                df.attrs['inttime'] = inttime
                df.attrs['obstime'] = obstime
                df.attrs['sec1970'] = fdata.time[0]
                df['sec1970'] = fdata.time
                df['sec1970'].attrs['dimname'] = 'Time, '
                df['pol'] = np.array([0, 1, 2, 3])
                df['pol'].attrs['dimname'] = 'Polarization, '
                
                nfreq = fdata.freq.shape[0]
                df.attrs['nfreq'] = nfreq
                df.attrs['freqstart'] = fdata.freq[0]
                df.attrs['freqstep'] = fdata.freq[1] - fdata.freq[0]

                ## get ra dec according to meridian scan
                #ra, dec = get_pointing_meridian_scan(fdata.time, dec0, 
                #        time_format='unix', feed_rotation=feed_rotation)
                ra, dec = get_pointing_meridian_scan(fdata.time, alt, az,
                        time_format='unix', feed_rotation=feed_rotation)

                cal_on = fdata.cal_on[:, None] * np.ones(beam_n)[None, :]
                cal_on = cal_on.astype('bool')
                df['ns_on'] = cal_on
                df['ns_on'].attrs['dimname']  = 'Time, Baseline'

            else:
                if not np.array_equal(df['sec1970'][:], fdata.time):
                    emsg = 'Time not match between M%03d and M001'%(beam_list[ii])
                    raise ValueError(emsg)
                if df.attrs['freqstart'] != fdata.freq[0]:
                    emsg = 'Freq not match between M%03d and M001'%(beam_list[ii])
                    raise ValueError(emsg)

        
            df['vis'][...,ii] = fdata.data.astype('float32')
            df['vis_mask'][..., ii] = fdata.mask.astype('uint8')

            del fdata
            gc.collect()
            
            print
        
        # get ra dec according to meridian scan
        beam_indx = [x-1 for x in beam_list]
        ra  = ra[:,  beam_indx]
        dec = dec[:, beam_indx]

        df['ra'] = ra
        df['ra'].attrs['dimname']  = 'Time, Baseline'

        df['dec'] = dec.astype('float32')
        df['dec'].attrs['dimname'] = 'Time, Baseline'

        df.attrs['history'] = history
        df.attrs['nants'] = data_shp[3]
        df.attrs['npol']  = data_shp[2]
        
        channo = []
        feedpos= []
        feedno = np.array(beam_list)
        for _b in beam_list:
            channo.append([2 * _b -1, 2 * _b])
            feedpos.append([0, 0, 0])

        blorder = [[feedno[i], feedno[i]] for i in range(len(beam_list))]
        df['blorder'] = blorder
        df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'
        df['feedno'] = feedno
        df['channo'] = np.array(channo)
        df['channo'].attrs['dimname'] = 'Feed No., (HPolarization VPolarization)'
        df['feedpos'] = np.array(feedpos)
        df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
        df['feedpos'].attrs['unit'] = 'degree'
        

def fill_info(df):
    
    df.attrs['nickname'] = 'HIIMPilotSurvey'
    df.attrs['comment']  = 'Share risk'
    df.attrs['observer'] = 'None'
    #df.attrs['history'] = history
    df.attrs['keywordver'] = '0.0' # Keyword version.

    # Type B Keywords
    df.attrs['sitename'] = 'FAST'
    df.attrs['sitelat'] = _Lat.value #25. + 39./60. + 10.6/3600.
    df.attrs['sitelon'] = _Lon.value #106. + 51./60. + 24.0/3600.
    df.attrs['siteelev'] = 1000.0    # Not precise
    df.attrs['timezone'] = 'UTC+08'  #
    df.attrs['epoch'] = 2000.0  # year

    df.attrs['telescope'] = 'FAST' #
    df.attrs['dishdiam'] = 300.
    df.attrs['cylen'] = -1 # For dish: -1
    df.attrs['cywid'] = -1 # For dish: -1

    df.attrs['recvver'] = '0.0'    # Receiver version.
    df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.

    df.attrs['corrver'] = '0.0'    # Correlator version.
    df.attrs['samplingbits'] = 8 # ADC sampling bits.
    df.attrs['corrmode'] = 1 # 2, 3


def offset_by(skycoord, posang, distance):
    """
    Copied from Astropy 3.1.2v
    
    Point with the given offset from the given point.
    Parameters
    ----------
    lon, lat, posang, distance : `Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the starting point,
        position angle and distance to the final point.
        Quantities should be in angular units; floats in radians.
        Polar points at lat= +/-90 are treated as limit of +/-(90-epsilon) and same lon.
    Returns
    -------
    lon, lat : `~astropy.coordinates.Angle`
        The position of the final point.  If any of the angles are arrays,
        these will contain arrays following the appropriate `numpy` broadcasting rules.
        0 <= lon < 2pi.
    Notes
    -----
    """
    from astropy.coordinates.angles import Angle
    from astropy.coordinates.representation import UnitSphericalRepresentation
    
    lat = skycoord.represent_as(UnitSphericalRepresentation).lat
    lon = skycoord.represent_as(UnitSphericalRepresentation).lon

    # Calculations are done using the spherical trigonometry sine and cosine rules
    # of the triangle A at North Pole,   B at starting point,   C at final point
    # with angles     A (change in lon), B (posang),            C (not used, but negative reciprocal posang)
    # with sides      a (distance),      b (final co-latitude), c (starting colatitude)
    # B, a, c are knowns; A and b are unknowns
    # https://en.wikipedia.org/wiki/Spherical_trigonometry

    cos_a = np.cos(distance)
    sin_a = np.sin(distance)
    cos_c = np.sin(lat)
    sin_c = np.cos(lat)
    cos_B = np.cos(posang)
    sin_B = np.sin(posang)

    # cosine rule: Know two sides: a,c and included angle: B; get unknown side b
    cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
    # sin_b = np.sqrt(1 - cos_b**2)
    # sine rule and cosine rule for A (using both lets arctan2 pick quadrant).
    # multiplying both sin_A and cos_A by x=sin_b * sin_c prevents /0 errors
    # at poles.  Correct for the x=0 multiplication a few lines down.
    # sin_A/sin_a == sin_B/sin_b    # Sine rule
    xsin_A = sin_a * sin_B * sin_c
    # cos_a == cos_b * cos_c + sin_b * sin_c * cos_A  # cosine rule
    xcos_A = cos_a - cos_b * cos_c

    A = Angle(np.arctan2(xsin_A, xcos_A), u.radian)
    # Treat the poles as if they are infinitesimally far from pole but at given lon
    # The +0*xsin_A is to broadcast a scalar to vector as necessary
    w_pole = np.argwhere((sin_c + 0*xsin_A) < 1e-12)
    if len(w_pole) > 0:
        # For south pole (cos_c = -1), A = posang; for North pole, A=180 deg - posang
        A_pole = (90*u.deg + cos_c*(90*u.deg-Angle(posang, u.radian))).to(u.rad)
        try:
            A[w_pole] = A_pole[w_pole]
        except TypeError as e: # scalar
            A = A_pole

    outlon = (Angle(lon, u.radian) + A).wrap_at(360.0*u.deg).to(u.deg)
    outlat = Angle(np.arcsin(cos_b), u.radian).to(u.deg)

    return SkyCoord(outlon, outlat, frame=skycoord.frame)

def get_pointing_meridian_scan(time, alt, az, time_format='unix', feed_rotation=0):
    
    '''
    estimate the pointing RA Dec accoriding obs time and init Dec 
    for drift scan pointing at meridian
    
    time: obs time
    '''
    
    alt = alt * np.ones_like(time)
    az  = az  * np.ones_like(time)
    time = Time(time, format=time_format, location=_Location)
    c0   = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame='altaz', 
                    location=_Location, obstime=time)
    c0   = c0.transform_to('icrs')

    ## pointing at meridian, RA = LST
    #ra0  = time.sidereal_time('apparent').to(u.deg)
    #dec0 = np.ones_like(ra0) * dec0

    # pointing RA Dec of the center beam
    #c0 = SkyCoord(ra0, dec0)

    # position of 19 beam in unit or arcmin, from Wenkai's calculation
    # already rotated by 23.4 deg
    x_position = np.array([  0.000,   5.263,   0.659,  -4.604,  -5.263,
                            -0.659,   4.604,  10.526,   5.922,   1.318,
                            -3.945,  -9.208,  -9.867, -10.526,  -5.922,
                            -1.318,   3.945,   9.208,   9.867])
    y_position = np.array([  0.000,  -2.277,  -5.6965, -3.419,   2.277,
                             5.6965,  3.419,  -4.555,  -7.974, -11.393,
                             -9.116, -6.838,  -1.142,   4.555,   7.974,
                             11.393,  9.116,   6.838,   1.142])

    separation = np.sqrt(x_position ** 2 + y_position ** 2) * u.arcmin
    position_angle  = np.arctan2(x_position, -y_position) * u.rad + 23.4 * u.deg
    position_angle -= feed_rotation * u.deg

    _c = offset_by(c0[:, None], position_angle[None, :], separation[None, :])
    
    return _c.ra.deg, _c.dec.deg 

if __name__ == '__main__':

    data_path = '/scratch/users/ycli/.test/'
    data_list = ['SDSS_N_2a_arcdrfit-M01_W_0001.fits',]
    data_file_list = [data_path + f for f in data_list]
    
    fdata = data_format.FASTfits_Spec(data_file_list, fmin=1300, fmax=1430)
