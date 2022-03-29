import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.signal import medfilt
from scipy.signal import convolve2d

from fpipe.plot import plot_map as pm
from fpipe.point_source import source
from fpipe.map import algebra as al

from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import NearestNDInterpolator

import h5py as h5

def plot_source_from_map(map_list, nvss_path, nvss_range, threshold=100, 
        output_path=None, beam_size=3./60., plot_hit=False, hitmap=None, 
        ra_dec_path =None):
    imap_info_list = []
    for map_name in map_list:
        with h5.File(map_name, 'r') as f:
            imap = al.load_h5(f, 'clean_map')
            imap = al.make_vect(imap, axis_names = imap.info['axes'])

            nmap = al.load_h5(f, 'noise_diag')
            nmap = al.make_vect(nmap, axis_names = imap.info['axes'])

            #imap = f['dirty_map'][:]
            pixs = f['map_pix'][:]
            nside = f['nside'][()]
        imap_info_list.append([imap, pixs, nside, nmap])

    #for _s in source.get_pointsource_spec(nvss_path, nvss_range, threshold, mJy=True):
    for _s in source.get_nvss_flux(nvss_path, nvss_range, threshold, mJy=True):

        #ax1.plot(_s[0], _s[1], 'k--')

        xx_min = 1.e30
        xx_max = -1.e30

        ymax = _s[1].max()*1.2

        #for map_name in map_list:
        spec_list = []
        for imap_info in imap_info_list:

            _r = source.get_map_spec(imap_info, _s[3], _s[4], mJy=True)
            if _r is not None:
                freq, spec, p_ra, p_dec, error = _r
                print spec.shape
            else:
                continue
            spec = np.ma.masked_equal(spec, 0)
            if not (np.all(spec.mask) or spec.shape[0] == 0):
                spec_list.append([freq, spec])

                ymax = max(ymax, np.ma.median(spec) * 1.5)

                xx_min = min(xx_min, freq.min())
                xx_max = max(xx_max, freq.max())

        if len(spec_list) == 0:
            continue

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_axes([0.06, 0.13, 0.52, 0.8])
        #ax0 = fig.add_axes([0.70, 0.1, 0.20, 0.8])
        cax = fig.add_axes([0.93, 0.13, 0.010, 0.8])
        for spec in spec_list:
            ax1.plot(spec[0], spec[1], 'r.', ms=3, mfc='none')

        #if beam_size is not None:
        #    beam_sig = beam_size * (2. * np.sqrt(2.*np.log(2.)))
        #    _w = np.exp(-0.5*(((p_ra - _s[3])*np.cos(np.radians(p_dec)))**2 \
        #            + (p_dec - _s[4])**2)/beam_sig **2 )
        #else:
        #    _w = 1.
        _w = 1.
        #ax1.plot(_s[0], _s[1]*_w, 'k--')
        #ax1.errorbar(_s[5][:, 0], _s[5][:, 1], _s[5][:, 2], fmt='go')
        ax1.plot(_s[0], _s[1]*_w, 'go')


        #ax1.set_ylabel(r'$T$ K')
        ax1.set_ylabel(r'Flux mJy')
        ax1.set_xlabel('Frequency MHz')
        ax1.set_xlim(xmin=xx_min, xmax=xx_max)
        ax1.text(0.03, 0.1, '%s'%_s[2], transform=ax1.transAxes)
        ax1.set_ylim(ymin=0, ymax=ymax)

        if plot_hit and (hitmap is not None):
            map_name = hitmap
            map_key  = 'hitmap'
            vmin = 0
            vmax = None
        else:
            map_name = map_list[0]
            map_key  = 'clean_map'
            vmin = None
            vmax = None

        ax0 = pm.plot_map_hp(map_name, map_key=map_key,
                #map_key='noise_diag',
                pix=0.2/60., indx=(slice(0, None), ), imap_shp = (100, 100),
                field_center=[(_s[3], _s[4]),], figsize=(6, 3), 
                vmin=vmin, vmax=vmax, sigma = None,
                title='', proj='ZEA', cmap='Blues',
                axes = (fig, [0.65, 0.13, 0.27, 0.8], cax),
                verbose=False)[1][0]

        _nvss_range = [_s[3] - 20 * 0.5/60., _s[3] + 20 * 0.5/60.,
                       _s[4] - 20 * 0.5/60., _s[4] + 20 * 0.5/60.]
        pm.plot_nvss(nvss_path, [_nvss_range,], (fig, ax0), 1)

        ax0.plot(_s[3], _s[4], 'o', mec='r', mfc='none', ms=10, mew=1,
                   transform=ax0.get_transform('icrs'))
        #ax0.plot(_s[3]+8./60., _s[4], 'o', mec='g', mfc='none', ms=10, mew=1,
        #           transform=ax0.get_transform('icrs'))
        ax0.plot(p_ra, p_dec, '+', mec='y', mfc='none', ms=10, mew=1,
                   transform=ax0.get_transform('icrs'))
        #ax0.add_artist(plt.Circle((_s[3], _s[4]), beam_size/2., color='k', fill=False,
        #        transform=ax0.get_transform('icrs')))
        ax0.set_title('')

        beam_sig = beam_size * (2. * np.sqrt(2.*np.log(2.)))
        if ra_dec_path is not None:
            with h5.File(ra_dec_path, 'r') as f:
                #for dset in f.keys():
                _w_list = []
                for i in range(7):
                    dset = 'ra_dec_%02d'%i
                    radec = f[dset][:]
                    mask  = f['mask_%02d'%i][:]
                    for i in range(19):
                        #_ra  = np.ma.array(radec[:, 0, i], mask=mask[:, i])
                        #_dec = np.ma.array(radec[:, 1, i], mask=mask[:, i])
                        _ra  = radec[:, 0, i]
                        _dec = radec[:, 1, i]
                        _sel = (_ra  > _nvss_range[0]) \
                             * (_ra  < _nvss_range[1]) \
                             * (_dec > _nvss_range[2]) \
                             * (_dec < _nvss_range[3])
                        if np.any(_sel):
                            #ax0.plot(_ra[_sel], _dec[_sel], 'w.',
                            #    transform=ax0.get_transform('icrs'), zorder=1000)
                            _w = np.exp(-0.5*(((_ra[_sel] - p_ra)\
                                    *np.cos(np.radians(_dec[_sel])))**2 \
                                    + (_dec[_sel] - p_dec)**2)/beam_sig **2 )
                            _w = min(1, _w.max())
                            _w_list.append(_w)
                #print _w_list
                ax1.plot(_s[0], _s[1]*max(_w_list), 'b--')

            #ax0.plot()

        if output_path is not None:
            #fig.savefig(output_path + '%s.pdf'%(_s[2].replace(' ', '-')), 
            #        formate='pdf')
            if plot_hit and (hitmap is not None):
                fig.savefig(output_path + '%s_hit.png'%(_s[2].replace(' ', '-')), 
                        formate='png', dpi=300)
            else:
                fig.savefig(output_path + '%s.png'%(_s[2].replace(' ', '-')), 
                        formate='png', dpi=300)
        plt.show()
        plt.clf()


def check_spec(source_list, output=None):

    fig = plt.figure(figsize=[8, 3])
    ax  = fig.add_axes([0.1, 0.18, 0.85, 0.78])

    x = np.logspace(np.log10(0.05), np.log10(10), 100)

    for source in source_list:
        cal_path = source['path']
        cal_name = source['name']


        cal_data = np.loadtxt(cal_path)
        cal_spec_func = np.poly1d(np.polyfit(np.log10(cal_data[:,0]*1.e-9),
                                             np.log10(cal_data[:,1]*1.e3),
                                             deg=2,))
        cal_flux = 10. ** cal_spec_func(np.log10(x)) # in mJy
        _l = ax.plot(x, cal_flux, '-', label=cal_name)
        _c = _l[0].get_color()
        ax.errorbar(cal_data[:, 0]*1.e-9, cal_data[:, 1]*1.e3, cal_data[:,2]*1.e3,
                fmt='o', ecolor=_c, mec=_c, mfc=_c)

    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Flux [mJy]')
    ax.legend()
    ax.loglog()

    if output is not None:
        fig.savefig(output + '.pdf', formate='pdf')

