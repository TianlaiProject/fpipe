import numpy as np
import scipy as sp
import scipy.ndimage
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate.rbf import Rbf
from fpipe.map import algebra
from astropy.cosmology import Planck15 as cosmology
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
# 21cm transition frequency (in MHz)
__nu21__ = 1420.40575177

import logging

logger = logging.getLogger(__name__)


Rz = lambda th: np.array([[ np.cos(th), -np.sin(th),           0],
                          [ np.sin(th),  np.cos(th),           0],
                          [          0,           0,           1]])
Ry = lambda th: np.array([[ np.cos(th),           0,  np.sin(th)],
                          [          0,           1,           0],
                          [-np.sin(th),           0,  np.cos(th)]])

def centering_to_fieldcenter(ra, dec):

    shp = ra.shape

    ra_c  = 0.5 * (ra.min() + ra.max())
    dec_c = 90. - 0.5 * (dec.min() + dec.max())

    alpha = ra
    delta = dec
    alpha = np.deg2rad(alpha)
    delta = np.deg2rad(delta)
    x = np.cos(alpha) * np.cos(delta)
    y = np.cos(delta) * np.sin(alpha)
    z = np.sin(delta)
    xyz = np.array([x, y, z])

    xyz.shape = (3, -1)
    xyz = np.dot(xyz.T, Rz( ra_c * np.pi/180.)).T
    xyz = np.dot(xyz.T, Ry(dec_c * np.pi/180.)).T
    xyz = np.dot(xyz.T, Rz(          np.pi/2.)).T

    x, y, z = xyz[0], xyz[1], xyz[2]

    h   = np.hypot(x, y)
    ra  = np.rad2deg(np.arctan2(y, x))
    dec = np.rad2deg(np.arctan2(z, h))

    ra.shape = shp
    dec.shape = shp

    return ra, dec

def physical_grid_lf(input_array, refinement=1, pad=2, order=0, feedback=1, 
        mode='constant'):
    r"""Project from freq, ra, dec into physical coordinates

    Parameters
    ----------
    input_array: np.ndarray
        The freq, ra, dec map

    Returns
    -------
    cube: np.ndarray
        The cube projected back into physical coordinates

    """
    if not hasattr(pad, '__iter__'):
        pad = [pad, pad, pad]
    pad = np.array(pad)

    freq_axis = input_array.get_axis('freq') #/ 1.e6
    ra_axis   = input_array.get_axis('ra')
    dec_axis  = input_array.get_axis('dec')

    _dec, _ra = np.meshgrid(dec_axis, ra_axis)
    _ra, _dec = centering_to_fieldcenter(_ra, _dec)

    # convert the freq, ra and dec axis to physical distance
    z_axis = __nu21__ / freq_axis - 1.0
    d_axis = (cosmology.comoving_transverse_distance(z_axis) * cosmology.h).value
    c_axis = (cosmology.comoving_distance(z_axis) * cosmology.h).value

    d_axis = d_axis[:, None, None]
    c_axis = c_axis[:, None, None]
    _ra    = _ra[None,  :, :]
    _dec   = _dec[None, :, :]

    xx = d_axis * np.cos(np.deg2rad(_ra)) * np.cos(np.deg2rad(_dec))
    yy = d_axis * np.sin(np.deg2rad(_ra)) * np.cos(np.deg2rad(_dec))
    zz = c_axis * np.sin(np.deg2rad(_dec)) 

    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    zz = zz.flatten()[:, None]
    dd = input_array.flatten()
    coord = np.concatenate([zz, xx, yy], axis=1)
    #input_array_f = NearestNDInterpolator(coord, input_array.flatten())
    #input_array_f = Rbf(xx, yy, zz, input_array.flatten()[:, None], function='linear')

    (numz, numx, numy) = input_array.shape

    c1, c2 = zz.min(), zz.max()
    c_center = 0.5 * (c1 + c2)

    phys_dim = np.array([c2 - c1, xx.max() - xx.min(), yy.max() - yy.min()])

    n = np.array([numz, numx, numy])

    # Enlarge cube size by `pad` in each dimension, so raytraced cube
    # sits exactly within the gridded points.
    phys_dim = phys_dim * (n + pad).astype(float) / n.astype(float)
    c1 = c_center - (c_center - c1) * (n[0] + pad[0]) / float(n[0])
    c2 = c_center + (c2 - c_center) * (n[0] + pad[0]) / float(n[0])
    n = n + pad
    # now multiply by scaling for a finer sub-grid
    n = (refinement * n).astype('int')

    if feedback > 0:
        msg = "converting from obs. to physical coord\n"\
               "refinement=%s, pad=(%s, %s, %s)\n "\
               "(%d, %d, %d)->(%f to %f) x %f x %f\n "\
               "(%d, %d, %d) (h^-1 cMpc)^3\n" % \
                       ((refinement, ) +  tuple(pad) + (
                        numz, numx, numy, c1, c2, 
                        phys_dim[1], phys_dim[2], 
                        n[0], n[1], n[2]))
        msg += "dx = %f, dy = %f, dz = %f"%(abs(phys_dim[1]) / float(n[1] - 1),
                                            abs(phys_dim[2]) / float(n[2] - 1),
                                            abs(c2 - c1) / float(n[0] - 1) )
        logger.debug(msg)
        #print msg

    phys_map = algebra.make_vect(np.zeros(n), axis_names=('freq', 'ra', 'dec'))

    # TODO: should this be more sophisticated? N-1 or N?
    info = {}
    info['axes'] = ('freq', 'ra', 'dec')
    info['type'] = 'vect'
    info['freq_delta'] = abs(c2 - c1) / float(n[0] - 1)
    info['freq_centre'] = c1 + info['freq_delta'] * float(n[0] // 2)
    info['ra_delta'] = abs(phys_dim[1]) / float(n[1] - 1)
    info['ra_centre'] = 0.5 * (xx.max() + xx.min())
    info['dec_delta'] = abs(phys_dim[2]) / float(n[2] - 1)
    info['dec_centre'] = 0.5 * (yy.max() + yy.min())
    phys_map.info = info

    # same as np.linspace(c1, c2, n[0], endpoint=True)
    z_axis = phys_map.get_axis_edges("freq")
    x_axis = phys_map.get_axis_edges("ra")
    y_axis = phys_map.get_axis_edges("dec")
    
    norm = np.histogramdd(coord, bins=[z_axis, x_axis, y_axis])[0] * 1.
    phys_map[:] = np.histogramdd(coord, bins=[z_axis, x_axis, y_axis], weights=dd)[0]
    
    norm[norm==0] = np.inf
    phys_map /= norm

    return phys_map, info

def physical_grid(input_array, refinement=1, pad=2, order=0, feedback=1, 
        mode='constant'):
    r"""Project from freq, ra, dec into physical coordinates

    Parameters
    ----------
    input_array: np.ndarray
        The freq, ra, dec map

    Returns
    -------
    cube: np.ndarray
        The cube projected back into physical coordinates

    """
    if not hasattr(pad, '__iter__'):
        pad = [pad, pad, pad]
    pad = np.array(pad)

    freq_axis = input_array.get_axis('freq') #/ 1.e6
    ra_axis = input_array.get_axis('ra')
    dec_axis = input_array.get_axis('dec')

    nu_lower, nu_upper = freq_axis.min(), freq_axis.max()
    ra_fact = sp.cos(sp.pi * input_array.info['dec_centre'] / 180.0)
    thetax, thetay = np.ptp(ra_axis), np.ptp(dec_axis)
    thetax *= ra_fact
    (numz, numx, numy) = input_array.shape

    z1 = __nu21__ / nu_upper - 1.0
    z2 = __nu21__ / nu_lower - 1.0
    d1 = (cosmology.comoving_transverse_distance(z1) * cosmology.h).value
    d2 = (cosmology.comoving_transverse_distance(z2) * cosmology.h).value
    c1 = (cosmology.comoving_distance(z1) * cosmology.h).value
    c2 = (cosmology.comoving_distance(z2) * cosmology.h).value
    #c1 = np.sqrt(c1**2. - (((0.5 * thetax * u.deg).to(u.rad)).value * d1)**2)
    c_center = (c1 + c2) / 2.

    # Make cube pixelisation finer, such that angular cube will
    # have sufficient resolution on the closest face.
    phys_dim = np.array([c2 - c1,
                         ((thetax * u.deg).to(u.rad)).value * d2,
                         ((thetay * u.deg).to(u.rad)).value * d2])

    # Note that the ratio of deltas in Ra, Dec in degrees may
    # be different than the Ra, Dec in physical coordinates due to
    # rounding onto this grid
    #n = np.array([numz, int(d2 / d1 * numx), int(d2 / d1 * numy)])
    n = np.array([numz, numx, numy])

    # Enlarge cube size by `pad` in each dimension, so raytraced cube
    # sits exactly within the gridded points.
    phys_dim = phys_dim * (n + pad).astype(float) / n.astype(float)
    c1 = c_center - (c_center - c1) * (n[0] + pad[0]) / float(n[0])
    c2 = c_center + (c2 - c_center) * (n[0] + pad[0]) / float(n[0])
    n = n + pad
    # now multiply by scaling for a finer sub-grid
    n = (refinement * n).astype('int')

    if feedback > 0:
        msg = "converting from obs. to physical coord\n"\
               "refinement=%s, pad=(%s, %s, %s)\n "\
               "(%d, %d, %d)->(%f to %f) x %f x %f\n "\
               "(%d, %d, %d) (h^-1 cMpc)^3\n" % \
                       ((refinement, ) +  tuple(pad) + (
                        numz, numx, numy, c1, c2, 
                        phys_dim[1], phys_dim[2], 
                        n[0], n[1], n[2]))
        msg += "dx = %f, dy = %f, dz = %f"%(abs(phys_dim[1]) / float(n[1] - 1),
                                            abs(phys_dim[2]) / float(n[2] - 1),
                                            abs(c2 - c1) / float(n[0] - 1) )
        logger.debug(msg)
        #print msg


    # this is wasteful in memory, but numpy can be pickled
    phys_map_npy = np.zeros(n)
    phys_map = algebra.make_vect(phys_map_npy, axis_names=('freq', 'ra', 'dec'))
    #mask = np.ones_like(phys_map)
    mask = np.ones_like(phys_map_npy)

    # TODO: should this be more sophisticated? N-1 or N?
    info = {}
    info['axes'] = ('freq', 'ra', 'dec')
    info['type'] = 'vect'

    #info = {'freq_delta': abs(phys_dim[0])/float(n[0]),
    #        'freq_centre': abs(c2+c1)/2.,
    info['freq_delta'] = abs(c2 - c1) / float(n[0] - 1)
    info['freq_centre'] = c1 + info['freq_delta'] * float(n[0] // 2)

    info['ra_delta'] = abs(phys_dim[1]) / float(n[1] - 1)
    #info['ra_centre'] = info['ra_delta'] * float(n[1] // 2)
    info['ra_centre'] = 0.

    info['dec_delta'] = abs(phys_dim[2]) / float(n[2] - 1)
    #info['dec_centre'] = info['dec_delta'] * float(n[2] // 2)
    info['dec_centre'] = 0.

    phys_map.info = info
    #print info

    # same as np.linspace(c1, c2, n[0], endpoint=True)
    radius_axis = phys_map.get_axis("freq")
    x_axis = phys_map.get_axis("ra")
    y_axis = phys_map.get_axis("dec")

    # Construct an array of the redshifts on each slice of the cube.
    #comoving_inv = cosmo.inverse_approx(cosmology.comoving_distance, z1 * 0.9, z2 * 1.1)
    #za = comoving_inv(radius_axis)  # redshifts on the constant-D spacing
    _xp = np.linspace(z1 * 0.9, z2 * 1.1, 500)
    _fp = (cosmology.comoving_distance(_xp) * cosmology.h).value
    #comoving_inv = interp1d(_fp, _xp)
    #za = comoving_inv(radius_axis)  # redshifts on the constant-D spacing
    za = np.interp(radius_axis, _fp, _xp)
    nua = __nu21__ / (1. + za)

    gridy, gridx = np.meshgrid(y_axis, x_axis)
    interpol_grid = np.zeros((3, n[1], n[2]))

    for i in range(n[0]):

        interpol_grid[0, :, :] = (nua[i] - freq_axis[0]) / \
                                 (freq_axis[-1] - freq_axis[0]) * numz
        proper_z = cosmology.comoving_transverse_distance(za[i]) * cosmology.h
        proper_z = proper_z.value

        angscale = ((proper_z * u.deg).to(u.rad)).value
        interpol_grid[1, :, :] = gridx / angscale / thetax * numx + numx / 2
        interpol_grid[2, :, :] = gridy / angscale / thetay * numy + numy / 2

        phys_map_npy[i, :, :] = sp.ndimage.map_coordinates(input_array,
                                                           interpol_grid,
                                                           order=order,
                                                           mode=mode)

        interpol_grid[1, :, :] = np.logical_or(interpol_grid[1, :, :] >= numx,
                                             interpol_grid[1, :, :] < 0)
        interpol_grid[2, :, :] = np.logical_or(interpol_grid[2, :, :] >= numy,
                                             interpol_grid[2, :, :] < 0)
        mask = np.logical_not(np.logical_or(interpol_grid[1, :, :],
                                            interpol_grid[2, :, :]))
        phys_map_npy *= mask


    phys_map_npy = algebra.make_vect(phys_map_npy, axis_names=('freq', 'ra', 'dec'))
    phys_map_npy.info = info
    return phys_map_npy, info


if __name__ == '__main__':
    pass
