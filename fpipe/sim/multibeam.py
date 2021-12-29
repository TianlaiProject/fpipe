import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

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

def get_pointing_meridian_scan(time, alt, az, location, x_position, y_position,
                              altaz=False):

    '''
    estimate the pointing RA Dec accoriding obs time and init Dec
    for drift scan pointing at meridian

    time: obs time
    '''

    c0   = SkyCoord(alt=alt, az=az, frame='altaz', location=location, obstime=time)
    c0   = c0.transform_to('icrs')

    separation = np.sqrt(x_position ** 2 + y_position ** 2) * u.arcmin
    position_angle  = np.arctan2(x_position, -y_position) * u.rad

    # _c = offset_by(c0[:, None], position_angle[None, :], separation[None, :])
    _c = offset_by(c0[:, None], position_angle.values.value[None, :], np.radians(separation.values.value[None, :] / 60.0))

    #return _c.ra.deg, _c.dec.deg
    if altaz:
        _altaz = _c.transform_to(AltAz(obstime=time, location=location)).altaz
        return _altaz.alt.deg, _altaz.az.deg
    else:
        return _c.ra.deg, _c.dec.deg
