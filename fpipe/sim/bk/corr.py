import scipy
import scipy.ndimage
import scipy.fftpack
import scipy.special
import numpy as np
import math

from os.path import exists

import fftutil
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

from gaussianfield import RandomField

_feedback = False

class RedshiftCorrelation(object):
    r"""A class for calculating redshift-space correlations.

    The mapping from real to redshift space produces anisotropic
    correlations, this class calculates them within the linear
    regime. As a minimum the velocity power spectrum `ps_vv` must be
    specified, the statistics of the observable can be specified
    explicitly (in `ps_dd` and `ps_dv`), or as a `bias` relative to
    the velocity spectrum.

    As the integrals to calculate the correlations can be slow, this
    class can construct a table for interpolating them. This table can
    be saved to a file, and reloaded as need.

    Parameters
    ----------
    ps_vv : function, optional
        A function which gives the velocity power spectrum at a
        wavenumber k (in units of h Mpc^{-1}).

    ps_dd : function, optional
        A function which gives the power spectrum of the observable.

    ps_dv : function, optional
        A function which gives the cross power spectrum of the
        observable and the velocity.

    redshift : scalar, optional
        The redshift at which the power spectra are
        calculated. Defaults to a redshift, z = 0.

    bias : scalar, optional
        The bias between the observable and the velocities (if the
        statistics are not specified directly). Defaults to a bias of
        1.0.

    Attributes
    ----------
    ps_vv, ps_dd, ps_dv : function
        The statistics of the obserables and velocities (see Parameters).

    ps_redshift : scalar
        Redshift of the power spectra (see Parameters).

    bias : scalar
        Bias of the observable (see Parameters).

    Notes
    -----
    To allow more sophisticated behaviour the four methods
    `growth_factor`, `growth_rate`, `bias_z` and `prefactor` may be
    replaced. These return their respective quantities as functions of
    redshift. See their method documentation for details. This only
    really makes sense when using `_vv_only`, though the functions can
    be still be used to allow some redshift scaling of individual
    terms.
    """

    ps_vv = None
    ps_dd = None
    ps_dv = None

    ps_redshift = 0.0
    bias = 1.0

    _vv_only = False

    cosmology = cosmo

    def __init__(self, ps_vv = None, ps_dd = None, ps_dv = None,
                 redshift = 0.0, bias = 1.0):
        self.ps_vv = ps_vv
        self.ps_dd = ps_dd
        self.ps_dv = ps_dv
        self.ps_redshift = redshift
        self.bias = bias
        self._vv_only = False if ps_dd and ps_dv else True

    def powerspectrum_1D(self, k_vec, z1, z2, numz, cross=False):
        r"""A vectorized routine for calculating the redshift space powerspectrum.

        Parameters
        ----------
        k_vec: array_like
            The magnitude of the k-vector
        redshift: scalar
            Redshift at which to evaluate other parameters

        Returns
        -------
        ps: array_like
            The redshift space power spectrum at the given k-vector and redshift.

        Note that this uses the same ps_vv as the realisation generator until
        the full dd, dv, vv calculation is ready.

        TODO: evaluate this using the same weight function in z as the data.
        """
        c1 = (self.cosmology.comoving_distance(z1) * self.cosmology.h).value
        c2 = (self.cosmology.comoving_distance(z2) * self.cosmology.h).value
        # Construct an array of the redshifts on each slice of the cube.
        #comoving_inv = cosmo.inverse_approx(self.cosmology.comoving_distance, z1, z2)
        #za = comoving_inv(da)
        _xp = np.linspace(z1 , z2 , 200)
        _fp = (self.cosmology.comoving_distance(_xp) * self.cosmology.h).value
        da = np.linspace(c1, c2, numz+1, endpoint=True)
        za = np.interp(da, _fp, _xp)

        # Calculate the bias and growth factors for each slice of the cube.
        mz = self.mean(za)
        bz = self.bias_z(za)
        fz = self.growth_rate(za)
        Dz = self.growth_factor(za) / self.growth_factor(self.ps_redshift)
        pz = self.prefactor(za)

        dfactor = np.mean(Dz * pz * bz)
        vfactor = np.mean(Dz * pz * fz)

        if cross:
            return self.ps_vv(k_vec) * dfactor * dfactor / np.mean(pz)
        else:
            return self.ps_vv(k_vec) * dfactor * dfactor

    def bias_z(self, z):
        r"""The linear bias at redshift z.

        The bias relative to the matter as a function of
        redshift. In this simple version the bias is assumed
        constant. Inherit, and override to use a more complicated
        model.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        bias : array_like
            The bias at `z`.
        """
        return self.bias * np.ones_like(z)


    def growth_factor(self, z):
        r"""The growth factor D_+ as a function of redshift.

        The linear growth factor at a particular
        redshift, defined as:
        .. math:: \delta(k; z) = D_+(z; z_0) \delta(k; z_0)

        Normalisation can be arbitrary. For the moment
        assume that \Omega_m ~ 1, and thus the growth is linear in the
        scale factor.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        growth_factor : array_like
            The growth factor at `z`.
        """
        return 1.0 / (1.0 + z)


    def growth_rate(self, z):
        r"""The growth rate f as a function of redshift.

        The linear growth rate at a particular redshift defined as:
        .. math:: f = \frac{d\ln{D_+}}{d\ln{a}}

        For the
        moment assume that \Omega_m ~ 1, and thus the growth rate is
        unity.

        Parameters
        ----------
        z : array_like
            The redshift to calculate at.

        Returns
        -------
        growth_rate : array_like
            The growth factor at `z`.
        """
        return 1.0 * np.ones_like(z)

    def prefactor(self, z):
        r"""An arbitrary scaling multiplying on each perturbation.

        This factor can be redshift dependent. It results in scaling
        the entire correlation function by prefactor(z1) *
        prefactor(z2).

        Parameters
        ----------
         z : array_like
            The redshift to calculate at.

        Returns
        -------
        prefactor : array_like
            The prefactor at `z`.
        """

        return 1.0 * np.ones_like(z)



    def mean(self, z):
        r"""Mean value of the field at a given redshift.

        Parameters
        ----------
        z : array_like
            redshift to calculate at.

        Returns
        -------
        mean : array_like
            the mean value of the field at each redshift.
        """
        return np.ones_like(z) * 0.0

    _sigma_v = 0.0
    def sigma_v(self, z):
        """Return the pairwise velocity dispersion at a given redshift.
        This is stored internally as `self._sigma_v` in units of km/s
        Note that e.g. WiggleZ reports sigma_v in h km/s
        """
        print "using sigma_v (km/s): " + repr(self._sigma_v)
        sigma_v_hinvMpc = (self._sigma_v / 100.)
        return np.ones_like(z) * sigma_v_hinvMpc


    def velocity_damping(self, kpar):
        """The velocity damping term for the non-linear power spectrum.
        """
        return (1.0 + (kpar * self.sigma_v(self.ps_redshift))**2.)**-1.


    def _realisation_dv(self, d, n):
        """Generate the density and line of sight velocity fields in a
        3d cube.
        """

        if not self._vv_only:
            raise Exception("Doesn't work for independent fields, I need to think a bit more first.")

        def psv(karray):
            """Assume k0 is line of sight"""
            k = (karray**2).sum(axis=3)**0.5
            return self.ps_vv(k) * self.velocity_damping(karray[..., 0])

        # Generate an underlying random field realisation of the
        # matter distribution.
        rfv = RandomField(npix = n, wsize = d)
        rfv.powerspectrum = psv

        vf0 = rfv.getfield()

        # Construct an array of \mu^2 for each Fourier mode.
        spacing = rfv._w / rfv._n
        kvec = fftutil.rfftfreqn(rfv._n, spacing / (2*math.pi))
        mu2arr = kvec[...,0]**2 / (kvec**2).sum(axis=3)
        mu2arr.flat[0] = 0.0
        del kvec

        df = vf0

        # Construct the line of sight velocity field.
        # TODO: is the s=rfv._n the correct thing here?
        vf = np.fft.irfftn(mu2arr * np.fft.rfftn(vf0), s=rfv._n)

        #return (df, vf, rfv, kvec)
        return (df, vf) #, rfv)


    def realisation(self, z1, z2, thetax, thetay, numz, numx, numy,
                    zspace=True, refinement=1, report_physical=False,
                    density_only=False, no_mean=False, no_evolution=False,
                    pad=5):
        r"""Simulate a redshift-space volume.

        Generates a 3D (angle-angle-redshift) volume from the given
        power spectrum. Currently only works with simply biased power
        spectra (i.e. vv_only). This routine uses a flat sky
        approximation, and so becomes inaccurate when a large volume
        of the sky is simulated.

        Parameters
        ----------
        z1, z2 : scalar
            Lower and upper redshifts of the box.
        thetax, thetay : scalar
            The angular size (in degrees) of the box.
        numz : integer
            The number of bins in redshift.
        numx, numy : integer
            The number of angular pixels along each side.
        zspace : boolean, optional
            If True (default) redshift bins are equally spaced in
            redshift. Otherwise space equally in the scale factor
            (useful for generating an equal range in frequency).
        density_only: boolean
            no velocity contribution
        no_mean: boolean
            do not add the mean temperature
        no_evolution: boolean
            do not let b(z), D(z) etc. evolve: take their mean
        pad: integer
            number of pixels over which to pad the physical region for
            interpolation onto freq, ra, dec; match spline order?

        Returns
        -------
        cube : np.ndarray
            The volume cube.

        """
        d1 = (self.cosmology.comoving_transverse_distance(z1) * self.cosmology.h).value
        d2 = (self.cosmology.comoving_transverse_distance(z2) * self.cosmology.h).value
        c1 = (self.cosmology.comoving_distance(z1) * self.cosmology.h).value
        c2 = (self.cosmology.comoving_distance(z2) * self.cosmology.h).value
        c_center = (c1 + c2) / 2.

        # Make cube pixelisation finer, such that angular cube will
        # have sufficient resolution on the closest face.
        #d = np.array([c2-c1, thetax * d2 * units.degree, thetay * d2 * units.degree])
        d = np.array([c2-c1, np.deg2rad(thetax * d2), np.deg2rad(thetay * d2)])
        # Note that the ratio of deltas in Ra, Dec in degrees may
        # be different than the Ra, Dec in physical coordinates due to
        # rounding onto this grid
        n = np.array([numz, int(d2 / d1 * numx), int(d2 / d1 * numy)])

        # Enlarge cube size by 1 in each dimension, so raytraced cube
        # sits exactly within the gridded points.
        d = d * (n + pad).astype(float) / n.astype(float)
        c1 = c_center - (c_center - c1)*(n[0] + pad) / float(n[0])
        c2 = c_center + (c2 - c_center)*(n[0] + pad) / float(n[0])
        n = n + pad
        # now multiply by scaling for a finer sub-grid
        n = refinement*n

        print "Generating cube: (%f to %f) x %f x %f (%d, %d, %d) (h^-1 cMpc)^3" % \
              (c1, c2, d[1], d[2], n[0], n[1], n[2])

        cube = self._realisation_dv(d, n)
        # TODO: this is probably unnecessary now (realisation used to change
        # shape through irfftn)
        n = cube[0].shape

        # Construct an array of the redshifts on each slice of the cube.
        #comoving_inv = cosmo.inverse_approx(self.cosmology.comoving_distance, 
        #        z1 - 0.1, z2 + 0.1)
        #da = np.linspace(c1, c2, n[0], endpoint=True)
        #za = comoving_inv(da)
        _xp = np.linspace(z1 - 0.1 , z2 + 0.1 , 200)
        _fp = (self.cosmology.comoving_distance(_xp) * self.cosmology.h).value
        da = np.linspace(c1, c2, numz+1, endpoint=True)
        za = np.interp(da, _fp, _xp)

        # Calculate the bias and growth factors for each slice of the cube.
        mz = self.mean(za)
        bz = self.bias_z(za)
        fz = self.growth_rate(za)
        Dz = self.growth_factor(za) / self.growth_factor(self.ps_redshift)
        pz = self.prefactor(za)

        # Construct the observable and velocity fields.
        if not no_evolution:
            df = cube[0] * (Dz * pz * bz)[:,np.newaxis,np.newaxis]
            vf = cube[1] * (Dz * pz * fz)[:,np.newaxis,np.newaxis]
        else:
            df = cube[0] * np.mean(Dz * pz * bz)
            vf = cube[1] * np.mean(Dz * pz * fz)

        # Construct the redshift space cube.
        rsf = df
        if not density_only:
            rsf += vf

        if not no_mean:
            rsf += mz[:,np.newaxis,np.newaxis]

        # Find the distances that correspond to a regular redshift
        # spacing (or regular spacing in a).
        if zspace:
            za = np.linspace(z1, z2, numz, endpoint = False)
        else:
            za = 1.0 / np.linspace(1.0 / (1+z2), 1.0 / (1+z1), numz, endpoint = False)[::-1] - 1.0

        da = (self.cosmology.comoving_transverse_distance(za) * self.cosmology.h).value
        xa = (self.cosmology.comoving_distance(za) * self.cosmology.h).value

        # Construct the angular offsets into cube
        tx = np.deg2rad( np.linspace(-thetax / 2., thetax / 2., numx) ) #* units.degree
        ty = np.deg2rad( np.linspace(-thetay / 2., thetay / 2., numy) ) #* units.degree

        #tgridx, tgridy = np.meshgrid(tx, ty)
        tgridy, tgridx = np.meshgrid(ty, tx)
        tgrid2 = np.zeros((3, numx, numy))
        acube = np.zeros((numz, numx, numy))

        # Iterate over redshift slices, constructing the coordinates
        # and interpolating into the 3d cube. Note that the multipliers scale
        # from 0 to 1, or from i=0 to i=N-1
        for i in range(numz):
            tgrid2[0,:,:] = (xa[i] - c1) / (c2-c1) * (n[0] - 1.)
            tgrid2[1,:,:] = (tgridx * da[i]) / d[1] * (n[1] - 1.) + \
                            0.5*(n[1] - 1.)
            tgrid2[2,:,:] = (tgridy * da[i]) / d[2] * (n[2] - 1.) + \
                            0.5*(n[2] - 1.)

            #if(zi > numz - 2):
            # TODO: what order here?; do end-to-end P(k) study
            #acube[i,:,:] = scipy.ndimage.map_coordinates(rsf, tgrid2, order=2)
            #acube[i,:,:] = scipy.ndimage.map_coordinates(rsf, tgrid2, order=1)
            acube[i,:,:] = scipy.ndimage.map_coordinates(rsf, tgrid2, order=0)

        if report_physical:
            return acube, rsf, (c1, c2, d[1], d[2])
        else:
            return acube

