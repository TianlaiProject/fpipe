
from os.path import join, dirname
#from fpipe.sim import corr21cm
from cora.signal import corr21cm
#from fpipe.plot import plot_map as pm
#from fpipe.sim.gaussianfield import RandomField, fftutil
from cora.core.gaussianfield import RandomField
from cora.util import fftutil

import numpy as np
from scipy import interpolate


def xi2ps_fft(xi_3d, n=256, dr=4., get_1d=False):

    ps_3d = np.fft.fftn(xi_3d)
    ps_3d *= dr**3

    if get_1d:

        k = np.fft.fftfreq(n, dr) * (2*np.pi)
        dk = (2*np.pi) / (n*dr)
        k_x = k[:, None, None]
        k_y = k[None, :, None]
        k_z = k[None, None, :]
        k_range = np.sqrt(k_x**2 + k_y**2 + k_z**2)

        #print ps_3d
        ps_3d = ps_3d.real

        #k_bin = np.linspace(0.01, 2., 20)
        #k = k_bin + 0.5 * (k_bin[1] - k_bin[0])
        k_bin = np.logspace(np.log10(0.001), np.log10(5.), 100)
        k = k_bin * (k_bin[1]/k_bin[0]) ** 0.5
        k = k[:-1]

        ps = np.histogram(k_range.flatten(), k_bin, weights=ps_3d.flatten())[0]
        normal = np.histogram(k_range.flatten(), k_bin)[0].astype(float)

        normal[normal==0] = np.inf
        ps /= normal

        return ps, k
    else:
        return ps_3d


def ps2xi_fft(pk, n=256, dr=4., get_1d=False):

    k = np.fft.fftfreq(n, dr) * (2*np.pi)
    dk = (2*np.pi) / (n*dr)
    k_x = k[:, None, None]
    k_y = k[None, :, None]
    k_z = k[None, None, :]
    k_range = np.sqrt(k_x**2 + k_y**2 + k_z**2)

    #pk = lambda k: np.interp(k, power[:,0], power[:,1])

    ps_3d = pk(k_range)

    xi_3d = np.fft.ifftn(ps_3d)
    xi_3d /= dr**3
    #print xi_3d.shape

    if get_1d:

        xi_3d = xi_3d.real

        r = (np.arange(n) + 1) * dr
        r_3d = r[:, None, None]**2 + r[None, :, None]**2 + r[None, None, :]**2
        r_3d = np.sqrt(r_3d)
        #r_3d = r_3d[:,:,:xi_3d.shape[2]]
        #print r_3d.shape

        r_bin = np.linspace(10, 200, 50)
        r = r_bin + 0.5 * (r_bin[1] - r_bin[0])
        r = r[:-1]

        xi     = np.histogram(r_3d.flatten(), r_bin, weights=xi_3d.flatten())[0]
        normal = np.histogram(r_3d.flatten(), r_bin)[0].astype(float)
        normal[normal==0] = np.inf
        xi /= normal

        return xi, r
    else:
        return xi_3d

class EoR(corr21cm.Corr21cm):

    def __init__(self, ps=None, sigma_v=0.0, redshift=0.0, psfile=None, pk_input=True, **kwargs):

        if psfile is None:
            psfile = join(dirname(__file__),"data/input_matterpower.dat")
            redshift = 0.2
        print("loading matter power file: " + psfile)
        pwrspec_data = np.genfromtxt(psfile)
        if not pk_input:
            factor = pwrspec_data[:,0] ** 3 / 2. / np.pi**2
        else:
            factor = 1
        (log_k, log_pk) = (np.log(pwrspec_data[:,0]), \
                           np.log(pwrspec_data[:,1] / factor))
        logpk_interp = interpolate.interp1d(log_k, log_pk,
                                            bounds_error=False,
                                            fill_value=np.min(log_pk))
        pk_interp = lambda k: np.exp(logpk_interp(np.log(k)))

        kstar = 7.0
        pk = lambda k: np.exp(-0.5 * k**2 / kstar**2) * pk_interp(k)

        super(EoR, self).__init__(ps=pk, sigma_v=sigma_v, redshift=redshift, **kwargs)

    def T_b(self, z):

        print('EoR uses Pk of brightness, ignore Tb ')

        return 1.

class Normal(corr21cm.Corr21cm):

    def __init__(self, ps=None, sigma_v=0.0, redshift=0.0, psfile=None, pk_input=True, **kwargs):

        if psfile is None:
            psfile = join(dirname(__file__),"data/input_matterpower.dat")
            redshift = 0.2
        print("loading matter power file: " + psfile)
        pwrspec_data = np.genfromtxt(psfile)
        if not pk_input:
            factor = pwrspec_data[:,0] ** 3 / 2. / np.pi**2
        else:
            factor = 1
        (log_k, log_pk) = (np.log(pwrspec_data[:,0]), \
                           np.log(pwrspec_data[:,1] / factor))
        logpk_interp = interpolate.interp1d(log_k, log_pk,
                                            bounds_error=False,
                                            fill_value=np.min(log_pk))
        pk_interp = lambda k: np.exp(logpk_interp(np.log(k)))

        kstar = 7.0
        pk = lambda k: np.exp(-0.5 * k**2 / kstar**2) * pk_interp(k)

        super(Normal, self).__init__(ps=pk, sigma_v=sigma_v, redshift=redshift, **kwargs)


class LogNormal(corr21cm.Corr21cm):

    def __init__(self, ps=None, sigma_v=0.0, redshift=0.0, psfile=None, pk_input=True, **kwargs):

        if psfile is None:
            psfile = join(dirname(__file__),"data/input_matterpower.dat")
            redshift = 0.2
        print("loading matter power file: " + psfile)
        pwrspec_data = np.genfromtxt(psfile)
        if not pk_input:
            factor = pwrspec_data[:,0] ** 3 / 2. / np.pi**2
        else:
            factor = 1
        (log_k, log_pk) = (np.log(pwrspec_data[:,0]), \
                           np.log(pwrspec_data[:,1] / factor))
        logpk_interp = interpolate.interp1d(log_k, log_pk,
                                            bounds_error=False,
                                            fill_value=np.min(log_pk))
        pk_interp = lambda k: np.exp(logpk_interp(np.log(k)))

        xi = ps2xi_fft(pk_interp, n=256, dr=4., get_1d=False)
        xi = np.log(1. + xi)
        ps_G, kh = xi2ps_fft(xi, n=256, dr=4., get_1d=True)
        kh   = kh[ps_G>0]
        ps_G = ps_G[ps_G>0]
        #plt.plot(kh, ps_G)
        #plt.loglog()
        #plt.show()
        ps_G = interpolate.interp1d(kh, ps_G, bounds_error=False,
                                    fill_value=np.min(np.exp(log_pk)))

        kstar = 7.0
        ps = lambda k: np.exp(-0.5 * k**2 / kstar**2) * ps_G(k)

        super(LogNormal, self).__init__(ps=ps, sigma_v=sigma_v, redshift=redshift, **kwargs)

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

        vf0  = rfv.getfield()
        sigG = np.var(vf0)
        vf0  = np.exp(vf0 - sigG/2.) - 1.

        # Construct an array of \mu^2 for each Fourier mode.
        spacing = rfv._w / rfv._n
        kvec = fftutil.rfftfreqn(rfv._n, spacing / (2*np.pi))
        mu2arr = kvec[...,0]**2 / (kvec**2).sum(axis=3)
        mu2arr.flat[0] = 0.0
        del kvec

        df = vf0

        # Construct the line of sight velocity field.
        # TODO: is the s=rfv._n the correct thing here?
        vf = np.fft.irfftn(mu2arr * np.fft.rfftn(vf0), s=rfv._n)

        #return (df, vf, rfv, kvec)
        return (df, vf) #, rfv)
