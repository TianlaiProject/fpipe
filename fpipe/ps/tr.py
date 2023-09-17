'''

Transfer function estimation 

'''
import numpy as np
import h5py as h5
import gc
import copy
import logging

from meerKAT_sim.ps import power_spectrum as ps
from meerKAT_utils import algebra as al
from meerKAT_utils import physical_gridding as gridding
from meerKAT_utils import binning

from meerKAT_sim.ps import pwrspec_estimator as pse, fgrm

logger = logging.getLogger(__name__)

class AutoTransferFunction(ps.AutoPS_OneByOne):

    prefix = 'atr_'

    params_init = {
            'ps3d_ref' : None,
            'sim' : True,
            }

    def init_output(self):

        super(AutoTransferFunction, self).init_output()

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp   = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            c, c_info = gridding.physical_grid(map_tmp)
            c.info    = pse.make_k_axes(c)[0]

        self.create_dataset_like('ps3d', c)

    def process(self, input):

        ii = 0.
        for tind_o, cube, cube_w in self.iterpstasks(input):

            ps3d, ps2d, ps1d = pse.calculate_xspec(
                    cube[0], cube[1], cube_w[0], cube_w[1],
                    window='blackman', #None, 
                    bins=self.kbin_edges, bins_x = self.kbin_x_edges, 
                    bins_y = self.kbin_y_edges,
                    logbins=self.params['logk'],
                    logbins_2d=self.params['logk_2d'],
                    unitless=self.params['unitless'],
                    nonorm = self.params['nonorm'],
                    feedback = self.feedback,
                    return_3d=True)

            self.df['ps3d'][:] += ps3d
            ii += 1.

            self.df['binavg_1d'][tind_o + (slice(None), )] = ps1d['binavg']
            self.df['counts_1d'][tind_o + (slice(None), )] = ps1d['counts_histo']

            self.df['binavg_2d'][tind_o + (slice(None), )] = ps2d['binavg']
            self.df['counts_2d'][tind_o + (slice(None), )] = ps2d['counts_histo']

            del ps2d, ps1d, cube, cube_w
            gc.collect()

        self.df['ps3d'][:] /= ii

        if self.params['ps3d_ref'] is not None:
            with h5.File(self.params['ps3d_ref'], 'r') as f:
                ps3d_ref = f['ps3d'][:]

            ps3d_ref[ps3d_ref==0] = np.inf
            self.df['ps3d'][:] /= ps3d_ref

        for ii in range(self.input_files_num):
            input[ii].close()


class CrossTransferFunction(ps.CrossPS_OneByOne):

    prefix = 'ctr_'

    params_init = {
            'ps3d_ref' : None,
            'sim' : True,
            }

    def init_output(self):

        super(CrossTransferFunction, self).init_output()

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp   = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            c, c_info = gridding.physical_grid(map_tmp)
            c.info    = pse.make_k_axes(c)[0]

        self.create_dataset_like('ps3d', c)

    def process(self, input):

        ii = 0.
        for tind_o, cube, cube_w in self.iterpstasks(input):

            ps3d, ps2d, ps1d = pse.calculate_xspec(
                    cube[0], cube[1], cube_w[0], cube_w[1],
                    window='blackman', #None, 
                    bins=self.kbin_edges, bins_x = self.kbin_x_edges, 
                    bins_y = self.kbin_y_edges,
                    logbins=self.params['logk'],
                    logbins_2d=self.params['logk_2d'],
                    unitless=self.params['unitless'],
                    nonorm = self.params['nonorm'],
                    feedback = self.feedback,
                    return_3d=True)

            self.df['ps3d'][:] += ps3d
            ii += 1.

            self.df['binavg_1d'][tind_o + (slice(None), )] = ps1d['binavg']
            self.df['counts_1d'][tind_o + (slice(None), )] = ps1d['counts_histo']

            self.df['binavg_2d'][tind_o + (slice(None), )] = ps2d['binavg']
            self.df['counts_2d'][tind_o + (slice(None), )] = ps2d['counts_histo']

            del ps2d, ps1d, cube, cube_w
            gc.collect()

        self.df['ps3d'][:]  /= ii
        #self.df['ps3d'][:] **= 2.

        if self.params['ps3d_ref'] is not None:
            with h5.File(self.params['ps3d_ref'], 'r') as f:
                ps3d_ref = f['ps3d'][:]

            ps3d_ref[ps3d_ref==0] = np.inf
            self.df['ps3d'][:] /= ps3d_ref

        for ii in range(self.input_files_num):
            input[ii].close()

