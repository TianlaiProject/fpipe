import logging
from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path

from fpipe.map import algebra as al
from fpipe.utils import fits_cat
from astropy.cosmology import Planck15 as cosmology
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
# 21cm transition frequency (in MHz)
__nu21__ = 1420.40575177
#from meerKAT_utils import physical_gridding as gridding
from fpipe.ps import physical_gridding as gridding
physical_grid = gridding.physical_grid
refinement = 1
order = 1

from fpipe.map import mapbase
from caput import mpiutil
import healpy as hp
import numpy as np
import scipy as sp
import h5py
import copy
import sys
import gc


logger = logging.getLogger(__name__)

class BinOpticalCat(pipeline.OneAndOne, mapbase.MapBase):

    params_init = {

            'tmp_map'   : None,
            'tmp_grid'  : None,
            'ra_label'  : 'RA',
            'dec_label' : 'DEC',
            'z_label'   : 'Z',
            'mock_n'    : 100,
            }
    
    prefix = 'bo_'
    def __init__(self, *arg, **kwargs):

        super(BinOpticalCat, self).__init__(*arg, **kwargs)
        mapbase.MapBase.__init__(self)


    def setup(self):

        '''
        setup the tmp 
        '''

        tmp_map = self.params['tmp_map']
        if tmp_map is not None:
            with h5py.File(tmp_map, 'r') as f:
                tmp = al.make_vect(al.load_h5(f, 'clean_map'))
                self.tmp        = tmp
                self.freq_edges = tmp.get_axis_edges('freq')
                self.ra_edges   = tmp.get_axis_edges('ra')
                self.dec_edges  = tmp.get_axis_edges('dec')
                freq = tmp.get_axis('freq')
        elif self.params['tmp_grid'] is not None:
            with np.load(self.params['tmp_grid']) as f:
                freq = f['freq'] * 1.e-6
                ra   = f['ra']
                dec  = f['dec']

                nfreq = freq.shape[0]
                dfreq = freq[1] - freq[0]

                nra   = ra.shape[0]
                dra   = ra[1]  - ra[0]

                ndec  = dec.shape[0]
                ddec  = dec[1] - dec[0]

                map_shp   = [nfreq, nra, ndec]
                self.tmp  = al.make_vect(np.zeros(map_shp), 
                                         axis_names=('freq', 'ra', 'dec'))
                self.tmp.set_axis_info('freq', freq[nfreq//2], dfreq)
                self.tmp.set_axis_info('ra',   ra[nra//2], dra)
                self.tmp.set_axis_info('dec',  dec[ndec//2], ddec)

                self.freq_edges = self.tmp.get_axis_edges('freq')
                self.ra_edges   = self.tmp.get_axis_edges('ra')
                self.dec_edges  = self.tmp.get_axis_edges('dec')
        else:
            msg = "Need Temp map!!"
            raise ValueError(msg)

        zz_edges = __nu21__ / self.freq_edges - 1.
        zz_centr = __nu21__ / freq - 1.
        r_edges = (cosmology.comoving_distance(zz_edges) * cosmology.h).value
        x_edges = (cosmology.comoving_transverse_distance(zz_centr) * cosmology.h).value
        S = np.abs(self.ra_edges[1] - self.ra_edges[0])\
                * np.abs(self.dec_edges[1] - self.dec_edges[0])
        S = S * ( np.pi / 180. ) ** 2.
        self.pix_volume = (x_edges ** 2) * S * np.abs(r_edges[1:] - r_edges[:-1])

        self.bin_func = bin_catalog_data
        self.ra0 = None
        self.dec0 = None

        self.init_output()

    def init_output(self):

        output_file = self.output_files[0]
        output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
        self.output_file = output_file
        #self.allocate_output(output_file, 'w')
        #self.create_dataset('realmap', self.tmp.shape)

    def process(self, input):

        self.realmap(input[0])
        self.selection(input[1])
        self.separable()
        self.produce_delta_map()
        self.save()


    def read_input(self):

        input = []
        #for input_file in self.input_files:

        # load real catalogue
        c = fits_cat.CAT('', self.input_files[0], '', feedback=0)
        c.ra_label  = self.params['ra_label'] #'RA'
        c.dec_label = self.params['dec_label'] #'DEC'
        c.z_label   = self.params['z_label'] #'Z'

        mask = c.z < 0.02
        mask += ~ ((c.ra < self.ra_edges.max())   * (c.ra > self.ra_edges.min()))
        mask += ~ ((c.dec < self.dec_edges.max()) * (c.dec > self.dec_edges.min()))
        c.data = c.data[~mask]
        c.mask = c.mask[~mask]

        input.append(c)

        # load random catalogue
        if len(self.input_files[1:]) > 0:
            c = fits_cat.CATs('', self.input_files[1:], '', feedback=0)
            c.ra_label  = self.params['ra_label'] #'RA'
            c.dec_label = self.params['dec_label'] #'DEC'
            c.z_label   = self.params['z_label'] #'Z'

            mask  = c.z < 0.02
            mask += ~ ((c.ra < self.ra_edges.max())   * (c.ra > self.ra_edges.min()))
            mask += ~ ((c.dec < self.dec_edges.max()) * (c.dec > self.dec_edges.min()))
            c.data = c.data[~mask]
            c.mask = c.mask[~mask]
            input.append(c)
        else:
            print("shuffle real as mock")
            _mock = []
            for i in range(self.params['mock_n']):
                _data = copy.deepcopy(c.data)
                _z = _data[c.z_label][~c.mask]
                np.random.shuffle(_z)
                _data[c.z_label][~c.mask] = _z
                _mock.append(_data)
            _mock = np.concatenate(_mock, axis=0)
            c_mock = copy.deepcopy(c)
            c_mock.data  = _mock
            c_mock._mask = np.zeros(c_mock.data.shape[0]).astype('bool')
            mask = c_mock.z < 0.02
            c_mock.mask = mask
            print( c_mock.ra.min(), c_mock.ra.max() )
            print( c_mock.z.min(), c_mock.z.max())
            input.append(c_mock)


        return input

    def realmap(self, cat):

        self.realcat_num = cat.ra.shape[0]

        self.realmap_binning = self.bin_func(cat, 
                self.freq_edges, self.ra_edges, self.dec_edges, 
                self.ra0, self.dec0)

        #map_optical = al.make_vect(self.realmap_binning, 
        #        axis_names=('freq', 'ra', 'dec'))
        #map_optical.copy_axis_info(self.tmp)

        #self.df['realmap'][:] = map_optical
        #with h5py.File(self.output_file + 'realmap.h5', 'w') as df:
        #    al.save_h5(df, 'optmap', map_optical)

    def selection(self, cat):

        #self.selection_function = np.zeros(self.tmp.shape)

        mockcat_num = cat.ra.shape[0]
        realcat_num = self.realcat_num
        realcat_in_num = np.sum(self.realmap_binning)
        alpha = int(float(mockcat_num) / float(realcat_num) + 0.1)
        mock_num = min(self.params['mock_n'], alpha)
        print('alpha = %d, mock number = %d, real number = %d(%d)'%(
                alpha, mock_num, realcat_in_num, realcat_num))
        self.mock_num = mock_num
        mockmap_binning = np.zeros((mock_num, ) + self.tmp.shape)
        for i in range(mock_num):
            #sel = (i * realcat_num, (i + 1) * realcat_num, None)
            sel = (i, mock_num * realcat_num,  mock_num)
            mockmap_binning[i, ...] = self.bin_func(cat, 
                self.freq_edges, self.ra_edges, self.dec_edges, 
                self.ra0, self.dec0, sel=sel)

        mockcat_in_num = np.sum(mockmap_binning)
        self.mockmap_binning = mockmap_binning

        self.selection_function = np.sum(mockmap_binning, axis=0)
        # adding the real map back to the selection function is a kludge which
        # ensures the selection function is not zero where there is real data
        # (limit of insufficient mocks)
        self.selection_function = self.selection_function + self.realmap_binning
        self.selection_function /= float(mock_num + 1)

        self.nbar = self.selection_function.copy() / self.pix_volume[:, None, None]

    def separable(self):

        # now assume separability of the selection function
        mask = (self.selection_function != 0).astype('int')
        spatial_selection  = np.sum(self.selection_function, axis=0)
        #norm = np.sum(mask, axis=0) * 1.
        #norm[norm==0] = np.inf
        #spatial_selection /= norm

        #freq_selection  = np.apply_over_axes(np.sum, self.selection_function, [1, 2])
        freq_selection  = np.sum( self.selection_function, axis = (1, 2))
        #norm = np.apply_over_axes(np.sum, mask, [1, 2]) * 1.
        #norm[norm==0] = np.inf
        #freq_selection /= norm
        # smooth freq_selection with polyfit
        _f = np.linspace(0, 1, freq_selection.shape[0])
        freq_selection  = np.poly1d(np.polyfit(_f, freq_selection, 4))(_f)
        freq_selection  = freq_selection[:, None, None]

        self.separable_selection = (freq_selection * spatial_selection)
        self.separable_selection /= np.sum(freq_selection.flatten())

        #norm = np.sum(freq_selection.flatten()) * np.sum(spatial_selection.flatten())
        #self.separable_selection /= norm
        #self.separable_selection = al.make_vect(self.separable_selection, 
        #        axis_names=('freq', 'ra', 'dec')) 
        #self.separable_selection.copy_axis_info(self.tmp)

        #al.save_h5(self.df, 'separable', self.separable_selection)

    def produce_delta_map(self):

        map_real = copy.deepcopy(self.realmap_binning)
        #map_nbar = copy.deepcopy(self.separable_selection)
        map_nbar = copy.deepcopy(self.selection_function)
        bad = map_nbar == 0
        map_nbar[bad] = np.inf

        map_real_delta = map_real / map_nbar - 1.
        #map_real_delta[bad] = 0.
        self.map_real_delta = map_real_delta

        #map_real_delta = al.make_vect(map_real_delta, 
        #        axis_names=('freq', 'ra', 'dec'))
        #map_real_delta.copy_axis_info(self.tmp)
        #al.save_h5(self.df, 'realmap_delta', map_real_delta)

        #for i in range(self.mock_num):

        map_mock = copy.deepcopy(self.mockmap_binning)

        map_mock_delta = map_mock / map_nbar[None, ...] - 1.
        #map_mock_delta[:, bad] = 0.
        self.map_mock_delta = map_mock_delta

        #    map_mock_delta = al.make_vect(map_mock_delta, 
        #            axis_names=('freq', 'ra', 'dec'))
        #    map_mock_delta.copy_axis_info(self.tmp)
        #    al.save_h5(self.df, 'mockmap%03d_delta'%i, map_mock_delta)

    def save(self):

        realmap_binning = al.make_vect(self.realmap_binning, 
                axis_names=('freq', 'ra', 'dec'))
        realmap_binning.copy_axis_info(self.tmp)

        realmap_delta = al.make_vect(self.map_real_delta, 
                axis_names=('freq', 'ra', 'dec'))
        realmap_delta.copy_axis_info(self.tmp)

        selection_function = al.make_vect(self.selection_function, 
                axis_names=('freq', 'ra', 'dec')) 
        selection_function.copy_axis_info(self.tmp)

        nbar = al.make_vect(self.nbar, 
                axis_names=('freq', 'ra', 'dec')) 
        nbar.copy_axis_info(self.tmp)

        separable_selection = al.make_vect(self.separable_selection, 
                axis_names=('freq', 'ra', 'dec')) 
        separable_selection.copy_axis_info(self.tmp)

        with h5py.File(self.output_file + '_realmap.h5', 'w') as df:
            al.save_h5(df, 'optmap',    realmap_binning)
            al.save_h5(df, 'delta',     realmap_delta)
            al.save_h5(df, 'selection', selection_function)
            al.save_h5(df, 'separable', separable_selection)
            al.save_h5(df, 'nbar',      nbar)

        for i in range(self.mock_num):
            mockmap_binning = al.make_vect(self.mockmap_binning[i, ...], 
                    axis_names=('freq', 'ra', 'dec'))
            mockmap_binning.copy_axis_info(self.tmp)

            mockmap_delta = al.make_vect(self.map_mock_delta[i, ...], 
                    axis_names=('freq', 'ra', 'dec'))
            mockmap_delta.copy_axis_info(self.tmp)
            with h5py.File(self.output_file + '_mockmap%03d.h5'%i, 'w') as df:
                al.save_h5(df, 'optmap',    mockmap_binning)
                al.save_h5(df, 'delta',     mockmap_delta)
                al.save_h5(df, 'selection', selection_function)
                al.save_h5(df, 'separable', separable_selection)
                al.save_h5(df, 'nbar',      nbar)


    def finish(self):
        mpiutil.barrier()
        #self.df.close()



class BinOpticalCat_Cube(BinOpticalCat):
    
    prefix = 'boc_'
    
    def setup(self):
        tmp_map = self.params['tmp_map']
        if tmp_map is not None:
            with h5py.File(tmp_map, 'r') as f:
                tmp = al.make_vect(al.load_h5(f, 'clean_map'))
                self.ra0 = tmp.info['ra_centre']
                self.dec0 = tmp.info['dec_centre']
                #tmp = gridding.physical_grid(tmp, refinement=1, order=0)[0]
                tmp = physical_grid(tmp, refinement=refinement, order=order)[0]
                self.tmp        = tmp
                self.freq_edges = tmp.get_axis_edges('freq')
                self.ra_edges   = tmp.get_axis_edges('ra')
                self.dec_edges  = tmp.get_axis_edges('dec')
                
                #print self.freq_edges
                #print self.ra_edges
                #print self.dec_edges
        elif self.params['tmp_grid'] is not None:
            #with np.load(self.params['tmp_grid']) as f:
            with h5py.File(self.params['tmp_grid'], 'r') as f:
                print('load tmp_grid : %s'%self.params['tmp_grid'])
                freq = f['freq'][:] * 1.e-6
                ra   = f['ra'][:]
                dec  = f['dec'][:]

                nfreq = freq.shape[0]
                dfreq = freq[1] - freq[0]

                nra   = ra.shape[0]
                dra   = ra[1]  - ra[0]

                ndec  = dec.shape[0]
                ddec  = dec[1] - dec[0]

                map_shp   = [nfreq, nra, ndec]
                tmp  = al.make_vect(np.zeros(map_shp), axis_names=('freq', 'ra', 'dec'))
                tmp.set_axis_info('freq', freq[nfreq//2], dfreq)
                tmp.set_axis_info('ra',   ra[nra//2], dra)
                tmp.set_axis_info('dec',  dec[ndec//2], ddec)
                self.ra0 = tmp.info['ra_centre']
                self.dec0 = tmp.info['dec_centre']
                #tmp = gridding.physical_grid(tmp, refinement=1, order=0)[0]
                tmp = physical_grid(tmp, refinement=refinement, order=order)[0]
                self.tmp        = tmp
                self.freq_edges = self.tmp.get_axis_edges('freq')
                self.ra_edges   = self.tmp.get_axis_edges('ra')
                self.dec_edges  = self.tmp.get_axis_edges('dec')
        else:
            msg = "Need Temp map!!"
            raise ValueError(msg)
        
        self.bin_func = bin_catalog_data_cube

        self.init_output()
        
def bin_catalog_data(catalog, freq_edges, ra_edges, dec_edges, ra0, dec0, 
        sel=(None, None, None)):
    """
    bin catalog data onto a grid in RA, Dec, and frequency
    This currently assumes that all of the axes are uniformly spaced
    """
    sel = slice(sel[0], sel[1], sel[2])
    num_catalog = catalog.ra[sel].shape[0]
    sample = np.zeros((num_catalog, 3))
    sample[:, 0] = catalog.freq[sel]
    sample[:, 1] = catalog.ra[sel]
    sample[:, 2] = catalog.dec[sel]

    #print freq_edges
    #print ra_edges
    #print dec_edges

    count_cube = histogram3d(sample, freq_edges, ra_edges, dec_edges)
    return count_cube

def bin_catalog_data_cube(catalog, freq_edges, ra_edges, dec_edges, ra0, dec0,
        sel=(None, None, None)):
    """
    bin catalog data onto a grid in RA, Dec, and frequency
    This currently assumes that all of the axes are uniformly spaced
    """

    #cosmology = cosmo.Cosmology()

    sel = slice(sel[0], sel[1], sel[2])
    num_catalog = catalog.ra[sel].shape[0]

    ra_fact = np.cos(dec0 * np.pi / 180.)
    #c   = cosmology.comoving_distance(catalog.z[sel])
    #d   = cosmology.proper_distance(catalog.z[sel])
    z_axis = catalog.z[sel]
    d = (cosmology.comoving_transverse_distance(z_axis) * cosmology.h).value
    c = (cosmology.comoving_distance(z_axis) * cosmology.h).value
    ra  = (catalog.ra[sel] - ra0) * ra_fact
    ra  = ra * np.pi / 180. * d
    dec = catalog.dec[sel] - dec0
    dec = dec * np.pi / 180. * d

    #print c.min(), c.max()
    #print ra.min(), ra.max()
    #print dec.min(), dec.max()
    #print

    sample = np.zeros((num_catalog, 3))
    sample[:, 0] = c
    sample[:, 1] = ra
    sample[:, 2] = dec

    count_cube = histogram3d(sample, freq_edges, ra_edges, dec_edges)
    return count_cube

def histogram3d(sample, xedges, yedges, zedges):
    """Make a 3D histogram from the sample and edge specification
    indices in the sample: 0=x, 1=y, 2=z;
    histogramdd was having problems with the galaxy catalogs
    """
    numcatalog = sample.size
    x_size = xedges.size - 1
    y_size = yedges.size - 1
    z_size = zedges.size - 1
    box_index = np.zeros(numcatalog)
    count_array = np.zeros((x_size + 1) * (y_size + 1) * (z_size + 1))
    # the final array to return is the value within the bin
    count_cube = np.zeros((x_size, y_size, z_size))

    # find which bin each galaxies lies in
    x_index = np.digitize(sample[:, 0], xedges)
    y_index = np.digitize(sample[:, 1], yedges)
    z_index = np.digitize(sample[:, 2], zedges)

    # digitize puts values outside of the bins either to 0 or len(bins)
    x_out = np.logical_or((x_index == 0), (x_index == (x_size + 1)))
    y_out = np.logical_or((y_index == 0), (y_index == (y_size + 1)))
    z_out = np.logical_or((z_index == 0), (z_index == (z_size + 1)))
    # now flag all those point which are inside the region
    box_in = np.logical_not(np.logical_or(np.logical_or(x_out, y_out), z_out))

    # the 0th bin center is recorded in the digitized index=1, so shift
    # also throw out points that are not in the volume
    x_index = x_index[box_in] - 1
    y_index = y_index[box_in] - 1
    z_index = z_index[box_in] - 1

    box_index = x_index + y_index * x_size + z_index * x_size * y_size

    # note that bincount will only count up to the largest object in the list,
    # which may be smaller than the dimension of the full count cube
    try:
        count_array[0:max(box_index) + 1] = np.bincount(box_index)

        # make the x y and z axes which index the bincount output
        count_index = np.arange(x_size * y_size * z_size)
        zind = count_index // (x_size * y_size)
        yind = (count_index - x_size * y_size * zind) // x_size
        xind = count_index - x_size * y_size * zind - x_size * yind

        #count_cube[xind, yind, zind] = count_array[xind + yind * x_size +
        #                                           zind * x_size * y_size]
        count_cube[xind, yind, zind] = count_array[count_index]
        #split_indices = cartesian((np.arange(z_size),
        #                           np.arange(y_size),
        #                           np.arange(x_size)))
        #count_cube[split_indices] = count_array[count_index]
    except MemoryError:
        print("histogram3d: all points out of the volume")

    return count_cube
