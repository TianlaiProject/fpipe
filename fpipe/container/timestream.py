"""Container class for the MeerKAT  timestream data.


Inheritance diagram
-------------------

.. inheritance-diagram:: tlpipe.container.container.BasicTod tlpipe.container.timestream_common.TimestreamCommon tlpipe.container.raw_timestream.RawTimestream Timestream
   :parts: 2

"""

from tlpipe.container import timestream
from caput import mpiarray
from caput import memh5
import numpy as np

class FAST_Timestream(timestream.Timestream):
    """Container class for the timestream data.

    This timestream data container is to hold time stream data that has polarization
    and baseline separated from the channel pairs in the raw timestream.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = ('time', 'frequency', 'polarization', 'baseline')
    _main_axes_ordered_datasets_ = { 'vis': (0, 1, 2, 3),
                                     'vis_mask': (0, 1, 2, 3),
                                     'sec1970': (0,),
                                     'jul_date': (0,),
                                     'freq': (1,),
                                     'pol': (2,),
                                     'blorder': (3,),
                                     'ra' : (0,3), # for meerKAT
                                     'dec': (0,3), # for meerKAT
                                     'az' : (0,3), # for meerKAT
                                     'el' : (0,3), # for meerKAT
                                     'flags' : (0, 1, 2, 3), # for meerKAT
                                     'ns_on' : (0, 3), # for meerKAT
                                   }
    _time_ordered_datasets_ = {'weather': (0,)}
    _time_ordered_attrs_ = {}
    _feed_ordered_datasets_ = { 'antpointing': (None, 0),
                                'feedno': (0,),
                                'feedpos': (0,),
                                'polerr': (0,),
                              }

    pol_dict = {0: 'hh', 1: 'vv', 2: 'hv', 3: 'vh', 4: 'I', 5: 'Q', 6: 'U', 7: 'V',
                 'hh': 0, 'vv': 1, 'hv': 2, 'vh': 3, 'I': 4, 'Q': 5, 'U': 6, 'V':7}

    @property
    def is_dish(self):
        return True

    def lin2I(self):
        """Convert the linear polarized data to Stokes I only."""
        try:
            pol = self.pol
        except KeyError:
            raise RuntimeError('Polarization of the data is unknown, can not convert')

        try:
            pol_type = pol.attrs['pol_type']
        except KeyError:
            Warning('pol_type is not recorded, assume linear')
            pol.attrs['pol_type'] = 'linear'

        if pol.attrs['pol_type'] == 'stokes' and pol.shape[0] == 4:
            warning.warn('Data is already Stokes polarization, no need to convert')
            return

        if pol.attrs['pol_type'] == 'linear' and pol.shape[0] >=2:

            # redistribute to 0 axis if polarization is the distributed axis
            original_dist_axis = self.main_data_dist_axis
            if 'polarization' == self.main_data_axes[self.main_data_dist_axis]:
                self.redistribute(0)

            pol = pol[:].tolist()
            p = self.pol_dict
            if pol[0] == b'hh':
                # for some reason, pol attr recorded in MeerKAT data are
                # ['hh', 'vv', 'hv', 'vh'], change it back to [0, 1, 2, 3]
                pol = [p[_p.decode()] for _p in pol]

            # create a new MPIArray to hold the new data
            shp = self.main_data.shape
            shp = shp[:-2] + (1, ) + shp[-1:]
            md = mpiarray.MPIArray(shp, axis=self.main_data_dist_axis, 
                    comm=self.comm, dtype=self.main_data.dtype)
            # convert to Stokes I, Q, U, V
            #print "convert to Stokes I, Q, U, V "
            md.local_array[:, :, 0] = 0.5 * ( 
                      self.main_data.local_data[:, :, pol.index(p['hh'])]\
                    + self.main_data.local_data[:, :, pol.index(p['vv'])]) # I

            attr_dict = {} # temporarily save attrs of this dataset
            memh5.copyattrs(self.main_data.attrs, attr_dict)
            del self[self.main_data_name]
            # create main data
            self.create_dataset(self.main_data_name, shape=md.shape, 
                    dtype=md.dtype, data=md, distributed=True, 
                    distributed_axis=self.main_data_dist_axis)
            memh5.copyattrs(attr_dict, self.main_data.attrs)

            if self.main_data_name + '_mask' in list(self.keys()):
                mk = mpiarray.MPIArray(shp, axis=self.main_data_dist_axis, 
                        comm=self.comm, dtype=self[self.main_data_name + '_mask'].dtype)
                mk.local_array[:, :, 0] = \
                self[self.main_data_name + '_mask'].local_data[:, :, pol.index(p['hh'])]\
                + self[self.main_data_name + '_mask'].local_data[:, :, pol.index(p['vv'])] 
                attr_dict = {} # temporarily save attrs of this dataset
                memh5.copyattrs(self[self.main_data_name + '_mask'].attrs, attr_dict)
                del self[self.main_data_name + '_mask']
                # create main data mask
                self.create_dataset(self.main_data_name + '_mask', shape=md.shape, 
                        dtype=mk.dtype, data=mk, distributed=True, 
                        distributed_axis=self.main_data_dist_axis)
                memh5.copyattrs(attr_dict, self[self.main_data_name + '_mask'].attrs)

            del self['pol']
            self.create_dataset('pol', data=np.array(['I', ]), dtype='S1')
            self['pol'].attrs['pol_type'] = 'stokes'

            # redistribute self to original axis
            self.redistribute(original_dist_axis)

        else:
            raise RuntimeError('Can not convert to Stokes polarization')

