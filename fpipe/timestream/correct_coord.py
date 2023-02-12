from fpipe.timestream import timestream_task
from caput import mpiarray
from fpipe.utils import coord
from scipy.interpolate import interp1d


class Correct_Coord(timestream_task.TimestreamTask):
    """
    """

    params_init = {
            'coord_file': '',
            'feed_rotation': 23.4,

            }

    prefix = 'cc_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        
        coord_file = self.params['coord_file']
        feed_rotation = self.params['feed_rotation']
        time_coord, az_coord, alt_coord = coord.xyz2azalt(coord_file)
        
        az = az_coord.copy()
        alt= alt_coord.copy()

        az_f  = interp1d(time_coord.unix, az , kind='nearest', 
                 bounds_error=False, fill_value="extrapolate")
        alt_f = interp1d(time_coord.unix, alt, kind='nearest', 
                 bounds_error=False, fill_value="extrapolate")
        
        time = ts['sec1970'][:].local_array

        alt0 = alt_f(time)
        az0 = az_f(time)
        az, alt, ra_new, dec_new = coord.get_pointing_any_scan(time, alt0, az0, 
                                time_format='unix', feed_rotation=feed_rotation)

        dec_new = mpiarray.MPIArray.wrap(dec_new, 1)
        ra_new = mpiarray.MPIArray.wrap(ra_new, 1)
        
        ts.delete_a_dataset('dec')
        ts.create_time_and_bl_ordered_dataset('dec', dec_new)
        
        ts.delete_a_dataset('ra')
        ts.create_time_and_bl_ordered_dataset('ra', ra_new)

        return super(Correct_Coord, self).process(ts)
