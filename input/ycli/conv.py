# -*- mode: python; -*-
#
import numpy as np
from fpipe.timestream import data_conv as dc
from fpipe.utils import coord

from scipy.interpolate import interp1d

coord_path = '/idia/users/ycli/fdata/coord/'
coord_file = 'SDSS-0508-01_2020_05_08_18_00_00_000.xlsx'

data_path = '/idia/users/ycli/fdata/raw/SDSS-0508-01/20200508/'
data_file = 'SDSS-0508-01_arcdrift-M%02d_W_%04d.fits'

output_path = '/scratch/users/ycli/fanalysis/raw/'
output_name = 'SDSS_N_2.5/20200508/SDSS_N_2.5_arcdrift%04d-%04d.h5'

beam_list = range(1, 20)[:2]
block_list = [1,]
fmin = 1050
fmax = 1430

time, az, alt = coord.xyz2azalt(coord_path + coord_file)

az_f  = interp1d(time.unix, az )
alt_f = interp1d(time.unix, alt)

for b in block_list:

    _out = output_path + output_name%(b, b)

    dc.convert_to_tl(data_path, data_file, _out, alt_f, az_f, 
            feed_rotation=23.4, beam_list = beam_list, block_list = [b, ],
            fmin=fmin, fmax=fmax)
