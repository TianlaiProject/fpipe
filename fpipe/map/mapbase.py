#! 
import logging
import numpy as np
import healpy as hp
import h5py

from caput import mpiutil

import os, fcntl

logger = logging.getLogger(__name__)

if h5py.get_config().mpi:
    h5_kwargs = {
            'driver' : 'mpio',
            'comm' : mpiutil._comm,
            }
else:
    h5_kwargs = {
            'driver' : None,
            }

class MapBase(object):

    #def __init__(self, *args, **kwargs):

    #    self.df = None
    df = None

    def __del__(self):

        if self.df is not None:
            self.df.close()

    def allocate_output(self, fname, mode='w'):

        logger.debug('allocate outpuf file %s'%fname)

        self.df = h5py.File(fname, mode, **h5_kwargs)

    def create_dataset(self, name, dset_shp, dset_info={}, dtype='f'):

        d = self.df.create_dataset(name, dset_shp, dtype=dtype)
        for key, value in dset_info.items():
            d.attrs[key] = repr(value)

    def create_dataset_like(self, name, dset_tmp):

        self.create_dataset(name, dset_tmp.shape, dset_tmp.info, dset_tmp.dtype)

    def write_block_to_dset(self, dset_name, indx, data, chunk_size=1024**3):

        _write_block_to_dset(self.df, dset_name, indx, data, chunk_size=chunk_size)

    def read_block_from_dset(self, dset_name, indx, data, chunk_size=1024**3):

        _read_block_from_dset(self.df, dset_name, indx, data, chunk_size=1024**3)


class MultiMapBase(object):

    #df_out = []
    #df_in  = []

    #def __init__(self,):

    #    self.df_in  = []
    #    self.df_out = []

    df_in  = []
    df_out = []

    def __del__(self):

        for df in self.df_out:
            df.close()
            self.df_out.remove(df)

        for df in self.df_in:
            df.close()
            self.df_in.remove(df)


    def open(self, fname, mode='r'):

        self.df_in += [h5py.File(fname, mode, **h5_kwargs), ]

    def allocate_output(self, fname, mode='w'):

        self.df_out += [h5py.File(fname, mode, **h5_kwargs), ]

    def create_dataset(self, df_idx, name, dset_shp, dset_info={}, dtype='f'):

        d = self.df_out[df_idx].create_dataset(name, dset_shp, dtype=dtype)
        for key, value in dset_info.items():
            d.attrs[key] = repr(value)

    def create_dataset_like(self, df_idx, name, dset_tmp):

        self.create_dataset(df_idx, name, dset_tmp.shape, dset_tmp.info, dset_tmp.dtype)

    def write_block_to_dset(self, df_idx, dset_name, indx, data, chunk_size=1024**3):

        _write_block_to_dset(self.df_out[df_idx], dset_name, indx, data,
                chunk_size=chunk_size)

    def read_block_from_dset(self, df_idx, dset_name, indx, data, chunk_size=1024**3):

        _read_block_from_dset(self.df_in[df_idx], dset_name, indx, data,
                chunk_size= chunk_size )


def _read_block_from_dset(df, dset_name, indx, data, chunk_size=1024**3):

    dset = df[dset_name]
    dset_off = dset.id.get_offset()

    block_siz = data.size * data.itemsize
    block_off = np.sum([indx[i] * np.prod(dset.shape[i+1:]) * data.itemsize 
        for i in range(len(indx))])
    total_off = dset_off + block_off
    msg = 'rank: %03d | block size: %12.6f MB | item size: %2d | block off: %6dB'%(
            mpiutil.rank, block_siz/ (1024*1024.), data.itemsize, block_off)
    logger.debug(msg)

    _f = os.open(str(df.filename), os.O_RDWR | os.O_CREAT )
    fcntl.lockf(_f, fcntl.LOCK_EX, block_siz, total_off , os.SEEK_SET)
    os.lseek(_f, total_off, os.SEEK_SET)
    for ii in range(0, block_siz, chunk_size):
        read_size = min(chunk_size, block_siz - ii)
        _s = slice(ii//data.itemsize, (ii+read_size)//data.itemsize)
        data.flat[_s] += np.frombuffer(os.read(_f, read_size), dtype=dset.dtype)

    fcntl.lockf(_f, fcntl.LOCK_UN)
    os.close(_f)


def _write_block_to_dset(df, dset_name, indx, data, chunk_size=1024**3):

    dset = df[dset_name]
    dset_off = dset.id.get_offset()

    block_siz = data.size * data.itemsize
    block_off = np.sum([indx[i] * np.prod(dset.shape[i+1:]) * data.itemsize 
        for i in range(len(indx))])
    total_off = dset_off + block_off
    msg = 'rank: %03d | dset offset: %12dB'%(mpiutil.rank, dset_off)
    logger.debug(msg)
    msg = 'rank: %03d | block size: %12.6f MB | item size: %2d | block off: %6dB'%(
            mpiutil.rank, block_siz/ (1024*1024.), data.itemsize, block_off)
    logger.debug(msg)

    _f = os.open(str(df.filename), os.O_RDWR | os.O_CREAT )
    fcntl.lockf(_f, fcntl.LOCK_EX, block_siz, total_off , os.SEEK_SET)
    os.lseek(_f, total_off, os.SEEK_SET)
    try:
        for ii in range(0, block_siz, chunk_size):
            read_size = min(chunk_size, block_siz - ii)
            _s = slice(ii//data.itemsize, (ii+read_size)//data.itemsize)
            data.flat[_s] += np.frombuffer(os.read(_f, read_size), dtype=dset.dtype)
    except ValueError as err:
        msg = 'empty array, igore read'
        logger.debug(msg)

    #buf = buffer(data)
    os.lseek(_f, total_off, os.SEEK_SET)
    for ii in range(0, block_siz, chunk_size):
        read_size = min(chunk_size, block_siz - ii)
        _s = slice(ii//data.itemsize, (ii+read_size)//data.itemsize)
        #buf = buffer(data.flat[_s])
        buf = memoryview(data.flat[_s])
        os.write(_f, buf)
    fcntl.lockf(_f, fcntl.LOCK_UN)
    os.close(_f)


