"""
Module for adding young, M- and L-type dwarf spectra from
`Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...772...79A/abstract>`_ to the
database. All  spectra are also available in the
`SpeX Prism Library Analysis Toolkit <https://github.com/aburgasser/splat>`_.
"""

import os
import shutil
import tarfile
import urllib.request

import h5py
import numpy as np

from astropy.io import fits
from typeguard import typechecked


@typechecked
def add_allers2013(input_path: str,
                   database: h5py._hl.files.File) -> None:
    """
    Function for adding the spectra of young, M- and L-type dwarfs from
    `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...772...79A/abstract>`_  to
    the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        The HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    print_text = 'spectra of young M/L type objects from Allers & Liu 2013'

    data_url = 'https://home.strw.leidenuniv.nl/~stolker/species/allers_liu_2013.tgz'
    data_file = os.path.join(input_path, 'allers_liu_2013.tgz')
    data_folder = os.path.join(input_path, 'allers+2014/')

    if not os.path.isfile(data_file):
        print(f'Downloading {print_text} (173 kB)...', end='', flush=True)
        urllib.request.urlretrieve(data_url, data_file)
        print(' [DONE]')

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    print(f'Unpacking {print_text} (173 kB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    sources = np.genfromtxt(os.path.join(data_folder, 'sources.csv'),
                            delimiter=',', dtype=None, encoding='ASCII')

    source_names = sources[:, 0]
    source_sptype = sources[:, 7]

    database.create_group('spectra/allers+2013')

    print_message = ''

    for _, _, files in os.walk(data_folder):
        for _, filename in enumerate(files):
            if filename.endswith('.fits'):
                sp_data, header = fits.getdata(os.path.join(data_folder, filename), header=True)

            else:
                continue

            sp_data = np.transpose(sp_data)

            # (erg s-1 cm-2 A-1) -> (W m-2 um-1)
            sp_data[:, 1:] *= 10.

            name = header['OBJECT']

            index = np.argwhere(source_names == name)

            if len(index) == 0:
                sptype = None
            else:
                sptype = source_sptype[index][0][0][:2]

            empty_message = len(print_message) * ' '
            print(f'\r{empty_message}', end='')

            print_message = f'Adding spectra... {name}'
            print(f'\r{print_message}', end='')

            dset = database.create_dataset(f'spectra/allers+2013/{name}', data=sp_data)

            dset.attrs['name'] = str(name).encode()
            dset.attrs['sptype'] = str(sptype).encode()

    empty_message = len(print_message) * ' '
    print(f'\r{empty_message}', end='')

    print_message = 'Adding spectra... [DONE]'
    print(f'\r{print_message}')

    database.close()
