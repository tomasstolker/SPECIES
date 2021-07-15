"""
Module for adding O5 through L3 SDSS stellar spectra from Kesseli et al. (2017) to the database.
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
def add_kesseli2017(input_path: str,
                    database: h5py._hl.files.File) -> None:
    """
    Function for adding the SDSS stellar spectra from Kesseli et al. (2017) to the database.

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

    data_url = 'https://cdsarc.unistra.fr/viz-bin/nph-Cat/tar.gz?J/ApJS/230/16'
    data_file = os.path.join(input_path, 'J_ApJS_230_16.tar.gz')
    data_folder = os.path.join(input_path, 'kesseli+2017/')

    if not os.path.isfile(data_file):
        print('Downloading SDSS spectra from Kesseli et al. 2017 (145 MB)...', end='', flush=True)
        urllib.request.urlretrieve(data_url, data_file)
        print(' [DONE]')

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    print('Unpacking SDSS spectra from Kesseli et al. 2017 (145 MB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    database.create_group('spectra/kesseli+2017')

    fits_folder = os.path.join(data_folder, 'fits')

    print_message = ''

    for _, _, files in os.walk(fits_folder):
        for _, filename in enumerate(files):
            with fits.open(os.path.join(fits_folder, filename)) as hdu_list:
                data = hdu_list[1].data

                wavelength = 1e-4*10.**data['LogLam']  # (um)
                flux = data['Flux']  # Normalized units
                error = data['PropErr']  # Normalized units

                name = filename[:-5].replace('_', ' ')

                file_split = filename.split('_')
                file_split = file_split[0].split('.')

                sptype = file_split[0]

                spdata = np.column_stack([wavelength, flux, error])

                empty_message = len(print_message) * ' '
                print(f'\r{empty_message}', end='')

                print_message = f'Adding spectra... {name}'
                print(f'\r{print_message}', end='')

                dset = database.create_dataset(f'spectra/kesseli+2017/{name}', data=spdata)

                dset.attrs['name'] = str(name).encode()
                dset.attrs['sptype'] = str(sptype).encode()

    empty_message = len(print_message) * ' '
    print(f'\r{empty_message}', end='')

    print_message = 'Adding spectra... [DONE]'
    print(f'\r{print_message}')

    database.close()
