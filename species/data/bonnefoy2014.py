"""
Module for adding young, M- and L-type dwarf spectra from
`Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...562A.127B/abstract>`_ to
the database.
"""

import gzip
import os
import shutil
import tarfile
import urllib.request

import h5py
import numpy as np

from astropy.io import fits
from typeguard import typechecked


@typechecked
def add_bonnefoy2014(input_path: str,
                     database: h5py._hl.files.File) -> None:
    """
    Function for adding the SINFONI spectra of young, M- and L-type dwarfs from
    `Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...562A.127B/abstract>`_ to
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

    print_text = 'spectra of young M/L type objects from Bonnefoy et al. 2014'

    data_url = 'http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/A+A/562/A127/'
    data_file = os.path.join(input_path, 'J_A+A_562_A127.tar.gz')
    data_folder = os.path.join(input_path, 'bonnefoy+2014/')

    if not os.path.isfile(data_file):
        print(f'Downloading {print_text} (2.3 MB)...', end='', flush=True)
        urllib.request.urlretrieve(data_url, data_file)
        print(' [DONE]')

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    print(f'Unpacking {print_text} (2.3 MB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    spec_dict = {}

    with gzip.open(os.path.join(data_folder, 'stars.dat.gz'), 'r') as gzip_file:
        for line in gzip_file:
            name = line[:13].decode().strip()
            files = line[80:].decode().strip().split()
            sptype = line[49:56].decode().strip()

            if name == 'NAME 2M1207A':
                name = '2M1207A'

            if len(sptype) == 0:
                sptype = None
            elif '.' in sptype:
                sptype = sptype[:4]
            else:
                sptype = sptype[:2]

            if name == 'Cha1109':
                sptype = 'M9'
            elif name == 'DH Tau B':
                sptype = 'M9'
            elif name == 'TWA 22A':
                sptype = 'M6'
            elif name == 'TWA 22B':
                sptype = 'M6'
            elif name == 'CT Cha b':
                sptype = 'M9'

            spec_dict[name] = {'name': name, 'sptype': sptype, 'files': files}

    database.create_group('spectra/bonnefoy+2014')

    fits_folder = os.path.join(data_folder, 'sp')

    print_message = ''

    for _, _, files in os.walk(fits_folder):
        for _, filename in enumerate(files):
            fname_split = filename.split('_')

            data = fits.getdata(os.path.join(fits_folder, filename))

            for name, value in spec_dict.items():
                if filename in value['files']:
                    if name == 'TWA 22AB':
                        # Binary spectrum
                        continue

                    if 'JHK.fits' in fname_split:
                        spec_dict[name]['JHK'] = data

                    elif 'J' in fname_split:
                        spec_dict[name]['J'] = data

                    elif 'H+K' in fname_split or 'HK' in fname_split:
                        spec_dict[name]['HK'] = data

    for name, value in spec_dict.items():
        empty_message = len(print_message) * ' '
        print(f'\r{empty_message}', end='')

        print_message = f'Adding spectra... {name}'
        print(f'\r{print_message}', end='')

        if 'JHK' in value:
            sp_data = value['JHK']

        elif 'J' in value and 'HK' in value:
            sp_data = np.vstack((value['J'], value['HK']))

        else:
            continue

        dset = database.create_dataset(f'spectra/bonnefoy+2014/{name}', data=sp_data)

        dset.attrs['name'] = str(name).encode()
        dset.attrs['sptype'] = str(value['sptype']).encode()

    empty_message = len(print_message) * ' '
    print(f'\r{empty_message}', end='')

    print_message = 'Adding spectra... [DONE]'
    print(f'\r{print_message}')

    database.close()
