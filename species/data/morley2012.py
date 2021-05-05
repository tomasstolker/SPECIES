"""
Module for T/Y dwarf model spectra from Morley et al. (2012).
"""

import os
import tarfile
import urllib.request

from typing import Optional, Tuple

import h5py
import spectres
import numpy as np

from typeguard import typechecked

from species.util import data_util, read_util


@typechecked
def add_morley2012(input_path: str,
                   database: h5py._hl.files.File,
                   wavel_range: Optional[Tuple[float, float]],
                   teff_range: Optional[Tuple[float, float]],
                   spec_res: Optional[float]) -> None:
    """
    Function for adding the T/Y dwarf model spectra from Morley et al. (2012) to the database.
    The spectra have been at solar metallicity in the Teff range from 500 to 1300 K. The spectra
    have been downloaded from the Theoretical spectra web server
    (http://svo2.cab.inta-csic.es/svo/theory/newov2/index.php?models=morley12) and resampled
    to a spectral resolution of 5000 from 0.6 to 30 um.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float), None
        Wavelength range (um). The original wavelength points are used if set to ``None``.
    teff_range : tuple(float, float), None
        Effective temperature range (K). All temperatures are selected if set to ``None``.
    spec_res : float, None
        Spectral resolution for resampling. Not used if ``wavel_range`` is set to ``None``.

    Returns
    -------
    NoneType
        None
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    input_file = 'morley2012.tgz'

    data_folder = os.path.join(input_path, 'morley2012/')
    data_file = os.path.join(input_path, input_file)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    url = 'https://home.strw.leidenuniv.nl/~stolker/species/morley2012.tgz'

    if not os.path.isfile(data_file):
        print('Downloading Morley et al. (2012) model spectra (141 MB)...', end='', flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(' [DONE]')

    print('Unpacking Morley et al. (2012) model spectra (141 MB)...', end='', flush=True)
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(' [DONE]')

    teff = []
    logg = []
    fsed = []
    flux = []

    if wavel_range is not None and spec_res is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
    else:
        wavelength = None

    print_message = ''

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):
            if filename[:11] == 'morley2012_':
                file_split = filename.split('_')

                teff_val = float(file_split[2])
                logg_val = float(file_split[4])
                fsed_val = float(file_split[6])

                if teff_range is not None:
                    if teff_val < teff_range[0] or teff_val > teff_range[1]:
                        continue

                empty_message = len(print_message) * ' '
                print(f'\r{empty_message}', end='')

                print_message = f'Adding Morley et al. (2012) model spectra... {filename}'
                print(f'\r{print_message}', end='')

                data_wavel, data_flux = np.loadtxt(os.path.join(data_folder, filename), unpack=True)

                teff.append(teff_val)
                logg.append(logg_val)
                fsed.append(fsed_val)

                if wavel_range is None or spec_res is None:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError('The wavelengths are not all sorted by increasing value.')

                    flux.append(data_flux)  # (W m-2 um-1)

                else:
                    flux_resample = spectres.spectres(wavelength,
                                                      data_wavel,
                                                      data_flux,
                                                      spec_errs=None,
                                                      fill=np.nan,
                                                      verbose=False)

                    if np.isnan(np.sum(flux_resample)):
                        raise ValueError(f'Resampling is only possible if the new wavelength '
                                         f'range ({wavelength[0]} - {wavelength[-1]} um) falls '
                                         f'sufficiently far within the wavelength range '
                                         f'({data_wavel[0]} - {data_wavel[-1]} um) of the input '
                                         f'spectra.')

                    flux.append(flux_resample)  # (W m-2 um-1)

    empty_message = len(print_message) * ' '
    print(f'\r{empty_message}', end='')

    print_message = 'Adding Morley et al. (2012) model spectra... [DONE]'
    print(f'\r{print_message}', end='')

    data_sorted = data_util.sort_data(np.asarray(teff),
                                      np.asarray(logg),
                                      None,
                                      None,
                                      np.asarray(fsed),
                                      wavelength,
                                      np.asarray(flux))

    data_util.write_data('morley-2012',
                         ['teff', 'logg', 'fsed'],
                         database,
                         data_sorted)
