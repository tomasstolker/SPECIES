"""
Module for downloading filter data from the website of the SVO Filter Profile Service.
"""

import os
import warnings
import urllib.request

from typing import Optional, Tuple
from astropy.io.votable import parse_single_table

import numpy as np

from typeguard import typechecked


@typechecked
def download_filter(filter_id: str) -> Tuple[Optional[np.ndarray],
                                             Optional[np.ndarray],
                                             Optional[str]]:
    """
    Function for downloading filter profile data from the SVO Filter Profile Service.

    Parameters
    ----------
    filter_id : str
        Filter name as listed on the website of the SVO Filter Profile Service
        (see http://svo2.cab.inta-csic.es/svo/theory/fps/).

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Fractional transmission.
    str
        Detector type ('energy' or 'photon').
    """

    if filter_id == 'Magellan/VisAO.rp':
        url = 'https://xwcl.science/magao/visao/VisAO_rp_filter_curve.dat'
        urllib.request.urlretrieve(url, 'VisAO_rp_filter_curve.dat')

        wavelength, transmission, _, _ = np.loadtxt('VisAO_rp_filter_curve.dat', unpack=True)

        # Not sure if energy- or photon-counting detector
        det_type = 'energy'

        os.remove('VisAO_rp_filter_curve.dat')

    elif filter_id == 'Magellan/VisAO.ip':
        url = 'https://xwcl.science/magao/visao/VisAO_ip_filter_curve.dat'
        urllib.request.urlretrieve(url, 'VisAO_ip_filter_curve.dat')

        wavelength, transmission, _, _ = np.loadtxt('VisAO_ip_filter_curve.dat', unpack=True)

        # Not sure if energy- or photon-counting detector
        det_type = 'energy'

        os.remove('VisAO_ip_filter_curve.dat')

    elif filter_id == 'Magellan/VisAO.zp':
        url = 'https://xwcl.science/magao/visao/VisAO_zp_filter_curve.dat'
        urllib.request.urlretrieve(url, 'VisAO_zp_filter_curve.dat')

        wavelength, transmission, _, _ = np.loadtxt('VisAO_zp_filter_curve.dat', unpack=True)

        # Not sure if energy- or photon-counting detector
        det_type = 'energy'

        os.remove('VisAO_zp_filter_curve.dat')

    elif filter_id == 'LCO/VisAO.Ys' or filter_id == 'Magellan/VisAO.Ys':
        url = 'https://xwcl.science/magao/visao/VisAO_Ys_filter_curve.dat'
        urllib.request.urlretrieve(url, 'VisAO_Ys_filter_curve.dat')

        wavelength, transmission, _, _ = np.loadtxt('VisAO_Ys_filter_curve.dat', unpack=True)

        # Remove wavelengths with zero transmission
        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

        # Not sure if energy- or photon-counting detector
        det_type = 'energy'

        os.remove('VisAO_Ys_filter_curve.dat')

    elif filter_id == 'ALMA/band6':
        url = 'https://home.strw.leidenuniv.nl/~stolker/species/alma_band6.dat'
        urllib.request.urlretrieve(url, 'alma_band6.dat')

        wavelength, transmission = np.loadtxt('alma_band6.dat', unpack=True)

        det_type = 'energy'

        os.remove('alma_band6.dat')

    elif filter_id == 'ALMA/band7':
        url = 'https://home.strw.leidenuniv.nl/~stolker/species/alma_band7.dat'
        urllib.request.urlretrieve(url, 'alma_band7.dat')

        wavelength, transmission = np.loadtxt('alma_band7.dat', unpack=True)

        det_type = 'energy'

        os.remove('alma_band7.dat')

    else:
        url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID='+filter_id
        urllib.request.urlretrieve(url, 'filter.xml')

        try:
            table = parse_single_table('filter.xml')

            wavelength = table.array['Wavelength']
            transmission = table.array['Transmission']

        except IndexError:
            wavelength = None
            transmission = None
            det_type = None

            warnings.warn(f'Filter \'{filter_id}\' is not available on the SVO Filter Profile '
                          f'Service.')

        except:
            os.remove('filter.xml')

            raise ValueError(f'The filter data of \'{filter_id}\' could not be downloaded. '
                             f'Perhaps the website of the SVO Filter Profile Service '
                             f'(http://svo2.cab.inta-csic.es/svo/theory/fps/) is not '
                             f'available?')

        if transmission is not None:
            try:
                det_type = table.get_field_by_id('DetectorType').value

                # For backward compatibility
                if not isinstance(det_type, str):
                    det_type = det_type.decode('utf-8')

                if int(det_type) == 1:
                    det_type = 'photon'

            except KeyError:
                # Energy-counting detector if the DetectorType key is not present
                det_type = 'energy'

            wavelength *= 1e-4  # (um)

        os.remove('filter.xml')

    if wavelength is not None:
        indices = []

        for i in range(transmission.size):
            if i == 0 and transmission[i] == 0. and transmission[i+1] == 0.:
                indices.append(i)

            elif i == transmission.size-1 and transmission[i-1] == 0. and transmission[i] == 0.:
                indices.append(i)

            elif transmission[i-1] == 0. and transmission[i] == 0. and transmission[i+1] == 0.:
                indices.append(i)

        wavelength = np.delete(wavelength, indices)
        transmission = np.delete(transmission, indices)

        if np.amin(transmission) < 0.:
            warnings.warn(f'The minimum transmission value of {filter_id} is smaller than zero '
                          f'({np.amin(transmission):.2e}). Wavelengths with negative transmission '
                          f'values will be removed.')

            indices = []

            for i, item in enumerate(transmission):
                if item > 0.:
                    indices.append(i)

            wavelength = wavelength[indices]
            transmission = transmission[indices]

    return wavelength, transmission, det_type
