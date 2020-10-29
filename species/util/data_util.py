"""
Utility functions for data processing.
"""

from typing import Optional, List

import h5py
import numpy as np

from typeguard import typechecked

from scipy.interpolate import griddata


def update_sptype(sptypes):
    """
    Function to update a list with spectral types to two characters (e.g., M8, L3, or T1).

    Parameters
    ----------
    sptypes : np.ndarray
        Input spectral types.

    Returns
    -------
    np.ndarray
        Updated spectral types.
    """

    sptype_list = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y']

    for i, spt_item in enumerate(sptypes):
        if spt_item == 'None':
            pass

        elif spt_item == 'null':
            sptypes[i] = 'None'

        else:
            for list_item in sptype_list:
                try:
                    sp_index = spt_item.index(list_item)
                    sptypes[i] = spt_item[sp_index:sp_index+2]

                except ValueError:
                    pass

    return sptypes


def update_filter(filter_in):
    """
    Function to update a filter ID from the Vizier Photometry viewer VOTable to the filter ID from
    the SVO Filter Profile Service.

    Parameters
    ----------
    filter_in : str
        Filter ID in the format of the Vizier Photometry viewer.

    Returns
    -------
    str
        Filter ID in the format of the SVO Filter Profile Service.
    """

    if filter_in[0:5] == b'2MASS':
        filter_out = str(b'2MASS/2MASS.'+filter_in[6:])

    elif filter_in[0:4] == b'WISE':
        filter_out = str(b'WISE/WISE.'+filter_in[5:])

    elif filter_in[0:10] == b'GAIA/GAIA2':
        filter_out = str(filter_in[0:9]+b'0'+filter_in[10:])

    else:
        filter_out = None

    return filter_out


@typechecked
def sort_data(param_teff: np.ndarray,
              param_logg: Optional[np.ndarray],
              param_feh: Optional[np.ndarray],
              param_co: Optional[np.ndarray],
              param_fsed: Optional[np.ndarray],
              wavelength: np.ndarray,
              flux: np.ndarray) -> List[np.ndarray]:
    """
    Parameters
    ----------
    param_teff : np.ndarray
        Array with the effective temperature (K) of each spectrum.
    param_logg : np.ndarray, None
        Array with the log10 surface gravity (cgs) of each spectrum.
    param_feh : np.ndarray, None
        Array with the metallicity of each spectrum. Not used if set to ``None``.
    param_co : np.ndarray, None
        Array with the carbon-to-oxygen ratio of each spectrum. Not used if set to ``None``.
    param_fsed : np.ndarray, None
        Array with the sedimentation parameter of each spectrum. Not used if set to ``None``.
    wavelength : np.ndarray
        Array with the wavelengths (um).
    flux : np.ndarray
        Array with the spectra with dimensions ``(n_spectra, n_wavelengths)``.

    Returns
    -------
    list(np.ndarray, )
        List with the unique values of the atmosphere parameters (each in a separate array), an
        array with the wavelengths, and a multidimensional array with the sorted spectra.
    """

    n_spectra = param_teff.shape[0]

    teff_unique = np.unique(param_teff)
    spec_shape = [teff_unique.shape[0]]

    print('Grid points stored in the database:')
    print(f'   - Teff = {teff_unique}')

    if param_logg is not None:
        logg_unique = np.unique(param_logg)
        spec_shape.append(logg_unique.shape[0])
        print(f'   - log(g) = {logg_unique}')

    if param_feh is not None:
        feh_unique = np.unique(param_feh)
        spec_shape.append(feh_unique.shape[0])
        print(f'   - [Fe/H] = {feh_unique}')

    if param_co is not None:
        co_unique = np.unique(param_co)
        spec_shape.append(co_unique.shape[0])
        print(f'   - C/O = {co_unique}')

    if param_fsed is not None:
        fsed_unique = np.unique(param_fsed)
        spec_shape.append(fsed_unique.shape[0])
        print(f'   - f_sed = {fsed_unique}')

    spec_shape.append(wavelength.shape[0])

    spectrum = np.zeros(spec_shape)

    for i in range(n_spectra):
        # The parameter order is: Teff, log(g), [Fe/H], C/O, f_sed
        # Not all parameters have to be included but the order matters

        index_teff = np.argwhere(teff_unique == param_teff[i])[0][0]
        spec_select = [index_teff]

        if param_logg is not None:
            index_logg = np.argwhere(logg_unique == param_logg[i])[0][0]
            spec_select.append(index_logg)

        if param_feh is not None:
            index_feh = np.argwhere(feh_unique == param_feh[i])[0][0]
            spec_select.append(index_feh)

        if param_co is not None:
            index_co = np.argwhere(co_unique == param_co[i])[0][0]
            spec_select.append(index_co)

        if param_fsed is not None:
            index_fsed = np.argwhere(fsed_unique == param_fsed[i])[0][0]
            spec_select.append(index_fsed)

        spec_select.append(...)

        spectrum[tuple(spec_select)] = flux[i]

    sorted_data = [teff_unique]

    if param_logg is not None:
        sorted_data.append(logg_unique)

    if param_feh is not None:
        sorted_data.append(feh_unique)

    if param_co is not None:
        sorted_data.append(co_unique)

    if param_fsed is not None:
        sorted_data.append(fsed_unique)

    sorted_data.append(wavelength)
    sorted_data.append(spectrum)

    return sorted_data


@typechecked
def write_data(model: str,
               parameters: List[str],
               database: h5py._hl.files.File,
               data_sorted: List[np.ndarray]) -> None:
    """
    Function for writing the model spectra and parameters to the database.

    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : list(str, )
        Model parameters.
    database: h5py._hl.files.File
        Database.
    data_sorted : list(np.ndarray, )
        Sorted model data with the parameter values, wavelength points (um), and flux
        densities (W m-2 um-1).

    Returns
    -------
    NoneType
        None
    """

    n_param = len(parameters)

    if f'models/{model}' in database:
        del database[f'models/{model}']

    dset = database.create_group(f'models/{model}')

    dset.attrs['n_param'] = n_param

    for i, item in enumerate(parameters):
        dset.attrs[f'parameter{i}'] = item

        database.create_dataset(f'models/{model}/{item}',
                                data=data_sorted[i])

    database.create_dataset(f'models/{model}/wavelength',
                            data=data_sorted[n_param])

    database.create_dataset(f'models/{model}/flux',
                            data=data_sorted[n_param+1])


@typechecked
def add_missing(model: str,
                parameters: List[str],
                database: h5py._hl.files.File) -> None:
    """
    Function for adding missing grid points with a linear interpolation.

    Parameters
    ----------
    model : str
        Atmosphere model.
    parameters : list(str, )
        Model parameters.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    print('Number of grid points per parameter:')

    grid_shape = []
    param_data = []

    for i, item in enumerate(parameters):
        grid_shape.append(database[f'models/{model}/{item}'].shape[0])
        param_data.append(np.asarray(database[f'models/{model}/{item}']))
        print(f'   - {item}: {grid_shape[i]}')

    flux = np.asarray(database[f'models/{model}/flux'])  # (W m-1 um-1)
    flux = np.log10(flux)

    count_total = 0
    count_interp = 0
    count_missing = 0

    if len(parameters) == 1:
        # Blackbody spectra
        pass

    elif len(parameters) == 2:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], []]
        new_points = [[], []]

        print('Fix missing grid points with a linear interpolation:')

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if np.isinf(np.sum(flux[i, j, ...])):
                    print('   - ', end='')
                    print(f'{parameters[0]} = {param_data[0][i]}, ', end='')
                    print(f'{parameters[1]} = {param_data[1][j]}')

                    if 0 < i < grid_shape[0]-1:
                        check_low = np.isinf(np.sum(flux[i-1, j, ...]))
                        check_up = np.isinf(np.sum(flux[i+1, j, ...]))

                        # Linear scaling of the intermediate Teff point
                        scaling = (param_data[0][i] - param_data[0][i-1]) / \
                                  (param_data[0][i+1] - param_data[0][i-1])

                        if not check_low and not check_up:
                            flux_low = flux[i-1, j, ...]
                            flux_up = flux[i+1, j, ...]
                            flux[i, j, ...] = flux_low*(1.-scaling) + flux_up*scaling
                            count_interp += 1

                        else:
                            find_missing[i, j] = True

                    else:
                        find_missing[i, j] = True

                else:
                    points[0].append(param_data[0][i])
                    points[1].append(param_data[1][j])

                    values.append(flux[i, j, ...])

                new_points[0].append(param_data[0][i])
                new_points[1].append(param_data[1][j])

                count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        if np.sum(find_missing) > 0:
            flux_int = griddata(points.T, values, new_points.T, method='linear', fill_value=np.nan)

            count = 0

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    if np.isnan(np.sum(flux_int[count, :])):
                        count_missing += 1

                    elif np.isinf(np.sum(flux[i, j, ...])):
                        flux[i, j, :] = flux_int[count, :]
                        count_interp += 1

                    count += 1

            if count_missing > 0:
                print(f'Could not interpolate {count_missing} grid points so storing zeros '
                      f'instead. [WARNING]\nThe grid points that are missing:')

                for i in range(flux_int.shape[0]):
                    if np.isnan(np.sum(flux_int[i, :])):
                        print('   - ', end='')
                        print(f'{parameters[0]} = {new_points[0][i]}, ', end='')
                        print(f'{parameters[1]} = {new_points[1][i]}')

    elif len(parameters) == 3:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], []]
        new_points = [[], [], []]

        print('Fix missing grid points with a linear interpolation:')

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    if np.isinf(np.sum(flux[i, j, k, ...])):
                        print('   - ', end='')
                        print(f'{parameters[0]} = {param_data[0][i]}, ', end='')
                        print(f'{parameters[1]} = {param_data[1][j]}, ', end='')
                        print(f'{parameters[2]} = {param_data[2][k]}')

                        if 0 < i < grid_shape[0]-1:
                            check_low = np.isinf(np.sum(flux[i-1, j, k, ...]))
                            check_up = np.isinf(np.sum(flux[i+1, j, k, ...]))

                            # Linear scaling of the intermediate Teff point
                            scaling = (param_data[0][i] - param_data[0][i-1]) / \
                                      (param_data[0][i+1] - param_data[0][i-1])

                            if not check_low and not check_up:
                                flux_low = flux[i-1, j, k, ...]
                                flux_up = flux[i+1, j, k, ...]
                                flux[i, j, k, ...] = flux_low*(1.-scaling) + flux_up*scaling
                                count_interp += 1

                            else:
                                find_missing[i, j, k] = True

                        else:
                            find_missing[i, j, k] = True

                    else:
                        points[0].append(param_data[0][i])
                        points[1].append(param_data[1][j])
                        points[2].append(param_data[2][k])

                        values.append(flux[i, j, k, ...])

                    new_points[0].append(param_data[0][i])
                    new_points[1].append(param_data[1][j])
                    new_points[2].append(param_data[2][k])

                    count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        if np.sum(find_missing) > 0:
            flux_int = griddata(points.T, values, new_points.T, method='linear', fill_value=np.nan)

            count = 0

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        if np.isnan(np.sum(flux_int[count, :])):
                            count_missing += 1

                        elif np.isinf(np.sum(flux[i, j, k, ...])):
                            flux[i, j, k, :] = flux_int[count, :]
                            count_interp += 1

                        count += 1

            if count_missing > 0:
                print(f'Could not interpolate {count_missing} grid points so storing zeros '
                      f'instead. [WARNING]\nThe grid points that are missing:')

                for i in range(flux_int.shape[0]):
                    if np.isnan(np.sum(flux_int[i, :])):
                        print('   - ', end='')
                        print(f'{parameters[0]} = {new_points[0][i]}, ', end='')
                        print(f'{parameters[1]} = {new_points[1][i]}, ', end='')
                        print(f'{parameters[2]} = {new_points[2][i]}')

    elif len(parameters) == 4:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], [], []]
        new_points = [[], [], [], []]

        print('Fix missing grid points with a linear interpolation:')

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        if np.isinf(np.sum(flux[i, j, k, m, ...])):
                            print('   - ', end='')
                            print(f'{parameters[0]} = {param_data[0][i]}, ', end='')
                            print(f'{parameters[1]} = {param_data[1][j]}, ', end='')
                            print(f'{parameters[2]} = {param_data[2][k]}, ', end='')
                            print(f'{parameters[3]} = {param_data[3][m]}')

                            if 0 < i < grid_shape[0]-1:
                                check_low = np.isinf(np.sum(flux[i-1, j, k, m, ...]))
                                check_up = np.isinf(np.sum(flux[i+1, j, k, m, ...]))

                                # Linear scaling of the intermediate Teff point
                                scaling = (param_data[0][i] - param_data[0][i-1]) / \
                                          (param_data[0][i+1] - param_data[0][i-1])

                                if not check_low and not check_up:
                                    flux_low = flux[i-1, j, k, m, ...]
                                    flux_up = flux[i+1, j, k, m, ...]
                                    flux[i, j, k, m, ...] = flux_low*(1.-scaling) + flux_up*scaling
                                    count_interp += 1

                                else:
                                    find_missing[i, j, k, m] = True

                            else:
                                find_missing[i, j, k, m] = True

                        else:
                            points[0].append(param_data[0][i])
                            points[1].append(param_data[1][j])
                            points[2].append(param_data[2][k])
                            points[3].append(param_data[3][m])

                            values.append(flux[i, j, k, m, ...])

                        new_points[0].append(param_data[0][i])
                        new_points[1].append(param_data[1][j])
                        new_points[2].append(param_data[2][k])
                        new_points[3].append(param_data[3][m])

                        count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        if np.sum(find_missing) > 0:
            flux_int = griddata(points.T, values, new_points.T, method='linear', fill_value=np.nan)

            count = 0

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        for m in range(grid_shape[3]):
                            if np.isnan(np.sum(flux_int[count, :])):
                                count_missing += 1

                            elif np.isinf(np.sum(flux[i, j, k, m, ...])):
                                flux[i, j, k, m, :] = flux_int[count, :]
                                count_interp += 1

                            count += 1

            if count_missing > 0:
                print(f'Could not interpolate {count_missing} grid points so storing zeros '
                      f'instead. [WARNING]\nThe grid points that are missing:')

                for i in range(flux_int.shape[0]):
                    if np.isnan(np.sum(flux_int[i, :])):
                        print('   - ', end='')
                        print(f'{parameters[0]} = {new_points[0][i]}, ', end='')
                        print(f'{parameters[1]} = {new_points[1][i]}, ', end='')
                        print(f'{parameters[2]} = {new_points[2][i]}, ', end='')
                        print(f'{parameters[3]} = {new_points[3][i]}')

        # ran_par_0 = np.random.randint(grid_shape[0], size=1000)
        # ran_par_1 = np.random.randint(grid_shape[1], size=1000)
        # ran_par_2 = np.random.randint(grid_shape[2], size=1000)
        # ran_par_3 = np.random.randint(grid_shape[3], size=1000)
        #
        # for z in range(ran_par_0.shape[0]):
        #     i = ran_par_0[z]
        #     j = ran_par_1[z]
        #     k = ran_par_2[z]
        #     m = ran_par_3[z]
        #
        #     if 0 < i < grid_shape[0]-1:
        #         check_low = np.isinf(np.sum(flux[i-1, j, k, m, ...]))
        #         check_up = np.isinf(np.sum(flux[i+1, j, k, m, ...]))
        #
        #         # Linear scaling of the intermediate Teff point
        #         scaling = (param_data[0][i] - param_data[0][i-1]) / \
        #                   (param_data[0][i+1] - param_data[0][i-1])
        #
        #         if not check_low and not check_up:
        #             flux_low = flux[i-1, j, k, m, ...]
        #             flux_up = flux[i+1, j, k, m, ...]
        #             flux[i, j, k, m, ...] = flux_low*(1.-scaling) + flux_up*scaling

    elif len(parameters) == 5:
        find_missing = np.zeros(grid_shape, dtype=bool)

        values = []
        points = [[], [], [], [], []]
        new_points = [[], [], [], [], []]

        print('Fix missing grid points with a linear interpolation:')

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    for m in range(grid_shape[3]):
                        for n in range(grid_shape[4]):
                            if np.isinf(np.sum(flux[i, j, k, m, n, ...])):
                                print('   - ', end='')
                                print(f'{parameters[0]} = {param_data[0][i]}, ', end='')
                                print(f'{parameters[1]} = {param_data[1][j]}, ', end='')
                                print(f'{parameters[2]} = {param_data[2][k]}, ', end='')
                                print(f'{parameters[3]} = {param_data[3][m]}, ', end='')
                                print(f'{parameters[4]} = {param_data[4][n]}')

                                if 0 < i < grid_shape[0]-1:
                                    check_low = np.isinf(np.sum(flux[i-1, j, k, m, n, ...]))
                                    check_up = np.isinf(np.sum(flux[i+1, j, k, m, n, ...]))

                                    # Linear scaling of the intermediate Teff point
                                    scaling = (param_data[0][i] - param_data[0][i-1]) / \
                                              (param_data[0][i+1] - param_data[0][i-1])

                                    if not check_low and not check_up:
                                        flux_low = flux[i-1, j, k, m, n, ...]
                                        flux_up = flux[i+1, j, k, m, n, ...]
                                        flux[i, j, k, m, n, ...] = flux_low*(1.-scaling) + \
                                            flux_up*scaling
                                        count_interp += 1

                                    else:
                                        find_missing[i, j, k, m, n] = True

                                else:
                                    find_missing[i, j, k, m, n] = True

                            else:
                                points[0].append(param_data[0][i])
                                points[1].append(param_data[1][j])
                                points[2].append(param_data[2][k])
                                points[3].append(param_data[3][m])
                                points[4].append(param_data[4][n])

                                values.append(flux[i, j, k, m, n, ...])

                            new_points[0].append(param_data[0][i])
                            new_points[1].append(param_data[1][j])
                            new_points[2].append(param_data[2][k])
                            new_points[3].append(param_data[3][m])
                            new_points[4].append(param_data[4][n])

                            count_total += 1

        values = np.asarray(values)
        points = np.asarray(points)
        new_points = np.asarray(new_points)

        if np.sum(find_missing) > 0:
            flux_int = griddata(points.T, values, new_points.T, method='linear', fill_value=np.nan)

            count = 0

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        for m in range(grid_shape[3]):
                            for n in range(grid_shape[4]):
                                if np.isnan(np.sum(flux_int[count, :])):
                                    count_missing += 1

                                elif np.isinf(np.sum(flux[i, j, k, m, n, ...])):
                                    flux[i, j, k, m, n, :] = flux_int[count, :]
                                    count_interp += 1

                                count += 1

            if count_missing > 0:
                print(f'Could not interpolate {count_missing} grid points so storing zeros '
                      f'instead. [WARNING]\nThe grid points that are missing:')

                for i in range(flux_int.shape[0]):
                    if np.isnan(np.sum(flux_int[i, :])):
                        print('   - ', end='')
                        print(f'{parameters[0]} = {new_points[0][i]}, ', end='')
                        print(f'{parameters[1]} = {new_points[1][i]}, ', end='')
                        print(f'{parameters[2]} = {new_points[2][i]}, ', end='')
                        print(f'{parameters[3]} = {new_points[3][i]}, ', end='')
                        print(f'{parameters[4]} = {new_points[4][i]}')

    else:
        raise ValueError('The add_missing function is currently not compatible with more than 5 '
                         'model parameters.')

    print(f'Number of stored grid points: {count_total}')
    print(f'Number of interpolated grid points: {count_interp}')
    print(f'Number of missing grid points: {count_missing}')

    del database[f'models/{model}/flux']
    database.create_dataset(f'models/{model}/flux', data=10.**flux)


def correlation_to_covariance(cor_matrix,
                              spec_sigma):
    """
    Parameters
    ----------
    cor_matrix : np.ndarray
        Correlation matrix of the spectrum.
    spec_sigma : np.ndarray
        Uncertainties (W m-2 um-1).

    Returns
    -------
    np.ndarrays
        Covariance matrix of the spectrum.
    """

    cov_matrix = np.zeros(cor_matrix.shape)

    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            cov_matrix[i, j] = cor_matrix[i, j]*spec_sigma[i]*spec_sigma[j]

            if i == j:
                assert cor_matrix[i, j] == 1.

    return cov_matrix
