"""
Module with a frontend for atmospheric retrieval with the radiative transfer code
``petitRADTRANS`` (see https://petitradtrans.readthedocs.io).
"""

import os
import json
import time

from math import isclose
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pymultinest

from molmass import Formula
from scipy.integrate import simps
from scipy.stats import invgamma
from typeguard import typechecked

from species.analysis import photometry
from species.core import constants
from species.data import database
from species.read import read_filter, read_object
from species.util import dust_util, retrieval_util


os.environ['OMP_NUM_THREADS'] = '1'


class AtmosphericRetrieval:
    """
    Class for atmospheric retrieval with ``petitRADTRANS`` and Bayesian inference.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 line_species: Optional[List[str]],
                 cloud_species: Optional[List[str]],
                 output_folder: str,
                 wavel_range: Optional[Tuple[float, float]],
                 scattering: bool = True,
                 inc_spec: Union[bool, List[str]] = True,
                 inc_phot: Union[bool, List[str]] = False,
                 pressure_grid: str = 'smaller',
                 weights: Optional[Dict[str, float]] = None,
                 lbl_species: Optional[List[str]] = None) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        line_species : list, None
            List with the line species. No line species are used if set to ``None``.
        cloud_species : list, None
            List with the cloud species. No cloud species are used if set to ``None``.
        output_folder : str
            Folder name that is used for the output files from ``MultiNest``. The folder is created
            if it does not exist.
        wavel_range : tuple(float, float), None
            The wavelength range (um) of the forward model. Should be a bit broader than the
            minimum and maximum wavelength of the data. The wavelength range is set automatically
            if the argument is set to ``None``.
        scattering : bool
            Include scattering in the radiative transfer. Only recommended at infrared wavelengths
            when clouds are included. Using scattering will increase the computation time.
        inc_spec : bool, list(str)
            Include spectroscopic data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of spectrum names (as stored
            in the database with :func:`~species.data.database.Database.add_object`) can be
            provided.
        inc_phot : bool, list(str)
            Include photometric data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of filter names (as stored in
            the database) can be provided.
        pressure_grid : str
            The type of pressure grid that is used for the radiative transfer. Either 'standard',
            to use 180 layers both for the atmospheric structure (e.g. when interpolating the
            abundances) and 180 layers with the radiative transfer, or 'smaller' to use 60 (instead
            of 180) with the radiative transfer, or 'clouds' to start with 1440 layers but resample
            to ~100 layers (depending on the number of cloud species) with a refinement around the
            cloud decks. For cloudless atmospheres it is recommended to use 'smaller', which runs
            faster than 'standard' and provides sufficient accuracy. For cloudy atmosphere, one can
            test with 'smaller' but it is recommended to use 'clouds' for improved accuracy fluxes.
        weights : dict(str, float), None
            Weights to be applied to the log-likelihood components of the different spectroscopic
            and photometric data that are provided with ``inc_spec`` and ``inc_phot``. This
            parameter can for example be used to bias the weighting of the photometric data points.
            An equal weighting is applied if the argument is set to ``None``.
        lbl_species : list, None
            List with the line species that will be used for calculating line-by-line spectra for
            the list of high-resolution spectra that are provided as argument of ``cross_corr``
            when running :func:`species.analysis.retrieval.AtmosphericRetrieval.run_multinest`. The
            argument can be set to ``None`` when ``cross_corr=None``.

        Returns
        -------
        NoneType
            None
        """

        # Input parameters

        self.object_name = object_name
        self.line_species = line_species
        self.cloud_species = cloud_species
        self.scattering = scattering
        self.output_folder = output_folder
        self.pressure_grid = pressure_grid
        self.lbl_species = lbl_species

        # Get object data

        self.object = read_object.ReadObject(self.object_name)
        self.distance = self.object.get_distance()[0]  # (pc)

        print(f'Object: {self.object_name}')
        print(f'Distance: {self.distance}')

        # Line species

        if self.line_species is None:
            print('Line species: None')
            self.line_species = []

        else:
            print('Line species:')
            for item in self.line_species:
                print(f'   - {item}')

        # Cloud species

        if self.cloud_species is None:
            print('Cloud species: None')
            self.cloud_species = []

        else:
            print('Cloud species:')
            for item in self.cloud_species:
                print(f'   - {item}')

        # Line species (high-resolution / line-by-line)

        if self.lbl_species is None:
            print('Line-by-line species: None')
            self.lbl_species = []

        else:
            print('Line-by-line species:')
            for item in self.lbl_species:
                print(f'   - {item}')

        # Scattering

        print(f'Scattering: {self.scattering}')

        # Get ObjectBox

        species_db = database.Database()

        objectbox = species_db.get_object(object_name,
                                          inc_phot=True,
                                          inc_spec=True)

        # Copy the cloud species into a new list because the values will be adjusted by Radtrans

        self.cloud_species_full = self.cloud_species.copy()

        # Get photometric data

        self.objphot = []
        self.synphot = []

        if isinstance(inc_phot, bool):
            if inc_phot:
                # Select all filters if True
                species_db = database.Database()
                inc_phot = objectbox.filters

            else:
                inc_phot = []

        if len(objectbox.filters) != 0:
            print('Photometric data:')

        for item in inc_phot:
            obj_phot = self.object.get_photometry(item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

            print(f'   - {item} (W m-2 um-1) = {obj_phot[2]:.2e} +/- {obj_phot[3]:.2e}')

            sphot = photometry.SyntheticPhotometry(item)
            self.synphot.append(sphot)

        # Get spectroscopic data

        if isinstance(inc_spec, bool):
            if inc_spec:
                # Select all filters if True
                species_db = database.Database()
                inc_spec = list(objectbox.spectrum.keys())

            else:
                inc_spec = []

        if inc_spec:
            # Select all spectra
            self.spectrum = self.object.get_spectrum()

            # Select the spectrum names that are not in inc_spec
            spec_remove = []
            for item in self.spectrum:
                if item not in inc_spec:
                    spec_remove.append(item)

            # Remove the spectra that are not included in inc_spec
            for item in spec_remove:
                del self.spectrum[item]

        if not inc_spec or self.spectrum is None:
            raise ValueError('At least one spectrum is required for AtmosphericRetrieval. Please '
                             'add a spectrum with the add_object method of Database. ')

        # Set wavelength bins and add to spectrum dictionary

        self.wavel_min = []
        self.wavel_max = []

        print('Spectroscopic data:')

        for key, value in self.spectrum.items():
            dict_val = list(value)
            wavel_data = dict_val[0][:, 0]

            wavel_bins = np.zeros_like(wavel_data)
            wavel_bins[:-1] = np.diff(wavel_data)
            wavel_bins[-1] = wavel_bins[-2]

            dict_val.append(wavel_bins)
            self.spectrum[key] = dict_val

            # Min and max wavelength for the Radtrans object

            self.wavel_min.append(wavel_data[0])
            self.wavel_max.append(wavel_data[-1])

            print(f'   - {key}')
            print(f'     Wavelength range (um) = {wavel_data[0]:.2f} - {wavel_data[-1]:.2f}')
            print(f'     Spectral resolution = {self.spectrum[key][3]:.2f}')

        # Set the wavelength range for the Radtrans object

        if wavel_range is None:
            self.wavel_range = (0.95*min(self.wavel_min), 1.15*max(self.wavel_max))

        else:
            self.wavel_range = (wavel_range[0], wavel_range[1])

        # Create the pressure layers for the Radtrans object

        if self.pressure_grid == 'standard' or self.pressure_grid == 'smaller':
            # Initiate 180 pressure layers but use only 60 layers during the radiative transfer
            # when pressure_grid is set to 'smaller'
            n_pressure = 180

        elif self.pressure_grid == 'clouds':
            # Initiate 1140 pressure layers but use fewer layers (~100) during the radiative
            # tranfer after running make_half_pressure_better
            n_pressure = 1440

        else:
            raise ValueError(f'The argument of pressure_grid (\'{self.pressure_grid}\') is not '
                             f'recognized. Please use \'standard\', \'smaller\', or \'clouds\'.')

        self.pressure = np.logspace(-6, 3, n_pressure)

        print(f'Initiating {self.pressure.size} pressure levels (bar): '
              f'{self.pressure[0]:.2e} - {self.pressure[-1]:.2e}')

        # Initiate parameter list and counters

        self.parameters = []

        # Initiate the optional P-T smoothing parameter

        self.pt_smooth = None

        # Weighting of the photometric and spectroscopic data

        print('Weights for the log-likelihood function:')

        if weights is None:
            self.weights = {}
        else:
            self.weights = weights

        for item in inc_spec:
            if item not in self.weights:
                self.weights[item] = 1.

            print(f'   - {item} = {self.weights[item]:.2e}')

        for item in inc_phot:
            if item not in self.weights:
                self.weights[item] = 1.

            print(f'   - {item} = {self.weights[item]:.2e}')

    @typechecked
    def set_parameters(self,
                       bounds: dict,
                       chemistry: str,
                       quenching: Optional[str],
                       pt_profile: str,
                       fit_corr: List[str]) -> None:
        """
        Function to set the list with parameters.

        Parameters
        ----------
        bounds : dict
            Dictionary with the parameter boundaries.
        chemistry : str
            The chemistry type: 'equilibrium' for equilibrium chemistry or 'free' for retrieval
            of free abundances (but constant with altitude).
        quenching : str, None
            Quenching type for CO/CH4/H2O abundances. Either the quenching pressure (bar) is a free
            parameter (``quenching='pressure'``) or the quenching pressure is calculated from the
            mixing and chemical timescales (``quenching='diffusion'``). The quenching is not
            applied if the argument is set to ``None``.
        pt_profile : str
            The parametrization for the pressure-temperature profile ('molliere', 'free', or
            'monotonic').
        fit_corr : list(str), None
            List with spectrum names for which the correlation length and fractional amplitude are
            fitted (see Wang et al. 2020).

        Returns
        -------
        NoneType
            None
        """

        # Generic parameters

        self.parameters.append('logg')
        self.parameters.append('radius')

        # P-T profile parameters

        if pt_profile in ['molliere', 'mod-molliere']:
            self.parameters.append('tint')
            self.parameters.append('alpha')
            self.parameters.append('log_delta')
            self.parameters.append('log_sigma_alpha')

            if pt_profile == 'molliere':
                self.parameters.append('t1')
                self.parameters.append('t2')
                self.parameters.append('t3')

        elif pt_profile in ['free', 'monotonic']:
            for i in range(15):
                self.parameters.append(f't{i}')

            if pt_profile == 'free':
                self.parameters.append('gamma_r')
                self.parameters.append('beta_r')

        # Abundance parameters

        if chemistry == 'equilibrium':

            self.parameters.append('metallicity')
            self.parameters.append('c_o_ratio')

        elif chemistry == 'free':

            for item in self.line_species:
                self.parameters.append(item)

            for item in self.cloud_species:
                self.parameters.append(item[:-3])

        # Non-equilibrium chemistry

        if quenching == 'pressure':
            # Fit quenching pressure
            self.parameters.append('log_p_quench')

        elif quenching == 'diffusion':
            # Calculate quenching pressure from Kzz and timescales
            pass

        # Cloud parameters

        if 'log_kappa_0' in bounds:
            self.parameters.append('fsed')
            self.parameters.append('log_kappa_0')
            self.parameters.append('opa_index')
            self.parameters.append('log_p_base')
            self.parameters.append('albedo')

        elif len(self.cloud_species) > 0:
            self.parameters.append('fsed')
            self.parameters.append('log_kzz')
            self.parameters.append('sigma_lnorm')

            for item in self.cloud_species:
                cloud_lower = item[:-6].lower()

                if f'{cloud_lower}_tau' in bounds:
                    self.parameters.append(f'{cloud_lower}_tau')

                elif 'log_tau_cloud' not in bounds:
                    self.parameters.append(f'{cloud_lower}_fraction')

        # Add the flux scaling parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][0] is not None:
                    self.parameters.append(f'scaling_{item}')

        # Add the error offset parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][1] is not None:
                    self.parameters.append(f'error_{item}')

        # Add the wavelength calibration parameters

        for item in self.spectrum:
            if item in bounds:
                if bounds[item][2] is not None:
                    self.parameters.append(f'wavelength_{item}')

        # Add extinction parameters

        if 'ism_ext' in bounds:
            self.parameters.append('ism_ext')

        if 'ism_red' in bounds:
            if 'ism_ext' not in bounds:
                raise ValueError('The \'ism_red\' parameter can only be used in combination '
                                 'with \'ism_ext\'.')

            self.parameters.append('ism_red')

        # Add covariance parameters

        for item in self.spectrum:
            if item in fit_corr:
                self.parameters.append(f'corr_len_{item}')
                self.parameters.append(f'corr_amp_{item}')

        # Add P-T smoothing parameter

        if 'pt_smooth' in bounds:
            self.parameters.append('pt_smooth')

        # Add cloud optical depth parameter

        if 'log_tau_cloud' in bounds:
            self.parameters.append('log_tau_cloud')

            if len(self.cloud_species) > 1:
                for item in self.cloud_species[1:]:
                    cloud_1 = item[:-6].lower()
                    cloud_2 = self.cloud_species[0][:-6].lower()

                    self.parameters.append(f'{cloud_1}_{cloud_2}_ratio')

        # List all parameters

        print(f'Fitting {len(self.parameters)} parameters:')

        for item in self.parameters:
            print(f'   - {item}')

    @typechecked
    def rebin_opacities(self,
                        spec_res: float,
                        out_folder: str = 'rebin_out'):
        """
        Function for downsampling the ``c-k`` opacities. The downsampled opacities should be stored
        in the `opacities/lines/corr_k/` folder of the ``pRT_input_data_path``.

        Parameters
        ----------
        spec_res : float
            Spectral resolution to which the opacities are downsampled.
        out_folder : str
            Path of the output folder where the opacities will be stored.

        Returns
        -------
        NoneType
            None
        """

        print('Importing petitRADTRANS...', end='', flush=True)
        from petitRADTRANS.radtrans import Radtrans
        print(' [DONE]')

        # https://petitradtrans.readthedocs.io/en/latest/content/notebooks/Rebinning_opacities.html

        rt_object = Radtrans(line_species=self.line_species,
                             rayleigh_species=['H2', 'He'],
                             cloud_species=self.cloud_species_full.copy(),
                             continuum_opacities=['H2-H2', 'H2-He'],
                             wlen_bords_micron=(0.1, 251.),
                             mode='c-k',
                             test_ck_shuffle_comp=self.scattering,
                             do_scat_emis=self.scattering)

        mol_masses = {}

        for item in self.line_species:
            if item[-8:] == '_all_iso':
                mol_masses[item[:-8]] = Formula(item[:-8]).isotope.massnumber

            else:
                mol_masses[item] = Formula(item).isotope.massnumber

        rt_object.write_out_rebin(spec_res,
                                  path=out_folder,
                                  species=self.line_species,
                                  masses=mol_masses)

    @typechecked
    def run_multinest(self,
                      bounds: dict,
                      chemistry: str = 'equilibrium',
                      quenching: Optional[str] = 'pressure',
                      pt_profile: str = 'molliere',
                      fit_corr: Optional[List[str]] = None,
                      cross_corr: Optional[List[str]] = None,
                      n_live_points: int = 2000,
                      resume: bool = False,
                      plotting: bool = False,
                      check_isothermal: bool = False,
                      pt_smooth: float = 0.3,
                      check_flux: Optional[float] = None) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the ``MultiNest`` sampler. While
        ``PyMultiNest`` can be installed with ``pip`` from the PyPI repository, ``MultiNest``
        has to to be build manually. See the ``PyMultiNest`` documentation for details:
        http://johannesbuchner.github.io/PyMultiNest/install.html. Note that the library path
        of ``MultiNest`` should be set to the environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac. Alternatively, the variable can be
        set before importing the ``species`` package, for example:

        .. code-block:: python

            >>> import os
            >>> os.environ['DYLD_LIBRARY_PATH'] = '/path/to/MultiNest/lib'
            >>> import species

        When using MPI, it is also required to install ``mpi4py`` (e.g. ``pip install mpi4py``),
        otherwise an error may occur when the ``output_folder`` is created by multiple processes.

        Parameters
        ----------
        bounds : dict
            Dictionary with the prior boundaries.
        chemistry : str
            The chemistry type: 'equilibrium' for equilibrium chemistry or 'free' for retrieval
            of free abundances (but constant with altitude).
        quenching : str, None
            Quenching type for CO/CH4/H2O abundances. Either the quenching pressure (bar) is a free
            parameter (``quenching='pressure'``) or the quenching pressure is calculated from the
            mixing and chemical timescales (``quenching='diffusion'``). The quenching is not
            applied if the argument is set to ``None``.
        pt_profile : str
            The parametrization for the pressure-temperature profile ('molliere', 'free', or
            'monotonic').
        fit_corr : list(str), None
            List with spectrum names for which the correlation length and fractional amplitude are
            fitted (see Wang et al. 2020).
        cross_corr : list(str), None
            List with spectrum names for which a cross-correlation to log-likelihood mapping is
            calculated instead of the regular least-squares approach (see Brogi & Line 2019). This
            cross-correlation approach can be applied on high-resolution spectra. Currently, this
            option only supports spectra that have been shifted to the planet's rest frame.
        n_live_points : int
            Number of live points.
        resume : bool
            Resume from a previous run.
        plotting : bool
            Plot sample results for testing.
        check_isothermal : bool
            Check if there is an isothermal region below 1 bar. If so, discard the sample. This
            parameter has not been properly tested. It is recommended to use the ``check_flux``
            parameter instead.
        pt_smooth : float
            Standard deviation of the Gaussian kernel that is used for smoothing the sampled
            temperature nodes of the P-T profile. Only required with `pt_profile='free'` or
            `pt_profile='monotonic'`. The argument should be given as log10(P/bar) with the default
            value set to 0.3 dex.
        check_flux : float, None
            Relative tolerance that is used for ensuring a consistent bolometric flux between the
            top, bottom, and 3 intermediate pressures in the atmosphere. This makes the retrieval
            much slower. To use this parameter, the opacities should be recreated with
            :meth:`~species.analysis.retrieval.AtmosphericRetrieval.rebin_opacities` at $R = 10$
            (i.e. ``spec_res=30``).

        Returns
        -------
        NoneType
            None
        """

        # Check if quenching parameter is used with equilibrium chemistry

        if quenching is not None and chemistry != 'equilibrium':
            raise ValueError('The \'quenching\' parameter can only be used in combination with '
                             'chemistry=\'equilibrium\'.')

        # Check quenching parameter

        if quenching is not None and quenching not in ['pressure', 'diffusion']:
            raise ValueError('The argument of \'quenching\' should by of the following: '
                             '\'pressure\', \'diffusion\', or None.')

        # Check if clouds are used in combination with equilibrium chemistry

        # if len(self.cloud_species) > 0 and chemistry != 'equilibrium':
        #     raise ValueError('Clouds are currently only implemented in combination with '
        #                      'equilibrium chemistry.')

        # Check if the Mollière P-T profile is used in combination with equilibrium chemistry

        # if pt_profile == 'molliere' and chemistry != 'equilibrium':
        #     raise ValueError('The \'molliere\' P-T parametrization can only be used in '
        #                      'combination with equilibrium chemistry.')

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(self.output_folder):
            print(f'Creating output folder: {self.output_folder}')
            os.mkdir(self.output_folder)

        # if not os.path.exists(self.output_folder):
        #     raise ValueError(f'The output folder (\'{self.output_folder}\') does not exist.')

        # Import petitRADTRANS and interpol_abundances here because it is slow

        print('Importing petitRADTRANS...', end='', flush=True)
        from petitRADTRANS.radtrans import Radtrans
        # from petitRADTRANS.fort_spec import feautrier_rad_trans
        # from petitRADTRANS.fort_spec import feautrier_pt_it
        print(' [DONE]')

        print('Importing chemistry module...', end='', flush=True)
        from poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances
        print(' [DONE]')

        print('Importing rebin module...', end='', flush=True)
        from petitRADTRANS.retrieval.rebin_give_width import rebin_give_width
        print(' [DONE]')

        # List with spectra for which the correlated noise is fitted

        if fit_corr is None:
            fit_corr = []

        for item in self.spectrum:
            if item in fit_corr:
                bounds[f'corr_len_{item}'] = (-3., 0.)  # log10(corr_len/um)
                bounds[f'corr_amp_{item}'] = (0., 1.)

        # List with spectra that will be used for a cross-correlation instead of least-squares

        if cross_corr is None:
            cross_corr = []

        # Create list with parameters for MultiNest

        self.set_parameters(bounds, chemistry, quenching, pt_profile, fit_corr)

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(self.parameters):
            cube_index[item] = i

        # Delete C/H and O/H boundaries if the chemistry is not free

        if chemistry != 'free':
            if 'c_h_ratio' in bounds:
                del bounds['c_h_ratio']

            if 'o_h_ratio' in bounds:
                del bounds['o_h_ratio']

        # Update the P-T smoothing parameter

        self.pt_smooth = pt_smooth

        # Create an instance of Ratrans
        # The names in self.cloud_species are changed after initiating Radtrans

        print('Setting up petitRADTRANS...')

        rt_object = Radtrans(line_species=self.line_species,
                             rayleigh_species=['H2', 'He'],
                             cloud_species=self.cloud_species,
                             continuum_opacities=['H2-H2', 'H2-He'],
                             wlen_bords_micron=self.wavel_range,
                             mode='c-k',
                             test_ck_shuffle_comp=self.scattering,
                             do_scat_emis=self.scattering)

        lbl_radtrans = {}

        for item in cross_corr:
            lbl_wavel_range = (0.95*self.spectrum[item][0][0, 0], 1.05*self.spectrum[item][0][-1, 0])

            lbl_cloud_species = self.cloud_species_full.copy()

            lbl_radtrans[item] = Radtrans(line_species=self.lbl_species,
                                          rayleigh_species=['H2', 'He'],
                                          cloud_species=lbl_cloud_species,
                                          continuum_opacities=['H2-H2', 'H2-He'],
                                          wlen_bords_micron=lbl_wavel_range,
                                          mode='lbl',
                                          test_ck_shuffle_comp=self.scattering,
                                          do_scat_emis=self.scattering)

        if check_flux is not None:

            line_species_low_res = []
            for item in self.line_species:
                line_species_low_res.append(item+'_R_10')

            lowres_radtrans = Radtrans(line_species=line_species_low_res,
                                       rayleigh_species=['H2', 'He'],
                                       cloud_species=self.cloud_species_full.copy(),
                                       continuum_opacities=['H2-H2', 'H2-He'],
                                       wlen_bords_micron=(0.5, 30.),
                                       mode='c-k',
                                       test_ck_shuffle_comp=self.scattering,
                                       do_scat_emis=self.scattering)

        # Create the RT arrays

        if self.pressure_grid == 'standard':
            print(f'Number of pressure levels used with the radiative transfer: {self.pressure.size}')

            rt_object.setup_opa_structure(self.pressure)

            for item in lbl_radtrans.values():
                item.setup_opa_structure(self.pressure)

            if check_flux is not None:
                lowres_radtrans.setup_opa_structure(self.pressure)

        elif self.pressure_grid == 'smaller':
            print(f'Number of pressure levels used with the radiative transfer: {self.pressure[::3].size}')

            rt_object.setup_opa_structure(self.pressure[::3])

            for item in lbl_radtrans.values():
                item.setup_opa_structure(self.pressure[::3])

            if check_flux is not None:
                lowres_radtrans.setup_opa_structure(self.pressure[::3])

        elif self.pressure_grid == 'clouds':
            if len(self.cloud_species) == 0:
                raise ValueError('Please select a different pressure_grid. Setting the argument '
                                 'to \'clouds\' is only possible with the use of cloud species.')

            # The pressure structure is reinitiated after the refinement around the cloud deck
            # so the current initializiation to 60 pressure points is not used
            print('Number of pressure levels used with the radiative transfer: variable}')

            rt_object.setup_opa_structure(self.pressure[::24])

            for item in lbl_radtrans.values():
                item.setup_opa_structure(self.pressure[::24])

            if check_flux is not None:
                lowres_radtrans.setup_opa_structure(self.pressure[::24])

        # Create the knot pressures

        if pt_profile in ['free', 'monotonic']:
            knot_press = np.logspace(np.log10(self.pressure[0]), np.log10(self.pressure[-1]), 15)

        else:
            knot_press = None

        @typechecked
        def prior(cube,
                  n_dim: int,
                  n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter cube.

            Parameters
            ----------
            cube : LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            NoneType
                None
            """

            # Surface gravity log10(g/cgs)

            if 'logg' in bounds:
                logg = bounds['logg'][0] + (bounds['logg'][1]-bounds['logg'][0])*cube[cube_index['logg']]
            else:
                # Default: 2 - 5.5
                logg = 2. + 3.5*cube[cube_index['logg']]

            cube[cube_index['logg']] = logg

            # Planet radius (Rjup)

            if 'radius' in bounds:
                radius = bounds['radius'][0] + (bounds['radius'][1]-bounds['radius'][0])*cube[cube_index['radius']]
            else:
                # Defaul: 0.8-2 Rjup
                radius = 0.8 + 1.2*cube[cube_index['radius']]

            cube[cube_index['radius']] = radius

            if pt_profile in ['molliere', 'mod-molliere']:

                # Internal temperature (K) of the Eddington approximation (middle altitudes)
                # see Eq. 2 in Mollière et al. (2020)
                if 'tint' in bounds:
                    tint = bounds['tint'][0] + (bounds['tint'][1]-bounds['tint'][0])*cube[cube_index['tint']]
                else:
                    # Default: 500 - 3000 K
                    tint = 500. + 2500.*cube[cube_index['tint']]

                cube[cube_index['tint']] = tint

                if pt_profile == 'molliere':
                    # Connection temperature (K)
                    t_connect = (3./4.*tint**4.*(0.1+2./3.))**0.25

                    # The temperature (K) at temp_3 is scaled down from t_connect
                    temp_3 = t_connect*(1-cube[cube_index['t3']])
                    cube[cube_index['t3']] = temp_3

                    # The temperature (K) at temp_2 is scaled down from temp_3
                    temp_2 = temp_3*(1-cube[cube_index['t2']])
                    cube[cube_index['t2']] = temp_2

                    # The temperature (K) at temp_1 is scaled down from temp_2
                    temp_1 = temp_2*(1-cube[cube_index['t1']])
                    cube[cube_index['t1']] = temp_1

                # alpha: power law index in tau = delta * press_cgs**alpha
                # see Eq. 1 in Mollière et al. (2020)

                if 'alpha' in bounds:
                    alpha = bounds['alpha'][0] + (bounds['alpha'][1]-bounds['alpha'][0])*cube[cube_index['alpha']]
                else:
                    # Default: 1 - 2
                    alpha = 1. + cube[cube_index['alpha']]

                cube[cube_index['alpha']] = alpha

                # Photospheric pressure (bar)

                if pt_profile == 'molliere':
                    if 'log_delta' in bounds:
                        p_phot = 10.**(bounds['log_delta'][0] + (bounds['log_delta'][1]-bounds['log_delta'][0])*cube[cube_index['log_delta']])
                    else:
                        # 1e-3 - 1e2 bar
                        p_phot = 10.**(-3. + 5.*cube[cube_index['log_delta']])

                elif pt_profile == 'mod-molliere':
                    # 1e-6 - 1e2 bar
                    p_phot = 10.**(-6. + 8.*cube[cube_index['log_delta']])

                # delta: proportionality factor in tau = delta * press_cgs**alpha
                # see Eq. 1 in Mollière et al. (2020)
                delta = (p_phot*1e6)**(-alpha)
                log_delta = np.log10(delta)

                cube[cube_index['log_delta']] = log_delta

                # sigma_alpha: fitted uncertainty on the alpha index
                # see Eq. 6 in GRAVITY Collaboration et al. (2020)

                if 'log_sigma_alpha' in bounds:
                    log_sigma_alpha = bounds['log_sigma_alpha'][0] + (bounds['log_sigma_alpha'][1]-bounds['log_sigma_alpha'][0])*cube[cube_index['log_sigma_alpha']]
                else:
                    # Default: -4 - 1
                    log_sigma_alpha = 1. - 5.*cube[cube_index['log_sigma_alpha']]

                cube[cube_index['log_sigma_alpha']] = log_sigma_alpha

            elif pt_profile == 'free':
                # 15 temperature knots (K)
                for i in range(15):
                    # Default: 0 - 8000 K
                    cube[cube_index[f't{i}']] = 8000.*cube[cube_index[f't{i}']]

                # Penalization of wiggles in the P-T profile
                # inverse gamma: a=1, b=5e-5
                beta_r = cube[cube_index['beta_r']]
                gamma_r = invgamma.ppf(cube[cube_index['gamma_r']], a=1., scale=beta_r)
                cube[cube_index['beta_r']] = beta_r
                cube[cube_index['gamma_r']] = gamma_r

            elif pt_profile == 'monotonic':
                # 15 temperature knots (K)
                cube[cube_index['t14']] = 10000.*cube[cube_index['t14']]

                for i in range(13, -1, -1):
                    cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] * \
                        (1.-cube[cube_index[f't{i}']])

                    # Increasing temperature steps with increasing pressure
                    # if i == 13:
                    #     cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] * \
                    #         (1.-cube[cube_index[f't{i}']])
                    #
                    # else:
                    #     temp_diff = cube[cube_index[f't{i+2}']] - cube[cube_index[f't{i+1}']]
                    #
                    #     if cube[cube_index[f't{i+1}']] - temp_diff < 0.:
                    #         temp_diff = cube[cube_index[f't{i+1}']]
                    #
                    #     cube[cube_index[f't{i}']] = cube[cube_index[f't{i+1}']] - \
                    #         cube[cube_index[f't{i}']]*temp_diff

            # Chemical composition

            if chemistry == 'equilibrium':
                # Metallicity [Fe/H] for the nabla_ad interpolation
                if 'metallicity' in bounds:
                    metallicity = bounds['metallicity'][0] + (bounds['metallicity'][1]-bounds['metallicity'][0])*cube[cube_index['metallicity']]
                else:
                    # Default: -1.5 - 1.5 dex
                    metallicity = -1.5 + 3.*cube[cube_index['metallicity']]

                cube[cube_index['metallicity']] = metallicity

                # Carbon-to-oxygen ratio for the nabla_ad interpolation
                if 'c_o_ratio' in bounds:
                    c_o_ratio = bounds['c_o_ratio'][0] + (bounds['c_o_ratio'][1]-bounds['c_o_ratio'][0])*cube[cube_index['c_o_ratio']]
                else:
                    # Default: 0.1 - 1.6
                    c_o_ratio = 0.1 + 1.5*cube[cube_index['c_o_ratio']]

                cube[cube_index['c_o_ratio']] = c_o_ratio

            elif chemistry == 'free':
                # log10 abundances of the line species
                log_x_abund = {}

                for item in self.line_species:
                    if item in bounds:
                        cube[cube_index[item]] = bounds[item][0] + (bounds[item][1]-bounds[item][0])*cube[cube_index[item]]

                    elif item not in ['K', 'K_lor_cut', 'K_burrows']:
                        # Default: -10. - 0. dex
                        cube[cube_index[item]] = -10.*cube[cube_index[item]]

                        # Add the log10 of the mass fraction to the abundace dictionary
                        log_x_abund[item] = cube[cube_index[item]]

                if 'Na' in self.line_species or 'Na_lor_cut' in self.line_species or \
                        'Na_burrows' in self.line_species:
                    log_x_k_abund = retrieval_util.potassium_abundance(log_x_abund)

                if 'K' in self.line_species:
                    cube[cube_index['K']] = log_x_k_abund

                elif 'K_lor_cut' in self.line_species:
                    cube[cube_index['K_lor_cut']] = log_x_k_abund

                elif 'K_burrows' in self.line_species:
                    cube[cube_index['K_burrows']] = log_x_k_abund

                # log10 abundances of the cloud species

                for item in self.cloud_species:
                    if item in bounds:
                        cube[cube_index[item]] = bounds[item][0] + (bounds[item][1]-bounds[item][0])*cube[cube_index[item]]

                    else:
                        # Default: -10. - 0. dex
                        cube[cube_index[item]] = -10.*cube[cube_index[item]]

            # CO/CH4/H2O quenching pressure (bar)

            if quenching == 'pressure':
                if 'log_p_quench' in bounds:
                    log_p_quench = bounds['log_p_quench'][0] + (bounds['log_p_quench'][1]-bounds['log_p_quench'][0])*cube[cube_index['log_p_quench']]
                else:
                    # Default: -6 - 3. (i.e. 1e-6 - 1e3 bar)
                    log_p_quench = -6. + 9.*cube[cube_index['log_p_quench']]

                cube[cube_index['log_p_quench']] = log_p_quench

            # Cloud parameters

            if 'log_kappa_0' in bounds:
                # Cloud model 2 from Mollière et al. (2020)

                if 'fsed' in bounds:
                    fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[cube_index['fsed']]
                else:
                    # Default: 0 - 10
                    fsed = 10.*cube[cube_index['fsed']]

                cube[cube_index['fsed']] = fsed

                if 'log_kappa_0' in bounds:
                    log_kappa_0 = bounds['log_kappa_0'][0] + (bounds['log_kappa_0'][1]-bounds['log_kappa_0'][0])*cube[cube_index['log_kappa_0']]
                else:
                    # Default: -8 - 3
                    log_kappa_0 = -8. + 11.*cube[cube_index['log_kappa_0']]

                cube[cube_index['log_kappa_0']] = log_kappa_0

                if 'opa_index' in bounds:
                    opa_index = bounds['opa_index'][0] + (bounds['opa_index'][1]-bounds['opa_index'][0])*cube[cube_index['opa_index']]
                else:
                    # Default: -6 - 1
                    opa_index = -6. + 7.*cube[cube_index['opa_index']]

                cube[cube_index['opa_index']] = opa_index

                if 'log_p_base' in bounds:
                    log_p_base = bounds['log_p_base'][0] + (bounds['log_p_base'][1]-bounds['log_p_base'][0])*cube[cube_index['log_p_base']]
                else:
                    # Default: -6 - 3
                    log_p_base = -6. + 9.*cube[cube_index['log_p_base']]

                cube[cube_index['log_p_base']] = log_p_base

                if 'albedo' in bounds:
                    albedo = bounds['albedo'][0] + (bounds['albedo'][1]-bounds['albedo'][0])*cube[cube_index['albedo']]
                else:
                    # Default: 0 - 1
                    albedo = cube[cube_index['albedo']]

                cube[cube_index['albedo']] = albedo

                if 'log_tau_cloud' in bounds:
                    log_tau_cloud = bounds['log_tau_cloud'][0] + \
                        (bounds['log_tau_cloud'][1] - bounds['log_tau_cloud'][0]) * \
                        cube[cube_index['log_tau_cloud']]

                    cube[cube_index['log_tau_cloud']] = log_tau_cloud

            elif len(self.cloud_species) > 0:
                # Sedimentation parameter: ratio of the settling and mixing velocities of the
                # cloud particles (used in Eq. 3 of Mollière et al. 2020)

                if 'fsed' in bounds:
                    fsed = bounds['fsed'][0] + (bounds['fsed'][1]-bounds['fsed'][0])*cube[cube_index['fsed']]
                else:
                    # Default: 0 - 10
                    fsed = 10.*cube[cube_index['fsed']]

                cube[cube_index['fsed']] = fsed

                # Log10 of the eddy diffusion coefficient (cm2 s-1)

                if 'log_kzz' in bounds:
                    log_kzz = bounds['log_kzz'][0] + (bounds['log_kzz'][1]-bounds['log_kzz'][0])*cube[cube_index['log_kzz']]
                else:
                    # Default: 5 - 13
                    log_kzz = 5. + 8.*cube[cube_index['log_kzz']]

                cube[cube_index['log_kzz']] = log_kzz

                # Geometric standard deviation of the log-normal size distribution

                if 'sigma_lnorm' in bounds:
                    sigma_lnorm = bounds['sigma_lnorm'][0] + (bounds['sigma_lnorm'][1] -
                                                              bounds['sigma_lnorm'][0])*cube[cube_index['sigma_lnorm']]
                else:
                    # Default: 1.05 - 3.
                    sigma_lnorm = 1.05 + 1.95*cube[cube_index['sigma_lnorm']]

                cube[cube_index['sigma_lnorm']] = sigma_lnorm

                if 'log_tau_cloud' in bounds:
                    log_tau_cloud = bounds['log_tau_cloud'][0] + \
                        (bounds['log_tau_cloud'][1] - bounds['log_tau_cloud'][0]) * \
                        cube[cube_index['log_tau_cloud']]

                    cube[cube_index['log_tau_cloud']] = log_tau_cloud

                    if len(self.cloud_species) > 1:
                        for item in self.cloud_species[1:]:
                            cloud_1 = item[:-3].lower()
                            cloud_2 = self.cloud_species[0][:-3].lower()

                            mass_ratio = bounds[f'{cloud_1}_{cloud_2}_ratio'][0] + \
                                (bounds[f'{cloud_1}_{cloud_2}_ratio'][1] - bounds[f'{cloud_1}_{cloud_2}_ratio'][0]) * \
                                cube[cube_index[f'{cloud_1}_{cloud_2}_ratio']]

                            cube[cube_index[f'{cloud_1}_{cloud_2}_ratio']] = mass_ratio

                elif chemistry == 'equilibrium':
                    # Cloud mass fractions at the cloud base, relative to the maximum values allowed
                    # from elemental abundances (see Eq. 3 in Mollière et al. 2020)

                    for item in self.cloud_species_full:
                        cloud_lower = item[:-6].lower()

                        if f'{cloud_lower}_fraction' in bounds:
                            cloud_bounds = bounds[f'{cloud_lower}_fraction']

                            cube[cube_index[f'{cloud_lower}_fraction']] = \
                                cloud_bounds[0] + (cloud_bounds[1] - cloud_bounds[0]) * \
                                cube[cube_index[f'{cloud_lower}_fraction']]

                        elif f'{cloud_lower}_tau' in bounds:
                            cloud_bounds = bounds[f'{cloud_lower}_tau']

                            cube[cube_index[f'{cloud_lower}_tau']] = \
                                cloud_bounds[0] + (cloud_bounds[1] - cloud_bounds[0]) * \
                                cube[cube_index[f'{cloud_lower}_tau']]

                        else:
                            # Default: 0.05 - 1.
                            cube[cube_index[f'{cloud_lower}_fraction']] = \
                                np.log10(0.05) + (np.log10(1.) - np.log10(0.05)) * \
                                cube[cube_index[f'{cloud_lower}_fraction']]

            # Add flux scaling parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][0] is not None:
                        cube[cube_index[f'scaling_{item}']] = bounds[item][0][0] + \
                            (bounds[item][0][1]-bounds[item][0][0])*cube[cube_index[f'scaling_{item}']]

            # Add error inflation parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][1] is not None:
                        cube[cube_index[f'error_{item}']] = bounds[item][1][0] + \
                            (bounds[item][1][1]-bounds[item][1][0]) * \
                            cube[cube_index[f'error_{item}']]

            # Add wavelength calibration parameter if the boundaries are provided

            for item in self.spectrum:
                if item in bounds:
                    if bounds[item][2] is not None:
                        cube[cube_index[f'wavelength_{item}']] = bounds[item][2][0] + \
                            (bounds[item][2][1]-bounds[item][2][0]) * \
                            cube[cube_index[f'wavelength_{item}']]

            # Add covariance parameters if any spectra are provided to fit_corr

            for item in self.spectrum:
                if item in fit_corr:
                    cube[cube_index[f'corr_len_{item}']] = bounds[f'corr_len_{item}'][0] + \
                        (bounds[f'corr_len_{item}'][1]-bounds[f'corr_len_{item}'][0]) * \
                        cube[cube_index[f'corr_len_{item}']]

                    cube[cube_index[f'corr_amp_{item}']] = bounds[f'corr_amp_{item}'][0] + \
                        (bounds[f'corr_amp_{item}'][1]-bounds[f'corr_amp_{item}'][0]) * \
                        cube[cube_index[f'corr_amp_{item}']]

            # ISM extinction

            if 'ism_ext' in bounds:
                ism_ext = bounds['ism_ext'][0] + (bounds['ism_ext'][1]-bounds['ism_ext'][0])*cube[cube_index['ism_ext']]

                cube[cube_index['ism_ext']] = ism_ext

            if 'ism_red' in bounds:
                ism_red = bounds['ism_red'][0] + (bounds['ism_red'][1]-bounds['ism_red'][0])*cube[cube_index['ism_red']]

                cube[cube_index['ism_red']] = ism_red

            # Standard deviation of the Gaussian kernel for smoothing the P-T profile

            if 'pt_smooth' in bounds:
                cube[cube_index['pt_smooth']] = bounds['pt_smooth'][0] + (bounds['pt_smooth'][1]-bounds['pt_smooth'][0])*cube[cube_index['pt_smooth']]

        @typechecked
        def loglike(cube,
                    n_dim: int,
                    n_param: int) -> float:
            """
            Function for the logarithm of the likelihood, computed from the parameter cube.

            Parameters
            ----------
            cube : LP_c_double
                Cube with the model parameters.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            float
                Sum of the logarithm of the prior and likelihood.
            """

            # Initiate the logarithm of the prior and likelihood

            ln_prior = 0.
            ln_like = 0.

            # Initiate abundance and cloud base dictionaries to None

            log_x_abund = None
            log_x_base = None

            # Create dictionary with flux scaling parameters

            scaling = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][0] is not None:
                    scaling[item] = cube[cube_index[f'scaling_{item}']]
                else:
                    scaling[item] = 1.

            # Create dictionary with error offset parameters

            err_offset = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][1] is not None:
                    err_offset[item] = cube[cube_index[f'error_{item}']]
                else:
                    err_offset[item] = None

            # Create dictionary with wavelength calibration parameters

            wavel_cal = {}

            for item in self.spectrum:
                if item in bounds and bounds[item][2] is not None:
                    wavel_cal[item] = cube[cube_index[f'wavelength_{item}']]
                else:
                    wavel_cal[item] = 0.

            # Create dictionary with covariance parameters

            corr_len = {}
            corr_amp = {}

            for item in self.spectrum:
                if f'corr_len_{item}' in bounds:
                    corr_len[item] = 10.**cube[cube_index[f'corr_len_{item}']]  # (um)

                if f'corr_amp_{item}' in bounds:
                    corr_amp[item] = cube[cube_index[f'corr_amp_{item}']]

            # Check if the cloud optical depth is a free parameter

            calc_tau_cloud = False

            for item in self.cloud_species:
                if item[:-3].lower()+'_tau' in bounds:
                    calc_tau_cloud = True

            # Read the P-T smoothing parameter or use the argument of run_multinest otherwise

            if 'pt_smooth' in cube_index:
                pt_smooth = cube[cube_index['pt_smooth']]
            else:
                pt_smooth = self.pt_smooth

            # C/O and [Fe/H]

            if chemistry == 'equilibrium':
                metallicity = cube[cube_index['metallicity']]
                c_o_ratio = cube[cube_index['c_o_ratio']]

            elif chemistry == 'free':
                # TODO Set [Fe/H] = 0 for Molliere P-T profile and cloud condensation profiles
                metallicity = 0.

                # Create a dictionary with the mass fractions

                log_x_abund = {}

                for item in self.line_species:
                    log_x_abund[item] = cube[cube_index[item]]

                # Check if the sum of fractional abundances is smaller than unity

                if np.sum(10.**np.asarray(list(log_x_abund.values()))) > 1.:
                    return -np.inf

                # Check if the C/H and O/H ratios are within the prior boundaries

                c_h_ratio, o_h_ratio, c_o_ratio = \
                    retrieval_util.calc_metal_ratio(log_x_abund)

                if 'c_h_ratio' in bounds and (c_h_ratio < bounds['c_h_ratio'][0] or
                                              c_h_ratio > bounds['c_h_ratio'][1]):

                    return -np.inf

                if 'o_h_ratio' in bounds and (o_h_ratio < bounds['o_h_ratio'][0] or
                                              o_h_ratio > bounds['o_h_ratio'][1]):

                    return -np.inf

                if 'c_o_ratio' in bounds and (c_o_ratio < bounds['c_o_ratio'][0] or
                                              c_o_ratio > bounds['c_o_ratio'][1]):

                    return -np.inf

            # Create the P-T profile

            temp, knot_temp, phot_press, conv_press = retrieval_util.create_pt_profile(
                cube, cube_index, pt_profile, self.pressure, knot_press,
                metallicity, c_o_ratio, pt_smooth)

            # if conv_press is not None and (conv_press > 1. or conv_press < 0.01):
            #     # Maximum pressure (bar) for the radiative-convective boundary
            #     return -np.inf

            # Prepare the scaling based on the cloud optical depth

            if calc_tau_cloud:
                if quenching == 'pressure':
                    # Quenching pressure (bar)
                    p_quench = 10.**cube[cube_index['log_p_quench']]

                elif quenching == 'diffusion':
                    pass

                else:
                    p_quench = None

                # Interpolate the abundances, following chemical equilibrium
                abund_in = interpol_abundances(np.full(self.pressure.size, cube[cube_index['c_o_ratio']]),
                                               np.full(self.pressure.size, cube[cube_index['metallicity']]),
                                               temp,
                                               self.pressure,
                                               Pquench_carbon=p_quench)

                # Extract the mean molecular weight
                mmw = abund_in['MMW']

            # Check for isothermal regions

            if check_isothermal:
                # Get knot indices where the pressure is larger than 1 bar
                indices = np.where(knot_press > 1.)[0]

                # Remove last index because temp_diff.size = knot_press.size - 1
                indices = indices[:-1]

                temp_diff = np.diff(knot_temp)
                temp_diff = temp_diff[indices]

                small_temp = np.where(temp_diff < 100.)[0]

                if len(small_temp) > 0:
                    # Return zero probability if there is a temperature step smaller than 10 K
                    return -np.inf

            # Penalize P-T profiles with oscillations

            if pt_profile == 'free':
                temp_sum = np.sum((knot_temp[2:] + knot_temp[:-2] - 2.*knot_temp[1:-1])**2.)
                # temp_sum = np.sum((temp[::3][2:] + temp[::3][:-2] - 2.*temp[::3][1:-1])**2.)

                ln_prior += -1.*temp_sum/(2.*cube[cube_index['gamma_r']]) - \
                    0.5*np.log(2.*np.pi*cube[cube_index['gamma_r']])

            # Return zero probability if the minimum temperature is negative

            if np.min(temp) < 0.:
                return -np.inf

            # Set the quenching pressure

            if quenching == 'pressure':
                # Fit the quenching pressure
                p_quench = 10.**cube[cube_index['log_p_quench']]

            elif quenching == 'diffusion':
                # Calculate the quenching pressure from timescales
                p_quench = retrieval_util.quench_pressure(
                    self.pressure, temp, cube[cube_index['metallicity']],
                    cube[cube_index['c_o_ratio']], cube[cube_index['logg']],
                    cube[cube_index['log_kzz']])

            else:
                p_quench = None

            # Calculate the emission spectrum

            start = time.time()

            if len(self.cloud_species) > 0 or 'log_kappa_0' in bounds:
                # Cloudy atmosphere

                tau_cloud = None

                if 'log_kappa_0' in bounds:
                    if 'log_tau_cloud' in self.parameters:
                        tau_cloud = 10.**cube[cube_index['log_tau_cloud']]

                elif chemistry == 'equilibrium':
                    cloud_fractions = {}

                    for item in self.cloud_species:
                        if f'{item[:-3].lower()}_fraction' in self.parameters:
                            cloud_fractions[item] = cube[cube_index[f'{item[:-3].lower()}_fraction']]

                        elif f'{item[:-3].lower()}_tau' in self.parameters:
                            params = retrieval_util.cube_to_dict(cube, cube_index)

                            cloud_fractions[item] = retrieval_util.scale_cloud_abund(
                                params, rt_object, self.pressure, temp, mmw, chemistry,
                                abund_in, item, params[f'{item[:-3].lower()}_tau'],
                                pressure_grid=self.pressure_grid)

                            if len(self.cross_corr) != 0:
                                raise ValueError('Check if it works correctly with lbl species.')

                    if 'log_tau_cloud' in self.parameters:
                        tau_cloud = 10.**cube[cube_index['log_tau_cloud']]

                        for i, item in enumerate(self.cloud_species):
                            if i == 0:
                                cloud_fractions[item] = 0.

                            else:
                                cloud_1 = item[:-3].lower()
                                cloud_2 = self.cloud_species[0][:-3].lower()

                                cloud_fractions[item] = cube[cube_index[f'{cloud_1}_{cloud_2}_ratio']]

                    log_x_base = retrieval_util.log_x_cloud_base(cube[cube_index['c_o_ratio']],
                                                                 cube[cube_index['metallicity']],
                                                                 cloud_fractions)

                elif chemistry == 'free':
                    # Add the log10 mass fractions of the clouds to the dictionary

                    log_x_base = {}

                    for item in self.cloud_species:
                        log_x_base[item[:-3]] = cube[cube_index[item]]

                # Create dictionary with cloud parameters

                cloud_param = ['fsed', 'log_kzz', 'sigma_lnorm', 'log_kappa_0',
                               'opa_index', 'log_p_base', 'albedo']

                cloud_dict = {}

                for item in cloud_param:
                    if item in self.parameters:
                        cloud_dict[item] = cube[cube_index[item]]
                    # elif item in ['log_kzz', 'sigma_lnorm']:
                    #     cloud_dict[item] = None

                # Calculate a cloudy spectrum for low- and medium-resolution data (i.e. corr-k)

                wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clouds(
                    rt_object, self.pressure, temp, c_o_ratio, metallicity,
                    p_quench, log_x_abund, log_x_base, cloud_dict, cube[cube_index['logg']],
                    chemistry=chemistry, pressure_grid=self.pressure_grid, plotting=plotting,
                    contribution=False, tau_cloud=tau_cloud)

                if wlen_micron is None and flux_lambda is None:
                    return -np.inf

                if check_flux is not None:
                    # Check if the bolometric flux is conserved in the radiative region

                    # Pressure index at the radiative-convective boundary
                    if conv_press is None:
                        i_conv = lowres_radtrans.press.shape[0]
                    else:
                        i_conv = np.argmax(conv_press < 1e-6*lowres_radtrans.press)

                    # Calculate low-resolution spectrum (R = 10) to initiate the attributes

                    wlen_lowres, flux_lowres, _ = retrieval_util.calc_spectrum_clouds(
                        lowres_radtrans, self.pressure, temp, c_o_ratio, metallicity,
                        p_quench, log_x_abund, log_x_base, cloud_dict, cube[cube_index['logg']],
                        chemistry=chemistry, pressure_grid=self.pressure_grid, plotting=plotting,
                        contribution=False, tau_cloud=tau_cloud)

                    if wlen_lowres is None and flux_lowres is None:
                        return -np.inf

                    f_bol = simps(flux_lowres, wlen_lowres)

                    # Calculate again a low-resolution spectrum (R = 10) but now
                    # with the new Feautrier function from petitRADTRANS

                    # flux_lowres, __, _, h_bol, _, _, _, _, __, __ = \
                    #     feautrier_pt_it(lowres_radtrans.border_freqs,
                    #                     lowres_radtrans.total_tau[:, :, 0, :],
                    #                     lowres_radtrans.temp,
                    #                     lowres_radtrans.mu,
                    #                     lowres_radtrans.w_gauss_mu,
                    #                     lowres_radtrans.w_gauss,
                    #                     lowres_radtrans.photon_destruction_prob,
                    #                     False,
                    #                     lowres_radtrans.reflectance,
                    #                     lowres_radtrans.emissivity,
                    #                     np.zeros_like(lowres_radtrans.freq),
                    #                     lowres_radtrans.geometry,
                    #                     lowres_radtrans.mu_star,
                    #                     True,
                    #                     lowres_radtrans.do_scat_emis,
                    #                     lowres_radtrans.line_struc_kappas[:, :, 0, :],
                    #                     lowres_radtrans.continuum_opa_scat_emis)

                    h_bol = -1e-3*lowres_radtrans.h_bol

                    # (erg s-1 cm-2 Hz-1) -> (W m-2 um-1)
                    flux_lowres *= 1e3*constants.LIGHT/wlen_lowres**2.

                    for i in range(i_conv):
                        if not isclose(f_bol, 4.*np.pi*h_bol[i], rel_tol=check_flux, abs_tol=0.):
                            # Remove the sample if the bolometric flux of the output spectrum
                            # is different from the bolometric flux deeper in the atmosphere
                            return -np.inf

                    if plotting:
                        plt.plot(wlen_lowres, flux_lowres)
                        plt.xlabel(r'Wavelength ($\mu$m)')
                        plt.ylabel(r'Flux (W m$^{-2}$ $\mu$m$^{-1}$)')
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.savefig('lowres_spec.pdf', bbox_inches='tight')
                        plt.clf()

                if phot_press/rt_object.pphot > 5. or phot_press/rt_object.pphot < 0.2:
                    # Remove the sample if the photospheric pressure from the P-T profile is more
                    # than a factor 5 larger than the photospheric pressure that is calculated from
                    # the Rosseland mean opacity, using the non-gray opacities of the atmosphere
                    # See Eq. 7 in GRAVITY Collaboration et al. (2020)
                    return -np.inf

                # if np.abs(cube[cube_index['alpha']]-rt_object.tau_pow) > 0.1:
                #     # Remove the sample if the parametrized, pressure-dependent opacity is not
                #     # consistent with the atmosphere's non-gray opacity structure
                #     # See Eq. 5 in GRAVITY Collaboration et al. (2020)
                #     return -np.inf

                # Penalize samples if the parametrized, pressure-dependent opacity is not
                # consistent with the atmosphere's non-gray opacity structure
                # See Eqs. 5 and 6 in GRAVITY Collaboration et al. (2020)

                sigma_alpha = 10.**cube[cube_index['log_sigma_alpha']]

                ln_like += -0.5 * (cube[cube_index['alpha']]-rt_object.tau_pow)**2. / \
                    sigma_alpha**2. - 0.5*np.log(2.*np.pi*sigma_alpha**2.)

                # Calculate cloudy spectra for high-resolution data (i.e. line-by-line)

                lbl_wavel = {}
                lbl_flux = {}

                for item in cross_corr:

                    lbl_wavel[item], lbl_flux[item], _ = retrieval_util.calc_spectrum_clouds(
                        lbl_radtrans[item], self.pressure, temp, c_o_ratio, metallicity,
                        p_quench, log_x_abund, log_x_base, cloud_dict, cube[cube_index['logg']],
                        chemistry=chemistry, pressure_grid=self.pressure_grid, plotting=plotting,
                        contribution=False, tau_cloud=tau_cloud)

                    if lbl_wavel[item] is None and lbl_flux[item] is None:
                        return -np.inf

            else:
                # Clear atmosphere

                if chemistry == 'equilibrium':
                    # Calculate a clear spectrum for low- and medium-resolution data (i.e. corr-k)
                    wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clear(
                        rt_object, self.pressure, temp, cube[cube_index['logg']],
                        cube[cube_index['c_o_ratio']], cube[cube_index['metallicity']],
                        p_quench, None, chemistry=chemistry, pressure_grid=self.pressure_grid,
                        contribution=False)

                    # Calculate clear spectra for high-resolution data (i.e. line-by-line)

                    lbl_wavel = {}
                    lbl_flux = {}

                    for item in cross_corr:
                        lbl_wavel[item], lbl_flux[item], _ = retrieval_util.calc_spectrum_clear(
                            lbl_radtrans[item], self.pressure, temp, cube[cube_index['logg']],
                            cube[cube_index['c_o_ratio']], cube[cube_index['metallicity']],
                            p_quench, None, chemistry=chemistry, pressure_grid=self.pressure_grid,
                            contribution=False)

                elif chemistry == 'free':
                    # Calculate a clear spectrum for low- and medium-resolution data (i.e. corr-k)

                    wlen_micron, flux_lambda, _ = retrieval_util.calc_spectrum_clear(
                        rt_object, self.pressure, temp, cube[cube_index['logg']],
                        None, None, None, log_x_abund, chemistry,
                        pressure_grid=self.pressure_grid, contribution=False)

                    # Calculate clear spectra for high-resolution data (i.e. line-by-line)

                    lbl_wavel = {}
                    lbl_flux = {}

                    for item in cross_corr:
                        log_x_lbl = {}

                        if 'CO_all_iso' in self.lbl_species:
                            log_x_lbl['CO_all_iso'] = log_x_abund['CO_all_iso']

                        if 'H2O_main_iso' in self.lbl_species:
                            log_x_lbl['H2O_main_iso'] = log_x_abund['H2O']

                        if 'CH4_main_iso' in self.lbl_species:
                            log_x_lbl['CH4_main_iso'] = log_x_abund['CH4']

                        lbl_wavel[item], lbl_flux[item], _ = retrieval_util.calc_spectrum_clear(
                            lbl_radtrans[item], self.pressure, temp, cube[cube_index['logg']],
                            None, None, None, log_x_lbl, chemistry,
                            pressure_grid=self.pressure_grid, contribution=False)

            end = time.time()

            print(f'\rRadiative transfer time: {end-start:.2e} s', end='', flush=True)

            # Return zero probability if the spectrum contains NaN values

            if np.sum(np.isnan(flux_lambda)) > 0:
                # if len(flux_lambda) > 1:
                #     warnings.warn('Spectrum with NaN values encountered.')

                return -np.inf

            for item in lbl_flux.values():
                if np.sum(np.isnan(item)) > 0:
                    return -np.inf

            # Scale the emitted spectra to the observation

            flux_lambda *= (cube[cube_index['radius']]*constants.R_JUP / (self.distance*constants.PARSEC))**2.

            for item in cross_corr:
                lbl_flux[item] *= (cube[cube_index['radius']]*constants.R_JUP / (self.distance*constants.PARSEC))**2.

            # Evaluate the spectra

            for i, item in enumerate(self.spectrum.keys()):
                # Select model spectrum

                if item in cross_corr:
                    model_wavel = lbl_wavel[item]
                    model_flux = lbl_flux[item]

                else:
                    model_wavel = wlen_micron
                    model_flux = flux_lambda

                # Shift the wavelengths of the data with the fitted calibration parameter
                data_wavel = self.spectrum[item][0][:, 0] + wavel_cal[item]

                # Flux density
                data_flux = self.spectrum[item][0][:, 1]

                # Variance with optional inflation
                if err_offset[item] is None:
                    data_var = self.spectrum[item][0][:, 2]**2
                else:
                    data_var = (self.spectrum[item][0][:, 2] + 10.**err_offset[item])**2

                # Apply ISM extinction to the model spectrum

                if 'ism_ext' in self.parameters:
                    if 'ism_red' in self.parameters:
                        ism_reddening = cube[cube_index['ism_red']]

                    else:
                        # Use default interstellar reddening (R_V = 3.1)
                        ism_reddening = 3.1

                    flux_ext = dust_util.apply_ism_ext(model_wavel,
                                                       model_flux,
                                                       cube[cube_index['ism_ext']],
                                                       ism_reddening)

                else:
                    flux_ext = model_flux

                # Convolve with Gaussian LSF

                flux_smooth = retrieval_util.convolve(model_wavel,
                                                      flux_ext,
                                                      self.spectrum[item][3])

                # Resample to the observation

                flux_rebinned = rebin_give_width(model_wavel,
                                                 flux_smooth,
                                                 data_wavel,
                                                 self.spectrum[item][4])

                if item not in cross_corr:
                    # Difference between the observed and modeled spectrum
                    flux_diff = flux_rebinned - scaling[item]*data_flux

                    # Shortcut for the weight
                    weight = self.weights[item]

                    if self.spectrum[item][2] is not None:
                        # Use the inverted covariance matrix

                        if err_offset[item] is None:
                            data_cov_inv = self.spectrum[item][2]

                        else:
                            # Ratio of the inflated and original uncertainties
                            sigma_ratio = np.sqrt(data_var) / self.spectrum[item][0][:, 2]
                            sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                            # Calculate the inversion of the infalted covariances
                            data_cov_inv = np.linalg.inv(self.spectrum[item][1]*sigma_i*sigma_j)

                        # Use the inverted covariance matrix
                        dot_tmp = np.dot(flux_diff, np.dot(data_cov_inv, flux_diff))
                        ln_like += -0.5 * weight * dot_tmp - \
                            0.5 * weight * np.nansum(np.log(2.*np.pi*data_var))

                    else:
                        if item in fit_corr:
                            # Covariance model (Wang et al. 2020)
                            wavel_j, wavel_i = np.meshgrid(data_wavel, data_wavel)

                            error = np.sqrt(data_var)  # (W m-2 um-1)
                            error_j, error_i = np.meshgrid(error, error)

                            cov_matrix = corr_amp[item]**2 * error_i * error_j * \
                                np.exp(-(wavel_i-wavel_j)**2 / (2.*corr_len[item]**2)) + \
                                (1.-corr_amp[item]**2) * np.eye(data_wavel.shape[0])*error_i**2

                            dot_tmp = np.dot(flux_diff, np.dot(np.linalg.inv(cov_matrix), flux_diff))

                            ln_like += -0.5 * weight * dot_tmp - \
                                0.5 * weight * np.nansum(np.log(2.*np.pi*data_var))

                        else:
                            # Calculate the log-likelihood without the covariance matrix
                            ln_like += -0.5 * weight * \
                                np.sum(flux_diff**2/data_var + np.log(2.*np.pi*data_var))

                else:
                    # Cross-correlation to log(L) mapping
                    # See Eq. 9 in Brogi & Line (2019)

                    # Number of wavelengths
                    n_wavel = float(data_flux.shape[0])

                    # Apply the optional flux scaling to the data
                    data_flux_scaled = scaling[item]*data_flux

                    # Variance of the data and model

                    cc_var_dat = np.sum((data_flux_scaled-np.mean(data_flux_scaled))**2) / n_wavel
                    cc_var_mod = np.sum((flux_rebinned-np.mean(flux_rebinned))**2) / n_wavel

                    # Cross-covariance
                    cross_cov = np.sum(data_flux_scaled * flux_rebinned) / n_wavel

                    # Log-likelihood
                    if cc_var_dat - 2.*cross_cov + cc_var_mod > 0.:
                        ln_like += -0.5 * n_wavel * np.log(cc_var_dat - 2.*cross_cov + cc_var_mod)

                    else:
                        # Return -inf if logarithm of negative value
                        return -np.inf

                if plotting:
                    plt.errorbar(data_wavel, scaling[item]*data_flux, yerr=np.sqrt(data_var),
                                 marker='o', ms=3, color='tab:blue', markerfacecolor='tab:blue', alpha=0.2)

                    plt.plot(data_wavel, flux_rebinned, marker='o', ms=3, color='tab:orange', alpha=0.2)

            # Evaluate the photometric fluxes

            for i, obj_item in enumerate(self.objphot):
                # Calculate the photometric flux from the model spectrum
                phot_flux, _ = self.synphot[i].spectrum_to_flux(wlen_micron, flux_lambda)

                if np.isnan(phot_flux):
                    raise ValueError(f'The synthetic flux of {self.synphot[i].filter_name} '
                                     f'is NaN. Perhaps the \'wavel_range\' should be broader '
                                     f'such that it includes the full filter profile?')

                # Shortcut for weight
                weight = self.weights[self.synphot[i].filter_name]

                if plotting:
                    read_filt = read_filter.ReadFilter(self.synphot[i].filter_name)

                    plt.errorbar(read_filt.mean_wavelength(), phot_flux, xerr=read_filt.filter_fwhm(),
                                 marker='s', ms=5., color='tab:green', mfc='white')

                if obj_item.ndim == 1:
                    # Filter with one flux
                    ln_like += -0.5 * weight * (obj_item[0] - phot_flux)**2 / obj_item[1]**2

                    if plotting:
                        plt.errorbar(read_filt.mean_wavelength(), obj_item[0], xerr=read_filt.filter_fwhm(),
                                     yerr=obj_item[1], marker='s', ms=5., color='tab:green', mfc='tab:green')

                else:
                    # Filter with multiple fluxes
                    for j in range(obj_item.shape[1]):
                        ln_like += -0.5 * weight * (obj_item[0, j] - phot_flux)**2 / obj_item[1, j]**2

            if plotting:
                plt.plot(wlen_micron, flux_smooth, color='black', zorder=-20)
                plt.xlabel(r'Wavelength ($\mu$m)')
                plt.ylabel(r'Flux (W m$^{-2}$ $\mu$m$^{-1}$)')
                plt.savefig('spectrum.pdf', bbox_inches='tight')
                plt.clf()

            return ln_prior + ln_like

        # Store the model parameters in a JSON file

        json_filename = os.path.join(self.output_folder, 'params.json')
        print(f'Storing the model parameters: {json_filename}')

        with open(json_filename, 'w') as json_file:
            json.dump(self.parameters, json_file)

        # Store the Radtrans arguments in a JSON file

        radtrans_filename = os.path.join(self.output_folder, 'radtrans.json')
        print(f'Storing the Radtrans arguments: {radtrans_filename}')

        radtrans_dict = {}
        radtrans_dict['line_species'] = self.line_species
        radtrans_dict['cloud_species'] = self.cloud_species_full
        radtrans_dict['lbl_species'] = self.lbl_species
        radtrans_dict['distance'] = self.distance
        radtrans_dict['scattering'] = self.scattering
        radtrans_dict['chemistry'] = chemistry
        radtrans_dict['quenching'] = quenching
        radtrans_dict['pt_profile'] = pt_profile
        radtrans_dict['pressure_grid'] = self.pressure_grid
        radtrans_dict['wavel_range'] = self.wavel_range

        if 'pt_smooth' not in bounds:
            radtrans_dict['pt_smooth'] = self.pt_smooth

        with open(radtrans_filename, 'w', encoding='utf-8') as json_file:
            json.dump(radtrans_dict, json_file, ensure_ascii=False, indent=4)

        # Run the nested sampling with MultiNest

        print('Sampling the posterior distribution with MultiNest...')

        pymultinest.run(loglike,
                        prior,
                        len(self.parameters),
                        outputfiles_basename=os.path.join(self.output_folder, ''),
                        resume=resume,
                        verbose=True,
                        const_efficiency_mode=True,
                        sampling_efficiency=0.05,
                        n_live_points=n_live_points,
                        evidence_tolerance=0.5)
