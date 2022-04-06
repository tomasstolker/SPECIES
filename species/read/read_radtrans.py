"""
Module for generating atmospheric model spectra with ``petitRADTRANS``.
Details on the  transfer code can be found in Mollière et al. (2019).
"""

import warnings

from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spectres

from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from typeguard import typechecked

from species.analysis import photometry
from species.core import box, constants
from species.read import read_filter
from species.util import dust_util, read_util, retrieval_util


class ReadRadtrans:
    """
    Class for generating a model spectrum with ``petitRADTRANS``.
    """

    @typechecked
    def __init__(
        self,
        line_species: Optional[List[str]] = None,
        cloud_species: Optional[List[str]] = None,
        scattering: bool = False,
        wavel_range: Optional[Tuple[float, float]] = None,
        filter_name: Optional[str] = None,
        pressure_grid: str = "smaller",
        res_mode: str = "c-k",
        cloud_wavel: Optional[Tuple[float, float]] = None,
        max_press: float = None,
        pt_manual: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        line_species : list, None
            List with the line species. No line species are used if set
            to ``None``.
        cloud_species : list, None
            List with the cloud species. No clouds are used if set to
            ``None``.
        scattering : bool
            Include scattering in the radiative transfer.
        wavel_range : tuple(float, float), None
            Wavelength range (um). The wavelength range is set to
            0.8-10 um if set to ``None`` or not used if ``filter_name``
            is not ``None``.
        filter_name : str, None
            Filter name that is used for the wavelength range. The
            ``wavel_range`` is used if ''filter_name`` is set to
            ``None``.
        pressure_grid : str
            The type of pressure grid that is used for the radiative
            transfer. Either 'standard', to use 180 layers both for
            the atmospheric structure (e.g. when interpolating the
            abundances) and 180 layers with the radiative transfer,
            or 'smaller' to use 60 (instead of 180) with the radiative
            transfer, or 'clouds' to start with 1440 layers but
            resample to ~100 layers (depending on the number of cloud
            species) with a refinement around the cloud decks. For
            cloudless atmospheres it is recommended to use 'smaller',
            which runs faster than 'standard' and provides sufficient
            accuracy. For cloudy atmosphere, one can test with
            'smaller' but it is recommended to use 'clouds' for
            improved accuracy fluxes.
        res_mode : str
            Resolution mode ('c-k' or 'lbl'). The low-resolution mode
            ('c-k') calculates the spectrum with the correlated-k
            assumption at :math:`\\lambda/\\Delta \\lambda = 1000`. The
            high-resolution mode ('lbl') calculates the spectrum with a
            line-by-line treatment at
            :math:`\\lambda/\\Delta \\lambda = 10^6`.
        cloud_wavel : tuple(float, float), None
            Tuple with the wavelength range (um) that is used for
            calculating the median optical depth of the clouds at the
            gas-only photosphere and then scaling the cloud optical
            depth to the value of ``log_tau_cloud``. The range of
            ``cloud_wavel`` should be encompassed by the range of
            ``wavel_range``.  The full wavelength range (i.e.
            ``wavel_range``) is used if the argument is set to
            ``None``.
        max_pressure : float, None
            Maximum pressure (bar) for the free temperature nodes. The
            default is set to 1000 bar.
        pt_manual : np.ndarray, None
            A 2D array that contains the P-T profile that is used
            when ``pressure_grid="manual"``. The shape of array should
            be (n_pressure, 2), with pressure (bar) as first column
            and temperature (K) as second column. It is recommended
            that the pressures are logarithmically spaced.

        Returns
        -------
        NoneType
            None
        """

        # Set several of the required ReadRadtrans attributes

        self.filter_name = filter_name
        self.wavel_range = wavel_range
        self.scattering = scattering
        self.pressure_grid = pressure_grid
        self.cloud_wavel = cloud_wavel
        self.pt_manual = pt_manual

        # Set maximum pressure

        if max_press is None:
            self.max_press = 1e3
        else:
            self.max_press = max_press

        # Set the wavelength range

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()
            self.wavel_range = (0.9 * self.wavel_range[0], 1.2 * self.wavel_range[1])

        elif self.wavel_range is None:
            self.wavel_range = (0.8, 10.0)

        # Set the list with line species

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        # Set the list with cloud species and the number of P-T points

        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        # Set the number of pressures

        if self.pressure_grid in ["standard", "smaller"]:
            # Initiate 180 pressure layers but use only
            # 60 layers during the radiative transfer
            # when pressure_grid is set to 'smaller'
            n_pressure = 180

        elif self.pressure_grid == "clouds":
            # Initiate 1140 pressure layers but use fewer
            # layers (~100) during the radiative tranfer
            # after running make_half_pressure_better
            n_pressure = 1440

        else:
            raise ValueError(
                f"The argument of pressure_grid "
                f"('{self.pressure_grid}') is "
                f"not recognized. Please use "
                f"'standard', 'smaller', or 'clouds'."
            )

        # Create 180 pressure layers in log space

        if self.pressure_grid == "manual":
            if self.pt_manual is None:
                raise UserWarning(
                    "A 2D array with the P-T profile "
                    "should be provided as argument "
                    "of pt_manual when using "
                    "pressure_grid='manual'."
                )

            self.pressure = self.pt_manual[:, 0]

        else:
            self.pressure = np.logspace(-6, np.log10(self.max_press), n_pressure)

        # Import petitRADTRANS here because it is slow

        print("Importing petitRADTRANS...", end="", flush=True)
        from petitRADTRANS.radtrans import Radtrans

        print(" [DONE]")

        # Create the Radtrans object

        self.rt_object = Radtrans(
            line_species=self.line_species,
            rayleigh_species=["H2", "He"],
            cloud_species=self.cloud_species,
            continuum_opacities=["H2-H2", "H2-He"],
            wlen_bords_micron=self.wavel_range,
            mode=res_mode,
            test_ck_shuffle_comp=self.scattering,
            do_scat_emis=self.scattering,
        )

        # Setup the opacity arrays

        if self.pressure_grid == "standard":
            self.rt_object.setup_opa_structure(self.pressure)

        elif self.pressure_grid == "manual":
            self.rt_object.setup_opa_structure(self.pressure)

        elif self.pressure_grid == "smaller":
            self.rt_object.setup_opa_structure(self.pressure[::3])

        elif self.pressure_grid == "clouds":
            self.rt_object.setup_opa_structure(self.pressure[::24])

    @typechecked
    def get_model(
        self,
        model_param: Dict[str, float],
        quenching: Optional[str] = None,
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
        plot_contribution: Optional[Union[bool, str]] = False,
        temp_nodes: Optional[int] = None,
    ) -> box.ModelBox:
        """
        Function for calculating a model spectrum with
        ``petitRADTRANS``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.
        quenching : str, None
            Quenching type for CO/CH4/H2O abundances. Either the
            quenching pressure (bar) is a free parameter
            (``quenching='pressure'``) or the quenching pressure is
            calculated from the mixing and chemical timescales
            (``quenching='diffusion'``). The quenching is not applied
            if the argument is set to ``None``.
        spec_res : float, None
            Spectral resolution, achieved by smoothing with a Gaussian
            kernel. No smoothing is applied when the argument is set to
            ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum will be
            resampled. The original wavelengths points will be used if
            the argument is set to ``None``.
        plot_contribution : bool, str, None
            Filename for the plot with the emission contribution. The
            plot is not created if the argument is set to ``False`` or
            ``None``. If set to ``True``, the plot is shown in an
            interface window instead of written to a file.
        temp_nodes : int, None
            Number of free temperature nodes.

        Returns
        -------
        species.core.box.ModelBox
            Box with the petitRADTRANS model spectrum.
        """

        # Set chemistry type

        if "metallicity" in model_param and "c_o_ratio" in model_param:
            chemistry = "equilibrium"

        else:
            chemistry = "free"

            # Check if all line species from the Radtrans object
            # are also present in the model_param dictionary

            for item in self.line_species:
                if item not in model_param:
                    raise RuntimeError(
                        f"The abundance of {item} is not found "
                        f"in the dictionary with parameters of "
                        f"'model_param'. Please add the log10 "
                        f"mass fraction of {item}."
                    )

        # Check quenching parameter

        if not hasattr(self, "quenching"):
            self.quenching = quenching

        if self.quenching is not None and chemistry != "equilibrium":
            raise ValueError(
                "The 'quenching' parameter can only be used in combination with "
                "chemistry='equilibrium'."
            )

        if self.quenching is not None and self.quenching not in [
            "pressure",
            "diffusion",
        ]:
            raise ValueError(
                "The argument of 'quenching' should be of the following: "
                "'pressure', 'diffusion', or None."
            )

        # C/O and [Fe/H]

        if chemistry == "equilibrium":
            # Equilibrium chemistry
            metallicity = model_param["metallicity"]
            c_o_ratio = model_param["c_o_ratio"]

            log_x_abund = None

        elif chemistry == "free":
            # Free chemistry

            # TODO Set [Fe/H] = 0 for Molliere P-T profile and
            # cloud condensation profiles
            metallicity = 0.0

            # Create a dictionary with the mass fractions

            log_x_abund = {}

            for item in self.line_species:
                log_x_abund[item] = model_param[item]

            _, _, c_o_ratio = retrieval_util.calc_metal_ratio(log_x_abund)

        # Create the P-T profile

        if self.pressure_grid == "manual":
            temp = self.pt_manual[:, 1]

        elif (
            "tint" in model_param
            and "log_delta" in model_param
            and "alpha" in model_param
        ):
            temp, _, _ = retrieval_util.pt_ret_model(
                np.array([model_param["t1"], model_param["t2"], model_param["t3"]]),
                10.0 ** model_param["log_delta"],
                model_param["alpha"],
                model_param["tint"],
                self.pressure,
                metallicity,
                c_o_ratio,
            )

        elif "tint" in model_param and "log_delta" in model_param:
            tau = self.pressure * 1e6 * 10.0 ** model_param["log_delta"]
            temp = (0.75 * model_param["tint"] ** 4.0 * (2.0 / 3.0 + tau)) ** 0.25

        else:
            if temp_nodes is None:
                temp_nodes = 0

                for i in range(100):
                    if f"t{i}" in model_param:
                        temp_nodes += 1
                    else:
                        break

            knot_press = np.logspace(
                np.log10(self.pressure[0]), np.log10(self.pressure[-1]), temp_nodes
            )

            knot_temp = []
            for i in range(temp_nodes):
                knot_temp.append(model_param[f"t{i}"])

            knot_temp = np.asarray(knot_temp)

            if "pt_smooth" in model_param:
                pt_smooth = model_param["pt_smooth"]

            else:
                pt_smooth = None

            temp = retrieval_util.pt_spline_interp(
                knot_press,
                knot_temp,
                self.pressure,
                pt_smooth=pt_smooth,
            )

        # Set the log quenching pressure, log(P/bar)

        if self.quenching == "pressure":
            p_quench = 10.0 ** model_param["log_p_quench"]

        elif self.quenching == "diffusion":
            p_quench = retrieval_util.quench_pressure(
                self.pressure,
                temp,
                model_param["metallicity"],
                model_param["c_o_ratio"],
                model_param["logg"],
                model_param["log_kzz"],
            )

        else:
            if "log_p_quench" in model_param:
                warnings.warn(
                    "The 'model_param' dictionary contains the "
                    "'log_p_quench' parameter but 'quenching=None'. "
                    "The quenching pressure from the dictionary is "
                    "therefore ignored."
                )

            p_quench = None

        if (
            len(self.cloud_species) > 0
            or "log_kappa_0" in model_param
            or "log_kappa_gray" in model_param
        ):

            tau_cloud = None
            log_x_base = None

            if "log_kappa_0" in model_param or "log_kappa_gray" in model_param:
                if "log_tau_cloud" in model_param:
                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    tau_cloud = model_param["tau_cloud"]

            elif chemistry == "equilibrium":
                # Create the dictionary with the mass fractions of the
                # clouds relative to the maximum values allowed from
                # elemental abundances

                cloud_fractions = {}

                for item in self.cloud_species:

                    if f"{item[:-3].lower()}_fraction" in model_param:
                        cloud_fractions[item] = model_param[
                            f"{item[:-3].lower()}_fraction"
                        ]

                    elif f"{item[:-3].lower()}_tau" in model_param:
                        # Import the chemistry module here because it is slow

                        from poor_mans_nonequ_chem.poor_mans_nonequ_chem import (
                            interpol_abundances,
                        )

                        # Interpolate the abundances, following chemical equilibrium

                        abund_in = interpol_abundances(
                            np.full(self.pressure.size, c_o_ratio),
                            np.full(self.pressure.size, metallicity),
                            temp,
                            self.pressure,
                            Pquench_carbon=p_quench,
                        )

                        # Extract the mean molecular weight

                        mmw = abund_in["MMW"]

                        # Calculate the scaled mass fraction of the clouds

                        cloud_fractions[item] = retrieval_util.scale_cloud_abund(
                            model_param,
                            self.rt_object,
                            self.pressure,
                            temp,
                            mmw,
                            "equilibrium",
                            abund_in,
                            item,
                            model_param[f"{item[:-3].lower()}_tau"],
                            pressure_grid=self.pressure_grid,
                        )

                if "log_tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = model_param["tau_cloud"]

                if tau_cloud is not None:
                    for i, item in enumerate(self.cloud_species):
                        if i == 0:
                            cloud_fractions[item] = 0.0

                        else:
                            cloud_1 = item[:-3].lower()
                            cloud_2 = self.cloud_species[0][:-3].lower()

                            cloud_fractions[item] = model_param[
                                f"{cloud_1}_{cloud_2}_ratio"
                            ]

                # Create a dictionary with the log mass fractions at the cloud base

                log_x_base = retrieval_util.log_x_cloud_base(
                    c_o_ratio, metallicity, cloud_fractions
                )

            elif chemistry == "free":
                # Add the log10 mass fractions of the clouds to the dictionary

                log_x_base = {}

                if "log_tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = model_param["tau_cloud"]

                if tau_cloud is None:
                    for item in self.cloud_species:
                        # Set the log10 of the mass fractions at th
                        # cloud base equal to the value from the
                        # parameter dictionary
                        log_x_base[item[:-3]] = model_param[item]

                else:
                    # Set the log10 of the mass fractions with the
                    # ratios from the parameter dictionary and
                    # scale to the actual mass fractions with
                    # tau_cloud that is used in calc_spectrum_clouds
                    for i, item in enumerate(self.cloud_species):
                        if i == 0:
                            log_x_base[item[:-3]] = 0.0

                        else:
                            cloud_1 = item[:-3].lower()
                            cloud_2 = self.cloud_species[0][:-3].lower()

                            log_x_base[item[:-3]] = model_param[
                                f"{cloud_1}_{cloud_2}_ratio"
                            ]

            # Calculate the petitRADTRANS spectrum
            # for a cloudy atmosphere

            if "fsed_1" in model_param and "fsed_2" in model_param:
                cloud_dict = model_param.copy()
                cloud_dict["fsed"] = cloud_dict["fsed_1"]

                (
                    wavelength,
                    flux_1,
                    emission_contr_1,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    cloud_dict,
                    model_param["logg"],
                    chemistry=chemistry,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

                cloud_dict = model_param.copy()
                cloud_dict["fsed"] = cloud_dict["fsed_2"]

                (
                    wavelength,
                    flux_2,
                    emission_contr_2,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    cloud_dict,
                    model_param["logg"],
                    chemistry=chemistry,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

                flux = (
                    model_param["f_clouds"] * flux_1
                    + (1.0 - model_param["f_clouds"]) * flux_2
                )

                emission_contr = (
                    model_param["f_clouds"] * emission_contr_1
                    + (1.0 - model_param["f_clouds"]) * emission_contr_2
                )

            else:
                (
                    wavelength,
                    flux,
                    emission_contr,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    model_param,
                    model_param["logg"],
                    chemistry=chemistry,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

        elif chemistry == "equilibrium":
            # Calculate the petitRADTRANS spectrum for a clear atmosphere

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object,
                self.pressure,
                temp,
                model_param["logg"],
                model_param["c_o_ratio"],
                model_param["metallicity"],
                p_quench,
                None,
                pressure_grid=self.pressure_grid,
                chemistry=chemistry,
                contribution=True,
            )

        elif chemistry == "free":
            log_x_abund = {}

            for ab_item in self.rt_object.line_species:
                log_x_abund[ab_item] = model_param[ab_item]

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object,
                self.pressure,
                temp,
                model_param["logg"],
                None,
                None,
                None,
                log_x_abund,
                chemistry=chemistry,
                pressure_grid=self.pressure_grid,
                contribution=True,
            )

        if "radius" in model_param:
            # Calculate the planet mass from log(g) and radius

            model_param["mass"] = read_util.get_mass(
                model_param["logg"], model_param["radius"]
            )

            if "distance" in model_param:
                # Use the radius and distance to
                # scale the fluxes to the observer

                scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                    model_param["distance"] * constants.PARSEC
                ) ** 2

                flux *= scaling

        # Apply ISM extinction

        if "ism_ext" in model_param:
            if "ism_red" in model_param:
                ism_reddening = model_param["ism_red"]

            else:
                # Use default ISM reddening (R_V = 3.1) if ism_red is not provided
                ism_reddening = 3.1

            flux = dust_util.apply_ism_ext(
                wavelength, flux, model_param["ism_ext"], ism_reddening
            )

        # Plot 2D emission contribution

        if plot_contribution:
            # Calculate the total optical depth (line and continuum opacities)
            # self.rt_object.calc_opt_depth(10.**model_param['logg'])

            # From Paul: The first axis of total_tau is the coordinate
            # of the cumulative opacity distribution function (ranging
            # from 0 to 1). A correct average is obtained by
            # multiplying the first axis with self.w_gauss, then
            # summing them. This is then the actual wavelength-mean.

            if self.scattering:
                # From petitRADTRANS: Only use 0 index for species
                # because for lbl or test_ck_shuffle_comp = True
                # everything has been moved into the 0th index
                w_gauss = self.rt_object.w_gauss[..., np.newaxis, np.newaxis]
                optical_depth = np.sum(
                    w_gauss * self.rt_object.total_tau[:, :, 0, :], axis=0
                )

            else:
                # TODO Is this correct?
                w_gauss = self.rt_object.w_gauss[
                    ..., np.newaxis, np.newaxis, np.newaxis
                ]
                optical_depth = np.sum(
                    w_gauss * self.rt_object.total_tau[:, :, :, :], axis=0
                )

                # Sum over all species
                optical_depth = np.sum(optical_depth, axis=1)

            mpl.rcParams["font.serif"] = ["Bitstream Vera Serif"]
            mpl.rcParams["font.family"] = "serif"

            plt.rc("axes", edgecolor="black", linewidth=2.5)

            plt.figure(1, figsize=(8.0, 4.0))
            gridsp = mpl.gridspec.GridSpec(1, 1)
            gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            ax = plt.subplot(gridsp[0, 0])

            ax.tick_params(
                axis="both",
                which="major",
                colors="black",
                labelcolor="black",
                direction="in",
                width=1,
                length=5,
                labelsize=12,
                top=True,
                bottom=True,
                left=True,
                right=True,
            )

            ax.tick_params(
                axis="both",
                which="minor",
                colors="black",
                labelcolor="black",
                direction="in",
                width=1,
                length=3,
                labelsize=12,
                top=True,
                bottom=True,
                left=True,
                right=True,
            )

            ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=13)
            ax.set_ylabel("Pressure (bar)", fontsize=13)

            ax.get_xaxis().set_label_coords(0.5, -0.09)
            ax.get_yaxis().set_label_coords(-0.07, 0.5)

            ax.set_yscale("log")

            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))

            press_bar = 1e-6 * self.rt_object.press  # (Ba) -> (Bar)

            xx_grid, yy_grid = np.meshgrid(wavelength, press_bar)

            ax.pcolormesh(
                xx_grid,
                yy_grid,
                emission_contr,
                cmap=plt.cm.bone_r,
                shading="gouraud",
            )

            photo_press = np.zeros(wavelength.shape[0])

            for i in range(photo_press.shape[0]):
                press_interp = interp1d(optical_depth[i, :], self.rt_object.press)
                photo_press[i] = press_interp(1.0) * 1e-6  # cgs to (bar)

            ax.plot(wavelength, photo_press, lw=0.5, color="gray")

            ax.set_xlim(np.amin(wavelength), np.amax(wavelength))
            ax.set_ylim(np.amax(press_bar), np.amin(press_bar))

            if isinstance(plot_contribution, str):
                plt.savefig(plot_contribution, bbox_inches="tight")
            else:
                plt.show()

            plt.clf()
            plt.close()

        # Convolve the spectrum with a Gaussian LSF

        if spec_res is not None:
            flux = retrieval_util.convolve(wavelength, flux, spec_res)

        # Resample the spectrum

        if wavel_resample is not None:
            flux = spectres.spectres(
                wavel_resample,
                wavelength,
                flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

            wavelength = wavel_resample

        if hasattr(self.rt_object, "h_bol"):
            pressure = 1e-6 * self.rt_object.press  # (bar)
            f_bol = -4.0 * np.pi * self.rt_object.h_bol
            f_bol *= 1e-3  # (erg s-1 cm-2) -> (W m-2)
            bol_flux = np.column_stack((pressure, f_bol))

        else:
            bol_flux = None

        return box.create_box(
            boxtype="model",
            model="petitradtrans",
            wavelength=wavelength,
            flux=flux,
            parameters=model_param,
            quantity="flux",
            contribution=emission_contr,
            bol_flux=bol_flux,
        )

    @typechecked
    def get_flux(self, model_param: Dict[str, float]) -> Tuple[float, None]:
        """
        Function for calculating the average flux density for the
        ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.

        Returns
        -------
        float
            Flux (W m-2 um-1).
        NoneType
            Error (W m-2 um-1). Always set to ``None``.
        """

        spectrum = self.get_model(model_param)

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)
