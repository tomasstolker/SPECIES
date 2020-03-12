import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestModel:

    def setup_class(self):
        self.limit = 1e-10
        self.test_path = os.path.dirname(__file__) + '/'
        self.model_param = {'teff': 2200., 'logg': 4.5, 'radius': 1., 'distance': 10.}

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_model(self):
        database = species.Database()

        database.add_model('ames-cond',
                           wavel_range=(1., 5.),
                           spec_res=100.,
                           teff_range=(2000., 2500))

        read_model = species.ReadModel('ames-cond')
        assert read_model.model == 'ames-cond'

    def test_get_model(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')

        model_box = read_model.get_model(self.model_param,
                                         spec_res=100.,
                                         magnitude=False,
                                         smooth=True)

        assert np.sum(model_box.wavelength) == pytest.approx(39.36850660634572, rel=self.limit, abs=0.)
        assert np.sum(model_box.flux) == pytest.approx(7.549674442955264e-13, rel=self.limit, abs=0.)

        model_box = read_model.get_model(self.model_param,
                                         spec_res=100.,
                                         magnitude=True,
                                         smooth=True)

        assert np.sum(model_box.wavelength) == pytest.approx(39.36850660634572, rel=self.limit, abs=0.)
        assert np.sum(model_box.flux) == pytest.approx(275.6061491339208, rel=self.limit, abs=0.)

    def test_get_data(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        model_box = read_model.get_data(self.model_param)

        assert np.sum(model_box.wavelength) == pytest.approx(47.859770458276856, rel=self.limit, abs=0.)
        assert np.sum(model_box.flux) == pytest.approx(8.275363683007199e-13, rel=self.limit, abs=0.)

    def test_get_flux(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        flux = read_model.get_flux(self.model_param)

        assert flux[0] == pytest.approx(3.3368963026400554e-14, rel=self.limit, abs=0.)

    def test_get_magnitude(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        magnitude = read_model.get_magnitude(self.model_param)

        assert magnitude[0] == pytest.approx(11.357124426046317, rel=self.limit, abs=0.)
        assert magnitude[1] == pytest.approx(11.357124426046317, rel=self.limit, abs=0.)

    def test_get_bounds(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        bounds = read_model.get_bounds()

        assert bounds['teff'] == (2000., 2500.)
        assert bounds['logg'] == (0., 6.)

    def test_get_wavelengths(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        wavelengths = read_model.get_wavelengths()

        assert np.sum(wavelengths) == pytest.approx(401.2594, rel=1e-7, abs=0.)

    def test_get_points(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        points = read_model.get_points()

        assert np.sum(points['teff']) == 13500.
        assert np.sum(points['logg']) == 39.

    def test_get_parameters(self):
        read_model = species.ReadModel('ames-cond', filter_name='Paranal/NACO.H')
        parameters = read_model.get_parameters()

        assert parameters == ['teff', 'logg']
