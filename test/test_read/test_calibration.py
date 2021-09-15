import os
import shutil

import pytest
import numpy as np

import species
from species.util import test_util


class TestCalibration:

    def setup_class(self):
        self.limit = 1e-8
        self.test_path = os.path.dirname(__file__) + '/'
        self.model_param = {'scaling': 1.}

    def teardown_class(self):
        os.remove('species_database.hdf5')
        os.remove('species_config.ini')
        shutil.rmtree('data/')

    def test_species_init(self):
        test_util.create_config('./')
        species.SpeciesInit()

    def test_read_calibration(self):
        database = species.Database()
        database.add_spectra('vega')

        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        assert read_calib.wavel_range == pytest.approx((1.44, 1.88), rel=1e-7, abs=0.)

    def test_resample_spectrum(self):
        read_calib = species.ReadCalibration('vega')
        spec_box = read_calib.resample_spectrum(np.linspace(1., 2., 10), apply_mask=True)

        assert np.sum(spec_box.wavelength) == 15.
        assert np.sum(spec_box.flux) == pytest.approx(2.288734760321133e-08, rel=self.limit, abs=0.)

    def test_get_spectrum(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.Lp')
        spec_box = read_calib.get_spectrum(self.model_param, apply_mask=True, spec_res=100.)

        assert np.sum(spec_box.wavelength) == pytest.approx(183.34793783628004, rel=self.limit, abs=0.)
        assert np.sum(spec_box.flux) == pytest.approx(2.363239096767195e-09, rel=self.limit, abs=0.)

    def test_get_flux(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        flux = read_calib.get_flux(model_param=self.model_param)

        assert flux[0] == pytest.approx(1.132902304231636e-09, rel=self.limit, abs=0.)

    def test_get_magnitude(self):
        read_calib = species.ReadCalibration('vega', filter_name='Paranal/NACO.H')
        app_mag, abs_mag = read_calib.get_magnitude(model_param=self.model_param)

        assert app_mag[0] == 0.03
        assert abs_mag[0] is None
