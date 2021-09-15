"""
Module for setting up species in the working folder.
"""

import configparser
import json
import os
import socket
import urllib.request

import h5py
import species


class SpeciesInit:
    """
    Class for initiating species by creating the database and configuration file in case they are
    not present in the working folder, and creating the data folder for storage of input data.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        print(f'Initiating species v{species.__version__}...', end='', flush=True)

        working_folder = os.path.abspath(os.getcwd())

        config_file = os.path.join(working_folder, 'species_config.ini')

        print(' [DONE]')

        try:
            contents = urllib.request.urlopen('https://pypi.org/pypi/species/json',
                                              timeout=1.).read()

            data = json.loads(contents)
            latest_version = data['info']['version']

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and species.__version__ != latest_version:
            print(f'A new version ({latest_version}) is available!')
            print('Want to stay informed about updates, bug fixes, and new features?')
            print('Please consider using \'Watch\' button on the Github page:')
            print('https://github.com/tomasstolker/species')

        if not os.path.isfile(config_file):

            print('Creating species_config.ini...', end='', flush=True)

            with open(config_file, 'w') as file_obj:
                file_obj.write('[species]\n')
                file_obj.write('database = species_database.hdf5\n')
                file_obj.write('data_folder = ./data/\n')

            print(' [DONE]')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database_file = os.path.abspath(config['species']['database'])
        data_folder = os.path.abspath(config['species']['data_folder'])

        print(f'Database: {database_file}')
        print(f'Data folder: {data_folder}')
        print(f'Working folder: {working_folder}')

        if not os.path.isfile(database_file):
            print('Creating species_database.hdf5...', end='', flush=True)
            h5_file = h5py.File(database_file, 'w')
            h5_file.close()
            print(' [DONE]')

        if not os.path.exists(data_folder):
            print('Creating data folder...', end='', flush=True)
            os.makedirs(data_folder)
            print(' [DONE]')
