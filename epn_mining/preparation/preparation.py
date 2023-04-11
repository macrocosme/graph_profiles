from os import listdir, makedirs
from os.path import isfile, join, exists
from pandas import DataFrame
import json

def check_directory_exists(directory):
    """Check if directory (string) ends with a slash.
    If directory does not end with a slash, add one at the end.
    Parameters
    ----------
    directory  : str
    Returns
    -------
    directory  : str
    """
    if not exists(directory):
        makedirs(directory)
    return directory

def list_files_in_current_path(path):
    """ Returns files in the current folder only """
    return [ f for f in listdir(path) if isfile(join(path,f)) ]

def cast_cm_to_mhz(freq):
    cm_mhz = {
        10: 3100, # used values from EPN webpage
        20: 1369,
        50: 728,
    }
    return cm_mhz[freq]

def read_metadata_file(base_path = '../www.epta.eu.org/epndb/psrfits', input_filename='file_log.txt', verbose=False):
    """Read EPN metadata into Pandas data frame

    Parameters
    ----------
    base_path: str
        Base path to metadata input file
    input_filename: str
        Input filename

    Returns
    -------
    df_good: Pandas.Dataframe
        The database as data frame
    """
    if verbose:
        print ('Reading metadata file')

    if not exists(base_path):
        print ('download epn data from the root of the repository using the following:')
        print ('wget -O www.epta.eu.org/epndb/psrfits/ http://www.epta.eu.org/epndb/psrfits/ --recursive -A .fits -l 10 --no-parent')
        exit()
        # print ('Downloading data from epta.eu.org/epndb. This may take a few minutes. Possibly enough time for a cuppa.')
        # check_directory_exists('../www.epta.eu.org/epndb/psrfits/')
        # wget.download('../www.epta.eu.org/epndb/psrfits/ http://www.epta.eu.org/epndb/psrfits/ --recursive -A .fits -l 10 --no-parent')
        # os.system('wget -O ../www.epta.eu.org/epndb/psrfits/ http://www.epta.eu.org/epndb/psrfits/ --recursive -A .fits -l 10 --no-parent')

    #Main output
    pulsars = []

    #Main inputs
    states={'header': 0, 'candidates': 1}
    state=0
    counter = 0

    files = []

    with open(input_filename, 'w+') as f:
        f.write('Start logging.\n')

    with open('../input_metadata/meta.txt', 'r') as f:   # To be replaced with http://www.epta.eu.org/epndb/list.php
        for line in f:
            if state == states['header']:
                try:
                    jname = line.split(' ')[0]
                    obs_count = counter = int(line.split(' ')[1].split('[')[1].split(']')[0])
                    bname = 'n/a'
                    state = states['candidates']
                except IndexError:
                    jname = line.split(' ')[0]
                    bname = line.split(' ')[-2].split('(')[1].split(')')[0]
                    obs_count = counter = int(line.split(' ')[-1].split('[')[1].split(']')[0])
                    state = states['candidates']

            elif state == states['candidates']:
                no_error = True
                observation = {}
                observation['jname'] = jname
                observation['bname'] = bname
                observation['frequency (MHz)'] = line.split(',')[0].split(' MHz')[0]
                observation['stokes'] = line.split(', ')[1].split(' [')[0]
                observation['reference'] = line.split(', ')[1].split('[')[1].split(']')[0]
                file = list_files_in_current_path(base_path)

                no_files = False
                # Get files list
                try:
                    file_location = "%s/%s/%s" % (
                        base_path,
                        observation['reference'],
                        observation['jname'],
                    )
                    files = list_files_in_current_path(file_location)
                except FileNotFoundError:
                    try:
                        file_location = "%s/%s/%s" % (
                        base_path,
                        observation['reference'],
                        observation['bname'],
                        )
                        files = list_files_in_current_path(file_location)
                    except FileNotFoundError:
                        try:
                            file_location = "%s/%s" % (
                                base_path,
                                observation['reference']
                            )
                            files = list_files_in_current_path(file_location)
                        except FileNotFoundError:
                            no_files = True

                if len(files) >= 1:
                    if len(files) == 1:
                        observation['file location'] = "%s/%s" % (
                            file_location,
                            files[0]
                        )
                        observation['file name'] = files[0]
                    else:
                        for file in files:
                            try:
                                # Horrible ifs structure
                                if 'SFTC' not in file:
                                    freq = file.split('_')[-1].split('.')[0]
                                    if 'cm' in freq:
                                        freq = cast_cm_to_mhz(int(freq.split('cm')[0]))
                                    elif 'MHz' in freq:
                                        freq = freq.split('MHz')[0]
                                else:
                                    with open("%s/%s" % (
                                        file_location,
                                        file
                                    )) as f:
                                        data = json.load(f)
                                    freq = data['hdr']['freq']

                                if int(observation['frequency (MHz)'].split('.')[0]) == int(freq):
                                    observation['file location'] = "%s/%s" % (
                                        file_location,
                                        file
                                    )
                                    observation['file name'] = file

                            except ValueError:
                                no_error = False
                                pass
                else:
                    observation['file location'] = 'n/a'
                    observation['file name'] = 'n/a'

                if no_error:
                    pulsars.append(observation)

                counter -= 1
                if counter == 0:
                    state = states['header']
                    files = []

    # Make dataframes from pulsars
    df = DataFrame.from_dict(pulsars)
    df_good = df.loc[df['file location'] != 'n/a']
    del df

    return df_good
