import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

def count_n_byte(df_header):
    n_bytes = []

    for i, row in df_header.iterrows():
        try:
            _next = str(df_header.iloc[i+1]['position'])
            _current = str(row['position'])

            if _next.isdigit() and _current.isdigit():
                _next = int(_next)
                _current = int(_current)
                df_header.loc[
                    (df_header['position'] == row['position']),
                    'n_byte',
                ] = (_next - _current)
            else:
                print  (_next, _next.isdigit(), _current,  _current.isdigit())
                df_header.loc[
                    (df_header['position'] == row['position']),
                    'n_byte',
                ] = -4
        except ValueError:
            df_header.loc[
                    (df_header['position'] == row['position']),
                    'n_byte',
                ] = -3
        except IndexError:
            try:
                if  int(row['position']) == 401:
                    df_header.loc[
                        (df_header['position'] == row['position']),
                        'n_byte',
                    ] = 80
                elif int(row['position']) == 641:
                    df_header.loc[
                        (df_header['position'] == row['position']),
                        'n_byte',
                    ] = 4
                else:
                    df_header.loc[
                        (df_header['position'] == row['position']),
                        'n_byte',
                    ] = -1
            except ValueError:
                df_header.loc[
                        (df_header['position'] == row['position']),
                        'n_byte',
                    ] = -2

    df_header.n_byte = df_header.n_byte.astype(int)
    return df_header

def reorganise_columns(df_header):
    epn_params = {}
    for i, row in df_header.iterrows():
        position, name, _format, unit, comment = row.values
        epn_params[i] = {}
        epn_params[i]['name'] = name
        epn_params[i]['position'] = position
        epn_params[i]['format'] = _format
        epn_params[i]['unit'] = unit
        epn_params[i]['comment'] = comment
    return pd.DataFrame.from_dict(epn_params, orient='index')

# Main header
df_head = pd.read_csv('epn/tabula-h0535.csv')
df_head = reorganise_columns(df_head)
df_head = count_n_byte(df_head)
df_head.to_csv('epn/header_description.csv', index=False)


# Sub header
df_subheader = pd.read_csv('epn/tabula-h0535-subheader.csv')
df_subheader = reorganise_columns(df_subheader)
# df_subheader = df_subheader.rename(columns={k: k.lower() for k, v in df_subheader.items()})
df_subheader = count_n_byte(df_subheader)
df_subheader.to_csv('epn/header_subheader_description.csv', index=False)

'''
Data section:

4(Nbin − 1) + 641,Data(Nbin),I4,data for last bin of stream
640 +Nrecords ∗ 80,,,,"end of first stream,"
,,,,N mod 80)− 1)∗records = INT(N bin · 0.05) + Θ((4N bin

'''

# Combine both
df = pd.concat([df_head, df_subheader])
df.to_csv('epn/header_description_combined.csv', index=False)

EPN_DESCRIPTION_CSVs = {
    'header': 'epn/header_description.csv',
    'subheader': 'epn/header_subheader_description.csv',
    'combined': 'epn/header_description_combined.csv',
}

def get_epn_format_description(choice='header'):
    if choice not in EPN_DESCRIPTION_CSVs.keys():
        print ('choice must be one of:', EPN_DESCRIPTION_CSVs.keys())
        return False
    return pd.read_csv(EPN_DESCRIPTION_CSVs[choice])


get_epn_format_description('header')
# get_epn_format_description('subheader')
df_comb = get_epn_format_description('combined')
df_comb




# Read EPN data file
class RightAscension:
    def __init__(self, h, m, s):
        self.h = h
        self.m = m
        self.s = s

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = h

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = m

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        self._s = s

class Declination:
    def __init__(self, d, m, s):
        self.d = d
        self.m = m
        self.s = s

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        self._d = d

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = m

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        self._s = s

def cast_format(value, _format, name):
    def cast(value, _format):
        if 'A' in _format:
            return value
        if 'E' in _format or 'F' in _format:
            try:
                value = float(value)
            except:
                pass
            return value
        if 'I' in _format:
            return int(value)

    if ',' in _format:
        formats = _format.split(',')

        if name == 'α2000':
            h = cast(
                value[
                    0 : int(formats[0][1:])
                ], formats[0])

            m = cast(
                value[
                    int(formats[0][1:]) : int(formats[0][1:])+int(formats[1][1:])
                ], formats[1])

            s = cast(
                value[
                    int(formats[0][1:])+int(formats[1][1:]) :
                ], formats[2])

            value = RightAscension(h, m, s)

        if name  == 'δ2000':
            d = cast(
                value[
                    0 : int(formats[0][1:])
                ], formats[0])

            m = cast(
                value[
                    int(formats[0][1:]) : int(formats[0][1:])+int(formats[1][1:])
                ], formats[1])

            s = cast(
                value[
                    int(formats[0][1:])+int(formats[1][1:]) :
                ], formats[2])

            value = Declination(d, m, s)

        if name == 'CDATE':
            value = pd.to_datetime(value, format='%d%m%Y', errors='ignore')

    else:
        value = cast(value, _format)

    return value

def read_epn_file(input_file='epn/raw_files/antt94/B0011+47/antt94_800.epn'):
    epn_data = {}
    processing = 'header'
    with open(input_file, 'r') as f:
        if processing == 'header':
            df = get_epn_format_description(processing)
            for i, row in df.iterrows():
                if 'blank' not in row['name']:
                    f.seek(int(row['position'])-1)
                    epn_data[row['name']] = cast_format(
                        f.read(int(row['n_byte'])).strip(),
                        row['format'],
                        row['name']
                    )
            processing = 'subheader'

        if processing == 'subheader':
            df = get_epn_format_description(processing)
            for i, row in df.iterrows():
                if 'blank' not in row['name']:
                    if row['name'] == 'IDfield':
                        f.seek(int(row['position'])-1)
                        stokes = cast_format(
                            f.read(int(row['n_byte'])).strip(),
                            row['format'],
                            row['name']
                        )
                        epn_data[stokes] = {}
                    else:
                        epn_data[stokes][row['name']] = cast_format(
                            f.read(int(row['n_byte'])).strip(),
                            row['format'],
                            row['name']
                        )
            processing = 'data'

        if processing == 'data':
            pass

    return epn_data

epn_data = read_epn_file(input_file='epn/raw_files/antt94/B0011+47/antt94_800.epn')


formats = {
    'A1': str,
    'A6': str,
    'A12': str,
    'A68': str,
    'A8': str,
    'E12.6': float,
    'F10.3': float,
    'F12.6': float,
    'F12.8': float,
    'F16.12': float,
    'F17.5': float,
    'F8.3': float,
    'I2': int,
    'I4': int,
    'I6': int,
    'I2,I2,F6.3': {},
    'I2,I2,I4': {},
    'I3,I2,F6.3': {}
}

formats_list = np.unique(df_comb['format'])
