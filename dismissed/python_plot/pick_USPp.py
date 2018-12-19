from plot_routines import *
import os

short_planet_list = [
'PSR J1719-1438 b',
'KOI-55 b',
'KOI-55 c',
'Kepler-78 b',
'K2-131 b',
'K2-22 b',
'55 Cnc e',
'WASP-19 b',
'WASP-47 e',
'WASP-43 b',
'Kepler-10 b',
'HATS-18 b',
'CoRoT-7 b',
'WASP-103 b',
'WASP-18 b',
'HD 3167 b',
'KELT-16 b'
]

os.system('head -1 defaults_radec.csv > uspp_NASA.csv')
for planet in short_planet_list:


    os.system('more  ')