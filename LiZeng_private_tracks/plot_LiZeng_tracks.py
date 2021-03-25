import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

file_list = [
      'halfEarthlikecorehalfh2ocold.dat',
      'halfh2o01h1000K1mbar.dat',
      'halfh2o01h300K1mbar.dat',
      'halfh2o01h500K1mbar.dat',
      'halfh2o01h700K1mbar.dat',
      'listh2o1000K1mbar.dat',
      'listh2o500K1mbar.dat',
      'listh2o700K1mbar.dat',
      'listh2omrmix.dat',
      'listhalfh2o1000K1mbar.dat',
      'listhalfh2o500K1mbar.dat',
      'listhalfh2o700K1mbar.dat'
]




for file_i, file_name in enumerate(file_list):
    data_in = np.genfromtxt(file_name, skip_header=0)

    sel = data_in[:,1]> 0.0
    mass_in = data_in[sel,0]
    radius_in = data_in[sel,1]

    plt.plot(mass_in, radius_in)

plt.xlim(30,50)
plt.ylim(3,10)
plt.xlabel('Mass [Mearth]')
plt.ylabel('Radius [Rearth]')
plt.show()
