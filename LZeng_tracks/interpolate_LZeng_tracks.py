import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

file_list = [
    'Earthlike1h300K1mbar',
    'Earthlike1h500K1mbar',
]




mass_output = np.arange(0.3, 40, 0.1)


for file_i, file_name in enumerate(file_list):
    fileout = open(file_name + '_interpolated.txt', 'w')
    data_in = np.genfromtxt(file_name + '.txt', skip_header=0)



    #fit_coeff = np.polynomial.chebyshev.chebfit(data_in[:,0], data_in[:,1], 3)
    #radius_out = np.polynomial.chebyshev.chebval(mass_output, fit_coeff)

    f_interp = interpolate.interp1d(data_in[:,0], data_in[:,1], kind='cubic')
    radius_out = f_interp(mass_output)

    #radius_out = interpolate.spline(data_in[:,0], data_in[:,1], mass_output, order=3)
    for m, r in zip(mass_output, radius_out):
        fileout.write('{0:f} {1:f}\n'.format(m ,r))
    fileout.close()
    plt.scatter(data_in[:,0], data_in[:,1])
    plt.plot(mass_output, radius_out)

plt.xlabel('Mass [Mearth]')
plt.ylabel('Radius [Rearth]')
plt.show()
