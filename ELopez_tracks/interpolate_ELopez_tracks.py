import numpy as np
import matplotlib.pyplot as plt

file_list = [
    '1pc_HHe',
    '10pc_HHe',
    'Earth_composition',
    'MaxIron',
    'PureRock',
    'Purewater'
]




mass_output = np.arange(0.5, 22, 0.1)

array_output = np.zeros([len(mass_output), len(file_list)+1])

array_output[:,0] = mass_output


fileout = open('ELopez_tracks.dat', 'wb')
fileout.write('descriptor Mearth ')


for file_i, file_name in enumerate(file_list):
    data_in = np.genfromtxt(file_name + '.dat', skip_header=1)

    fit_coeff = np.polynomial.chebyshev.chebfit(data_in[:,0], data_in[:,1], 5)
    radius_out = np.polynomial.chebyshev.chebval(mass_output, fit_coeff)

    array_output[:,file_i+1] = radius_out

    fileout.write(file_name + ' ')

    plt.scatter(data_in[:,0], data_in[:,1])
    plt.plot(mass_output, radius_out)

plt.xlabel('Mass [Mearth]')
plt.ylabel('Radius [Rearth]')
plt.show()


fileout.write(' \n')
for i in xrange(0,len(mass_output)):
    for j in array_output[i,:]:
        fileout.write('{0:f}  '.format(j))
    fileout.write(' \n')

fileout.close()
