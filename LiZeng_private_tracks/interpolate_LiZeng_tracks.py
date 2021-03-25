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
    mass_in_log = np.log10(mass_in)



    f2 = interp1d(mass_in_log, radius_in, kind='cubic')



    fileout = open('interpolated_'+file_name, 'wb')

    mass_out_log = np.arange(np.log10(mass_in[0]), np.log10(mass_in[-1]), 0.05)
    mass_out = 10**(mass_out_log)
    radius_out = f2(mass_out_log)

    if mass_in[0] > 0.25:
        print file_name
        m = (radius_out[1]-radius_out[0])/(mass_out_log[1]-mass_out_log[0])
        q = ((radius_out[1]+radius_out[0])-m*(mass_out_log[1]+mass_out_log[0]))/2.
        mass_out_ext_log= np.arange(np.log10(0.200),np.log10(mass_in[0]-0.005), 0.1)
        mass_out_ext = 10**mass_out_ext_log
        radius_out_ext = m*mass_out_ext_log + q

        print mass_in[0], mass_in[1], '   ', mass_out_ext
        for m, r in zip(mass_out_ext, radius_out_ext):
            fileout.write('{0:f} {1:f} \n'.format(m,r))

        #plt.plot(np.log10(mass_out_ext), radius_out_ext)
        plt.plot(mass_out_ext,radius_out_ext)


    for m, r in zip(mass_out, radius_out):
        fileout.write('{0:f} {1:f} \n'.format(m,r))

    fileout.close()
    plt.scatter(mass_in, radius_in)
    plt.plot(mass_out, radius_out)

plt.xscale('log')
plt.xlim(0.01,22)
plt.ylim(0,3)
plt.xlabel('Mass [Mearth]')
plt.ylabel('Radius [Rearth]')
plt.show()
