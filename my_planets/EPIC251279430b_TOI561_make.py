import numpy as np

pl_names = ['HD 80653 b','TOI-561 b','TOI-561 c','TOI-561 d','TOI-561 e']
pl_orbper = [0.71957,0.446578,10.779,25.62,77.23]
pl_ttvflag = [0.0,0.0,0.0,0.0,0.0]
pl_upperlimit = [False,False,False,False, False]

st_rad =  [1.22,0.849,0.849,0.849,0.849]
st_mass = [1.18,0.785,0.785,0.785,0.785]
st_teff = [6000,5455,5455,5455,5455]

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = [5.60,1.59,5.40,11.95,16.0]
pl_masseerr1 = [0.43,0.36,0.98,1.28,2.3]
pl_masseerr2 = [0.43,0.36,0.98,1.28,2.3]
pl_rade = [1.613,1.423,2.878,2.53,2.67]
pl_radeerr1 = [0.071,0.066,0.096,0.13,0.11]
pl_radeerr2 = [0.071,0.066,0.096,0.13,0.11]


fileout = open('my_planets.dat','w')
fileout.write('name orbital_period mass mass_error_max mass_error_min radius radius_error_max radius_error_min star_mass star_radius star_teff pl_ttvflag textbox_ha textbox_va upper_limit\n')
if np.size(pl_masse)>1:
    for nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit in \
            zip(pl_names,pl_orbper,
            pl_masse,pl_masseerr1,pl_masseerr2,
            pl_rade,pl_radeerr1,pl_radeerr2,
            st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit):

            fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},left,bottom,{12:f} \n'.format(
            nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit))
else:
    fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},left,bottom,{12:f}\n'.format(
        pl_names,pl_orbper,
        pl_masse,pl_masseerr1,pl_masseerr2,
        pl_rade,pl_radeerr1,pl_radeerr2,
        st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit))

fileout.close()
