import numpy as np

pl_names = np.asarray(['Kepler-103 b', 'Kepler-103 c', 'PH-2 b'])
pl_orbper = np.asarray([15.965329, 179.60958, 282.52540])
pl_ttvflag = np.asarray([0.0, 0.0, 0.0])
pl_upperlimit = np.asarray([False, False, False])

st_rad =  np.asarray([1.482,1.482,0.959])
st_mass = np.asarray([1.214,1.214,0.953])
st_teff = np.asarray([6009., 6009., 5691.])

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = np.asarray([ 11.68, 58.54, 108.54])
pl_masseerr1 = np.asarray([4.32, 11.17, 29.53])
pl_masseerr2 = np.asarray([4.73, 11.43, 32.59])
pl_rade = np.asarray([3.421, 5.251, 9.47])
pl_radeerr1 = np.asarray([0.054, 0.086, 0.15])
pl_radeerr2 = np.asarray([0.053, 0.085, 0.14])


fileout = open('my_planets.dat','w')
fileout.write('name orbital_period mass mass_error_max mass_error_min radius radius_error_max radius_error_min star_mass star_radius star_teff pl_ttvflag textbox_ha textbox_va upper_limit\n')
if np.size(pl_masse)>1:
    for nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit in \
            zip(pl_names,pl_orbper,
            pl_masse,pl_masseerr1,pl_masseerr2,
            pl_rade,pl_radeerr1,pl_radeerr2,
            st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit):

            fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},right,bottom,{12:f} \n'.format(
            nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit))
else:
    fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},right,bottom,{12:f}\n'.format(
        pl_names,pl_orbper,
        pl_masse,pl_masseerr1,pl_masseerr2,
        pl_rade,pl_radeerr1,pl_radeerr2,
        st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit))

fileout.close()
