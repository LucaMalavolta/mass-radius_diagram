import numpy as np

pl_names = np.asarray(['Kepler-9 b', 'Kepler-9 c'])
pl_orbper = np.asarray([19.24, 38.9852])
pl_ttvflag = np.asarray([1.0, 1.0])
pl_upperlimit = np.asarray([False, False])

st_rad =  np.asarray([0.999,0.999])
st_mass = np.asarray([1.026,1.026])
st_teff = np.asarray([5801, 5801])

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = np.asarray([ 43.4, 30.0])
pl_masseerr1 = np.asarray([1.8,1.3]) #positive error bar
pl_masseerr2 = np.asarray([2.2,1.5]) #negative error bar
pl_rade = np.asarray([8.29, 8.08])
pl_radeerr1 = np.asarray([0.43, 0.41])
pl_radeerr2 = np.asarray([0.54, 0.54])


fileout = open('my_planets.dat','w')
fileout.write('name orbital_period mass mass_error_max mass_error_min radius radius_error_max radius_error_min star_mass star_radius star_teff pl_ttvflag textbox_ha textbox_va \n')
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
