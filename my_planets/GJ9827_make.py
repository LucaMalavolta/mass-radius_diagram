import numpy as np

pl_names = np.asarray(['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d'])
pl_orbper = np.asarray([1.2089, 3.6481, 6.21047])
pl_ttvflag = np.asarray([0.0, 0.0, 0.0])
pl_upperlimit = np.asarray([False, True, False])

st_rad =  np.asarray([0.602,0.602,0.602])
st_mass = np.asarray([0.606,0.606,0.606])
st_teff = np.asarray([4305, 4305,4305])

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = np.asarray([ 4.910488, 0.839642, 4.046201])
pl_masseerr1 = np.asarray([0.493563,0.666171,0.821256])
pl_masseerr2 = np.asarray([0.487035,0.494051,0.836228])
pl_rade = np.asarray([1.577, 1.241, 2.022])
pl_radeerr1 = np.asarray([0.027, 0.024, 0.046])
pl_radeerr2 = np.asarray([0.031, 0.026, 0.043])


fileout = open('my_planets.dat','w')
fileout.write('name orbital_period mass mass_error_max mass_error_min radius radius_error_max radius_error_min star_mass star_radius star_teff pl_ttvflag textbox_ha textbox_va upper_limit\n')
if np.size(pl_masse)>1:
    for nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit in \
            zip(pl_names,pl_orbper,
            pl_masse,pl_masseerr1,pl_masseerr2,
            pl_rade,pl_radeerr1,pl_radeerr2,
            st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit):

            fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},left,top,{12:f} \n'.format(
            nm, p, m, mp, mm, r, rp, rm, mass, rad, teff, ttv, upp_limit))
else:
    fileout.write('{0:s},{1:f},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f},{9:f},{10:f},{11:5.1f},left,top,{12:f}\n'.format(
        pl_names,pl_orbper,
        pl_masse,pl_masseerr1,pl_masseerr2,
        pl_rade,pl_radeerr1,pl_radeerr2,
        st_mass, st_rad, st_teff,pl_ttvflag, pl_upperlimit))

fileout.close()
