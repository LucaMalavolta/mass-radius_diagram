import numpy as np

pl_names = np.asarray(['FAKE-1 b', 'FAKE-1 d'])
pl_orbper = np.asarray([2.1089, 4.45047])
pl_ttvflag = np.asarray([0.0, 1.0])
pl_upperlimit = np.asarray([False, False])

st_rad =  np.asarray([0.702,0.702])
st_mass = np.asarray([0.706,0.706])
st_teff = np.asarray([5125,5125])

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = np.asarray([ 5.04, 4.80])
pl_masseerr1 = np.asarray([0.930, 1.93])
pl_masseerr2 = np.asarray([0.930, 2.02])
pl_rade = np.asarray([1.253,  2.834])
pl_radeerr1 = np.asarray([0.067,  0.056])
pl_radeerr2 = np.asarray([0.041,  0.033])

fileout = open('other_planets.dat','w')
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
