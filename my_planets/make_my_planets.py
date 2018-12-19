import numpy as np

pl_names = np.asarray(['FAKE-1 b', 'FAKE-1 c', 'FAKE-1 d'])
pl_orbper = np.asarray([2.1089, 13.314, 4.45047])
pl_ttvflag = np.asarray([0.0, 0.0, 1.0])
pl_upperlimit = np.asarray([False, True, False])

st_rad =  np.asarray([0.702,0.702,0.702])
st_mass = np.asarray([0.706,0.706,0.706])
st_teff = np.asarray([5125, 5125,5125])

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima

pl_masse = np.asarray([ 7.04, 1.030, 2.80])
pl_masseerr1 = np.asarray([0.630,1.040,1.13])
pl_masseerr2 = np.asarray([0.630,0.740,1.02])
pl_rade = np.asarray([1.453, 1.141, 2.234])
pl_radeerr1 = np.asarray([0.037, 0.034, 0.026])
pl_radeerr2 = np.asarray([0.021, 0.036, 0.013])


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
