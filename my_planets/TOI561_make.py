import numpy as np

pl_names = np.asarray(['TOI-561c', 'TOI-561b', 'TOI-561d'])
pl_orbper = np.asarray([10.781327, 0.446547, 16.371651])
pl_ttvflag = np.asarray([0.0, 0.0, 0.0])
pl_upperlimit = np.asarray([False, False, True])

st_rad =  np.asarray([0.850374,0.850374,0.850374])
st_mass = np.asarray([0.775885,0.775885,0.775885])
st_teff = np.asarray([5437.880175, 5437.880175,5437.880175])

#pl_masseerr1: +1 sigma
#pl_masseerr2: -1 sigma

pl_masse = np.asarray([ 5.12, 1.835849, 3.00])
pl_masseerr1 = np.asarray([1.02,0.50000,0.26])
pl_masseerr2 = np.asarray([1.07,0.50000,0.187940])
pl_rade = np.asarray([2.88, 1.43, 2.61])
pl_radeerr1 = np.asarray([0.10, 0.06, 0.09])
pl_radeerr2 = np.asarray([0.10, 0.06, 0.09])


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
