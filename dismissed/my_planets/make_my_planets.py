import numpy as np

G_grav = 6.67398e-11
M_sun = 1.98892e30
M_jup = 1.89813e27
M_ratio = M_sun / M_jup
Mu_sun = 132712440018.9
seconds_in_day = 86400
AU_km = 1.4960 * 10 ** 8

pl_names = 'Kepler-19b'
pl_orbper = 9.2867592445
st_rad = 0.859
st_mass = 0.936
st_teff = 5541
pl_ttvflag = 1.000000

#pl_masseerr1: +1 sgima
#pl_masseerr2: -1 sgima


pl_masse = 8.40247027767
pl_masseerr1 = 1.54
pl_masseerr2 = 1.62
pl_rade = 2.209
pl_radeerr1 = 0.049
pl_radeerr2 = 0.049


pl_masseerr_avg = (pl_masseerr1 + pl_masseerr2)/2.0
pl_radeerr_avg = (pl_radeerr1 + pl_radeerr2)/2.0

pl_dens = pl_masse/pl_rade**3
pl_denserr1 =pl_dens * np.sqrt( (pl_masseerr1/pl_masse)**2 + 9*(pl_radeerr2/pl_rade)**2)
pl_denserr2 =pl_dens * np.sqrt( (pl_masseerr2/pl_masse)**2 + 9*(pl_radeerr1/pl_rade)**2)
pl_denserr_avg = pl_dens * np.sqrt( (pl_masseerr_avg/pl_masse)**2 + 9*(pl_radeerr_avg/pl_rade)**2)


a_smj_AU = np.power((Mu_sun * np.power(pl_orbper * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * st_mass, 1.00 / 3.00)
insol = st_rad**2 * (st_teff/5777.0)**4 / a_smj_AU**2

perc_error = pl_masseerr_avg/pl_masse * 100.

perc_error = pl_denserr_avg/pl_dens * 100.


fileout = open('my_planets.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
if np.size(pl_masse)>1:
    for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ep, ins, ttv in \
            zip(pl_names,pl_orbper,pl_masse,pl_masseerr1,pl_masseerr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_dens,pl_denserr1,pl_denserr2,perc_error,insol,pl_ttvflag):
            fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:15f} {13:5.1f}\n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
else:
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:15f} {13:5.1f}\n'.format(
        pl_names,pl_orbper,pl_masse,pl_masseerr1,pl_masseerr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_dens,pl_denserr1,pl_denserr2,perc_error,insol,pl_ttvflag))

fileout.close()
