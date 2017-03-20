import numpy as np

data = np.genfromtxt(
    'defaults.csv',           # file name
    skip_header=1,          # lines to skip at the top
    skip_footer=0,          # lines to skip at the bottom
    delimiter=',',          # column delimiter
    dtype='float32',        # data type
    filling_values=-0.0001,       # fill missing values with 0
    #usecols = (0,2,3,5),    # columns to read
    names=['pl_name','pl_orbper','pl_masse','pl_masseerr1','pl_masseerr2','pl_rade','pl_radeerr1','pl_radeerr2','st_mass','st_rad','st_teff','pl_ttvflag'])     # column names

names = np.genfromtxt(
    'defaults.csv',           # file name
    skip_header=1,          # lines to skip at the top
    skip_footer=0,          # lines to skip at the bottom
    delimiter=',',          # column delimiter
    dtype=str,        # data type
    filling_values=-1.000,       # fill missing values with 0
    usecols = (0))    # columns to read

print names


sel = (data['pl_masse'] > 0.0) & (data['pl_masse'] > 0.0) & (data['pl_orbper'] > 0.0) & (data['st_mass']>0.0) & (data['st_rad']>0.) & \
    (np.abs(data['pl_masseerr1'])>0.01 ) & (np.abs(data['pl_masseerr2'])>0.01 )

G_grav = 6.67398e-11
M_sun = 1.98892e30
M_jup = 1.89813e27
M_ratio = M_sun / M_jup
Mu_sun = 132712440018.9
seconds_in_day = 86400
AU_km = 1.4960 * 10 ** 8

pl_names     = names[sel]
pl_orbper    = data['pl_orbper'][sel]
st_rad       = data['st_rad'][sel]
st_teff      = data['st_teff'][sel]
st_mass      = data['st_mass'][sel]
pl_masse     = data['pl_masse'][sel]
pl_masseerr1 = np.abs(data['pl_masseerr1'][sel])
pl_masseerr2 = np.abs(data['pl_masseerr2'][sel])
pl_rade      = data['pl_rade'][sel]
pl_radeerr1  = np.abs(data['pl_radeerr1'][sel])
pl_radeerr2  = np.abs(data['pl_radeerr2'][sel])
pl_ttvflag  = data['pl_ttvflag'][sel]

a_smj_AU = np.power((Mu_sun * np.power(pl_orbper * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * st_mass, 1.00 / 3.00)

insol = st_rad**2 * (st_teff/5777.0)**4 / a_smj_AU**2

pl_masseerr_avg = (pl_masseerr1 + pl_masseerr2)/2.0
pl_radeerr_avg = (pl_radeerr1 + pl_radeerr2)/2.0

pl_dens = pl_masse/pl_rade**3
pl_denserr1 =pl_dens * np.sqrt( (pl_masseerr1/pl_masse)**2 + 9*(pl_radeerr2/pl_rade)**2)
pl_denserr2 =pl_dens * np.sqrt( (pl_masseerr2/pl_masse)**2 + 9*(pl_radeerr1/pl_rade)**2)
pl_denserr_avg = pl_dens * np.sqrt( (pl_masseerr_avg/pl_masse)**2 + 9*(pl_radeerr_avg/pl_rade)**2)

perc_error = pl_masseerr_avg/pl_masse * 100.

perc_error = pl_denserr_avg/pl_dens * 100.

sel20 = (perc_error <= 20.)
sel40 = (perc_error > 20.) & (perc_error <= 40.)
sel60 = (perc_error > 40.) & (perc_error <= 60.)
sel80 = (perc_error > 60.)
#sel100 = (perc_error > 60.) & (perc_error <= 70.)

nottv_sel20 = (perc_error <= 20.)& (pl_ttvflag<1)
nottv_sel40 = (perc_error > 20.) & (perc_error <= 40.) & (pl_ttvflag<1)
nottv_sel60 = (perc_error > 40.) & (perc_error <= 60.)& (pl_ttvflag<1)
nottv_sel80 = (perc_error > 60.)& (pl_ttvflag<1)

ttv_sel20 = (perc_error <= 20.) & (pl_ttvflag>0)
ttv_sel40 = (perc_error > 20.) & (perc_error <= 40.) & (pl_ttvflag>0)
ttv_sel60 = (perc_error > 40.) & (perc_error <= 60.) & (pl_ttvflag>0)
ttv_sel80 = (perc_error > 60.) & (pl_ttvflag>0)


fileout = open('veusz.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, asm, ins, ttv in \
        zip(pl_names,pl_orbper,pl_masse,pl_masseerr1,pl_masseerr2,pl_rade,pl_radeerr1,pl_radeerr2,pl_dens,pl_denserr1,pl_denserr2,a_smj_AU,insol,pl_ttvflag):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:15f} {13:5.1f}\n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, asm, ins, ttv))
fileout.close()


fileout = open('veusz_P20.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[sel20],pl_orbper[sel20],pl_masse[sel20],pl_masseerr1[sel20],pl_masseerr2[sel20],pl_rade[sel20],pl_radeerr1[sel20],pl_radeerr2[sel20],pl_dens[sel20],pl_denserr1[sel20],pl_denserr2[sel20],insol[sel20],pl_ttvflag[sel20]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()

fileout = open('veusz_P40.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[sel40],pl_orbper[sel40],pl_masse[sel40],pl_masseerr1[sel40],pl_masseerr2[sel40],pl_rade[sel40],pl_radeerr1[sel40],pl_radeerr2[sel40],pl_dens[sel40],pl_denserr1[sel40],pl_denserr2[sel40],insol[sel40],pl_ttvflag[sel40]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()



fileout = open('veusz_P60.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[sel60],pl_orbper[sel60],pl_masse[sel60],pl_masseerr1[sel60],pl_masseerr2[sel60],pl_rade[sel60],pl_radeerr1[sel60],pl_radeerr2[sel60],pl_dens[sel60],pl_denserr1[sel60],pl_denserr2[sel60],insol[sel60],pl_ttvflag[sel60]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()

fileout = open('veusz_P80.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[sel80],pl_orbper[sel80],pl_masse[sel80],pl_masseerr1[sel80],pl_masseerr2[sel80],pl_rade[sel80],pl_radeerr1[sel80],pl_radeerr2[sel80],pl_dens[sel80],pl_denserr1[sel80],pl_denserr2[sel80],insol[sel80],pl_ttvflag[sel80]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()




fileout = open('veusz_P20_ttv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[ttv_sel20],pl_orbper[ttv_sel20],pl_masse[ttv_sel20],pl_masseerr1[ttv_sel20],pl_masseerr2[ttv_sel20],pl_rade[ttv_sel20],pl_radeerr1[ttv_sel20],pl_radeerr2[ttv_sel20],pl_dens[ttv_sel20],pl_denserr1[ttv_sel20],pl_denserr2[ttv_sel20],insol[ttv_sel20],pl_ttvflag[ttv_sel20]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()

fileout = open('veusz_P40_ttv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[ttv_sel40],pl_orbper[ttv_sel40],pl_masse[ttv_sel40],pl_masseerr1[ttv_sel40],pl_masseerr2[ttv_sel40],pl_rade[ttv_sel40],pl_radeerr1[ttv_sel40],pl_radeerr2[ttv_sel40],pl_dens[ttv_sel40],pl_denserr1[ttv_sel40],pl_denserr2[ttv_sel40],insol[ttv_sel40],pl_ttvflag[ttv_sel40]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()

fileout = open('veusz_P60_ttv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[ttv_sel60],pl_orbper[ttv_sel60],pl_masse[ttv_sel60],pl_masseerr1[ttv_sel60],pl_masseerr2[ttv_sel60],pl_rade[ttv_sel60],pl_radeerr1[ttv_sel60],pl_radeerr2[ttv_sel60],pl_dens[ttv_sel60],pl_denserr1[ttv_sel60],pl_denserr2[ttv_sel60],insol[ttv_sel60],pl_ttvflag[ttv_sel60]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()

fileout = open('veusz_P80_ttv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins ttv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv in \
        zip(pl_names[ttv_sel80],pl_orbper[ttv_sel80],pl_masse[ttv_sel80],pl_masseerr1[ttv_sel80],pl_masseerr2[ttv_sel80],pl_rade[ttv_sel80],pl_radeerr1[ttv_sel80],pl_radeerr2[ttv_sel80],pl_dens[ttv_sel80],pl_denserr1[ttv_sel80],pl_denserr2[ttv_sel80],insol[ttv_sel80],pl_ttvflag[ttv_sel80]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, ttv))
fileout.close()



fileout = open('veusz_P20_nottv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins nottv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv in \
        zip(pl_names[nottv_sel20],pl_orbper[nottv_sel20],pl_masse[nottv_sel20],pl_masseerr1[nottv_sel20],pl_masseerr2[nottv_sel20],pl_rade[nottv_sel20],pl_radeerr1[nottv_sel20],pl_radeerr2[nottv_sel20],pl_dens[nottv_sel20],pl_denserr1[nottv_sel20],pl_denserr2[nottv_sel20],insol[nottv_sel20],pl_ttvflag[nottv_sel20]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv))
fileout.close()

fileout = open('veusz_P40_nottv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins nottv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv in \
        zip(pl_names[nottv_sel40],pl_orbper[nottv_sel40],pl_masse[nottv_sel40],pl_masseerr1[nottv_sel40],pl_masseerr2[nottv_sel40],pl_rade[nottv_sel40],pl_radeerr1[nottv_sel40],pl_radeerr2[nottv_sel40],pl_dens[nottv_sel40],pl_denserr1[nottv_sel40],pl_denserr2[nottv_sel40],insol[nottv_sel40],pl_ttvflag[nottv_sel40]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv))
fileout.close()

fileout = open('veusz_P60_nottv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins nottv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv in \
        zip(pl_names[nottv_sel60],pl_orbper[nottv_sel60],pl_masse[nottv_sel60],pl_masseerr1[nottv_sel60],pl_masseerr2[nottv_sel60],pl_rade[nottv_sel60],pl_radeerr1[nottv_sel60],pl_radeerr2[nottv_sel60],pl_dens[nottv_sel60],pl_denserr1[nottv_sel60],pl_denserr2[nottv_sel60],insol[nottv_sel60],pl_ttvflag[nottv_sel60]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv))
fileout.close()

fileout = open('veusz_P80_nottv.dat','w')
fileout.write('descriptor name p m,+,- r,+,- d,+,- ins nottv\n')
for nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv in \
        zip(pl_names[nottv_sel80],pl_orbper[nottv_sel80],pl_masse[nottv_sel80],pl_masseerr1[nottv_sel80],pl_masseerr2[nottv_sel80],pl_rade[nottv_sel80],pl_radeerr1[nottv_sel80],pl_radeerr2[nottv_sel80],pl_dens[nottv_sel80],pl_denserr1[nottv_sel80],pl_denserr2[nottv_sel80],insol[nottv_sel80],pl_ttvflag[nottv_sel80]):
    fileout.write('"{0:s}" {1:15f} {2:15f} {3:15f} {4:15f} {5:15f} {6:15f} {7:15f} {8:15f} {9:15f} {10:15f} {11:15f} {12:5.1f} \n'.format(nm, p, m, mp, mm, r, rp, rm, d, dp, dm, ins, nottv))
fileout.close()
