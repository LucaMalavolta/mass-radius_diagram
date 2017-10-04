import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas
csfont = {'fontname':'Times New Roman'}



data_eu = pandas.read_csv('../Exoplanets_eu/exoplanet.eu_catalog.csv')

data_nasa = np.genfromtxt(
    '../NASA_data/defaults_radec.csv',           # file name
    skip_header=1,          # lines to skip at the top
    skip_footer=0,          # lines to skip at the bottom
    delimiter=',',          # column delimiter
    dtype='float32',        # data type
    filling_values=0.00000000,       # fill missing values with 0
    #usecols = (0,2,3,5),    # columns to read
    names=['name','orbital_period','mass','mass_error_max','mass_error_min','radius','radius_error_max','radius_error_min','star_mass','star_radius','star_teff','pl_ttvflag','ra','dec'])     # column names

names_nasa = np.genfromtxt(
    '../NASA_data/defaults.csv',           # file name
    skip_header=1,          # lines to skip at the top
    skip_footer=0,          # lines to skip at the bottom
    delimiter=',',          # column delimiter
    dtype=str,        # data type
    filling_values=-1.000,       # fill missing values with 0
    usecols = (0))    # columns to read



n_planets = len(data_eu['# name'])

parameters_list = ['mass', 'mass_error_min', 'mass_error_max', \
                   'radius', 'radius_error_min', 'radius_error_max', \
                   'star_radius', 'star_teff', 'star_mass', 'orbital_period']

data_combined = {}
for key in parameters_list:
    data_combined[key] = np.zeros(n_planets, dtype=np.double) - 0.0001

factor_dict = {'mass':317.83, 'mass_error_min':317.83, 'mass_error_max':317.83, \
    'radius':11.209, 'radius_error_min':11.209, 'radius_error_max':11.209, \
    'star_radius':1.00000, 'star_teff':1.00000, 'star_mass':1.00000, 'orbital_period':1.00000 }


ii = 0
for name_i, name_val in enumerate(data_eu['# name']):
    index = np.where(names_nasa==name_val)
    ind = -1
    found = False

    if np.size(index) >0:
        ind = index[0]
    else:
        #dist = (data_eu['ra'][name_i] -  data_nasa['ra'])**2 + (data_eu['dec'][name_i] -  data_nasa['dec'])**2
        #cos(A) = sin(Decl.1)sin(Decl.2) + cos(Decl.1)cos(Decl.2)cos(RA.1 - RA.2) and thus, A = arccos(A)
        cos_A = np.sin(data_eu['dec'][name_i]/180.0*np.pi) * np.sin(data_nasa['dec']/180.0*np.pi) + \
            np.cos(data_eu['dec'][name_i]/180.0*np.pi) * np.cos(data_nasa['dec']/180.0*np.pi) * np.cos(data_eu['ra'][name_i]/180.0*np.pi - data_nasa['ra']/180.0*np.pi)
        A = np.arccos(cos_A)

        ind_sort = np.argsort(A)

        ind_where = np.where( np.abs(data_eu['orbital_period'][name_i] - data_nasa['orbital_period'][ind_sort[:8]]) < 0.1)[0]
        if np.size(ind_where)  > 0:
            ind = ind_sort[ind_where[0]]

    if ind > 0:

        for key in parameters_list:
            if np.abs(data_nasa[key][ind])>0.00001:
                data_combined[key][name_i] = np.abs(data_nasa[key][ind]).copy()
            elif data_eu[key][name_i]>0.00000000001:
                data_combined[key][name_i] = data_eu[key][name_i].copy() * factor_dict[key]
            #print data_nasa[key][ind], data_eu[key][name_i], data_combined[key][name_i]


        #"Attenzione!

        #print 'mass', data_eu['mass'][name_i]
        #if name_val == 'Kepler-223 b': quit()

G_grav = 6.67398e-11
M_sun = 1.98892e30
M_jup = 1.89813e27
M_ratio = M_sun / M_jup
Mu_sun = 132712440018.9
seconds_in_day = 86400
AU_km = 1.4960 * 10 ** 8

sel = (data_combined['mass'] > 0.2) & (data_combined['radius'] > 0.2) & (data_combined['orbital_period'] > 0.01) & \
    (data_combined['star_mass']>0.0) & (data_combined['star_radius']>0.) & (data_combined['star_teff']>0.) & \
    (data_combined['radius_error_min']/data_combined['radius']<1.0 ) & (data_combined['radius_error_max']/data_combined['radius']<1.0 ) & \
    (data_combined['mass_error_min']/data_combined['mass']<1.0 ) & (data_combined['mass_error_max']/data_combined['mass']<1.0 ) & \
    (data_combined['mass_error_min']>0.0) & (data_combined['mass_error_max']>0.0) & (data_combined['radius_error_min']>0.0) & (data_combined['radius_error_max']>0.0)


pl_names     = data_eu['# name'][sel]
pl_orbper    = data_combined['orbital_period'][sel]
st_rad       = data_combined['star_radius'][sel]
st_teff      = data_combined['star_teff'][sel]
st_mass      = data_combined['star_mass'][sel]
pl_mass     = data_combined['mass'][sel]
pl_mass_error_max = data_combined['mass_error_max'][sel]
pl_mass_error_min = data_combined['mass_error_min'][sel]
pl_radius      = data_combined['radius'][sel]
pl_radius_error_max  = data_combined['radius_error_max'][sel]
pl_radius_error_min  = data_combined['radius_error_min'][sel]
mass_detection_type  = data_eu['mass_detection_type'][sel]

a_smj_AU = np.power((Mu_sun * np.power(pl_orbper * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * st_mass, 1.00 / 3.00)

insol = st_rad**2 * (st_teff/5777.0)**4 / a_smj_AU**2

#for on, a, st1, st2 in zip(insol, a_smj_AU, st_rad, st_teff):
#    print on, a, st1, st2

pl_masserr_avg = (pl_mass_error_max + pl_mass_error_min)/2.0
pl_radiuserr_avg = (pl_radius_error_max + pl_radius_error_min)/2.0

pl_dens = pl_mass/pl_radius**3
pl_denserr1 =pl_dens * np.sqrt( (pl_mass_error_max/pl_mass)**2 + 9*(pl_radius_error_min/pl_radius)**2)
pl_denserr2 =pl_dens * np.sqrt( (pl_mass_error_min/pl_mass)**2 + 9*(pl_radius_error_max/pl_radius)**2)
pl_denserr_avg = pl_dens * np.sqrt( (pl_masserr_avg/pl_mass)**2 + 9*(pl_radiuserr_avg/pl_radius)**2)

perc_error = pl_masserr_avg/pl_mass * 100.

perc_error = pl_denserr_avg/pl_dens * 100.


color_map = plt.get_cmap('plasma')

log_insol = np.log10(insol)

insol_01 = (log_insol) / (3.5)

#alphas = [1 - np.abs(err/mass) for mass, err in zip(pl_mass, pl_masserr_avg)]

alphas = 1 - np.abs(pl_masserr_avg/pl_mass)
alphas_original = alphas.copy()
alphas[alphas > 0.8] = 1.0
alphas[alphas > 1.0] = 1.0
alphas[alphas < 0.1] = 0.1

colors = [color_map(i, alpha=a) for i, a in zip(insol_01, np.power(alphas,2))]
colors_noalpha = [color_map(i, alpha=1.0) for i in insol_01]


fig, ax1 = plt.subplots(1, figsize=(12, 12))

xlims = [1, 30]
ylims = [0.8, 4.2]
#index_sort = numpy.argsort(alphas)

for pos, ypt, m_err1, m_err2, r_err1, r_err2, color, alpha_orig, pl_name, m_detect_type in \
        zip(pl_mass, pl_radius, pl_mass_error_max, pl_mass_error_min, pl_radius_error_max, pl_radius_error_min,
        colors, alphas_original, pl_names, mass_detection_type):
    #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
    #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
    #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
    #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig)
    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig)

    #ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig)
    #ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig)

    ax1.scatter(pos, ypt, color='white', zorder=alpha_orig+0.0001)
    if m_detect_type == 'TTV':
        ax1.scatter(pos, ypt, color=color, zorder=alpha_orig+0.0002, marker="s")
    else:
        ax1.scatter(pos, ypt, color=color, zorder=alpha_orig+0.0002)

    #ax1.scatter(pos, ypt, color=color, zorder=2, edgecolor=color_noalpha)

    #if pos*0.98 < xlims[0] or pos*1.02 > xlims[1] or ypt*0.98 < ylims[0] or ypt > ylims[1]: continue
    #ax1.annotate(pl_name, xy=(pos, ypt),
    #             xytext=(-2, 1), textcoords='offset points', ha='right', va='bottom',
    #             color=color, zorder=alpha_orig, annotation_clip=True)


ax1.set_xlim(xlims)
ax1.set_ylim(ylims)
ax1.set_xscale('log')
ax1.set_xticks([1, 2, 5, 10, 20, 30])
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))

ax1.set_ylabel('Radius [R$_{\oplus}$]')
ax1.set_xlabel('Mass [M$_{\oplus}$]')

plt.savefig('ExoEU_NASA_MR.pdf', bbox_inches='tight', dpi=300)
