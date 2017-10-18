import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas
csfont = {'fontname':'Times New Roman'}

data = pandas.read_csv('../Exoplanets_eu/exoplanet.eu_catalog.csv')

sel = (data['mass'] > 0.0) & (data['orbital_period'] > 0.0) & (data['star_mass']>0.0) & (data['star_radius']>0.) & \
    (np.abs(data['mass_error_min']/data['mass'])<1.0 ) & (np.abs(data['mass_error_max']/data['mass'])<1.0 )

G_grav = 6.67398e-11
M_sun = 1.98892e30
M_jup = 1.89813e27
M_ratio = M_sun / M_jup
Mu_sun = 132712440018.9
seconds_in_day = 86400
AU_km = 1.4960 * 10 ** 8



pl_names     = data['# name'][sel]
pl_orbper    = data['orbital_period'][sel]
st_rad       = data['star_radius'][sel]
st_teff      = data['star_teff'][sel]
st_mass      = data['star_mass'][sel]
pl_masse     = data['mass'][sel]*317.83
pl_masseerr1 = np.abs(data['mass_error_max'][sel])*317.83
pl_masseerr2 = np.abs(data['mass_error_min'][sel])*317.83
pl_rade      = data['radius'][sel]*11.209
pl_radeerr1  = np.abs(data['radius_error_max'][sel])*11.209
pl_radeerr2  = np.abs(data['radius_error_min'][sel])*11.209
mass_detection_type  = data['mass_detection_type'][sel]

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


color_map = plt.get_cmap('plasma')

log_insol = np.log10(insol)

insol_01 = (log_insol) / (3.5)

#alphas = [1 - np.abs(err/mass) for mass, err in zip(pl_masse, pl_masseerr_avg)]

alphas = 1 - np.abs(pl_masseerr_avg/pl_masse)
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

for pos, ypt, m_err1, m_err2, r_err1, r_err2, color, alpha_orig, pl_name in \
        zip(pl_masse, pl_rade, pl_masseerr1, pl_masseerr2, pl_radeerr1, pl_radeerr2, colors, alphas_original, pl_names):
    #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
    #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
    #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
    #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig)
    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig)

    ax1.scatter(pos, ypt, color='white', zorder=alpha_orig+0.0001)
    ax1.scatter(pos, ypt, color=color, zorder=alpha_orig+0.0002)
    #ax1.scatter(pos, ypt, color=color, zorder=2, edgecolor=color_noalpha)

    if pos < xlims[0] or pos > xlims[1] or ypt < ylims[0] or ypt > ylims[1]: continue
    ax1.annotate(pl_name, xy=(pos, ypt),
                 xytext=(-2, 1), textcoords='offset points', ha='right', va='bottom',
                 color=color, zorder=alpha_orig)

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

plt.savefig('ExoEU_MR.pdf', bbox_inches='tight', dpi=300)
