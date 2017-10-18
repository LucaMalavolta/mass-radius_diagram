import numpy as np
import math
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas
from matplotlib import rc
import matplotlib as mpl


def text_slope_match_line(ax, xdata, ydata ,x_pos):
    global rotated_labels

    # find the slope

    ind = np.argmin(np.abs(xdata-x_pos))

    x1 = xdata[ind-1]
    x2 = xdata[ind+1]
    y1 = ydata[ind-1]
    y2 = ydata[ind+1]

    y_pos = y1 + (x_pos-x1)*(y2-y1)/(x2-x1)

    p1 = np.array((x1, y1))
    p2 = np.array((x2, y2))

    # get the line's data transform
    #ax = ax.get_axes()

    sp1 = ax.transData.transform_point(p1)
    sp2 = ax.transData.transform_point(p2)

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])


    return math.degrees(math.atan(rise/run)), y_pos


def transform_log_colorscale(val, vmin, vmax):
    return(np.log10(val)-np.log10(vmin))/(np.log10(vmax)-np.log10(vmin))


def combine_catalogues():

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

    data_combined['name'] = data_eu['# name']
    data_combined['mass_detection_type'] = data_eu['mass_detection_type']

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

            try:
                A = np.arccos(cos_A)
            except:
                print A
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

    return data_combined

def make_plot(properties_dict):


    alpha_upper_limit = 0.8
    alpha_lower_limit = 0.2

    define_thick_markers = False
    define_planet_names = False
    define_alpha_density = False
    define_plot_USPP = False
    define_plot_USPP_name = False

    name_thick_markers = ''
    name_planet_names = ''
    name_alpha_density = ''
    name_plot_USPP = ''
    name_plot_USPP_name = ''

    for key_name, key_val in properties_dict.iteritems():
        if key_name == 'define_thick_markers':
            define_thick_markers = properties_dict['define_thick_markers']
        if key_name == 'define_planet_names':
            define_planet_names = properties_dict['define_planet_names']
        if key_name == 'define_alpha_density':
            define_alpha_density = properties_dict['define_alpha_density']
            if properties_dict['define_alpha_density']:
                alpha_upper_limit = 0.6
                alpha_lower_limit = 0.3
        if key_name == 'define_plot_USPP':
            define_plot_USPP = properties_dict['define_plot_USPP']
        if key_name == 'define_plot_USPP_name':
            define_plot_USPP_name = properties_dict['define_plot_USPP_name']
            if properties_dict['define_plot_USPP_name']:
                define_plot_USPP = properties_dict['define_plot_USPP_name']

    if define_thick_markers:
        name_thick_markers = '_ThickMarkers'

    if define_planet_names:
        name_planet_names = '_PNames'

    if define_alpha_density:
        name_alpha_density = '_rho'

    if define_plot_USPP:
        name_plot_USPP = '_USPPlanets'

    if define_plot_USPP_name:
        name_plot_USPP_name = '_name'


    csfont = {'fontname':'Times New Roman'}

    data_combined = combine_catalogues()

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


    pl_names     = data_combined['name'][sel]
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
    mass_detection_type  = data_combined['mass_detection_type'][sel]

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

    if define_alpha_density:
        perc_error = pl_denserr_avg/pl_dens
    else:
        perc_error = pl_masserr_avg/pl_mass



    insol_min = 1.0
    insol_max = 3000.0


    insol_01 = transform_log_colorscale(insol, insol_min, insol_max)
    insol_01 = [0.0 if i < 0.0 else i for i in insol_01]
    insol_01 = [1.00 if i > 1.00 else i for i in insol_01]


    alphas = 1 - np.abs(perc_error)
    alphas_original = alphas.copy()
    alphas *= alphas
    alphas[alphas > alpha_upper_limit] = 1.0
    alphas[alphas < alpha_lower_limit] = alpha_lower_limit

    #color_map = plt.get_cmap('plasma')
    #color_map = plt.get_cmap('nipy_spectral_r')
    import cmocean
    color_map = cmocean.cm.thermal

    #colors = [color_map(i, alpha=a) for i, a in zip(insol_01, np.power(alphas,2))]
    colors = [color_map(i, alpha=a) for i, a in zip(insol_01, alphas)]
    colors_noalpha = [color_map(i, alpha=1.0) for i in insol_01]



    xlims = [0.4, 30]
    ylims = [0.8, 4.3]
    #index_sort = numpy.argsort(alphas)


    #rc('text', usetex=True, fontsize=12)
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}

    matplotlib.rc('font', **font)
    fig, ax1 = plt.subplots(1, figsize=(12, 12))

    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.set_xscale('log')
    ax1.set_xticks([0.5, 1, 2, 5, 10, 20, 30])
    #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))
    #ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))

    ax1.set_ylabel('Radius [R$_{\oplus}$]')
    ax1.set_xlabel('Mass [M$_{\oplus}$]')
    #plt.show()


    x_vector = [1, 3, 10, 30, 100, 300, 1000, 3000]
    ax2 = fig.add_axes([0.17, 0.55, 0.02, 0.30])

    x_logvector = transform_log_colorscale(x_vector, insol_min, insol_max)


    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=color_map,
                                    orientation='vertical',
                                    ticks=x_logvector)
    ax2.set_title('F$_{\mathrm{p}}$/F$_{\oplus}$   ')
    cb1.ax.set_yticklabels(x_vector)



    for pos, ypt, m_err1, m_err2, r_err1, r_err2, color, alpha_orig, pl_name, m_detect_type, pl_period, insol01_val in \
            zip(pl_mass, pl_radius, pl_mass_error_max, pl_mass_error_min, pl_radius_error_max, pl_radius_error_min,
            colors, alphas_original, pl_names, mass_detection_type, pl_orbper, insol_01):
        #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
        #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
        #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
        #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

        if m_detect_type == 'TTV':
            marker_point = "s"
        else:
            marker_point = "o"

        if define_plot_USPP:
            if pl_period < 1.0000:
                marker_point = "H"

        if define_thick_markers:
            ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig, marker=marker_point, mfc='white')
            ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig, marker=marker_point, mfc='white')
            if insol01_val> 0.9 :
                color_mod = color_map(0.9, alpha=color[3])
            else:
                color_mod = color
            ax1.plot(pos, ypt, color=color, zorder=alpha_orig+0.3, marker=marker_point, mfc=color, mec=color_mod)

        else:
            ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig, marker=marker_point, mfc='white', markersize=0)
            ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig, marker=marker_point, mfc='white', markersize=0)
            ax1.plot(pos, ypt, color='white', mfc='white', zorder=alpha_orig+0.1, marker=marker_point, mec="none")
            ax1.plot(pos, ypt, color=color, zorder=alpha_orig+0.3, marker=marker_point, mfc=color, mec="none")



        if define_planet_names or (define_plot_USPP_name and pl_period < 1.0000):

            if pos*0.98 < xlims[0] or pos*1.02 > xlims[1] or ypt*0.98 < ylims[0] or ypt > ylims[1]: continue
            ax1.annotate(pl_name, xy=(pos, ypt),
                         xytext=(-3, 0), textcoords='offset points', ha='right', va='bottom',
                         color=color, fontsize=10,  zorder=alpha_orig+0.5, annotation_clip=True)

    # adding the composition tracks
    LZeng_tracks = np.genfromtxt('../LZeng_tracks/Zeng2016_mrtable2.txt', skip_header=1, \
         names = ['Mearth','100_fe','75_fe','50_fe','30_fe','25_fe','20_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip'] )

    lizeng_plots = {
    '100_fe': {'x_pos':28, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'label':'100% Fe'},
    '75_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'label':'75% Fe'},
    '50_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'label':'50% Fe'},
    #'30_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'label':'30% Fe'},
    '25_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'label':'25% Fe'},
    #'20_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'label':'20% Fe'},
    'rocky':  {'x_pos':28, 'cmap':'Greens', 'color':0.50, 'alpha':0.8, 'label':'Rocky'},
    '25_h2o':     {'x_pos':28, 'cmap':'winter', 'color':0.75, 'alpha':0.8, 'label':'25% H$_{2}$O'},
    '50_h2o':     {'x_pos':28, 'cmap':'winter', 'color':0.50, 'alpha':0.8, 'label':'50% H$_{2}$O'},
    '100_h2o':    {'x_pos':28, 'cmap':'winter', 'color':0.00, 'alpha':0.8, 'label':'100% H$_{2}$O'},
    'cold_h2_he': {'x_pos':3.5, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'label':'Cold H$_{2}$/He'},
    'max_coll_strip': {'x_pos':28, 'cmap':'binary', 'color':1.00, 'alpha':0.2, 'label':'', 'fill_below':True}
    }

    x_pos = 28

    for key_name, key_val in lizeng_plots.iteritems():
        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)
        line = ax1.plot(LZeng_tracks['Mearth'],LZeng_tracks[key_name],color=color, zorder=0)
        rotation, y_pos = text_slope_match_line(ax1, LZeng_tracks['Mearth'],LZeng_tracks[key_name], key_val['x_pos'])
        ax1.annotate(key_val['label'], xy=(key_val['x_pos'], y_pos), \
                         xytext=(0, 5), textcoords='offset points', ha='right', va='bottom', \
                         color=color_noalpha, zorder=1000, rotation=rotation, rotation_mode="anchor", fontsize=12, weight='bold')
        if 'fill_below' in key_val:
            ax1.fill_between(LZeng_tracks['Mearth'],0, LZeng_tracks[key_name], color=color, alpha=0.15)

    output_name = 'ExoEU_NASA_MR' + name_thick_markers \
                                + name_planet_names \
                                + name_alpha_density \
                                + name_plot_USPP \
                                + name_plot_USPP_name \
                                + '.pdf'

    print 'OUTPUT plot:  ', output_name
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    plt.close()





properties_dict = {}
properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = True
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)

properties_dict = {}
properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = True
properties_dict['define_plot_USPP_name'] = True
make_plot(properties_dict)

properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = False
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)

properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = True
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = False
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)


# Same but with thick markers
properties_dict = {}
properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = True
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)

properties_dict = {}
properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = True
properties_dict['define_plot_USPP_name'] = True
make_plot(properties_dict)

properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = False
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)

properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = True
properties_dict['define_alpha_density'] = True
properties_dict['define_plot_USPP'] = False
properties_dict['define_plot_USPP_name'] = False
make_plot(properties_dict)
