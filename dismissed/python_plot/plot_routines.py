import numpy as np
import math
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas
from matplotlib import rc
import matplotlib as mpl

def start_up(parameter_dict):
    global insol_min, insol_max, color_map, color_map_backup, \
    xlims, ylims, xticks, colorbar_xvector, add_overplot, \
    alpha_upper_limit, alpha_lower_limit, alpha_upper_value, alpha_lower_value, \
    font_label, font_planet_name, font_my_planet, font_tracks, \
    font_USP_name, font_Solar_name


    xticks = parameter_dict['xticks']
    xlims = parameter_dict['xlims']
    ylims = parameter_dict['ylims']

    alpha_upper_limit = parameter_dict['alpha_upper_limit']
    alpha_lower_limit = parameter_dict['alpha_lower_limit']
    alpha_upper_value = parameter_dict['alpha_upper_value']
    alpha_lower_value = parameter_dict['alpha_lower_value']
    insol_min = parameter_dict['insol_min']
    insol_max = parameter_dict['insol_max']
    colorbar_xvector = parameter_dict['colorbar_xvector']
    add_overplot = parameter_dict['add_overplot']
    color_map = parameter_dict['color_map']
    color_map_backup = parameter_dict['color_map']

    font_label = 12
    font_planet_name = 10
    font_my_planet = 14
    font_tracks = 12
    font_USP_name = 12
    font_Solar_name = 12

    for key_name, key_val in parameter_dict.iteritems():
        if key_name == 'font_label':
            font_label = parameter_dict['font_label']
        if key_name == 'font_planet_name':
            font_planet_name = parameter_dict['font_planet_name']
        if key_name == 'font_my_planet':
            font_my_planet = parameter_dict['font_my_planet']
        if key_name == 'font_tracks':
            font_tracks = parameter_dict['font_tracks']
        if key_name == 'font_USP_name':
            font_USP_name = parameter_dict['font_USP_name']
        if key_name == 'font_Solar_name':
            font_Solar_name = parameter_dict['font_Solar_name']

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

def transform_linear_colorscale(val, vmin, vmax):
    return(np.asarray(val)-np.asarray(vmin))/(np.asarray(vmax)-np.asarray(vmin))


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
        '../NASA_data/defaults_radec.csv',           # file name
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
                print
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

def constants():
    global G_grav, M_sun, M_jup, M_ratio, Mu_sun, seconds_in_day, AU_km
    G_grav = 6.67398e-11
    M_sun = 1.98892e30
    M_jup = 1.89813e27
    M_ratio = M_sun / M_jup
    Mu_sun = 132712440018.9
    seconds_in_day = 86400
    AU_km = 1.4960 * 10 ** 8

def create_flags(properties_dict):
    global define_thick_markers, define_planet_names, define_planet_names_usp, define_alpha_density, \
        skip_plot_USPP, define_short_names, no_color_scale, no_alpha_colors, \
        name_thick_markers, name_planet_names, name_alpha_density,markersize, \
        name_plot_USPP, z_offset, mark_ttvs

    define_thick_markers = False
    define_planet_names = False
    define_planet_names_usp = False
    define_alpha_density = False
    skip_plot_USPP = False
    define_short_names = False
    no_color_scale = False
    no_alpha_colors = False
    mark_ttvs = True
    name_thick_markers = ''
    name_planet_names = ''
    name_alpha_density = ''
    name_plot_USPP = ''
    markersize = 6
    z_offset = 0.0


    for key_name, key_val in properties_dict.iteritems():
        if key_name == 'define_thick_markers':
            define_thick_markers = properties_dict['define_thick_markers']
        if key_name == 'define_planet_names':
            define_planet_names = properties_dict['define_planet_names']
        if key_name == 'define_planet_names_usp':
            define_planet_names_usp = properties_dict['define_planet_names_usp']
        if key_name == 'define_alpha_density':
            define_alpha_density = properties_dict['define_alpha_density']
        if key_name == 'skip_plot_USPP':
            skip_plot_USPP = properties_dict['skip_plot_USPP']
        if key_name == 'define_short_names':
            define_short_names = properties_dict['define_short_names']
        if key_name == 'no_color_scale':
            no_color_scale = properties_dict['no_color_scale']
        if key_name == 'no_alpha_colors':
            no_alpha_colors = properties_dict['no_alpha_colors']
        if key_name == 'z_offset':
            z_offset = properties_dict['z_offset']
        if key_name == 'markersize':
            markersize = properties_dict['markersize']
        if key_name == 'mark_ttvs':
            mark_ttvs = properties_dict['mark_ttvs']



    if no_color_scale:
        color_map = mpl.cm.binary_r
    else:
        color_map = color_map_backup
    matplotlib.rcParams.update({'font.size': font_label})


    if define_thick_markers:
        name_thick_markers = '_ThickMarkers'

    if define_planet_names:
        name_planet_names = '_PNames'

    if define_alpha_density:
        name_alpha_density = '_rho'

def perform_selection(data_combined):
    global pl_names, pl_orbper, st_rad, st_teff, st_mass, \
    pl_mass, pl_mass_error_max, pl_mass_error_min, pl_radius, \
    pl_radius_error_max, pl_radius_error_min, mass_detection_type, \
    a_smj_AU, insol, pl_masserr_avg, pl_radiuserr_avg, pl_dens, pl_denserr1, \
    pl_denserr2, pl_denserr_avg, perc_error, insol_01

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
    insol_01 = pl_orbper*0.0

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

    print ' DENSITY ALPHA FLAG: ', define_alpha_density
    if define_alpha_density:
        perc_error = pl_denserr_avg/pl_dens
    else:
        perc_error = pl_masserr_avg/pl_mass
        sel_tmp = (perc_error<0.20) & (pl_mass<20) & (pl_radius>0.5) & (pl_radius<2.6) & (pl_radiuserr_avg/pl_radius < 0.20)
        print '--> Number of super-Earths with masses better than 20pc: ', np.sum(sel_tmp)


def input_dataset(input_NASA_format):
    global pl_names, pl_orbper, st_rad, st_teff, st_mass, \
    pl_mass, pl_mass_error_max, pl_mass_error_min, pl_radius, \
    pl_radius_error_max, pl_radius_error_min, mass_detection_type, \
    a_smj_AU, insol, pl_masserr_avg, pl_radiuserr_avg, pl_dens, pl_denserr1, \
    pl_denserr2, pl_denserr_avg, perc_error, insol_01


    data_nasa = np.genfromtxt(
        input_NASA_format,           # file name
        skip_header=1,          # lines to skip at the top
        skip_footer=0,          # lines to skip at the bottom
        delimiter=',',          # column delimiter
        dtype='float32',        # data type
        filling_values=0.00000000,       # fill missing values with 0
        #usecols = (0,2,3,5),    # columns to read
        names=['name','orbital_period','mass','mass_error_max','mass_error_min','radius','radius_error_max','radius_error_min','star_mass','star_radius','star_teff','pl_ttvflag','ra','dec'])     # column names

    names_nasa = np.genfromtxt(
        input_NASA_format,           # file name
        skip_header=1,          # lines to skip at the top
        skip_footer=0,          # lines to skip at the bottom
        delimiter=',',          # column delimiter
        dtype=str,        # data type
        filling_values=-1.000,       # fill missing values with 0
        usecols = (0))    # columns to read

    pl_names     = names_nasa
    pl_orbper    = data_nasa['orbital_period']
    st_rad       = data_nasa['star_radius']
    st_teff      = data_nasa['star_teff']
    st_mass      = data_nasa['star_mass']
    pl_mass     = data_nasa['mass']
    pl_mass_error_max = data_nasa['mass_error_max']
    pl_mass_error_min = np.abs(data_nasa['mass_error_min'])
    pl_radius      = data_nasa['radius']
    pl_radius_error_max  = data_nasa['radius_error_max']
    pl_radius_error_min  = np.abs(data_nasa['radius_error_min'])
    mass_detection_type  = np.ones(np.size(pl_orbper))
    insol_01 = pl_orbper*0.0

    a_smj_AU = np.power((Mu_sun * np.power(pl_orbper * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * st_mass, 1.00 / 3.00)

    insol = st_rad**2 * (st_teff/5777.0)**4 / a_smj_AU**2
    #for pn, ii in zip(pl_names, insol):
    #    print pn, ii

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

def insolation_scale(linear_in=False):
    global insol_01, linear
    linear = linear_in
    if linear:
        insol_01 = transform_linear_colorscale(insol, insol_min, insol_max)
    else:
        insol_01 = transform_log_colorscale(insol, insol_min, insol_max)
    insol_01 = [0.0 if i < 0.0 else i for i in insol_01]
    insol_01 = [1.00 if i > 1.00 else i for i in insol_01]

    if no_color_scale:
        insol_01 =[0.0 if i < 0.0 else 0.0 for i in insol_01]

def define_alpha_colors():
    global alphas, alphas_original, colors
    alphas = 1 - np.abs(perc_error)
    alphas_original = alphas.copy()
    alphas *= alphas
    alphas[alphas_original > alpha_upper_limit] = alpha_upper_value**2
    alphas[alphas_original < alpha_lower_limit] = alpha_lower_value**2

    if no_alpha_colors:
        alphas *= 0
        alphas += 1.
    #colors = [color_map(i, alpha=a) for i, a in zip(insol_01, np.power(alphas,2))]
    colors = [color_map(i, alpha=a) for i, a in zip(insol_01, alphas)]
    colors_noalpha = [color_map(i, alpha=1.0) for i in insol_01]


def setup_plot(x=12, y=12):
    global fig, ax1
    fig, ax1 = plt.subplots(1, figsize=(x, y))
    fig.tight_layout()
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.set_xscale('log')
    ax1.set_xticks(xticks)
    #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))
    #ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))

    ax1.set_ylabel('Radius [R$_{\oplus}$]')
    ax1.set_xlabel('Mass [M$_{\oplus}$]')
    #plt.show()


def add_color_bar(axes_list=[0.17, 0.55, 0.02, 0.30]):
    ax2 = fig.add_axes(axes_list)

    if linear:
        x_logvector = transform_linear_colorscale(colorbar_xvector, insol_min, insol_max)
    else:
        x_logvector = transform_log_colorscale(colorbar_xvector, insol_min, insol_max)


    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=color_map,
                                    orientation='vertical',
                                    ticks=x_logvector)
    ax2.set_title('F$_{\mathrm{p}}$/F$_{\oplus}$   ')
    cb1.ax.set_yticklabels(colorbar_xvector)


def short_name_subtitutions():
    global name_subtitutions
    name_subtitutions = {
        'Kepler':'Kp',
        'WASP':'W',
        'HAT-P':'HP',
        'HATS':'HS',
        'Qatar': 'Q',
        'TRAPPIST': 'T',
        ' ':''
    }

def plot_lizeng_tracks(z_overcome=0):
    # adding the composition tracks
    LZeng_tracks = np.genfromtxt('../LZeng_tracks/Zeng2016_mrtable2.txt', skip_header=1, \
         names = ['Mearth','100_fe','75_fe','50_fe','30_fe','25_fe','20_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip'] )

    x_pos = 28
    lizeng_plots = {
    '100_fe': {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'label':'100% Fe'},
    '75_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'label':'75% Fe'},
    '50_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'label':'50% Fe'},
    #'30_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'label':'30% Fe'},
    '25_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'label':'25% Fe'},
    #'20_fe':  {'x_pos':28, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'label':'20% Fe'},
    'rocky':  {'x_pos':x_pos, 'cmap':'Greens', 'color':0.50, 'alpha':0.8, 'label':'Rocky'},
    '25_h2o':     {'x_pos':x_pos, 'cmap':'winter', 'color':0.75, 'alpha':0.8, 'label':'25% H$_{2}$O'},
    '50_h2o':     {'x_pos':x_pos, 'cmap':'winter', 'color':0.50, 'alpha':0.8, 'label':'50% H$_{2}$O'},
    '100_h2o':    {'x_pos':x_pos, 'cmap':'winter', 'color':0.00, 'alpha':0.8, 'label':'100% H$_{2}$O'},
    'cold_h2_he': {'x_pos':3.5, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'label':'Cold H$_{2}$/He'},
    'cold_h2_he': {'x_pos':1.0, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'label':'Cold H$_{2}$/He'},
    'max_coll_strip': {'x_pos':x_pos, 'cmap':'binary', 'color':1.00, 'alpha':0.2, 'label':'', 'fill_below':True}
    }

    for key_name, key_val in lizeng_plots.iteritems():
        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)
        line = ax1.plot(LZeng_tracks['Mearth'],LZeng_tracks[key_name],color=color, zorder=0+z_offset+z_overcome)
        rotation, y_pos = text_slope_match_line(ax1, LZeng_tracks['Mearth'],LZeng_tracks[key_name], key_val['x_pos'])
        ax1.annotate(key_val['label'], xy=(key_val['x_pos'], y_pos), \
                         xytext=(0, 5), textcoords='offset points', ha='right', va='bottom', \
                         color=color_noalpha, zorder=1000+z_offset+z_overcome, rotation=rotation, rotation_mode="anchor", fontsize=font_tracks, weight='bold')
        if 'fill_below' in key_val:
            ax1.fill_between(LZeng_tracks['Mearth'],0, LZeng_tracks[key_name], color=color, alpha=0.15)

def plot_lizeng_tracks_alternative(z_overcome=0):
    # adding the composition tracks
    LZeng_tracks = np.genfromtxt('../LZeng_tracks/Zeng2016_mrtable2.txt', skip_header=1, \
         names = ['Mearth','100_fe','75_fe','50_fe','30_fe','25_fe','20_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip'] )

    x_pos = 13.5
    lizeng_plots = {
    '100_fe': {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'label':'100% Fe'},
    #'75_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'label':'75% Fe'},
    '50_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'label':'50% Fe'},
    '30_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'label':'30% Fe'},
    #'25_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'label':'25% Fe', 'fill_between':True},
    #'20_fe':  {'x_pos':x_pos, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'label':'20% Fe'},
    'rocky':  {'x_pos':x_pos, 'cmap':'Greens', 'color':0.50, 'alpha':0.8, 'label':'100% MgSiO$_3$'},
    #'25_h2o':     {'x_pos':x_pos, 'cmap':'winter', 'color':0.75, 'alpha':0.8, 'label':'25% H$_{2}$O'},
    '50_h2o':     {'x_pos':x_pos, 'cmap':'winter', 'color':0.50, 'alpha':0.8, 'label':'50% H$_{2}$O'},
    #'100_h2o':    {'x_pos':x_pos, 'cmap':'winter', 'color':0.00, 'alpha':0.8, 'label':'100% H$_{2}$O'},
    #'cold_h2_he': {'x_pos':3.5, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'label':'Cold H$_{2}$/He'},
    'max_coll_strip': {'x_pos':x_pos, 'cmap':'binary', 'color':1.00, 'alpha':0.2, 'label':'', 'fill_below':True}
    }


    for key_name, key_val in lizeng_plots.iteritems():
        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)
        line = ax1.plot(LZeng_tracks['Mearth'],LZeng_tracks[key_name],color=color, zorder=0+z_offset+z_overcome)
        rotation, y_pos = text_slope_match_line(ax1, LZeng_tracks['Mearth'],LZeng_tracks[key_name], key_val['x_pos'])
        ax1.annotate(key_val['label'], xy=(key_val['x_pos'], y_pos), \
                         xytext=(0, -6), textcoords='offset points', ha='right', va='top', \
                         color=color_noalpha, zorder=1000+z_offset+z_overcome, rotation=rotation, rotation_mode="anchor",
                     fontsize=font_tracks, weight='bold')
        if 'fill_below' in key_val:
            ax1.fill_between(LZeng_tracks['Mearth'],0, LZeng_tracks[key_name], color=color, alpha=0.15)
        if 'fill_between' in key_val:
            ax1.fill_between(LZeng_tracks['Mearth'], LZeng_tracks[key_name]-0.01,LZeng_tracks[key_name]+0.01 ,
                             color=color, zorder=0 + z_offset+z_overcome, alpha=0.5)


def add_points_from_dataset():

    for pos, ypt, m_err1, m_err2, r_err1, r_err2, color, alpha_orig, pl_name, m_detect_type, pl_period, insol01_val in \
            zip(pl_mass, pl_radius, pl_mass_error_max, pl_mass_error_min, pl_radius_error_max, pl_radius_error_min,
            colors, alphas_original, pl_names, mass_detection_type, pl_orbper, insol_01):
        #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
        #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
        #plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
        #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

        #if pl_name == 'Kepler-19 b':
        #    print color,  pos, ypt, alpha_orig

        if color[-1] < 0.0001:
            continue

        if m_detect_type == 'TTV' and mark_ttvs:
            marker_point = "s"
        else:
            marker_point = "o"

        if skip_plot_USPP:
            if pl_period < 1.0000:
                continue

        if define_planet_names and define_short_names:
            for key_name, key_val in name_subtitutions.iteritems():
                pl_name_tmp = pl_name.replace(key_name, key_val)
                pl_name = pl_name_tmp

        if define_thick_markers:
            ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig+add_overplot+z_offset, marker=marker_point, mfc='white', markersize=markersize)
            ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig+add_overplot+z_offset, marker=marker_point, mfc='white', markersize=markersize)
            if insol01_val> 0.9 :
                color_mod = color_map(0.9, alpha=color[3])
            else:
                color_mod = color
            ax1.plot(pos, ypt, color=color, zorder=alpha_orig+add_overplot+0.3+z_offset, marker=marker_point, mfc=color, mec=color_mod, markersize=markersize)

        else:
            ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig+add_overplot+z_offset, marker=marker_point, mfc='white', markersize=0)
            ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig+add_overplot+z_offset, marker=marker_point, mfc='white', markersize=0)
            ax1.plot(pos, ypt, color='white', mfc='white', zorder=alpha_orig+add_overplot+0.1+z_offset, marker=marker_point, mec="none", markersize=markersize)
            ax1.plot(pos, ypt, color=color, zorder=alpha_orig+add_overplot+0.3+z_offset, marker=marker_point, mfc=color, mec="none", markersize=markersize)

        if define_planet_names:
            if pos*0.98 < xlims[0] or pos*1.02 > xlims[1] or ypt*0.98 < ylims[0] or ypt > ylims[1]: continue
            ax1.annotate(pl_name, xy=(pos, ypt),
                         xytext=(-5, 2), textcoords='offset points', ha='right', va='bottom',
                         color=color, fontsize=font_planet_name,  zorder=alpha_orig+add_overplot+0.5+z_offset, annotation_clip=True)

        if define_planet_names_usp:
            if pos*0.98 < xlims[0] or pos*1.02 > xlims[1] or ypt*0.98 < ylims[0] or ypt > ylims[1]: continue
            bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor=color, pad=0.1)
            bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='red', pad=0.1)
            if pl_name=='K2-141 b':
                ax1.annotate(pl_name, xy=(pos, ypt),
                             xytext=(4, -4), textcoords='offset points', ha='left', va='top',
                             color=color, fontsize=font_my_planet, zorder=alpha_orig + add_overplot + 0.5 + z_offset,
                             annotation_clip=True, bbox=bbox_props_mine)

            elif pl_name == 'HD 3167 b G17':
                ax1.annotate(pl_name, xy=(pos, ypt),
                             xytext=(4, -4), textcoords='offset points', ha='left', va='top',
                             color=color, fontsize=font_USP_name, zorder=alpha_orig + add_overplot/2. + 0.5 + z_offset,
                            annotation_clip=True, bbox=bbox_props)

            elif pl_name == 'HD 3167 b C17' or pl_name =='55 Cnc e':
                ax1.annotate(pl_name, xy=(pos, ypt),
                             xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom',
                             color=color, fontsize=font_USP_name, zorder=alpha_orig + add_overplot/2. + 0.5 + z_offset,
                             annotation_clip=True, bbox=bbox_props)

            elif pl_name=='K2-131 b' or pl_name=='Kepler-10 b':
                ax1.annotate(pl_name, xy=(pos, ypt),
                             xytext=(-4, -4), textcoords='offset points', ha='right', va='top',
                             color=color, fontsize=font_USP_name, zorder=alpha_orig + add_overplot/2. + 0.5 + z_offset,
                             annotation_clip=True, bbox=bbox_props)

            else:
                ax1.annotate(pl_name, xy=(pos, ypt),
                         xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
                         color=color, fontsize=font_USP_name,  zorder=alpha_orig+add_overplot/2.+0.5+z_offset, annotation_clip=True, bbox=bbox_props)

def add_solar_system():
    bbox_props = dict(boxstyle="square", fc="w", alpha=0.9, edgecolor='b', pad=0.1)

    ax1.plot([0.815, 1.00],[0.949,1.00],'ob', markersize=markersize+4, marker='*', zorder= 10000+ z_offset)
    ax1.annotate('Earth', xy=(1.0, 1.0),
                 xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                 color='b', fontsize=font_Solar_name, zorder= 10000+ z_offset,
                 annotation_clip=True, bbox=bbox_props)
    ax1.annotate('Venus', xy=(0.815, 0.949),
                 xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                 color='b', fontsize=font_Solar_name, zorder= 10000+ z_offset,
                 annotation_clip=True, bbox=bbox_props)

def save_fig(prefix_output_name):

    output_name = prefix_output_name + name_thick_markers \
                                + name_planet_names \
                                + name_alpha_density \
                                + name_plot_USPP \
                                + '.pdf'
    print 'OUTPUT plot:  ', output_name
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
