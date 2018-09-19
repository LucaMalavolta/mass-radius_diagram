from plot_routines import *
import cmocean

properties_dict = {}
properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = False
properties_dict['define_plot_USPP'] = True
properties_dict['no_color_scale'] = True
properties_dict['mark_ttvs'] = False


#color_map = plt.get_cmap('plasma')
#color_map = plt.get_cmap('nipy_spectral_r')

"""
This has been modified to accomodate USPp
"""
parameters_dict = {
    'xticks': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
    #'xlims': [0.9, 13.],
    #'ylims': [0.9, 2.2],
    #'xlims': [0.4, 30],
    #'ylims': [0.8, 4.3],
    'xlims': [0.7, 14],
    'ylims': [0.9, 2.6],
    'alpha_upper_limit': 0.70,
    'alpha_upper_value': 0.70,
    'alpha_lower_limit': 0.70,
    'alpha_lower_value': 0.00,
    'insol_min': 1000.0,
    'insol_max': 6000.0,
    'colorbar_xvector': [1000, 2000, 3000, 4000, 5000, 6000],
    'add_overplot': 0.0,
    'color_map': cmocean.cm.thermal,
    'font_label': 24,
    'font_planet_name': 14,
    'font_tracks':22,
    'font_Solar_name':18
}

if properties_dict['define_alpha_density']:
    parameters_dict['alpha_upper_limit'] = 0.6
    parameters_dict['alpha_lower_limit'] = 0.6
    parameters_dict['alpha_upper_value'] = 0.6
    parameters_dict['alpha_lower_value'] = 0.0

properties_dict['skip_plot_USPP'] = True
properties_dict['no_color_scale'] = True
start_up(parameters_dict)


#csfont = {'fontname':'Times New Roman','size': 12}
#matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman'],'size': 14})

matplotlib.rcParams.update({'font.family': 'serif', 'font.serif':'Times New Roman'})
matplotlib.rcParams.update({'font.size': 18})

create_flags(properties_dict)

constants()
short_name_subtitutions()

data_combined = combine_catalogues()
perform_selection(data_combined)

define_alpha_colors()
setup_plot(11,12)

add_points_from_dataset()


input_NASA_format = '../NASA_data/ussp_NASA.csv'


properties_dict['skip_plot_USPP'] = False
properties_dict['no_color_scale'] = False
properties_dict['no_alpha_colors'] = True
properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names_usp'] = True
properties_dict['define_short_names'] = False

properties_dict['z_offset'] = 1000000.0
properties_dict['markersize'] = 10

create_flags(properties_dict)
input_dataset(input_NASA_format)

insolation_scale(linear_in=True)
define_alpha_colors()

add_points_from_dataset()
#add_color_bar(axes_list=[0.2, 0.55, 0.02, 0.35])

plot_lizeng_tracks_alternative()

add_solar_system()

prefix_output_name = 'ExoEU_NASA_USPp_MR'
save_fig(prefix_output_name)
