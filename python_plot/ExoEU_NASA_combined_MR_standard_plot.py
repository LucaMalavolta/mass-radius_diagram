from plot_routines import *


def make_plot(properties_dict):

    import cmocean
    #color_map = plt.get_cmap('plasma')
    #color_map = plt.get_cmap('nipy_spectral_r')


    parameters_dict = {
        'xticks': [0.5, 1, 2, 5, 10, 20, 30],
        'xlims': [0.4, 30],
        'ylims': [0.8, 4.3],
        'alpha_upper_limit': 0.8,
        'alpha_lower_limit': 0.2,
        'alpha_upper_value': 1.0,
        'alpha_lower_value': 0.2,
        'insol_min': 1.0,
        'insol_max': 3000.0,
        'colorbar_xvector': [1, 3, 10, 30, 100, 300, 1000],
        #'colorbar_xvector': [1, 3, 10, 30, 100, 300, 1000, 3000],
        'add_overplot': 0.0,
        'color_map': cmocean.cm.thermal,
        'font_label': 24,
        'font_planet_name': 10,
        'font_tracks':18,
        'font_Solar_name':18
    }

    if properties_dict['define_alpha_density']:
        parameters_dict['alpha_upper_limit'] = 0.6
        parameters_dict['alpha_lower_limit'] = 0.3
        parameters_dict['alpha_upper_value'] = 0.6
        parameters_dict['alpha_lower_value'] = 0.3




    start_up(parameters_dict)


    csfont = {'fontname':'Times New Roman'}
    matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman']})

    create_flags(properties_dict)
    constants()
    short_name_subtitutions()

    data_combined = combine_catalogues()
    perform_selection(data_combined)

    insolation_scale()
    define_alpha_colors()
    setup_plot(12,8)

    add_points_from_dataset()

    if not properties_dict['no_color_scale']:
        add_color_bar()
    plot_lizeng_tracks()




    prefix_output_name = 'ExoEU_NASA_MR'
    save_fig(prefix_output_name)


#properties_dict['define_plot_USPP'] = False

properties_dict = {}
properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = False
properties_dict['define_short_names'] = True
properties_dict['no_color_scale'] = True
make_plot(properties_dict)


properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = True
properties_dict['define_alpha_density'] = False
properties_dict['define_short_names'] = True
properties_dict['no_color_scale'] = True
make_plot(properties_dict)


# Same but with thick markers
properties_dict = {}
properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = False
properties_dict['define_short_names'] = True
properties_dict['no_color_scale'] = False
make_plot(properties_dict)


properties_dict['define_thick_markers'] = True
properties_dict['define_planet_names'] = True
properties_dict['define_alpha_density'] = False
properties_dict['define_short_names'] = True
properties_dict['no_color_scale'] = False
make_plot(properties_dict)
