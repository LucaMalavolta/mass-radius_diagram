from subroutines.plotting_classes import *
from subroutines.dataset_classes import *
import cPickle as pickle

#properties_dict['define_plot_USPP'] = False

combinations ={
'comb1': {'thick_markers': True, 'planet_names':False}
#'comb2': {'thick_markers': True, 'planet_names':False},
#'comb3': {'thick_markers': True, 'planet_names':False},
#'comb4': {'thick_markers': True, 'planet_names':False}
}

for key_name, key_comb in combinations.iteritems():

    exo_dataset = Dataset_ExoplanetEU()
    my_planets = Dataset_Input('./my_planets/my_planets.dat')

    MR_plot = MR_Plot()

    if key_name == 'comb1':
        MR_plot.font_label = 18
        MR_plot.font_planet_name = 12
        MR_plot.font_tracks =18
        MR_plot.font_my_planet = 18
        MR_plot.font_USP_name = 16
        MR_plot.font_Solar_name =16
        MR_plot.skip_plot_USPP = True
        MR_plot.markersize_USP = 12
        MR_plot.prefix_output_name = './plots/mega_ExoEU_MR_USP_standard_fulton'
    if key_name == 'comb2':
        MR_plot.font_label = 14
        MR_plot.font_planet_name = 10
        MR_plot.font_tracks =14
        MR_plot.font_my_planet = 16
        MR_plot.font_USP_name = 14
        MR_plot.font_Solar_name =14
        MR_plot.skip_plot_USPP = True
        MR_plot.markersize_USP = 12
        MR_plot.prefix_output_name = './plots/large_ExoEU_MR_USP_standard_fulton'
    if key_name == 'comb3':
        MR_plot.font_label = 16
        MR_plot.font_planet_name = 10
        MR_plot.font_tracks =16
        MR_plot.font_my_planet = 16
        MR_plot.font_USP_name = 14
        MR_plot.font_Solar_name =14
        MR_plot.skip_plot_USPP = True
        MR_plot.markersize_USP = 12
        MR_plot.prefix_output_name = './plots/giant_ExoEU_MR_USP_standard_fulton'
    if key_name == 'comb4':
        MR_plot.plot_size = [9.6,8]
        MR_plot.skip_plot_USPP = True
        MR_plot.prefix_output_name = './plots/lplot_ExoEU_MR_USP_standard_fulton'

    #MR_plot.plot_size = [9.6,8]


    MR_plot.define_thick_markers = key_comb['thick_markers']
    MR_plot.define_planet_names = key_comb['planet_names']
    MR_plot.define_alpha_density = False
    MR_plot.define_planet_names_USPP = True
    MR_plot.define_short_names = True
    MR_plot.no_color_scale = False
    MR_plot.mark_ttvs = False

    MR_plot.xlims = [0.4, 20]
    MR_plot.ylims = [0.8, 2.8]
    MR_plot.xticks = [0.5, 1, 2, 5, 10, 20]

    MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]



    #MR_plot.prefix_output_name = './plots/ExoEU_MR_standard_fulton'

    MR_plot.add_lzeng_tracks = True
    #MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']
    MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
    MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5
    for key_name in MR_plot.lzeng_plot_parameters:
        MR_plot.lzeng_plot_parameters[key_name]['linestyle'] = '--'



    MR_plot.fulton_gap = [1.7, 0.1]
    MR_plot.add_fulton_gap = True
    MR_plot.fulton_gap_shaded = True



    MR_plot.set_update_properties()
    #MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    MR_plot.make_plot(exo_dataset)
    MR_plot.add_USP_planets(exo_dataset)

    MR_plot.save_figure()

def other_plots():
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
