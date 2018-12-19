from subroutines.plotting_classes import *
from subroutines.dataset_classes import *



#properties_dict['define_plot_USPP'] = False

exo_dataset = Dataset_ExoplanetEU()
my_planets = Dataset_Input('./my_planets/my_planets.dat')
other_planets = Dataset_Input('./my_planets/other_planets.dat')
MR_plot = MR_Plot()


MR_plot.define_thick_markers = True
MR_plot.define_planet_names = True
MR_plot.define_alpha_density = False
MR_plot.define_short_names = False
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False
MR_plot.mark_flag_ttvs = True

MR_plot.xlims = [1, 92]
MR_plot.ylims = [1.0, 12.]
MR_plot.xticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]

MR_plot.add_lzeng_tracks = False
MR_plot.logM = False

MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]


MR_plot.prefix_output_name = './plots/ExoEU_NASA_MR_lowdensity'

MR_plot.set_update_properties()
#MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)

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
