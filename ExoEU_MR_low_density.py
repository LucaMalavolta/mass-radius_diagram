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
MR_plot.mark_ttvs = True
MR_plot.mark_flag_ttvs = True

MR_plot.xlims = [8, 92]
MR_plot.ylims = [6.0, 12.]
MR_plot.xticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]

MR_plot.add_lzeng_tracks = False
MR_plot.logM = False

MR_plot.plot_size = [12,10]


#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14


MR_plot.font_label = 18
MR_plot.font_planet_name = 12
MR_plot.font_tracks =18
MR_plot.font_my_planet = 18
MR_plot.font_USP_name = 18
MR_plot.font_Solar_name =16

MR_plot.font_label = 22
MR_plot.font_planet_name = 14
MR_plot.font_tracks =22
MR_plot.font_my_planet = 22
MR_plot.font_USP_name = 22
MR_plot.font_Solar_name =16


MR_plot.colorbar_axes_list=[0.12, 0.55, 0.03, 0.35]


MR_plot.prefix_output_name = './plots/ExoEU_MR_lowdensity'

MR_plot.set_update_properties()
#MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)

MR_plot.save_figure()
