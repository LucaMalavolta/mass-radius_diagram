from subroutines.plotting_classes import *
from subroutines.dataset_classes import *
import pickle

#properties_dict['define_plot_USPP'] = False

exo_dataset = Dataset_ExoplanetEU()

try:
    my_planets = Dataset_Input('./my_planets/my_planets.dat')
except:
    my_planets = None

try:
    other_planets = Dataset_Input('./my_planets/other_planets.dat')
except:
    other_planets = None

other_planets = None
MR_plot = MR_Plot()

MR_plot.plot_size = [8.75,7]

MR_plot.exclude_planet_names.extend(['Kepler-51 c'])


MR_plot.fp_foplus_spaces = '    ' #Manually increase the distance between the Fp_Foplus label and the colorbar tick labels

MR_plot.define_thick_markers = True
MR_plot.define_planet_names = False
MR_plot.define_alpha_density = False
MR_plot.define_short_names = True
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False
MR_plot.mark_flag_ttvs = False

#MR_plot.logM = False
#MR_plot.xticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

MR_plot.add_lzeng_tracks = False
MR_plot.add_jupiter_densities = True

MR_plot.logM = False

MR_plot.xticks = [0.1, 0.2, 0.5, 1, 1.5]
MR_plot.xlims = [0.001, 1.5]
MR_plot.ylims = [0.00, 3.0]
MR_plot.xy_labels = [1.48, 3.00]

#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14


#MR_plot.font_label = 18
#MR_plot.font_planet_name = 12
#MR_plot.font_tracks =18
#MR_plot.font_my_planet = 18
#MR_plot.font_USP_name = 18
#MR_plot.font_Solar_name =16

font_reducer = 2

MR_plot.font_label = 22 -font_reducer - 4
MR_plot.font_planet_name = 12 -font_reducer
MR_plot.font_tracks = 20 - font_reducer
MR_plot.font_my_planet = 20 - font_reducer
MR_plot.font_USP_name = 18 - font_reducer
MR_plot.font_Solar_name =16 - font_reducer

MR_plot.add_solar_system_flag = False

MR_plot.colorbar_axes_list=[0.15, 0.55, 0.03, 0.35]
MR_plot.fp_foplus_spaces = '  ' #Manually increase the distance between the Fp_Foplus label and the colorbar tick labels


MR_plot.prefix_output_name = './plots/ExoEU_MR_lowdensity_jupiter_units'

MR_plot.jupiter_units = True
MR_plot.define_alpha_density = True

MR_plot.insol_max = 9000.0
MR_plot.colorbar_xvector = [1, 3, 10, 30, 100, 300, 1000, 3000, 9000]


MR_plot.set_update_properties()
#MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
#MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)

#MR_plot.make_plot(exo_dataset)


if my_planets is None:
    MR_plot.make_plot(exo_dataset)
else:
    if other_planets is None:
        MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    else:
        MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)


MR_plot.save_figure()
