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

MR_plot = MR_Plot()

MR_plot.fp_foplus_spaces = '    ' #Manually increase the distance between the Fp_Foplus label and the colorbar tick labels

#Combination 1
MR_plot.font_label = 24
MR_plot.font_planet_name = 16
MR_plot.font_tracks =24
MR_plot.font_my_planet = 20
MR_plot.font_USP_name = 18
MR_plot.font_Solar_name =24
MR_plot.prefix_output_name = './plots/ExoEU_MR_standard_noplanets_169'

#Combination 2
#MR_plot.font_label = 14
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =14
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/large_ExoEU_MR_standard_fulton'

#Combination 3
#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/giant_ExoEU_MR_standard_fulton'

#Combination 4
MR_plot.plot_size = [16,9]
#MR_plot.prefix_output_name = './plots/lplot_ExoEU_MR_standard_fulton'


MR_plot.define_thick_markers = True
MR_plot.define_planet_names = False
MR_plot.define_alpha_density = False
MR_plot.define_short_names = True
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False

#MR_plot.exclude_planet_names.extend(['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d'])

MR_plot.xlims = [0.4, 22]
MR_plot.ylims = [0.8, 2.8]
MR_plot.xy_labels = [20.2, 2.80]

MR_plot.xticks = [0.5, 1, 2, 5, 10, 20]

#MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]
MR_plot.colorbar_axes_list=[0.10, 0.50, 0.03, 0.40]

MR_plot.tracks_on_top = True


#MR_plot.prefix_output_name = './plots/ExoEU_MR_standard_fulton'

MR_plot.add_lzeng_tracks = True
#MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']

#MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
#MR_plot.lzeng_plot_parameters['cold_h2_he']['y_pos'] = 2.8
#MR_plot.lzeng_plot_parameters['cold_h2_he']['rotation'] = 53.38

#MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5
#MR_plot.lzeng_plot_parameters['100_h2o']['y_pos'] = 2.71
#MR_plot.lzeng_plot_parameters['100_h2o']['rotation'] = 45.71

for key_name in MR_plot.lzeng_plot_parameters:
    MR_plot.lzeng_plot_parameters[key_name]['linestyle'] = '--'

MR_plot.add_elopez_tracks = False
#MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']
for key_name in MR_plot.elopez_plot_parameters:
    MR_plot.elopez_plot_parameters[key_name]['linestyle'] = '--'


MR_plot.fulton_gap = [1.7, 0.1]
MR_plot.add_fulton_gap = False
MR_plot.fulton_gap_shaded = True


MR_plot.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','max_coll_strip']


MR_plot.set_update_properties()
MR_plot.make_plot(exo_dataset)


MR_plot.save_figure()
