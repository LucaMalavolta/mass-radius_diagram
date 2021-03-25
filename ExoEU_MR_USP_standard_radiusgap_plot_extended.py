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


#Ultra-short period-planets addition:
MR_plot.skip_plot_USPP = True
MR_plot.define_planet_names_USPP = False

MR_plot.no_alpha_colors = True

MR_plot.MR_limit = 0.3
MR_plot.only_precise_MR = True

#Combination 1
MR_plot.font_label = 24
MR_plot.font_planet_name = 12
MR_plot.font_tracks = 20
MR_plot.font_my_planet = 20
MR_plot.font_USP_name = 20
MR_plot.font_Solar_name =20
MR_plot.prefix_output_name = './plots/ExoEU_MR_USP_standard_radius_extended'

#Combination 2
#MR_plot.font_label = 14
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =14
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/large_ExoEU_MR_standard_radius'

#Combination 3
#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/giant_ExoEU_MR_standard_radius'

#Combination 4
#MR_plot.plot_size = [9.6,8]
#MR_plot.prefix_output_name = './plots/lplot_ExoEU_MR_standard_radius'

MR_plot.plot_size = [10,10]


MR_plot.exclude_planet_names.extend(['NONE'])

MR_plot.define_thick_markers = True
MR_plot.define_planet_names = False
MR_plot.define_alpha_density = False
MR_plot.define_short_names = True
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False

#MR_plot.exclude_planet_names.extend(['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d'])

MR_plot.xlims = [0.4, 23]
MR_plot.ylims = [0.8, 3.2]
MR_plot.xy_labels = [20.7, 2.80]

MR_plot.xticks = [0.5, 1, 2, 5, 10, 20]

#MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]
MR_plot.colorbar_axes_list=[0.17, 0.50, 0.03, 0.40]
MR_plot.fp_foplus_spaces = '     '

MR_plot.tracks_on_top = True


#MR_plot.prefix_output_name = './plots/ExoEU_MR_standard_radius'

MR_plot.add_lzeng_tracks = True
#MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']

#MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 2.1
MR_plot.lzeng_plot_parameters['cold_h2_he']['y_pos'] = 3.2
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


MR_plot.radius_gap = [1.85, 0.15]
MR_plot.add_radius_gap = True
MR_plot.radius_gap_shaded = True


MR_plot.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']
MR_plot.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']
MR_plot.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']
MR_plot.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']


MR_plot.set_update_properties()
if my_planets is None:
    MR_plot.make_plot(exo_dataset)
else:
    if other_planets is None:
        MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    else:
        MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)


#Uncomment this snippet to include additional tracks from private files
plot_parameters = MR_plot.default_plot_parameters.copy()
plot_parameters['cmap']='winter'
plot_parameters['color']=0.60
plot_parameters['linestyle'] = '--'
plot_parameters['label']='+1% H$_{2}$'
plot_parameters['use_box'] = False
plot_parameters['x_pos'] = 3.5
plot_parameters['y_pos'] = 2.15
MR_plot.add_track_from_files('LiZeng_private_tracks/interpolated_halfh2o01h300K1mbar.dat', plot_parameters)


MR_plot.add_USP_planets(exo_dataset)


MR_plot.save_figure()
