from subroutines.plotting_classes import *
from subroutines.dataset_classes import *
import pickle

exo_dataset = Dataset_ExoplanetEU()
my_planets = Dataset_Input('./my_planets/my_planets.dat')

try:
    my_planets = Dataset_Input('./my_planets/my_planets.dat')
except:
    my_planets = None

try:
    other_planets = Dataset_Input('./my_planets/other_planets.dat')
except:
    other_planets = None

MR_plot = MR_Plot()

#MR_plot.fp_foplus_spaces = '    ' #Manually increase the distance between the Fp_Foplus label and the colorbar tick labels


#MR_plot.font_label = 18
#MR_plot.font_planet_name = 12
#MR_plot.font_tracks =18
#MR_plot.font_my_planet = 18
#MR_plot.font_USP_name = 16
#MR_plot.font_Solar_name =16
#MR_plot.skip_plot_USPP = True
#MR_plot.markersize_USP = 12
MR_plot.skip_plot_USPP = True
MR_plot.markersize_USP = 12
MR_plot.font_label = 24
MR_plot.font_planet_name = 12
MR_plot.font_tracks =16
MR_plot.font_my_planet = 20
MR_plot.font_USP_name = 16
MR_plot.font_Solar_name =18

MR_plot.prefix_output_name = './plots/ExoEU_MR_USPplanets_radius'

## Different combinations of size

# Combination 1
#MR_plot.font_label = 14
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =14
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.skip_plot_USPP = True
#MR_plot.markersize_USP = 12

# Combination 2
#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.skip_plot_USPP = True
#MR_plot.markersize_USP = 12

# Combination 3
#MR_plot.plot_size = [9.6,8]
#MR_plot.skip_plot_USPP = True
#MR_plot.prefix_output_name = './plots/lplot_ExoEU_MR_USP_standard_radius'

MR_plot.define_thick_markers = True
MR_plot.define_planet_names = False
MR_plot.define_alpha_density = False
MR_plot.define_planet_names_USPP = True
MR_plot.define_short_names = True
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False

# Use this to exclude planets from the archive
#MR_plot.exclude_planet_names.extend(['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d'])

MR_plot.xlims = [0.4, 20]
MR_plot.ylims = [0.8, 2.8]
MR_plot.xticks = [0.5, 1, 2, 5, 10, 20]
#MR_plot.xy_labels = [20.7, 2.80]

MR_plot.colorbar_axes_list=[0.13, 0.50, 0.03, 0.40]

#MR_plot.prefix_output_name = './plots/ExoEU_MR_standard_radius'

MR_plot.add_lzeng_tracks = True
#MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']
MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5
for key_name in MR_plot.lzeng_plot_parameters:
    MR_plot.lzeng_plot_parameters[key_name]['linestyle'] = '--'


MR_plot.radius_gap = [1.7, 0.1]
MR_plot.add_radius_gap = True
MR_plot.radius_gap_shaded = True


MR_plot.set_update_properties()
#MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
print(my_planets)
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
plot_parameters['use_box'] = True
plot_parameters['x_pos'] = 19.0
MR_plot.add_track_from_files('LiZeng_private_tracks/interpolated_halfh2o01h300K1mbar.dat', plot_parameters)



MR_plot.add_USP_planets(exo_dataset)

MR_plot.save_figure()
