from subroutines.plotting_classes import *
from subroutines.rescaledR_classes import *
from subroutines.rescaledM_classes import *
from subroutines.dataset_classes import *
import cPickle as pickle

def create_datasets():
    exo_dataset = Dataset_ExoplanetEU()
    my_planets = Dataset_Input('./my_planets/my_planets.dat')

    pickle.dump(exo_dataset, open("exo_dataset.p", "wb"))
    pickle.dump(my_planets, open("my_planets.p", "wb"))

def load_datasets():
    exo_dataset = pickle.load(open("exo_dataset.p", "rb"))
    my_planets = pickle.load(open("my_planets.p", "rb"))
    return exo_dataset, my_planets


exo_dataset = Dataset_ExoplanetEU()

try:
    my_planets = Dataset_Input('./my_planets/my_planets.dat')
except:
    my_planets = None

try:
    other_planets = Dataset_Input('./my_planets/other_planets.dat')
except:
    other_planets = None


MR_plot = MR_rescaledM_plot()

MR_plot.fp_foplus_spaces = '    ' #Manually increase the distance between the Fp_Foplus label and the colorbar tick labels

MR_plot.colorbar_ticks_position = 'left'

MR_plot.font_label = 24
MR_plot.font_planet_name = 12
MR_plot.font_tracks =18
MR_plot.font_my_planet = 20
MR_plot.font_USP_name = 18
MR_plot.font_Solar_name =18
MR_plot.prefix_output_name = './plots/ExoEU_MR_rescaledM_fulton'

#combination 1
#MR_plot.font_label = 14
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =14
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/large_ExoEU_MR_rescaledM_fulton'

#combination 2
#MR_plot.font_label = 16
#MR_plot.font_planet_name = 10
#MR_plot.font_tracks =16
#MR_plot.font_my_planet = 16
#MR_plot.font_USP_name = 14
#MR_plot.font_Solar_name =14
#MR_plot.prefix_output_name = './plots/giant_ExoEU_MR_rescaledM_fulton'

#combination 3
#MR_plot.plot_size = [9.6,8]
#MR_plot.prefix_output_name = './plots/plot_ExoEU_MR_rescaledM_fulton'


MR_plot.define_thick_markers = True
MR_plot.define_planet_names = True
MR_plot.define_alpha_density = False
MR_plot.define_short_names = True
MR_plot.no_color_scale = False
MR_plot.mark_ttvs = False

MR_plot.exclude_planet_names.extend(['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d'])

MR_plot.logR = False
MR_plot.logM = False
MR_plot.xlims = [0.01, 2.19]
MR_plot.ylims = [0.8, 2.9]

MR_plot.xy_labels = [0.8, 2.80]

MR_plot.xticks = [0.5, 1, 2, 5, 10, 20, 30]
MR_plot.yticks = [0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 ]

#MR_plot.colorbar_axes_list=[0.10, 0.58, 0.03, 0.40]
#MR_plot.colorbar_axes_list=[0.11, 0.65, 0.03, 0.40]
MR_plot.colorbar_axes_list=[0.87, 0.50, 0.03, 0.40]

MR_plot.add_lzeng_tracks = True

MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']
MR_plot.lzeng_plot_list = ['75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o']
#MR_plot.lzeng_plot_list = []


for key_name in MR_plot.lzeng_plot_parameters:

    MR_plot.lzeng_plot_parameters[key_name]['linestyle'] = '--'
    #MR_plot.lzeng_plot_parameters[key_name]['x_pos'] =

#MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90


#MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5

MR_plot.fulton_gap = [1.7, 0.1]
MR_plot.add_fulton_gap = True
MR_plot.fulton_gap_shaded = True
MR_plot.fulton_label_position = ['left', 'bottom']

MR_plot.set_update_properties()
if my_planets is None:
    MR_plot.make_plot(exo_dataset)
else:
    if other_planets is None:
        MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    else:
        MR_plot.make_plot_with_mine_and_other_planets(exo_dataset, my_planets, other_planets)


plot_parameters = MR_plot.default_plot_parameters.copy()
plot_parameters['cmap']='winter'
plot_parameters['color']=0.60
plot_parameters['linestyle'] = '--'
plot_parameters['label']='+1% H$_{2}$'
plot_parameters['use_box'] = True
plot_parameters['x_pos'] = 0.01
plot_parameters['y_pos'] = 2.40
plot_parameters['rotation'] = -80.0
MR_plot.add_track_from_files('LiZeng_private_tracks/interpolated_halfh2o01h300K1mbar.dat', plot_parameters)



MR_plot.save_figure()
