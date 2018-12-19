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



combinations ={
'comb1': {'thick_markers': True, 'planet_names':False},
'comb2': {'thick_markers': False, 'planet_names':False},
'comb3': {'thick_markers': False, 'planet_names':True},
'comb4': {'thick_markers': True, 'planet_names':True},
}

combinations ={
'comb1': {'thick_markers': True, 'planet_names':True},
'comb2': {'thick_markers': True, 'planet_names':True},
'comb3': {'thick_markers': True, 'planet_names':True},
'comb4': {'thick_markers': True, 'planet_names':True}
}

for key_name, key_comb in combinations.iteritems():

    exo_dataset = Dataset_ExoplanetEU()
    my_planets = Dataset_Input('./my_planets/my_planets.dat')

    MR_plot = MR_rescaledM_plot()

    if key_name == 'comb1':
        MR_plot.font_label = 18
        MR_plot.font_planet_name = 12
        MR_plot.font_tracks =18
        MR_plot.font_my_planet = 18
        MR_plot.font_USP_name = 18
        MR_plot.font_Solar_name =18
        MR_plot.prefix_output_name = './plots/mega_ExoEU_MR_rescaledM_fulton'
    if key_name == 'comb2':
        MR_plot.font_label = 14
        MR_plot.font_planet_name = 10
        MR_plot.font_tracks =14
        MR_plot.font_my_planet = 16
        MR_plot.font_USP_name = 14
        MR_plot.font_Solar_name =14
        MR_plot.prefix_output_name = './plots/large_ExoEU_MR_rescaledM_fulton'
    if key_name == 'comb3':
        MR_plot.font_label = 16
        MR_plot.font_planet_name = 10
        MR_plot.font_tracks =16
        MR_plot.font_my_planet = 16
        MR_plot.font_USP_name = 14
        MR_plot.font_Solar_name =14
        MR_plot.prefix_output_name = './plots/giant_ExoEU_MR_rescaledM_fulton'
    if key_name == 'comb4':
        MR_plot.plot_size = [9.6,8]
        MR_plot.prefix_output_name = './plots/plot_ExoEU_MR_rescaledM_fulton'

    MR_plot.colorbar_ticks_position = 'left'

    MR_plot.define_thick_markers = key_comb['thick_markers']
    MR_plot.define_planet_names = key_comb['planet_names']
    MR_plot.define_alpha_density = False
    MR_plot.define_short_names = True
    MR_plot.no_color_scale = False
    MR_plot.mark_ttvs = False

    MR_plot.logR = False
    MR_plot.logM = False
    MR_plot.xlims = [0.01, 2.19]
    MR_plot.ylims = [0.8, 2.9]
    MR_plot.xticks = [0.5, 1, 2, 5, 10, 20, 30]
    MR_plot.yticks = [0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 ]

    MR_plot.colorbar_axes_list=[0.10, 0.58, 0.03, 0.40]

    MR_plot.add_lzeng_tracks = True

    MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']
    MR_plot.lzeng_plot_list = ['75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o']
    #MR_plot.lzeng_plot_list = []


    MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
    for key_name in MR_plot.lzeng_plot_parameters:
        MR_plot.lzeng_plot_parameters[key_name]['linestyle'] = '--'

    #MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5

    MR_plot.fulton_gap = [1.7, 0.1]
    MR_plot.add_fulton_gap = True
    MR_plot.fulton_gap_shaded = True
    MR_plot.fulton_label_position = ['left', 'bottom']

    MR_plot.set_update_properties()
    #MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)

    MR_plot.save_figure()
