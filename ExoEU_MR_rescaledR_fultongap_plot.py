from subroutines.plotting_classes import *
from subroutines.rescaledR_classes import *
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
'comb2': {'thick_markers': False, 'planet_names':True},
'comb3': {'thick_markers': True, 'planet_names':True},
'comb4': {'thick_markers': False, 'planet_names':False},
}

for key_name, key_comb in combinations.iteritems():

    exo_dataset = Dataset_ExoplanetEU()
    my_planets = Dataset_Input('./my_planets/my_planets.dat')

    MR_plot = MR_rescaledR_plot()


    MR_plot.define_thick_markers = key_comb['thick_markers']
    MR_plot.define_planet_names = key_comb['planet_names']
    MR_plot.define_alpha_density = False
    MR_plot.define_short_names = True
    MR_plot.no_color_scale = False
    MR_plot.mark_ttvs = False

    MR_plot.logR = True
    MR_plot.xlims = [0.4, 30]
    MR_plot.ylims = [0.5, 2.4]
    MR_plot.xticks = [0.5, 1, 2, 5, 10, 20, 30]
    MR_plot.yticks = [0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 ]

    MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]


    MR_plot.prefix_output_name = './plots/ExoEU_MR_rescaledR_fulton'

    MR_plot.lzeng_plot_list = ['100_fe','rocky','100_h2o']


    MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
    #MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5

    MR_plot.fulton_gap = [1.7, 0.05]
    MR_plot.add_fulton_gap = True


    MR_plot.set_update_properties()
    #MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
    MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)

    MR_plot.save_figure()
