from subroutines.plotting_classes import *
from subroutines.density_classes_vertical import *
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
my_planets = Dataset_Input('./my_planets/my_planets.dat')


MR_plot = MR_densityM_map()


MR_plot.logR = False
MR_plot.logM = False
MR_plot.xlims = [0.01, 1.99]
MR_plot.ylims = [0.8, 2.9]
MR_plot.xticks = [0.5, 1, 2, 5, 10, 20, 30]
MR_plot.yticks = [0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0 ]


MR_plot.add_color_bar = False

MR_plot.colorbar_axes_list=[0.10, 0.52, 0.03, 0.40]


MR_plot.prefix_output_name = './plots/ExoEU_MR_density_M'

MR_plot.lzeng_plot_list = []



MR_plot.lzeng_plot_parameters['cold_h2_he']['x_pos'] = 0.90
#MR_plot.lzeng_plot_parameters['100_h2o']['x_pos'] = 13.5

MR_plot.fulton_gap = [1.7, 0.05]
MR_plot.add_fulton_gap = True


MR_plot.set_update_properties()
#MR_plot.make_plot_with_my_planets(exo_dataset, my_planets)
MR_plot.make_plot(exo_dataset)

MR_plot.save_figure()
