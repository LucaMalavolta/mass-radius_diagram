from subroutines.constants import *

def transform_log_colorscale(val, vmin, vmax):
    return(np.log10(val)-np.log10(vmin))/(np.log10(vmax)-np.log10(vmin))

def transform_linear_colorscale(val, vmin, vmax):
    return(np.asarray(val)-np.asarray(vmin))/(np.asarray(vmax)-np.asarray(vmin))



class MR_Plot():

    def __init__(self):
        self.xticks = [0.5, 1, 2, 5, 10, 20, 30]
        self.yticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4,]
        self.xlims = [0.4, 30]
        self.ylims = [0.8, 4.3]
        self.xy_labels = [19.0, 4.2]

        self.define_alpha_density = False
        self.alpha_upper_limit = 0.8
        self.alpha_lower_limit = 0.1
        self.alpha_upper_value = 1.0
        self.alpha_lower_value = 0.2
        self.default_alpha_density = True
        self.insol_min = 1.0
        self.insol_max = 6000.0
        self.colorbar_xvector = [1, 3, 10, 30, 100, 300, 1000, 3000, 6000]
        self.add_overplot = 0.0
        #self.color_map = cmocean.cm.thermal
        self.color_map = matplotlib.cm.plasma
        self.font_label = 12
        self.font_planet_name = 10
        self.font_tracks =12
        self.font_my_planet = 14
        self.font_USP_name = 12
        self.font_Solar_name =12
        self.define_thick_markers = False
        self.define_planet_names = False
        self.define_planet_names_USPP = False
        self.skip_plot_USPP = False
        self.define_short_names = False
        self.no_color_scale = False
        self.no_alpha_colors = False
        self.mark_ttvs = True
        self.mark_flag_ttvs = False

        self.add_solar_system_flag = True
        self.add_color_bar = True

        self.name_thick_markers = ''
        self.name_planet_names = ''
        self.name_alpha_density = ''
        self.name_plot_USPP = ''

        self.exclude_planet_names = ['HAT-P-47 b']

        self.markersize = 6
        self.markersize_USP = 8
        self.z_offset = 0.0
        self.linear_insolation_scale = False

        self.logM = True
        self.logR = False
        self.add_lzeng_tracks = True
        self.add_elopez_tracks = False
        self.add_jupiter_densities = False

        self.jupiter_densities_list = [1.0, 0.50, 0.25, 0.10, 0.05, 0.030]

        self.plot_size = [12,10]


        self.prefix_output_name = ''

        self.marker_point = "o"
        self.marker_point_ttv = "s"

        self.jupiter_units = False
        self.e2j_mass = 1.
        self.e2j_radius = 1.

        self.name_subtitutions = {
            'Kepler':'Kp',
            'WASP':'W',
            'HAT-P':'HP',
            'HATS':'HS',
            'Qatar': 'Q',
            'TRAPPIST': 'T',
            ' ':''
        }

        self.default_plot_parameters = {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'linestyle':'-', 'label':'100% Fe'}

        self.tracks_on_top = False

        self.lzeng_tracks = np.genfromtxt('./LZeng_tracks/Zeng2016_mrtable2.txt', skip_header=1, \
         names = ['Mearth','100_fe','75_fe','50_fe','30_fe','25_fe','20_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip'] )

        self.lzeng_plot_list = ['100_fe','75_fe','50_fe','25_fe','rocky','25_h2o','50_h2o','100_h2o','cold_h2_he','max_coll_strip']
        self.lzeng_plot_parameters = {
            '100_fe':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.00, 'alpha':0.8, 'linestyle':'-', 'label':'100% Fe'},
            '75_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.25, 'alpha':0.8, 'linestyle':'-', 'label':'75% Fe'},
            '50_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'50% Fe'},
            '30_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.70, 'alpha':0.8, 'linestyle':'-', 'label':'30% Fe'},
            '25_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.75, 'alpha':0.8, 'linestyle':'-', 'label':'25% Fe'},
            '20_fe':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'gist_heat', 'color':0.80, 'alpha':0.8, 'linestyle':'-', 'label':'20% Fe'},
            'rocky':          {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'Greens',    'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'Rocky'},
            '25_h2o':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.75, 'alpha':0.8, 'linestyle':'-', 'label':'25% H$_{2}$O'},
            '50_h2o':         {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.50, 'alpha':0.8, 'linestyle':'-', 'label':'50% H$_{2}$O'},
            '100_h2o':        {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':True , 'cmap':'winter', 'color':0.00, 'alpha':0.8, 'linestyle':'-', 'label':'100% H$_{2}$O'},
            'cold_h2_he':     {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'winter', 'color':0.90, 'alpha':0.8, 'linestyle':'-', 'label':'Cold H$_{2}$/He'},
            'max_coll_strip': {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'binary', 'color':1.00, 'alpha':0.2, 'linestyle':'-', 'label':'', 'fill_below':True}
        }

        self.elopez_tracks = np.genfromtxt('./ELopez_tracks/ELopez_tracks.dat', skip_header=1, \
         names = ['Mearth', '1_HHe', '10_HHe', 'Earth_composition', 'MaxIron', 'PureRock', 'Purewater'] )

        self.elopez_plot_list = ['1_HHe','10_HHe']
        self.elopez_plot_parameters = {
            '1_HHe' : {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'winter', 'color':0.10, 'alpha':0.8, 'linestyle':'-', 'label':'1% H$_{2}$/He'},
            '10_HHe': {'x_pos':None, 'y_pos': None, 'rotation':None, 'use_box':False, 'cmap':'winter', 'color':0.20, 'alpha':0.8, 'linestyle':'-', 'label':'10% H$_{2}$/He'},
        }

        self.radius_gap_on_top = False
        self.add_radius_gap = False
        self.radius_gap_parameters = {'x_pos':None, 'y_pos': None, 'rotation':None, 'cmap':'Blues', 'color':0.70, 'alpha':0.2, 'label':'Radius gap'}
        self.radius_gap = [1.7, 0.2]
        self.radius_gap_shaded = True
        self.radius_label_position = ['bottom', 'left']

        self.font_my_planet = 12

        self.colorbar_axes_list=[0.10, 0.65, 0.03, 0.30]
        self.colorbar_ticks_position = 'right'

        self.fp_foplus_spaces = '   '

        csfont = {'fontname':'Times New Roman'}
        matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman']})

    def set_update_properties(self):

        if self.define_alpha_density and self.default_alpha_density:
            self.alpha_upper_limit = 0.8
            self.alpha_lower_limit = 0.3
            self.alpha_upper_value = 1.0
            self.alpha_lower_value = 0.3

        if self.no_color_scale:
            self.color_map = mpl.cm.binary_r

        matplotlib.rcParams.update({'font.size': self.font_label})

        if self.define_thick_markers and (self.name_thick_markers is ''):
            self.name_thick_markers = '_ThickMarkers'

        if self.define_planet_names and (self.name_planet_names is ''):
            self.name_planet_names = '_PNames'

        if self.define_planet_names_USPP and (self.name_plot_USPP is ''):
            self.name_plot_USPP = '_PNames'

        if self.define_alpha_density and (self.name_alpha_density is ''):
            self.name_alpha_density = '_rho'

        if self.jupiter_units:
            self.e2j_mass = 1./M_jup_to_ear
            self.e2j_radius = 1./R_jup_to_ear

    def make_plot(self, dataset):

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)

        self.setup_plot()

        if self.add_color_bar:
            self.plot_color_bar()

        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()

        if self.add_elopez_tracks:
            self.plot_elopez_tracks()

        if self.add_jupiter_densities:
            self.plot_jupiter_densities()

        if self.add_radius_gap:
            self.plot_radius_gap()
        self.add_points_from_dataset(dataset)
        if self.add_solar_system_flag:
            self.add_solar_system()

    def make_plot_with_my_planets(self, dataset, my_planets):

        self.exclude_planet_names.extend(my_planets.pl_names)

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)
        self.insolation_scale(my_planets)
        self.define_alpha_colors(my_planets)

        self.setup_plot()

        if self.add_color_bar:
            self.plot_color_bar()
        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()

        if self.add_elopez_tracks:
            self.plot_elopez_tracks()

        if self.add_jupiter_densities:
            self.plot_jupiter_densities()

        if self.add_radius_gap:
            self.plot_radius_gap()

        self.add_points_from_dataset(dataset)
        self.add_my_planets(my_planets)

        if self.add_solar_system_flag:
            self.add_solar_system()


    def make_plot_with_mine_and_other_planets(self, dataset, my_planets, other_planets):

        self.exclude_planet_names.extend(my_planets.pl_names)
        self.exclude_planet_names.extend(other_planets.pl_names)


        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)

        self.insolation_scale(my_planets)
        self.define_alpha_colors(my_planets)

        self.insolation_scale(other_planets)
        self.define_alpha_colors(other_planets)

        self.setup_plot()

        if self.add_color_bar:
            self.plot_color_bar()

        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()

        if self.add_elopez_tracks:
            self.plot_elopez_tracks()

        if self.add_jupiter_densities:
            self.plot_jupiter_densities()

        if self.add_radius_gap:
            self.plot_radius_gap()

        self.add_points_from_dataset(dataset)
        self.add_my_planets(my_planets)
        self.add_other_planets(other_planets)
        if self.add_solar_system_flag:
            self.add_solar_system()

    def insolation_scale(self, dataset):

        if self.linear_insolation_scale:
            dataset.insol_01 = transform_linear_colorscale(dataset.insol, self.insol_min, self.insol_max)
        else:
            dataset.insol_01 = transform_log_colorscale(dataset.insol, self.insol_min, self.insol_max)
        dataset.insol_01 = [0.0 if i < 0.0 else i for i in dataset.insol_01]
        dataset.insol_01 = [1.00 if i > 1.00 else i for i in dataset.insol_01]

        if self.no_color_scale:
            dataset.insol_01 =[0.0 if i < 0.0 else 0.0 for i in dataset.insol_01]


    def define_alpha_colors(self, dataset):

        if self.define_alpha_density:
            dataset.alphas = 1 - np.abs(dataset.perc_error_density)
        else:
            dataset.alphas = 1 - np.abs(dataset.perc_error_mass)

        dataset.alphas_original = dataset.alphas.copy()
        dataset.alphas *= dataset.alphas
        dataset.alphas[dataset.alphas_original > self.alpha_upper_limit] = self.alpha_upper_value**2
        dataset.alphas[dataset.alphas_original < self.alpha_lower_limit] = self.alpha_lower_value**2

        if self.no_alpha_colors:
            dataset.alphas *= 0
            dataset.alphas += 1.

        #colors = [self.color_map(i, alpha=a) for i, a in zip(dataset.insol_01, np.power(dataset.alphas,2))]
        #colors_noalpha = [self.color_map(i, alpha=1.0) for i in dataset.insol_01]
        dataset.colors = [self.color_map(i, alpha=a) for i, a in zip(dataset.insol_01, dataset.alphas)]


    def setup_plot(self):

        self.xrange = self.xlims[1]-self.xlims[0]
        self.yrange = self.ylims[1]-self.ylims[0]


        self.fig, self.ax1 = plt.subplots(1, figsize=(self.plot_size[0], self.plot_size[1]))
        self.fig.tight_layout()
        self.ax1.set_xlim(self.xlims)
        self.ax1.set_ylim(self.ylims)

        if self.logM:
            self.ax1.set_xscale('log')
            self.ax1.set_xticks(self.xticks)
            self.ax1.minorticks_off()
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            self.ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))
        if self.logR:
            self.ax1.set_yscale('log')
            self.ax1.set_yticks(self.yticks)
            self.ax1.minorticks_off()
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
            self.ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(x)))

        #self.ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        #self.ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
        if self.jupiter_units:
            self.ax1.set_ylabel('Radius [R$_\mathrm{J}$]')
            self.ax1.set_xlabel('Mass [M$_\mathrm{J}$]')
        else:
            self.ax1.set_ylabel('Radius [R$_{\oplus}$]')
            self.ax1.set_xlabel('Mass [M$_{\oplus}$]')
        #plt.show()

        self.compute_radius_gap()


    def compute_radius_gap(self):
        self.radius_gap_x = np.arange(self.xlims[0]-self.xlims[0]/2., self.xlims[1]+0.1, 0.01)
        self.radius_gap_y = np.ones(len(self.radius_gap_x))*self.radius_gap[0]

    def plot_radius_gap(self):

        if  self.radius_gap_on_top:
            radius_z_order = self.z_offset + 1000.0
        else:
            radius_z_order = self.z_offset

        #self.ax1.fill_between(self.radius_gap_x, self.radius_gap_y-self.radius_gap[1], self.radius_gap_y+self.radius_gap[1], alpha=0.3)
        xytext = [0, 0]

        key_val = self.radius_gap_parameters #shortcut

        if 'top' in self.radius_label_position:
            va='top'
            xytext[1] = 3
            y_axis = self.radius_gap_y+self.radius_gap[1]

        if 'bottom' in self.radius_label_position:
            va='bottom'
            xytext[1] = -6
            y_axis = self.radius_gap_y-self.radius_gap[1]

        if 'left' in self.radius_label_position:
            ha='left'
            xytext[0] = 3
            x_pos = self.xlims[0]

        if 'right' in self.radius_label_position:
            ha='right'
            xytext[0] = -3
            x_pos = self.xlims[1]

        if key_val['x_pos']:
            x_pos = key_val['x_pos']

        if key_val['y_pos']:
            y_pos = key_val['y_pos']
        else:
            x_pos, y_pos = self.interpolate_line_value(self.radius_gap_x, y_axis, x_pos=x_pos)

        if key_val['rotation']:
            rotation = key_val['rotation']
        else:
            rotation = self.text_slope_match_line(self.ax1, self.radius_gap_x, y_axis, x_pos)

        color_map = plt.get_cmap(key_val['cmap'])
        color = color_map(key_val['color'], alpha=key_val['alpha'])
        color_noalpha = color_map(key_val['color'], alpha=1.0)

        if self.radius_gap_shaded:
            self.shade_radius_gap(key_val, color)
        else:
            self.ax1.fill_between(self.radius_gap_x, self.radius_gap_y-self.radius_gap[1], self.radius_gap_y+self.radius_gap[1], color=color, zorder=0)

        self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos), \
                         xytext=(xytext[0], xytext[1]), textcoords='offset points', ha=ha, va=va, \
                         color=color_noalpha, zorder=1000+radius_z_order, rotation=rotation, rotation_mode="anchor",
                     fontsize=self.font_tracks, weight='bold')

    def shade_radius_gap(self, key_val, color):

        N = len(self.radius_gap_x)
        M = 1000

        ymin, ymax = min(self.radius_gap_y - 5 * self.radius_gap[1]), max(self.radius_gap_y + 5 * self.radius_gap[1])
        yy = np.linspace(ymin, ymax, M)
        a = [np.exp(-((Y - yy) / self.radius_gap[1]) ** 2) / self.radius_gap[1] for Y in zip(self.radius_gap_y)]
        A = np.array(a)
        A = A.reshape(N, M)
        self.ax1.imshow(A.T, cmap=key_val['cmap'], alpha=key_val['alpha'], aspect='auto',
                   origin='lower', extent=(min(self.radius_gap_x), max(self.radius_gap_x), ymin, ymax))


    def plot_color_bar(self):
        # use axes_list=[0.17, 0.55, 0.02, 0.30] for large font plots

        ax2 = self.fig.add_axes(self.colorbar_axes_list)

        if self.linear_insolation_scale:
            x_logvector = transform_linear_colorscale(self.colorbar_xvector, self.insol_min, self.insol_max)
        else:
            x_logvector = transform_log_colorscale(self.colorbar_xvector, self.insol_min, self.insol_max)

        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=self.color_map,
                                        orientation='vertical',
                                        ticks=x_logvector)

        ax2.yaxis.set_ticks_position(self.colorbar_ticks_position)
        #if self.colorbar_ticks_position == 'left':
        #    ax2.set_title(self.fp_foplus_spaces + 'F$_{\mathrm{p}}$/F$_{\oplus}$')
        #else:
        #    ax2.set_title('F$_{\mathrm{p}}$/F$_{\oplus}$' + self.fp_foplus_spaces)

        ax2.annotate('F$_{\mathrm{p}}$/F$_{\oplus}$', xy=(0, 0), xytext=(-1.1, 1.025),
                     textcoords='axes fraction', ha='left', va='bottom',
                     fontsize=self.font_label + 2, color='k', annotation_clip=True)

        cb1.ax.set_yticklabels(self.colorbar_xvector)

    def plot_jupiter_densities(self):

        mass = np.arange(0.001, 3.0, 0.001)
        n_dens = len(self.jupiter_densities_list)
        for i_d, density in enumerate(self.jupiter_densities_list):

            val = '{0:1.2f}'.format(density)
            radius = np.power(mass/density ,1./3.)
            dict_pams = {
                'x_pos'    : None,
                'y_pos'    : None,
                'rotation' : None,
                'use_box'  : True,
                'cmap'     : 'viridis',
                #'cmap'     : 'gist_heat',
                'color'    : 1./(n_dens+2) *i_d,
                'alpha'    : 0.8,
                'linestyle': '-',
                'label'    : val + ' $\\rho_{\mathrm{J}}$',
                'overplot' : True
                }

            self.add_tracks(mass, radius, key_val=dict_pams)

    def plot_lzeng_tracks(self):
        for key_name in self.lzeng_plot_list:
            #key_val = self.lzeng_plot_parameters[key_name]
            #mass = self.lzeng_tracks['Mearth']
            #radius = self.lzeng_tracks[key_name]
            self.add_tracks(self.lzeng_tracks['Mearth'],
                    self.lzeng_tracks[key_name],
                    self.lzeng_plot_parameters[key_name])

    def plot_elopez_tracks(self):
        for key_name in self.elopez_plot_list:
            #key_val = self.lzeng_plot_parameters[key_name]
            #mass = self.lzeng_tracks['Mearth']
            #radius = self.lzeng_tracks[key_name]
            self.add_tracks(self.elopez_tracks['Mearth'],
                       self.elopez_tracks[key_name],
                       self.elopez_plot_parameters[key_name])

    def add_track_from_files(self, track_filename, key_val=None, skip_header=0):
            tracks = np.genfromtxt(track_filename, skip_header=skip_header)
            mass = tracks[:,0]
            radius = tracks[:,1]
            self.add_tracks(mass, radius, key_val)
            print('Added additional track: ', track_filename)

    def add_tracks(self, mass, radius, key_val=None):

            if  self.tracks_on_top:
                z_order = self.z_offset + 1000.0
            else:
                z_order = self.z_offset

            if key_val is None:
                key_val = self.default_plot_parameters

            try:

                if not (key_val['x_pos'] or key_val['y_pos']):
                    x_pos, y_pos = self.interpolate_line_value(mass, radius)
                elif key_val['x_pos']:
                    x_pos, y_pos = self.interpolate_line_value(mass, radius, x_pos=key_val['x_pos'])
                elif key_val['y_pos']:
                    x_pos, y_pos = self.interpolate_line_value(mass, radius, y_pos=key_val['y_pos'])
                else:
                    x_pos = y_pos=key_val['x_pos']
                    y_pos = y_pos=key_val['y_pos']

                if key_val['rotation']:
                    rotation = key_val['rotation']
                else:
                    rotation = self.text_slope_match_line(self.ax1, mass, radius, x_pos)

            except:
                print(key_val['label'] + " composition track outside the boundaries of the plot")
                return

            bbox_props = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='w', pad=0.0)

            color_map = plt.get_cmap(key_val['cmap'])
            color = color_map(key_val['color'], alpha=key_val['alpha'])
            color_noalpha = color_map(key_val['color'], alpha=1.0)

            if 'overplot' in key_val:
                line = self.ax1.plot(mass, radius, color=color, zorder=950+z_order, ls=key_val['linestyle'])
            else:
                line = self.ax1.plot(mass, radius, color=color, zorder=0+z_order, ls=key_val['linestyle'])



            print('   Track: ', key_val['label'], ' x_pos: ', x_pos, ' y_pos: ', y_pos, ' rotation: ', rotation)


            if key_val['use_box']:
                self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos), \
                                 xytext=(0, -6), textcoords='offset points', ha='right', va='top', \
                                 color=color_noalpha, zorder=1000+z_order, rotation=rotation, rotation_mode="anchor",
                                 fontsize=self.font_tracks, weight='bold', bbox=bbox_props)
            else:
                self.ax1.annotate(key_val['label'], xy=(x_pos, y_pos), \
                                 xytext=(0, -6), textcoords='offset points', ha='right', va='top', \
                                 color=color_noalpha, zorder=1000+z_order, rotation=rotation, rotation_mode="anchor",
                                 fontsize=self.font_tracks, weight='bold' )#, bbox=bbox_props)
            if 'fill_below' in key_val:
                self.ax1.fill_between(mass, 0, radius, color=color, alpha=0.15)
            if 'fill_between' in key_val:
                self.ax1.fill_between(mass, radius-0.01,radius+0.01, color=color, zorder=0 + z_order, alpha=0.5)

    def add_points_from_dataset(self, dataset):

        n_planets = len(dataset.pl_mass)
        for ind in range(0, n_planets):

            pos = dataset.pl_mass[ind] * self.e2j_mass
            ypt = dataset.pl_radius[ind] * self.e2j_radius
            m_err1 = dataset.pl_mass_error_max[ind] * self.e2j_mass
            m_err2 = dataset.pl_mass_error_min[ind] * self.e2j_mass
            r_err1 = dataset.pl_radius_error_max[ind] * self.e2j_radius
            r_err2 = dataset.pl_radius_error_min[ind] * self.e2j_radius
            color = dataset.colors[ind]
            alpha_orig = dataset.alphas_original[ind]

            pl_name = dataset.pl_names[ind]
            #pl_name_original = dataset.pl_names[ind]
            m_detect_type = dataset.mass_detection_type[ind]
            pl_period = dataset.pl_orbper[ind]
            insol01_val = dataset.insol_01[ind]
            pl_ttvflag = dataset.pl_ttvflag[ind]

            #plotline, caplines, (barlinecols,) = self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
            #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
            #plotline, caplines, (barlinecols,) = self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
            #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

            #if pl_name == 'Kepler-19 b':
            #    print color,  pos, ypt, alpha_orig

            if pl_name in self.exclude_planet_names:
                continue

            if color[-1] < 0.0001:
                continue

            if (m_detect_type == 'TTV' and self.mark_ttvs) or (pl_ttvflag>0.5 and self.mark_flag_ttvs):
                marker_point = self.marker_point_ttv
            else:
                marker_point = self.marker_point

            if self.skip_plot_USPP and pl_period < 1.0000:
                continue

            if self.define_planet_names and self.define_short_names:
                for key_name, key_val in self.name_subtitutions.items():
                    pl_name = pl_name.replace(key_name, key_val)

            if self.define_thick_markers:
                self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=self.markersize)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=self.markersize)
                if insol01_val> 0.9 :
                    color_mod = self.color_map(0.9, alpha=color[3])
                else:
                    color_mod = color
                self.ax1.plot(pos, ypt, color=color, zorder=alpha_orig+self.add_overplot+0.3+self.z_offset, marker=marker_point, mfc=color, mec=color_mod, markersize=self.markersize)

            else:
                self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=0)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=0)
                self.ax1.plot(pos, ypt, color='white', mfc='white', zorder=alpha_orig+self.add_overplot+0.1+self.z_offset, marker=marker_point, mec="none", markersize=self.markersize)
                self.ax1.plot(pos, ypt, color=color, zorder=alpha_orig+self.add_overplot+0.3+self.z_offset, marker=marker_point, mfc=color, mec="none", markersize=self.markersize)

            if self.define_planet_names and not (self.define_planet_names_USPP and pl_period < 1.0000):
                #if pos*0.97 < self.xlims[0] or pos*1.02 > self.xlims[1] or ypt*0.98 < self.ylims[0] or ypt > self.ylims[1]: continue

                if pos < self.xlims[0] or pos > self.xlims[1] or ypt < self.ylims[0] or ypt > self.ylims[1]: continue

                if pos-self.xrange/10. < self.xlims[0] and ypt+self.yrange/20. > self.ylims[1]:
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(5, -2), textcoords='offset points', ha='left', va='top',
                                 color=color, fontsize=self.font_planet_name,  zorder=alpha_orig+self.add_overplot+0.5+self.z_offset, annotation_clip=True)
                elif pos-self.xrange/10. < self.xlims[0]:
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(5, 2), textcoords='offset points', ha='left', va='bottom',
                                 color=color, fontsize=self.font_planet_name,  zorder=alpha_orig+self.add_overplot+0.5+self.z_offset, annotation_clip=True)
                elif ypt+self.yrange/20. > self.ylims[1]:
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(-5, -2), textcoords='offset points', ha='right', va='top',
                                 color=color, fontsize=self.font_planet_name,  zorder=alpha_orig+self.add_overplot+0.5+self.z_offset, annotation_clip=True)
                else:
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(-5, 2), textcoords='offset points', ha='right', va='bottom',
                                 color=color, fontsize=self.font_planet_name,  zorder=alpha_orig+self.add_overplot+0.5+self.z_offset, annotation_clip=True)


    def add_USP_planets(self, dataset):

        if not self.skip_plot_USPP:
            print('you should activate skip_plot_USPP to avoid a mess of points')

        n_planets = len(dataset.pl_mass)
        for ind in range(0, n_planets):

            pos = dataset.pl_mass[ind]  * self.e2j_mass
            ypt = dataset.pl_radius[ind] * self.e2j_radius
            m_err1 = dataset.pl_mass_error_max[ind] * self.e2j_mass
            m_err2 = dataset.pl_mass_error_min[ind] * self.e2j_mass
            r_err1 = dataset.pl_radius_error_max[ind] * self.e2j_radius
            r_err2 = dataset.pl_radius_error_min[ind] * self.e2j_radius
            color = dataset.colors[ind]
            alpha_orig = dataset.alphas_original[ind] + 10


            pl_name = dataset.pl_names[ind]
            #pl_name_original = dataset.pl_names[ind]
            m_detect_type = dataset.mass_detection_type[ind]
            pl_period = dataset.pl_orbper[ind]
            insol01_val = dataset.insol_01[ind]
            pl_ttvflag = dataset.pl_ttvflag[ind]

            #plotline, caplines, (barlinecols,) = self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color,
            #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)
            #plotline, caplines, (barlinecols,) = self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color,
            #                                                  capsize=5, capthick=2, markerfacecolor='white', zorder=alpha_orig)

            #if pl_name == 'Kepler-19 b':
            #    print color,  pos, ypt, alpha_orig

            if pl_name in self.exclude_planet_names:
                continue

            if color[-1] < 0.0001:
                continue

            if (m_detect_type == 'TTV' and self.mark_ttvs) or (pl_ttvflag>0.5 and self.mark_flag_ttvs):
                marker_point = self.marker_point_ttv
            else:
                marker_point = self.marker_point

            if pl_period > 1.0000:
                continue

            if self.define_planet_names and self.define_short_names:
                for key_name, key_val in self.name_subtitutions.iteritems():
                    pl_name = pl_name.replace(key_name, key_val)

            if self.define_thick_markers:
                self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color='k', zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=self.markersize_USP)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color='k', zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=self.markersize_USP)
                if insol01_val> 0.9 :
                    color_mod = self.color_map(0.9, alpha=color[3])
                else:
                    color_mod = color
                #self.ax1.plot(pos, ypt, color=color, zorder=alpha_orig+self.add_overplot+1.3+self.z_offset, marker=marker_point, mfc=color, mec='k', markersize=self.markersize_USP, mew=self.markersize_USP/6 )
                self.ax1.plot(pos, ypt, color=color, zorder=alpha_orig+self.add_overplot+1.3+self.z_offset, marker=marker_point, mfc=color_mod, mec='k', markersize=self.markersize_USP, mew=self.markersize_USP/6 )

            else:
                self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=0)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color, zorder=alpha_orig+self.add_overplot+self.z_offset, marker=marker_point, mfc='white', markersize=0)
                self.ax1.plot(pos, ypt, color='white', mfc='white', zorder=alpha_orig+self.add_overplot+0.1+self.z_offset, marker=marker_point, mec="none", markersize=self.markersize_USP)
                self.ax1.plot(pos, ypt, color=color, zorder=alpha_orig+self.add_overplot+0.3+self.z_offset, marker=marker_point, mfc=color, mec="none", markersize=self.markersize_USP)

            if self.define_planet_names_USPP and pl_period < 1.0000:
                if pos*0.98 < self.xlims[0] or pos*1.02 > self.xlims[1] or ypt*0.98 < self.ylims[0] or ypt > self.ylims[1]: continue
                bbox_props = dict(boxstyle="square", fc="w", alpha=0.7, edgecolor=color, pad=0.1)
                if pl_name=='K2-141 b' or pl_name == 'K2-229 b':
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(4, -4), textcoords='offset points', ha='left', va='top',
                                 color='k', fontsize=self.font_USP_name, zorder=alpha_orig + self.add_overplot/2. + 0.5 + self.z_offset,
                                 annotation_clip=True, bbox=bbox_props)

                elif pl_name == 'HD 3167 b G17':
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(4, -4), textcoords='offset points', ha='left', va='top',
                                 color='k', fontsize=self.font_USP_name, zorder=alpha_orig + self.add_overplot/2. + 0.5 + self.z_offset,
                                annotation_clip=True, bbox=bbox_props)

                elif pl_name =='55 Cnc e':
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom',
                                 color='k', fontsize=self.font_USP_name, zorder=alpha_orig + self.add_overplot/2. + 0.5 + self.z_offset,
                                 annotation_clip=True, bbox=bbox_props)

                elif pl_name=='K2-131 b' or pl_name=='Kepler-10 b' or pl_name == 'HD 3167 b' or pl_name =='CoRoT-7 b':
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                                 xytext=(-4, -4), textcoords='offset points', ha='right', va='top',
                                 color='k', fontsize=self.font_USP_name, zorder=alpha_orig + self.add_overplot/2. + 0.5 + self.z_offset,
                                 annotation_clip=True, bbox=bbox_props)

                else:
                    self.ax1.annotate(pl_name, xy=(pos, ypt),
                             xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
                             color='k', fontsize=self.font_USP_name,  zorder=alpha_orig+self.add_overplot/2.+0.5+self.z_offset, annotation_clip=True, bbox=bbox_props)


    def add_my_planets(self, dataset):
        z_order = self.add_overplot+self.z_offset+2000.00
        n_planets = len(dataset.pl_mass)
        for ind in range(0, n_planets):

            pos = dataset.pl_mass[ind] * self.e2j_mass
            ypt = dataset.pl_radius[ind]  * self.e2j_radius
            m_err1 = dataset.pl_mass_error_max[ind] * self.e2j_mass
            m_err2 = dataset.pl_mass_error_min[ind] * self.e2j_mass
            r_err1 = dataset.pl_radius_error_max[ind]  * self.e2j_radius
            r_err2 = dataset.pl_radius_error_min[ind]  * self.e2j_radius
            color = dataset.colors[ind]
            alpha_orig = dataset.alphas_original[ind]
            pl_name = dataset.pl_names[ind]
            m_detect_type = dataset.mass_detection_type[ind]
            pl_period = dataset.pl_orbper[ind]
            insol01_val = dataset.insol_01[ind]
            pl_ttvflag = dataset.pl_ttvflag[ind]
            bbox_ha = dataset.textbox_ha[ind]
            bbox_va = dataset.textbox_va[ind]
            pl_upper_limit = dataset.pl_upper_limit[ind]

            if (m_detect_type == 'TTV' and self.mark_ttvs) or (pl_ttvflag>0.5 and self.mark_flag_ttvs):
                marker_point = self.marker_point_ttv
            else:
                marker_point = self.marker_point

            if insol01_val> 0.9 :
                insol01_val = 0.9

            color_mod = self.color_map(insol01_val, alpha=1.0)

            if pl_upper_limit:
                color_bar = self.color_map(insol01_val, alpha=0.5)
                self.ax1.errorbar(0.0, ypt, xerr=pos + m_err1, color=color_bar, zorder=z_order, lw=4, xlolims=True, capsize=5)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color_mod, zorder=z_order, marker=marker_point, mfc='white', markersize=self.markersize*1.5, lw=2)
                #self.ax1.plot(pos, ypt, color='white', zorder=z_order+0.2, marker=marker_point, mfc='white', mec='k', mew = 0 , markersize=self.markersize*2, lw=0)
                #self.ax1.plot(pos, ypt, color=color_mod, zorder=z_order+0.3, marker=marker_point, mfc=color_bar, mec='k', mew = 2 , markersize=self.markersize*2, lw=4)
                self.ax1.plot(pos, ypt, color=color_bar, zorder=z_order+0.3, marker=marker_point, mfc=color_mod, mec='k', mew = 1 , markersize=self.markersize*1.5, lw=2)
            else:
                color_mod = [0.961336,0.548636,0.275305,1.      ]
                self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color_mod, zorder=z_order, marker=marker_point, mfc='white', markersize=self.markersize, lw=4)
                self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color_mod, zorder=z_order, marker=marker_point, mfc='white', markersize=self.markersize, lw=4)
                self.ax1.plot(pos, ypt, color=color_mod, zorder=z_order+0.3, marker=marker_point, mfc=color_mod, mec='k', mew = 2 , markersize=self.markersize*2, lw=4)


            if pos*0.98 < self.xlims[0] or pos*1.02 > self.xlims[1] or ypt*0.98 < self.ylims[0] or ypt > self.ylims[1]: continue
            bbox_props = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='k', pad=0.2)

            if bbox_ha=='right':
                if bbox_va=='bottom':
                    xytext = [-6, 6]
                else:
                    xytext = [-6, -6]
            else:
                if bbox_va=='bottom':
                    xytext = [6, 6]
                else:
                    xytext = [6, -6]

            self.ax1.annotate(pl_name, xy=(pos, ypt),
                xytext=xytext, textcoords='offset points', ha=bbox_ha, va=bbox_va,
                color='k', fontsize=self.font_my_planet, zorder=z_order,
                annotation_clip=True, bbox=bbox_props)

    def add_other_planets(self, dataset):
        z_order = 1.0+self.add_overplot+self.z_offset+1000.0


        n_planets = len(dataset.pl_mass)
        for ind in range(0, n_planets):

            pos = dataset.pl_mass[ind]  * self.e2j_mass
            ypt = dataset.pl_radius[ind]  * self.e2j_radius
            m_err1 = dataset.pl_mass_error_max[ind]  * self.e2j_mass
            m_err2 = dataset.pl_mass_error_min[ind]  * self.e2j_mass
            r_err1 = dataset.pl_radius_error_max[ind]  * self.e2j_radius
            r_err2 = dataset.pl_radius_error_min[ind]  * self.e2j_radius
            color = dataset.colors[ind]
            alpha_orig = dataset.alphas_original[ind]
            pl_name = dataset.pl_names[ind]
            m_detect_type = dataset.mass_detection_type[ind]
            pl_period = dataset.pl_orbper[ind]
            insol01_val = dataset.insol_01[ind]
            pl_ttvflag = dataset.pl_ttvflag[ind]
            bbox_ha = dataset.textbox_ha[ind]
            bbox_va = dataset.textbox_va[ind]

            if (m_detect_type == 'TTV' and self.mark_ttvs) or (pl_ttvflag>0.5 and self.mark_flag_ttvs):
                marker_point = self.marker_point_ttv
            else:
                marker_point = self.marker_point

            alpha = 1.0

            if insol01_val> 0.9 :
                insol01_val = 0.9

            color_mod = self.color_map(insol01_val, alpha=alpha)

            p1 = self.ax1.errorbar(pos, ypt, xerr=([m_err2], [m_err1]), color=color_mod, zorder=z_order, marker=marker_point, mfc='white', markersize=self.markersize, lw=3)#, ls='-.')
            p1[-1][0].set_linestyle('dashed')

            p2 = self.ax1.errorbar(pos, ypt, yerr=([r_err2], [r_err1]), color=color_mod, zorder=z_order, marker=marker_point, mfc='white', markersize=self.markersize, lw=3)#, linestyle='dashed')
            p2[-1][0].set_linestyle('dashed')

            self.ax1.plot(pos, ypt, color=color_mod, zorder=z_order+0.3, marker=marker_point, mfc=color_mod, mec='k', mew = 1 , markersize=self.markersize+1, lw=2)


            if pos*0.98 < self.xlims[0] or pos*1.02 > self.xlims[1] or ypt*0.98 < self.ylims[0] or ypt > self.ylims[1]: continue
            bbox_props = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='k', pad=0.2)

            if bbox_ha=='right':
                if bbox_va=='bottom':
                    xytext = [-6, 6]
                else:
                    xytext = [-6, -6]
            else:
                if bbox_va=='bottom':
                    xytext = [6, 6]
                else:
                    xytext = [6, -6]

            self.ax1.annotate(pl_name, xy=(pos, ypt),
                xytext=xytext, textcoords='offset points', ha=bbox_ha, va=bbox_va,
                color='k', fontsize=(self.font_my_planet + self.font_planet_name)/2., zorder=alpha_orig + self.add_overplot + 0.5 + self.z_offset,
                annotation_clip=True, bbox=bbox_props)

    def add_solar_system(self):
        bbox_props = dict(boxstyle="square", fc="w", alpha=0.9, edgecolor='b', pad=0.1)

        self.ax1.plot([0.815, 1.00],[0.949,1.00],'ob', markersize=self.markersize+4, marker='*', zorder= 10000+ self.z_offset)
        self.ax1.annotate('Earth', xy=(1.0, 1.0),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)
        self.ax1.annotate('Venus', xy=(0.815, 0.949),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='top',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)


    def show_figure(self):
        plt.show()



    def save_figure(self):

        output_name = self.prefix_output_name + self.name_thick_markers \
                                    + self.name_planet_names \
                                    + self.name_alpha_density \
                                    + self.name_plot_USPP \
                                    + '.pdf'
        print('OUTPUT plot:  ', output_name)
        #plt.tight_layout()
        plt.savefig(output_name, bbox_inches='tight')
        plt.close()

    def interpolate_line_value(self, mass, radius, x_pos=None, y_pos=None,
        default_position = 'top_right'):

        if default_position == 'top_right':
            ind = np.where((radius<=self.ylims[1]) & (mass<=self.xlims[1]))[0][-1]
        if default_position == 'bottom_left':
            ind = np.where((radius>=self.ylims[0]) & (mass>=self.xlims[0]))[0][0]

        x1 = mass[ind]
        x2 = mass[ind+1]
        y1 = radius[ind]
        y2 = radius[ind+1]

        if y1 == y2:
            if x_pos is None:
                return self.xy_labels[0], y1
            else:
                return x_pos, y1

        if x1 == x2:
            if y_pos is None:
                return x1, self.xy_labels[1]
            else:
                return x1, y_pos

        if x_pos:
            y_out = y1 + (x_pos-x1)*(y2-y1)/(x2-x1)
            return x_pos, y_out

        if y_pos:
            x_out = x1 + (y_pos-y1)*(x2-x1)/(y2-y1)
            return x_out, y_pos

        if x2>self.xlims[1]:
            # Case: the track is ending on the right side of the plot
            x_out = self.xy_labels[0]
            y_out = y1 + (x_out-x1)*(y2-y1)/(x2-x1)
        else:
            # Case: the track is ending on the upper part of the plot
            y_out = self.xy_labels[1]
            x_out = x1 + (y_out-y1)*(x2-x1)/(y2-y1)

        return x_out, y_out


    def text_slope_match_line(self, ax, xdata, ydata, pos, nearly_vertical=False):

        # find the slope

        if nearly_vertical:
            ind = np.where(ydata > pos)[0][0]
        else:
            ind = np.argmin(np.abs(xdata-pos))

        x1 = xdata[ind-1]
        x2 = xdata[ind+1]
        y1 = ydata[ind-1]
        y2 = ydata[ind+1]

        p1 = np.array((x1, y1))
        p2 = np.array((x2, y2))

        # get the line's data transform
        #ax = ax.get_axes()

        sp1 = ax.transData.transform_point(p1)
        sp2 = ax.transData.transform_point(p2)

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        return math.degrees(math.atan(rise/run))
