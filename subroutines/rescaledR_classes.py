from plotting_classes import *
from scipy.stats import multivariate_normal

class MR_rescaledR_plot(MR_Plot):


    def rescale_radius(self):
        M = self.lzeng_tracks['Mearth']
        R = self.lzeng_tracks['rocky']
        self.reference_density_lzeng = self.lzeng_tracks['rocky'].copy()
        self.reference_density_fit = np.polynomial.chebyshev.chebfit(np.log10(M), R, 3)
        #fit_output = np.polynomial.chebyshev.chebval(logM_plot, p)

        #density = M/R**3

        #self.ax1.scatter(M, R)
        #self.ax1.plot(10**logM_plot, fit_output)
        #self.ax1.scatter(M, density, c='r')


    def rescale_dataset(self, dataset):

        logM = np.log10(dataset.pl_mass)
        dataset.pl_radius /= np.polynomial.chebyshev.chebval(logM, self.reference_density_fit)
        dataset.pl_radius_error_max /= np.polynomial.chebyshev.chebval(logM, self.reference_density_fit)
        dataset.pl_radius_error_min /= np.polynomial.chebyshev.chebval(logM, self.reference_density_fit)
        #self.add_points_from_dataset(dataset)
        return dataset

    def rescale_lzeng_tracks(self):

        for keyword in self.lzeng_plot_list:
            self.lzeng_tracks[keyword] /= self.reference_density_lzeng

    def rescale_fulton_gap(self):
        self.fulton_gap_y /= np.polynomial.chebyshev.chebval(np.log10(self.fulton_gap_x), self.reference_density_fit)


    def add_solar_system_rescaled(self):
        bbox_props = dict(boxstyle="square", fc="w", alpha=0.9, edgecolor='b', pad=0.1)

        earth_factor= np.polynomial.chebyshev.chebval(0.00, self.reference_density_fit)
        venus_factor= np.polynomial.chebyshev.chebval(np.log10(0.949), self.reference_density_fit)

        self.ax1.plot([0.815, 1.00],[0.949/venus_factor, 1.00/earth_factor],'ob', markersize=self.markersize+4, marker='*', zorder= 10000+ self.z_offset)
        self.ax1.annotate('Earth', xy=(1.0, 1.0/earth_factor),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)
        self.ax1.annotate('Venus', xy=(0.815, 0.949/venus_factor),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)

    def make_plot(self, dataset):

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)

        self.setup_plot()
        self.ax1.set_ylabel('Radius/Radius$_{Rocky}$')

        self.rescale_radius()
        dataset = self.rescale_dataset(dataset)
        self.rescale_lzeng_tracks()
        self.rescale_fulton_gap()

        if self.add_color_bar:
            self.plot_color_bar()
        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_fulton_gap:
            self.plot_fulton_gap()
        self.add_points_from_dataset(dataset)
        self.add_solar_system_rescaled()

    def make_plot_with_my_planets(self, dataset, my_planets):

        self.exclude_planet_names.extend(my_planets.pl_names)

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)
        self.insolation_scale(my_planets)
        self.define_alpha_colors(my_planets)

        self.setup_plot()

        self.rescale_radius()
        dataset = self.rescale_dataset(dataset)
        my_planets = self.rescale_dataset(my_planets)
        self.rescale_lzeng_tracks()
        self.rescale_fulton_gap()

        if self.add_color_bar:
            self.plot_color_bar()
        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_fulton_gap:
            self.plot_fulton_gap()

        self.add_points_from_dataset(dataset)
        self.add_my_planets(my_planets)
        self.add_solar_system_rescaled()



class MR_density_map(MR_rescaledR_plot):



    def prepare_mgrid(self):
        self.MRmap_x, self.MRmap_y = np.mgrid[self.xlims[0]:self.xlims[1]:.05, self.ylims[0]:self.ylims[1]:.05]
        self.MRmap_pos = np.dstack((self.MRmap_x, self.MRmap_y))

    def add_points_from_dataset(self, dataset):
        self.prepare_mgrid()
        n_planets = len(dataset.pl_mass)
        for ind in xrange(0, n_planets):

            pos = dataset.pl_mass[ind]
            ypt = dataset.pl_radius[ind]
            m_err1 = dataset.pl_mass_error_max[ind]
            m_err2 = dataset.pl_mass_error_min[ind]
            r_err1 = dataset.pl_radius_error_max[ind]
            r_err2 = dataset.pl_radius_error_min[ind]

            planet_mvr = multivariate_normal([pos, ypt], [[(m_err1+m_err2)/2., 0.0], [0.0, (r_err1+r_err2)/2.]])

            if ind == 0:
                self.MRmap = planet_mvr.pdf(self.MRmap_pos)
            else:
                self.MRmap += planet_mvr.pdf(self.MRmap_pos)

        """
        for ind in xrange(0, 10000):
            x = np.random.uniform(0.4, 50)
            y = np.random.uniform(0.4, 4.0)
            #self.ax1.scatter(x,y)
            planet_mvr = multivariate_normal([x, y], [[x/10., 0.0], [0.0, y/10.]])
            if ind == 0:
                self.MRmap = planet_mvr.pdf(self.MRmap_pos)
            else:
                self.MRmap += planet_mvr.pdf(self.MRmap_pos)
        """
        self.ax1.contour(self.MRmap_x, self.MRmap_y, self.MRmap)


    def make_plot_with_my_planets(self, dataset, my_planets):

        self.exclude_planet_names.extend(my_planets.pl_names)

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)
        self.insolation_scale(my_planets)
        self.define_alpha_colors(my_planets)

        self.setup_plot()

        self.rescale_radius()
        dataset = self.rescale_dataset(dataset)
        my_planets = self.rescale_dataset(my_planets)
        self.rescale_lzeng_tracks()
        self.rescale_fulton_gap()

        self.add_color_bar()
        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_fulton_gap:
            self.plot_fulton_gap()

        self.add_points_from_dataset(dataset)
        self.add_my_planets(my_planets)
        self.add_solar_system_rescaled()

# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# from scipy.stats import multivariate_normal
# rv = multivariate_normal([-0.5, -0.2], [[0.5, 0.0], [0.0, 0.5]])
