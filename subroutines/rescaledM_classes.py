from plotting_classes import *
from scipy.stats import multivariate_normal

def text_slope_match_line_alternative(ax, xdata, ydata, y_pos):
    # for nearly vertical lines

    # find the slope
    ind = np.where(ydata > y_pos)[0][0]

    #ind = np.argmin(np.abs(xdata-x_pos))

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

class MR_rescaledM_plot(MR_Plot):


    def rescale_mass(self):
        M = self.lzeng_tracks['Mearth']
        R = self.lzeng_tracks['rocky']
        self.reference_density_lzeng = self.lzeng_tracks['Mearth'].copy()
        self.reference_density_fit = np.polynomial.chebyshev.chebfit(R, np.log10(M), 5)

    def rescale_dataset(self, dataset):

        dataset.pl_mass /= 10**np.polynomial.chebyshev.chebval(dataset.pl_radius, self.reference_density_fit)
        dataset.pl_mass_error_max /= 10**np.polynomial.chebyshev.chebval(dataset.pl_radius, self.reference_density_fit)
        dataset.pl_mass_error_min /= 10**np.polynomial.chebyshev.chebval(dataset.pl_radius, self.reference_density_fit)
        #self.add_points_from_dataset(dataset)
        return dataset


    def plot_lzeng_tracks(self):

        if  self.lzeng_tracks_on_top:
            lzeng_z_order = self.z_offset + 1000.0
        else:
            lzeng_z_order = self.z_offset

        for key_name in self.lzeng_plot_list:
            key_val = self.lzeng_plot_parameters[key_name]

            x_rescaled = self.lzeng_tracks['Mearth']/10**np.polynomial.chebyshev.chebval(self.lzeng_tracks[key_name], self.reference_density_fit)
            fit_coeff = np.polynomial.chebyshev.chebfit(self.lzeng_tracks[key_name], x_rescaled, 3)

            R_range = np.arange(0.1, 5.0, 0.01)
            M_range = np.polynomial.chebyshev.chebval(R_range, fit_coeff)



            #if key_val['x_pos'] == 0.0:
            ypos_sel = np.where(R_range > self.ylims[0])[0][0]
            key_val['x_pos'] = M_range[ypos_sel]
            y_pos = R_range[ypos_sel] * 1.00


            color_map = plt.get_cmap(key_val['cmap'])
            color = color_map(key_val['color'], alpha=key_val['alpha'])
            color_noalpha = color_map(key_val['color'], alpha=1.0)

            if key_name != 'rocky':
                line = self.ax1.plot(M_range,R_range, color=color, zorder=0+lzeng_z_order, ls=key_val['linestyle'])
            else:
                self.ax1.axvline(1.0, c=color, ls=key_val['linestyle'], zorder=0+lzeng_z_order,)


            rotation = text_slope_match_line_alternative(self.ax1, M_range, R_range, y_pos)

            if rotation < 0.0:
                rotation += 180.0

            #if key_name == 'rocky':
            #    color_noalpha = 'r'

            self.ax1.annotate(key_val['label'], xy=(key_val['x_pos'], y_pos), \
                             xytext=(-2, 2), textcoords='offset points', ha='left', va='bottom', \
                             color=color_noalpha, zorder=1000+lzeng_z_order, rotation=rotation, rotation_mode="anchor",
                         fontsize=self.font_tracks, weight='bold')




    def add_solar_system_rescaled(self):
        bbox_props = dict(boxstyle="square", fc="w", alpha=0.9, edgecolor='b', pad=0.1)

        earth_factor= 10**np.polynomial.chebyshev.chebval(1.000, self.reference_density_fit)
        venus_factor= 10**np.polynomial.chebyshev.chebval(0.949, self.reference_density_fit)

        self.ax1.plot([0.815/venus_factor, 1.00/earth_factor],[0.949, 1.00],'ob', markersize=self.markersize+4, marker='*', zorder= 10000+ self.z_offset)
        self.ax1.annotate('Earth', xy=(1.0/earth_factor, 1.0),
                     xytext=(5, -5), textcoords='offset points', ha='left', va='top',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)
        self.ax1.annotate('Venus', xy=(0.815/venus_factor, 0.949),
                     xytext=(5, -5), textcoords='offset points', ha='left', va='top',
                     color='b', fontsize=self.font_Solar_name, zorder= 10000+ self.z_offset,
                     annotation_clip=True, bbox=bbox_props)

    def make_plot(self, dataset):

        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)

        self.setup_plot()
        self.ax1.set_xlabel('Mass/Mass$_{\mathrm{Rocky}}$')

        self.ax1.axvline(1.0, c='r')
        self.rescale_mass()
        dataset = self.rescale_dataset(dataset)

        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_color_bar:
            self.plot_color_bar()
        if self.add_fulton_gap:
            self.plot_fulton_gap(put_left=True)
        self.add_points_from_dataset(dataset)
        self.add_solar_system_rescaled()

    def make_plot_with_my_planets(self, dataset, my_planets):
        self.exclude_planet_names.extend(my_planets.pl_names)


        self.insolation_scale(dataset)
        self.define_alpha_colors(dataset)
        self.insolation_scale(my_planets)
        self.define_alpha_colors(my_planets)

        self.colorbar_axes_list=[0.90, 0.55, 0.03, 0.35]

        self.setup_plot()

        self.ax1.set_xlabel('Mass/Mass$_{Rocky}$')
        #self.ax1.axvline(1.0, c='r', ls = )

        self.rescale_mass()
        dataset = self.rescale_dataset(dataset)
        my_planets = self.rescale_dataset(my_planets)


        if self.add_lzeng_tracks:
            self.plot_lzeng_tracks()
        if self.add_color_bar:
            self.plot_color_bar()
        if self.add_fulton_gap:
            self.plot_fulton_gap()

        self.add_points_from_dataset(dataset)
        self.add_my_planets(my_planets)
        self.add_solar_system_rescaled()


class MR_densityM_map(MR_rescaledM_plot):

    def prepare_mgrid(self):
        self.MRmap_x, self.MRmap_y = np.mgrid[self.xlims[0]:self.xlims[1]:.05, self.ylims[0]:self.ylims[1]:.05]
        self.MRmap_pos = np.dstack((self.MRmap_x, self.MRmap_y))

    def add_points_from_dataset(self, dataset):
        self.prepare_mgrid()
        n_planets = len(dataset.pl_mass)
        ii = 0
        for ind in xrange(0, n_planets):

            pos = dataset.pl_mass[ind]
            ypt = dataset.pl_radius[ind]
            m_err1 = dataset.pl_mass_error_max[ind]
            m_err2 = dataset.pl_mass_error_min[ind]
            r_err1 = dataset.pl_radius_error_max[ind]
            r_err2 = dataset.pl_radius_error_min[ind]

            try:

                planet_mvr = multivariate_normal([pos, ypt], [[(m_err1+m_err2)/2., 0.0], [0.0, (r_err1+r_err2)/2.]])

                if ii == 0:
                    self.MRmap = planet_mvr.pdf(self.MRmap_pos)
                else:
                    self.MRmap += planet_mvr.pdf(self.MRmap_pos)
                ii += 1
            except:
                continue

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
