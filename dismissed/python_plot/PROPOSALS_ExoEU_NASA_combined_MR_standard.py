from plot_routines import *


def make_plot(properties_dict):

    import cmocean
    #color_map = plt.get_cmap('plasma')
    #color_map = plt.get_cmap('nipy_spectral_r')


    parameters_dict = {
        'xticks': [0.5, 1, 2, 5, 10, 20, 30],
        'xlims': [0.4, 30],
        'ylims': [0.8, 3.5],
        'alpha_upper_limit': 0.6,
        'alpha_lower_limit': 0.2,
        'alpha_upper_value': 0.6,
        'alpha_lower_value': 0.2,
        'insol_min': 1.0,
        'insol_max': 3000.0,
        'colorbar_xvector': [1, 3, 10, 30, 100, 300, 1000],
        #'colorbar_xvector': [1, 3, 10, 30, 100, 300, 1000, 3000],
        'add_overplot': 0.0,
        'color_map': cmocean.cm.thermal
        #'font_label': 24,
        #'font_planet_name': 10,
        #'font_tracks':18,
        #'font_Solar_name':18
    }

    if properties_dict['define_alpha_density']:
        parameters_dict['alpha_upper_limit'] = 0.6
        parameters_dict['alpha_lower_limit'] = 0.3
        parameters_dict['alpha_upper_value'] = 0.6
        parameters_dict['alpha_lower_value'] = 0.3




    start_up(parameters_dict)


    csfont = {'fontname':'Times New Roman'}
    matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman']})

    create_flags(properties_dict)
    constants()
    short_name_subtitutions()

    data_combined = combine_catalogues()
    perform_selection(data_combined)

    insolation_scale()
    define_alpha_colors()
    setup_plot(12,8)

    add_points_from_dataset()

    if not properties_dict['no_color_scale']:
        add_color_bar(axes_list=[0.10, 0.55, 0.02, 0.30])
    plot_lizeng_tracks(z_overcome=1000.0)


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='red', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='red', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-101b'
    mass = 3.174736
    radius = 2.000
    r_err2 = 0.1 # lower error bar
    r_err1 = 0.1 # upper error bar
    m_err2 = 1.972261 # left error bar
    m_err1 = 1.728341 # right error bar

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='red', zorder=1000000, marker=marker_point, mfc='red', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='green', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='green', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-101b'
    mass = 6.236904
    radius = 2.000
    r_err2 = 0.1 # lower error bar
    r_err1 = 0.1 # upper error bar
    m_err2 = 1.306899 # left error bar
    m_err1 = 1.392773 # right error bar

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='green', zorder=1000000, marker=marker_point, mfc='green', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)



    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='blue', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='blue', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-101b'
    mass = 12.552851
    radius = 2.000
    r_err2 = 0.1 # lower error bar
    r_err1 = 0.1 # upper error bar
    m_err2 = 1.367502 # left error bar
    m_err1 = 1.470664 # right error bar

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='blue', zorder=1000000, marker=marker_point, mfc='blue', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)







##################################################################
###########     K2-136b
##################################################################


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='red', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='red', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136b'
    mass = 0.929394
    radius = 0.99
    r_err2 = 0.04 # lower error bar
    r_err1 = 0.06 # upper error bar
    m_err2 = 0.479255
    m_err1 = 0.905728

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='red', zorder=1000000, marker=marker_point, mfc='red', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='green', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='green', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136b'
    mass = 2.754726
    radius = 0.99
    r_err2 = 0.04 # lower error bar
    r_err1 = 0.06 # upper error bar
    m_err2 = 1.204926
    m_err1 = 0.910399

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='green', zorder=1000000, marker=marker_point, mfc='green', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)



    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='blue', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='blue', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136b'
    mass = 3.264450
    radius = 0.99
    r_err2 = 0.04 # lower error bar
    r_err1 = 0.06 # upper error bar
    m_err2 = 1.265018
    m_err1 = 1.104723

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='blue', zorder=1000000, marker=marker_point, mfc='blue', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)




##################################################################
###########     K2-136c
##################################################################


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='red', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='red', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136c'
    mass = 3.229902
    radius = 2.91
    r_err2 = 0.10 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.642065
    m_err1 = 1.496213

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='red', zorder=1000000, marker=marker_point, mfc='red', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='green', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='green', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136c'
    mass = 8.985295
    radius = 2.91
    r_err2 = 0.10 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.647561
    m_err1 = 1.489745

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='green', zorder=1000000, marker=marker_point, mfc='green', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)



    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='blue', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='blue', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136c'
    mass = 18.446796
    radius = 2.91
    r_err2 = 0.10 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.672911
    m_err1 = 1.801098

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='blue', zorder=1000000, marker=marker_point, mfc='blue', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)



##################################################################
###########     K2-136d
##################################################################


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='red', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='red', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136d'
    mass = 2.135329
    radius = 1.45
    r_err2 = 0.08 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.257667
    m_err1 = 1.478348

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='red', zorder=1000000, marker=marker_point, mfc='red', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)


    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='green', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='green', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136d'
    mass =2.804351
    radius = 1.45
    r_err2 = 0.08 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.327330
    m_err1 = 1.158479

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='green', zorder=1000000, marker=marker_point, mfc='green', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)



    bbox_props = dict(boxstyle="square", fc="w", alpha=0.4, edgecolor='blue', pad=0.1)
    bbox_props_mine = dict(boxstyle="square", fc="w", alpha=0.8, edgecolor='blue', pad=0.1)
    marker_point = "o"
    markersize = 12
    pl_name = 'K2-136d'
    mass =4.445044
    radius = 1.45
    r_err2 = 0.08 # lower error bar
    r_err1 = 0.11 # upper error bar
    m_err2 = 1.252377
    m_err1 = 1.221327

    plt.errorbar(mass, radius, yerr=([r_err2], [r_err1]), xerr=([m_err2], [m_err1]), color='blue', zorder=1000000, marker=marker_point, mfc='blue', markersize=markersize)
    plt.annotate(pl_name, xy=(mass, radius),
        xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',
        color='black', fontsize=12, zorder=1000000, annotation_clip=True, bbox=bbox_props_mine)

















    prefix_output_name = 'PROPOSAL_ExoEU_NASA_MR'
    save_fig(prefix_output_name)


#properties_dict['define_plot_USPP'] = False

properties_dict = {}
properties_dict['define_thick_markers'] = False
properties_dict['define_planet_names'] = False
properties_dict['define_alpha_density'] = False
properties_dict['define_short_names'] = True
properties_dict['no_color_scale'] = True
make_plot(properties_dict)

def dump():

    properties_dict = {}
    properties_dict['define_thick_markers'] = True
    properties_dict['define_planet_names'] = False
    properties_dict['define_alpha_density'] = False
    properties_dict['define_short_names'] = True
    properties_dict['no_color_scale'] = False
    make_plot(properties_dict)
