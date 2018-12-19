from constants import *

<<<<<<< HEAD



class Dataset():
    def fix_TTVs(self):
        TTV_list = [
            'Kepler-30 b',
            'Kepler-30 c',
            'Kepler-30 d',
            'Kepler-35 (AB) b',
            'Kepler-18 d',
            'Kepler-56 b',
            'Kepler-89 e',
            'K2-19 b',
            'Kepler-34 (AB) b'
        ]
        for planet_i, planet_name in enumerate(self.pl_names):
            if planet_name in TTV_list:
                self.pl_ttvflag[planet_i] = 1.00

=======
class Dataset():
>>>>>>> 294d8892ec0b362ed7bcc8280d22867ee64e1211
    def compute_derived_parameters(self):
        self.insol_01 = self.pl_orbper*0.0

        self.a_smj_AU = np.power((Mu_sun * np.power(self.pl_orbper * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * self.st_mass, 1.00 / 3.00)

        self.insol = self.st_rad**2 * (self.st_teff/5777.0)**4 / self.a_smj_AU**2

        self.pl_masserr_avg = (self.pl_mass_error_max + self.pl_mass_error_min)/2.0
        self.pl_radiuserr_avg = (self.pl_radius_error_max + self.pl_radius_error_min)/2.0

        self.pl_dens = self.pl_mass/self.pl_radius**3
        self.pl_dens_error_max =self.pl_dens * np.sqrt( (self.pl_mass_error_max/self.pl_mass)**2 + 9*(self.pl_radius_error_min/self.pl_radius)**2)
        self.pl_dens_error_min =self.pl_dens * np.sqrt( (self.pl_mass_error_min/self.pl_mass)**2 + 9*(self.pl_radius_error_max/self.pl_radius)**2)
        self.pl_denserr_avg = self.pl_dens * np.sqrt( (self.pl_masserr_avg/self.pl_mass)**2 + 9*(self.pl_radiuserr_avg/self.pl_radius)**2)

        self.perc_error_density = self.pl_denserr_avg/self.pl_dens
        self.perc_error_mass = self.pl_masserr_avg/self.pl_mass

        self.alphas = None
        self.alphas_original = None
        self.colors = None

<<<<<<< HEAD


=======
>>>>>>> 294d8892ec0b362ed7bcc8280d22867ee64e1211
class Dataset_Input(Dataset):
    def __init__(self, input_planets):

        names_list = ['name','orbital_period',
            'mass','mass_error_max','mass_error_min',
            'radius','radius_error_max','radius_error_min',
            'star_mass','star_radius','star_teff','pl_ttvflag','textbox_ha','textbox_va', 'pl_upper_limit']

        data_input = np.genfromtxt(
            input_planets,           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype='float32',        # data type
            filling_values=0.00000000,       # fill missing values with 0
            #usecols = (0,2,3,5),    # columns to read
            names=names_list) # column names

        self.pl_names = np.genfromtxt(
            input_planets,           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype=str,        # data type
            filling_values=-1.000,       # fill missing values with 0
            usecols = (0) )    #

        self.textbox_ha = np.genfromtxt(
            input_planets,           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype=str,        # data type
            filling_values=-1.000,       # fill missing values with 0
            usecols = (12) )    #

        self.textbox_va = np.genfromtxt(
            input_planets,           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype=str,        # data type
            filling_values=-1.000,       # fill missing values with 0
            usecols = (13) )    #

        # this didn't work

        #self.pl_names = string_nasa['name']
        #self.textbox_ha = string_nasa['textbox_ha']
        #self.textbox_va = string_nasa['textbox_va']

        self.pl_orbper = data_input['orbital_period']
        self.st_rad = data_input['star_radius']
        self.st_teff = data_input['star_teff']
        self.st_mass = data_input['star_mass']
        self.pl_mass = data_input['mass']
        self.pl_mass_error_max = data_input['mass_error_max']
        self.pl_mass_error_min = np.abs(data_input['mass_error_min'])
        self.pl_radius = data_input['radius']
        self.pl_radius_error_max = data_input['radius_error_max']
        self.pl_radius_error_min = np.abs(data_input['radius_error_min'])
        self.mass_detection_type = np.ones(np.size(self.pl_orbper))
        self.pl_ttvflag = data_input['pl_ttvflag']
        self.pl_upper_limit = data_input['pl_upper_limit']

<<<<<<< HEAD

        self.fix_TTVs()
=======
>>>>>>> 294d8892ec0b362ed7bcc8280d22867ee64e1211
        self.compute_derived_parameters()


class Dataset_Combined(Dataset):
    def __init__(self):

        data_eu = pandas.read_csv('./Exoplanets_eu/exoplanet.eu_catalog.csv')

        data_nasa = np.genfromtxt(
            './NASA_data/defaults_radec.csv',           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype='float32',        # data type
            filling_values=0.00000000,       # fill missing values with 0
            #usecols = (0,2,3,5),    # columns to read
            names=['name','orbital_period','mass','mass_error_max','mass_error_min','radius','radius_error_max','radius_error_min','star_mass','star_radius','star_teff','pl_ttvflag','ra','dec'])     # column names

        names_nasa = np.genfromtxt(
            './NASA_data/defaults_radec.csv',           # file name
            skip_header=1,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype=str,        # data type
            filling_values=-1.000,       # fill missing values with 0
            usecols = (0))    # columns to read

        n_planets = len(data_eu['# name'])

        parameters_list = ['mass', 'mass_error_min', 'mass_error_max', \
                           'radius', 'radius_error_min', 'radius_error_max', \
                           'star_radius', 'star_teff', 'star_mass', 'orbital_period', 'pl_ttvflag']

        data_combined = {}

        data_combined['name'] = data_eu['# name'].values
        data_combined['mass_detection_type'] = data_eu['mass_detection_type'].values

        for key in parameters_list:
            data_combined[key] = np.zeros(n_planets, dtype=np.double) - 0.0001

        factor_dict = {'mass':317.83, 'mass_error_min':317.83, 'mass_error_max':317.83, \
            'radius':11.209, 'radius_error_min':11.209, 'radius_error_max':11.209, \
            'star_radius':1.00000, 'star_teff':1.00000, 'star_mass':1.00000, 'orbital_period':1.00000, 'pl_ttvflag':0.00000 }

        ii = 0
        for name_i, name_val in enumerate(data_eu['# name']):
            index = np.where(names_nasa==name_val)
            ind = -1
            found = False

            if np.size(index) >0:
                ind = index[0]
            else:
                #dist = (data_eu['ra'][name_i] -  data_nasa['ra'])**2 + (data_eu['dec'][name_i] -  data_nasa['dec'])**2
                #cos(A) = sin(Decl.1)sin(Decl.2) + cos(Decl.1)cos(Decl.2)cos(RA.1 - RA.2) and thus, A = arccos(A)
                try:

                    cos_A = np.sin(data_eu['dec'][name_i]/180.0*np.pi) * np.sin(data_nasa['dec']/180.0*np.pi) + \
                        np.cos(data_eu['dec'][name_i]/180.0*np.pi) * np.cos(data_nasa['dec']/180.0*np.pi) * np.cos(data_eu['ra'][name_i]/180.0*np.pi - data_nasa['ra']/180.0*np.pi)

                    A = np.arccos(cos_A)

                    ind_sort = np.argsort(A)

                    ind_where = np.where( np.abs(data_eu['orbital_period'][name_i] - data_nasa['orbital_period'][ind_sort[:8]]) < 0.1)[0]
                    if np.size(ind_where)  > 0:
                        ind = ind_sort[ind_where[0]]
                except:
                    print

            if ind >= 0:

                for key in parameters_list:
                    try:
                        if np.abs(data_nasa[key][ind])>0.00001:
                            data_combined[key][name_i] = np.abs(data_nasa[key][ind]).copy()
                        elif data_eu[key][name_i]>0.00000000001:
                            data_combined[key][name_i] = data_eu[key][name_i].copy() * factor_dict[key]
                        #print data_nasa[key][ind], data_eu[key][name_i], data_combined[key][name_i]
                    except:
                        continue

        sel = (data_combined['mass'] > 0.2) & (data_combined['radius'] > 0.2) & (data_combined['orbital_period'] > 0.01) & \
            (data_combined['star_mass']>0.0) & (data_combined['star_radius']>0.) & (data_combined['star_teff']>0.) & \
            (data_combined['radius_error_min']/data_combined['radius']<0.99 ) & (data_combined['radius_error_max']/data_combined['radius']<0.99 ) & \
            (data_combined['mass_error_min']/data_combined['mass']<0.99 ) & (data_combined['mass_error_max']/data_combined['mass']<0.99 ) & \
            (data_combined['mass_error_min']>0.0) & (data_combined['mass_error_max']>0.0) & (data_combined['radius_error_min']>0.0) & (data_combined['radius_error_max']>0.0)

        self.pl_names = data_combined['name'][sel]
        self.pl_orbper = data_combined['orbital_period'][sel]
        self.st_rad = data_combined['star_radius'][sel]
        self.st_teff = data_combined['star_teff'][sel]
        self.st_mass = data_combined['star_mass'][sel]
        self.pl_mass = data_combined['mass'][sel]
        self.pl_mass_error_max = data_combined['mass_error_max'][sel]
        self.pl_mass_error_min = data_combined['mass_error_min'][sel]
        self.pl_radius = data_combined['radius'][sel]
        self.pl_radius_error_max = data_combined['radius_error_max'][sel]
        self.pl_radius_error_min = data_combined['radius_error_min'][sel]
        self.mass_detection_type = data_combined['mass_detection_type'][sel]
        self.pl_ttvflag = data_combined['pl_ttvflag'][sel]

<<<<<<< HEAD
        self.fix_TTVs()
=======
>>>>>>> 294d8892ec0b362ed7bcc8280d22867ee64e1211
        self.compute_derived_parameters()


class Dataset_ExoplanetEU(Dataset):
    def __init__(self):

        data_eu = pandas.read_csv('./Exoplanets_eu/exoplanet.eu_catalog.csv')

        n_planets = len(data_eu['# name'])

        data_eu['name'] = data_eu['# name']

        data_eu['mass_error_min'] *= M_jup_to_ear
        data_eu['mass_error_max'] *= M_jup_to_ear
        data_eu['mass'] *= M_jup_to_ear

        data_eu['radius'] *= R_jup_to_ear
        data_eu['radius_error_min'] *= R_jup_to_ear
        data_eu['radius_error_max'] *= R_jup_to_ear

        sel = (data_eu['mass'] > 0.2) & (data_eu['radius'] > 0.2) & (data_eu['orbital_period'] > 0.01) & \
            (data_eu['star_mass']>0.0) & (data_eu['star_radius']>0.) & (data_eu['star_teff']>0.) & \
            (data_eu['radius_error_min']/data_eu['radius']<0.99 ) & (data_eu['radius_error_max']/data_eu['radius']<0.99 ) & \
            (data_eu['mass_error_min']/data_eu['mass']<0.99 ) & (data_eu['mass_error_max']/data_eu['mass']<0.99) & \
            (data_eu['mass_error_min']>0.0) & (data_eu['mass_error_max']>0.0) & (data_eu['radius_error_min']>0.0) & (data_eu['radius_error_max']>0.0)

        self.pl_names = data_eu['# name'][sel].values
        self.pl_orbper = data_eu['orbital_period'][sel].values
        self.st_rad = data_eu['star_radius'][sel].values
        self.st_teff = data_eu['star_teff'][sel].values
        self.st_mass = data_eu['star_mass'][sel].values
        self.pl_mass = data_eu['mass'][sel].values
        self.pl_mass_error_max = data_eu['mass_error_max'][sel].values
        self.pl_mass_error_min = data_eu['mass_error_min'][sel].values
        self.pl_radius = data_eu['radius'][sel].values
        self.pl_radius_error_max = data_eu['radius_error_max'][sel].values
        self.pl_radius_error_min = data_eu['radius_error_min'][sel].values
        self.mass_detection_type = data_eu['mass_detection_type'][sel].values
        self.pl_ttvflag = [True if ttv_flag == 'TTV' else False for ttv_flag in data_eu['mass_detection_type'][sel].values]

<<<<<<< HEAD
        self.fix_TTVs()
=======
>>>>>>> 294d8892ec0b362ed7bcc8280d22867ee64e1211
        self.compute_derived_parameters()
