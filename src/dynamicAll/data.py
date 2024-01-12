
import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.table import Table
from astroquery.simbad import Simbad

Simbad.add_votable_fields('plx', 'distance')  
from jax._src.config import config
config.update("jax_enable_x64", True)  
import warnings
from sklearn.mixture import GaussianMixture as GMM
import pynbody
from .base import Data
import matplotlib.pyplot as plt

class KeckData(Data):
    def __init__(self,table,galaxy:SkyCoord=None,unit=[u.deg,u.deg]):
        r'''
        Subclass of Data class used to handle observations from Keck-DEIMOS

        Parameters
        ----------
        table : 
            any table where i can access things by name
            e.g table['RA'],table['DEC'] etc.
            Currently I am using the naming convention from Marla's latest schema
        galaxy : SkyCoord
            SkyCoord object must have a distance.
            See from_name method for initializing the class using astroquery
        unit : list, optional
            units for RA and DEC, by default [u.deg,u.deg]

        NOTES
        -----
        I will not be attempting to write a function to read the data table -- I will leave that to the user.

        TODO: Place the stars at some distance that is not just the distance associated with the galaxy
        TODO: I should make sure the units are idiot (me) proof somehow
        TODO: Ask Marla about names and what not
        '''
        ###### from Marla's latest schema #########
        self.table    = table                           # Don't think we actually need this, but i'll keep i around for now
        self._prob_m  = table['Pmem_pure']              # membership probabilities
        
        self.member     = np.where(self._prob_m  ==1)[0]        # use probabilities to get rid of everything that is not a member
        self.non_member = np.where(self._prob_m  !=1)[0]

        self._mag     = table['gmag_o'][self.member]    # g-band magnitude
        self._R       = table['rproj_kpc'][self.member] # projected radii in kpc
        self._vlos    = table['v'][self.member]         # line of sight velocities in km/s
        self.d_vlos   = table['v_err'][self.member]     # line of sight velocity errors in km/s


        # proper motions: originally in mas/yr, but I'm converting to km/s
        self._pmra    = table['gaia_pmra'][self.member]  * 4.74  # mas/yr to km/s
        self._pmdec   = table['gaia_pmdec'][self.member] * 4.74 # mas/yr to km/s

        # Get Data we need from the data table
        self.table  = table                    # Don't think we actually need this, but i'll keep i around for now
        self.ra     = table['RA'][self.member]  *unit[0]    # RA of all stars 
        self.dec    = table['DEC'][self.member] *unit[1]    # DEC of all stars 
        self.N_star = len(self.ra)


        ###########################################


        # I'm assuming that the galaxy is already a SkyCoord object, but maybe I should do a check?
        # self.galaxy = galaxy

        # self.pos    = SkyCoord(self.ra,self.dec, frame='icrs')
        # # project radii in radians
        # self.Rrad      = np.array(self.pos.separation(galaxy).to(u.rad))
        
        # # projected radi in kpc -- i'll do all the analysis in kpc
        # self._R  =2* jnp.tan(self.Rrad/2) * self.galaxy.distance *self.galaxy.distance.unit
        # # I should be able to do this using astropy -- but for now this is fine

        # self._vlos  = jnp.array(table['v']) * u.km/u.s
        # self.d_vlos = jnp.array(table['v_err']) * u.km/u.s

        # self.bin_edges={}
        # # self.pos    = SkyCoord(self.ra,self.dec, distance=galaxy.distance[0], frame='icrs')
        # # self.x = self.galaxy.cartesian.x - self.pos.cartesian.x 
        # # self.y = self.galaxy.cartesian.y - self.pos.cartesian.y
        # # self.z = self.galaxy.cartesian.z - self.pos.cartesian.z
        # # self.pos_cartesian = self.pos.cartesian
        # # # self.r      = self.pos.separation_3d(galaxy)

    def plot_pos(self,plot_all=True):
        r'''
        Plot the positions (RA and DEC) of the stars in the data set

        Parameters
        ----------
        plot_all : bool, optional
            If True, plot all stars in the data set, and label members and non members. By default True
        
        Returns
        -------
        fig,ax : matplotlib.pyplot.figure, matplotlib.pyplot.axis
            figure and axis objects in case you want to do more with the plot
        '''
        fig,ax = plt.subplots(figsize=(10,8))
        if plot_all:
            ax.scatter(self.table['RA'],self.table['DEC'], # first plot all objects in table
                    marker='s',
                    alpha=1.0,
                    color='black',
                    label='all',
                    facecolor='none',
            )
            ax.scatter(self.ra,self.dec, # plot members 
                    alpha=0.8,
                    marker='+',
                    label='members',
            )
            ax.scatter(self.table['RA'][self.non_member],self.table['DEC'][self.non_member],
                    marker='.',
                    # alpha=1,
                    # facecolor='none',
                    label='non-members',
            )
        else:
            ax.scatter(self.ra,self.dec, # plot members 
            alpha=0.8,
            label='members',
            )

        ax.set(
            xlabel='RA',
            ylabel='DEC'
        )
        ax.legend()
        return fig,ax 

    @staticmethod
    def from_name(table,name):
        '''
        So that I dont have to keep looking up coordinates for all objects

        Parameters
        ----------
        table : _type_
            _description_
        name : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        Some of the names have "NAME" in front of them e.g. Draco: NAME Dra dSph
        The naming comvention is all over the place so I guess I should deala with that
        '''

        galaxies = Table.read('/Users/juan/phd/projects/dynamicAll/src/data/simbad.xml')
        df       = galaxies['MAIN_ID','RA_d','DEC_d','Distance_distance','Distance_unit'].to_pandas()
        # first try to see if the galaxy name is cached in a pre-prepared file
        try: 
            temp = df.loc[df['MAIN_ID'] ==name]
            ra   = np.float64(temp['RA_d'] ) * u.deg
            dec  = np.float64(temp['DEC_d']) * u.deg
            D    = np.float64(temp['Distance_distance']) * u.Unit(temp['Distance_unit'])
            galaxy = SkyCoord(ra,dec,distance=D.to(u.kpc), frame='icrs')
            return KeckData(table,galaxy)
        except:
            warnings.warn("\n Something went wrong, the Name provided was not in our database,",stacklevel=2)
        
        # Try Adding "Name" to the beginning of name.. cause that works sometimes
        try: 
            temp = df.loc[df['MAIN_ID'] =='Name '+name]
            ra   = np.float64(temp['RA_d'] ) * u.deg
            dec  = np.float64(temp['DEC_d']) * u.deg
            D    = np.float64(temp['Distance_distance']) * u.Unit(temp['Distance_unit'])
            galaxy = SkyCoord(ra,dec,distance=D.to(u.kpc), frame='icrs')
            return KeckData(table,galaxy)
        except:
            warnings.warn("\n We tried changing the name -- Last attempt will try to use simbad.query_object to search for object",stacklevel=2)
        # Last attempt: check to see if you can use simbad to find the object
        try: 
            temp= Simbad.query_object(name)
            galaxy = SkyCoord(temp['RA'][0],temp['DEC'][0],unit=(u.hourangle,u.deg),distance=(temp['Distance_distance']* u.Unit(temp['Distance_unit'][0])).to(u.kpc))
            print('success using simbad!')
            return KeckData(table,galaxy)
        except:
            warnings.warn("\n Something went wrong, data set was not initialized properly.",stacklevel=2)

class DCJLData(Data):
    '''
    Class for data from DCJL simulations
    '''
    
    def __init__(self,halo):
        '''
        
        Notes
        -----
        TODO: add a function to test virial theorem
        '''
        self.halo = halo
        
        #set everything to physical units
        # self.halo.physical_units()
        # center on the halo
        # pynbody.analysis.halo.center_of_mass(self.halo)
        # make sure there are stars in the halo
        # print('ngas = %e, ndark = %e, nstar = %e\n'%(len(self.halo.gas),len(self.halo.dark),len(self.halo.star)))

        # sort all data from halo that I need
        # Cartesian
        self._x = self.halo.s['x']
        self._y = self.halo.s['y']
        self._z = self.halo.s['z']
        
        self._vx = self.halo.s['vx']
        self._vy = self.halo.s['vy']
        self._vz = self.halo.s['vz']

        # Spherical
        # self.r = self.halo.s['r']   
        # self._vr     = self.halo.s['vr']    
        # self._vtheta = self.halo.s['vtheta']
        # self._vphi   = self.halo.s['vphi']  

        # `Projected` -- choose 'z' direction to be line-of-sight
        self.R    = jnp.sqrt(self.halo.s['x']**2 + self.halo.s['y']**2)
        
        self._vlos = self.halo.s['z']

        # Additional information
        self._feh    = self.halo.s['feh']   
        self._tForm  = self.halo.s['tform']

        super().__init__()
        
    def mass_bspline(self):
        return -1

    def random_rotation(self):

        self.halo.rotate_x(np.random.uniform(-90,90))
        self.halo.rotate_y(np.random.uniform(-90,90))
        self.halo.rotate_z(np.random.uniform(-90,90))

    def split_feh(self,nComponents: int ,fehCutOff:float = -4):
        '''
        split sample of stars into multiple populations using a Gaussian Mixture Model
        using the metallicity of each star particle in the snapshot


        This really only works for 2 components.

        Parameters
        ----------
        nComponents : int
        number of gaussians to fit to data
        fehCutOff : float, optional
            cut off metallicities at some point, by default -4

        Returns
        -------
        _type_
            _description_
        '''
        if fehCutOff != -np.inf:
            nGarbage = len(self._data.s['feh'][self._data.s['feh'] <= fehCutOff])
            warnings.warn(f"\nMetallicity cut off is removing {nGarbage} stars from data set")
        
        feh    = self._data.s['feh'][self._data.s['feh'] > fehCutOff]

        clf    = GMM(n_components=nComponents).fit(np.array([feh]).T)
        out = clf.predict(np.array([feh]).T)
        means = clf.means_
        where_0 = np.where(out == 0)
        where_1 = np.where(out == 1)

        if means[0,0] > means[1,0]:
            # out = 1 - out # this is more clever but only works for two samples.
            out[where_0] = 1
            out[where_1] = 0
        self._labels['feh'] = out
        # out = 1 - out
        
        return self._labels['feh']


    def Lum(sim,band='v'):
        sun_abs_mag = {'u':5.56,'b':5.45,'v':4.8,'r':4.46,'i':4.1,'j':3.66,
                        'h':3.32,'k':3.28}[band]
        return 10.0 ** ((sun_abs_mag - sim.star[band + '_mag']) / 2.5)

    

class MockData(Data):
    def __init__(self,dataSet):
        '''
        _summary_

        Parameters
        ----------
        dataSet : _type_
            _description_
        Notes
        -----
        I'm not sure if i loveee the notation here, but there arent that many options
        '''
        
        # Cartesian
        self._x = jnp.array(dataSet['x'])
        self._y = jnp.array(dataSet['y'])
        self._z = jnp.array(dataSet['z'])
        
        self._vx = jnp.array(dataSet['vx'])
        self._vy = jnp.array(dataSet['vy'])
        self._vz = jnp.array(dataSet['vz'])
        self.N_star = len(self._x)

        try:
            vel_error = jnp.array(dataSet['error'])
        except:
            # set all of them to None        self._vlos   = jnp.array(dataSet['z'])
            self. d_vlos = None
            self.d_vr   = None
            self.d_vtheta = None
            self.d_vphi = None
            self.d_pmr = None
            self.d_pmt = None


        # Spherical
        try:
            self._r      = jnp.array(dataSet['r'])   
            self._vr     = jnp.array(dataSet['vr'])    
            self._vtheta = jnp.array(dataSet['vtheta'])
            self._vphi   = jnp.array(dataSet['vphi'])  
        except:
            self.spherical() #this will define the spherical components as above 
            # also defines self._R

        # proper motions
        try:
            self._pmr     = jnp.array(dataSet['pmr'])
            self._pmt     = jnp.array(dataSet['pmt'])
        except:
            self.cylindrical()
        
        # `Projected' -- choose 'z' direction to be line-of-sight
        self._vlos   = jnp.array(dataSet['vz'])
        
        self. d_vlos = jnp.array(dataSet['error'])
        
        self.d_vr   = jnp.sqrt(jnp.array(3*dataSet['error']**2))
        self.d_vtheta = self.d_vr
        self.d_vphi = self.d_vr
        self.d_pmr = self.d_vr
        self.d_pmt = self.d_vr

        # errors on observations
        self._error = jnp.array(dataSet['error'])

        super().__init__()


