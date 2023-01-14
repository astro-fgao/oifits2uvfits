# vlbi_imaging_utils.py
# Andrew Chael, 10/15/2015
# Utilities for generating and manipulating VLBI images, datasets, and arrays

# TODO:
#       Fix save_uvfits
#       Add non-circular errors
#       Add closure amplitude debiasing
#       Add different i,q,u,v SEFDs and calibration errors?
#       Incorporate Katherine's scattering code?

import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import itertools as it
import astropy.io.fits as fits
import datetime
import writeData
import oifits_new as oifits
import time as ttime
import pulses
#from mpl_toolkits.basemap import Basemap # for plotting baselines on globe

##################################################################################################
# Constants
##################################################################################################
EP = 1.0e-10
C = 299792458.0
DEGREE = np.pi/180.
RADPERAS = DEGREE/3600
RADPERUAS = RADPERAS/1e6

# Telescope elevation cuts (degrees)
ELEV_LOW = 15.0
ELEV_HIGH = 85.0

# Default Optical Depth and std. dev % on gain
TAUDEF = 0.1
GAINPDEF = 0.1

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = 78 # in degree, E of N

# Observation recarray datatypes
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8'), ('sefd','f8')]

DTPOL = [('time','f8'),('tint','f8'),
         ('t1','a32'),('t2','a32'),
         ('el1','f8'),('el2','f8'),
         ('tau1','f8'),('tau2','f8'),
         ('u','f8'),('v','f8'),
         ('vis','c16'),('qvis','c16'),('uvis','c16'),('sigma','f8')]

DTBIS = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),
         ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
         ('bispec','c16'),('sigmab','f8')]
                                             
DTCPHASE = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),
            ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
            ('cphase','f8'),('sigmacp','f8')]
            
DTCAMP = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),('t4','a32'),
          ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),
          ('u3','f8'),('v3','f8'),('u4','f8'),('v4','f8'),
          ('camp','f8'),('sigmaca','f8')]

# Observation fields for plotting and retrieving data
FIELDS = ['time','tint','u','v','uvdist',
          't1','t2','el1','el2','tau1','tau2',
          'vis','amp','phase','snr','sigma',
          'qvis','qamp','qphase','qsnr',
          'uvis','uamp','uphase','usnr',
          'pvis','pamp','pphase',
          'm','mamp','mphase']
                  
##################################################################################################
# Classes
##################################################################################################

class Image(object):
    """A radio frequency image array (in Jy/pixel).
    
    Attributes:
        pulse: The function convolved with pixel value dirac comb for continuous image rep. (function from pulses.py)
        psize: The pixel dimension in radians (float)
        xdim: The number of pixels along the x dimension (int)
        ydim: The number of pixels along the y dimension (int)
        ra: The source Right Ascension (frac hours)
        dec: The source Declination (frac degrees)
        rf: The radio frequency (Hz)
        imvec: The xdim*ydim vector of jy/pixel values (array)
        source: The astrophysical source name (string)
        mjd: The mjd of the image
    """
    
    def __init__(self, image, psize, ra, dec, rf=230e9, pulse=pulses.trianglePulse2D, source="SgrA", mjd="0"):
        if len(image.shape) != 2:
            raise Exception("image must be a 2D numpy array")
        
        self.pulse = pulse
        self.psize = float(psize)
        self.xdim = image.shape[1]
        self.ydim = image.shape[0]
        self.imvec = image.flatten()
                
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = float(mjd)
        
        self.qvec = []
        self.uvec = []
        
    def add_qu(self, qimage, uimage):
        """Add Q and U images
        """
        
        if len(qimage.shape) != len(uimage.shape):
            raise Exception("image must be a 2D numpy array")
        if qimage.shape != uimage.shape != (self.ydim, self.xdim):
            raise Exception("Q & U image shapes incompatible with I image!")
        self.qvec = qimage.flatten()
        self.uvec = uimage.flatten()
    
    def copy(self):
        """Copy the image object"""
        newim = Image(self.imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)
        if len(self.qvec):
            newim.add_qu(self.qvec.reshape(self.ydim,self.xdim), self.uvec.reshape(self.ydim,self.xdim))
        return newim
        
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis)
        """
        self.qvec = - self.qvec
        return
           
    def observe_same(self, obs, sgrscat=False):
        """Observe the image on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
        """
        
        # Check for agreement in coordinates and frequency
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
        if (self.rf != obs.rf):
            raise Exception("Image frequency is not the same as observation frequency!")
        
        # Get data
        obslist = obs.tlist()
        
        # Remove possible conjugate baselines:
        obsdata = []
        blpairs = []
        for tlist in obslist:
            for dat in tlist:
                if not ((dat['t1'], dat['t2']) in blpairs
                     or (dat['t2'], dat['t1']) in blpairs):
                     obsdata.append(dat)
                     
        obsdata = np.array(obsdata, dtype=DTPOL)
                          
        # Extract uv data
        uv = obsdata[['u','v']].view(('f8',2))
           
        # Perform DFT
        mat = ftmatrix(self.psize, self.xdim, self.ydim, uv, pulse=self.pulse)
        vis = np.dot(mat, self.imvec)
        
        # If there are polarized images, observe them:
        qvis = np.zeros(len(vis))
        uvis = np.zeros(len(vis))
        if len(self.qvec):
            qvis = np.dot(mat, self.qvec)
            uvis = np.dot(mat, self.uvec)
        
        # Scatter the visibilities with the SgrA* kernel
        if sgrscat:
            print('Scattering Visibilities with Sgr A* kernel!')
            for i in range(len(vis)):
                ker = sgra_kernel_uv(self.rf, uv[i,0], uv[i,1])
                vis[i]  *= ker
                qvis[i] *= ker
                uvis[i] *= ker
   
        # Put the visibilities back in the obsdata array
        obsdata['vis'] = vis
        obsdata['qvis'] = qvis
        obsdata['uvis'] = uvis
        
        # Return observation object
        obs_no_noise = Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata, obs.tarr, source=self.source, mjd=self.mjd)
        return obs_no_noise
        
    def observe(self, array, tint, tadv, tstart, tstop, bw, tau=TAUDEF, gainp=GAINPDEF, opacity_errs=True, ampcal=True, phasecal=True, sgrscat=False):
        """Observe the image with an array object to produce an obsdata object.
           tstart and tstop should be hrs in GMST.
           tint and tadv should be seconds.
           tau is the estimated optical depth. This can be a single number or a dictionary giving one tau per site
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel at the appropriate frequency
        """
        
        # Generate empty observation
        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, tau=tau, opacity_errs=opacity_errs)
        
        # Observe
        obs = self.observe_same(obs, sgrscat=sgrscat)
        
        # Add noise
        obs = add_noise(obs, opacity_errs=opacity_errs, ampcal=ampcal, phasecal=phasecal, gainp=gainp)
        
        return obs
        
    def display(self, cfun='afmhot', nvec=20, pcut=0.01, plotp=False, interp='nearest'):
        """Display the image with matplotlib
        """
        
        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'nearest'
            
        plt.figure()
        plt.clf()
        
        image = self.imvec;
        
        if len(self.qvec) and plotp:
            thin = self.xdim/nvec
            mask = (self.imvec).reshape(self.ydim, self.xdim) > pcut * np.max(self.imvec)
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = (np.cos(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

            m = (np.abs(self.qvec + 1j*self.uvec)/self.imvec).reshape(self.ydim, self.xdim)
            m[-mask] = 0
            
            plt.suptitle('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
            
            # Stokes I plot
            plt.subplot(121)
            im = plt.imshow(image.reshape(self.ydim, self.xdim), cmap=plt.get_cmap(cfun), interpolation=interp, vmin=0)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
            plt.title('Stokes I')
        
            # m plot
            plt.subplot(122)
            im = plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation=interp, vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='|m|')
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
            plt.title('m (above %0.2f max flux)' % pcut)
        
        else:
            plt.subplot(111)
            plt.title('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
            
            im = plt.imshow(image.reshape(self.ydim,self.xdim), cmap=plt.get_cmap(cfun), interpolation=interp)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
        
        plt.show(block=False)
            
    def save_txt(self, fname):
        """Save image data to text file"""
        
        # Coordinate values
        pdimas = self.psize/RADPERAS
        xs = np.array([[j for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        xs = pdimas * (xs[::-1] - self.xdim/2.0)
        ys = np.array([[i for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        ys = pdimas * (ys[::-1] - self.xdim/2.0)
        
        # Data
        if len(self.qvec):
            outdata = np.hstack((xs, ys, (self.imvec).reshape(self.xdim*self.ydim, 1),
                                         (self.qvec).reshape(self.xdim*self.ydim, 1),
                                         (self.uvec).reshape(self.xdim*self.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

            fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"
        else:
            outdata = np.hstack((xs, ys, (self.imvec).reshape(self.xdim*self.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)"
            fmts = "%10.10f %10.10f %10.10f"
     
        # Header
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %.4f \n" % self.mjd +
                    "RF: %.4f GHz \n" % (self.rf/1e9) +
                    "FOVX: %i pix %f as \n" % (self.xdim, pdimas * self.xdim) +
                    "FOVY: %i pix %f as \n" % (self.ydim, pdimas * self.ydim) +
                    "------------------------------------\n" + hf)
         
        # Save
        np.savetxt(fname, outdata, header=head, fmt=fmts)

    def save_fits(self, fname):
        """Save image data to FITS file"""
                
        # Create header and fill in some values
        header = fits.Header()
        header['OBJECT'] = self.source
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CDELT1'] = -self.psize/DEGREE
        header['CDELT2'] = self.psize/DEGREE
        header['OBSRA'] = self.ra * 180/12.
        header['OBSDEC'] = self.dec
        header['FREQ'] = self.rf
        header['MJD'] = self.mjd
        header['TELESCOP'] = 'VLBI'
        header['BUNIT'] = 'JY/PIXEL'
        header['STOKES'] = 'I'
        
        # Create the fits image
        image = np.reshape(self.imvec,(self.ydim,self.xdim))[::-1,:] #flip y axis!
        hdu = fits.PrimaryHDU(image, header=header)
        if len(self.qvec):
            qimage = np.reshape(self.qvec,(self.xdim,self.ydim))[::-1,:]
            uimage = np.reshape(self.uvec,(self.xdim,self.ydim))[::-1,:]
            header['STOKES'] = 'Q'
            hduq = fits.ImageHDU(qimage, name='Q', header=header)
            header['STOKES'] = 'U'
            hduu = fits.ImageHDU(uimage, name='U', header=header)
            hdulist = fits.HDUList([hdu, hduq, hduu])
        else: hdulist = fits.HDUList([hdu])
      
        # Save fits
        hdulist.writeto(fname, clobber=True)
        
        return
                
##################################################################################################
class Array(object):
    """A VLBI array of telescopes with locations and SEFDs
    
        Attributes:
        tarr: The array of telescope data (name, x, y, z, SEFD) where x,y,z are geocentric coordinates.
    """
    
    def __init__(self, tarr):
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        # !AC better way?
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
            
    def listbls(self):
        """List all baselines"""
 
        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])
                    
        return np.array(bls)
            
    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=0.0, tau=TAUDEF, opacity_errs=True):
        """Generate u,v points and baseline errors for the array.
           Return an Observation object with no visibilities.
           tstart and tstop are hrs in GMST
           tint and tadv are seconds.
           rf and bw are Hz
           ra is fractional hours
           dec is fractional degrees
           tau can be a single number or a dictionary giving one per site
        """
        
        if mjdtogmt(mjd)-tstart > 1e-9: #!AC time!
            raise Exception("Initial time is greater than given mjd!")
            
        # Set up coordinate system
        sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
        projU = np.cross(np.array([0,0,1]), sourcevec)
        projU = projU / np.linalg.norm(projU)
        projV = -np.cross(projU, sourcevec)
        
        # Set up time start and steps
        tstep = tadv/3600.0
        if tstop < tstart:
            tstop = tstop + 24.0;
        
        # Wavelength
        l = C/rf
        
        # Observing times
        times = np.arange(tstart, tstop, tstep)

        # Generate uv points at all times
        outlist = []
        for k in range(len(times)):
            time = times[k]
            theta = np.mod((time-ra)*360./24, 360)
            blpairs = []
            for i1 in range(len(self.tarr)):
                for i2 in range(len(self.tarr)):
                    coord1 = np.array((self.tarr[i1]['x'], self.tarr[i1]['y'], self.tarr[i1]['z']))
                    coord2 = np.array((self.tarr[i2]['x'], self.tarr[i2]['y'], self.tarr[i2]['z']))
                    if (i1!=i2 and
                        self.tarr[i1]['z'] <= self.tarr[i2]['z'] and # Choose the north one first
                        not ((i2, i1) in blpairs) and # This cuts out the conjugate baselines
                        elevcut(earthrot(coord1, theta),sourcevec) and
                        elevcut(earthrot(coord2, theta),sourcevec)):
                        
                        # Optical Depth
                        if type(tau) == dict:
                            try:
                                tau1 = tau[i1]
                                tau2 = tau[i2]
                            except KeyError:
                                tau1 = tau2 = TAUDEF
                        else:
                            tau1 = tau2 = tau
                        
                        # Append data to list
                        blpairs.append((i1,i2))
                        outlist.append(np.array((
                                  time,
                                  tint, # Integration
                                  self.tarr[i1]['site'], # Station 1
                                  self.tarr[i2]['site'], # Station 2
                                  elev(earthrot(coord1, theta),sourcevec), # Station 1 elevation
                                  elev(earthrot(coord2, theta),sourcevec), # Station 2 elevation
                                  tau1, # Station 1 optical depth
                                  tau2, # Station 1 optical depth
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projU), # u (lambda)
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projV), # v (lambda)
                                  0.0, 0.0, 0.0, # Stokes I, Q, U visibilities (Jy)
                                  blnoise(self.tarr[i1]['sefd'], self.tarr[i2]['sefd'], tint, bw) # Sigma (Jy)
                                ), dtype=DTPOL
                                ))
        obsarr = np.array(outlist)
         
        if not len(obsarr):
            raise Exception("No mutual visibilities in the specified time range!")
            
        # Elevation dependence on noise using estimated opacity
        if opacity_errs:
            elevs = obsarr[['el1','el2']].view(('f8',2))
            taus = obsarr[['tau1','tau2']].view(('f8',2))
            obsarr['sigma'] *= np.sqrt(np.exp(taus[:,0]/(EP+np.sin(elevs[:,0]*DEGREE)) + taus[:,1]/(EP+np.sin(elevs[:,1]*DEGREE))))
        
        # Return
        obs = Obsdata(ra, dec, rf, bw, np.array(outlist), self.tarr, source=str(ra) + ":" + str(dec), mjd=mjd, ampcal=True, phasecal=True)
        return obs
     
    def save_array(self, fname):
        """Save the array data in a text file"""
         
        out = ""
        for scope in range(len(self.tarr)):
            dat = (self.tarr[scope]['site'],
                   self.tarr[scope]['x'], self.tarr[scope]['y'],
                   self.tarr[scope]['z'], self.tarr[scope]['sefd']
                  )
            out += "%-8s %15.5f  %15.5f  %15.5f  %6.4f \n" % dat
        f = open(fname,'w')
        f.write(out)
        f.close()
        return
         
#    def plotbls(self):
#        """Plot all baselines on a globe"""
#
#        lat = []
#        lon = []
#        for t1 in range(len(tarr)):
#            (x,y,z) = (self.tarr[t1]['x'], self.tarr[t1]['y'], self.tarr[t1]['z'])
#            lon.append(np.arctan2(y, x)/DEGREE)
#            lat.append(90 - np.arccos(z/np.sqrt(x**2 + y**2 + z**2))/DEGREE)

#        map = Basemap(projection='moll', lon_0=-90)
#        map.drawmapboundary(fill_color='blue')
#        map.fillcontinents(color='green', lake_color='blue')
#        map.drawcoastlines()
#        for i in range(len(lon)):
#            for j in range(len(lon)):
#                x,y = map([lon[i],lon[j]], [lat[i],lat[j]])
#                map.plot(x, y, marker='D', color='r')
#
#        plt.show()
        
##################################################################################################
class Obsdata(object):
    """A VLBI observation of visibility amplitudes and phases.
    
       Attributes:
        source: the source name
        ra: the source right ascension (frac. hours)
        dec: the source declination (frac. degrees)
        mjd: the observation start date
        tstart: the observation start time (GMT, frac. hr.)
        tstop: the observation end time (GMT, frac. hr.)
        rf: the observing frequency (Hz)
        bw: the observing bandwidth (Hz)
        ampcal: amplitudes calibrated T/F
        phasecal: phases calibrated T/F
        data: recarray with the data (time, t1, t2, tint, u, v, vis, qvis, uvis, sigma)
    """
    
    def __init__(self, ra, dec, rf, bw, datatable, tarr, timemjd, dateobs, ins, arrname, nwav, source="SgrA", mjd=0, ampcal=True, phasecal=True):
        
        print("mjd ori")
        print(mjd)
        if len(datatable) == 0:
            raise Exception("No data in input table!")
        if (datatable.dtype != DTPOL):
            raise Exception("Data table should be a recarray with datatable.dtype = %s" % DTPOL)
        
        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        #self.rf = float(rf)
        #self.bw = float(bw)
        self.rf = rf
        self.bw = bw
        self.ampcal = bool(ampcal)
        self.phasecal = bool(phasecal)
        self.tarr = tarr
        self.timemjd = timemjd
        self.nwav = nwav
        self.ins = ins
        self.datatable = datatable # added by myself
        self.arrname = arrname
        self.dateobs = dateobs
        
        # Dictionary of array indices for site names
        # !AC better way?
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        
        # Time partition the datatable
        datalist = []
        for key, group in it.groupby(datatable, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        # Remove conjugate baselines
        # Make the north site first in each pair
        obsdata = []
#        print 'Obsdata loop: ',len(datalist)
        for tlist in datalist:
            blpairs = []
#            print 'Obsdata loop: ',len(tlist)
            for dat in tlist:
#                if not (set((dat['t1'], dat['t2']))) in blpairs:
                     # Reverse the baseline if not north
#                print 'Obsdata loop: ',dat['vis'].shape
                if (self.tarr[self.tkey[dat['t1']]]['z']) <= (self.tarr[self.tkey[dat['t2']]]['z']):
                    (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                    (dat['el1'], dat['el2']) = (dat['el2'], dat['el1'])
                    dat['u'] = -dat['u']
                    dat['v'] = -dat['v']
                    dat['vis'] = np.conj(dat['vis'])
                    dat['uvis'] = np.conj(dat['uvis'])
                    dat['qvis'] = np.conj(dat['qvis'])
                     # Append the data point
                blpairs.append(set((dat['t1'],dat['t2'])))
                obsdata.append(dat)

        obsdata = np.array(obsdata, dtype=DTPOL)
#        print 'Obsdata constructor shape: ',obsdata.shape
        
        # Sort the data by time
        obsdata = obsdata[np.argsort(obsdata, order=['time','t1'])]
        
        # Save the data
        self.data = obsdata
#        print 'final data shape: ',self.data.shape
            
        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        print("newtest")
        print(mjd)
        self.mjd = fracmjd(mjd, self.tstart)
        print(self.mjd)
        print(mjd)
        print("***self.tstart=")
        print(self.tstart)
        self.tstop = times[-1]
        if self.tstop < self.tstart:
            self.tstop += 24.0
        
        self.timemjd = timemjd
        #self.mjd = timemjd
        print("test in CLASS Obsdata")
        print(self.timemjd)
  
    def copy(self):
        """Copy the observation object"""
        newobs = Obsdata(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, ampcal=self.ampcal, phasecal=self.phasecal)
        return newobs
        
    def data_conj(self):
        #!AC
        """Return a data array of same format as self.data but including all conjugate baselines"""
        
        data = np.empty(2*len(self.data),dtype=DTPOL)
        
        # Add the conjugate baseline data
        for f in DTPOL:
            f = f[0]
            if f in ["t1", "t2", "el1", "el2", "tau1", "tau2"]:
                if f[-1]=='1': f2 = f[:-1]+'2'
                else: f2 = f[:-1]+'1'
                data[f] = np.hstack((self.data[f], self.data[f2]))
            elif f in ["u","v"]:
                data[f] = np.hstack((self.data[f], -self.data[f]))
            elif f in ["vis","qvis","uvis"]:
                data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
            else:
                data[f] = np.hstack((self.data[f], self.data[f]))
        
        # Sort the data
        #!AC sort within equal times?
        data = data[np.argsort(data['time'])]
        return data
        
    def unpack(self, fields, conj=False):
        """Return a recarray of all the data for the given fields from the data table
           If conj=True, will return conjugate baselines"""
        
        # If we only specify one field
        if type(fields) == str: fields = [fields]
            
        # Get conjugates
        if conj:
            data = self.data_conj()
        else:
            data = self.data
        
        # Get field data
        allout = []
        for field in fields:
             
            if field in ["u","v","sigma","tint","time","el1","el2","tau1","tau2"]:
                out = data[field]
                ty = 'f8'
            elif field in ["t1","t2"]:
                out = data[field]
                ty = 'a32'
            elif field in ["vis","amp","phase","snr"]:
                out = data['vis']
                ty = 'c16'
            elif field in ["qvis","qamp","qphase","qsnr"]:
                out = data['qvis']
                ty = 'c16'
            elif field in ["uvis","uamp","uphase","usnr"]:
                out = data['uvis']
                ty = 'c16'
            elif field in ["pvis","pamp","pphase"]:
                out = data['qvis'] + 1j * data['uvis']
                ty = 'c16'
            elif field in ["m","mamp","mphase"]:
                out = (data['qvis'] + 1j * data['uvis']) / data['vis']
                ty = 'c16'
            elif field in ["uvdist"]:
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            else: raise Exception("%s is not valid field \n" % field +
                                  "valid field values are " + string.join(FIELDS))

            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp","pamp","mamp"]:
                out = np.abs(out)
                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "pphase", "mphase"]:
                out = np.angle(out)/DEGREE
                ty = 'f8'
            elif field in ["snr","qsnr","usnr"]:
                out = np.abs(out)/data['sigma']
                ty = 'f8'
             
                    
            # Reshape and stack with other fields
            out = np.array(out, dtype=[(field, ty)])
            if len(allout) > 0:
                allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
            else:
                allout = out
            
        return allout
    
    def tlist(self, conj=False):
        """Return partitioned data in a list of equal time observations"""
        
        if conj:
            data = self.data_conj()
        else:
            data = self.data
        
        # Use itertools groupby function to partition the data
        datalist = []
        for key, group in it.groupby(data, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        return datalist
    
    def res(self):
        """Return the nominal resolution of the observation in radian"""
        return 1.0/np.max(self.unpack('uvdist')['uvdist'])
        
    def bispectra(self, vtype='vis', mode='time', count='min'):
        """Return all independent equal time bispectrum values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all bispectrum values
           Get Q, U, P bispectra by changing vtype
        """

        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('min', 'max'):
            raise Exception("possible options for count are 'min' and 'max'")
        
        # Generate the time-sorted data with conjugate baselines
        tlist = self.tlist(conj=True)
        outlist = []
        bis = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = list(set(np.hstack((tdata['t1'],tdata['t2']))))
                                        
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            # Determine the triangles in the time step
            if count == 'min':
                # If we want a minimal set, choose triangles with the minimum sefd reference
                # Unless there is no sefd data, in which case choose the northernmost
                if len(set(self.tarr['sefd'])) > 1:
                    ref = sites[np.argmin([self.tarr[self.tkey[site]]['sefd'] for site in sites])]
                else:
                    ref = sites[np.argmax([self.tarr[self.tkey[site]]['z'] for site in sites])]
                sites.remove(ref)
                
                # Find all triangles that contain the ref
                tris = list(it.combinations(sites,2))
                tris = [(ref, t[0], t[1]) for t in tris]
            elif count == 'max':
                # Find all triangles
                tris = list(it.combinations(sites,3))
            
            # Generate bispectra for each triangle
            for tri in tris:
                # The ordering is north-south
                a1 = np.argmax([self.tarr[self.tkey[site]]['z'] for site in tri])
                a3 = np.argmin([self.tarr[self.tkey[site]]['z'] for site in tri])
                a2 = 3 - a1 - a3
                tri = (tri[a1], tri[a2], tri[a3])
                    
                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1],tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue
                    
                # Choose the appropriate polarization and compute the bs and err
                if vtype in ["vis", "qvis", "uvis"]:
                    bi = l1[vtype]*l2[vtype]*l3[vtype]
                    bisig = np.abs(bi) * np.sqrt((l1['sigma']/np.abs(l1[vtype]))**2 +
                                                 (l2['sigma']/np.abs(l2[vtype]))**2 +
                                                 (l3['sigma']/np.abs(l3[vtype]))**2)
                    #Katie's 2nd + 3rd order corrections - see CHIRP supplement
                    bisig = np.sqrt(bisig**2 + (l1['sigma']*l2['sigma']*np.abs(l3[vtype]))**2 +
                                               (l1['sigma']*l3['sigma']*np.abs(l2[vtype]))**2 +
                                               (l3['sigma']*l2['sigma']*np.abs(l1[vtype]))**2 +
                                               (l1['sigma']*l2['sigma']*l3['sigma'])**2 )
                elif vtype == "pvis":
                    p1 = l1['qvis'] + 1j*l2['uvis']
                    p2 = l2['qvis'] + 1j*l2['uvis']
                    p3 = l3['qvis'] + 1j*l3['uvis']
                    bi = p1 * p2 * p3
                    bisig = np.abs(bi) * np.sqrt((l1['sigma']/np.abs(p1))**2 +
                                                 (l2['sigma']/np.abs(p2))**2 +
                                                 (l3['sigma']/np.abs(p3))**2)
                    #Katie's 2nd + 3rd order corrections - see CHIRP supplement
                    bisig = np.sqrt(bisig**2 + (l1['sigma']*l2['sigma']*np.abs(p3))**2 +
                                               (l1['sigma']*l3['sigma']*np.abs(p2))**2 +
                                               (l3['sigma']*l2['sigma']*np.abs(p1))**2 +
                                               (l1['sigma']*l2['sigma']*l3['sigma'])**2 )

                    bisig = np.sqrt(2) * bisig
                
                # Append to the equal-time list
                bis.append(np.array((time, tri[0], tri[1], tri[2],
                                     l1['u'], l1['v'], l2['u'], l2['v'], l3['u'], l3['v'],
                                     bi, bisig), dtype=DTBIS))
            
            # Append equal time bispectra to outlist
            if mode=='time' and len(bis) > 0:
                outlist.append(np.array(bis))
                bis = []
     
        if mode=='all':
            outlist = np.array(bis)
        
        return outlist
   
        
    def c_phases(self, vtype='vis', mode='time', count='min'):
        """Return all independent equal time closure phase values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all closure phases
        """
        #!AC Error formula for closure phases only true in high SNR limit!
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")
        
        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count)
        
        # Reformat into a closure phase list/array
        outlist = []
        cps = []
        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = bi['sigmacp']/np.abs(bi['cphase'])/DEGREE
                bi['cphase'] = (np.angle(bi['cphase'])/DEGREE).real
                cps.append(bi.astype(np.dtype(DTCPHASE)))
            if mode == 'time' and len(cps) > 0:
                outlist.append(np.array(cps))
                cps = []
                
        if mode == 'all':
            outlist = np.array(cps)

        return outlist
         
    def c_amplitudes(self, vtype='vis', mode='time', count='min'):
        """Return equal time closure amplitudes
           Set count='max' to return all closure amplitudes up to inverses
        """
        
        #!AC Error formula for closure amplitudes only true in high SNR limit!
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")
        
        tlist = self.tlist(conj=True)
        outlist = []
        cas = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = np.array(list(set(np.hstack((tdata['t1'],tdata['t2'])))))
            if len(sites) < 4:
                continue
                                            
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            if count == 'min':
                # If we want a minimal set, choose the minimum sefd reference
                # !AC sites are ordered by sefd - does that make sense?
                sites = sites[np.argsort([self.tarr[self.tkey[site]]['sefd'] for site in sites])]
                ref = sites[0]
                
                # Loop over other sites >=3 and form minimal closure amplitude set
                for i in range(3, len(sites)):
                    blue1 = l_dict[ref, sites[i]] #!!
                    for j in range(1, i):
                        if j == i-1: k = 1
                        else: k = j+1
                        
                        red1 = l_dict[sites[i], sites[j]]
                        red2 = l_dict[ref, sites[k]]
                        blue2 = l_dict[sites[j], sites[k]]
                        
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis"]:
                            e1 = blue1['sigma']
                            e2 = blue2['sigma']
                            e3 = red1['sigma']
                            e4 = red2['sigma']
                            
                            p1 = amp_debias(blue1[vtype], e1)
                            p2 = amp_debias(blue2[vtype], e2)
                            p3 = amp_debias(red1[vtype], e3)
                            p4 = amp_debias(red2[vtype], e4)
                                                                             
                        elif vtype == "pvis":
                            e1 = np.sqrt(2)*blue1['sigma']
                            e2 = np.sqrt(2)*blue2['sigma']
                            e3 = np.sqrt(2)*red1['sigma']
                            e4 = np.sqrt(2)*red2['sigma']

                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], e1)
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], e2)
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], e3)
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], e4)
                            
                        
                        camp = np.abs((p1*p2)/(p3*p4))
                        camperr = camp * np.sqrt((e1/np.abs(p1))**2 +
                                                 (e2/np.abs(p2))**2 +
                                                 (e3/np.abs(p3))**2 +
                                                 (e4/np.abs(p4))**2)
                                        
                        # Add the closure amplitudes to the equal-time list
                        # Our site convention is (12)(34)/(14)(23)
                        cas.append(np.array((time,
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP))


            elif count == 'max':
                # Find all quadrangles
                quadsets = list(it.combinations(sites,4))
                for q in quadsets:
                    # Loop over 3 closure amplitudes
                    # Our site convention is (12)(34)/(14)(23)
                    for quad in (q, [q[0],q[2],q[1],q[3]], [q[0],q[1],q[3],q[2]]):
                        
                        # Blue is numerator, red is denominator
                        blue1 = l_dict[quad[0], quad[1]]
                        blue2 = l_dict[quad[2], quad[3]]
                        red1 = l_dict[quad[0], quad[3]]
                        red2 = l_dict[quad[1], quad[2]]
                                      
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis"]:
                            e1 = blue1['sigma']
                            e2 = blue2['sigma']
                            e3 = red1['sigma']
                            e4 = red2['sigma']
                            
                            p1 = amp_debias(blue1[vtype], e1)
                            p2 = amp_debias(blue2[vtype], e2)
                            p3 = amp_debias(red1[vtype], e3)
                            p4 = amp_debias(red2[vtype], e4)
                                                                             
                        elif vtype == "pvis":
                            e1 = np.sqrt(2)*blue1['sigma']
                            e2 = np.sqrt(2)*blue2['sigma']
                            e3 = np.sqrt(2)*red1['sigma']
                            e4 = np.sqrt(2)*red2['sigma']

                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], e1)
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], e2)
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], e3)
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], e4)
                            
                        
                        camp = np.abs((p1*p2)/(p3*p4))
                        camperr = camp * np.sqrt((e1/np.abs(p1))**2 +
                                                 (e2/np.abs(p2))**2 +
                                                 (e3/np.abs(p3))**2 +
                                                 (e4/np.abs(p4))**2)
                                        
                                        
                        # Add the closure amplitudes to the equal-time list
                        cas.append(np.array((time,
                                             quad[0], quad[1], quad[2], quad[3],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP))

            # Append all equal time closure amps to outlist
            if mode=='time':
                outlist.append(np.array(cas))
                cas = []
            elif mode=='all':
                outlist = np.array(cas)
        
        return outlist
    
    def dirtybeam(self, npix, fov, pulse=pulses.trianglePulse2D):
        """Return a square Image object of the observation dirty beam
           fov is in radian
        """
        # !AC this is a slow way of doing this
        # !AC add different types of beam weighting
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0
        
        im = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
        
        # !AC think more carefully about the different image size cases
        im = im[0:npix, 0:npix]
        
        # Normalize to a total beam power of 1
        im = im/np.sum(im)
        
        src = self.source + "_DB"
        return Image(im, pdim, self.ra, self.dec, rf=self.rf, source=src, mjd=self.mjd, pulse=pulse)
    
    def dirtyimage(self, npix, fov, pulse=pulses.trianglePulse2D):
       

        """Return a square Image object of the observation dirty image
           fov is in radian
        """
        # !AC this is a slow way of doing this
        # !AC add different types of beam weighting
        # !AC is it possible for Q^2 + U^2 > I^2 in the dirty image?
        
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        vis = self.unpack('vis')['vis']
        qvis = self.unpack('qvis')['qvis']
        uvis = self.unpack('uvis')['uvis']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0

        # Take the DFTS
        # Shouldn't need to real about conjugate baselines b/c unpack does not return them
        im  = np.array([[np.mean(np.real(vis)*np.cos(2*np.pi*(i*u + j*v)) -
                                 np.imag(vis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
        qim = np.array([[np.mean(np.real(qvis)*np.cos(2*np.pi*(i*u + j*v)) -
                                 np.imag(qvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
        uim = np.array([[np.mean(np.real(uvis)*np.cos(2*np.pi*(i*u + j*v)) -
                                 np.imag(uvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
                                           
        dim = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
           
        # !AC is this the correct normalization?
        im = im/np.sum(dim)
        qim = qim/np.sum(dim)
        uim = uim/np.sum(dim)
 
        # !AC think more carefully about the different image size cases here
        im = im[0:npix, 0:npix]
        qim = qim[0:npix, 0:npix]
        uim = uim[0:npix, 0:npix]
        
        out = Image(im, pdim, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
        out.add_qu(qim, uim)
        return out
    
    def cleanbeam(self, npix, fov, pulse=pulses.trianglePulse2D):
        """Return a square Image object of the observation fitted (clean) beam
           fov is in radian
        """
        # !AC include other beam weightings
        im = make_square(self, npix, fov, pulse=pulse)
        beamparams = self.fit_beam()
        im = add_gauss(im, 1.0, beamparams)
        return im
        
    def fit_beam(self):
        """Fit a gaussian to the dirty beam and return the parameters (fwhm_maj, fwhm_min, theta).
           All params are in radian and theta is measured E of N.
           Fit the quadratic expansion of the Gaussian (normalized to 1 at the peak)
           to the expansion of dirty beam with the same normalization
        """
        # !AC include other beam weightings
          
        # Define the sum of squares function that compares the quadratic expansion of the dirty image
        # with the quadratic expansion of an elliptical gaussian
        def fit_chisq(beamparams, db_coeff):
            
            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2/fwhm_min2 + np.sin(theta)**2/fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2/fwhm_maj2 + np.sin(theta)**2/fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1/fwhm_maj2 - 1/fwhm_min2)
            gauss_coeff = np.array((a,b,c))
            
            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)
            
            return chisq
        
        # These are the coefficients (a,b,c) of a quadratic expansion of the dirty beam
        # For a point (x,y) in the image plane, the dirty beam expansion is 1-ax^2-by^2-cxy
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        n = float(len(u))
        abc = (2.*np.pi**2/n) * np.array([np.sum(u**2), np.sum(v**2), 2*np.sum(u*v)])
        abc = 1e-20 * abc # Decrease size of coefficients
        
        # Fit the beam
        guess = [(50)**2, (50)**2, 0.0]
        params = scipy.optimize.minimize(fit_chisq, guess, args=(abc,), method='Powell')
        
        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10*np.sqrt(params.x[0])
            fwhm_min = 1e-10*np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10*np.sqrt(params.x[1])
            fwhm_min = 1e-10*np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi/2, np.pi)

        return np.array((fwhm_maj, fwhm_min, theta))
            
    def plotall(self, field1, field2, ebar=True, rangex=False, rangey=False, conj=False, show=True, axis=False, color='b'):
        """Make a scatter plot of 2 real observation fields with errors
           If conj==True, display conjugate baselines"""
        
        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + string.join(FIELDS))
                              
        # Unpack x and y axis data
        data = self.unpack([field1,field2], conj=conj)
        
        # X error bars
        if field1 in ['amp', 'qamp', 'uamp']:
            sigx = self.unpack('sigma',conj=conj)['sigma']
        elif field1 in ['phase', 'uphase', 'qphase']:
            sigx = (self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('amp',conj=conj)['amp'])/DEGREE
        elif field1 == 'pamp':
            sigx = np.sqrt(2)*self.unpack('sigma',conj=conj)['sigma']
        elif field1 == 'pphase':
            sigx = np.sqrt(2)*(self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('pamp',conj=conj)['pamp'])/DEGREE
        elif field1 == 'mamp':
            sigx = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'],
                        self.unpack('mamp',conj=conj)['mamp'])
        elif field1 == 'mphase':
            sigx = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'],
                        self.unpack('mamp',conj=conj)['mamp']) / self.unpack('mamp',conj=conj)['mamp']
        else:
            sigx = None
            
        # Y error bars
        if field2 in ['amp', 'qamp', 'uamp']:
            sigy = self.unpack('sigma',conj=conj)['sigma']
        elif field2 in ['phase', 'uphase', 'qphase']:
            sigy = (self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('amp',conj=conj)['amp'])/DEGREE
        elif field2 == 'pamp':
            sigy = np.sqrt(2)*self.unpack('sigma',conj=conj)['sigma']
        elif field2 == 'pphase':
            sigy = np.sqrt(2)*(self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('pamp',conj=conj)['pamp'])/DEGREE
        elif field2 == 'mamp':
            sigy = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'],
                        self.unpack('mamp',conj=conj)['mamp'])
        elif field2 == 'mphase':
            sigy = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'],
                        self.unpack('mamp',conj=conj)['mamp']) / self.unpack('mamp',conj=conj)['mamp']
        else:
            sigy = None
        
        # Debias amplitudes if appropriate:
        if field1 in ['amp', 'qamp', 'uamp', 'pamp', 'mamp']:
            print("De-biasing amplitudes for plot x values!")
            data[field1] = amp_debias(data[field1], sigx)
        
        if field2 in ['amp', 'qamp', 'uamp', 'pamp', 'mamp']:
            print("De-biasing amplitudes for plot y values!")
            data[field2] = amp_debias(data[field2], sigy)
           
        # Data ranges
        if not rangex:
            rangex = [np.min(data[field1]) - 0.2 * np.abs(np.min(data[field1])),
                      np.max(data[field1]) + 0.2 * np.abs(np.max(data[field1]))]
        if not rangey:
            rangey = [np.min(data[field2]) - 0.2 * np.abs(np.min(data[field2])),
                      np.max(data[field2]) + 0.2 * np.abs(np.max(data[field2]))]
        
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)
         
        if ebar and (np.any(sigy) or np.any(sigx)):
            x.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, fmt='b.', color=color)
        else:
            x.plot(data[field1], data[field2], 'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel(field1)
        x.set_ylabel(field2)

        if show:
            plt.show(block=False)
        return x
        
    def plot_bl(self, site1, site2, field, ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b'):
        """Plot a field over time on a baseline"""
        
        # Determine if fields are valid
        if field not in FIELDS:
            raise Exception("valid fields are " + string.join(FIELDS))
        
        # Get the data from data table on the selected baseline
        # !AC TODO do this with unpack instead?
        plotdata = []
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    time = obs['time']
                    if field == 'uvdist':
                        plotdata.append([time, np.abs(obs['u'] + 1j*obs['v']), 0])
                    
                    elif field in ['amp', 'qamp', 'uamp']:
                        print("De-biasing amplitudes for plot!")
                        if field == 'amp': l = 'vis'
                        elif field == 'qamp': l = 'qvis'
                        elif field == 'uamp': l = 'uvis'
                        plotdata.append([time, amp_debias(obs[l], obs['sigma']), obs['sigma']])
                    
                    elif field in ['phase', 'qphase', 'uphase']:
                        if field == 'phase': l = 'vis'
                        elif field == 'qphase': l = 'qvis'
                        elif field == 'uphase': l = 'uvis'
                        plotdata.append([time, np.angle(obs[l])/DEGREE, obs['sigma']/np.abs(obs[l])/DEGREE])
                    
                    elif field == 'pamp':
                        print("De-biasing amplitudes for plot!")
                        pamp = amp_debias(obs['qvis'] + 1j*obs['uvis'], np.sqrt(2)*obs['sigma'])
                        plotdata.append([time, pamp, np.sqrt(2)*obs['sigma']])
                    
                    elif field == 'pphase':
                        plotdata.append([time,
                                         np.angle(obs['qvis'] + 1j*obs['uvis'])/DEGREE,
                                         np.sqrt(2)*obs['sigma']/np.abs(obs['qvis'] + 1j*obs['uvis'])/DEGREE
                                       ])
                    
                    elif field == 'mamp':
                        print("NOT de-baising polarimetric ratio amplitudes!")
                        plotdata.append([time,
                                         np.abs((obs['qvis'] + 1j*obs['uvis'])/obs['vis']),
                                         merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])
                                       ])
                    
                    elif field == 'mphase':
                        plotdata.append([time,
                                        np.angle((obs['qvis'] + 1j*obs['uvis'])/obs['vis'])/DEGREE,
                                        (merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])/
                                           np.abs((obs['qvis']+1j*obs['uvis'])/obs['vis'])/DEGREE)
                                       ])
                    
                    else:
                        plotdata.append([time, obs[field], 0])
                    
                    # Assume only one relevant entry per scan
                    break
        


        if not rangex:
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])),
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))]
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)

        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMST (h)')
        x.set_ylabel(field)
        x.set_title('%s - %s'%(site1,site2))
        
        if show:
            plt.show(block=False)
        return x
                
                
    def plot_cphase(self, site1, site2, site3, ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b'):
        """Plot closure phase over time on a triangle"""

        # Get closure phases (maximal set)
        
        cphases = self.c_phases(mode='time', count='max')
        
        # Get requested closure phases over time
        tri = (site1, site2, site3)
        plotdata = []
        for entry in cphases:
            for obs in entry:
                obstri = (obs['t1'],obs['t2'],obs['t3'])
                if set(obstri) == set(tri):
                    # Flip the sign of the closure phase if necessary
                    parity = paritycompare(tri, obstri)
                    plotdata.append([obs['time'], parity*obs['cphase'], obs['sigmacp']])
                    continue
        
        plotdata = np.array(plotdata)
        
        if len(plotdata) == 0:
            print("No closure phases on this triangle!")
            return
        
        # Data ranges
        if not rangex:
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])),
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))]
        
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)

        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        x.set_ylabel('Closure Phase (deg)')
        x.set_title('%s - %s - %s' % (site1,site2,site3))
        if show:
            plt.show(block=False)
        return x
        
    def plot_camp(self, site1, site2, site3, site4, ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b'):
        """Plot closure amplitude over time on a quadrange
           (1-2)(3-4)/(1-4)(2-3)
        """
        quad = (site1, site2, site3, site4)
        b1 = set((site1, site2))
        r1 = set((site1, site4))
              
        # Get the closure amplitudes
        camps = self.c_amplitudes(mode='time', count='max')
        plotdata = []
        for entry in camps:
            for obs in entry:
                obsquad = (obs['t1'],obs['t2'],obs['t3'],obs['t4'])
                if set(quad) == set(obsquad):
                    num = [set((obs['t1'], obs['t2'])), set((obs['t3'], obs['t4']))]
                    denom = [set((obs['t1'], obs['t4'])), set((obs['t2'], obs['t3']))]
                    
                    if (b1 in num) and (r1 in denom):
                        plotdata.append([obs['time'], obs['camp'], obs['sigmaca']])
                    elif (r1 in num) and (b1 in denom):
                        plotdata.append([obs['time'], 1./obs['camp'], obs['sigmaca']/(obs['camp']**2)])
                    continue
                
                    
        plotdata = np.array(plotdata)
        if len(plotdata) == 0:
            print("No closure amplitudes on this quadrangle!")
            return

        # Data ranges
        if not rangex:
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])),
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))]
        
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)
            
        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        x.set_ylabel('Closure Amplitude')
        x.set_title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)'%(site1,site2,site3,site4,
                                                           site1,site4,site2,site3))
        if show:
            plt.show(block=False)
            return
        else:
            return x
            
    def save_txt(self, fname):
        """Save visibility data to a text file"""
        
        # Get the necessary data and the header
        outdata = self.unpack(['time', 'tint', 't1', 't2', 'el1', 'el2', 'tau1','tau2',
                               'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'sigma'])
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %.4f - %.4f \n" % (fracmjd(self.mjd,self.tstart), fracmjd(self.mjd,self.tstop)) +
                    "RF: %.4f GHz \n" % (self.rf/1e9) +
                    "BW: %.4f GHz \n" % (self.bw/1e9) +
                    "PHASECAL: %i \n" % self.phasecal +
                    "AMPCAL: %i \n" % self.ampcal +
                    "----------------------------------------------------------------\n" +
                    "Site       X(m)             Y(m)             Z(m)           SEFD\n"
                )
        
        for i in range(len(self.tarr)):
            head += "%-8s %15.5f  %15.5f  %15.5f  %6.4f \n" % (self.tarr[i]['site'],
                                                               self.tarr[i]['x'], self.tarr[i]['y'], self.tarr[i]['z'],
                                                               self.tarr[i]['sefd'])

        head += ("----------------------------------------------------------------\n" +
                "time (hr) tint    T1     T2    Elev1   Elev2  Tau1   Tau2   U (lambda)       V (lambda)         "+
                "Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   sigma (Jy)"
                )
          
        # Format and save the data
        fmts = ("%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %4.2f   %4.2f  %16.4f %16.4f    "+
               "%10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f")
        np.savetxt(fname, outdata, header=head, fmt=fmts)
        return
    
    def save_uvfits(self, fname, n_subscan=1):
        """Save visibility data to uvfits
            Needs template.UVP file
        """
        
        # Open template UVFITS
        #hdulist = fits.open('./template.UVP')          # this template doesn't contain FQ table
        hdulist = fits.open('./template-FQ.UVP')
        
        print("self.bw=", self.bw)
        print("self.rf=", self.rf)
        
        # Load the array data
        tarr = self.tarr
        tnames = tarr['site']
        tnums = np.arange(1, len(tarr)+1)
        xyz = np.array([[tarr[i]['x'],tarr[i]['y'],tarr[i]['z']] for i in np.arange(len(tarr))])
        sefd = tarr['sefd']
        
        print("xyz=", xyz)
        
        nsta = len(tnames)
        col1 = fits.Column(name='ANNAME', format='8A', array=tnames)
        col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
        col3 = fits.Column(name='NOSTA', format='1J', array=tnums)
        colfin = fits.Column(name='SEFD', format='1D', array=sefd)
        
        #!AC these antenna fields+header are questionable - look into them
        col25= fits.Column(name='ORBPARM', format='1E', array=np.zeros(0))
        col4 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta))
        col5 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta))
        col6 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1'))
        col7 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta))
        col8 = fits.Column(name='POLCALA', format='3E', array=np.zeros((nsta,3)))
        col9 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1'))
        col10 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=(90.*np.ones(nsta)))
        col11 = fits.Column(name='POLCALB', format='3E', array=np.zeros((nsta,3)))
        
        #!AC Change more antenna header params?
        head = hdulist['AIPS AN'].header
        #head = fits.Header()
        head['EXTVER'] = 1
        head['GSTIA0'] = 119.85 # for mjd 48277
        head['FREQ']= self.rf[0]-self.bw[0]/2.0
        #head['RDATE'] = '1991-01-21' # !AC change??
        head['ARRNAM'] = self.arrname[0:8] # just to comment for the moment
        head['XYZHAND'] = 'RIGHT'
        head['ARRAYX'] = 0.e0
        head['ARRAYY'] = 0.e0
        head['ARRAYZ'] = 0.e0
        head['DEGPDY'] = 360.985
        head['POLARX'] = 0.e0
        head['POLARY'] = 0.e0
        head['UT1UTC'] = 0.e0
        head['DATUTC'] = 0.e0
        head['TIMESYS'] = 'UTC'
        head['FRAME'] = '????'
        head['NUMORB'] = 0
        head['NO_IF'] = self.nwav
        head['NOPCAL'] = 2
        head['POLTYPE'] = 'APPROX'
        head['FREQID'] = 1
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col25,col3,col4,col5,col6,col7,col8,col9,col10,col11]), name='AIPS AN', header=head)
        hdulist['AIPS AN'] = tbhdu
        
        ### fgao: now let's add the FQ table
        head_fq = hdulist['AIPS FQ'].header
        head_fq['NO_IF'] = self.nwav
        head_fq['EXTVER'] = 1
        col1_fq = fits.Column(name='FRQSEL', format='1J', array=[1])
        #iffreq = np.array([[self.rf[i][0],self.rf[i][1],self.rf[i][2]] for i in range(3)])
        #col2_fq = fits.Column(name='IF FREQ', format='3D', array=iffreq
        #col2_fq = fits.Column(name='IF FREQ', format='1D', array=self.rf-self.rf[0])
        #iffreq = [[self.rf[i]-self.rf[0] for i in range(self.nwav)]]
        #print "iffreq=", iffreq
        print("self.nwav=",self.nwav)
        col2_fq = fits.Column(name='IF FREQ', format=str(self.nwav)+'D', unit='Hz', array=[[self.rf[i]-self.rf[0] for i in range(self.nwav)]])
        col3_fq = fits.Column(name='CH WIDTH', format=str(self.nwav)+'E', unit='Hz', array=[[self.bw[i] for i in range(self.nwav)]])
        col4_fq = fits.Column(name='TOTAL BANDWIDTH', format=str(self.nwav)+'E', unit='Hz', array=[[self.bw[i] for i in range(self.nwav)]])
        col5_fq = fits.Column(name='SIDEBAND', format=str(self.nwav)+'J', array=[[np.ones(self.nwav)]])
        
        
        
        #col6_fq = fits.Column(name='RXCODE', format='8A', array=np.array('' for i in range(self.nwav)))
        tbhdu_fq = fits.BinTableHDU.from_columns(fits.ColDefs([col1_fq,col2_fq,col3_fq,col4_fq,col5_fq]), name='AIPS FQ', header=head_fq)
        #tbhdu_fq = fits.BinTableHDU.from_columns(fits.ColDefs([col1_fq,col2_fq]), name='AIPS FQ', header=head_fq)
        #tbhdu_fq = fits.BinTableHDU.from_columns(fits.ColDefs([col1_fq]), name='AIPS FQ', header=head_fq)
        hdulist['AIPS FQ'] = tbhdu_fq
        
        print("self.rf.shape=", self.rf.shape)


        # Data header (based on the BU format)
        ###
        header = hdulist[0].header
        #header = fits.Header()
        #header['EXTEND'] = True
        header['OBSRA'] = self.ra        # * 180./12.
        print("self.ra=", self.ra)
        header['OBSDEC'] = self.dec
        print("self.dec=", self.dec)
        header['OBJECT'] = self.source
        header['MJD'] = self.mjd
        print("self.mjd=", self.mjd)
        header['DATE-OBS'] = self.dateobs
        print("self.dateobs=",self.dateobs)
        header['BUNIT'] = 'JY'
        header['VELREF'] = 3 #??
        header['ALTRPIX'] = 1.e0
        header['TELESCOP'] = self.arrname[0:8]
        #header['TELESCOP'] = 'ALMA'
        header['INSTRUME'] = self.ins[0,0][0:7]     # here I just use the first record in the list ins
        header['RESTFREQ'] = self.rf[0]     # here I just quote the first reference frequency as the rest frequency
        print("self.ins.shape=", self.ins.shape)
        
        header['NAXIS3'] = 2
        header['NAXIS4'] = self.nwav
        
        print("header[NAXIS4]=", header['NAXIS4'])
        
        header['CTYPE2'] = 'COMPLEX'
        header['CRVAL2'] = 1.e0
        header['CDELT2'] = 1.e0
        header['CRPIX2'] = 1.e0
        header['CROTA2'] = 0.e0
        header['CTYPE3'] = 'STOKES'
        header['CRVAL3'] = -1.e0
        header['CRDELT3'] = -1.e0
        header['CRPIX3'] = 1.e0
        header['CROTA3'] = 0.e0
        header['CTYPE4'] = 'FREQ'
        header['CRVAL4'] = self.rf[0]          #don't need to subtract the half bandwidth self.bw[0]/2.0, this frequency should also be at the ncenter of the channel
        header['CDELT4'] = self.bw[0]
        header['CRPIX4'] = 1.e0
        header['CROTA4'] = 0.e0
        header['CTYPE5'] = 'IF'
        header['CRVAL5'] = 1.e0
        header['CDELT5'] = 1.e0
        header['CRPIX5'] = 1.e0
        header['CROTA5'] = 0.e0
        header['CTYPE6'] = 'RA'
        header['CRVAL6'] = header['OBSRA']
        header['CDELT6'] = 1.e0
        header['CRPIX6'] = 1.e0
        header['CROTA6'] = 0.e0
        header['CTYPE7'] = 'DEC'
        header['CRVAL7'] = header['OBSDEC']
        header['CDELT7'] = 1.e0
        header['CRPIX7'] = 1.e0
        header['CROTA7'] = 0.e0
        
        print("self.rf=", self.rf)
        
        header['PTYPE1'] = 'UU---SIN'
        #header['PSCAL1'] = 1/self.rf     #this is AC's original
        header['PSCAL1'] = 1/self.rf[0]    # it seems this parameter is not been used except with FITTP
        header['PZERO1'] = 0.e0
        header['PTYPE2'] = 'VV---SIN'
        header['PSCAL2'] = 1/self.rf[0]   # it seems this parameter is not been used except with FITTP
        header['PZERO2'] = 0.e0
        header['PTYPE3'] = 'WW---SIN'
        header['PSCAL3'] = 1/self.rf[0]    # it seems this parameter is not been used except with FITTP
        header['PZERO3'] = 0.e0
        header['PTYPE4'] = 'BASELINE'
        header['PSCAL4'] = 1.e0
        header['PZERO4'] = 0.e0
        header['PTYPE5'] = 'DATE'
        header['PSCAL5'] = 1.e0
        header['PZERO5'] = 0.e0
        header['PTYPE6'] = 'DATE'
        header['PSCAL6'] = 1.e0
        header['PZERO6'] = 0.0
        header['PTYPE7'] = 'INTTIM'
        header['PSCAL7'] = 1.e0
        header['PZERO7'] = 0.e0
        header['PTYPE8'] = 'ELEV1'
        header['PSCAL8'] = 1.e0
        header['PZERO8'] = 0.e0
        header['PTYPE9'] = 'ELEV2'
        header['PSCAL9'] = 1.e0
        header['PZERO9'] = 0.e0
        header['PTYPE10'] = 'TAU1'
        header['PSCAL10'] = 1.e0
        header['PZERO10'] = 0.e0
        header['PTYPE11'] = 'TAU2'
        header['PSCAL11'] = 1.e0
        header['PZERO11'] = 0.e0
        header['PTYPE12'] = 'FRQSEL'
        header['PSCAL12'] = 1.e0
        header['PZERO12'] = 0.e0
        
        # Get data
        obsdata = self.unpack(['time','tint','u','v','vis','qvis','uvis','sigma','t1','t2','el1','el2','tau1','tau2'])
        #ndatt = len(obsdata['time'])
        
        print("ins in save_uvfits")
        print("self.ins.shape=", self.ins.shape)
        self.ins = np.reshape(self.ins,(len(obsdata)))
        print("self.ins.shape=", self.ins.shape)
        
        
        ndatt = len(self.datatable['time'])
        
        print("ndatt=", ndatt)
        nwav = self.nwav
        ndat = int(ndatt/nwav)
        print("nwav=", nwav)
        print("ndat=", ndat)
        
        rrpol = 0
        llpol = 0

        for i in range(len(self.datatable['vis'])):
            if self.ins[i].find('P1') != -1:
                rrpol = 1
            if self.ins[i].find('P2') != -1:
                llpol = 1

        dualpol = False
        if rrpol*llpol == 1:
            dualpol = True

        print("dualpol=", dualpol)
        
        if dualpol == True:
            ndat = int(ndat/2)
        
        #here this 2 corresponds to counting rr and ll as two records, which really should be one record

        # here ndat corresponds to the number of scans * n_bl * number of subscans(within each exposure)
        print("ndat=", ndat)


        
        # uv are in lightseconds
        #u = obsdata['u']
        #v = obsdata['v']
        u = self.datatable['u']
        v = self.datatable['v']

        print("len(self.datatable['u'])=", len(self.datatable['u']))
        print("len(self.datatable['time'])=", len(self.datatable['time']))
        #for i in range(0, 100, 1):
        #    print i, self.datatable['sigma'][i]
        #u = u[::-1]
        #v = v[::-1]
        
        #print "self.datatable['u']=", self.datatable['u']



        print("u.shape=", u.shape)
        #print "u=", u
        
        # times and tints
        #jds = (self.mjd + 2400000.5) * np.ones(len(obsdata))
        #print "self.mjd=", self.mjd
        print("self.timemjd=", self.timemjd)
        print("self.timemjd.shape=", self.timemjd.shape)
        #print "obsdata.shape="
        #print obsdata.shape
        self.timemjd = np.reshape(self.timemjd,(len(obsdata)))
        print("len(obsdata)=",len(obsdata))
        print("self.timemjd.shape=", self.timemjd.shape)
        jds = np.round(self.timemjd,0)+2400000.5
        fractimes = self.timemjd - np.round(self.timemjd,0)
        #for i in range (0,2904,132):
        #    print "jds=", i, self.timemjd[i],jds[i],fractimes[i], jds[i]+fractimes[i]
        #print "jds.shape=", jds.shape
        
        #print "fractimes=",fractimes
        #i = 0

        #for i in range(len(obsdata)):
            #print i, 'jds=', jds[i], 'fractimes=', fractimes[i]
            
        #fractimes = np.ones(ndatt)*fractimes
        #print fractimes
        #fractimes = (obsdata['time'] / 24.0)
        tints = self.datatable['tint']
        
        print("tints=", tints)

        # Baselines
        # !GAO 21.09.17: In AIPS, to let e.g. VPLOT working, for each baseline, t1 is alwayls smaller than t2. But in our GRAVITY data, t1 is always bigger than t2. So here I need to swap them, opposite the baseline vector direction.
        t2 = [self.tkey[scope] + 1 for scope in self.datatable['t1']]
        t1 = [self.tkey[scope] + 1 for scope in self.datatable['t2']]
        bl = 256*np.array(t1) + np.array(t2)
        
        #print "t1=", t1
        #print "t2=", t2
        
        # elevations
        el1 = self.datatable['el1']
        el2 = self.datatable['el2']
        
        # opacities
        tau1 = self.datatable['tau1']
        tau2 = self.datatable['tau2']
        
        print("self.datatable['vis'].shape=", self.datatable['vis'].shape)
        for i in range(10):
            print("self.datatable['vis'][i]=", i, self.datatable['vis'][i])
        #for i in range(0,ndatt):
        #    print "self.datatable[i]=", i, self.datatable[i]
        
        
        
        # rr, ll, lr, rl, weights
        # !AC Assume V = 0 (linear polarization only)
        #if dualpol == True:
        rr = np.zeros(int(ndat*nwav),dtype=complex)
        ll = np.zeros(int(ndat*nwav),dtype=complex)
        weight_rr = np.zeros(int(ndat*nwav))
        weight_ll = np.zeros(int(ndat*nwav))
        #rr = np.zeros(len(self.datatable['vis']))
        #ll = np.zeros(len(self.datatable['vis']))


 
        kr = 0
        kl = 0
        for i in range(len(self.datatable['vis'])):
            if (self.ins[i].find('P1') != -1) or (dualpol == False):
                rr[kr] = rr[kr] + self.datatable['vis'][i]
                weight_rr[kr] = weight_rr[kr] + 1 / (2 * self.datatable['sigma'][i]**2)
                kr = kr + 1
            if self.ins[i].find('P2') != -1:
                ll[kl] = ll[kl] + self.datatable['vis'][i]
                weight_ll[kl] = weight_ll[kl] + 1 / (2 * self.datatable['sigma'][i]**2)
                kl = kl + 1

        #rr = rr + self.datatable['vis'] # complex
        #ll = ll + self.datatable['vis'] # complex
        #rl = self.datatable['qvis'] + 1j*self.datatable['uvis']
        #lr = self.datatable['qvis'] - 1j*self.datatable['uvis']
        #didn't consider the cross polarization term
        #weight = 1 / (2 * obsdata['sigma']**2)       #AC's original setting
        #weight = 1 / (2 * self.datatable['sigma']**2)
        
        #print "obsdata['sigma']=", obsdata['sigma']
        #print "obsdata['sigma'].shape=", obsdata['sigma'].shape
        print("self.datatable['sigma']=", self.datatable['sigma'])
        print("self.datatable['sigma'].shape=", self.datatable['sigma'].shape)
        #for i in range(len(self.datatable['vis'])):
            #if self.ins[i].find('P1') < 0:
            #    rr[i] = 0
            #if self.ins[i].find('P2') < 0:
            #    ll[i] = 0

        print("rr.shape=", rr.shape)
        print("rr=", rr)
        print("rr type=", type(rr))
        print("self.datatable['vis'].shape=", self.datatable['vis'].shape)

        print('1st time rr[0]=',rr[0], ll[0])
        #rr[0] = rr[0]*0
        print("rr[0].shape=", rr[0].shape)
        print('2nd time rr[0]=',rr[0], ll[0])
        print('2nd time rr[1]=',rr[1], ll[1])
        print("self.datatable['vis'][0]=", self.datatable['vis'][0])
        print("self.datatable['vis'][1]=", self.datatable['vis'][1])
    


        print("len(self.datatable['vis'])=")
        print(len(self.datatable['vis']))
        
        print("rr.shape=", rr.shape)
        #for i in range(0,2904,100):
        #    print i, rr[i], ll[i], rl[i], lr[i], self.ins[i], self.ins[i].find('P1')
        # Data array
        
        
        outdat = np.zeros((ndat, 1, 1, nwav, 1, 2, 3))    #now implementing data into multi-IFs
        
        print("outdat shape")
        k = 0
        print(outdat.shape)
        
        loop1 = list(range(ndat))
        loop2 = list(range(nwav))
        for i in loop1:
            for j in loop2:
                outdat[i,0,0,j,0,0,0] = np.real(rr[k])
                outdat[i,0,0,j,0,0,1] = np.imag(rr[k])
                outdat[i,0,0,j,0,0,2] = weight_rr[k]
                outdat[i,0,0,j,0,1,0] = np.real(ll[k])
                outdat[i,0,0,j,0,1,1] = np.imag(ll[k])
                outdat[i,0,0,j,0,1,2] = weight_ll[k]
                #outdat[i,0,0,0,j,2,0] = np.real(rl[k])
                #outdat[i,0,0,0,j,2,1] = np.imag(rl[k])
                #outdat[i,0,0,0,j,2,2] = weight[k]
                #outdat[i,0,0,0,j,3,0] = np.real(lr[k])
                ##outdat[i,0,0,0,j,3,1] = np.imag(lr[k])
                #outdat[i,0,0,0,j,3,2] = weight[k]
                k = k + 1
        #print "outdat[0,*] begins"
        #print outdat[0,]
        #print "outdat end"


        #print "before reshape u.shape=", u.shape
        u=np.reshape(u,(int(ndatt/nwav),nwav))
        v=np.reshape(v,(int(ndatt/nwav),nwav))
        bl=np.reshape(bl,(int(ndatt/nwav),nwav))
        jds=np.reshape(jds,(int(ndatt/nwav),nwav))
        fractimes=np.reshape(fractimes,(int(ndatt/nwav),nwav))
        tints=np.reshape(tints,(int(ndatt/nwav),nwav))
        el1=np.reshape(el1,(int(ndatt/nwav),nwav))
        el2=np.reshape(el2,(int(ndatt/nwav),nwav))
        tau1=np.reshape(tau1,(int(ndatt/nwav),nwav))
        tau2=np.reshape(tau2,(int(ndatt/nwav),nwav))
        
        
        #print "after 1st reshape, u=", u.shape, u
        u = -u[:,0]/C*self.rf[0]   #divide by the speed of light, convert the unit of u,v from meters into seconds
        v = -v[:,0]/C*self.rf[0]   #the "minus" sign is added in corresponding to the baseline swap performed up.
        #print "after 1st slice, u=", u.shape, u
        bl = bl[:,0]
        jds = jds[:,0]
        fractimes = fractimes[:,0]
        tints = tints[:,0]
        el1 = el1[:,0]
        el2 = el2[:,0]
        tau1 = tau1[:,0]
        tau2 = tau2[:,0]
        
        print("### n_subscan=",n_subscan)
        
        #n_subscan = 32
        if dualpol == True:
            u = np.reshape(u,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            v = np.reshape(v,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            bl = np.reshape(bl,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            jds = np.reshape(jds,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            fractimes = np.reshape(fractimes,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            tints = np.reshape(tints,(int(ndatt/nwav/6/n_subscan),n_subscan,6))
            el1 = np.reshape(el1,(int(ndatt/nwav/6),6))
            el2 = np.reshape(el2,(int(ndatt/nwav/6),6))
            tau1 = np.reshape(tau1,(int(ndatt/nwav/6),6))
            tau2 = np.reshape(tau2,(int(ndatt/nwav/6),6))
            #print "after 2nd reshape, u=", u.shape, u
            print("in dualpol=True,fractimes.shape=",fractimes.shape)
            print("fractimes[0,0,:]=",fractimes[0,0,:])
            print("fractimes[0,:,0]=",fractimes[0,:,0])
            print(fractimes[0:100,:,0])
            u = np.ravel(u[::2,:,:])
            v = np.ravel(v[::2,:,:])
            bl = np.ravel(bl[::2,:,:])
            jds = np.ravel(jds[::2,:,:])
            #fractimes = np.ravel(fractimes[:,::2,:])
            fractimes = np.ravel(fractimes[::2,:,:])
            #print "now",fractimes[0:100,0,0]
            tints = np.ravel(tints[::2,:,:])
            el1 = np.ravel(el1[::2])
            el2 = np.ravel(el2[::2])
            tau1 = np.ravel(tau1[::2])
            tau2 = np.ravel(tau2[::2])
            print("after ravel")
            print(fractimes[0:100])
        
        frqsel = tau2 + 2


        #print "then drop and flatten, u=", u.shape, u
        print("reshape finished!")
        print("u=",u.shape,u)
    

        
        print("tau2=", tau2)
        
        print("jds.shape=**", jds.shape)
        #for i in range(0,264,12):
        #    print "jds=", i, jds[i], fractimes[i], jds[i]+fractimes[i]
        #print "fractimes=", fractimes
        
        print("frqsel.shape=", frqsel.shape)
        #print frqsel
        

        print('outdat[0].shape')
        print(outdat[0].shape)
        print("outdat.shape=",outdat.shape)
        print("##########################")
        print("jds.shape=", jds.shape)
        print("jds=",jds[0:200])
        print("fractimes.shape=", fractimes.shape)
        print("fractimes=",fractimes[0:200])
        
        
        # Save data

        pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE',
                   'INTTIM', 'ELEV1', 'ELEV2', 'TAU1', 'TAU2', 'FREQSEL']
        #pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', '_DATE',
        #        'INTTIM']
        #pardata=[u, v, np.zeros(ndat,nwav), bl, jds, fractimes, tints, el1, el2,tau1,tau2]
  
        #print 'pardata.shape='
        #print pardata
        x = fits.GroupData(outdat, parnames=pars,
            pardata=[u, v, np.zeros(ndat), bl, jds, fractimes, tints, el1, el2, tau1, tau2, frqsel],
            bitpix=-32)
        #x = fits.GroupData(outdat, parnames=pars,
        #                   pardata=[u, v, np.zeros(ndat), bl, jds, np.zeros(ndat), tints],
        #                   bitpix=-32)
        #print 'pardata='
        #print [u, v, np.zeros(ndat), bl, jds, fractimes, tints, el1, el2,tau1,tau2]
        print("x.shape=", x.shape)
        #for i in range(0,61,12):
        #    print i, x[i]
        #print 'x[0]='
        #print x[0]
                
        #hdulist[0] = fits.GroupsHDU(data=x, header=header)
        hdulist[0].data = x
        hdulist[0].header = header
        hdulist.writeto(fname)
        
        #print "hdulist[0].data"
        #print hdulist[0].data
                
        return
    
    def save_oifits(self, fname, flux=1.0):
        """Save visibility data to oifits
            Antenna diameter currently incorrect and the exact times are not correct in the datetime object
            Please contact Katie Bouman (klbouman@mit.edu) for any questions on this function
        """
        #todo: Add polarization to oifits??
        print('Warning: save_oifits does NOT save polarimetric visibility data!')
        
        # Normalizing by the total flux passed in - note this is changing the data inside the obs structure
        self.data['vis'] /= flux
        self.data['sigma'] /= flux
        
        data = self.unpack(['u','v','amp','phase', 'sigma', 'time', 't1', 't2', 'tint'])
        biarr = self.bispectra(mode="all", count="min")

        # extract the telescope names and parameters
        antennaNames = self.tarr['site'] #np.array(self.tkey.keys())
        sefd = self.tarr['sefd']
        antennaX = self.tarr['x']
        antennaY = self.tarr['y']
        antennaZ = self.tarr['z']
        #antennaDiam = -np.ones(antennaX.shape) #todo: this is incorrect and there is just a dummy variable here
        antennaDiam = sefd # replace antennaDiam with SEFD for radio observtions
        
        # create dictionary
        union = {};
        union = writeData.arrayUnion(antennaNames, union)

        # extract the integration time
        intTime = data['tint'][0]
        if not all(data['tint'][0] == item for item in np.reshape(data['tint'], (-1)) ):
            raise TypeError("The time integrations for each visibility are different")

        # get visibility information
        amp = data['amp']
        phase = data['phase']
        viserror = data['sigma']
        u = data['u']
        v = data['v']
        
        # convert antenna name strings to number identifiers
        ant1 = writeData.convertStrings(data['t1'], union)
        ant2 = writeData.convertStrings(data['t2'], union)
        
        # convert times to datetime objects
        time = data['time']
        dttime = np.array([datetime.datetime.utcfromtimestamp(x*60*60) for x in time]); #todo: these do not correspond to the acutal times
        
        # get the bispectrum information
        bi = biarr['bispec']
        t3amp = np.abs(bi);
        t3phi = np.angle(bi, deg=1)
        t3amperr = biarr['sigmab']
        t3phierr = 180.0/np.pi * (1/t3amp) * t3amperr;
        uClosure = np.transpose(np.array([np.array(biarr['u1']), np.array(biarr['u2'])]));
        vClosure = np.transpose(np.array([np.array(biarr['v1']), np.array(biarr['v2'])]));
        
        # convert times to datetime objects
        timeClosure = biarr['time']
        dttimeClosure = np.array([datetime.datetime.utcfromtimestamp(x) for x in timeClosure]); #todo: these do not correspond to the acutal times

        # convert antenna name strings to number identifiers
        biarr_ant1 = writeData.convertStrings(biarr['t1'], union)
        biarr_ant2 = writeData.convertStrings(biarr['t2'], union)
        biarr_ant3 = writeData.convertStrings(biarr['t3'], union)
        antOrder = np.transpose(np.array([biarr_ant1, biarr_ant2, biarr_ant3]))

        # todo: check that putting the negatives on the phase and t3phi is correct
        writeData.writeOIFITS(fname, self.ra, self.dec, self.rf, self.bw, intTime, amp, viserror, phase, viserror, u, v, ant1, ant2, dttime,
                              t3amp, t3amperr, t3phi, t3phierr, uClosure, vClosure, antOrder, dttimeClosure, antennaNames, antennaDiam, antennaX, antennaY, antennaZ)
   
        # Un-Normalizing by the total flux passed in - note this is changing the data inside the obs structure back to what it originally was
        self.data['vis'] *= flux
        self.data['sigma'] *= flux
        
        return
        
##################################################################################################
# Object Construction Functions
##################################################################################################
           
def load_array(filename):
    """Read an array from a text file and return an Array object
    """
    
    tdata = np.loadtxt(filename,dtype=str)
    if tdata.shape[1] != 5:
        raise Exception("Array file should have format: (name, x, y, z, SEFD)")
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4])), dtype=DTARR) for x in tdata]
    tdata = np.array(tdata)
    return Array(tdata)
      
def load_obs_txt(filename):
    """Read an observation from a text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    
    # Read the header parameters
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60. + float(ra[6])/3600.
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60. + float(dec[6])/3600.)
    mjd = float(file.readline().split()[2])
    rf = float(file.readline().split()[2]) * 1e9
    bw = float(file.readline().split()[2]) * 1e9
    phasecal = bool(file.readline().split()[2])
    ampcal = bool(file.readline().split()[2])
    file.readline()
    file.readline()
    
    # read the tarr
    line = file.readline().split()
    tarr = []
    while line[1][0] != "-":
        tarr.append(np.array((line[1], line[2], line[3], line[4], line[5]), dtype=DTARR))
        line = file.readline().split()
    tarr = np.array(tarr, dtype=DTARR)
    file.close()

    
    # Load the data, convert to list format, return object
    datatable = np.loadtxt(filename, dtype=str)
    datatable2 = []
    for row in datatable:
        time = float(row[0])
        tint = float(row[1])
        t1 = row[2]
        t2 = row[3]
        el1 = float(row[4])
        el2 = float(row[5])
        tau1 = float(row[6])
        tau2 = float(row[7])
        u = float(row[8])
        v = float(row[9])
        vis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
        if datatable.shape[1] == 17:
            qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
            uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
            sigma = float(row[16])
        elif datatable.shape[1] == 13:
            qvis = 0+0j
            uvis = 0+0j
            sigma = float(row[12])
        else:
            raise Exception('Text file does not have the right number of fields!')
            
        datatable2.append(np.array((time, tint, t1, t2, el1, el2, tau1, tau2,
                                    u, v, vis, qvis, uvis, sigma), dtype=DTPOL))
    
    # Return the datatable
    datatable2 = np.array(datatable2)
    return Obsdata(ra, dec, rf, bw, datatable2, tarr, source=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)

def load_obs_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, src='SgrA', mjd=0, ampcal=False, phasecal=False):
    """Read an observation from a maps text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    # Read telescope parameters from the array file
    tdata = np.loadtxt(arrfile, dtype=str)
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[-1])), dtype=DTARR) for x in tdata]
    tdata = np.array(tdata)

    # Read parameters from the obs_spec
    f = open(obsspec)
    stop = False
    while not stop:
        line = f.readline().split()
        if line==[] or line[0]=='\\':
            continue
        elif line[0] == 'FOV_center_RA':
            x = line[2].split(':')
            ra = float(x[0]) + float(x[1])/60. + float(x[2])/3600.
        elif line[0] == 'FOV_center_Dec':
            x = line[2].split(':')
            dec = np.sign(float(x[0])) * (abs(float(x[0])) + float(x[1])/60. + float(x[2])/3600.)
        elif line[0] == 'Corr_int_time':
            tint = float(line[2])
        elif line[0] == 'Corr_chan_bw':  #!AC what if multiple channels?
            bw = float(line[2]) * 1e6 #MHz
        elif line[0] == 'Channel': #!AC what if multiple scans with different params?
            rf = float(line[2].split(':')[0]) * 1e6
        elif line[0] == 'Scan_start':
            x = line[2].split(':') #!AC properly compute MJD!
        elif line[0] == 'Endscan':
            stop=True
    f.close()
    
    # Load the data, convert to list format, return object
    datatable = []
    f = open(ifile)
    
    for line in f:
        line = line.split()
        if not (line[0] in ['UV', 'Scan','\n']):
            time = line[0].split(':')
            time = float(time[2]) + float(time[3])/60. + float(time[4])/3600.
            u = float(line[1]) * 1000
            v = float(line[2]) * 1000
            bl = line[4].split('-')
            t1 = tdata[int(bl[0])-1]['site']
            t2 = tdata[int(bl[1])-1]['site']
            el1 = 0.
            el2 = 0.
            tau1 = 0.
            tau2 = 0.
            vis = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
            sigma = float(line[10])
            datatable.append(np.array((time, tint, t1, t2, el1, el2, tau1, tau2,
                                        u, v, vis, 0.0,0.0, sigma), dtype=DTPOL))
    
    datatable = np.array(datatable)
    #!AC: qfile and ufile must have exactly the same format as ifile
    #!AC: add some consistency check
    if not qfile==0:
        f = open(qfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['qvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                i += 1
            
    if not ufile==0:
        f = open(ufile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['uvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                i += 1
    
    # Return the datatable
    return Obsdata(ra, dec, rf, bw, datatable, tdata, source=src, mjd=mjd)

def load_obs_uvfits(filename, flipbl=False):
    """Load uvfits data from a uvfits file.
    """
        
    # Load the uvfits file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    
    # Load the array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    try:
        sefd = hdulist['AIPS AN'].data['SEFD']
    except KeyError:
        print("Warning! no SEFD data in UVfits file")
        sefd = np.zeros(len(tnames))
        
    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2], sefd[i]),
            dtype=DTARR) for i in range(len(tnames))]
            
    tarr = np.array(tarr)
    
    # Various header parameters
    ra = header['OBSRA'] * 12./180.
    dec = header['OBSDEC']
    src = header['OBJECT']
    if header['CTYPE4'] == 'FREQ':
        rf = header['CRVAL4']
        bw = header['CDELT4']
    else: raise Exception('Cannot find observing frequency!')
    
    
    # Mask to screen bad data
    rrweight = data['DATA'][:,0,0,0,0,0,2]
    llweight = data['DATA'][:,0,0,0,0,1,2]
    rlweight = data['DATA'][:,0,0,0,0,2,2]
    lrweight = data['DATA'][:,0,0,0,0,3,2]
    mask = (rrweight > 0) * (llweight > 0) * (rlweight > 0) * (lrweight > 0)
    
    # Obs Times
    jds = data['DATE'][mask]
    mjd = int(jdtomjd(np.min(jds)))
    
    #!AC: There seems to be different behavior here -
    #!AC: BU puts date in _DATE
    if len(set(data['DATE'])) > 2:
        times = np.array([mjdtogmt(jdtomjd(jd)) for jd in jds])
    else:
        times = data['_DATE'][mask] * 24.0
    
    # Integration times
    tints = data['INTTIM'][mask]
    
    # Sites - add names
    t1 = data['BASELINE'][mask].astype(int)/256
    t2 = data['BASELINE'][mask].astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    scopes_num = np.sort(list(set(np.hstack((t1,t2)))))
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])
    
    # Elevations (not in BU files)
    try:
        el1 = data['ELEV1'][mask]
        el2 = data['ELEV2'][mask]
    except KeyError:
        el1 = el2 = np.zeros(len(t1))

    # Opacities (not in BU files)
    try:
        tau1 = data['TAU1'][mask]
        tau2 = data['TAU2'][mask]
    except KeyError:
        tau1 = tau2 = np.zeros(len(t1))
        
    # Convert uv in lightsec to lambda by multiplying by rf
    try:
        u = data['UU---SIN'][mask] * rf
        v = data['VV---SIN'][mask] * rf
    except KeyError:
        try:
            u = data['UU'][mask] * rf
            v = data['VV'][mask] * rf
        except KeyError:
            try:
                u = data['UU--'][mask] * rf
                v = data['VV--'][mask] * rf
            except KeyError:
                raise Exception("Cant figure out column label for UV coords")
                    
    # Get vis data
    rr = data['DATA'][:,0,0,0,0,0,0][mask] + 1j*data['DATA'][:,0,0,0,0,0,1][mask]
    ll = data['DATA'][:,0,0,0,0,1,0][mask] + 1j*data['DATA'][:,0,0,0,0,1,1][mask]
    rl = data['DATA'][:,0,0,0,0,2,0][mask] + 1j*data['DATA'][:,0,0,0,0,2,1][mask]
    lr = data['DATA'][:,0,0,0,0,3,0][mask] + 1j*data['DATA'][:,0,0,0,0,3,1][mask]
    rrsig = 1/np.sqrt(rrweight[mask])
    llsig = 1/np.sqrt(llweight[mask])
    rlsig = 1/np.sqrt(rlweight[mask])
    lrsig = 1/np.sqrt(lrweight[mask])
    
    # Form stokes parameters
    ivis = (rr + ll)/2.0
    qvis = (rl + lr)/2.0
    uvis = (rl - lr)/(2.0j)
    isig = np.sqrt(rrsig**2 + llsig**2)/2.0
    qsig = np.sqrt(rlsig**2 + lrsig**2)/2.0
    usig = qsig
    
    # !AC Should sigma be the avg of the stokes sigmas, or just the I sigma?
    sigma = isig
    
    # !AC reverse sign of baselines for correct imaging?
    if flipbl:
        u = -u
        v = -v
    
    # Make a datatable
    # !AC Can I make this faster?
    datatable = []
    for i in range(len(times)):
        datatable.append(np.array((
                           times[i], tints[i],
                           t1[i], t2[i], el1[i], el2[i], tau1[i], tau2[i],
                           u[i], v[i],
                           ivis[i], qvis[i], uvis[i], sigma[i]
                           ), dtype=DTPOL
                         ))
    datatable = np.array(datatable)

    return Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=mjd, ampcal=True, phasecal=True)

def load_obs_oifits(filename, flux=1.0, specavg=-1, specbin=1, rescale_flux=False, renorm_flux=False, renorm_num=1, airmass=False,visdata=False):
    """Load data from an oifits file
        Does NOT currently support polarization
        """
    
    print('Warning: load_obs_oifits does NOT currently support polarimetric data!')
    if (len(filename.split('-')) >1):
        date_label = filename.split('-')[1]
        print("date_label=",date_label)
    #open oifits file and get visibilities
    oidata=oifits.open(filename)
    vis_data = oidata.vis
    print("vis_data.shape=",vis_data.shape)
    print("len(vis_data)=",len(vis_data))
    #amp = np.array([vis_data[i].visamp for i in range(len(vis_data))])


    #print "flux_data[0].station=",flux_data[1].station
    # get source info
    src = oidata.target[0].target
    ra = oidata.target[0].raep0.angle
    dec = oidata.target[0].decep0.angle
    
    # get annena info
    nAntennas = len(oidata.array[list(oidata.array.keys())[0]].station)
    #sites = np.array([str((oidata.array[oidata.array.keys()[0]].station[i])).replace(" ", "") for i in range(nAntennas)])
    sites = np.array([oidata.array[list(oidata.array.keys())[0]].station[i].sta_name for i in range(nAntennas)])
    arrayX = oidata.array[list(oidata.array.keys())[0]].arrxyz[0]
    arrayY = oidata.array[list(oidata.array.keys())[0]].arrxyz[1]
    arrayZ = oidata.array[list(oidata.array.keys())[0]].arrxyz[2]
    x = np.array([arrayX + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[0] for i in range(nAntennas)])
    y = np.array([arrayY + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[1] for i in range(nAntennas)])
    z = np.array([arrayZ + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[2] for i in range(nAntennas)])
    
    # get wavelength and corresponding frequencies
    wavelength = oidata.wavelength[list(oidata.wavelength.keys())[0]].eff_wave
    print("wavelength=", wavelength)
    nWavelengths = wavelength.shape[0]
    nwav_ori = nWavelengths
    print('load obs oifits nwave: ',nWavelengths)
    bandpass = oidata.wavelength[list(oidata.wavelength.keys())[0]].eff_band
    print("bandpass=", bandpass)
    frequency = C/wavelength
    print("frequency=", frequency)        #so this is the central frequency for each "channel"
    # todo: this result seems wrong...
    #bw = np.mean(2*(np.sqrt( bandpass**2*frequency**2 + C**2) - C)/bandpass)
    # this is AC's original treatment of bandwidth
    # since for the GRAVITY data now (low resolution), the effective bandwidth is wider than the separation between adjacent channels, here I need to put each channel as independent IFs in AIPS
    #rf = np.mean(frequency)
    #rf = frequency     # here I change the reference frequency to the first channel central frequency
    
    bw_ori = C/(wavelength-bandpass/2.0) - C/(wavelength+bandpass/2.0)
    print("bw_ori=",bw_ori)
    #print "rf=", rf
    #print "bw=", bw
    

    if specavg==1:
        print("*** specavg = 1, nwav_ori=",nwav_ori)
        if nwav_ori == 210:
            print("detect 210 wavelengths, I suppose you're using the MED mode.")
            print("specavg = 1, re-binning the data with every",specbin,"channels")
            n_bin = specbin
            nwav = nwav_ori/n_bin
            print("now nwav=",nwav)
            rf = np.zeros(nwav)
            bw = np.zeros(nwav)
            
            for i in range(nwav):
                freq_sum = 0.0
                bw_sum = 0.0
                for j in range(n_bin):
                    freq_sum = freq_sum + frequency[i*n_bin+j]
                    bw_sum = bw_sum + bw_ori[i*n_bin+j]
                rf[i] = freq_sum/n_bin
                bw[i] = bw_sum
            #rf[i] = frequency[1::n_bin]
            print("rf.shape=",rf.shape)
            print("rf=",rf)
            #for i in range(nwav):
            #    bw[i] = bw_ori[i*n_bin]+bw_ori[i*n_bin+1]+bw_ori[i*n_bin+2]
            print("bw.shape=",bw.shape)
            print("bw=",bw)
            
            #load in the u and v coordinate, this is the same as specavg=-1
            u = np.array([vis_data[i].ucoord for i in range(len(vis_data))])
            v = np.array([vis_data[i].vcoord for i in range(len(vis_data))])
            print("u.shape=",u.shape)
            print("u=",u)
            
            print("now do the transpose on u v")
            u = np.transpose(np.array([u for j in range(nwav)]))
            v = np.transpose(np.array([v for j in range(nwav)]))
            print("u.shape=",u.shape)
            print("u=",u)
            print("len(vis_data)=", len(vis_data))

        elif nwav_ori == 233:
            print("detect 233 wavelengths, I suppose you're using the MED mode.")
            print("specavg = 1, re-binning the data with every",specbin,"channels")
            n_bin = specbin
            nwav = int(np.floor(nwav_ori/n_bin))
            print("now nwav=",nwav)
            rf = np.zeros(nwav)
            bw = np.zeros(nwav)
            
            for i in range(nwav):
                freq_sum = 0.0
                bw_sum = 0.0
                for j in range(n_bin):
                    freq_sum = freq_sum + frequency[i*n_bin+j]
                    bw_sum = bw_sum + bw_ori[i*n_bin+j]
                rf[i] = freq_sum/n_bin
                bw[i] = bw_sum
            #rf[i] = frequency[1::n_bin]
            print("rf.shape=",rf.shape)
            print("rf=",rf)
            #for i in range(nwav):
            #    bw[i] = bw_ori[i*n_bin]+bw_ori[i*n_bin+1]+bw_ori[i*n_bin+2]
            print("bw.shape=",bw.shape)
            print("bw=",bw)
            
            #load in the u and v coordinate, this is the same as specavg=-1
            u = np.array([vis_data[i].ucoord for i in range(len(vis_data))])
            v = np.array([vis_data[i].vcoord for i in range(len(vis_data))])
            print("u.shape=",u.shape)
            print("u=",u)
            
            print("now do the transpose on u v")
            u = np.transpose(np.array([u for j in range(nwav)]))
            v = np.transpose(np.array([v for j in range(nwav)]))
            print("u.shape=",u.shape)
            print("u=",u)
            print("len(vis_data)=", len(vis_data))
            
        elif nwav_ori == 1628:
            print("detect 1628 wavelengths, I suppose you're using the HIGH mode.")
            print("specavg = 1, re-binning the data with every",specbin,"channels")
            n_bin = specbin
            nwav = int(np.floor(nwav_ori/n_bin))
            print("now nwav=",nwav)
            rf = np.zeros(nwav)
            bw = np.zeros(nwav)
            
            for i in range(nwav):
                freq_sum = 0.0
                bw_sum = 0.0
                for j in range(n_bin):
                    freq_sum = freq_sum + frequency[i*n_bin+j]
                    bw_sum = bw_sum + bw_ori[i*n_bin+j]
                rf[i] = freq_sum/n_bin
                bw[i] = bw_sum
            #rf[i] = frequency[1::n_bin]
            print("rf.shape=",rf.shape)
            print("rf=",rf)
            #for i in range(nwav):
            #    bw[i] = bw_ori[i*n_bin]+bw_ori[i*n_bin+1]+bw_ori[i*n_bin+2]
            print("bw.shape=",bw.shape)
            print("bw=",bw)
            
            #load in the u and v coordinate, this is the same as specavg=-1
            u = np.array([vis_data[i].ucoord for i in range(len(vis_data))])
            v = np.array([vis_data[i].vcoord for i in range(len(vis_data))])
            print("u.shape=",u.shape)
            print("u=",u)
            
            print("now do the transpose on u v")
            u = np.transpose(np.array([u for j in range(nwav)]))
            v = np.transpose(np.array([v for j in range(nwav)]))
            print("u.shape=",u.shape)
            print("u=",u)
            print("len(vis_data)=", len(vis_data))

##        sz=len(vis_data[0].visamp)
#        u=np.zeros((len(vis_data))); v=np.zeros((len(vis_data)))
#        amp=np.zeros((len(vis_data))); phase=np.zeros((len(vis_data)))
#        amperr=np.zeros((len(vis_data))); visphierr=np.zeros((len(vis_data)))
#        time=np.zeros((len(vis_data)))
#        for i in range(len(vis_data)):
##            good=np.where(vis_data[i].visamp > 0.)
##            w=1./vis_data[i].visamperr[good]**2.a
##            wph=1./vis_data[i].visphierr[good]**2.
##            w[np.where(vis_data[i].visamp==0.)]=0.
##            wph[np.where(vis_data[i].visamp==0.)]=0.
##            wsum=np.ma.sum(w)
#            if np.sum(vis_data[i].visamperr) > 0:
##                print 'shape u: ',u.shape,len(vis_data)
#                u[i]=np.ma.average(vis_data[i].ucoord/wavelength,weights=1./vis_data[i].visamperr**2.)
#                v[i]=np.ma.average(vis_data[i].vcoord/wavelength,weights=1./vis_data[i].visamperr**2.)
#                ampt,wsum=np.ma.average(vis_data[i].visamp,weights=1./vis_data[i].visamperr**2.,returned=True)
#                amp[i]=ampt
#                amperr[i]=np.sqrt(np.ma.sum(1./vis_data[i].visamperr**2.*(vis_data[i].visamp-ampt)**2.)/wsum)
##                amperr[i]=1./np.sqrt(wsum)
##                amperr[i]=np.ma.average(vis_data[i].visamperr,weights=1./vis_data[i].visamperr**2.)
#                if np.max(np.abs(vis_data[i].visphierr))==0:
#                    phase[i]=np.ma.average(vis_data[i].visphi,weights=1./vis_data[i].visamperr**2.)
#                else:
#                    phaset,wphsum=np.ma.average(vis_data[i].visphi,weights=1./vis_data[i].visphierr**2.,returned=True)
#                    phase[i]=phaset#; visphierr[i]=1./np.sqrt(wphsum)
#                    visphierr[i]=np.sqrt(np.ma.sum(1./vis_data[i].visphierr**2.*(vis_data[i].visphi-phaset)**2.)/wphsum)
#                time[i] = (ttime.mktime((vis_data[i].timeobs + datetime.timedelta(days=1)).timetuple()))/(60.0*60.0)
##                if i==0:
##                    print 'i=0: ',ampt,1./np.sqrt(wsum),phaset,1./np.sqrt(wphsum)
#        good=np.where(amp > 0.)
#        amp=amp[good]; u=u[good]; v=v[good]; phase=phase[good]
#        amperr=amperr[good]; visphierr=visphierr[good]; time=time[good]
##            if i==0:
##                print 'amp: ',np.ma.average(vis_data[i].visamp,weights=1./vis_data[i].visamperr**2.)
##                print 'phase: ',np.ma.average(vis_data[i].visphi,weights=1./vis_data[i].visamperr**2.)
    else:
    # get the u-v point for each visibility
    #AC's original setting of u and v
        #u = np.array([vis_data[i].ucoord/wavelength for i in range(len(vis_data))])
        #v = np.array([vis_data[i].vcoord/wavelength for i in range(len(vis_data))])
        
        print("specavg=-1, no spectral averaging.")
        nwav = nwav_ori
        rf = frequency     # here I change the reference frequency to the first channel central frequency
        bw = C/(wavelength-bandpass/2.0) - C/(wavelength+bandpass/2.0)
        print("rf=", rf)
        print("bw=", bw)
        
        #fgao's update
        u = np.array([vis_data[i].ucoord for i in range(len(vis_data))])
        v = np.array([vis_data[i].vcoord for i in range(len(vis_data))])

        print("u.shape=", u.shape)
        #print "original v=", v
        
        print("now do the transpose")
        u = np.transpose(np.array([u for j in range(nwav)]))
        v = np.transpose(np.array([v for j in range(nwav)]))
        print("len(vis_data)=", len(vis_data))
        
        print("u.shape=", u.shape)
        #print "u=", u
        
        print("v.shape=", v.shape)
        #print "v=", v
    
# get visibility info - currently the phase error is not being used properly
    amp = np.array([vis_data[i].visamp for i in range(len(vis_data))])
    print("before rescale flux:")
    print("amp[*][*]=", amp[0:15,0:6])

    
    print("amp.shape=", amp.shape)
    print("amp[*][*]=", amp[0:15,0:6])
    #print "flux_data_applied[**][**]=", flux_data_applied[0:15,0:6]
    #AC's original
    phase = np.array([vis_data[i].visphi for i in range(len(vis_data))])
    if visdata:
        phase = phase*180.0/np.pi
    #here I changed the sign of the phase in corresponding to the swap of baseline direction
    phase = -1*phase
 #   print 'load obs oifits amp phase shape: ',amp.shape,phase.shape
    print("phase[*][*]=", phase[0:6][0:10])

    amperr = np.array([vis_data[i].visamperr for i in range(len(vis_data))])
    visphierr = np.array([vis_data[i].visphierr for i in range(len(vis_data))])
    #print "notes on vis_data[i].timeobs"
    #for i in range(len(vis_data)):
    #    print i, vis_data[i].timeobs
    timeobs = np.array([vis_data[i].timeobs for i in range(len(vis_data))]) #convert to single number
    #print "timeobs=", timeobs
    print("timeobs.shape=",timeobs.shape)
    print("the first timeobs record : vis_data[0].timeobs =", vis_data[0].timeobs)
    dateobs = str(vis_data[0].timeobs)
    dateobs = str.strip(dateobs)[0:10]
    #print "dateobs=", dateobs

#return timeobs
    #!AC TODO - datetime not working!!!
    time = np.transpose(np.tile(np.array([(ttime.mktime((timeobs[i] + datetime.timedelta(days=1)).timetuple()))/(60.0*60.0)
                                          for i in range(len(timeobs))]), [nWavelengths, 1]))
    print("***")
    #print "time=", time
    timemjd = np.array([vis_data[i].timemjd for i in range(len(vis_data))])
    print("timemjd=", timemjd)
    print(timemjd.shape)
    timemjd_seq = timemjd.reshape(int(len(vis_data)/6),6)[:,0]
    print("timemjd_seq.shape=",timemjd_seq.shape)
    print("timemjd_seq=",timemjd_seq)
    timemjd = np.transpose(np.array([timemjd for j in range(nwav)]))
    print(timemjd.shape)
    
    ins = np.array([vis_data[i].ins for i in range(len(vis_data))])
    print("ins=", ins)
    print("ins.shape=", ins.shape)
    print("ins type=", type(ins))
    
    P1 = False
    P2 = False
    for xx in ins:
        if ('P1' in xx):
            P1 = True
        elif ('P2' in xx):
            P2 = True
    if P1*P2 == 1:
        dualpol = True
    elif P1*P2 == 0:
        dualpol = False
    else:
        print("don't recognize P1 and P2")

    if dualpol:
        n_frame = len(vis_data)/6/2
    else:
        n_frame = len(vis_data)

    print("dualpol=",dualpol)
    print("n_frame=",n_frame)

    ins = np.transpose(np.array([ins for j in range(nwav)]))
    
    #
    print("ins.shape=", ins.shape)
    print("ins type=", type(ins))

    arrname = np.array([vis_data[i].arrname for i in range(len(vis_data))])
    
    print("arrname=", arrname)
    
    arrname = arrname[0]   # here I just quote the first arrname in the whole list
    
    #######################################################################################################
    # here I need to resort the OI_FLUX data in the same order of mjd as the visibility data, then re-normalized the visibility amplitude

    # first read in the OI_FLUX data
    if renorm_flux:
        flux_data_raw = oidata.flux
        flux_data = np.array([flux_data_raw[i].fluxdata for i in range(len(flux_data_raw))])
        ins_flux = np.array([flux_data_raw[i].ins for i in range(len(flux_data_raw))])
        mjd_flux = np.array([flux_data_raw[i].timemjd for i in range(len(flux_data_raw))])
        print("flux_data.shape=",flux_data.shape)
        print("ins_flux.shape=",ins_flux.shape)
        print("mjd_flux.shape=",mjd_flux.shape)
        ins_flux = ins_flux.reshape(len(flux_data_raw)/4,4)
        mjd_flux = mjd_flux.reshape(len(flux_data_raw)/4,4)
    
        print("flux_data_raw.ins",ins_flux[:,0])
        print("flux_data=",flux_data[:,0])
        print("mjd_flux=",mjd_flux[:,0])
        print("timemjd_seq.shape=",timemjd_seq.shape)
        print("timemjd_seq=",timemjd_seq)
    
        # here I have to re-arrange the flux_data array because in OI_FLUX the P1 P2 is arranged differently than in OI_VIS in the raw data
        flux_data = flux_data.reshape(len(flux_data_raw)/4,4,nwav)
        print("FGAO2021: nwav=",nwav)
        print("before re-arrange flux_data=",flux_data[:,0,0])
        flux_data_tmp = np.zeros(len(flux_data_raw)*nwav).reshape(len(flux_data_raw)/4,4,nwav)
        mjd_flux_tmp = np.zeros(len(flux_data_raw)).reshape(len(flux_data_raw)/4,4)

        for i in range(n_frame):
            for j in range(len(mjd_flux)):
                if np.abs(timemjd_seq[i*2] - mjd_flux[j,0]) < 0.001 and ('P1' in ins_flux[j,0]):
                    flux_data_tmp[i*2,:,:] = flux_data[j,:,:]
                    mjd_flux_tmp[i*2,:] = mjd_flux[j,:]
                elif np.abs(timemjd_seq[i*2] - mjd_flux[j,0]) < 0.001 and ('P2' in ins_flux[j,0]):
                    flux_data_tmp[i*2+1,:,:] = flux_data[j,:,:]
                    mjd_flux_tmp[i*2+1,:] = mjd_flux[j,:]
                #else:
                #    print "not found matched mjd_flux for",flux_data_tmp[i*2,0,0],"i=",i


        print("flux_data_tmp=",flux_data_tmp[:,0])
        #if dualpol:
        #    for i in range(n_frame):
        #        flux_data_tmp[i*2,:,:] = flux_data[i,:,:]
        #        flux_data_tmp[i*2+1,:,:] = flux_data[i+n_frame,:,:]
        #        mjd_flux_tmp[i*2,:] = mjd_flux[i,:]
        #        mjd_flux_tmp[i*2+1,:] = mjd_flux[i+n_frame,:]

        flux_data = flux_data_tmp
        print("mjd_flux_tmp.shape=",mjd_flux_tmp.shape)
        print("mjd_flux_tmp=",mjd_flux_tmp[:,0])
        print("flux_data.shape=",flux_data.shape)
        #print "flux_data[0:30]=",flux_data[0:30,:,0:5]

        # here I read in the airmass correction from an external file to be applied on to the flux_data at different times/airmass/strehl
    if airmass:
        airmass_file = 'airmass-correction-'+date_label+'-new.txt'
        print("read in air mass data file from:",airmass_file)
        f_airmass = open(airmass_file,'r')
        airmass_raw = f_airmass.readlines()
        n_frame_airmass = len(airmass_raw) - 5
        airmass_poly_term = np.zeros(len(airmass_raw[0].split(' ')))
        airmass_poly_par = np.zeros(len(airmass_raw[1].split(' '))*4).reshape(4, len(airmass_raw[1].split(' ')))
        print("airmass_poly_par.shape=",airmass_poly_par.shape)
        airmass_data = np.zeros(n_frame_airmass*6).reshape(n_frame_airmass,6)
        airmass_coe = np.zeros(n_frame_airmass*5).reshape(n_frame_airmass,5) # mjd, UT4, UT3, UT2, UT1
        
        for i in range(4):# loop around 4 telescopes
            k = 0
            for j in range(len(airmass_raw[i+1].split(' '))):
                if (airmass_raw[i+1].split(' ')[j] != '') and (airmass_raw[i+1].split(' ')[j] != '\n'):
                    airmass_poly_par[i,k] = float(airmass_raw[i+1].split(' ')[j])
                    print("i=",i,"j=",j,"k=",k,"airmass_poly_par[i,k]=",airmass_poly_par[i,k])
                    k = k + 1
        my_poly1 = np.poly1d(np.trim_zeros(airmass_poly_par[0,:]))
        my_poly2 = np.poly1d(np.trim_zeros(airmass_poly_par[1,:]))
        my_poly3 = np.poly1d(np.trim_zeros(airmass_poly_par[2,:]))
        my_poly4 = np.poly1d(np.trim_zeros(airmass_poly_par[3,:]))

        print(" my_poly1=", my_poly1)
        print(" my_poly2=", my_poly2)
        print(" my_poly3=", my_poly3)
        print(" my_poly4=", my_poly4)

        for i in range(n_frame_airmass):
            for j in range(6):
                airmass_data[i,j] = airmass_raw[i+5].split(' ')[j]

        for i in range(n_frame_airmass):
            airmass_coe[i,0] = airmass_data[i,0]
            airmass_coe[i,1] = my_poly1(airmass_data[i,2])
            airmass_coe[i,2] = my_poly2(airmass_data[i,3])
            airmass_coe[i,3] = my_poly3(airmass_data[i,4])
            airmass_coe[i,4] = my_poly4(airmass_data[i,5])
#            airmass_coe[i,1] = np.exp(-2*airmass_data[i,1])/np.max(np.exp(-2*airmass_data[:,1]))*my_poly1(airmass_data[i,2])
#            airmass_coe[i,2] = np.exp(-2*airmass_data[i,1])/np.max(np.exp(-2*airmass_data[:,1]))*my_poly2(airmass_data[i,3])
#            airmass_coe[i,3] = np.exp(-2*airmass_data[i,1])/np.max(np.exp(-2*airmass_data[:,1]))*my_poly3(airmass_data[i,4])
#            airmass_coe[i,4] = np.exp(-2*airmass_data[i,1])/np.max(np.exp(-2*airmass_data[:,1]))*my_poly4(airmass_data[i,5])

        print("airmass_coe[:,0]=",airmass_coe[:,0])

        k=0
        for i in range(len(flux_data_raw)/4):
            for j in range(n_frame_airmass):
                if np.abs(mjd_flux_tmp[i,0] - airmass_coe[j,0]) < 0.001:
                    print("bingo")
                    k = k + 1
                    flux_data[i,0,:] = flux_data[i,0,:] / airmass_coe[j,1]
                    flux_data[i,1,:] = flux_data[i,1,:] / airmass_coe[j,2]
                    flux_data[i,2,:] = flux_data[i,2,:] / airmass_coe[j,3]
                    flux_data[i,3,:] = flux_data[i,3,:] / airmass_coe[j,4]
        print("k=",k)
        
    if rescale_flux and renorm_flux:
        print("FGAO2021: nwav=",nwav)
        # prepare to expand the flux table from telescope-base to baseline-base
        flux_data_applied = np.zeros(len(flux_data_raw)/4*6*nwav).reshape(len(flux_data_raw)/4,6,nwav)
        flux_data_applied_renorm = np.zeros(len(flux_data_raw)/4*6*nwav).reshape(len(flux_data_raw)/4,6,nwav)
        for i in range(len(flux_data_raw)/4):
            for j in range(4):
                for k in range(nwav):
                    if flux_data[i,j,k] < 0:
                        print("negative flux at i=",i,"j=",j,"k=",k,flux_data[i,j,k])
                        #flux_data[i,j,k] = flux_data[i,j,k] * (-1)
                        flux_data[i,j,k] = 0
            flux_data_applied[i,0,:] = np.sqrt(flux_data[i,0,:]*flux_data[i,1,:])
            flux_data_applied[i,1,:] = np.sqrt(flux_data[i,0,:]*flux_data[i,2,:])
            flux_data_applied[i,2,:] = np.sqrt(flux_data[i,0,:]*flux_data[i,3,:])
            flux_data_applied[i,3,:] = np.sqrt(flux_data[i,1,:]*flux_data[i,2,:])
            flux_data_applied[i,4,:] = np.sqrt(flux_data[i,1,:]*flux_data[i,3,:])
            flux_data_applied[i,5,:] = np.sqrt(flux_data[i,2,:]*flux_data[i,3,:])

    
        print("flux_data_applied.shape=",flux_data_applied.shape)
        print("flux_data_applied[0,0,:]=",flux_data_applied[0,0,:])
        #print "flux_data_applied[0:30] before renorm=",flux_data_applied[0:10,:,0:5]
        #flux_data_applied = np.nan_to_num(flux_data_applied,nan=-1111111,copy=False)
        flux_data_applied[np.isnan(flux_data_applied)] = 1.0
        #print "flux_data_applied[0:30] before renorm after nan_to_num=",flux_data_applied[0:10,:,0:5]
    
    if renorm_flux:
        print("***** reform_flux = TRUE, using frame#",renorm_num)
        for i in range(len(flux_data_raw)/4/2):
            j = (renorm_num-1)*2
            k = (renorm_num-1)*2 + 1
            
            #flux_data_applied_renorm[i*2,:,:]   = flux_data_applied[i*2,:,:]/flux_data_applied[renorm_num-1,:,:]
            #flux_data_applied_renorm[i*2+1,:,:] = flux_data_applied[i*2+1,:,:]/flux_data_applied[renorm_num,:,:]
            
            flux_data_applied_renorm[i*2,:,:]   = flux_data_applied[i*2,:,:]/flux_data_applied[j,:,:]
            flux_data_applied_renorm[i*2+1,:,:] = flux_data_applied[i*2+1,:,:]/flux_data_applied[k,:,:]
            print("the data I'm using for renorm is j:",flux_data[j,0,0],flux_data[j,1,0],flux_data_applied[j,0,0])
            print("the data I'm using for renorm is k:",flux_data[k,0,0],flux_data[k,1,0],flux_data_applied[k,0,0])
            print("mjd_flux_tmp[j]=",mjd_flux_tmp[j,:])
    #else:
    #    flux_data_applied_renorm = flux_data_applied

    #print "flux_data_applied.shape=",flux_data_applied.shape
    #print "flux_data_applied[0:30] after renorm=",flux_data_applied_renorm[0:10,:,0:5]
    if renorm_flux:
        flux_data_applied_renorm = flux_data_applied_renorm.reshape(len(flux_data_raw)/4*6,nwav)
    #print "flux_data_applied_renorm.shape=",flux_data_applied_renorm.shape
    #print "flux_data_applied[*][*] after reshape=",flux_data_applied[0:10,3:8]

    if rescale_flux:
        amp = amp * flux_data_applied_renorm



    #######################################################################################################


    #timemjd = np.array([vis_data[i].timemjd for i in range(len(vis_data))])
    #timemjd = vis_data[0].timemjd
#print timemjd
#    print 'vb oifits tint: ',tint
        #time = np.array([vis_data[i].timeobs for i in range(len(vis_data))])   ###added by myself
    #time = np.transpose(np.tile(np.array([(ttime.mktime(timeobs[i].timetuple()) - ttime.mktime(datetime.datetime.utcfromtimestamp(0).timetuple()))/(60.0*60.0)
    #                                      for i in range(len(timeobs))]), [nWavelengths, 1]))
    
    #my own test
    #time = np.array([vis_data[i].mjd for i in range(len(vis_data))])

    # integration time
    tint = np.array([vis_data[i].int_time for i in range(len(vis_data))])
#    print 'vb oifits tint: ',tint
#    if not all(tint[0] == item for item in np.reshape(tint, (-1)) ):
#        raise TypeError("The time integrations for each visibility are different")
    tint = tint[0]
    tint = tint * np.ones( amp.shape )

    # get telescope names for each visibility
    t1 = np.transpose(np.tile( np.array([ vis_data[i].station[0].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))
    print("t1=",t1)
    print("t1.shape=",t1.shape)
    t2 = np.transpose(np.tile( np.array([ vis_data[i].station[1].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))
    #t1 = np.transpose(np.tile( np.array([ str(vis_data[i].station[0]).replace(" ", "") for i in range(len(vis_data))]), [nWavelengths,1]))
    #t2 = np.transpose(np.tile( np.array([ str(vis_data[i].station[1]).replace(" ", "") for i in range(len(vis_data))]), [nWavelengths,1]))

    # dummy variables
    el1 = -np.ones(amp.shape)
    el2 = -np.ones(amp.shape)
    tau1 = -np.ones(amp.shape)
    tau2 = -np.ones(amp.shape)
    qvis = -np.ones(amp.shape)
    uvis = -np.ones(amp.shape)
    sefd = -np.ones(x.shape)

# vectorize
    print("now doing the ravel")
    time = time.ravel()
    tint = tint.ravel()
    t1 = t1.ravel()
    t2 = t2.ravel()
    el1 = el1.ravel()
    el2 = el2.ravel()
    tau1 = tau1.ravel()
    tau2 = tau2.ravel()
    u = u.ravel()
    v = v.ravel()
    #vis = amp.ravel() * np.exp ( -1j * phase.ravel() * np.pi/180.0 )
    vis = np.zeros(u.shape,dtype=complex).reshape(len(vis_data),nwav)
    vis_sum = np.zeros(u.shape,dtype=complex).reshape(len(vis_data),nwav)
    vis_mid = np.zeros(u.shape,dtype=complex).reshape(int(len(vis_data)/6),6,nwav)
    vis_mid_cal = np.zeros(u.shape,dtype=complex).reshape(int(len(vis_data)/6),6,nwav)
    print("###############################################")
    print("vis.shape=",vis.shape)
    vis_ori = amp * np.exp ( -1j * phase * np.pi/180.0 )
    print("vis_ori.shape=",vis_ori.shape)
    #print "vis_ori[12,:]=",vis_ori[12,:]
    if visdata:
        vis_mid = vis_ori.reshape(len(vis_data)/6,6,nwav)
        print("vis_mid.shape=",vis_mid.shape)
        print("vis_mid[0,:]=",vis_mid[:,0,0])
        print("vis_mid[6,:,:]=",vis_mid[6,:,:])
        print("vis_mid[6,:,:].shape=",vis_mid[6,:,:].shape)
        print("vis_mid[0,:,:]=",vis_mid[0,:,:])
        for i in range(len(vis_data)/6/2):
            vis_mid_cal[i*2,:,:] = vis_mid[i*2,:,:]/vis_mid[0,:,:]
            vis_mid_cal[i*2+1,:,:] = vis_mid[i*2+1,:,:]/vis_mid[1,:,:]

        print("vis_mid_cal[0,:,:]=",vis_mid_cal[0,:,:])
        print("vis_mid_cal[0,:,:].shape=",vis_mid_cal[0,:,:].shape)
    #re-binning the visibility data
    if specavg==1:
        if nwav_ori == 210 or nwav_ori == 233 or nwav_ori == 1628:
            for i in range(nwav):
                for j in range(n_bin):
                    vis_sum[:,i] = vis_sum[:,i] + vis_ori[:,i*n_bin+j]
                    #vis_sum = vis_sum + vis_ori[:,i*3+j]
                #vis[:,i] = (vis_ori[:,i*3]+vis_ori[:,i*3+1]+vis_ori[:,i*3+2])/n_bin
                vis[:,i] = vis_sum[:,i]/n_bin
            print("nwav=",nwav)
    else:
        vis = vis_ori

    print("vis[0,:]=",vis[0,:])
    
    
    #now do the ravel of vis
    if visdata:
        vis = vis_mid_cal.ravel()
    else:
        vis = vis.ravel()

    print("vis.shape=", vis.shape)
    #for i in range(100):
    #    print i, vis[i]
    print("amp.ravel=", amp.ravel())
    
    print("vis=",vis)
    qvis = qvis.ravel()
    uvis = uvis.ravel()
    amperr = amperr.ravel()
    phrad = phase.ravel()*np.pi/180.; pherr = visphierr.ravel()*np.pi/180.
    #realerr = np.abs(np.real(vis))*np.sqrt(np.cos(phrad)**2.*amperr**2.+amp.ravel()**2.*np.sin(phrad)**2.*pherr**2./phrad**2.)
    #imagerr = np.abs(np.imag(vis))*np.sqrt(np.sin(phrad)**2.*amperr**2.+amp.ravel()**2.*np.cos(phrad)**2.*pherr**2./phrad**2.)
    
    print("amperr.shape=", amperr.shape)
    #for i in range(0,100,1):
    #    print i, amperr.ravel()[i]
    
    print('load obs oifits after ravel shape: ',vis.shape)
    print("len(vis)=", len(vis))
    
    print("u[0]=",u[0])
    
    #TODO - check that we are properly using the error from the amplitude and phase
    
    # create data tables
    datatable = np.array([ (time[i], tint[i], t1[i], t2[i], el1[i], el2[i], tau1[i], tau2[i], u[i], v[i], flux*vis[i], qvis[i], uvis[i], flux*amperr[i]) for i in range(len(vis))], dtype=DTPOL)
    #    datatable = np.ma.array([ (time[i], tint[i], t1[i], t2[i], el1[i], el2[i], tau1[i], tau2[i], u[i], v[i], flux*vis[i], qvis[i], uvis[i], flux*(realerr[i]+1j*imagerr[i])) for i in range(len(vis))], dtype=DTPOL)
    tarr = np.array([ (sites[i], x[i], y[i], z[i], sefd[i]) for i in range(nAntennas)], dtype=DTARR)
    
    #print "printing datatable now"
    #for i in range(150):
    #    print i, datatable[i]
    
    print('load obs oifits datatable shape: ',datatable.shape)
    print('test in load_obs_oifits')
    #print time[0]
    # return object
    print("nwav=",nwav)
    print("bw=",bw)
    print("rf=",rf)
    return Obsdata(ra, dec, rf, bw, datatable, tarr, timemjd, dateobs, ins, arrname, nwav, source=src, mjd=time[0], ampcal=False, phasecal=False)
    
def load_im_txt(filename, pulse=pulses.trianglePulse2D):
    """Read in an image from a text file and create an Image object
       Text file should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """
    
    # TODO !AC should pulse type be in header?
    # Read the header
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60. + float(ra[6])/3600.
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60. + float(dec[6])/3600.)
    mjd = float(file.readline().split()[2])
    rf = float(file.readline().split()[2]) * 1e9
    xdim = file.readline().split()
    xdim_p = int(xdim[2])
    psize_x = float(xdim[4])*RADPERAS/xdim_p
    ydim = file.readline().split()
    ydim_p = int(ydim[2])
    psize_y = float(ydim[4])*RADPERAS/ydim_p
    file.close()
    
    if psize_x != psize_y:
        raise Exception("Pixel dimensions in x and y are inconsistent!")
    
    # Load the data, convert to list format, make object
    datatable = np.loadtxt(filename, dtype=float)
    image = datatable[:,2].reshape(ydim_p, xdim_p)
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    # Look for Stokes Q and U
    qimage = uimage = np.zeros(image.shape)
    if datatable.shape[1] == 5:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)
    
    if np.any((qimage != 0) + (uimage != 0)):
        print('Loaded Stokes I, Q, and U images')
        outim.add_qu(qimage, uimage)
    else:
        print('Loaded Stokes I image only')
    
    return outim
    
def load_im_fits(filename, punit="deg", pulse=pulses.trianglePulse2D):
    """Read in an image from a FITS file and create an Image object
    """

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
        
    # Open the FITS file
    hdulist = fits.open(filename)
    
    # Assume stokes I is the primary hdu
    header = hdulist[0].header
    
    # Read some header values
    ra = header['OBSRA']*12/180.
    dec = header['OBSDEC']
    xdim_p = header['NAXIS1']
    psize_x = np.abs(header['CDELT1']) * pscl
    dim_p = header['NAXIS2']
    psize_y = np.abs(header['CDELT2']) * pscl
    
    if 'MJD' in list(header.keys()): mjd = header['MJD']
    else: mjd = 48277.0
    
    if 'FREQ' in list(header.keys()): rf = header['FREQ']
    else: rf = 230e9
    
    if 'OBJECT' in list(header.keys()): src = header['OBJECT']
    else: src = 'SgrA'
    
    # Get the image and create the object
    data = hdulist[0].data
    data = data.reshape((data.shape[-2],data.shape[-1]))
    image = data[::-1,:] # flip y-axis!
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    # Look for Stokes Q and U
    qimage = uimage = np.array([])
    for hdu in hdulist[1:]:
        header = hdu.header
        data = hdu.data
        data = data.reshape((data.shape[-2],data.shape[-1]))
        if 'STOKES' in list(header.keys()) and header['STOKES'] == 'Q':
            qimage = data[::-1,:] # flip y-axis!
        if 'STOKES' in list(header.keys()) and header['STOKES'] == 'U':
            uimage = data[::-1,:] # flip y-axis!
    if qimage.shape == uimage.shape == image.shape:
        print('Loaded Stokes I, Q, and U images')
        outim.add_qu(qimage, uimage)
    else:
        print('Loaded Stokes I image only')
                
    return outim

##################################################################################################
# Image Construction Functions
##################################################################################################

def resample_square(im, xdim_new, ker_size=5):
    """Return a new image object that is resampled to the new dimensions xdim x ydim"""
    
    # !AC TODO work with not square image? New xdim & ydim must be compatible!
    if im.xdim != im.ydim:
        raise Exception("Image must be square (for now)!")
    if im.pulse == pulses.deltaPulse2D:
        raise Exception("This function only works on continuously parametrized images: does not work with delta pulses!")
    
    ydim_new = xdim_new
    fov = im.xdim * im.psize
    psize_new = fov / xdim_new
    ij = np.array([[[i*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0, j*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0]
                    for i in np.arange(0, -im.xdim, -1)]
                    for j in np.arange(0, -im.ydim, -1)]).reshape((im.xdim*im.ydim, 2))
    def im_new(x,y):
        mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
        return np.sum([im.imvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
    
    out = np.array([[im_new(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)]
                      for y in np.arange(0, -ydim_new, -1)] )

                      
    # TODO !AC check if this normalization is correct!
    scaling = np.sum(im.imvec) / np.sum(out)
    out *= scaling
    outim = Image(out, psize_new, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    
    # Q and U images
    if len(im.qvec):
        def im_new_q(x,y):
            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) *
                    ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
            return np.sum([im.qvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
        def im_new_u(x,y):
            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) *
                    ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
            return np.sum([im.uvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
        
        outq = np.array([[im_new_q(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)]
                      for y in np.arange(0, -ydim_new, -1)] )
        outu = np.array([[im_new_u(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)]
                      for y in np.arange(0, -ydim_new, -1)] )
        outq *= scaling
        outu *= scaling
        outim.add_qu(outq, outu)
        
    return outim
    
def make_square(obs, npix, fov,pulse=pulses.trianglePulse2D):
    """Make an empty prior image
       obs is an observation object
       fov is in radians
    """
    pdim = fov/npix
    im = np.zeros((npix,npix))
    return Image(im, pdim, obs.ra, obs.dec, rf=obs.rf, source=obs.source, mjd=obs.mjd, pulse=pulse)

def add_flat(im, flux):
    """Add flat background to an image"""
    
    imout = (im.imvec + (flux/float(len(im.imvec))) * np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return out

def add_tophat(im, flux, radius):
    """Add tophat flux to an image"""
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize
    
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
    # !AC handle actual zeros?
    hat = np.array([[1.0 if np.sqrt(i**2+j**2) <= radius else EP
                      for i in xlist]
                      for j in ylist])
    
    # !AC think more carefully about the different cases for array size here
    hat = hat[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (hat * flux/np.sum(hat))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)
    return out

def add_gauss(im, flux, beamparams, x=0, y=0):
    """Add a gaussian to an image
       beamparams is [fwhm_maj, fwhm_min, theta], all in rad
       x,y are gaussian position in rad
       theta is the orientation angle measured E of N
    """
    

    sigma_maj = beamparams[0] / (2. * np.sqrt(2. * np.log(2.)))
    sigma_min = beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
    cth = np.cos(beamparams[2])
    sth = np.sin(beamparams[2])
    
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
    
    gauss = np.array([[np.exp(-((j-y)*cth + (i-x)*sth)**2/(2*sigma_maj**2) - ((i-x)*cth - (j-y)*sth)**2/(2.*sigma_min**2))
                      for i in xlist]
                      for j in ylist])
  
    # !AC think more carefully about the different cases for array size here
    gauss = gauss[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (gauss * flux/np.sum(gauss))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return out


def add_const_m(im, mag, angle):
    """Add a constant fractional polarization to image
       angle is in radians"""
    
    if not (0 < mag < 1):
        raise Exception("fractional polarization magnitude must be beween 0 and 1!")
    
    imi = im.imvec.reshape(im.ydim,im.xdim)
    imq = qimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    imu = uimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imi, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    out.add_qu(imq, imu)
    return out
    
##################################################################################################
# Image domain blurring Functions
##################################################################################################
def blur_gauss(image, beamparams, frac, frac_pol=0):
    """Blur image with a Gaussian beam defined by beamparams
       beamparams is [FWHMmaj, FWHMmin, theta], all in radian
    """
    
    im = (image.imvec).reshape(image.ydim, image.xdim)
    if len(image.qvec):
        qim = (image.qvec).reshape(image.ydim, image.xdim)
        uim = (image.uvec).reshape(image.ydim, image.xdim)
    xfov = image.xdim * image.psize
    yfov = image.ydim * image.psize
    xlist = np.arange(0,-image.xdim,-1)*image.psize + (image.psize*image.xdim)/2.0 - image.psize/2.0
    ylist = np.arange(0,-image.ydim,-1)*image.psize + (image.psize*image.ydim)/2.0 - image.psize/2.0
    
    if beamparams[0] > 0.0:
        sigma_maj = frac * beamparams[0] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = frac * beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])
        gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*sigma_maj**2) - (i*cth - j*sth)**2/(2.*sigma_min**2))
                                  for i in xlist]
                                  for j in ylist])

        # !AC think more carefully about the different image size cases here
        gauss = gauss[0:image.ydim, 0:image.xdim]
        gauss = gauss / np.sum(gauss) # normalize to 1
        
        # Convolve
        im = scipy.signal.fftconvolve(gauss, im, mode='same')


    if frac_pol:
        if not len(image.qvec):
            raise Exception("There is no polarized image!")
                
        sigma_maj = frac_pol * beamparams[0] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = frac_pol * beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])
        gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*sigma_maj**2) - (i*cth - j*sth)**2/(2.*sigma_min**2))
                                  for i in xlist]
                                  for j in ylist])
        

        # !AC think more carefully about the different cases here
        gauss = gauss[0:image.ydim, 0:image.xdim]
        gauss = gauss / np.sum(gauss) # normalize to 1
        
        # Convolve
        qim = scipy.signal.fftconvolve(gauss, qim, mode='same')
        uim = scipy.signal.fftconvolve(gauss, uim, mode='same')
                                  
    
    out = Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd, pulse=image.pulse)
    if len(image.qvec):
        out.add_qu(qim, uim)
    return out
        
##################################################################################################
# Scattering Functions
##################################################################################################
def deblur(obs):
    """Deblur the observation obs by dividing with the Sgr A* scattering kernel.
       Returns a new observation.
    """
    
    datatable = np.array(obs.data, copy=True)
    vis = datatable['vis']
    qvis = datatable['qvis']
    uvis = datatable['uvis']
    sigma = datatable['sigma']
    u = datatable['u']
    v = datatable['v']
    
    for i in range(len(vis)):
        ker = sgra_kernel_uv(obs.rf, u[i], v[i])
        vis[i] = vis[i] / ker
        qvis[i] = qvis[i] / ker
        uvis[i] = uvis[i] / ker
        sigma[i] = sigma[i] / ker
    
    datatable['vis'] = vis
    datatable['qvis'] = qvis
    datatable['uvis'] = uvis
    datatable['sigma'] = sigma
    
    obsdeblur = Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, datatable, obs.tarr)
    return obsdeblur
    
def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda),
       at a given frequency rf (in Hz).
       Values from Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = POS_ANG * DEGREE
    
    
    # Covarience matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u,v])
    
    
    x2 = np.dot(uv, np.dot(m, uv))
    g = np.exp(-2 * np.pi**2 * x2)
    
    return g

def sgra_kernel_params(rf):
    """Return elliptical gaussian parameters in radian for the Sgr A* scattering ellipse at a given frequency
       Values from Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    fwhm_maj_rf = FWHM_MAJ * (lcm**2)  * RADPERUAS
    fwhm_min_rf = FWHM_MIN * (lcm**2)  * RADPERUAS
    theta = POS_ANG * DEGREE
    
    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])
                                     
##################################################################################################
# Other Functions
##################################################################################################

def paritycompare(perm1, perm2):
    """Compare the parity of two permutations.
       Assume both lists are equal length and with same elements
       Copied from: http://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-equal-parity
    """
    
    perm2 = list(perm2)
    perm2_map = dict((v, i) for i,v in enumerate(perm2))
    transCount=0
    for loc, p1 in enumerate(perm1):
        p2 = perm2[loc]
        if p1 != p2:
            sloc = perm2_map[p1]
            perm2[loc], perm2[sloc] = p1, p2
            perm2_map[p1], perm2_map[p2] = sloc, loc
            transCount += 1
    
    if not (transCount % 2): return 1
    else: return  -1
    
def merr(sigma, I, m):
    """Return the error in mbreve real and imaginary parts"""
    return sigma * np.sqrt((2 + np.abs(m)**2)/ (np.abs(I) ** 2))
       
def ticks(axisdim, psize, nticks=8):
    """Return a list of ticklocs and ticklabels
       psize should be in desired units
    """
    
    axisdim = int(axisdim)
    nticks = int(nticks)
    if not axisdim % 2: axisdim += 1
    if nticks % 2: nticks -= 1
    tickspacing = float((axisdim-1))/nticks
    ticklocs = np.arange(0, axisdim+1, tickspacing)
    ticklabels= np.around(psize * np.arange((axisdim-1)/2., -(axisdim)/2., -tickspacing), decimals=1)
    return (ticklocs, ticklabels)
    
def rastring(ra):
    """Convert a ra in fractional hours to formatted string"""
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-m/60.)*3600.
    out = "%2i h %2i m %2.4f s" % (h,m,s)
    return out

def decstring(dec):
    """Convert a dec in fractional degrees to formatted string"""
    
    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-m/60.)*3600.
    out = "%2i deg %2i m %2.4f s" % (deg,m,s)
    return out

def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string"""
    
    if gmt > 24.0: gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-m/60.)*3600.
    out = "%02i:%02i:%2.4f" % (h,m,s)
    return out

def fracmjd(mjd, gmt):
    """Convert a int mjd + gmt (frac. hr.) into a fractional mjd"""
    
    return int(mjd) + gmt/24.

def mjdtogmt(mjd):
    """Return the gmt of a fractional mjd, in days"""
    
    return (mjd - int(mjd)) * 24.0
    
def jdtomjd(jd):
    """Return the mjd of a jd"""
    
    return jd - 2400000.5
    
def earthrot(vec, theta):
    """Rotate a vector about the z-direction by theta (degrees)"""
    
    x = theta * DEGREE
    return np.dot(np.array(((np.cos(x),-np.sin(x),0),(np.sin(x),np.cos(x),0),(0,0,1))),vec)

def elev(obsvec, sourcevec):
    """Return the elevation of a source with respect to an observer in degrees"""
    anglebtw = np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec)
    el = 90 - np.arccos(anglebtw)/DEGREE
    return el
        
def elevcut(obsvec,sourcevec):
    """Return True if a source is observable by a telescope vector"""
    angle = elev(obsvec, sourcevec)
    return ELEV_LOW < angle < ELEV_HIGH

        
def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline (2-bit quantization)"""
    
    return np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88

def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean"""
    
    return np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var"""
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1"""
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

      
def ftmatrix(pdim, xdim, ydim, uvlist, pulse=pulses.deltaPulse2D):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """

    # TODO : there is a residual value for the center being around 0, maybe we should chop this off to be exactly 0
    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0

    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))
    return ftmatrices

def amp_debias(vis, sigma):
    """Return debiased visibility amplitudes"""
    
    # TODO: what to do if deb2 < 0?
    deb2 = np.abs(vis)**2 - np.abs(sigma)**2
    if type(deb2) == float or type(deb2)==np.float64:
        if deb2 < 0.0: return np.abs(vis)
        else: return np.sqrt(deb2)
    else:
        lowsnr = deb2 < 0.0
        deb2[lowsnr] = np.abs(vis[lowsnr])**2
        return np.sqrt(deb2)
#        return vis

def add_noise(obs, opacity_errs=True, ampcal=True, phasecal=True, gainp=GAINPDEF):
    """Re-compute sigmas from SEFDS and add noise with gain & phase errors
       Be very careful using outside of Image.observe!"""
    
    if (not opacity_errs) and (not ampcal):
        raise Exception("ampcal=False requires opacity_errs=True!")
        
    # Get data
    obslist = obs.tlist()
    
    # Remove possible conjugate baselines:
    obsdata = []
    blpairs = []
    for tlist in obslist:
        for dat in tlist:
            if not ((dat['t1'], dat['t2']) in blpairs
                 or (dat['t2'], dat['t1']) in blpairs):
                 obsdata.append(dat)
                 
    obsdata = np.array(obsdata, dtype=DTPOL)
                      
    # Extract data
    sites = obsdata[['t1','t2']].view(('a32',2))
    elevs = obsdata[['el1','el2']].view(('f8',2))
    taus = obsdata[['tau1','tau2']].view(('f8',2))
    time = obsdata[['time']].view(('f8',1))
    tint = obsdata[['tint']].view(('f8',1))
    uv = obsdata[['u','v']].view(('f8',2))
    vis = obsdata[['vis']].view(('c16',1))
    qvis = obsdata[['qvis']].view(('c16',1))
    uvis = obsdata[['uvis']].view(('c16',1))

    bw = obs.bw
    
    # Overwrite sigma_cleans with values computed from the SEFDs
    sigma_clean = np.array([blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefd'], obs.tarr[obs.tkey[sites[i][1]]]['sefd'], tint[i], bw) for i in range(len(tint))])
 
    # Estimated noise using no gain and estimated opacity
    if opacity_errs:
        sigma_est = sigma_clean * np.sqrt(np.exp(taus[:,0]/(EP+np.sin(elevs[:,0]*DEGREE)) + taus[:,1]/(EP+np.sin(elevs[:,1]*DEGREE))))
    else:
        sigma_est = sigma_clean
        
    # Add gain and opacity fluctuations to the true noise
    if not ampcal:
        # Amplitude gain
        gain1 = np.abs(np.array([1.0 + gainp * hashrandn(sites[i,0], 'gain')
                        + gainp * hashrandn(sites[i,0], 'gain', time[i]) for i in range(len(time))]))
        gain2 = np.abs(np.array([1.0 + gainp * hashrandn(sites[i,1], 'gain')
                        + gainp * hashrandn(sites[i,1], 'gain', time[i]) for i in range(len(time))]))
        
        # Opacity
        tau1 = np.array([taus[i,0]*(1 + gainp * hashrandn(sites[i,0], 'tau', time[i])) for i in range(len(time))])
        tau2 = np.array([taus[i,1]*(1 + gainp * hashrandn(sites[i,1], 'tau', time[i])) for i in range(len(time))])

        # Correct noise RMS for gain variation and opacity
        sigma_true = sigma_clean / np.sqrt(gain1 * gain2)
        sigma_true = sigma_true * np.sqrt(np.exp(tau1/(EP+np.sin(elevs[:,0]*DEGREE)) + tau2/(EP+np.sin(elevs[:,1]*DEGREE))))
    
    else:
        sigma_true = sigma_est
    
    # Add the noise and the gain error to the true visibilities
    vis  = (vis + cerror(sigma_true))  * (sigma_est/sigma_true)
    qvis = (qvis + cerror(sigma_true)) * (sigma_est/sigma_true)
    uvis = (uvis + cerror(sigma_true)) * (sigma_est/sigma_true)
  
    # Add random atmospheric phases
    if not phasecal:
        phase1 = np.array([2 * np.pi * hashrand(sites[i,0], 'phase', time[i]) for i in range(len(time))])
        phase2 = np.array([2 * np.pi * hashrand(sites[i,1], 'phase', time[i]) for i in range(len(time))])
        
        vis *= np.exp(1j * (phase2-phase1))
        qvis *= np.exp(1j * (phase2-phase1))
        uvis *= np.exp(1j * (phase2-phase1))
                
    # Put the visibilities estimated errors back in the obsdata array
    obsdata['vis'] = vis
    obsdata['qvis'] = qvis
    obsdata['uvis'] = uvis
    obsdata['sigma'] = sigma_est
    
    # Return observation object
    out =  Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr, source=obs.source, mjd=obs.mjd, ampcal=ampcal, phasecal=phasecal)
    return out



