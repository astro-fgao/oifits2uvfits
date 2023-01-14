import vlbi_imaging_utils0 as vb
import oifits_new as oifits
import numpy as np
import datetime
import numpy as np
from numpy import pi, cos, sin
import imp
imp.reload(oifits)

def writeOIFITS(filename, RA, DEC, frequency, bandWidth, intTime,
                visamp, visamperr, visphi, visphierr, u, v, ant1, ant2, timeobs,
                t3amp, t3amperr, t3phi, t3phierr, uClosure, vClosure, antOrder, timeClosure,
                antennaNames, antennaDiam, antennaX, antennaY, antennaZ):
                
              
    speedoflight = vb.C;
    flagVis = False; # do not flag any data
    
    # open a new oifits file
    data = oifits.oifits();

    # put in the target information - RA and DEC should be in degrees
    name = 'TARGET_NAME';
    data.target = np.append(data.target, oifits.OI_TARGET(name, RA, DEC, veltyp='LSR') )
    
    #calulate wavelength and bandpass
    wavelength = speedoflight/frequency;
    bandlow = speedoflight/(frequency+(0.5*bandWidth) );
    bandhigh = speedoflight/(frequency-(0.5*bandWidth) );
    bandpass = bandhigh-bandlow;
    
    # put in the wavelength information - only using a single frequency
    data.wavelength['WAVELENGTH_NAME'] = oifits.OI_WAVELENGTH(wavelength, eff_band=bandpass)
    
    # put in information about the telescope stations in the array
    stations = [];
    for i in range(0, len(antennaNames)):
        stations.append( (antennaNames[i], antennaNames[i], i+1, antennaDiam[i], [antennaX[i], antennaY[i], antennaZ[i]]) )
    print("FG: stations=",stations,stations[0])


    timemjd = timeobs
    ins='GRAVITY'
    arrname = 'VLTI'
    data.array['ARRAY_NAME'] = oifits.OI_ARRAY('GEOCENTRIC', np.array([0, 0, 0]), stations);
    ttt = oifits.OI_ARRAY('GEOCENTRIC', np.array([0, 0, 0]), stations)
    print("FG: ttt.frame=",ttt.frame)
    print("FG: ttt.arrxyz=",ttt.arrxyz)
    print("FG: ttt.stations=",ttt.station)
    print("FG: ### oifits.OI_ARRAY=",oifits.OI_ARRAY('GEOCENTRIC', np.array([0, 0, 0]), stations))
    print("FG: data.array[ARRAY_NAME]=",data.array['ARRAY_NAME'].station)
    print('Warning: set cflux and cfluxerr = False because otherwise problems were being generated...are they the total flux density?')
    print('Warning: are there any true flags?')
    # put in the visibility information - note this does not include phase errors!
    for i in range(0, len(u)):
        print("FG: in writeData.py len(u)=",len(u))
        #timemjd[i] = timeobs[i]*0.0+2456000.0
        print("FG:data.array[ARRAY_NAME].station[0]=",data.array['ARRAY_NAME'].station[ant1[i][0]-1].sta_name);
        print("FG:data.array[ARRAY_NAME].station[1]=",data.array['ARRAY_NAME'].station[ant2[i][0]-1].sta_name);
        #print "FG:data.array[ARRAY_NAME].station[2]=",data.array['ARRAY_NAME'].station[2];
        #print "FG:data.array[ARRAY_NAME].station[3]=",data.array['ARRAY_NAME'].station[3];
        station_curr = (data.array['ARRAY_NAME'].station[ant1[i][0]-1].sta_name , data.array['ARRAY_NAME'].station[ant2[i][0]-1].sta_name);
        print("FG: in writeData.py station_curr=",station_curr);
        print("FG: in writeData.py timeobs=",i,timeobs);
        print("FG: in writeData.py intTime=",i,intTime);
        print("FG: in writeData.py ins=",i,ins);
        print("FG: in writeData.py visamp[i]=",i,visamp[i]);
        print("FG: in writeData.py visphi[i]=",i,visphi[i]);
        print("FG: in writeData.py flagVis[i]=",i,flagVis);
        print("FG: in writeData.py u[i]=",i,u[i]);
        print("FG: in writeData.py u[i][0]=",i,u[i][0]);
        print("FG: in writeData.py u[i]*wavelength=",i,u[i]*wavelength);
        print("FG: in writeData.py data.wavelength['WAVELENGTH_NAME']=",i,data.wavelength['WAVELENGTH_NAME']);
        print("FG: in writeData.py data.target[0]=",i,data.target[0]);
        print("FG: in writeData.py data.array['ARRAY_NAME']=",i,data.array['ARRAY_NAME']);
        currVis = oifits.OI_VIS(timeobs, timeobs, intTime, ins, arrname, visamp[i], visamperr[i], visphi[i], visphierr[i], flagVis, u[i][0]*wavelength[0], v[i][0]*wavelength[0],data.wavelength['WAVELENGTH_NAME'], data.target[0], station=station_curr);
        #currVis = oifits.OI_VIS(timeobs[i], intTime, ins, visamp[i], visamperr[i], visphi[i], visphierr[i], flagVis, u[i]*wavelength, v[i]*wavelength, data.wavelength['WAVELENGTH_NAME'], data.target[0], array=data.array['ARRAY_NAME'], station=station_curr, cflux=False, cfluxerr=False);
        data.vis = np.append( data.vis, currVis );
        

        # put in visibility squared information
    for k in range(0, len(u)):
        station_curr = (data.array['ARRAY_NAME'].station[ ant1[k][0] - 1 ] , data.array['ARRAY_NAME'].station[ ant2[k][0] - 1 ]);
        currVis2 = oifits.OI_VIS2(timeobs, timeobs, intTime, ins, visamp[k]**2, 2.0*visamp[k]*visamperr[k], flagVis, u[k][0]*wavelength[0], v[k][0]*wavelength[0], data.wavelength['WAVELENGTH_NAME'], data.target[0], array=arrname, station=station_curr);
        data.vis2 = np.append(data.vis2, currVis2 );

    # put in bispectrum information
    for j in range(0, len(uClosure)):
        station_curr = (data.array['ARRAY_NAME'].station[ antOrder[j][0] - 1 ] , data.array['ARRAY_NAME'].station[ antOrder[j][1] - 1 ], data.array['ARRAY_NAME'].station[ antOrder[j][2] - 1 ]);
        currT3 = oifits.OI_T3(timeobs, timeobs, intTime, ins, t3amp[j], t3amperr[j], t3phi[j], t3phierr[j], flagVis, uClosure[j][0]*wavelength[0], vClosure[j][0]*wavelength[0], uClosure[j][1]*wavelength[0], vClosure[j][1]*wavelength[0], data.wavelength['WAVELENGTH_NAME'], data.target[0], array=arrname, station=station_curr);
        data.t3 = np.append(data.t3, currT3 );
        


    #save oifits file
    data.save(filename)

def arrayUnion(array, union):
    for item in array:
        if not (item in list(union.keys())):
            union[item] = len(union)+1
    return union

def convertStrings(array, union):
    returnarray = np.zeros((array.shape),dtype=int)
    for i in range(len(array)):
        returnarray[i] = union[array[i]]
    return returnarray


