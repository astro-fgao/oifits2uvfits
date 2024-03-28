# oifits2uvfits
Python scripts to convert oifits file into uvfits format.

This package convert oifits format (widely used in optical/IR interferometry) to uvfits format (widely used in radio interferometry). Initially based on the eht-imaging package, the code was much tailored to the GRAVITY instrument (espeically to the Galactic Center user cases). 


Code usage: 

	python oi2uv.py [-h] [--output OUTPUT] [--rescale_flux RESCALE_FLUX]
                [--renorm_flux RENORM_FLUX] [--renorm_num RENORM_NUM]
                [--airmass AIRMASS] [--visdata VISDATA] [--specavg SPECAVG]
                [--specbin SPECBIN]
                input

    
Options:

	-h, --help            			show this help message and exit
 	
	--output OUTPUT       			output uvfits name
 
 	--rescale_flux RESCALE_FLUX		whether to rescale flux (default: False)
  
  	--renorm_flux RENORM_FLUX		(*) whether to renormalize flux (default: False)
  
  	--renorm_num RENORM_NUM			(*) renormalize number (default: 0)
  
  	--airmass AIRMASS     			(*) whether to correct for airmass (default: False)
  						
  	--visdata VISDATA     			whether to read in visdata instead of vis_amp and 
   						vis_phi (default: False)
  
  	--specavg SPECAVG     			whether to do spectral averaging (default: False)
  
  	--specbin SPECBIN     			how many channels to averaging (default: 1)

Please note that (*) denotes the options which are used internally and not yet available in this public branch yet.

For further information or feedback, please contact me at fgao@mpifr-bonn.mpg.de
