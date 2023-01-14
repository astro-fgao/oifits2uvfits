# oifits2uvfits
scripts to convert oifits file into uvfits format

This package convert oifits format (widely used in optical/IR interferometry) to uvfits format (widely used in radio interferometry). Initially based on the eht-imaging package, the code was much tailored to the GRAVITY instrument (espeically to the Galactic Center user cases). 


Code usage:

    python oifits2uvfits input_oifits_file_name output_oifits_file_name
    
VISDATA or VIS_AMP & VIS_PHI
During the conversion, the user can choose whether to use the VISDATA column (--VISDATA = True) or use the VIS_AMP and VIS_PHI column (default case). 

spectral averaging
The user can also specify how many channels they want to average during the conversion step (--specavg=True and --specbin= number of channels to average. default: no averaging)
