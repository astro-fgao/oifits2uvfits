import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument('input', type=str, default='*.oifits',help='input oifits name')
parser.add_argument('--output', dest='output', type=str, default='output.uvfits',help='output uvfits name')
parser.add_argument('--rescale_flux', dest='rescale_flux', type=bool, default=False ,help='whether to rescale flux (default: False)')
parser.add_argument('--renorm_flux', dest='renorm_flux', type=bool, default=False ,help='whether to renormalize flux (default: False)')
parser.add_argument('--renorm_num', dest='renorm_num', type=int, default = 0 ,help='renormalize number (default: 0)')
parser.add_argument('--airmass', dest='airmass', type=bool, default=False ,help='whether to correct for airmass (default: False)')
parser.add_argument('--visdata', dest='visdata', type=bool, default=False ,help='whether to read in visdata instead of vis_amp and vis_phi (default: False)')
parser.add_argument('--specavg', dest='specavg', type=bool, default=False ,help='whether to do spectral averaging (default: False)')
parser.add_argument('--specbin', dest='specbin', type=int, default=1 ,help='how many channels to averaging (default: 1)')

args = parser.parse_args()


import vlbi_imaging_utils_gvraid_flux_2021_py3 as vb


obs = vb.load_obs_oifits(args.input, rescale_flux=args.rescale_flux, renorm_flux=args.renorm_flux, renorm_num=args.renorm_num, airmass=args.airmass, visdata=args.visdata, specavg=args.specavg, specbin=args.specbin)
print("load oifits file %s finished!" % args.input)
obs.save_uvfits(args.output)
print("save uvfits file %s finished" % args.output)
