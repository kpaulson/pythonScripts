import numpy as np
import argparse

def main(energy, velocity):
	
	inputEnergy = energy
	inputVelocity = velocity
	
	#KE = 1/2 * m * v^2
	m_p = 0.010438870 #938.27 MeV/c² converted into eV
	
	if inputVelocity != 0:
		energy = 0.5*m_p*(velocity**2)
		
		print(energy)
		#return(energy)
		
	if inputEnergy != 0:
		#energy = 0.5*m_p*(velocity**2)
		velocity = np.sqrt(args.energy * 2 / m_p)
		
	print('proton with energy {:} moves at velocity {:}'.format(energy, velocity))
		#return(energy)
	
#args = none
#convertVelocityToEnergy = True

#inputVelocity = 670

#main(velocity=inputVelocity)

	
#####################################################
##
#####################################################
def setup():
	"""Get user command-line input and set things up"""
	
	#defaults
	inputVelocity_default = 0
	inputEnergy_default = 0
	
	#Get User Input
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-e', '--energy', default=inputVelocity_default, help='Input energy value to convert to velocity', required=False, type=float)
	parser.add_argument('-v', '--velocity', default=inputVelocity_default, help='Input velocity value to convert to velocity', required=False, type=float)
	
	# parser.add_argument('-o', '--overwrite', default=False, action="store_true", help='Overwrite existing L1 CDF file, if necessary', required=False)
	# parser.add_argument('-sb', '--startbyte', help='Start byte [-1==All] [default={:}]'.format(sb_default), required=False, default=sb_default, type=int)
	# parser.add_argument('-eb', '--endbyte', help='End byte [-1==All] [default={:}]'.format(eb_default), required=False, default=eb_default, type=int)
	# parser.add_argument('-l1', '--l1file', help='Input L0 File [default={:}]'.format(l1file_default), required=False, default=l1file_default)
	# parser.add_argument('-d', '--l1dir', help='Input L1 Directory (for use with -b or -r [default={:}]'.format(l1dir_default), required=False, default=l1dir_default)
	# parser.add_argument('-dl2', '--l2dir', help='Output L2 Directory [default={:}]'.format(l2dir_default), required=False, default=l2dir_default)
	# parser.add_argument('-dlog', '--logdir', help='Output for Log Files [default={:}]'.format(logdir_default), required=False, default=logdir_default)
	# parser.add_argument('-a', '--apid', help='APID to create L1 file for [0==all] [default={:}]'.format(apid_default), required=False, default=apid_default, type=str)
	
	#Read in the arguments
	args = parser.parse_args()
		
	#Version of the data product
	#args.version=1
	
	#Convert APID to an integer (it is read as a string from the command line)
	# try:
		# if args.apid[0:2]=='0x': 
			# base=16
		# else:
			# base=10
		# args.apid = int(args.apid, base)
	# except TypeError:
		# statusmsg('Trouble parsing desired APID....exiting.', screen=True)
		# statusmsg(sys.exc_info(), screen=True)
		# sys.exit()
	
	#Return to main routine
	return(args)


############################################
####
############################################
if __name__=='__main__':	
	args = setup()
	print(args)
	main(energy=args.energy, velocity=args.velocity)