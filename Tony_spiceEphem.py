import os, spiceypy, sys, glob, argparse, datetime
from matplotlib import pyplot as plt
from ftplib import FTP_TLS
import numpy as np

def get_ephem(args):

	#List of date/times that we want ephemeris data for
	duration_seconds = int((args.end-args.start).total_seconds())
	dt = np.array([args.start+datetime.timedelta(seconds=i) for i in range(0,duration_seconds+1,args.step)])
	
	#Get the URLs for SPDF ephemeris files
	#(the relative URLS listed right here are all at spdf.gsfc.nasa.gov [hard-coded below])
	urls = dict()	
	base = '/pub/data/psp/ephemeris/spice/'
	urls['lsk'] = base+'Leap_Second_Kernel/*.tls'
	urls['sclk'] = base+'SCLK_files/*.tsc'
	urls['ah'] = base+'Attitude_History_Kernels/*.ah.bc'
	urls['recon_ephem'] = base+'Reconstructed_Ephemerides/spp_recon_*.bsp' #we're just gonna get all of these, even though we *could* get certain ones for certain dates
	urls['frame'] = base+'PSP_Frame_Kernels/spp_v*.tf'
	urls['dyn'] = base+'PSP_Frame_Kernels/spp_dyn*.tf'
	urls['pck'] = base+'Planetary_Constant_Kernel/pck*.tpc'
	urls['planetary_ephem'] = base+'Planetary_Ephemerides/de*.bsp'
		
	#Attitude history (AH) files are daily	
	#First figure out which AH files we need
	dt_dates = [datetime.datetime(i.year,i.month,i.day) for i in dt]
	date_list = np.unique(dt_dates)
	
	#Download the ephemeris files (FTP) to the local cache
	kernel_paths = get_all_kernels(urls, date_list, get_ah=True)
	
	#furnsh all the kernels to SPICE
	for file in kernel_paths:		
		print(file)
		spiceypy.furnsh(file)
		

				
	######################################	
	#Above here is basically setup: furnshing all the files so SPICE has the kernels it needs
	#Below here is using SPICE to calculate spacecraft position
	######################################	



	met = np.array([(i-datetime.datetime(2010,1,1)).total_seconds() for i in dt])
	met_sec = np.floor(met).astype(int)
	met_sec_str = np.array(['{:1.0f}'.format(i) for i in met_sec])
	met_subsec_base50000 = ((met-met_sec)*50000).astype(int)
	met_subsec_str = np.array(['{:05.0f}'.format(i) for i in met_subsec_base50000])
	
	targ = '-96'
	targ_int = int(targ)
	targ_name = 'PSP'
	obs = 'Sun'
	corr = 'None'
	
	et_array = np.array([spiceypy.scs2e(targ_int, met_sec_str[i_dt]+':'+met_subsec_str[i_dt]) for i_dt in range(len(dt))])
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'SPP_RTN', corr, obs)
	state = np.array(state)
	sc_pos_km_rtn = state[:,0:3]
	sc_vel_kms_rtn = state[:,3:6]
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'IAU_SUN', corr, obs)
	state = np.array(state)	
	sc_pos_km_carr = state[:,0:3]
	sc_vel_kms_carr = state[:,3:6]
	
	rad = np.array([spiceypy.reclat(vec)[0] for vec in sc_pos_km_carr])
	carrlon = np.array([spiceypy.reclat(vec)[1] for vec in sc_pos_km_carr])
	carrlat = np.array([spiceypy.reclat(vec)[2] for vec in sc_pos_km_carr])
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'ECLIPJ2000', corr, obs)
	state = np.array(state)	
	sc_pos_km_hae = state[:,0:3]
	sc_vel_kms_hae = state[:,3:6]
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'SPP_HCI', corr, obs)
	state = np.array(state)	
	sc_pos_km_hci = state[:,0:3]
	sc_vel_kms_hci = state[:,3:6]
	
	cmat = np.zeros((len(dt),3,3))
	cmat_sunvec = np.zeros((len(dt),3))
	
	SCX_RTN = np.zeros((len(dt),3))
	SCY_RTN = np.zeros((len(dt),3))
	SCZ_RTN = np.zeros((len(dt),3))
	sunangle = np.zeros((len(dt)))
	for i_dt in range(len(dt)):
	
		thisdt = dt[i_dt]
		
		#Convert MET to ET
		et = et_array[i_dt]
		
		#get attitude at given ET
		cmat[i_dt,...] = spiceypy.pxform('SPP_SPACECRAFT','SPP_RTN',et)
		
		SCX_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[1,0,0]) 
		SCY_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[0,1,0]) 
		SCZ_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[0,0,1]) 
		
		cmat_sunvec[i_dt,...] = np.dot(cmat[i_dt,...],[0,0,-1]) 		
		scz_rcomp = cmat_sunvec[i_dt,0] #the R component of the S/C Z axis
		
		sunangle[i_dt,...] = np.arccos(scz_rcomp)*180./np.pi
	
	#This puts values into something like the format you'd get from the SPC ephemeris CDF files
	dat = {}
	dat['sc_pos_km_rtn'] =  sc_pos_km_rtn
	dat['sc_pos_km_carr'] = sc_pos_km_carr
	dat['sc_pos_km_hci'] =  sc_pos_km_hci	
	dat['sc_pos_km_hae'] = sc_pos_km_hae
	dat['sc_vel_kms_rtn'] = sc_vel_kms_rtn
	dat['sc_vel_kms_carr'] = sc_vel_kms_carr
	dat['sc_vel_kms_hci'] = sc_vel_kms_hci	
	dat['sc_vel_kms_hae'] = sc_vel_kms_hae
	dat['SCX_RTN'] = SCX_RTN
	dat['SCY_RTN'] = SCY_RTN
	dat['SCZ_RTN'] = SCZ_RTN
	dat['Epoch'] = dt
	dat['SUN_ANGLE'] = sunangle
	dat['CARR_LAT'] = carrlat
	dat['CARR_LON'] = carrlon
	dat['SC_R_AU'] = rad/149597870.7 #convert to AU from km
	dat['cmat'] = cmat


	#plot some of the values
	fig,ax = plt.subplots(3,1)
	ax[0].plot(dt, dat['SC_R_AU']*215., 'bo-')
	ax[1].plot(dt, dat['CARR_LAT'], 'ro-')
	ax[1].plot(dt, dat['CARR_LON'], 'bo-')
	ax[2].plot(dt,sunangle,'ro-')
	ax[0].set_ylabel('Radius (Rs)')
	ax[1].set_ylabel('Lat/Lon (deg)')
	ax[2].set_ylabel('Sun Angle (deg)')
	fig.autofmt_xdate()
	fig.show()

	import pdb; pdb.set_trace()
	
########################################################
##
########################################################
def get_all_kernels(urls, date_list, get_ah=False):
	
	#check whether the cache directory exists and make it if need be
	cache_path = 'ezephem_cache/'
	if not os.path.isdir(cache_path): os.makedirs(cache_path)
	already_cached_files = glob.glob(cache_path+'*.*')
	already_cached_files = [os.path.basename(f) for f in already_cached_files]
	
	kernel_file_list = []

	#connect to FTP server at SPDF
	ftp = FTP_TLS('spdf.gsfc.nasa.gov')
	ftp.login()
		
	#loop through each type of kernel we need to download
	for key in urls.keys():
		if key=='ah': continue
		try:
			#get a list of all files in that directory
			files = ftp.nlst(urls[key])
			
			#take the newest one
			sortedfiles = sorted(files)
			newest = [sortedfiles[-1]]
			if key=='recon_ephem': newest=sortedfiles #use all of them
			
			for file in newest:
				#see if we have that file
				if not os.path.basename(file) in already_cached_files:
					with open(os.path.join(cache_path,os.path.basename(file)), 'wb') as f:					
						ftp.retrbinary('RETR ' + file, f.write)
					
				#add this file to the list of kernels we're using
				kernel_file_list.append(os.path.join(cache_path, os.path.basename(file)))
			
		except:
			print('Could not retrieve files from server')
			print(sys.exc_info())
			import pdb; pdb.set_trace()
			
	if get_ah:
		#loop through each day we need to download
		for date in date_list:
			try:
				#get a list of all files for this day in the AH directory	
				search_path = os.path.dirname(urls['ah'])+'/*'+date.strftime('%Y_%j')+'*.ah.bc'
				files = ftp.nlst(search_path)
		
				#take the newest one
				newest = sorted(files)[-1]
		
				#see if we have that file
				if not os.path.basename(newest) in already_cached_files:
					with open(os.path.join(cache_path,os.path.basename(newest)), 'wb') as f:					
						ftp.retrbinary('RETR ' + newest, f.write)
				
				#add this file to the list of kernels we're using
				kernel_file_list.append(os.path.join(cache_path, os.path.basename(newest)))
		
			except:
				print('Could not retrieve AH file for day: {:}'.format(repr(date)))
				print(sys.exc_info())
				import pdb; pdb.set_trace()
	
	ftp.close()

	#Clean up cache, if you can
	cached_files = glob.glob(cache_path+'*.*')
	cached_files = [os.path.basename(f) for f in cached_files]
	for file in cached_files:
		if os.path.join(cache_path,file) not in kernel_file_list:
			try:
				os.remove(file)
			except:
				pass
	

	return(kernel_file_list)

########################################################
###
########################################################
def setup():
	"""Get user command-line input and set things up"""
	
	#available options
	sc_options = ['psp']
	
	#defaults
	sc_default = 'psp'
	start_default = '20180813000000'
	end_default = '20180814000000'
	step_default = 3600
	
	#Deine allowed command-line arguments
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-v', '--verbose', default=False, action="store_true", help='Increase verbosity', required=False)
	parser.add_argument('-sc', '--spacecraft', default=sc_default, help='Which spacecraft to use [default={:}]'.format(sc_default), required=False)
	parser.add_argument('-start', '--start', default=start_default, help='Start Date/Time.  Format=YYYYMMDDHHmmSS [default={:}]'.format(start_default), required=False, type=str)
	parser.add_argument('-end', '--end', default=end_default, help='Start Date/Time.  Format=YYYYMMDDHHmmSS  [default={:}]'.format(end_default), required=False, type=str)
	parser.add_argument('-step', '--step', default=step_default, help='Time Step in seconds [default={:}]'.format(step_default), required=False, type=int)
	
	#Read in the command-line arguments
	args = parser.parse_args()

	#Parse dates and times
	try:
		startdt = datetime.datetime.strptime(args.start, '%Y%m%d%H%M%S')		
		args.start = startdt		
	except:
		raise ValueError('Could not parse the provided start time.')
		
	try:
		enddt = datetime.datetime.strptime(args.end, '%Y%m%d%H%M%S')
		args.end = enddt
	except:
		raise ValueError('Could not parse provided end time.')
		
	if args.start>args.end:
		raise ValueError('Start time must be before end time')
		
	if args.spacecraft not in sc_options:
		raise ValueError('Spacecraft must be one of '+repr(sc_options))
	
	
	#Return
	return(args)

   
if __name__=='__main__': 
	args = setup()
	get_ephem(args)
	
	
	