from datetime import datetime
from datetime import timedelta
import numpy as np
import os, sys, glob
from scipy.io import readsav
from spacepy import pycdf


#Parker Solar Probe Mission Elapsed Time "Epoch" is Midnight, Jan 1, 2010
epoch = datetime(2010,1,1,0,0,0)

#SPC Modulation Frequency
spc_modfreq = 1171.875 #Hz

#NYS Length
nys_s = 1024./spc_modfreq

#AU distance in km
au_km = 149597870.7

def met2dt(met):
	'''Convert Mission Elapsed Time (seconds since 2010,1,1 00:00:00) to a datetime object'''
	elapsed = timedelta(seconds=met)
	dt = epoch + elapsed
	return(dt)
	
def dt2met(dt):
	'''Convert a datetime object to Mission Elapsed Time (seconds since 2010,1,1, 00:00:00)'''
	met = (dt-epoch).total_seconds()
	return(met)
	
def cal2met(year=2010, month=1, day=1, hour=0, minute=0, second=0):
	'''Convert year, month, day, hour, minute, second to Mission Elapsed Time (seconds since 2010,1,1, 00:00:00)'''
	dtarg = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
	met = dt2met(dtarg)
	return(met)
	
def tick2sec(tick):
	'''Convert tick number within a NYS to time since beginning of NYS in seconds'''
	return(tick/spc_modfreq)

def get_spc_events():
	'''Define SPC events of interest'''
	events =[[datetime(2018,8,12),'Launch'],
			 [datetime(2018,8,30,5,35,0),'First SPC Turnon'],
			 [datetime(2018,9,3,8,33,28),'SPC_PowerOn'],
			 [datetime(2018,9,5,22,9,46),'SPC_PowerOn'],
			 [datetime(2018,9,8,0,57,0),'Mag Roll'],
			 [datetime(2018,9,8,4,50,0),'Transient Slew'],
			 [datetime(2018,9,17,14,20,0),'Nadir Viewing'],
			 [datetime(2018,9,17,16,29,0),'Electron Mode'],
			 [datetime(2018,9,21,3,35,16),'SPC_PowerOn'],
			 [datetime(2018,9,26,16,30,0),'Mini Encounter Campaign'],
			 [datetime(2018,10,2,4,0,0),'Umbra Pointing'],
			 [datetime(2018,10,2,6,0,0),'FluxAngle'],
			 [datetime(2018,10,2,10,0,0),'FluxAngle'],
			 [datetime(2018,10,2,14,0,0),'FluxAngle'],
			 [datetime(2018,10,3,3,21,0),'Venus#1'],
			 [datetime(2018,10,3,16,17,0),'SPC_PowerOn'],
			 [datetime(2018,10,6,7,7,0),'SPC_PowerOn'],
			 [datetime(2018,10,6,10,44,0),'Test_SuperFast'],
			 [datetime(2018,10,8,16,39,0),'SPC_PowerOn'],
			 [datetime(2018,10,11,15,17,0),'SPC_PowerOn'],
			 [datetime(2018,10,14,14,22,0),'SPC_PowerOn'],
			 [datetime(2018,10,16,10,44,0),'FSW patch and Seq Load'],
			 [datetime(2018,10,20,10,37,44),'SPC_PowerOn'],
			 [datetime(2018,10,20,12,14,12),'SPC_PowerOn'],
			 [datetime(2018,10,20,13,26,39),'SPC_PowerOn'],
			 [datetime(2018,10,31,2,15,0),'SWEM_Reset'],
			 [datetime(2018,11,3,11,50,0),'SWEM_Reset'],
			 [datetime(2018,11,4,16,44,0),'SWEM_Reset'],
			 [datetime(2018,11,5,3,41,0),'SWEM_Reset'],
			 [datetime(2018,11,5,11,42,0),'SWEM_Reset'],
			 [datetime(2018,11,6,3,27,0),'Perihelion #1'],
			 [datetime(2018,11,7,3,59,0),'SWEM_Reset'],
			 [datetime(2018,11,10,15,27),'ElectronMedium'],
			 [datetime(2018,11,10,15,32),'MediumMode'],
			 [datetime(2018,11,10,15,37,51),'Calibration'],
			 [datetime(2018,11,11,3,27),'SlowMode'],
			 [datetime(2018,11,13,15,0),'ElectronMedium'],
			 [datetime(2018,11,13,15,5),'SlowMode'],
			 [datetime(2018,11,15,2,30),'SlowMode'],
			 [datetime(2018,11,17,2,58),'SlowMode?'],
			 [datetime(2018,11,18,13,15,0),'Mag Rolls'],
			 [datetime(2018,11,18,13,20),'ElectronMedium'],
			 [datetime(2018,11,18,13,25),'MediumMode'],
			 [datetime(2018,11,18,23,0),'SlowMode'],
			 [datetime(2018,11,24,15,0),'ElectronMedium'],
			 [datetime(2018,11,24,15,5),'MediumMode'],
			 [datetime(2018,12,13,15,28,43),'SPC_PowerOn'],
			 [datetime(2018,12,15,8,8,52),'SPC_PowerOn'],
			 [datetime(2018,12,16,5,58,53),'SPC_PowerOn'],
			 [datetime(2019,1,20,12,3,50),'SPC_PowerOn (MedMode)'],
			 [datetime(2019,1,22,1,34,30),'MediumMode'],
			 [datetime(2019,1,22,15,30,47),'FastMode'],
			 [datetime(2019,1,25,15,31,50),'SuperFastMode'],
			 [datetime(2019,1,25,16,16,34),'FastMode'],
			 [datetime(2019,1,28,15,28,2),'MediumMode'],
			 [datetime(2019,1,28,16,28,5),'ElectronMedium'],
			 [datetime(2019,1,28,16,33,3),'MediumMode'],
			 [datetime(2019,2,20,13,18,50),'SPC_PowerOn'],
			 [datetime(2019,2,24),'6 kV Test'],
			 [datetime(2019,2,24,22,14,49),'SPC_PowerOn'],
			 [datetime(2019,2,25,0,24,16),'spc_shortcal'],
			 [datetime(2019,2,25,0,28,33),'Slow Mode'],
			 [datetime(2019,2,26,13,18,49),'SPC_PowerOn'],
			 [datetime(2019,3,6,10,35,8),'SPC_PowerOn'],
			 [datetime(2019,3,9,7,24,19),'SPC_PowerOn'],
			 [datetime(2019,3,21),'6 kV Switch (Slow)'],
			 [datetime(2019,3,29,23,30,13),'spc_shortcal'],
			 [datetime(2019,3,30,0,30,0),'Fast Mode'],
			 [datetime(2019,3,30,11,50,0),'FluxAngle'],
			 [datetime(2019,3,30,12,0,0),'FastMode'],
			 [datetime(2019,3,30,23,50,0),'FluxAngle'],
			 [datetime(2019,3,31,0,0,0),'FastMode'],
			 [datetime(2019,3,31,11,50,0),'FluxAngle'],
			 [datetime(2019,3,31,12,0,0),'FastMode'],
			 [datetime(2019,3,31,23,50,0),'FluxAngle'],
			 [datetime(2019,4,1,0,0,0),'FastMode'],
			 [datetime(2019,4,1,11,50,0),'FluxAngle'],
			 [datetime(2019,4,1,12,0,0),'FastMode'],
			 [datetime(2019,4,1,23,10,0),'FluxAngle'],
			 [datetime(2019,4,1,23,20,0),'MediumMode'],
			 [datetime(2019,4,2,0,0,0),'FastMode'],
			 [datetime(2019,4,2,11,50,0),'FluxAngle'],
			 [datetime(2019,4,2,12,0,0),'FastMode'],
			 [datetime(2019,4,2,23,40,0),'FluxAngle'],
			 [datetime(2019,4,2,23,50,0),'ElectronMedium'],
			 [datetime(2019,4,3,0,0,0),'FastMode'],
			 [datetime(2019,4,3,11,40,0),'FluxAngle'],
			 [datetime(2019,4,3,11,50,0),'ElectronMedium'],
			 [datetime(2019,4,3,12,0,0),'FastMode'],
			 [datetime(2019,4,3,23,40,0),'FluxAngle'],
			 [datetime(2019,4,3,23,50,0),'ElectronMedium'],
			 [datetime(2019,4,4,0,0,0),'FastMode'],
			 [datetime(2019,4,4,11,40,0),'FluxAngle'],
			 [datetime(2019,4,4,11,50,0),'ElectronMedium'],
			 [datetime(2019,4,4,12,0,0),'FastMode'],
			 [datetime(2019,4,4,23,0,0),'Perihelion #2'],
			 [datetime(2019,4,4,23,40,0),'FluxAngle'],
			 [datetime(2019,4,4,23,50,0),'ElectronMedium'],
			 [datetime(2019,4,5,0,0,0),'FastMode'],
			 [datetime(2019,4,5,11,40,0),'FluxAngle'],
			 [datetime(2019,4,5,11,50,0),'ElectronMedium'],
			 [datetime(2019,4,5,12,0,0),'FastMode'],
			 [datetime(2019,4,5,23,40,0),'FluxAngle'],
			 [datetime(2019,4,5,23,50,0),'ElectronMedium'],
			 [datetime(2019,4,6,0,0,0),'FastMode'],
			 [datetime(2019,4,6,11,40,0),'FluxAngle'],
			 [datetime(2019,4,6,11,50,0),'ElectronMedium'],
			 [datetime(2019,4,6,12,0,0),'FastMode'],
			 [datetime(2019,4,6,23,50,0),'FluxAngle'],
			 [datetime(2019,4,7,0,0,0),'FastMode'],
			 [datetime(2019,4,7,11,50,0),'FluxAngle'],
			 [datetime(2019,4,7,12,0,0),'FastMode'],
			 [datetime(2019,4,7,23,50,0),'FluxAngle'],
			 [datetime(2019,4,8,0,0,0),'FastMode'],
			 [datetime(2019,4,8,11,50,0),'FluxAngle'],
			 [datetime(2019,4,8,12,0,0),'FastMode'],
			 [datetime(2019,4,8,23,50,0),'FluxAngle'],
			 [datetime(2019,4,9,0,0,0),'FastMode'],
			 [datetime(2019,4,9,11,50,0),'FluxAngle'],
			 [datetime(2019,4,9,12,0,0),'FastMode'],
			 [datetime(2019,4,9,23,50,0),'FluxAngle'],
			 [datetime(2019,4,10,0,0,0),'FastMode'],
			 [datetime(2019,4,10,14,0,0),'spc_shortcal'],
			 [datetime(2019,4,10,14,20,0),'SlowMode'],
			 [datetime(2019,4,12,0,34,36),'SPC_PowerOn'],
			 [datetime(2019,4,12,12,54,36),'SPC_PowerOn'],
			 [datetime(2019,4,13,0,56,40),'SPC_PowerOn'],
			 [datetime(2019,4,13,23,19,27),'SPC_PowerOn'],
			 [datetime(2019,4,15,0,29,25),'SPC_PowerOn'],
			 [datetime(2019,4,15,17,9,27),'SPC_PowerOn'],
			 [datetime(2019,4,16,11,19,27),'SPC_PowerOn'],
			 [datetime(2019,4,17,16,49,27),'SPC_PowerOn'],
			 [datetime(2019,4,18,12,9,28),'SPC_PowerOn'],
			 [datetime(2019,4,19,11,14,32),'SPC_PowerOn'],
			 [datetime(2019,4,20,13,14,30),'SPC_PowerOn'],
			 [datetime(2019,4,21,11,44,34),'SPC_PowerOn'],
			 [datetime(2019,4,22,3,49,32),'SPC_PowerOn'],
			 [datetime(2019,4,23,9,44,28),'SPC_PowerOn'],
			 [datetime(2019,4,25,15,14,22),'SPC_PowerOn'],
			 [datetime(2019,4,26,7,3,41),'SPC_PowerOn'],
			 [datetime(2019,4,28,9,29,28),'SPC_PowerOn'],
			 [datetime(2019,4,30,2,4,28),'SPC_PowerOn'],
			 [datetime(2019,5,1,0,39,26),'SPC_PowerOn'],
			 [datetime(2019,5,1,13,4,26),'SPC_PowerOn'],
			 [datetime(2019,5,2,2,34,26),'SPC_PowerOn'],
			 [datetime(2019,5,2,22,29,28),'SPC_PowerOn'],
			 [datetime(2019,5,3,12,24,26),'SPC_PowerOn'],
			 [datetime(2019,5,4,13,4,26),'SPC_PowerOn'],
			 [datetime(2019,5,5,8,14,26),'SPC_PowerOn'],
			 [datetime(2019,5,7,3,49,28),'SPC_PowerOn'],
			 [datetime(2019,5,8,9,44,28),'SPC_PowerOn'],
			 [datetime(2019,5,10,3,49,28),'SPC_PowerOn'],
			 [datetime(2019,5,11,13,49,26),'SPC_PowerOn'],
			 [datetime(2019,5,12,0,34,28),'SPC_PowerOn'],
			 [datetime(2019,7,20,20,41,37),'SPC_PowerOn'],
			 [datetime(2019,7,22,22,18,57),'SPC_PowerOn'],
			 [datetime(2019,7,25,7,19,28),'SPC_PowerOn'],
			 [datetime(2019,7,27,14,4,28),'SPC_PowerOn'],
			 [datetime(2019,7,30,6,59,28),'SPC_PowerOn'],
			 [datetime(2019,7,31,16,4,28),'SPC_PowerOn'],
			 [datetime(2019,8,1,6,49,28),'SPC_PowerOn'],
			 [datetime(2019,8,4,7,29,28),'SPC_PowerOn'],
			 [datetime(2019,8,7,14,19,28),'SPC_PowerOn'],
			 [datetime(2019,8,9,7,39,30),'SPC_PowerOn'],
			 [datetime(2019,8,11,10,49,30),'SPC_PowerOn'],
			 [datetime(2019,8,13,14,54,34),'SPC_PowerOn'],
			 [datetime(2019,8,15,13,9,34),'SPC_PowerOn'],
			 [datetime(2019,8,16,9,59,34),'SPC_PowerOn'],
			 [datetime(2019,8,17,2,51,0),'SlowMode'],
			 [datetime(2019,8,17,12,0,0),'MediumMode'],
			 [datetime(2019,8,18,12,0,0),'spc_shortcal'],
			 [datetime(2019,8,19,12,0,0),'MediumMode'],
			 [datetime(2019,8,19,20,4,32),'HVOff_SC_OpMode2'],
			 [datetime(2019,8,21,23,59,34),'SPC_PowerOn'],
			 [datetime(2019,8,22,0,2,29),'MediumMode'],
			 [datetime(2019,8,23,12,0,0),'MediumMode'],
			 [datetime(2019,8,25,12,0,0),'MediumMode'],
			 [datetime(2019,8,27,3,0,0),'0.25AU,MedFastMode'],
			 [datetime(2019,8,28,0,5,0),'FluxAngle'],
			 [datetime(2019,8,29,0,5,0),'FluxAngle'],
			 [datetime(2019,8,30,0,5,0),'FluxAngle'],
			 [datetime(2019,8,30,17,40,0),'spc_shortcal'],
			 [datetime(2019,8,30,17,42,6),'HiCurrPwrOff'],
			 [datetime(2019,9,1,18,0,0),'Perihelion #3'],
			 [datetime(2019,9,18,3,4,0),'SPC LPT, Load 4 kV'],
			 [datetime(2019,9,18,4,1,0),'spc_shortcal, HV Ramp'],
			 [datetime(2019,9,18,6,28,0),'MediumMode, then Off'],
			 [datetime(2019,9,18,20,16,0),'Re-enable SPC Startup'],
			 [datetime(2019,9,18,23,16,43),'SPC_PowerOn'],
			 [datetime(2019,9,20,13,14,31),'SPC_PowerOn'],
			 [datetime(2019,9,21,13,24,31),'SPC_PowerOn'],
			 [datetime(2019,9,22,7,49,31),'SPC_PowerOn'],
			 [datetime(2019,10,15),'SPC_Off_0.82'],
			 [datetime(2019,12,16),'SPC_On_0.82'],
			 [datetime(2019,12,26,17,14),'Venus#2'],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 # [datetime(2020,,,,),''],
			 #[datetime(2020,,,,),''],]
			 []]
	return(events)

def dt2orbitnum(dt):
	ap = np.array(get_aphelia())
	laterpix = np.where(ap[:,1]>dt)[0]
	return(laterpix[0])
	
def get_perihelia():
	"""Return a list of datetime objects that contains each perihelion time and distance"""
	peri_distances = [35.67,35.67,35.67,
					  27.88,27.88,
					  20.36,20.36,
					  15.98,15.98,
					  13.28,13.28,13.28,13.28,13.28,13.28,13.28,
					  11.44,11.44,11.44,11.44,11.44,
					  9.86,9.86,9.86]
				 
	peri_times = [datetime(2018,11,6,3), datetime(2019,4,4,23), datetime(2019,9,1,18),
				  datetime(2020,1,29,10), datetime(2020,6,7,8),
				  datetime(2020,9,27,10), datetime(2021,1,17,17),
				  datetime(2021,4,29,9), datetime(2021,8,9,18),
				  datetime(2021,11,21,9), datetime(2022,2,25,16), datetime(2022,6,12,23), datetime(2022,9,6,6), datetime(2022,12,11,13), datetime(2023,3,17,21), datetime(2023,6,22,4),
				  datetime(2023,9,28,0), datetime(2023,12,29,2), datetime(2024,3,30,3), datetime(2024,6,30,4), datetime(2024,9,30,5),
				  datetime(2024,12,24,11), datetime(2025,3,22,22), datetime(2025,6,19,8)]		  
	peri = []
	for i_peri in range(len(peri_distances)):
		peri.append([i_peri, peri_times[i_peri], peri_distances[i_peri]])
	return(peri)

def get_aphelia():
	"""Return a list of datetime objects that contains each perihelion time and distance"""
	ap_distances = [217.9,201.72,201.72,201.72,
					  188.06,
					  175.61,175.61,
					  168.28,168.28,168.28,
					  163.55,163.55,163.55,163.55,163.55,163.55,163.55,
					  160.16,160.16,160.16,160.16,
					  157.21,157.21,157.21]
				 
	ap_times = [datetime(2018,8,12,8), datetime(2019,1,20,1), datetime(2019,6,18,20),datetime(2019,11,15,15), 
				datetime(2020,4,3,9),
				datetime(2020,8,2,5), datetime(2020,11,22,13),
				datetime(2021,3,9,4), datetime(2021,6,19,14),datetime(2021,9,30,0),
				datetime(2022,1,8,12), datetime(2022,4,14,19), datetime(2022,7,20,2), datetime(2022,10,24,10), datetime(2023,1,28,17), datetime(2023,5,5,0),datetime(2023,8,9,7),
				datetime(2023,11,13,0), datetime(2024,2,13,2), datetime(2024,5,15,3), datetime(2024,8,15,4),
				datetime(2024,11,10,6), datetime(2025,2,6,17), datetime(2025,5,6,4)]		  
	ap = []
	for i_ap in range(len(ap_distances)):
		ap.append([i_ap, ap_times[i_ap], ap_distances[i_ap]])
	return(ap)
	
def get_magfilename(dt, level, version=0, verbose=False):
	"""get filename for FIELDS mag files"""
	if version!=0:
		raise ValueError('version must be 0')
	
	if level==1:
		pass
	elif level==2:
		base_path = '/psp/data/sci/fields/staging/l2/mag_RTN_4_Sa_per_Cyc/'
		file_fmt = base_path+'{:04.0f}/{:02.0f}/psp_fld_l2_mag_RTN_4_Sa_per_Cyc_{:04.0f}{:02.0f}{:02.0f}_v{:02.0f}.cdf'
	
	filename = file_fmt.format(dt.year,dt.month,dt.year,dt.month,dt.day,version)
	return(filename)
	
def get_mag(level, dtstart, dtend=False, version=0, verbose=False):
	"""Return mag variables between requested times"""
	
	allowed_levels = [1,2]
	if level not in allowed_levels:
		raise ValueError('level must be in: '+repr(allowed_levels))
	
	if ( (type(dtstart)!=datetime) | ((type(dtend)!=datetime) ) ):
		raise TypeError('dtstart and dtend must be datetime objects')
	
	if dtend<dtstart:
		raise ValueError('dtstart must be before dtend')
		
	if level==2:
		varnames = ['epoch_mag_RTN','psp_fld_l2_mag_RTN']
	
	
	vars = {}
	for varname in varnames:
		vars[varname] = []
	attrs = {}
	
	thisday = datetime(dtstart.year,dtstart.month,dtstart.day,0,0,0)
	while thisday<=dtend:
		filepath = get_magfilename(thisday, level, version=version, verbose=False)

		if not filepath:
			if verbose: print('File not found for: {:}'.format(thisday.isoformat()))
		else:
			if verbose: print('Using file: '+repr(filepath))
			dat = pycdf.CDF(filepath)
 
			if verbose: print('done reading cdf')
			for i_var in range(len(varnames)):
				try:
					thisvar = dat[varnames[i_var]]
					vars[varnames[i_var]].extend(thisvar[...])
					if verbose: print('done with var: {:}'.format(varnames[i_var]))
				except KeyError:
					print('Key not in file: {:}'.format(varnames[i_var]))
					print('Available keys: '+repr(dat.keys()))
					print('Halting operation')
					import pdb; pdb.set_trace()
						
		#iterate to the next day
		thisday = thisday + timedelta(days=1)
	
	#Store the attribute variables
	for key in vars.keys():
		try:
			attrs[key] = dat[key].attrs
		except KeyError:
			print('Key not in file: {:}'.format(key))
			print('Available keys: '+repr(dat.keys()))
	
	#return numpy arrays, rather than lists
	for key in vars.keys():
		vars[key] = np.array(vars[key])
		
	gpix = np.where( (vars['epoch_mag_RTN']>=dtstart) & (vars['epoch_mag_RTN']<=dtend) )[0]	
	
	for key in vars.keys():
		vars[key] = vars[key][gpix,...]
		bpix = np.where(vars[key]==attrs[key]['FILLVAL'])[0]
		if len(bpix)>0:
			try:
				vars[key][bpix]=np.nan
			except ValueError:
				pass
	
	return(vars, attrs)
		
def get_spcvars(apid, level, varnames, dtstart, dtend=False, verbose=False, version=-1, all=False):
	"""return a long array of a variable from a range of dates"""
	
	if type(varnames)!=list:
		varnames = [varnames]
		
	if 'Epoch' not in varnames: varnames.append('Epoch')
	
	#Make sure user input is reasonable
	if not dtend:
		dtend = datetime(dtstart.year,dtstart.month,dtstart.day,23,59,59,999999) #default to end of the day requested
	
	allowed_levels = [0.5,1,2,3]
	if level not in allowed_levels:
		raise ValueError('level must be in: '+repr(allowed_levels))
	
	if ( (type(dtstart)!=datetime) | ((type(dtend)!=datetime) ) ):
		raise TypeError('dtstart and dtend must be datetime objects')
	
	if dtend<dtstart:
		raise ValueError('dtstart must be before dtend')
	
	vars = {}
	attrs = {}
	for varname in varnames:
		vars[varname] = []
		
	thisday = datetime(dtstart.year,dtstart.month,dtstart.day,0,0,0)
	firstloop=True
	
	while thisday<=dtend:
		filepath = getfilename(thisday, level, ionelectron='i', apid=apid, version=version, verbose=False)
		if not filepath:
			if verbose: print('File not found for: {:}'.format(thisday.isoformat()))
		else:
			if verbose: print('Using file: '+repr(filepath))
			dat = pycdf.CDF(filepath)
			if (all & firstloop):
				varnames = dat.keys()
				vars = {}
				for varname in varnames: vars[varname] = []
				
			if verbose: print('done reading cdf')
			for i_var in range(len(varnames)):
				try:
					thisvar = dat[varnames[i_var]]
					#thisvar._raw=True #so that we get raw Epoch values rather than datetimes
					vars[varnames[i_var]].extend(thisvar[...])
					if verbose: print('done with var: {:}'.format(varnames[i_var]))
				except KeyError:
					print('Key not in file: {:}'.format(varnames[i_var]))
					print('Available keys: '+repr(dat.keys()))
					print('Halting operation')
					import pdb; pdb.set_trace()
				except TypeError:
					print('TypeError on key: {:}'.format(varnames[i_var]))
					

		firstloop = False
		#iterate to the next day
		thisday = thisday + timedelta(days=1)
	 
	#Store the attribute variables
	for key in vars.keys():
		try:
			attrs[key] = dat[key].attrs
		except KeyError:
			print('Key not in file: {:}'.format(key))
			print('Available keys: '+repr(dat.keys()))
	
	#return numpy arrays, rather than lists
	for key in vars.keys():
		vars[key] = np.array(vars[key])
		
	gpix = np.where( (vars['Epoch']>=dtstart) & (vars['Epoch']<=dtend) )[0]	
	for key in vars.keys():
		try:
			vars[key] = vars[key][gpix,...]
			bpix = np.where(vars[key]==attrs[key]['FILLVAL'])[0]
			if len(bpix)>0:
				try:
					vars[key][bpix]=np.nan
				except ValueError:
					pass
		except:
			pass
		
	return(vars, attrs)

def load_kernels(dt, verbose=False):
	'''Load all necessary SPICE kernels for SWEAP ephemeris calcs'''
	import spiceypy
	
	#Load in leap second kernel
	tls_path = get_tls_path()
	spiceypy.furnsh(tls_path)
	
	#Load spacecraft clock file
	sclk_path = get_sclk_path()
	spiceypy.furnsh(sclk_path)
	
	#Find which unique dates we have
	#Since attitude history is in daily files, we'll need to load in all of them
	uniq_dates = []
	for thisdt in dt:
		date = datetime(thisdt.year,thisdt.month,thisdt.day)
		if date not in uniq_dates: uniq_dates.append(date)
		
	#Load in the daily attitude history files
	for date in uniq_dates:
		ah_path = get_attitude_history_path(date)
		if verbose: print('Loading AH file: {:}'.format(ah_path))
		if ah_path:
			spiceypy.furnsh(ah_path)
		else:
			raise ValueError('Attitude history file not available for this date')
	
	#Load spacecraft ephemeris
	ephem_path = get_ephemerides_path()
	spiceypy.furnsh(ephem_path)
	
	#Load frame kernel
	frame_path = get_frame_kernel_path()
	spiceypy.furnsh(frame_path)
	
	#Load PSP dyn kernel
	psp_dyn_path = get_psp_dyn_kernel_path()	
	spiceypy.furnsh(psp_dyn_path)
	
	#Load SWEAP frame kernels
	sweap_kernel_path = get_sweapframe_kernel_path()
	spiceypy.furnsh(sweap_kernel_path)
	
	#Load planetary ephemerides
	planetary_ephem_path = get_planetary_ephemeris_path()
	spiceypy.furnsh(planetary_ephem_path)
	
	#Load definition of heliospheric coordinate frames
	spiceypy.furnsh('/psp/code/spice_kernels/heliospheric.tf')
	
	#Load planetary body extent kernel
	spiceypy.furnsh('/psp/code/spice_kernels/pck00010.tpc')
		
	#Define the body of interest
	spiceypy.boddef('SPP_SPACECRAFT', -96)
	
def calculate_carrington(dt, verbose=False):
	'''Use SPICE to calculate carrington radius, lat, lon'''
	
	import spiceypy
	
	targ = '-96'
	obs = 'Sun'
	corr = 'None'
	
	#make sure it is an array 
	#input could be a single datetime object, or a list of datetime objects, or a numpy array of datetime objects
	dt = np.array(dt)
	
	#convert the datetime objects to MET and to the strings of seconds/subseconds that SPICE expects
	met = np.array([(i-datetime(2010,1,1)).total_seconds() for i in dt])
	met_sec = np.floor(met).astype(int)
	met_sec_str = np.array(['{:1.0f}'.format(i) for i in met_sec])
	met_subsec_base50000 = ((met-met_sec)*50000).astype(int)
	met_subsec_str = np.array(['{:05.0f}'.format(i) for i in met_subsec_base50000])
		
	#load all PSP/SWEAP kernels
	load_kernels(dt, verbose=verbose)
	
	#convert to ET from MET
	et_array = np.array([spiceypy.scs2e(-96, met_sec_str[i_dt]+':'+met_subsec_str[i_dt]) for i_dt in range(len(dt))])
	
	#Get cartesian position of s/c in carrington coords
	state, light_time = spiceypy.spkezr(targ, et_array, 'IAU_SUN', corr, obs)
	state = np.array(state)	
	sc_pos_km_carr = state[:,0:3]
	
	#Convert cartesian to radius/lat/lon
	rad = np.zeros(len(dt))
	carrlon = np.zeros(len(dt))
	carrlat = np.zeros(len(dt))
	
	#calculate radius latitude longitude for each cartesian vector
	for i_dt in range(len(dt)):
		thisvec = sc_pos_km_carr[i_dt]
		thisrad,thislon,thislat = spiceypy.reclat(thisvec)
		rad[i_dt] = thisrad/au_km
		carrlon[i_dt] = thislon
		carrlat[i_dt] = thislat
	
	return(rad,carrlon,carrlat)

	
def calculate_sunangle(dt, verbose=False):
	'''Use SPICE to calculate off-pointing angle'''
	import spiceypy
	
	#make sure it is an array 
	#input could be a single datetime object, or a list of datetime objects, or a numpy array of datetime objects
	dt = np.array(dt)
	
	#convert the datetime objects to MET and to the strings of seconds/subseconds that SPICE expects
	met = np.array([(i-datetime(2010,1,1)).total_seconds() for i in dt])
	met_sec = np.floor(met).astype(int)
	met_sec_str = np.array(['{:1.0f}'.format(i) for i in met_sec])
	met_subsec_base50000 = ((met-met_sec)*50000).astype(int)
	met_subsec_str = np.array(['{:05.0f}'.format(i) for i in met_subsec_base50000])
		
	#load all PSP/SWEAP kernels
	load_kernels(dt, verbose=verbose)
	
	#convert to ET from MET
	et_array = np.array([spiceypy.scs2e(-96, met_sec_str[i_dt]+':'+met_subsec_str[i_dt]) for i_dt in range(len(dt))])
	
	#array to store sun angle values
	sunangle = np.zeros((len(dt)))
	
	#loop through each date/time
	for i_dt in range(len(dt)):
			
		#et at this data point
		et = et_array[i_dt]
		
		#get attitude at given ET
		cmat = spiceypy.pxform('SPP_SPACECRAFT','PSP_RTN',et)
		scz_rtn = np.dot(cmat,[0,0,-1]) 		
		scz_rcomp = scz_rtn[0] #the R component of the S/C Z axis
		
		#calculate sunangle
		sunangle[i_dt,...] = np.arccos(scz_rcomp)*180./np.pi
	
	return(sunangle)
	
def calculate_ephemeris(dt, verbose=False):
	'''Calculate all useful ephemeris info for given dt array'''
	import spiceypy
	
	dt = np.array(dt)
	
	met = np.array([(i-datetime(2010,1,1)).total_seconds() for i in dt])
	met_sec = np.floor(met).astype(int)
	met_sec_str = np.array(['{:1.0f}'.format(i) for i in met_sec])
	met_subsec_base50000 = ((met-met_sec)*50000).astype(int)
	met_subsec_str = np.array(['{:05.0f}'.format(i) for i in met_subsec_base50000])
	
	targ = '-96'
	targ_int = int(targ)
	targ_name = 'PSP'
	obs = 'Sun'
	corr = 'None'
	
	load_kernels(dt, verbose=verbose)
	

	et_array = np.array([spiceypy.scs2e(targ_int, met_sec_str[i_dt]+':'+met_subsec_str[i_dt]) for i_dt in range(len(dt))])
	
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'PSP_RTN', corr, obs)
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
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'SPP_HCI', corr, obs)
	state = np.array(state)	
	sc_pos_km_hci = state[:,0:3]
	sc_vel_kms_hci = state[:,3:6]
	
	state, light_time = spiceypy.spkezr(targ, et_array, 'ECLIPJ2000', corr, obs)
	state = np.array(state)	
	sc_pos_km_hae = state[:,0:3]
	sc_vel_kms_hae = state[:,3:6]
	
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
		cmat[i_dt,...] = spiceypy.pxform('SPP_SPACECRAFT','PSP_RTN',et)
		
		SCX_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[1,0,0]) 
		SCY_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[0,1,0]) 
		SCZ_RTN[i_dt,...] = np.dot(cmat[i_dt,...],[0,0,1]) 
		
		cmat_sunvec[i_dt,...] = np.dot(cmat[i_dt,...],[0,0,-1]) 		
		scz_rcomp = cmat_sunvec[i_dt,0] #the R component of the S/C Z axis
		
		sunangle[i_dt,...] = np.arccos(scz_rcomp)*180./np.pi
	
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
	dat['SC_R_AU'] = rad/au_km
	dat['cmat'] = cmat

	return(dat)
	import pdb; pdb.set_trace()

	
def get_oaf_filename(orbit):
	'''Get the most recent OAF filename for a given orbit'''
	
	import os.path
	import glob
	
	#get start date of orbit
	#oaf files are in a directory named by the year of start of orbit
	aphelia = get_aphelia()
	ap_date = [i[1] for i in aphelia]
	ap_dist = [i[2] for i in aphelia]
	year = ap_date[orbit-1].year
	
	#create glob string to search
	dir = '/psp/data/moc_data_products/orbit_activity_file/{:04.0f}'.format(year)
	globstr = os.path.join(dir,'ORB_{:1.0f}_20[0-9][0-9]-'.format(orbit)+'[0-9]'*3+'-'+'[0-9]'*6+'.oaf') #good through the year 2099
	files = sorted(glob.glob(globstr))
	
	if len(files)>0: 
		filename = files[-1]
	else:
		filename = False
	
	return(filename)
	
def get_pasf_filename(orbit,segment):
	'''Get the most recent PAS filename for a given orbit'''
	
	import os.path
	import glob
	
	#get start date of orbit
	#oaf files are in a directory named by the year of start of orbit
	aphelia = get_aphelia()
	ap_date = [i[1] for i in aphelia]
	ap_dist = [i[2] for i in aphelia]
	year = ap_date[orbit-1].year
	
	#create glob string to search
	teamsdir = '/psp/data/teams/psp_soc/orbit_{:02.0f}/PASF'.format(orbit)
	mdpdir = '/psp/data/moc_data_products/payload_activity_schedule'
		
	teamsglob = os.path.join(teamsdir,'{:02.0f}_{:02.0f}_FORMOC_[0-9][0-9][0-9]'.format(orbit,segment)+'.pasf') #good through the year 2099
	mdpglob = os.path.join(mdpdir,'{:02.0f}_{:02.0f}_FORMOC_[0-9][0-9][0-9]'.format(orbit,segment)+'.pasf') #good through the year 2099
	
	#find files and sort them
	teams_files = sorted(glob.glob(teamsglob))
	mdp_files = sorted(glob.glob(mdpglob))
	
	#get modified date from both of the most recent files
	try:
		mdate_teams = os.path.getmtime(teams_files[-1])
	except:
		mdate_teams = 0	
	try:
		mdate_mdp = os.path.getmtime(mdp_files[-1])
	except:
		mdate_mdp = 0

	#return the most recent file; or False if no file found
	if mdate_teams > mdate_mdp:
		filename = teams_files[-1]
	elif mdate_teams < mdate_mdp:
		filename = mdp_files[-1]
	elif ( (mdate_teams==mdate_mdp) & (mdate_teams!=0) ):
		filename = mdp_files[-1]
	else:
		filename = False
	
	return(filename)
	
	
def get_planetary_ephemeris_path():
	globdir = '/psp/data/moc_data_products/planetary_ephemeris/'
	globstr = globdir + 'de[0-9][0-9][0-9].bsp'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)

	
def get_frame_kernel_path():
	globdir = '/psp/data/moc_data_products/frame_kernel/'
	globstr = globdir + 'spp_v[0-9][0-9][0-9].tf'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
		
def get_sweapframe_kernel_path():
	globdir = '/psp/data/moc_data_products/frame_kernel/'
	globstr = globdir + 'spp_sweap_v[0-9][0-9][0-9].ti'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
	
def get_psp_dyn_kernel_path():	
	globdir = '/psp/data/moc_data_products/frame_kernel/'
	globstr = globdir + 'psp_dyn_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9].tf'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
		
def get_ephemerides_path():
	globdir = '/psp/data/moc_data_products/ephemerides/'
	globstr = globdir + 'spp_nom_20180812_20250831_v[0-9][0-9][0-9]_*.bsp'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
	
	
def get_attitude_history_path(dt):
	globdir = '/psp/data/moc_data_products/attitude_history/{:04.0f}/'.format(dt.year)
	globstr = globdir + 'spp_{:04.0f}_{:03.0f}_[0-9][0-9].ah.bc'.format(dt.year,dt.timetuple().tm_yday)
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
	
def get_tls_path():
	globdir = '/psp/data/moc_data_products/leap_second_kernel/'
	globstr = globdir + 'naif00[0-9][0-9].tls'
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
		
def get_sclk_path():
	globdir = '/psp/data/moc_data_products/operations_sclk_kernel/'
	globstr = globdir + 'spp_sclk_[0-9][0-9][0-9][0-9].tsc'	
	files = glob.glob(globstr)
	try:
		return(sorted(files)[-1])
	except:
		return(False)
	
def get_ephem_data(dt, idl=False):
	filename = get_ephem_filename(dt,idl=idl)
	if filename:
		if idl:
			j2000epoch = datetime(2000,1,1,12)
			dat = readsav(filename)
			dat['Epoch'] = np.array([j2000epoch+timedelta(seconds=i/1e9) for i in dat['e']])
			dat['SUN_ANGLE'] = dat['cmat_sunang']*180/np.pi
		else:
			from spacepy import pycdf
			dat = pycdf.CDF(filename)
		return(dat)
	else:
		return(False)
	
def get_ephem_filename(dt, idl=False):
	'''Get the most recent ephemeris file for a given date'''
	
	import os.path
	import glob
	
	#create glob string to search
	dir = '/psp/data/sci/sweap/spc/EPHEMERIS/{:04.0f}/{:02.0f}'.format(dt.year, dt.month)
	yyyymmdd_str = '{:04.0f}{:02.0f}{:02.0f}'.format(dt.year, dt.month, dt.day)	
	if idl:
		globstr = os.path.join(dir,'spp_swp_spc_l3ephem_{:}.idl'.format(yyyymmdd_str))
	else:
		globstr = os.path.join(dir,'spp_swp_spc_ephem_{:}_v[0-9][0-9].cdf'.format(yyyymmdd_str))
		
	files = sorted(glob.glob(globstr))
	if len(files)>0: 
		filename = files[-1]
	else:
		filename = False
	
	return(filename)
	

	
	
def getfilename(dt, level, ionelectron='i', apid=0x354, version=-1, mode=0, verbose=False):
	"""Return path to a SPC data file of requested datetime, level, ion/elec/fa/cal, version=-1 gives most recent version"""
	
	import numpy, glob
	import numpy as np
	
	allowed_apids = [0x343,0x351,0x352,0x353,0x354,0x35e,0x35f]
	if apid not in allowed_apids:
		raise ValueError('Allowed apid values: '+ repr(allowed_apids))
	
	#Find the generic path string to search on (NB L0p5,L1 have diff. formats then L2,L3)
	base_path = '/psp/data/sci/sweap/spc/'
	if apid==0x343: base_path = '/psp/data/sci/sweap/swem/'
	
	#date/time strings in filenames
	yyyymmdd_str = '{:04.0f}{:02.0f}{:02.0f}'.format(dt.year, dt.month, dt.day)
	yyyy_mm = '{:04.0f}/{:02.0f}'.format(dt.year, dt.month)
	
	#text for apid as used in filenames
	apidstr = (hex(apid)[2:]).upper()
	
	#specify locations for each type of file
	if apid==0x343: 
		globdir = base_path+'L{:1.0f}/swem_ana_hkp/'.format(level)+yyyy_mm+'/'
		if level==0:
			raise ValueError ('Level==0 not allowed for 0x343 packets')
		if level==1:
			globfile = 'psp_swp_swem_ana_hkp_L{:1.0f}'.format(level)+'_'+yyyymmdd_str+'_v[0-9][0-9].cdf'
		if level>1:
			raise ValueError('Level>1 not allowed for 0x343 packets')			
	if apid==0x351:
		if level==0:
			raise ValueError ('Level==0 not allowed for 0x351 packets')
		if level==1:
			globfile = 'spp_swp_spc_APID{:}L{:1.0f}'.format(apidstr,level)+'_'+yyyymmdd_str+'_v[0-9][0-9].cdf'		
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/APID{:}/'.format(apidstr)
		if level>1: raise ValueError('Level>1 not allowed for 0x351 packets')
	if apid==0x352:
		raise ValueError('0x352 file finding not yet implemented')
	if apid in [0x353,0x354]:
		allowed_ionelectron = ['i','e','f','','ical','ifa'] #used to include 'hsk', but now ionelectron only works for 0x353,0x354 packets
		if ionelectron not in allowed_ionelectron:
			raise ValueError('Allowed ionelectron values: '+ repr(allowed_ionelectron))	
		if level==0:
			raise ValueError ('Level==0 not allowed for 0x353,354 packets')
		if level==1:		
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/APID{:}/'.format(apidstr)
			globfile = 'spp_swp_spc_APID{:}L{:1.0f}'.format(apidstr,level)+'_'+yyyymmdd_str+'_v[0-9][0-9].cdf'
		if level>1:
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/'
			globfile = 'spp_swp_spc_l{:1.0f}{:}'.format(level,ionelectron)+'_'+yyyymmdd_str+'_v[0-9][0-9].cdf'	
		if level>3: raise ValueError('L3 is highest available level for 0x353,0x354')			
	if apid==0x35e:
		if level==0:
			raise ValueError ('Level==0 not allowed for 0x35E packets')
		if level==1:
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/APID{:}/'.format(apidstr)
			globfile = 'spp_swp_spc_APID35EL1_'+yyyymmdd_str+'_v[0-9][0-9].cdf'
		if level>1: raise ValueError('L1 is highest available level for 0x35e')
	if apid==0x35f:
		if level==0:
			raise ValueError ('Level==0 not allowed for 0x35F packets')
		if level==1: 
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/APID{:}/'.format(apidstr)
			globfile = 'spp_swp_spc_APID35FL1_'+yyyymmdd_str+'_v[0-9][0-9].cdf'
		if level==2: 
			globdir = base_path+'L{:1.0f}/'.format(level)+yyyy_mm+'/'
			globfile = 'spp_swp_spc_APID35Fl2hsk_'+yyyymmdd_str+'_v[0-9][0-9].cdf'
		if level>2: raise ValueError('L2 is highest available level for 0x35f')
	
	
	globstr = globdir + globfile	
	if verbose: print('using globstr: {:}'.format(globstr))
	
	files = glob.glob(globstr)
	versions = [int(i[-6:-4]) for i in files]
	
	#And find the requested version
	if version==-1:
		try:
			maxind = np.argmax(versions)
		except ValueError:
			if verbose: print('No data files found')
			return(False)
		
		filepath = files[maxind]
		return(filepath)
	else:
		if version in versions:
			return(files[np.where(versions==version)[0]])
		else:
			return(False)

def get_dustevents(dtstart=datetime(2018,10,2), dtend=datetime.now(), verbose=False):
	"""Get list of dust events and sizes between dtstart and dtend"""
	dust_base = '/home/acase/data/Dust_data_Compare_3_orbits_TDS_V34/'
	dt = dtstart
	time = []
	amp =[]
	unix_epoch = datetime(1970,1,1)
	while dt < dtend+timedelta(days=1):
		savpath = dust_base+'PSP_dust_sav_file_TDS_{:04.0f}{:02.0f}{:02.0f}.sav'.format(dt.year,dt.month,dt.day)
		try:
			dat = readsav(savpath)
		except:
			pass
		thistime = dat['dust_hits_time_out_tds']
		try:
			thistime_dt = [unix_epoch+timedelta(seconds=i) for i in thistime]
		except:
			thistime_dt = [unix_epoch+timedelta(seconds=i) for i in [thistime]]
			
		time.extend(thistime_dt)
		thisamp = dat['dust_hits_amps_out_tds']
		try:
			amp.extend(dat['dust_hits_amps_out_tds'])
		except:
			amp.extend([dat['dust_hits_amps_out_tds']])
		
		dt = dt+timedelta(days=1)
	return(np.array(time),np.array(np.abs(amp)))
	
	
def get_dustrates(dtstart=datetime(2018,10,2), dtend=datetime.now(), verbose=False, timebin=60, threshold=0):
	"""Get rates of particles with amplitudes above "threshold" within bins of "timebin" seconds size"""
	tm,amp = get_dustevents(dtstart=dtstart, dtend=dtend, verbose=verbose)
	
	thisdt = dtstart
	tm_rates = []
	cnt_rates = []
	time_increment = timedelta(seconds=timebin)
	while thisdt < dtend:
		gpix = np.where( (tm < thisdt+time_increment) & (tm > thisdt)  & (amp>threshold))[0]
		tm_rates.append(thisdt)
		cnt_rates.append(len(gpix))
		thisdt = thisdt+time_increment 
	return(np.array(tm_rates), np.array(np.double(cnt_rates))/timebin)
	
def get_scpot(dtstart, dtend=False, basepath='/home/acase/data/Vsc/', verbose=False):
	"""Get average spacecraft potential between dtstart and dtend"""
	
	#Make sure user input is reasonable
	if not dtend:
		dtend = datetime(dtstart.year,dtstart.month,dtstart.day,23,59,59,999999)
	
	if ( (type(dtstart)!=datetime) | ((type(dtend)!=datetime) ) ):
		raise TypeError('dtstart and dtend must be datetime objects')
	
	if dtend<dtstart:
		raise ValueError('dtstart must be before dtend')
	
	#Define time epoch
	dt0 = datetime(1970,1,1,0,0,0) #times in files are unix time
	
	#lists to hold values from each day we read in
	dt = []
	volt = []
	
	#keep track of which day we're reading in
	thisday = datetime(dtstart.year,dtstart.month,dtstart.day,0,0,0)
	
	#loop through each day and read in sc potential file
	while thisday<=dtend:
		
		#construct the filename
		yyyymmss = '{:4.0f}{:02.0f}{:02.0f}'.format(thisday.year,thisday.month,thisday.day)
		file = basepath + 'Vsc_interp_to_1NYs_regular_grid_'+yyyymmss+'.sav'
		if verbose: print('***INFO*** Using file: {:}'.format(file))
		try:
			#read data from file
			dat = readsav(file)
			unix = dat[dat.keys()[0]][0]['x']
			thisdt = [dt0+timedelta(seconds=i) for i in unix]
			thisvolt = dat[dat.keys()[0]][0]['y']			
			
			#extend data arrays
			dt.extend(thisdt)
			volt.extend(thisvolt)
			if verbose: print('***INFO*** finished day: {:}'.format(thisday.isoformat()))
		except:
			if verbose: print('***ERROR*** could not read day: {:}'.format(thisday.isoformat()))
		
		#iterate to the next day
		thisday = thisday + timedelta(days=1)
	
	
	dt = np.array(dt)
	volt = np.array(volt)
	
	#now trim to the specific time periods that user wanted
	gpix = np.where( (dt>=dtstart) & (dt<=dtend) )[0]
	
	return(dt[gpix],volt[gpix])
	