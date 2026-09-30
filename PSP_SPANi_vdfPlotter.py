
from spacepy import pycdf 
import wget
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import os.path
from pathlib import Path
import datetime
import bisect
import autoplot as ap
import jpype

import argparse
import timeparser

import socket
localMachine = socket.gethostname()
if localMachine == 'fc.cfa.harvard.edu':
	googleDrive_path = '/home/kpaulson/MyDrive/'
else:
	googleDrive_path = 'G:/My Drive/'
	#This will be a terrible way to 
	
from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)

def getDataSet(variableName, fileName, tr, getDep0=False, getDep1=False, getDep2=False):
	'''
	Needs autoplot to be imported as ap and to be initialized
	'''
	ds = ap.APDataSet()
	ds.setDataSetURI(fileName+'?{}&timerange={}'.format(variableName,tr)) 
	ds.doGetDataSet()
	ds_array = ap.to_ndarray(ds, '{}'.format(variableName))
	return ds_array

def getTrimmedDataSet(variableName, fileName, tr, getDep0=False, getDep1=False, getDep2=False):
	'''
	Needs autoplot to be imported as ap and to be initialized
	'''
	ds = ap.APDataSet()
	ds.setDataSetURI(fileName+'?{}&timerange={}'.format(variableName,tr)) 
	ds.doGetDataSet()
	ds_array = ap.to_ndarray(ds, '{}'.format(variableName))
	ds_array = Ops.trim(ds_array, tr)
	return ds_array

#def doitallRange(yyyy=2021, mm=1, dd=20, HH=2, MM=14, SS=30, deltaSeconds=10, species='H+'):
def main(args):
	
	ap.init()

	Util= jpype.JClass('org.autoplot.jythonsupport.Util') 
	Ops=  jpype.JClass('org.das2.qds.ops.Ops')
	
	if args.verbose: print(f' Will generate VDF plots integrated over {args.secondsInterval} seconds on {args.timerange} for {args.ionSpecies}')
	
	localMachine = socket.gethostname()
	
	for species in args.ionSpecies:
		print('species:',species)
		level = 'L2'
		if species == 'H+' or species == 'H':
			speciesTag = '00'
			if species == 'H':
				species = 'H+'
		elif species == 'He++' or species == 'He':
			if localMachine == 'fc.cfa.harvard.edu':
				speciesTag = '0a'
				level = 'L2B'
			else:
				speciesTag = '01'
			if species == 'He':
				species = 'He++'
				
		if args.verbose: print('  Ion Species:',species)
		
		days = Util.generateTimeRanges('$Y-$m-$d', args.timerange)
		
		for date in days:
			if args.verbose: print('   ',date)
		
			# This time stuff needs to be updated. timeparser isn't working, try "whenever"?
			#timemid = timeparser.parsedate(str(date))
			timeid = datetime.datetime.strptime(str(date), '%Y-%m-%d')
			
			year     = timeid.year
			month    = timeid.month
			day      = timeid.day
			hour     = timeid.hour
			minute   = timeid.minute
			second   = timeid.second
			
			if len(str(month)) < 2:
				month = '0'+str(month)
			
			(epoch, theta, phi, energy, eflux, rotMat) = getDailyData(date, level, speciesTag)
			if True:
				epoch_datetime = np.array([datetime.datetime.strptime(str(d)[:-3], "%Y-%m-%dT%H:%M:%S.%f") for d in epoch])
			if False:
				epoch = np.array(epoch)
				epoch_datetime64 = epoch.astype(np.datetime64)
				epoch_datetime = np.array([d.astype(datetime.datetime) for d in epoch_datetime64])
			
			#def do time slicing and 
			slices = Util.generateTimeRanges('$Y-$m-$dT$H:$M:$(S,span=%s)'%(args.secondsInterval), date)
			
			for timeSlice in slices:
				#timeStart = timeparser.parsedate(timeSlice)
				timeStart = datetime.datetime.strptime(str(timeSlice), '%Y-%m-%dT%H:%M:%S')
				timeEnd   = timeStart + datetime.timedelta(seconds=args.secondsInterval)
			
				if args.verbose:
					print('Desired timerange:',timeStart, ' to ', timeEnd)
					
				#print(epoch[10:15])
				#print(epoch_datetime[10:15])	
				#print(timeStart)
				tSliceIndex_start  = bisect.bisect_left(epoch_datetime, timeStart)
				tSliceIndex_end    = bisect.bisect_right(epoch_datetime, timeEnd)
				tSliceIndex = tSliceIndex_start+int((tSliceIndex_end-tSliceIndex_start)/2.)
				
				outputDirectory = f'{googleDrive_path}Research/PSP/SPAN/SPANi/VDFs/sf{speciesTag}/{year}/{month}/{day}/'
				outputFilename  = str(epoch_datetime[tSliceIndex].strftime("%Y-%m-%dT%H%M%S"))+f'_delta-{str(args.secondsInterval)}-seconds_sf{speciesTag}_vdf.png'
				if not os.path.exists(outputDirectory):
				#	os.mkdirs(outputDirectory)
					path = Path(outputDirectory)
					path.mkdir(parents=True)
				
				if os.path.exists(outputDirectory + outputFilename):
					if args.verbose: print('   File already exists, moving on...') 
					continue
				
				if args.verbose:
					print('Time of closest start data point:',epoch[tSliceIndex_start])
					print('Time of closest end data point:',epoch[tSliceIndex_end])

				#print(timeEnd - timeStart)
				epochSlice  = epoch_datetime[tSliceIndex]
				#print(epochSlice)
				thetaSlice  = theta[tSliceIndex_start:tSliceIndex_end].mean(axis=0)
				phiSlice    = phi[tSliceIndex_start:tSliceIndex_end].mean(axis=0)
				energySlice = energy[tSliceIndex_start:tSliceIndex_end].mean(axis=0)
				efluxSlice  = eflux[tSliceIndex_start:tSliceIndex_end].mean(axis=0)

				thetaReshaped = thetaSlice.reshape((8,32,8))
				phiReshaped = phiSlice.reshape((8,32,8))
				energyReshaped = energySlice.reshape((8,32,8))
				efluxReshaped = efluxSlice.reshape((8,32,8))

				if speciesTag == '00':
					mass_p = 0.010438870      #eV/c^2 where c = 299792 km/s
					charge_p = 1              #eV
				elif speciesTag == '01' or speciesTag == '0a':
					mass_p = 4*0.010438870      #eV/c^2 where c = 299792 km/s
					charge_p = 2              #eV

				#Define VDF
				numberFlux = efluxReshaped/energyReshaped
				vdf = numberFlux*(mass_p**2)/((2E-5)*energyReshaped)

				#Convert to velocity units in each energy channel
				vel = np.sqrt(2*charge_p*energyReshaped/mass_p)

				vx = vel * np.cos(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
				vy = vel * np.sin(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
				vz = vel *                                   np.sin(np.radians(thetaReshaped))


				thetaplot_cut = 0
				phiplot_cut = 2

				#phi_avg = np.nanmean(phiReshaped,axis=thetaplot_cut)
				#theta_avg = np.nanmean(thetaReshaped,axis=thetaplot_cut)
				#energy_avg=np.nanmean(energyReshaped,axis=thetaplot_cut)

				phiAngle_sliceIndex = 0
				
				phi_plane = phiReshaped[phiAngle_sliceIndex,:,:]
				theta_plane = thetaReshaped[phiAngle_sliceIndex,:,:]
				energy_plane = energyReshaped[phiAngle_sliceIndex,:,:]
				vel_plane = np.sqrt(2*charge_p*energy_plane/mass_p)

				df=np.nansum(vdf,axis=thetaplot_cut)
				

				vx_plane = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
				vy_plane = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
				vz_plane = vel_plane *                                   np.sin(np.radians(theta_plane))
				
				vmin = 1
				vmax = 8
				levels = np.logspace(vmin, vmax, (2*(vmax-vmin))+1)
				fig,ax=plt.subplots()
				fig.set_size_inches(12, 9)
				cs=ax.contourf(vx_plane, vz_plane, df, locator=ticker.LogLocator(numticks=(vmax-vmin)+1), cmap=cm.turbo, levels=levels)
				cbar = fig.colorbar(cs)
				ax.set_xlim(-1000,0) #(-2000,0)
				ax.set_ylim(-750, 750) #(-1500, 1500)
				ax.set_xlabel('Vx (km/s) [~RTN R]', fontsize=22)
				plt.xticks(fontsize=22)
				ax.set_ylabel('Vz (km/s) [~RTN N]', fontsize=22)
				plt.yticks(fontsize=22)
				cbar.set_label('VDF ($cm^{-3} km^{-1} s$)', fontsize=22)
				ax.set_title(str(epoch_datetime[tSliceIndex].isoformat(timespec='seconds'))+f' +/- {(timeEnd - timeStart)/2}   '+species)
				#plt.show()
				
				fig.savefig(outputDirectory + outputFilename)
				plt.close(fig)

def getDailyData(date, level, speciesTag):
	
	timeid = datetime.datetime.strptime(str(date), '%Y-%m-%d')
	
	year     = timeid.year
	month    = timeid.month
	day      = timeid.day
	hour     = timeid.hour
	minute   = timeid.minute
	second   = timeid.second
	
	if len(str(month)) < 2:
		month = '0'+str(month)
				
	if localMachine == 'SI393828':
		localDirectory       = f'G:/My Drive/Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/{year}/{month}/'
		localDirectory_ap    = f'G:/My Drive/Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/$Y/$m/'
		localDirectory_ap_L2 = f'G:/My Drive/Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/data/sci/sweap/spi/L2/spi_sf{speciesTag}/$Y/$m/'
	elif localMachine == 'fc.cfa.harvard.edu':
		localDirectory       = f'/psp/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/{year}/{month}/'
		localDirectory_ap    = f'/psp/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/$Y/$m/'
		localDirectory_ap_Test = '/psp/data/sci/sweap/spi/{}/spi_sf{}/$Y/$m/'
	localDirectoryPub     = f'{googleDrive_path}Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/{year}/{month}/'
	localDirectoryPub_ap  = '{}Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/{}/spi_sf{}/$Y/$m/'.format(googleDrive_path, level, speciesTag)
	
	VDfile_directoryRemote_ap = f'http://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/$Y/$m/'
	VDfile_filename_ap = 'psp_swp_spi_sf{}_{}_{}_$Y$m$d_v$v.cdf'
	
	epoch_string  = 'Epoch'
	theta_string  = 'THETA'
	phi_string    = 'PHI'
	energy_string = 'ENERGY'
	eflux_string  = 'EFLUX'
	rotMat_string = 'ROTMAT_SC_INST'
	if level == 'L2B':
		VDfile_filename_ap_L2B = f'psp_swp_spi_sf{speciesTag}_L2B_mom_$Y$m$d_v$v.cdf'
		#epoch_string  = 'Epoch'
		#theta_string  = 'THETA'
		#phi_string    = 'PHI'
		#energy_string = 'ENERGY'
		eflux_string  = 'DATA'
		#rotMat_string = 'ROTMAT_SC_INST'
			
	if localMachine == 'fc.cfa.harvard.edu':
		try:
			if args.verbose: print('     First looking at local folders for SPAN data')
			if level == 'L2B':
				if args.verbose: print('    SPAN file:',localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, 'mom'))
				epoch   = getTrimmedDataSet(epoch_string,   localDirectory_ap_Test.format('L2', '01') + VDfile_filename_ap.format('01', 'L2', '8Dx32Ex8A'), date)
				theta   = getTrimmedDataSet(theta_string,   localDirectory_ap_Test.format('L2', '01') + VDfile_filename_ap.format('01', 'L2', '8Dx32Ex8A'), date)
				phi     = getTrimmedDataSet(phi_string,     localDirectory_ap_Test.format('L2', '01') + VDfile_filename_ap.format('01', 'L2', '8Dx32Ex8A'), date)
				energy  = getTrimmedDataSet(energy_string,  localDirectory_ap_Test.format('L2', '01') + VDfile_filename_ap.format('01', 'L2', '8Dx32Ex8A'), date)
				rotMat  = getTrimmedDataSet(rotMat_string,  localDirectory_ap_Test.format('L2', '01') + VDfile_filename_ap.format('01', 'L2', '8Dx32Ex8A'), date)
				eflux   = getTrimmedDataSet(eflux_string,   localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, 'mom'), date)
			else:
				if args.verbose: print('    SPAN file:',localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'))
				epoch   = getTrimmedDataSet(epoch_string,   localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
				theta   = getTrimmedDataSet(theta_string,   localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
				phi     = getTrimmedDataSet(phi_string,     localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
				energy  = getTrimmedDataSet(energy_string,  localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
				rotMat  = getTrimmedDataSet(rotMat_string,  localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
				eflux   = getTrimmedDataSet(eflux_string,   localDirectory_ap_Test.format(level, speciesTag) + VDfile_filename_ap.format(speciesTag, level, '8Dx32Ex8A'), date)
		except:
			if args.verbose: print('      ..that didn\'t work, let\'s look online') 
			raise 'noGoodStuffError'
			epoch   = getTrimmedDataSet(epoch_string,   VDfile_directoryRemote_ap + VDfile_filename_ap, date)
			theta   = getTrimmedDataSet(theta_string,   VDfile_directoryRemote_ap + VDfile_filename_ap, date)
			phi     = getTrimmedDataSet(phi_string,     VDfile_directoryRemote_ap + VDfile_filename_ap, date)
			energy  = getTrimmedDataSet(energy_string,  VDfile_directoryRemote_ap + VDfile_filename_ap, date)
			eflux   = getTrimmedDataSet(eflux_string,   VDfile_directoryRemote_ap + VDfile_filename_ap, date)
			rotMat  = getTrimmedDataSet(rotMat_string,  VDfile_directoryRemote_ap + VDfile_filename_ap, date)
	else:
		versionList = ['09', '08', '07', '06', '05', '04', '03', '02', '01', '00']
		# there is a html parser called "beautiful soup" that might work to solve this instead

		versionTest = 'notFound'
		for version in versionList:
			if versionTest == 'notFound':
				VDfile_directoryRemote = f'http://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/{level}/spi_sf{speciesTag}/{year}/{month}/'
				VDfile_filename = f'psp_swp_spi_sf{speciesTag}_{level}_8Dx32Ex8A_{year}{month}{day}_v{version}.cdf'
				
				
				if os.path.isfile(localDirectory + VDfile_filename):
					if args.verbose: print(f"Version {version} exists")
					VDfile = localDirectory + VDfile_filename
					versionTest = 'found'
				else:
					try:
						if args.verbose: print(f"Version {version} doesn't exist locally, searching online..")
						VDfile = wget.download(VDfile_directoryRemote + VDfile_filename + ' -P ' + localDirectory)
						versionTest = 'found'
						if args.verbose: print(f'Grabbed version {version}')
					except:
						continue
			elif versionTest == 'found':
				break

		print(localDirectory + VDfile_filename)
		cdf_VDfile = pycdf.CDF(VDfile)

		epoch           = cdf_VDfile['Epoch']
		theta           = cdf_VDfile['THETA']
		phi             = cdf_VDfile['PHI']
		energy          = cdf_VDfile['ENERGY']
		eflux           = cdf_VDfile['EFLUX']
		rotMat          = cdf_VDfile['ROTMAT_SC_INST']
	
	if eflux_string  == 'DATA':
		eflux = eflux*0.97E8
	
	return(epoch, theta, phi, energy, eflux, rotMat)
	

# print('Encounter 23 overview')
# for dd in np.arange(21,24,1):
	# for hh in np.arange(0,24,1):
		# for mm in np.arange(0,60,10):
			# try:
				# doitallRange(2025, 3, dd, hh, mm, 0, 15, species='H') 
				# doitallRange(2025, 3, dd, hh, mm, 0, 15, species='He') 
			# except:
				# continue



# Default inputs
#timerange = '2023-12-26 through 2023-12-30'
#stepSize = 50

def setup():
	'''get user input'''
	
	#outputDirectoryDefault = 'G:/My Drive/Research/PSP/WaveAnalysis/WaveSpectralFitting/'
	outputDirectoryDefault = googleDrive_path+'Research/PSP/WaveAnalysis/WaveSpectralFitting/'
	timerangeDefault = '2023-12-28'
	secondsIntervalDefault = 10
	ionSpeciesDefault = ['H+']
	savePlotDefault = False
	
	parser = argparse.ArgumentParser(description="process user inputs")
	parser.add_argument('-tr',  "--timerange", type=str, help="The total timerange to generate fits over", default=timerangeDefault)
	parser.add_argument('-sec', "--secondsInterval",  type=int, help="The number of seconds over which to integrate SPAN", default=secondsIntervalDefault)
	parser.add_argument('-ion', "--ionSpecies", type=str, nargs='+', help="The species of ions to plot (H+, or He++)", default=ionSpeciesDefault)
	parser.add_argument('-v',   "--verbose", action='store_true', help="Print out more progress messages if chosen (default: False)")
	#parser.add_argument('-dir', "--outputDirectory",  type=str, help="Where should these files go?", default=outputDirectoryDefault)
	#parser.add_argument('-plot', "--savePlots",  type=str2bool, help="Should we generate plot pngs?", default=savePlotDefault)

	args = parser.parse_args()
	return(args)

#timerange = args.timerange
#stepSize = args.stepSize

#trs = Util.generateTimeRanges('$Y-$m-$d $(H,span=6):00 ', '2023-12-26 through 2023-12-30') 


###
### DEFINE VARIABLES
###


if __name__=="__main__":
	args = setup()
	main(args)