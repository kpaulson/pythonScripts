import emcee, corner
import numpy as np
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.ion()

import autoplot as ap
import matplotlib.pyplot as plt
import numpy as np
import jpype
import csv
import os
import argparse

import socket
localMachine = socket.gethostname()
if localMachine == 'fc.cfa.harvard.edu':
	googleDrive_path = '/home/kpaulson/MyDrive/'
else:
	googleDrive_path = 'G:/My Drive/'

import sys
sys.path.append(googleDrive_path+'Software/Python/spectral_fits/')
import fit_spectra as srijanEmcee
import mcmc_functions as mcmc_funcs


def BPL_fit(x,y,init,verbose=False,calc_uncertainties=True):
	'''
	Fit data to form f(x) = { A(x/x_break)^(-α1) x < x_break, A(x/x_break)^(-α2) x >= x_break}
	
	Inputs:
	x   :  numpy array of x values
	y   :  numpy array of y values of the same size as x
	init   :   tuple of inital values for model (amplitude,x_break,alpha_1,alpha_2)

	'''
	init_model = amd.models.BrokenPowerLaw1D(*init)
	fit = amd.fitting.LMLSQFitter(calc_uncertainties=calc_uncertainties)

	p = fit(init_model, x, y, maxiter=200,filter_non_finite=True)
	param_cov = fit.fit_info['param_cov']
	
	if verbose:
		display(f'{p.amplitude.name}: {p.amplitude.value:.5f} +- {np.sqrt(param_cov[0,0]):.5f}')
		display(f'{p.x_break.name}: {p.x_break.value:.5f} +- {np.sqrt(param_cov[1,1]):.5f}')
		display(f'{p.alpha_1.name}: {p.alpha_1.value:.5f} +- {np.sqrt(param_cov[2,2]):.5f}')
		display(f'{p.alpha_2.name}: {p.alpha_2.value:.5f} +- {np.sqrt(param_cov[3,3]):.5f}')
	return p,param_cov


def getDataSet(variableName, fileName, tr, getDep0=False, getDep1=False, getDep2=False):
	'''
	Needs autoplot to be imported as ap and to be initialized
	'''
	ds = ap.APDataSet()
	ds.setDataSetURI(fileName+'?{}&timerange={}'.format(variableName,tr)) 
	ds.doGetDataSet()
	ds_array = ap.to_ndarray(ds, '{}'.format(variableName))
	return ds_array

def srijanFitter(xraw, yraw, outputDirectory='/tmp/'):
	# fitting the function
	X = np.log10(xraw)
	Y = np.log10(yraw)
	
	# pre-processing the data
	X, Y = srijanEmcee.preprocess_data(X, Y)
	Ndata = len(Y)
	
	#-----------------implementing MCMC fitting-----------------------#
	# limits to initialize walkers | all logs are of base 10
	init_pos = {}
	init_pos['log_xb'] = [-1, 1]
	init_pos['log_delta'] = [-5, 1]
	init_pos['log_alpha1'] = [-2, 2]
	init_pos['log_alpha2'] = [-2, 2]
	init_pos['const'] = [-3, 3]

	Nparams = len(init_pos.keys())
	Nwalkers = 20
	emcee_fitter = mcmc_funcs.mcmc_spectral_fit(init_pos, Nwalkers)

	sampler = emcee.EnsembleSampler(Nwalkers, Nparams, emcee_fitter.log_probability, args=(X, Y))
	sampler.run_mcmc(emcee_fitter.params_init, 200, progress=True)

	# plotting the mcmc results
	# Q is of the form: breakpoint, smoothnessOfBreak, slope1, slope2, intercept
	Q = srijanEmcee.plot_emcee_results(sampler, X, Y, Nparams, outputDirectory)
	return(Q)
	
	

def erase_csv_and_write_labels(filename, labels):
  """
  Erases the contents of a CSV file and writes the given labels 
  as the header row.

  Args:
    filename: The name of the CSV file.
    labels: A list of strings representing the column labels.
  """
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)

def append_line_to_csv(filename, line):
  """
  Appends a line to a CSV file. 

  Args:
    filename: The name of the CSV file.
    line: A list of values to be written as a row in the CSV.
  """
  with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(line)
	
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
	
	trs = Util.generateTimeRanges('$Y-$m-$d $(H,span=6):00 ', args.timerange) 
	outputDirectory = args.outputDirectory
	
	fitParameterLabels = ['time', 'breakpoint_inc', 'smoothness_inc', 'slope1_inc', 'slope2_inc', 'intercept_inc',
							'breakpoint_wave', 'smoothness_wave', 'slope1_wave', 'slope2_wave', 'intercept_wave',
							'breakpoint_all', 'smoothness_all', 'slope1_all', 'slope2_all', 'intercept_all']
	dayWrite = None
	
	for i in range(len(trs)-1):
		try:
			tr = trs[i] + ' to ' + trs[i+1]
			print(tr)
			
			if not os.path.exists(outputDirectory+'data/'):
				os.mkdir(outputDirectory+'data/')
			if not os.path.exists(outputDirectory+'plots/'):
				os.mkdir(outputDirectory+'plots/')
			
			dayNow = str(tr)[0:10]
			if dayNow == dayWrite:
				outputFile = outputDirectory+'data/'+'{}_spectralFitParameters.csv'.format(dayWrite)
			else:
				outputFile = outputDirectory+'data/'+'{}_spectralFitParameters.csv'.format(dayNow)
				erase_csv_and_write_labels(outputFile, fitParameterLabels)
				dayWrite = dayNow
				
			#if dailyFileExists:
				
			
			#tt=Ops.timegen('2024-09-01T00:00Z', '1hr', 10 )

			waveData   = ap.APDataSet()
			#coherency  = ap.APDataSet()
			#waveNormal = ap.APDataSet()

			if localMachine == 'fc.cfa.harvard.edu':
				waveAnalysis_file = '/mnt/nanook/waveAnalysis_PSP/WaveAnalysis_Files/v1.3/$Y/$m/PSP_WaveAnalysis_$Y-$m-$d_$(H,span=6)00_v1.3.cdf'
			else:
				waveAnalysis_file = googleDrive_path+'Research/PSP/WaveAnalysis/WaveAnalysis_Files/v1.3/$Y/$m/PSP_WaveAnalysis_$Y-$m-$d_$(H,span=6)00_v1.3.cdf'
			
			waveData.setDataSetURI(waveAnalysis_file+'?Wave_Power_b&timerange={}'.format(tr))  
			waveData.doGetDataSet()

			#coherency.setDataSetURI(waveAnalysis_file+'?coherency_b')  
			#coherency.doGetDataSet()

			#waveNormal.setDataSetURI(waveAnalysis_file+'?wave_normal_b')  
			#waveNormal.doGetDataSet()

			wavePower   = ap.to_ndarray(waveData,   'Wave_Power_b')
			fft_time    = ap.to_ndarray(waveData,   'FFT_time')     
			frequencies = ap.to_ndarray(waveData,   'Frequencies') 
			#coherency   = ap.to_ndarray(coherency,  'coherency_b')
			#waveNormal  = ap.to_ndarray(waveNormal, 'wave_normal_b')

			coherency  = getDataSet('coherency_b', waveAnalysis_file, tr)
			waveNormal = getDataSet('wave_normal_b', waveAnalysis_file, tr)
				   
			noise = np.where(wavePower<1E-4)
			wavePower[noise]='nan'
				
			tr_minis = Util.generateTimeRanges('$Y-$m-$d $H:(M,span=10)', tr) 

			step = args.stepSize #50
			index = 0
			while index < (len(wavePower)+step):
				#print(str(fft_time[index]).replace(':','')[:17])

				try:
					maxFreq = 40#64
					#index = 325	

					#@ Where statements not working right now
					waveLike  = np.where(coherency[index:index+step,1:maxFreq]>0.8)# & waveNormal<25).all() 
					turbulent = np.where(coherency[index:index+step,1:maxFreq]<=0.5)# or waveNormal<=25).any()
					
					frequencyArray = np.zeros((step, len(frequencies)))
					
					for ii in range(step):
						frequencyArray[ii,:] = frequencies
					
					xraw = frequencyArray[:,1:maxFreq][turbulent].flatten()
					yraw = wavePower[index:index+step][:,1:maxFreq][turbulent].flatten()
					
					xwave = frequencyArray[:,1:maxFreq][waveLike].flatten()
					ywave = wavePower[index:index+step][:,1:maxFreq][waveLike].flatten()
					
					xall = frequencyArray[:,1:maxFreq].flatten()
					yall = wavePower[index:index+step][:,1:maxFreq].flatten()
					
					sortedOrder = np.argsort(xraw)
					xraw = xraw[sortedOrder]
					yraw = yraw[sortedOrder]
					
					sortedOrder = np.argsort(xwave)
					xwave = xwave[sortedOrder]
					ywave = ywave[sortedOrder]
					
					sortedOrder = np.argsort(xall)
					xall = xall[sortedOrder]
					yall = yall[sortedOrder]
					
					#xraw = xraw[np.log10(xraw) > -0.4]
					#yraw = yraw[np.log10(xraw) > -0.4]
					
					#xraw = frequencies[:maxFreq][turbulent]
					#yraw = wavePower[index:index+step][:maxFreq][turbulent]

					srijanStyle = True
					if srijanStyle:
						Q_incoherent = srijanFitter(xraw, yraw)
						Q_wave = srijanFitter(xwave, ywave)
						Q_all = srijanFitter(xall, yall)

					#savePlots = False
					if args.savePlots:
						fig, ax = plt.subplots(figsize=(10, 8))
						ax.scatter(np.log10(frequencyArray[:,1:maxFreq][waveLike]), np.log10(wavePower[index:index+step][:,1:maxFreq][waveLike]), s=6, color='magenta', alpha=0.5, label='coherent')
						ax.scatter(np.log10(frequencyArray[:,1:maxFreq][turbulent]), np.log10(wavePower[index:index+step][:,1:maxFreq][turbulent]), s=6, color='black', alpha=0.5, label='incoherent')
						ax.plot(np.log10(xraw), mcmc_funcs.fit_func(np.log10(xraw), Q_incoherent[0], Q_incoherent[1], Q_incoherent[2], Q_incoherent[3], Q_incoherent[4]), 'r', label='fitted curve')
						ax.plot(np.log10(xall), mcmc_funcs.fit_func(np.log10(xall), Q_all[0], Q_all[1], Q_all[2], Q_all[3], Q_all[4]), 'blue', label='fitted curve')
						ax.axvline(Q_incoherent[0], color='r', ls='--', label='spectral break!cincoherent')
						ax.axvline(Q_all[0], color='blue', ls='--', label='spectral break!call')
						#ax.set_xscale('log') 
						#ax.set_yscale('log') 
						ax.grid(True)
						ax.set_xlabel('Log10(Frequency (Hz))')
						ax.set_ylabel('Log10(Wave Power (nT^2/Hz))')
						ax.set_title('Time: {:}'.format(str(fft_time[index+int(step/2)])[:19]))
						#plt.show()
						
						plt.savefig(outputDirectory+'plots/'+'%s_fitPoints.png'%str(fft_time[index+int(step/2)]).replace(':','')[:17])
						plt.close(fig)
					
					dataOut = [str(fft_time[index+int(step/2)])[:19]] + Q_incoherent.tolist() + Q_wave.tolist() + Q_all.tolist()
					append_line_to_csv(outputFile, dataOut)
					
					#np.save('/users/kpaulson/TMP_x1.npy', frequencies[:maxFreq][turbulent])
					#np.save('/users/kpaulson/TMP_y1.npy', wavePower[index][:maxFreq][turbulent])

					index += step
				except:
					index += step
		except:
			print('no wave file in timerange, moving on')
			continue

ap.init()

Util= jpype.JClass('org.autoplot.jythonsupport.Util') 
Ops=  jpype.JClass('org.das2.qds.ops.Ops')

# Default inputs
timerange = '2023-12-26 through 2023-12-30'
stepSize = 50

def setup():
	'''get user input'''
	
	#outputDirectoryDefault = 'G:/My Drive/Research/PSP/WaveAnalysis/WaveSpectralFitting/'
	outputDirectoryDefault = googleDrive_path+'Research/PSP/WaveAnalysis/WaveSpectralFitting/'
	timerangeDefault = '2023-12-28'
	stepSizeDefault = 50
	savePlotDefault = False
	
	parser = argparse.ArgumentParser(description="process user inputs")
	parser.add_argument('-tr', "--timerange", type=str, help="The total timerange to generate fits over", default=timerangeDefault)
	parser.add_argument('-s', "--stepSize",  type=int, help="The number of spectra to combine per fit", default=stepSizeDefault)
	parser.add_argument('-dir', "--outputDirectory",  type=str, help="Where should these files go?", default=outputDirectoryDefault)
	parser.add_argument('-plot', "--savePlots",  type=str2bool, help="Should we generate plot pngs?", default=savePlotDefault)

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