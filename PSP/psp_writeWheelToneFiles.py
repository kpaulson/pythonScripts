import autoplot as ap

# 1. Start the background Java instance and pull in the functions
ap.init()

from autoplot import getDataSet, formatDataSet

outDirectory = 'G:/My Drive/Research/PSP/ReactionWheels/'

for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
	print('Processing %s...' % year)
	
	try:
		uri = (
			'http://research.ssl.berkeley.edu/data/spp/data/sci/fields/l2/f2_100bps/$Y/$m/psp_fld_l2_f2_100bps_$Y$m$d_v$v.cdf'
			'?PSP_FLD_L2_F2_100bps_SC_Reaction_Wheel_Speed_RW1'
			';PSP_FLD_L2_F2_100bps_SC_Reaction_Wheel_Speed_RW2'
			';PSP_FLD_L2_F2_100bps_SC_Reaction_Wheel_Speed_RW3'
			';PSP_FLD_L2_F2_100bps_SC_Reaction_Wheel_Speed_RW4'
		)
		
		# Pass only ONE argument to getDataSet
		wheelTones = getDataSet(uri, timeRange=str(year))
		
		formatDataSet(wheelTones, f"{outDirectory}psp_reactionWheelTones_{year}.csv")
		print(f'  Successfully saved {year}')
	except Exception as e:
		print(f'  Skipping {year} due to error: {e}')
		