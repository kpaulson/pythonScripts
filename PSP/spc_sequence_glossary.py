from astropy import units as u
import numpy as np

'''
47.8 sec to run 
Add Uniqe event of echo value 77

cmd.SPC_HALT(1)
pkg_main.sleep(2)
cmd.SPP_INTTIME(60)
pkg_main.sleep(2)
cm.SPC_RUN(1)
pkg_main.sleep(2)
cmd.SPC_ECHO1(77)

'''

# Dictionary mapping sequence numbers to script names and their run times
sequence_script_echo_value_run_time_MAY2024 = {
    1: ('FAKE',1 * u.second), # TODO: Clarify what this represents
    2: ('FAKE', 1 * u.second), # TODO: Clarify what this represents
    5: ('spcoff_seq5', 11.5* u.second),
    6: ('SPCPWRCYCLE', 5 * u.second), 
    7: ('spchvoff_seq7', 71.5 * u.second),
    8: ('spchvon_seq8', 154.25 * u.second),
    23: ('spcstartup_seq23', 41 * u.second),
    32: ('spc_lpt_sc', 5 * u.second),
    34: ('spc_electron_slow', 15 * u.second),
    36: ('spc_proton_slow_4kv', 11 * u.second), # TODO: Clarify what this represents
    53: ('spc_nominal_pwroff', 47 * u.second),
    57: ('spc_testtable_20201214.py',73 * u.second), # TODO: Clarify what this represents
    58: ('spcfpgafail', 5 * u.second),
    68: ('spc_proton_medium_4kv', 12 * u.second ), # TODO: Clarify what this represents
    70: ('spc_electron_medium', 15 * u.second),
    88: ('spc_shortcal', 58.5 * u.second),
    89: ('spc_fluxangle_repeat', 10 * u.second), # TODO: Clarify what this represents
    90: ('spc_proton_medfast_4kV',11 * u.second), # TODO: Clarify what this represents
    98: ('spc_magroll', 12.6* u.second), # TODO: Clarify what this represents
    102: ('spc_proton_medium_4kv_cruise', 12 * u.second ), # TODO: Clarify what this represents
    112: ('spc_shortcal_enc', 45.5 * u.second),
    113: ('spchvon_enc', 728 * u.second),
    114: ('spcstartup_enc', 730 * u.second), ####
    116: ('spchvon_enc_1kv',228 * u.second),
    117: ('spc_proton_medium_1kv', 12 * u.second ), # TODO: Clarify what this represents
    118: ('spcstartup_enc_1kv', 29 * u.second) ###
}



# Combined dictionary for sequence info including run time and echo value
sequence_script_echo_value_plus_runtime = {
    1: ('FAKE', 1 * u.second,0), # TODO: Clarify what this represents
    2: ('FAKE',  1 * u.second ,0), # TODO: Clarify what this represents
    5: ('spcoff_seq5', 11.5* u.second , np.nan), # TODO: Clarify what this represents - none are specified in the scrip
    6: ('SPCPWRCYCLE',  5 * u.second ,np.nan), # TODO: Clarify what this represents
    7: ('spchvoff_seq7', 71.5 * u.second ,np.nan), # TODO: Clarify what this represents
    8: ('spchvon_seq8', 154.25 * u.second ,[44,45]), #44 starting, #45 ending high voltage
    23: ('spcstartup_seq23', 41 * u.second ,[46,47]),
    32: ('spc_lpt_sc', 5 * u.second ,np.nan), # TODO: Clarify what this represents
    34: ('spc_electron_slow', 15 * u.second ,60 ),

    53: ('spc_nominal_pwroff', 47 * u.second , np.nan), # TODO: Clarify what this represents
    57: ('spc_testtable_20201214.py',73 * u.second  ,[83,84]),
    58: ('spcfpgafail',  5 * u.second ,np.nan), # TODO: Clarify what this represents

    70: ('spc_electron_medium', 15 * u.second ,59 ),
    88: ('spc_shortcal', 58.5 * u.second ,[81,82]),
    89: ('spc_fluxangle_repeat', 10 * u.second , [79,80]), #*****

    98: ('spc_magroll', 12.6* u.second ,76),


    113: ('spchvon_enc',  728 * u.second ,[97,98]),
    114: ('spcstartup_enc', 730 * u.second ,[97,98]), # TODO: Clarify what this represents
    116: ('spchvon_enc_1kv', 228 * u.second ,[101,102]),
    117: ('spc_proton_medium_1kv',  12 * u.second  ,103),
    118: ('spcstartup_enc_1kv', 29 * u.second ,[99,100]) # TODO: Clarify what this represents #Hidden echo value in deeper files references in run
}



# Dictionary mapping sequence numbers to script names and their run times
sequence_script_echo_value_run_time = {
    1: ('FAKE',1 * u.second), # TODO: Clarify what this represents
    2: ('FAKE', 1 * u.second), # TODO: Clarify what this represents
    5: ('spcoff_seq5', 11.5* u.second),
    6: ('SPCPWRCYCLE', 5 * u.second), 
    7: ('spchvoff_seq7', 71.5 * u.second),
    8: ('spchvon_seq8', 154.25 * u.second),
    23: ('spcstartup_seq23', 41 * u.second),
    32: ('spc_lpt_sc', 5 * u.second),
    34: ('spc_electron_slow', 15 * u.second),
    36: ('spc_proton_slow_4kv', 11 * u.second), # TODO: Clarify what this represents
    53: ('spc_nominal_pwroff', 47 * u.second),
    57: ('spc_testtable_20201214.py',73 * u.second), # TODO: Clarify what this represents
    58: ('spcfpgafail', 5 * u.second),
    68: ('spc_proton_medium_4kv', 12 * u.second ), # TODO: Clarify what this represents
    70: ('spc_electron_medium', 15 * u.second),
    88: ('spc_shortcal', 58.5 * u.second),
    89: ('spc_fluxangle_repeat', 10 * u.second), # TODO: Clarify what this represents
    90: ('spc_proton_medfast_4kV',11 * u.second), # TODO: Clarify what this represents
    98: ('spc_magroll', 12.6* u.second), # TODO: Clarify what this represents
    102: ('spc_proton_medium_4kv_cruise', 12 * u.second ), # TODO: Clarify what this represents
    112: ('spc_shortcal_enc', 45.5 * u.second),
    113: ('spchvon_enc', 728 * u.second),
    114: ('spcstartup_enc', 730 * u.second), ####
    116: ('spchvon_enc_1kv',228 * u.second),
    117: ('spc_proton_medium_1kv', 12 * u.second ), # TODO: Clarify what this represents
    118: ('spcstartup_enc_1kv', 29 * u.second) ###
}


# Dictionary mapping sequence numbers to script names and echo values
sequence_script_echo_value = {
    1: ('FAKE', 0), # TODO: Clarify what this represents
    2: ('FAKE', 0), # TODO: Clarify what this represents
    5: ('spcoff_seq5', np.nan), # TODO: Clarify what this represents - none are specified in the scrip
    6: ('SPCPWRCYCLE', np.nan), # TODO: Clarify what this represents
    7: ('spchvoff_seq7', np.nan), # TODO: Clarify what this represents
    8: ('spchvon_seq8',[44,45]), #44 starting, #45 ending high voltage
    23: ('spcstartup_seq23', [46,47]),
    32: ('spc_lpt_sc',np.nan), # TODO: Clarify what this represents
    34: ('spc_electron_slow',60 ),
    36: ('spc_proton_slow_4kv',73),
    53: ('spc_nominal_pwroff', np.nan), # TODO: Clarify what this represents
    57: ('spc_testtable_20201214.py', [83,84]),
    58: ('spcfpgafail',np.nan), # TODO: Clarify what this represents
    68: ('spc_proton_medium_4kv', 74),
    70: ('spc_electron_medium', 59 ),
    88: ('spc_shortcal',[81,82]),
    89: ('spc_fluxangle_repeat', [79,80]), #*****
    90: ('spc_proton_medfast_4kV',75),
    98: ('spc_magroll',76),
    102: ('spc_proton_medium_4kv_cruise', 78),
    112: ('spc_shortcal_enc',(95,96)),# Run daily *****
    113: ('spchvon_enc', [97,98]),
    114: ('spcstartup_enc', [97,98]), # TODO: Clarify what this represents
    116: ('spchvon_enc_1kv',[101,102]),
    117: ('spc_proton_medium_1kv',103),
    118: ('spcstartup_enc_1kv', [99,100]) # TODO: Clarify what this represents #Hidden echo value in deeper files references in run
}


# Combined dictionary for sequence info including run time and echo value
sequence_script_echo_value_plus_runtime = {
    1: ('FAKE', 1 * u.second,0), # TODO: Clarify what this represents
    2: ('FAKE',  1 * u.second ,0), # TODO: Clarify what this represents
    5: ('spcoff_seq5', 11.5* u.second , np.nan), # TODO: Clarify what this represents - none are specified in the scrip
    6: ('SPCPWRCYCLE',  5 * u.second ,np.nan), # TODO: Clarify what this represents
    7: ('spchvoff_seq7', 71.5 * u.second ,np.nan), # TODO: Clarify what this represents
    8: ('spchvon_seq8', 154.25 * u.second ,[44,45]), #44 starting, #45 ending high voltage
    23: ('spcstartup_seq23', 41 * u.second ,[46,47]),
    32: ('spc_lpt_sc', 5 * u.second ,np.nan), # TODO: Clarify what this represents
    34: ('spc_electron_slow', 15 * u.second ,60 ),
    36: ('spc_proton_slow_4kv', 11 * u.second ,73),
    53: ('spc_nominal_pwroff', 47 * u.second , np.nan), # TODO: Clarify what this represents
    57: ('spc_testtable_20201214.py',73 * u.second  ,[83,84]),
    58: ('spcfpgafail',  5 * u.second ,np.nan), # TODO: Clarify what this represents
    68: ('spc_proton_medium_4kv',  12 * u.second ,74),
    70: ('spc_electron_medium', 15 * u.second ,59 ),
    88: ('spc_shortcal', 58.5 * u.second ,[81,82]),
    89: ('spc_fluxangle_repeat', 10 * u.second , [79,80]), #*****
    90: ('spc_proton_medfast_4kV', 11 * u.second, 75),
    98: ('spc_magroll', 12.6* u.second ,76),
    102: ('spc_proton_medium_4kv_cruise',  12 * u.second ,78),
    112: ('spc_shortcal_enc', 45.5 * u.second , [95,96]),# Run daily *****
    113: ('spchvon_enc',  728 * u.second ,[97,98]),
    114: ('spcstartup_enc', 730 * u.second ,[97,98]), # TODO: Clarify what this represents
    116: ('spchvon_enc_1kv', 228 * u.second ,[101,102]),
    117: ('spc_proton_medium_1kv',  12 * u.second  ,103),
    118: ('spcstartup_enc_1kv', 29 * u.second ,[99,100]) # TODO: Clarify what this represents #Hidden echo value in deeper files references in run
}

def get_script_w_run_time(sequence_num):
    """
    Retrieve the script name and run time for a given sequence number.

    Parameters:
    sequence_num (int): The sequence number to check.

    Returns:
    tuple: The script name and run time for the specified sequence number.
    """
    script_info = sequence_script_echo_value_run_time.get(sequence_num)
    if script_info:
        script_name, run_time = script_info
        return script_name, run_time
    else:
        print(f'Sequence number {sequence_num} not found.')
        return None, None



# This function prepares the sequence data by matching sequence numbers to their run times
def prepare_sequence_data(sequence_num_list, sequence_dict = sequence_script_echo_value_run_time):
    prepared_data = []
    for seq_str in sequence_num_list:
        try:
            seq = int(seq_str)  # Convert string to integer
            if seq in sequence_dict:
                name, run_time = sequence_dict[seq]
                # Handle the case where run_time is np.nan or an Astropy Quantity
                run_time_value = run_time.to(u.second).value if not isinstance(run_time, float) else run_time
                prepared_data.append((seq, run_time_value))
        except ValueError:
            # Handle the case where the sequence number is not an integer
            print(f"Invalid sequence number: {seq_str}")
        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred: {e}")
    return prepared_data


def get_script_w_echo_value(sequence_num):
    """
    Retrieve the script name and echo value for a given sequence number.

    Parameters:
    sequence_num (int): The sequence number to check.

    Returns:
    tuple: The script name and echo value for the specified sequence number.
    """
    script_info = sequence_script_echo_value.get(sequence_num)
    if script_info:
        script_name, echo_value = script_info
        return script_name, echo_value
    else:
        print(f'Sequence number {sequence_num} not found.')
        return None, None

# # Assuming `sequence_script_echo_value_run_time` is accessible and has been imported



def get_sequence_info_echo_value_plus_runtime(sequence_num):
    """
    Get script information based on the sequence number, including run time and echo value.

    Parameters:
    sequence_num (int): The sequence number to check.

    Returns:
    tuple: Script name, run time in seconds, and echo value(s).
    """
    info = sequence_script_echo_value_plus_runtime.get(sequence_num)
    if info:
        script_name, runtime, echo_value = info
        runtime_seconds = runtime.to(u.second).value if runtime is not None else None
        return script_name, runtime_seconds, echo_value
    else:
        print(f'Sequence number {sequence_num} not found.')
        return None, None, None


def prepare_sequence_echo_time_data(sequence_num):
     # Prepare data for the DataFrame
    data = []
    for i in sequence_num:
        script_name, runtime, echo_value = get_sequence_info_echo_value_plus_runtime(int(i))
        # If echo_value is a tuple or list, extract the first element, or use NaN if not available
        echo_value = echo_value[0] if isinstance(echo_value, (list, tuple)) else echo_value
        # Replace 'nan' with np.nan for compatibility
        echo_value = np.nan if echo_value == 'nan' else echo_value
        data.append([script_name, runtime, echo_value])

    return data


def main():
   
    # Example of how to use get_script_w_run_time function
    sequence_number = 118

    # Example Usage
    info = get_sequence_info_echo_value_plus_runtime(sequence_number)
    print(f"Script Name: {info[0]}, Runtime: {info[1]} seconds, Echo Value(s): {info[2]}")

    try:
        script, run_time = get_script_w_run_time(sequence_number)
        print(f'For sequence number {sequence_number}: Script is "{script}" and it runs for {run_time}.')
    except ValueError as e:
        print(e)
    
    # Example of how to use get_script_w_echo_value function
    try:
        script, echo_value = get_script_w_echo_value(sequence_number)
        print(f'For sequence number {sequence_number}: Script is "{script}" and echo value is {echo_value}.')
    except ValueError as e:
        print(e)
    
    # Uncomment below to enter Python debugger mode
    #import pdb; pdb.set_trace()
    
    return 'Main function completed'


if __name__ == "__main__":
    main_result = main()
    print(main_result)