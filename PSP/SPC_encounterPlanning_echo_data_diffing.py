"""
# SWEAP Telemetry Worksheet Analysis Script

Echo value location: http://sweap.cfa.harvard.edu/data/sci/sweap/spc/L2/2022/03/spp_swp_spc_APID35Fl2hsk_20220310_v00.cdf?_SPC_ECHO1

## Purpose
This script analyzes SWEAP Telemetry Worksheets (in xlsx format) to extract SPC commands. 
It serves the purpose of verifying if ratstat test results align with expected behavior by comparing the 
extracted SPC commands with the expected commands. This comparison helps identify any discrepancies or 
unexpected outcomes, aiding in the confirmation of test result correctness.

## How to Run 
To run the script, execute the following command:

```bash
python3 echo_data_diffing.py -o #-o orbit
#EXAMPLE -python3 echo_data_diffing.py -o 18 
"""
import pandas as pd
import numpy as np
import openpyxl
import re
import argparse
from datetime import datetime, timedelta
import csv
from collections import OrderedDict

from spc_sequence_glossary import get_script_w_run_time, get_script_w_echo_value

# Suppress openpyxl warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Define file paths
file_path_SWEAP_Telemetry = 'SWEAP_Telemetry_Worksheet.xlsx'
file_path_SWEAP_MEM = 'SWEAP_MEM.xlsx'

# import pdb; pdb.set_trace()


def process_sweap_telemetry(file_path, args):
    """
    Process the SWEAP Telemetry Excel sheet.

    :param file_path: Path to the Excel file.
    :param args: Command-line arguments.
    :return: List of processed telemetry data.
    """
    
    selected_orbit = args.orbit if args.orbit is not None else 15
    xls = pd.ExcelFile(file_path)
    selected_sheets = [sheet for sheet in xls.sheet_names if f'Orbit{selected_orbit}' in sheet]
    telemetry_data = []
    for sheet_name in selected_sheets:
        print(f"Loading data from sheet: {sheet_name}")
        data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        telemetry_data.extend(process_sheet_data(data, sheet_name, file_path))
    return telemetry_data


def process_sheet_data(data, sheet_name, file_path):
    """
    Process data from a single sheet and extract SPC commands.

    :param data: DataFrame containing sheet data.
    :param sheet_name: Name of the sheet.
    :param file_path: Path to the Excel file.
    :return: List of extracted SPC commands.
    """
    header_row_index = find_header_row(data)
    if header_row_index is None:
        return []
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
    event_description = data.iloc[:, 0].tolist()
    date = data.iloc[:, 3].tolist()
    time = data.iloc[:, 4].tolist()
    flight_GSEOS_command = data.iloc[:, 14].tolist()
    spc_events = [] 
    for row_num, (description, command) in enumerate(zip(event_description, flight_GSEOS_command), start=2):
        if isinstance(description, str) and 'SPC' in description:
            sequence_number = re.search(r'\((\d+)\)', command)
            sequence_number = sequence_number.group(1) if sequence_number else "Sequence # Not Found"
            corresponding_datetime = format_datetime(date[row_num - 2], time[row_num - 2])
            spc_events.append(f"{corresponding_datetime}|{command}|{sequence_number}")
    return spc_events


def find_header_row(data):
    """
    Find the header row in the data.

    :param data: DataFrame containing sheet data.
    :return: Index of the header row.
    """
    for index, row in data.iterrows():
        if row[0] == 'Event Description' or row[3] == 'Date' or row[4] == 'Time' or row[14] == 'FLIGHT GSEOS Command':
            return index
    return None

# def format_datetime(date_val, time_val):
#     """
#     Format date and time into datetime object.

#     :param date_val: Date value.
#     :param time_val: Time value.
#     :return: Formatted datetime string.
#     """
#     if isinstance(date_val, datetime):
#         date_part = date_val.date()
#         combined_datetime = datetime.combine(date_part, time_val)
#         return combined_datetime.strftime('%Y-%m-%d %H:%M:%S')
#     else:
#         return f"{date_val} {time_val}"

def format_datetime(date_val, time_val):
    """
    Normalize Excel/Pandas date & time fields to 'YYYY-MM-DD HH:MM:SS'.
    Handles datetime, pandas.Timestamp, timedelta, float (Excel fraction), and str.
    """
    # --- normalize date ---
    if isinstance(date_val, pd.Timestamp):
        date_part = date_val.date()
    elif isinstance(date_val, datetime):
        date_part = date_val.date()
    else:
        # fall back to pandas parsing (works for strings, excel serials, etc.)
        date_part = pd.to_datetime(date_val, errors="coerce").date()

    # --- normalize time -> datetime.time ---
    if isinstance(time_val, pd.Timestamp):
        t = time_val.time()
    elif isinstance(time_val, datetime):
        t = time_val.time()
    elif isinstance(time_val, timedelta):
        # Excel time as fraction-of-day becomes timedelta; map to time-of-day
        t = (datetime.min + time_val).time()
    elif isinstance(time_val, (int, float)):
        # Excel stores time as fraction of a day
        t = (datetime.min + timedelta(days=float(time_val))).time()
    elif isinstance(time_val, str):
        t = pd.to_datetime(time_val, errors="coerce").time()
    else:
        # last resort: midnight
        t = datetime.min.time()

    return datetime.combine(date_part, t).strftime("%Y-%m-%d %H:%M:%S")

#######################################################################################################


def process_sweap_mem(file_path):
    """
    Process the SWEAP MEM Excel sheet.

    :param file_path: Path to the SWEAP MEM Excel file.
    :return: Processed data as lists.
    """
    color_ordered_dict = OrderedDict([
        ('00000000', 'Transparent'), 
        ('00000001', 'Black'),
        ('FFC9DAF8', 'Grey'),
        ('FFFF0000', 'Red'),
        ('FF00FF00', 'Green'),
        ('FFFF00FF', 'Hot Pink'),
        ('FFFFFF00', 'Yellow'),
        ('FFFFC000', 'Orange-keys')
    ])
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook['Seqs']
    data_sweap_mem = pd.read_excel(file_path, sheet_name='Seqs')
    seq_num = data_sweap_mem.iloc[:, 2].tolist()
    seq_name = data_sweap_mem.iloc[:, 3].tolist()
    load_generated_script = data_sweap_mem.iloc[:, 6].tolist()
    column_letter = 'A'
    column_index = openpyxl.utils.column_index_from_string(column_letter)
    sweap_mem_data_value = []
    sweap_mem_information = []
    sweap_mem_information_w_color = []
    for row_num, (row, seq_num, seq_name, load_script) in enumerate(zip(sheet.iter_rows(min_col=column_index, max_col=column_index), seq_num, seq_name, load_generated_script), start=1):
        cell = row[0]
        color = cell.fill.start_color.index
        color_value = f"{color:0>8}"
        color_name = color_ordered_dict.get(color_value, "Unknown Color")
        if pd.isna(seq_num) or (color_value == '00000000' and color_name == 'Transparent'):
            continue
        else:
            data_info_w_color = f"Row {row_num + 1} | SEQ #: {seq_num} | SEQ name: {seq_name} | Load Generated: {load_script} | Color Value: {color_value} | Color is: {color_name}"
            data_info = f"SEQ #: {seq_num} | SEQ name: {seq_name} | Load Generated: {load_script}"
            data_value = f"{seq_num}|{seq_name}|{load_script}"
            sweap_mem_data_value.append(data_value)
            sweap_mem_information.append(data_info)
            sweap_mem_information_w_color.append(data_info_w_color)
    return sweap_mem_data_value 


SWEAP_MEM_data_value = process_sweap_mem(file_path_SWEAP_MEM)
SWEAP_MEM_data_value = [spc_info for spc_info in SWEAP_MEM_data_value if "SPC" in spc_info]



def process_echo_data(sequence_num, sequence_datetime):
    """
    Process echo data.

    :param sequence_num: Sequence numbers.
    :param sequence_datetime: Sequence datetimes.
    :return: Processed echo data.
    """
    echo_value = []
    dt_echo_value = []
    flightscripts_echo_info = []
    for index, s_num in enumerate(sequence_num):
        echo_info = get_script_w_echo_value(int(s_num))
        flightscripts_echo_info.append(echo_info[0])
        echo_value.append(echo_info[1])
        dt_echo_value.append(sequence_datetime[index])
    echo_value = np.array(echo_value)
    dt_echo_value = np.array(dt_echo_value)
    return echo_value, dt_echo_value, flightscripts_echo_info

######################################################################################################


def convert_quotes(text):
    return text.replace('"', "'")

def SWEAP_MEM_data_value_list(verbose_mode=False):
    flightscripts_type = []
    flightscripts_type_w_seq = []
    for i in SWEAP_MEM_data_value:
        seq = i.split("|")[0]
        data_values_split = i.split("|")[2]
        if data_values_split == 'nan':
            if verbose_mode:
                print("nan value:", data_values_split)
            continue
        try:
            converted_single_quotes = convert_quotes(data_values_split)
            start_quote = converted_single_quotes.index("'")
            end_quote = converted_single_quotes.rindex("'")
            flightscripts_types = converted_single_quotes[start_quote + 1:end_quote]
            fs_w_seq = f'{seq} {flightscripts_types}'
            flightscripts_type_w_seq.append(fs_w_seq)
            flightscripts_type.append(flightscripts_types)
        except Exception as e:
            if verbose_mode:
                print(f"Error processing data: {e}")
            continue
    return flightscripts_type_w_seq, flightscripts_type



flightscripts_type_w_seq, flightscripts_type = SWEAP_MEM_data_value_list(verbose_mode = False)



#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#*******************************************************************************************************


def setup_parser():
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-o', '--orbit', default=15, type=int, metavar='', help='(Optional) Specify the orbit number (default: 15)')
    # Add other arguments as needed
    
    return parser

# Parse the command-line arguments
args = setup_parser().parse_args()




sequence_num = []
sequence_datetime = []
sequence_command = []

for line_of_code in process_sweap_telemetry(file_path_SWEAP_Telemetry, args):
    splitted = line_of_code.split("|")
    if len(splitted) >= 3:
        datetime_value = splitted[0]
        command_value = splitted[1]
        sequence_value = splitted[2]

        if sequence_value != 'Sequence # Not Found':
            sequence_num.append(sequence_value)
            sequence_datetime.append(datetime_value)
            sequence_command.append(command_value)
        else:
            print(f"Sequence number not found for line: {line_of_code}")
        
        # Pass the file_path to process_sheet_data
        #echo_data.extend(process_sheet_data(pd.read_excel(file_path_SWEAP_Telemetry, sheet_name=selected_sheet_index, header=None), file_path_SWEAP_Telemetry))

#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************
#******************************************************************************************************

echo_value, dt_echo_value, flightscripts_echo_info = [], [], []

for index, s_num in enumerate(sequence_num):
    echo_info = get_script_w_echo_value(int(s_num))
    flightscripts_echo_info.append( echo_info[0] )
    echo_value.append(echo_info[1])
    dt_echo_value.append(sequence_datetime[index])

echo_value = [str(val) for val in echo_value]  # Convert all elements to strings
echo_value = np.array(echo_value)  # Convert to NumPy array
dt_echo_value = np.array(dt_echo_value)



################################################################################################################
################################################################################################################
def sequence36(inputDateTime):
    """spc_proton_slow_4kv"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=11))
    echoValues.append(73)
    return (timestamps, echoValues)

def sequence68(inputDateTime):
    """spc_proton_medium_4kv"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=12))
    echoValues.append(74)
    return (timestamps, echoValues)

def sequence89(inputDateTime):
    """spc_fluxangle_repeat"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=10))
    echoValues.append(78)
    timestamps.append(inputDateTime + timedelta(seconds=10))
    echoValues.append(80)
    return (timestamps, echoValues)

def sequence90(inputDateTime):
    """spc_proton_medfast_4kV"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=11))
    echoValues.append(75)
    return (timestamps, echoValues)

def sequence98(inputDateTime):
    """spc_magroll"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=12))
    echoValues.append(76)
    return (timestamps, echoValues)


def sequence102(inputDateTime):
    """spc_proton_medium_4kv_cruise"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=12))
    echoValues.append(78) 
    return (timestamps, echoValues)


def sequence112(inputDateTime):
    """spc_shortcal"""
    timestamps = []
    echoValues = []
    timestamps.append(inputDateTime + timedelta(seconds=20.4))
    echoValues.append(95) 
    timestamps.append(inputDateTime + timedelta(seconds=298.3))
    echoValues.append(96)
    return (timestamps, echoValues)

def calculate_timestamps_and_echo_values(sequence_number, input_datetime):
    sequence_functions = {
        '36': sequence36,
        '68': sequence68,
        '89': sequence89,
        '90': sequence90,
        '98': sequence98,
        '102': sequence102,
        '112': sequence112,# Add more sequences as needed
    }
    
    if sequence_number in sequence_functions:
        return sequence_functions[sequence_number](input_datetime)
    else:
        return [], []



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def main(args):
    """
    Main function
    """
    if args.verbose:
        print("Verbose mode is on.")
    
    # telemetry_data = SWEAP_Telemetry_WorkSheet(file_path_SWEAP_Telemetry)
    telemetry_data = process_sweap_telemetry(file_path_SWEAP_Telemetry,args)
    sequence_dt = [datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in dt_echo_value]
   
    # Create an eventList as a list of dictionaries
    eventList = [{'sequence_datetime': dt, 'sequence_number': sequence_number} for dt, sequence_number in zip(sequence_dt, sequence_num)]

    # Create lists to store timestamps and echo values
    timestamps_total = []
    echoValues_total = []

    for event in eventList:
        sequence_datetime = event['sequence_datetime']
        sequence_number = event['sequence_number']

        # Calculate timestamps and echo values based on sequence number and start time
        timestamps, echoValues = calculate_timestamps_and_echo_values(sequence_number, sequence_datetime)

        # Extend the total lists with the calculated values
        timestamps_total.extend(timestamps)
        echoValues_total.extend(echoValues)


    # Get the date from the first telemetry data
    first_data_date = datetime.strptime(telemetry_data[0][0:10], '%Y-%m-%d')

    # Replace "-" with "_" in the date string
    formatted_date_str = first_data_date.strftime("%Y_%m_%d")

    # Write the combined data to a CSV file
    with open(f'echovalues_orbit{args.orbit}_{formatted_date_str}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamp', 'Echo Value'])  # Write header
        for timestamp, echo_value in zip(timestamps_total, echoValues_total):
            csvwriter.writerow([timestamp, echo_value])


################################################################################################################
################################################################################################################

def setup():
    """
    Get user input using argparse.
    """
    parser = argparse.ArgumentParser(description='SWEAP Telemetry Worksheet Analysis')
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help='(Optional) Provide detailed information during execution')
    parser.add_argument('-o', '--orbit', default=15, type=int, metavar='', help='(Optional) Specify the orbit number (default: 15)')
    return parser.parse_args()

if __name__ == "__main__":
    args = setup()
    main(args)