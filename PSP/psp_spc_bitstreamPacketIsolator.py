#!/usr/bin/env python3
"""
psp_spc_bitstreamPacketIsolater.py
----------------------------------
Isolates SPC telemetry packets (APIDs 0x351, 0x352, 0x353, 0x354, 0x35E, 0x35F)
from raw spacecraft telemetry bitstreams and filters them by Mission Elapsed Time (MET)
or UTC ISO 8601 timestamps.
"""

import argparse
import datetime
import gzip
import os
import re
import struct
import sys

# Target SPC APIDs
TARGET_SPC_APIDS = {0x351, 0x352, 0x353, 0x354, 0x35E, 0x35F}
SWEM_WRAPPER_APIDS = set(range(0x348, 0x351))

# PSP Spacecraft MET Epoch: 2010-01-01 00:00:00 UTC
PSP_MET_EPOCH = datetime.datetime(2010, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

def iso_to_met(iso_str):
    """Converts ISO 8601 string to MET seconds based on 2010-01-01 epoch."""
    if not iso_str:
        return None
    clean_str = iso_str.strip().replace('Z', '+00:00')
    dt = datetime.datetime.fromisoformat(clean_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int((dt - PSP_MET_EPOCH).total_seconds())

def met_to_iso(met):
    """Converts MET seconds back to UTC ISO string."""
    if met is None or met == float('inf') or met == float('-inf'):
        return "N/A"
    dt = PSP_MET_EPOCH + datetime.timedelta(seconds=met)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def read_binary_stream(path, is_gzip=False):
    if is_gzip or path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return f.read()
    with open(path, 'rb') as f:
        return f.read()

def crop_spc_packets(bytestr, target_apids=TARGET_SPC_APIDS, keep_swem_wrapper=False, start_met=None, end_met=None):
    output_bytes = bytearray()
    extracted_counts = {apid: 0 for apid in sorted(target_apids)}
    
    file_min_met = float('inf')
    file_max_met = float('-inf')

    # Match CCSDS primary header start
    pattern_bytes = [struct.pack('>H', 0x0800 | (apid & 0x07FF)) for apid in target_apids]
    pattern = b'(' + b'|'.join(re.escape(b) for b in pattern_bytes) + b')'

    matches = [m.start(0) for m in re.finditer(pattern, bytestr)]

    for start_idx in matches:
        if start_idx + 10 > len(bytestr):
            continue

        hdr_w1, seq_ctrl, pkt_len_field = struct.unpack('>HHH', bytestr[start_idx:start_idx+6])

        version = (hdr_w1 >> 13) & 0x07
        pkt_type = (hdr_w1 >> 12) & 0x01
        sec_hdr = (hdr_w1 >> 11) & 0x01
        apid = hdr_w1 & 0x07FF

        if version != 0 or pkt_type != 0 or sec_hdr != 1:
            continue

        total_pkt_size = pkt_len_field + 7
        if start_idx + total_pkt_size > len(bytestr):
            continue

        # Extract 4-byte CCSDS MET from Bytes 6-9
        met = struct.unpack('>I', bytestr[start_idx+6 : start_idx+10])[0]

        if met < file_min_met: file_min_met = met
        if met > file_max_met: file_max_met = met

        # Time range filtering
        if start_met is not None and met < start_met:
            continue
        if end_met is not None and met > end_met:
            continue

        if apid in target_apids:
            extracted_counts[apid] += 1

            if keep_swem_wrapper and start_idx >= 12:
                wrap_hdr = bytestr[start_idx - 12 : start_idx]
                wrap_apid = struct.unpack('>H', wrap_hdr[0:2])[0] & 0x07FF
                if wrap_apid in SWEM_WRAPPER_APIDS:
                    output_bytes.extend(wrap_hdr)

            output_bytes.extend(bytestr[start_idx : start_idx + total_pkt_size])

    return bytes(output_bytes), extracted_counts, file_min_met, file_max_met

def main():
    parser = argparse.ArgumentParser(description="Isolate SPC APIDs and filter by ISO Time / MET range.")
    parser.add_argument("-i", "--input", required=True, help="Input L0 binary file (.dat)")
    parser.add_argument("-o", "--output", required=True, help="Output cropped binary file")
    parser.add_argument("-gz", "--gzip", action="store_true", help="Set if input file is gzipped")
    parser.add_argument("-w", "--wrapper", action="store_true", help="Preserve SWEM wrapper headers if present")
    
    # ISO Time filter options
    parser.add_argument("-st", "--start-time", type=str, default=None, help="Start ISO UTC Time (e.g., 2021-10-16T12:00:00)")
    parser.add_argument("-et", "--end-time", type=str, default=None, help="End ISO UTC Time (e.g., 2021-10-16T14:30:00)")
    
    # MET filter options
    parser.add_argument("-sm", "--start-met", type=int, default=None, help="Start MET in seconds")
    parser.add_argument("-em", "--end-met", type=int, default=None, help="End MET in seconds")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"***ERROR*** Input file {args.input} does not exist.")
        sys.exit(1)

    start_met = iso_to_met(args.start_time) if args.start_time else args.start_met
    end_met = iso_to_met(args.end_time) if args.end_time else args.end_met

    print(f"Reading raw telemetry: {args.input}")
    raw_stream = read_binary_stream(args.input, is_gzip=args.gzip)

    cropped_stream, counts, min_met, max_met = crop_spc_packets(
        raw_stream, 
        keep_swem_wrapper=args.wrapper, 
        start_met=start_met, 
        end_met=end_met
    )

    with open(args.output, "wb") as f:
        f.write(cropped_stream)

    print("\n" + "=" * 65)
    print(" Telemetry Stream Info:")
    print(f"   File Start : MET {min_met:<10} | UTC: {met_to_iso(min_met)}")
    print(f"   File End   : MET {max_met:<10} | UTC: {met_to_iso(max_met)}")
    print("-" * 65)
    if start_met or end_met:
        print(" Active Time Filter:")
        print(f"   Filter Start: MET {str(start_met):<10} | UTC: {met_to_iso(start_met)}")
        print(f"   Filter End  : MET {str(end_met):<10} | UTC: {met_to_iso(end_met)}")
        print("=" * 65)

    print(f"\nCropped output saved to: {args.output}\n")
    print("Packet Extraction Summary:")
    for apid, count in counts.items():
        print(f"  APID {hex(apid).upper()}: {count:7d} packets")
    print("-" * 65)

if __name__ == "__main__":
    main()