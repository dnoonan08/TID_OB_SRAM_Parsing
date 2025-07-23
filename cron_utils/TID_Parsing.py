#!/usr/bin/python3.11

import sys
sys.path.append('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/')
from parse_sc_data import getParsedTables
import glob
import os
import datetime

print(f'Try parsing files {datetime.datetime.now()}')

forceReprocess = False

#flist = glob.glob('/eos/user/d/dnoonan/July_2025_TID_Data/merged_jsons/report_TID_chip_COB-5Pct-1-3_ECOND_2025-07-22*.json')
flist = open('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/cron_utils/last_merged_files.txt','r').read().splitlines()

flist.sort()

for fname in flist[:]:
    print(f'Parsing {fname}')
    fname_totals = fname.replace('.json','_totals.csv').replace('merged_jsons','parsed_data')
    fname_packets = fname.replace('.json','_packets.csv').replace('merged_jsons','parsed_data')
    fname_bist = fname.replace('.json','_bist.csv').replace('merged_jsons','parsed_data')
    fname_settings = fname.replace('.json','_settings.csv').replace('merged_jsons','parsed_data')
    already_parsed = (os.path.exists(fname_totals) & 
                      os.path.exists(fname_packets) & 
                      os.path.exists(fname_bist) & 
                      os.path.exists(fname_settings)
                     )
    if forceReprocess or not already_parsed:
        try:
            d_tot, d_packets, d_bist, d_setting = getParsedTables(fname,forceReprocess=forceReprocess)
        except:
            print(f'issue with {fname}')
            continue
        if d_tot is None:
            print(f'issue with {fname}, no d_tot')
            continue
        if d_packets is None:
            print(f'issue with {fname}, no d_packets')
            continue
        if d_bist is None:
            print(f'issue with {fname}, no d_bist')
            continue
        if d_setting is None:
            print(f'issue with {fname}, no d_setting')
            continue
        d_tot.to_csv(fname_totals)
        d_packets.to_csv(fname_packets)
        d_bist.to_csv(fname_bist)
        d_setting.to_csv(fname_settings)
