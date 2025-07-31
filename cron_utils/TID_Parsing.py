#!/usr/bin/python3.11

import sys
sys.path.append('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/')
from parse_sc_data import getParsedTables
import glob
import os
import datetime
import pandas as pd

print(f'Try parsing files {datetime.datetime.now()}')

forceReprocess = False

_COB_='COB-15Pct-4-3'

#flist = glob.glob('/eos/user/d/dnoonan/July_2025_TID_Data/merged_jsons/report_TID_chip_COB-5Pct-1-3_ECOND_2025-07-22*.json')
flist = open('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/cron_utils/last_merged_files.txt','r').read().splitlines()

flist.sort()

newFilesParsed = False
lastFileName = None
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
        newFilesParsed = True
        lastFileName = fname

if newFilesParsed:
    print('TRYING TO MAKE PLOTS')
    from makePlots import makeSummaryPlots
    makeSummaryPlots(_COB_)

    fname = lastFileName
    print(f'Making voltage summary table for file: {fname}')
    fname_totals = fname.replace('.json','_totals.csv').replace('merged_jsons','parsed_data')
    fname_bist = fname.replace('.json','_bist.csv').replace('merged_jsons','parsed_data')
    d_tot = pd.read_csv(fname_totals,index_col='voltages')
    d_bist = pd.read_csv(fname_bist,index_col='voltages')
    cols = ['n_packets','word_count','error_count','isOBErrors','isSingleError_MultiBit','isSingleError_SingleBit','isBadPacketCRC','isBadPacketHeader','pass_PP_bist','pass_OB_bist']

    x = d_tot.loc[:,cols].copy(deep=True)
    x.columns = ['packets', 'words','errors','OB','Multi','Single','CRC','Hdr','PPbist','OBbist']
    v_pp_bist = float(d_bist[~d_bist.pass_PP_bist].index.max())+.01
    v_ob_bist = float(d_bist[~d_bist.pass_OB_bist].index.max())+.01
    v_i2c = float(d_bist[d_bist.PPbist_1<=-1].index.max())+.01

    with open(f'/eos/user/d/dnoonan/July_2025_TID_Data/plots/{_COB_}/last_run_{_COB_}.txt','w') as outputfile:
        outputfile.write(f'{fname}\n')
        outputfile.write('\n')
        outputfile.write(x.to_string())
        outputfile.write('\n')
        outputfile.write(f'I2C Voltage     = {v_i2c:.2f}\n')
        outputfile.write(f'PP Bist Voltage = {v_pp_bist:.2f}\n')
        outputfile.write(f'OB Bist Voltage = {v_ob_bist:.2f}\n')
