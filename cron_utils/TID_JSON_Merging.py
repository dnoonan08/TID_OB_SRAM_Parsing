#!/usr/bin/python3.12

import glob
import json
import os
import datetime


print(f'Try merging {datetime.datetime.now()}')
skipped_file_list = open('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/cron_utils/skipped_files.txt','r').read().splitlines()

### function which merges split-out daq capture data (from January TID runs) back into a single json file
def merge_jsons(fname, new_file_name):
    data = json.load(open(fname))
    for t in data['tests']:
        if t['outcome']=='error' and t['setup']['outcome']=='error':
            continue
        if 'streamCom' in t['nodeid']:
            try:
                v = t['metadata']['voltage']
            except:
                return -1
            sc_fname = fname.replace('.json',f'_streamcompare_{round(v*100):03d}.json')
            try:
                sc_data = json.load(open(sc_fname))
            except:
                print(f'Unable to find SC data for {sc_fname}')
                print('Skipping this file')
                return -1
            for k in sc_data.keys():
                t['metadata'][k] = sc_data[k]
    if new_file_name==fname:
        print('ISSUE WITH NAMING NEW FILE')
        print('  try checking that new and old directory names are appropriate')
        print('  old name={old_dir_name}')
        print('  new name={new_file_name}')
        return
    json.dump(data,open(new_file_name,'w'))
    return new_file_name

#location that Grace's cronjob copies data from hexacontroller onto eos
unmergedEOSdir = '/eos/user/d/dnoonan/July_2025_TID_Data/data/'
#location that new merged files should be put
mergedEOSdir = '/eos/user/d/dnoonan/July_2025_TID_Data/merged_jsons/'

chip='COB-10Pct-1-1'

a = glob.glob(f'{unmergedEOSdir}/report_TID_chip_{chip}_ECOND_2025-*.json')
b = glob.glob(f'{unmergedEOSdir}/report_TID_chip_{chip}_ECOND_2025-*_streamcompare_*.json')

flist = list(set(a) - set(b) - set(skipped_file_list))
flist.sort()

merged_file_list = []

forceReprocess = False

for f in flist[:]:
    newFileName = f.replace(unmergedEOSdir,mergedEOSdir)
    if forceReprocess or not os.path.exists(newFileName):
        print(f)
        print(newFileName)
        try:
            x = merge_jsons(f,newFileName)
            if x==-1:
                skipped_file_list.append(f)
            else:
                print(x)
                merged_file_list.append(newFileName)
        except:
            print('Issue')
            skipped_file_list.append(f)
        print('-'*20)

with open('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/cron_utils/skipped_files.txt','w') as _file:
    for f in skipped_file_list:
        _file.write(f'{f}\n')

with open('/afs/cern.ch/user/d/dnoonan/TID_OB_SRAM_Parsing/cron_utils/last_merged_files.txt','w') as _file:
    for f in merged_file_list:
        _file.write(f'{f}\n')
