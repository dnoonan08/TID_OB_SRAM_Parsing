import numpy as np
import glob
import json
import pandas as pd
from datetime import datetime
import os

packet_word_index = np.loadtxt('data/packet_word_index.csv',delimiter=',',dtype=int)
full_capture = np.loadtxt('data/full_clean_orbit.csv',delimiter=',',dtype=str)
full_capture = np.vectorize(lambda x: int(x,16))(full_capture)
daq_stream = full_capture.flatten()
isIdle = (daq_stream==0x55555500)
nonMandatoryIdle = (isIdle & np.roll(isIdle,1))
sram_data = daq_stream[~nonMandatoryIdle]

def getBISTresults(fname):
    data = json.load(open(fname))

    voltages = np.array(data['tests'][-1]['metadata']['voltages'])
    try:
        timestamps = np.array(data['tests'][-1]['metadata']['timestamps'])
    except:
        timestamps = np.datetime64('2025-01-01')

    try:
        b = np.array(data['tests'][-1]['metadata']['bist_results'])
        b_pp = b[:,4:]
        b_ob = b[:,:4]
    except:
        b_pp = np.array(data['tests'][-1]['metadata']['ppResults'])
        b_ob = np.array(data['tests'][-1]['metadata']['obResults'])

    if (b_pp==4095).any():
        _pp_min = voltages[(b_pp==4095).all(axis=1)].min()
    else:
        _pp_min = 1.5
    if (b_ob==4095).any():
        _ob_min = voltages[(b_ob==4095).all(axis=1)].min()
    else:
        _ob_min = 1.5
    print('PP BIST min voltage', _pp_min)
    print('OB BIST min voltage', _ob_min)

    d_bist = pd.DataFrame({"voltages":voltages,
                           "timestamps":timestamps,
                           "PPbist_1":b_pp[:,0],
                           "PPbist_2":b_pp[:,1],
                           "PPbist_3":b_pp[:,2],
                           "PPbist_4":b_pp[:,3],
                           "OBbist_1":b_ob[:,0],
                           "OBbist_2":b_ob[:,1],
                           "OBbist_3":b_ob[:,2],
                           "OBbist_4":b_ob[:,3],
                           "file":fname,
                          })
    d_bist.voltages = (d_bist.voltages.values).round(3)
    d_bist.set_index('voltages',inplace=True)

    d_bist['pass_PP_bist'] = (d_bist[[f'PPbist_{i}' for i in [1,2,3,4]]]==4095).all(axis=1)
    d_bist['pass_OB_bist'] = (d_bist[[f'OBbist_{i}' for i in [1,2,3,4]]]==4095).all(axis=1)
    d_bist['PP_bist_01'] = d_bist[f'PPbist_1'].apply(lambda x: f'{x:012b}')
    d_bist['OB_bist_01'] = d_bist[f'OBbist_1'].apply(lambda x: f'{x:012b}')

    return d_bist

def getParsedTables(fname, forceReprocess=False, debug_print=False):
    fname_totals = fname.replace('.json','_totals.csv').replace('merged_jsons','parsed')
    fname_packets = fname.replace('.json','_packets.csv').replace('merged_jsons','parsed')
    fname_bist = fname.replace('.json','_bist.csv').replace('merged_jsons','parsed')
    already_parsed = (os.path.exists(fname_totals) &
                      os.path.exists(fname_packets) &
                      os.path.exists(fname_bist)
                     ) & ~forceReprocess
    if already_parsed:
        d_tot = pd.read_csv(fname_totals,index_col='voltages')
        df = pd.read_csv(fname_packets,index_col=0)
        d_bist = pd.read_csv(fname_bist)
    else:
        print(fname)
        d_tot, df = checkErr(fname, debug_print=debug_print)

        d_bist = getBISTresults(fname)

        d_tot = d_tot.merge(d_bist[['pass_PP_bist','pass_OB_bist','PP_bist_01','OB_bist_01']],left_index=True,right_index=True,how='left').sort_index()
        d_tot.timestamp = pd.to_datetime(d_tot.timestamp)

        #if we pass bist at 1.19, fill rest as pass
        if d_bist.loc[1.19].pass_PP_bist:
            d_tot.PP_bist_01 = d_tot.PP_bist_01.fillna('111111111111').astype(object)
            d_tot.pass_PP_bist = d_tot.pass_PP_bist.astype(bool).fillna(True)
        if d_bist.loc[1.19].pass_OB_bist:
            d_tot.OB_bist_01 = d_tot.OB_bist_01.fillna('111111111111').astype(object)
            d_tot.pass_OB_bist = d_tot.pass_OB_bist.astype(bool).fillna(True)

        df['file'] = fname
        d_tot['file'] = fname
        d_tot.sort_index(inplace=True)
    return d_tot, df, d_bist


def checkErr(fname,i=0, debug_print=False):
    df=parse_sram_errors_per_packet(fname,sram_data, debug_print=debug_print)
    sum_cols = ['isOBErrors',
                'isSingleError',
                'isBadPacketCRC',
                'isBadPacketHeader',
                'isSingleError_SingleBit',
                'isSingleError_MultiBit',
                'isSingleError_SingleBit_SpecialPackets',
                'isSingleError_MultiBit_SpecialPackets',
                'isOBErrors_SpecialPackets',
               ]

    ###fix that allows dropping the last lc buffer readout from the sums, if there are more than 1
    # y=df[1].set_index('voltages')[['n_captured_bx','n_captures','n_packets','word_count','error_count','timestamp','current','temperature']]
    y=df[1][['voltages','n_captures','n_packets','n_captured_bx']].groupby('voltages').apply(lambda x: x.iloc[:-1] if len(x)>1 else x, include_groups=False).groupby('voltages').sum()
    y[['word_count','error_count','timestamp','current','temperature']]=df[1].groupby('voltages')[['word_count','error_count','timestamp','current','temperature']].first()

    if not df[0] is None:
        df[0]['isSpecialPacket'] = df[0].packet_number.isin([3,4,11,27,32,33,49])
        df[0]['isSingleError_SingleBit'] = df[0].isSingleError & (df[0].asic_emu_bit_diff_0==1)
        df[0]['isSingleError_MultiBit'] = df[0].isSingleError & (df[0].asic_emu_bit_diff_0>1)

        df[0]['isSingleError_SingleBit_SpecialPackets'] = df[0].isSingleError_SingleBit & df[0].isSpecialPacket
        df[0]['isSingleError_MultiBit_SpecialPackets'] = df[0].isSingleError_MultiBit & df[0].isSpecialPacket
        df[0]['isOBErrors_SpecialPackets'] = df[0].isOBErrors & df[0].isSpecialPacket

        ###fix that allows dropping the last lc buffer readout from the sums, if there are more than 1
        # x=df[0].groupby('voltages').sum()[sum_cols]
        x=df[0].groupby(['voltages','lc_number']).sum()[sum_cols].groupby(['voltages']).apply(lambda x: x.iloc[:-1] if len(x)>1 else x, include_groups=False).groupby('voltages').sum()
        z=y.merge(x,left_index=True,right_index=True,how='left').fillna(0)
    else:
        z = y
        z[sum_cols] = 0

    z['error_rate'] = (z.isOBErrors + z.isSingleError)/(z.n_captured_bx/3564*67)
    z['run'] = i
    z = z.astype({c: int for c in sum_cols})
    return z, df[0]

### function which merges split-out daq capture data (from January TID runs) back into a single json file
def merge_jsons(fname, old_dir_name='json_files', new_dir_name='merged_jsons'):
    data = json.load(open(fname))
    for t in data['tests']:
        if 'streamCom' in t['nodeid']:
            if t['outcome']=='error':
                continue
            v = t['metadata']['voltage']
            sc_fname = fname.replace('.json',f'_streamcompare_{round(v*100):03d}.json')
            sc_data = json.load(open(sc_fname))
            for k in sc_data.keys():
                t['metadata'][k] = sc_data[k]
    newName = fname.replace(old_dir_name,new_dir_name)
    if newName==fname:
        print('ISSUE WITH NAMING NEW FILE')
        print('  try checking that new and old directory names are appropriate')
        print('  old dir name={old_dir_name}')
        print('  new dir name={new_dir_name}')
        return
    json.dump(data,open(newName,'w'))
    return newName

def print_daq_capture(metadata, j, N=4096):
    print(metadata['voltage'])
    print(metadata['Timestamp'][0])
    for i in range(len(metadata['DAQ_asic'][j])):
        if i>N: break
        asic = np.array(metadata['DAQ_asic'][j][i][::-1])
        emu = np.array(metadata['DAQ_emu'][j][i][::-1])
        count= np.array(metadata['DAQ_counter'][j][i])
        count = count[0] + (count[1]<<32)
        agree = ''.join(['-' if i else 'X' for i in asic==emu])#"" if (asic==emu).all() else " <<<<"
        print(f"{i:04d} " + ' '.join([f'{x:08x}' for x in asic]) + " | " + ' '.join([f'{x:08x}' for x in emu]) + f" | {count} |" + agree)

def print_daq_capture_OB_errors(metadata, j, err_idx):
    l = [err_idx]
    for i in range(1,6):
        l.append(err_idx + i)
        l.append(err_idx - i)
    l = np.array(l).flatten()
    l = l[l>=0]
    l = l[l<len(metadata['DAQ_asic'][j])]
    l = np.unique(l)
    for i in l:
#         if i>N: break
        asic = np.array(metadata['DAQ_asic'][j][i][::-1])
        emu = np.array(metadata['DAQ_emu'][j][i][::-1])
        count= metadata['DAQ_counter'][j][i]
        count = count[0] + (count[1]<<32)
        agree = ''.join(['-' if i else 'X' for i in asic==emu])#"" if (asic==emu).all() else " <<<<"
        print(f"{i:04d} " + ' '.join([f'{x:08x}' for x in asic]) + " | " + ' '.join([f'{x:08x}' for x in emu]) + f" | {count} |" + agree)


def print_packet(d_asic,d_emu,d_count,d_idx, print_diff = False):
    da = d_asic.reshape(-1,6)
    de = d_emu.reshape(-1,6)
    dc = d_count
    di = d_idx
    agree = np.where(da==de,'-','X')

    for i in range(len(da)):
        _diff = ""
        if print_diff:
            if (agree[i]=='X').sum()==1:
                a=da[i][da[i]!=de[i]][0]
                b=de[i][da[i]!=de[i]][0]
                _10 = f'{(a^b)&a:032b}'.replace('0','-').replace('1','0')
                _01 = f'{(a^b)&b:032b}'.replace('0','-')
                _diff = ' '+''.join(['0' if _10[i]=='0' else '1' if _01[i]=='1' else '-' for i in range(32)])

        print(f"{di[i]:04d} " + ' '.join([f'{x:08x}' for x in da[i]]) + " | " + ' '.join([f'{x:08x}' for x in de[i]]) + f" | {dc[i]} | " + ''.join(agree[i])+_diff)

def parse_packet_errors(d_asic, d_emu, d_count, d_idx):
    mismatch_idx = np.argwhere(~(d_asic==d_emu)).flatten()
    if len(mismatch_idx)==0:
        print('Packet Has Zero Errors', d_idx[0])
        return -1,0,0,0,0,0,0,0,0
    first_mismatch_packet_word_index = packet_word_index[np.argwhere((full_capture[:,2:]==d_emu[20:24]).all(axis=1))[0,0]][mismatch_idx[0]%6]

    #we have a mismatch that has the 9-bit packet header the previous word is an IDLE
    isBadPacketHeader = (((d_emu[mismatch_idx]>>23)==0x1e6) & ((d_emu[mismatch_idx-1]>>8)==0x555555)).any()

    #append -1's to the end, such that we always have at least a few elements (and if not, we get -1)
    error_word_asic = d_asic[mismatch_idx].tolist()+[-1]*3
    error_word_emu = d_emu[mismatch_idx].tolist()+[-1]*3
    error_word_counter = d_count[(mismatch_idx/6).astype(int)].tolist()+[-1]*3
    #we have a mismatch that is NOT and IDLE but the next word is an IDLE
    try:
        isBadPacketCRC = (((d_emu[mismatch_idx]>>8)!=0x555555) & ((d_emu[mismatch_idx+1]>>8)==0x555555)).any()
    except:
        isBadPacketCRC = False

    if len(mismatch_idx)>1:
        is_ob_err = (((mismatch_idx - np.roll(mismatch_idx,1))%12==0) &
                     ((d_emu.flatten()[mismatch_idx]>>8)!=0x555555)
                    )
        isOBErrors = is_ob_err.all()
        n_ob_errors = is_ob_err.sum()
    else:
        isOBErrors = False
        n_ob_errors = 0


    n_tot_errors = len(mismatch_idx)
    return isOBErrors, isBadPacketHeader, isBadPacketCRC, n_tot_errors, n_ob_errors,first_mismatch_packet_word_index, error_word_asic, error_word_emu, error_word_counter

hex32=np.vectorize(lambda x: f'{x:08x}')
bin32=np.vectorize(lambda x: f'{x:032b}')

#debugging tool, for printing a packet given a single dataframe row as returned by parse_sram_errors_per_packet function
def print_data_from_parsed_dataframe(row, print_diff=False):
    data = json.load(open(row.file_names))
    t = data['tests'][row.test_number]['metadata']
    i_start = int(row.packet_start_idx)
    i_stop = int(row.packet_stop_idx)+1
    daq_asic = np.array(t['DAQ_asic'][row.lc_number][i_start:i_stop])[:,::-1]
    daq_emu = np.array(t['DAQ_emu'][row.lc_number][i_start:i_stop])[:,::-1]
    daq_count = np.array(t['DAQ_counter'][row.lc_number][i_start:i_stop])
    daq_count = (daq_count[:,1]<<32) + daq_count[:,0]
    daq_idx = np.arange(i_start, i_stop)
    print_packet(daq_asic, daq_emu, daq_count, daq_idx, print_diff)
    return daq_asic, daq_emu

def parse_sram_errors_per_packet(file_name, sram_data, nl1a=67, return_lists = False, debug_print=False, reset_sc_counts=False):

    if debug_print:
        print('DEBUG')
    #initialize lists
    isOBErrors, isBadPacketCRC, isBadPacketHeader = [],[],[],
    n_tot_errors, n_ob_errors = [],[]
    packet_number, packet_word, packet_start_idx, packet_stop_idx, packet_start_counter = [],[],[],[],[]
    voltages, file_names, test_number, test_name, lc_number = [],[],[],[],[]
    n_erx, n_etx = [],[]
    error_word_asic_0, error_word_asic_1, error_word_asic_2, error_word_asic_3 = [],[],[],[]
    error_word_emu_0, error_word_emu_1, error_word_emu_2, error_word_emu_3 = [],[],[],[]
    error_word_counter_0, error_word_counter_1, error_word_counter_2, error_word_counter_3 = [],[],[],[]

    total_captures, total_packets, total_fifo_occupancy = [], [], []
    total_timestamp = []
    total_word_count, total_error_count = [],[]
    total_mean_temperature, total_std_temperature, total_mean_current, total_std_current = [],[],[],[]
    total_capture_length_bx = []
    total_voltages, total_file_names, total_test_number, total_test_name, total_lc_number = [],[],[],[],[]
    total_n_erx, total_n_etx = [],[]

    settings_voltage, settings_file_names, settings_test_number, settings_test_name = [],[],[],[]
    settings_capbank = []
    settings_phase_select = []
    settings_delay_setting = []
    settings_delay_width = []

    #load file
    try:
        data = json.load(open(file_name))
    except:
        print(f'Issue with loading json file {file_name}')
        return None, None, None
    #loop through all tests
    for t_idx, _t in enumerate(data['tests']):
        tname = _t['nodeid']

        #skip tests that aren't stream compar loops
        if not 'test_streamCompareLoop' in tname:
            continue


        if debug_print: print(t_idx, tname)
        if not _t['setup']['outcome']=='passed':
            print('Setup erred out')
            continue
#         if _t['outcome']=='error':
#             if debug_print: print('Test erred out')
#             continue

        if not 'metadata' in _t:
            if debug_print: print('No Metadata')
            continue

        t = _t['metadata']
        daq_nl1a    = np.array(t['DAQ_nL1A'])
        voltage = t['voltage']

        _n_erx = 12
        _n_etx = 6
        if 'active_erx' in t:
            _n_erx = int(np.bitwise_count(t['active_erx']))
        if 'active_etx' in t:
            _n_etx = int(np.bitwise_count(t['active_etx']))

        settings_voltage.append(voltage)
        settings_file_names.append(file_name)
        settings_test_name.append(tname)
        settings_test_number.append(t_idx)
        settings_capbank.append(t['automatic_capbank_setting'])
        settings_phase_select.append(t['automatic_phase_select_settings'])
        settings_delay_setting.append(t['automatic_delayscan_settings'])
        settings_delay_width.append(t['automatic_delayscan_width'])

        # print(voltage)
        # print(t['Timestamp'][0])
        # print('Correct Voltages:', (abs(np.array(t['Voltage']) - t['voltage'])<0.015).all())

        #get captures where we took data with 67 l1a
        capt_idx67=np.argwhere(daq_nl1a==nl1a).flatten()

        wc = np.array(t['word_err_count'])
        hasl1a= np.array(t['HasL1A'])
#         if (hasl1a==0).any():

        n_sc_bx_0  = wc[hasl1a==0][:,1].astype(int)
        n_sc_bx_7  = wc[hasl1a==7][:,1].astype(int)
        n_sc_bx_67 = wc[hasl1a==67][:,1].astype(int)

        err_cnt_0  = wc[hasl1a==0][:,2].astype(int)
        err_cnt_7  = wc[hasl1a==7][:,2].astype(int)
        err_cnt_67 = wc[hasl1a==67][:,2].astype(int)

        n_sc_bx = 0
        err_cnt = 0
        if nl1a==7:
            if reset_sc_counts and len(n_sc_bx_0)>0:
                n_sc_bx = n_sc_bx_7[-1] - n_sc_bx_0[-1]
                err_cnt = err_cnt_7[-1] - err_cnt_0[-1]
            else:
                n_sc_bx = n_sc_bx_7[-1]
                err_cnt_bx = err_cnt_7[-1]
        if nl1a==67:
            if reset_sc_counts and len(n_sc_bx_7)>0:
                n_sc_bx = n_sc_bx_67[-1] - n_sc_bx_7[-1]
                err_cnt = err_cnt_67[-1] - err_cnt_7[-1]
            else:
                n_sc_bx = n_sc_bx_67[-1]
                err_cnt = err_cnt_67[-1]

        temperature = np.array(t['Temperature'])
        current = np.array(t['Current'])
        hasL1A    = np.array(t['HasL1A'])
        mean_current, std_current = current[hasL1A==nl1a].mean(), current[hasL1A==nl1a].std()
        mean_temperature, std_temperature = temperature[hasL1A==nl1a].mean(), temperature[hasL1A==nl1a].std()

        for c in capt_idx67:
#             print(c)
            daq_timestamp = t['Timestamp-DAQ'][c]
            try:
                daq_asic    = np.array(t['DAQ_asic'][c])
            except:
                print('Corrupted ASIC data capture, skipping')
                continue
            #skip if there are no errors
            if len(daq_asic)==0:
                total_fifo_occupancy.append(0)
                total_captures.append(0)
                total_packets.append(0)
                total_timestamp.append(daq_timestamp)
                total_mean_current.append(mean_current)
                total_mean_temperature.append(mean_temperature)
                total_capture_length_bx.append(n_sc_bx)
                total_word_count.append(n_sc_bx)
                total_error_count.append(err_cnt)
                total_voltages.append(voltage)
                total_file_names.append(file_name)
                total_test_number.append(t_idx)
                total_test_name.append(tname)
                total_lc_number.append(c)
                total_n_erx.append(_n_erx)
                total_n_etx.append(_n_etx)

                continue

            daq_asic = daq_asic[:,::-1]
            try:
                daq_emu     = np.array(t['DAQ_emu'][c])[:,::-1]
            except:
                print('Corrupted emulator data capture, skipping')
                continue
            if len(daq_asic)!=len(daq_emu):
                if debug_print: print(f'mismatch and asic and emulator capture lengths, skipping {tname} {c}')
                continue

            # if (daq_emu!=daq_asic).sum()>16000:
            #     total_fifo_occupancy.append(-1)
            #     total_captures.append(-1)
            #     total_packets.append(-1)
            #     total_timestamp.append(daq_timestamp)
            #     total_mean_current.append(mean_current)
            #     total_mean_temperature.append(mean_temperature)
            #     total_capture_length_bx.append(n_sc_bx)
            #     total_word_count.append(n_sc_bx)
            #     total_error_count.append(err_cnt)
            #     total_voltages.append(voltage)
            #     total_file_names.append(file_name)
            #     total_test_number.append(t_idx)
            #     total_test_name.append(tname)
            #     total_lc_number.append(c)
            #     total_n_erx.append(_n_erx)
            #     total_n_etx.append(_n_etx)

            #     print(f'Too many errors (V={voltage:.02f}), skipping')
            #     continue
            #looking for corrupted captures, identifiable in TID captures as spot where idles appear in column N but not N+1 of emulator
            isIdles = (daq_emu>>8)==0x555555
            corruptedData = False
            for i_col in range(5):
                if (isIdles[:,i_col] & ~isIdles[:,i_col+1]).any():
                    total_fifo_occupancy.append(-1)
                    total_captures.append(-1)
                    total_packets.append(-1)
                    total_timestamp.append(daq_timestamp)
                    total_mean_current.append(mean_current)
                    total_mean_temperature.append(mean_temperature)
                    total_capture_length_bx.append(n_sc_bx)
                    total_word_count.append(n_sc_bx)
                    total_error_count.append(err_cnt)
                    total_voltages.append(voltage)
                    total_file_names.append(file_name)
                    total_test_number.append(t_idx)
                    total_test_name.append(tname)
                    total_lc_number.append(c)
                    total_n_erx.append(_n_erx)
                    total_n_etx.append(_n_etx)
                    print(f'Corrupted data capture (V={voltage:.02f}), skipping')
                    corruptedData = True
                    break
            if corruptedData:
                continue

            try:
                daq_counter = np.array(t['DAQ_counter'][c])
                daq_counter = daq_counter[:,0] + (daq_counter[:,1]<<32)
            except:
                total_fifo_occupancy.append(-1)
                total_captures.append(-1)
                total_packets.append(-1)
                total_timestamp.append(daq_timestamp)
                total_mean_current.append(mean_current)
                total_mean_temperature.append(mean_temperature)
                total_capture_length_bx.append(n_sc_bx)
                total_word_count.append(n_sc_bx)
                total_error_count.append(err_cnt)
                total_voltages.append(voltage)
                total_file_names.append(file_name)
                total_test_number.append(t_idx)
                total_test_name.append(tname)
                total_lc_number.append(c)
                total_n_erx.append(_n_erx)
                total_n_etx.append(_n_etx)
                print('Corrupted counter data capture, skipping')
                continue

            if len(daq_counter)>len(daq_asic):
                daq_counter = daq_counter[:len(daq_asic)]

            #find boundaries between captures based on fact that counter isn't continuous
            new_capture = (daq_counter-np.roll(daq_counter,1))!=1
            n_captures = new_capture.sum()

            new_capture_arg = np.argwhere(new_capture).flatten().tolist() + [None]

            ###Update the capture arguments, to split single captures if there are more than one packet in it
            updated_capture_arg = []
            for i, capt_idx in enumerate(new_capture_arg[:-1]):

                start_idx = new_capture_arg[i]

                capture_emu = daq_emu[new_capture_arg[i]:new_capture_arg[i+1]]
                capture_asic = daq_asic[new_capture_arg[i]:new_capture_arg[i+1]]

                idx_has_idles_asic = ((capture_asic>>8)==0x555555).all(axis=1)
                idx_has_idles_emu  = ((capture_emu>>8)==0x555555).all(axis=1)
                idx_has_idles = idx_has_idles_asic & idx_has_idles_emu

                packet_split_idx = (np.argwhere((idx_has_idles & np.roll(idx_has_idles,-1) & np.roll(idx_has_idles,-2) & ~np.roll(idx_has_idles,-3))[:-4]).flatten() + start_idx).tolist()
                if not start_idx in packet_split_idx:
                    packet_split_idx.append(start_idx)
                updated_capture_arg += packet_split_idx
            updated_capture_arg.sort()
            new_capture_arg = np.array(updated_capture_arg + [None])

            #make lists of all packets
            packets_asic = []
            packets_emu = []
            packets_counter =[]
            packets_idx = []

            idx = np.arange(len(daq_emu))

            count_errors = 0
            #loop through all captures, grouping into packets
            for i, capt_idx in enumerate(new_capture_arg[:-1]):
                #if first capture, or previous capture was started more than 80 BX before, start new packet
                if i==0 or (daq_counter[new_capture_arg[i]] - daq_counter[new_capture_arg[i-1]])>80:
                    packets_asic.append(daq_asic[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                    packets_emu.append(daq_emu[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                    packets_counter.append(daq_counter[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                    packets_idx.append(idx[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                else:
                    #find packet number for this capture and previous capture
                    this_capt_daq_emu_line = daq_emu[new_capture_arg[i]][2:]
                    last_capt_daq_emu_line = daq_emu[new_capture_arg[i-1]+3][2:]

                    #if we can't find emulator data in full capture, indicates a problem
                    try:
                        this_capt_idx = np.argwhere(full_capture[:,2:]==this_capt_daq_emu_line)[0][0]
                        last_capt_idx = np.argwhere(full_capture[:,2:]==last_capt_daq_emu_line)[0][0]
                    except:
                        count_errors += 1
                        continue
                    this_packet_num = int(packet_word_index[this_capt_idx][0]/1000)
                    last_packet_num = int(packet_word_index[last_capt_idx][0]/1000)

                    #if this capture has same packet number as previous capture, add capture to previous packet
                    if this_packet_num==last_packet_num:
                        packets_asic[-1] += daq_asic[capt_idx:new_capture_arg[i+1]].flatten().tolist()
                        packets_emu[-1] += daq_emu[capt_idx:new_capture_arg[i+1]].flatten().tolist()
                        packets_counter[-1] += daq_counter[capt_idx:new_capture_arg[i+1]].flatten().tolist()
                        packets_idx[-1] += idx[capt_idx:new_capture_arg[i+1]].flatten().tolist()
                    else:
                        packets_asic.append(daq_asic[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                        packets_emu.append(daq_emu[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                        packets_counter.append(daq_counter[capt_idx:new_capture_arg[i+1]].flatten().tolist())
                        packets_idx.append(idx[capt_idx:new_capture_arg[i+1]].flatten().tolist())

            if (count_errors>0):
                print('Problems splitting packets, skipping')
                continue

            total_fifo_occupancy.append(len(daq_asic))
            total_captures.append(n_captures)
            total_packets.append(len(packets_asic))
            total_timestamp.append(daq_timestamp)
            total_mean_current.append(mean_current)
            total_mean_temperature.append(mean_temperature)
            if len(daq_asic)>=4095:
                total_capture_length_bx.append(daq_counter[-1] - daq_counter[0] + 1)
            else:
                total_capture_length_bx.append(n_sc_bx)
            total_word_count.append(n_sc_bx)
            total_error_count.append(err_cnt)
            total_voltages.append(voltage)
            total_file_names.append(file_name)
            total_test_number.append(t_idx)
            total_test_name.append(tname)
            total_lc_number.append(c)
            total_n_erx.append(_n_erx)
            total_n_etx.append(_n_etx)

            #loop through packets
            for i in range(len(packets_asic)):
                #load arrays from this packet
                d_asic = np.array(packets_asic[i])
                d_emu = np.array(packets_emu[i])
                d_count = np.array(packets_counter[i])
                d_idx = np.array(packets_idx[i])

#                 #parse errors from this packet
#                 try:
                _isOBErrors, _isBadPacketHeader, _isBadPacketCRC, _n_tot_errors, _n_ob_errors, _first_mismatch_packet_word_index, _error_word_asic, _error_word_emu, _error_word_counter = parse_packet_errors(d_asic, d_emu, d_count, d_idx)
#                 except:
#                     count_errors += 1
#                     continue
                if _isOBErrors==-1: continue
                #append outputs to arrays for all packets
                isOBErrors.append(_isOBErrors)
                isBadPacketHeader.append(_isBadPacketHeader)
                isBadPacketCRC.append(_isBadPacketCRC)
                n_tot_errors.append(_n_tot_errors)
                n_ob_errors.append(_n_ob_errors)
                packet_number.append(int(_first_mismatch_packet_word_index/1000)-1)
                packet_word.append(_first_mismatch_packet_word_index%1000)
                packet_start_idx.append(d_idx[0])
                packet_stop_idx.append(d_idx[-1])
                packet_start_counter.append(d_count[0])
                voltages.append(voltage)
                file_names.append(file_name)
                test_number.append(t_idx)
                test_name.append(tname)
                lc_number.append(c)

                error_word_asic_0.append(_error_word_asic[0])
                error_word_asic_1.append(_error_word_asic[1])
                error_word_asic_2.append(_error_word_asic[2])
                error_word_asic_3.append(_error_word_asic[3])
                error_word_emu_0.append(_error_word_emu[0])
                error_word_emu_1.append(_error_word_emu[1])
                error_word_emu_2.append(_error_word_emu[2])
                error_word_emu_3.append(_error_word_emu[3])
                error_word_counter_0.append(_error_word_counter[0])
                error_word_counter_1.append(_error_word_counter[1])
                error_word_counter_2.append(_error_word_counter[2])
                error_word_counter_3.append(_error_word_counter[3])

                n_erx.append(_n_erx)
                n_etx.append(_n_etx)


    #return the lists, or create pandas dataframe and return that
    if return_lists:
        return isOBErrors, isBadPacketCRC, isBadPacketHeader, n_tot_errors, n_ob_errors, packet_number, packet_word, packet_start_idx, packet_stop_idx, packet_start_counter, voltages, file_names, test_name, test_number, lc_number, n_erx, n_etx, error_word_asic_0, error_word_asic_1, error_word_asic_2, error_word_asic_3, error_word_emu_0, error_word_emu_1, error_word_emu_2, error_word_emu_3
    else:
        df_tot = pd.DataFrame({
            'voltages':total_voltages,
            'n_captured_bx':total_capture_length_bx,
            'n_captures':total_captures,
            'n_packets':total_packets,
            'file_names':total_file_names,
            'test_number':total_test_number,
            'test_name':total_test_name,
            'lc_number':total_lc_number,
            'word_count':total_word_count,
            'error_count':total_error_count,
            'timestamp':total_timestamp,
            'temperature':total_mean_temperature,
            'current':total_mean_current,
            'n_erx':total_n_erx,
            'n_etx':total_n_etx,
        })

        phase_selects = np.array(settings_phase_select)
        delay_setting = np.array(settings_delay_setting)
        delay_width   = np.array(settings_delay_width)
        df_settings = pd.DataFrame({
            'voltages':settings_voltage,
            'file_names':settings_file_names,
            'test_number':settings_test_number,
            'test_name':settings_test_name,
            'capbank':settings_capbank,
            'phase_select_00': phase_selects[:,0],
            'phase_select_01': phase_selects[:,1],
            'phase_select_02': phase_selects[:,2],
            'phase_select_03': phase_selects[:,3],
            'phase_select_04': phase_selects[:,4],
            'phase_select_05': phase_selects[:,5],
            'phase_select_06': phase_selects[:,6],
            'phase_select_07': phase_selects[:,7],
            'phase_select_08': phase_selects[:,8],
            'phase_select_09': phase_selects[:,9],
            'phase_select_10': phase_selects[:,10],
            'phase_select_11': phase_selects[:,11],
            'delay_setting_00': delay_setting[:,0],
            'delay_setting_01': delay_setting[:,1],
            'delay_setting_02': delay_setting[:,2],
            'delay_setting_03': delay_setting[:,3],
            'delay_setting_04': delay_setting[:,4],
            'delay_setting_05': delay_setting[:,5],
            'delay_width_00': delay_width[:,0],
            'delay_width_01': delay_width[:,1],
            'delay_width_02': delay_width[:,2],
            'delay_width_03': delay_width[:,3],
            'delay_width_04': delay_width[:,4],
            'delay_width_05': delay_width[:,5],
        })

        df = pd.DataFrame({
            'isOBErrors':isOBErrors,
            'isBadPacketCRC':isBadPacketCRC,
            'isBadPacketHeader':isBadPacketHeader,
            'n_tot_errors':n_tot_errors,
            'n_ob_errors':n_ob_errors,
            'packet_number':packet_number,
            'packet_word':packet_word,
            'packet_start_idx':packet_start_idx,
            'packet_stop_idx':packet_stop_idx,
            'packet_start_counter':packet_start_counter,
            'voltages':voltages,
            'file_names':file_names,
            'test_name':test_name,
            'test_number':test_number,
            'lc_number':lc_number,
            'n_erx':n_erx,
            'n_etx':n_etx,
            'error_word_asic_0':error_word_asic_0,
            'error_word_asic_1':error_word_asic_1,
            'error_word_asic_2':error_word_asic_2,
            'error_word_asic_3':error_word_asic_3,
            'error_word_emu_0':error_word_emu_0,
            'error_word_emu_1':error_word_emu_1,
            'error_word_emu_2':error_word_emu_2,
            'error_word_emu_3':error_word_emu_3,
            'error_word_counter_0':error_word_counter_0,
            'error_word_counter_1':error_word_counter_1,
            'error_word_counter_2':error_word_counter_2,
            'error_word_counter_3':error_word_counter_3,
        })
        if len(df)==0:
            return None, df_tot, df_settings

        df['isSingleError'] = ~df.isOBErrors & ~df.isBadPacketCRC & ~df.isBadPacketHeader & (df.n_tot_errors==1)

        df['asic_emu_bit_diff_0']= np.bitwise_count(df.error_word_asic_0 ^ df.error_word_emu_0)
        df['asic_emu_bit_diff_1']= np.bitwise_count(df.error_word_asic_1 ^ df.error_word_emu_1)
        df['asic_emu_bit_diff_2']= np.bitwise_count(df.error_word_asic_2 ^ df.error_word_emu_2)
        df['asic_emu_bit_diff_3']= np.bitwise_count(df.error_word_asic_3 ^ df.error_word_emu_3)


        N=len(df)
        sram_matrix=np.array([np.concatenate([sram_data,sram_data,sram_data])]*N).reshape(N,-1)
        ####only look at sram indices that would have come from same sram as the matching word from the emulator
        #make 2d array, with indices that are within +/- 256 sram rows
        max_back_search = 512
        b=np.array([(np.arange(max_back_search + 256+1)-max_back_search)*12]*N).reshape(N,-1)


        for i_error in range(4):

            #find where in the sram_data the
            match_idx = []
            for x in df[f'error_word_emu_{i_error}']:
                match = sram_data==x
                if match.sum()==1:
                    match_idx.append(np.argwhere(match)[0][0])
                else:
                    match_idx.append(-1)
            match_idx = np.array(match_idx)
            df[f'sram_idx_{i_error}'] = match_idx

            c=(b.T+match_idx+len(sram_data)).T
            #turn this list of indices into a boolean mask
            c_mask = np.zeros_like(sram_matrix,dtype=bool)
            for i in range(len(c_mask)):
                c_mask[i,c[i]]=True

            #apply boolean mask to sram data, and reshape to original shape
            data_in_same_sram=sram_matrix[c_mask].reshape(c.shape)

            xor_bit_count = np.bitwise_count(data_in_same_sram.T ^ df[f'error_word_asic_{i_error}'].values)
            closest_sram_offset = np.argmin(xor_bit_count,axis=0)-max_back_search
            #how many bits are different
            n_bits_diff = np.min(xor_bit_count,axis=0)
            #how many sram indices have the
            n_min_bitmatches = (xor_bit_count==n_bits_diff).sum(axis=0)

            df[f'closest_match_offset_{i_error}'] = closest_sram_offset
            df[f'closest_match_n_bits_diff_{i_error}'] = n_bits_diff
            df[f'n_closest_matches_{i_error}'] = n_min_bitmatches


        column_order = ['voltages','isOBErrors', 'isSingleError', 'isBadPacketCRC', 'isBadPacketHeader',
                        'n_tot_errors','n_ob_errors', 'packet_number', 'packet_word', 'packet_start_idx', 'packet_stop_idx', 'packet_start_counter',
                        'file_names', 'test_name', 'test_number', 'lc_number', 'n_erx', 'n_etx',
                        'error_word_asic_0', 'error_word_asic_1',  'error_word_asic_2',  'error_word_asic_3',
                        'error_word_emu_0', 'error_word_emu_1', 'error_word_emu_2', 'error_word_emu_3',
                        'error_word_counter_0', 'error_word_counter_1', 'error_word_counter_2', 'error_word_counter_3',
                        'asic_emu_bit_diff_0', 'asic_emu_bit_diff_1', 'asic_emu_bit_diff_2', 'asic_emu_bit_diff_3',
                        'sram_idx_0', 'sram_idx_1', 'sram_idx_2', 'sram_idx_3',
                        'closest_match_offset_0', 'closest_match_offset_1', 'closest_match_offset_2', 'closest_match_offset_3',
                        'closest_match_n_bits_diff_0', 'closest_match_n_bits_diff_1', 'closest_match_n_bits_diff_2', 'closest_match_n_bits_diff_3',
                        'n_closest_matches_0', 'n_closest_matches_1', 'n_closest_matches_2', 'n_closest_matches_3',
                       ]



        return df[column_order], df_tot, df_settings


xray_starts_stops = {1:[(np.datetime64('2024-07-28T17:46'),np.datetime64('2024-07-30T09:06')),],
                     2:[(np.datetime64('2024-07-30T18:53'),np.datetime64('2024-07-31T17:27')),],
                     3:[(np.datetime64('2024-07-18T20:55'),np.datetime64('2024-07-19T15:49')),(np.datetime64('2024-07-19T18:21'),np.datetime64('2024-07-21T23:27'))],
                     4:[(np.datetime64('2024-07-25T18:55'),np.datetime64('2024-07-28T13:11')),],
                     5:[(np.datetime64('2024-07-30T12:28'),np.datetime64('2024-07-30T13:58')),],
                  }

def datetime_to_TID(timestamp, doseRate=9.2, xray_start_stop = [(None,None)]):
    """
    Convert timestamps into TID doses
    """

    TID=[]
    xray_on = []
    for t in timestamp:
        TID.append(0)
        _xray_on = False
        if t < xray_start_stop[0][0]:
            TID[-1] = (t - xray_start_stop[0][0]).astype('timedelta64[s]').astype(int)/3600.*doseRate
        else:
            for x_start,x_stop in xray_start_stop:
                if t>x_start:
                    if (x_stop is None) or (t<x_stop):
                        _xray_on = True
                        TID[-1] += (t - x_start).astype('timedelta64[s]').astype(int)/3600.*doseRate
                    else:
                        _xray_on = False
                        TID[-1] += (x_stop - x_start).astype('timedelta64[s]').astype(int)/3600.*doseRate
        xray_on.append(_xray_on)
    return TID, xray_on


def getFileList(chip_num):
    #Defining the name of the directory where the JSON files are stored
    idir = f"../Results/ECOND_COB_{chip_num}"

    #Making a list of the files in the directory defined above, which start with the word "report" and ends with .json
    fnames = list(np.sort(glob.glob(f"{idir}/report*.json")))
    return fnames


bad_periods = [(np.datetime64('2024-07-20T11:17:59'), np.datetime64('2024-07-20T13:38:11'))]

def skip_bad_periods(timestamp):
    for t in bad_periods:
        if timestamp>t[0] and timestamp<t[1]:
            return True
    return False



test_idx=0 #test_streamCompareLoop at 1.2V


def parse_sc_from_flist(fnames, xray_start_stop, verbose=False):

    timestamps = []
    tid = []
    n_errors = []
    capture_i_file = []
    fname = []
    len_daq_capture = []
    n_bx_capture = []
    n_ob_errors = []
    n_ob_errors_per_SRAM = []
    n_ob_errors_per_packet_num = []
    curr_measured = []
    temp_measured = []

    full_i_file = []
    full_err_packet_word_num = []
    full_err_packet_num = []
    full_n_consec = []
    full_timestamp = []
    full_tid = []

    issue_count = 0

    #loop through files
    for _i, f in enumerate(fnames[:]):
        try:
            fname_date = np.datetime64(pd.to_datetime(f.replace('.json','').split('_ECOND_')[-1],format="%Y-%m-%d_%H-%M-%S"))
            if skip_bad_periods(fname_date):continue

            #load test data
            data = json.load(open(f))
            #get specific test (1.2V)
            t=data['tests'][test_idx]['metadata']
            if not ((abs(np.array(t['Voltage']) - t['voltage'])<0.015) | (np.array(t['Voltage'])==-1)).all():
#                 print(t['Voltage'])
#                 print((abs(np.array(t['Voltage']) - t['voltage'])<0.015))
#                 print((t['Voltage']==-1))
#                 print((abs(np.array(t['Voltage']) - t['voltage'])<0.015) | (t['Voltage']==-1))
                print(f'Bad Voltage Setting, expecting {t["voltage"]}')
                print(f'file_num {_i}, {f}')
                print(t['Voltage'])
                continue

            counts = t['word_err_count'][-1][1:]
            n_errors.append(counts)

            #get last timestamp, and convert into TID
            _t = np.datetime64(t['word_err_count'][-1][0])
            _tid = datetime_to_TID([_t],9.2,xray_start_stop)[0][0]


            hasl1a = np.array(t['HasL1A'])
            temperature = np.array(t['Temperature'])
            current = np.array(t['Current'])
            c_67 = current[hasl1a==67]
            t_67 = temperature[hasl1a==67]

            if (hasl1a==0).any():
                c_0 = current[hasl1a==0]
                t_0 = temperature[hasl1a==0]
            else:
                c_0 = np.array([-1])
                t_0 = np.array([-1])
#             if _t>xray_start:
#                 if _t<xray_stop:
#                     _tid = (_t-xray_start).astype('timedelta64[s]').astype(int)/3600.*9.2
#                 else:
#                     _tid = (xray_stop-xray_start).astype('timedelta64[s]').astype(int)/3600.*9.2

            if counts[1]==0:
                timestamps.append(_t)
                tid.append(_tid)
                capture_i_file.append(_i)
                fname.append(f)
                n_sc_bx_7 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==7][:,1].astype(int)
                n_sc_bx_67 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==67][:,1].astype(int)
                if len(n_sc_bx_7)>0:
                    n_bx = n_sc_bx_67[-1] - n_sc_bx_7[-1]
                else:
                    n_bx = n_sc_bx_67[-1]
                n_bx_capture.append(n_bx)
#                 n_bx_capture.append(1.2e9)
                len_daq_capture.append(0)
                n_ob_errors.append(0)
                n_ob_errors_per_SRAM.append(np.zeros(12,dtype=int))
                n_ob_errors_per_packet_num.append(np.zeros(67,dtype=int))
                curr_measured.append([c_67.mean(), c_67.min(), c_67.max(), c_67[0], c_0.mean(), c_0.min(), c_0.max(), c_0[0]])
                temp_measured.append([t_67.mean(), t_67.min(), t_67.max(), t_67[0], t_0.mean(), t_0.min(), t_0.max(), t_0[0]])
                continue

            #load DAQ arrays
            daq_nl1a    = np.array(t['DAQ_nL1A'])
            #find index of captures for where we have 67 L1As in an orbit
            capt_idx67=np.argwhere(daq_nl1a==67)[0]
            daq_asic    = np.array(np.array(t['DAQ_asic'])[daq_nl1a==67][0])[:,::-1]
            daq_emu     = np.array(np.array(t['DAQ_emu'])[daq_nl1a==67][0])[:,::-1]
            daq_counter = np.array(np.array(t['DAQ_counter'])[daq_nl1a==67][0])
            daq_counter = daq_counter[:,0] + (daq_counter[:,1]<<32)

            if len(daq_asic)!=len(daq_emu): continue

            timestamps.append(_t)
            tid.append(_tid)
            capture_i_file.append(_i)
            fname.append(f)
            len_daq_capture.append(len(daq_counter))
            curr_measured.append([c_67.mean(), c_67.min(), c_67.max(), c_67[0], c_0.mean(), c_0.min(), c_0.max(), c_0[0]])
            temp_measured.append([t_67.mean(), t_67.min(), t_67.max(), t_67[0], t_0.mean(), t_0.min(), t_0.max(), t_0[0]])

            if len(daq_counter)<4095:
                n_sc_bx_7 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==7][:,1].astype(int)
                n_sc_bx_67 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==67][:,1].astype(int)
                if len(n_sc_bx_7)>0:
                    n_bx = n_sc_bx_67[-1] - n_sc_bx_7[-1]
                else:
                    n_bx = n_sc_bx_67[-1]
                n_bx_capture.append(n_bx)
            else:
                n_bx_capture.append(daq_counter[-1] - daq_counter[0])

            hasl1a = np.array(t['HasL1A'])
            temperature = np.array(t['Temperature'])
            current = np.array(t['Current'])
            c_67 = current[hasl1a==67]
            t_67 = temperature[hasl1a==67]
            c_67[0],c_67.min(), c_67.max(), c_67.mean(), t_67[0],t_67.min(), t_67.max(), t_67.mean()

            if len(daq_asic)>0:
                #if the lengths are unequal, there is an issue with this capture and we continue

                #find index for all words where the asic and the emulator disagree
                mismatch_idx = np.argwhere(~(daq_asic==daq_emu).flatten()).flatten()

                #look for cases where the difference between neighboring mismatches is exactly 12

                is_ob_err= (((np.roll(mismatch_idx,-1) - mismatch_idx)==12)
                            & ((daq_emu.flatten()[mismatch_idx]>>8)!=0x555555)
                            & ((daq_emu.flatten()[np.roll(mismatch_idx,1)]>>8)!=0x555555)
                            & ((np.roll(mismatch_idx,-2) - mismatch_idx)>=24)
                           )

                n=is_ob_err.sum()

                #BX number of where the errors are that appear to be from OB SRAM errors
                err_idx=(mismatch_idx[is_ob_err]/6).astype(int)
                etx_idx = (mismatch_idx[is_ob_err]%6)

                #call it a new error "group" anytime the difference between errors is not 2 BX
                new_packet=(daq_counter[err_idx] - np.roll(daq_counter[err_idx],1))!=2
                #assign packet number to each error group
                capture_packet_number = (np.cumsum(new_packet))

                #count the number of consecutive errors in this error group
                n_consec=np.unique(capture_packet_number,return_counts=True)[1]+1
                if verbose:
                    print(_i, f, len(daq_asic), len(daq_emu), f"{_tid:.5f}", counts, n)

                n_ob_errors.append(n)
                #match up the emulator line to a complete orbit capture to identify the word
                #look for which packet number it is, and the location of the error in thatpacket
                err_packet_index = []
                for i in range(len(err_idx)):
                    x=daq_emu[err_idx[i]]
                    p=packet_word_index[(full_capture[:,:]==x[:]).all(axis=1)][0]
                    err_packet_index.append(p[etx_idx[i]])

                #cast as numpy array and take mod 1000 to get which l1a within the orbit was effected
                err_packet_index = np.array(err_packet_index)

                err_packet_word_num = err_packet_index[new_packet]%1000
                err_packet_num = (err_packet_index[new_packet]/1000).astype(int)
                n_ob_errors_per_SRAM.append(np.zeros(12,dtype=int))
                n_ob_errors_per_packet_num.append(np.zeros(67,dtype=int))

                #count how many times each sram and packet shows up in this stream comparison period
                if len(err_packet_index)>0:
                    _sram, _counts = np.unique(err_packet_word_num%12,return_counts=True)
                    n_ob_errors_per_SRAM[-1][_sram] = _counts

                    _packet, _counts = np.unique(err_packet_num,return_counts=True)
                    n_ob_errors_per_packet_num[-1][_packet] = _counts

                full_i_file += [_i]*len(n_consec)
                full_timestamp += [_t]*len(n_consec)
                full_tid += [_tid]*len(n_consec)
                full_err_packet_word_num += err_packet_word_num.tolist()
                full_err_packet_num += err_packet_num.tolist()
                full_n_consec += n_consec.tolist()
            else:
                n_ob_errors.append(0)
                n_ob_errors_per_SRAM.append(np.zeros(12,dtype=int))
                n_ob_errors_per_packet_num.append(np.zeros(67,dtype=int))


        except:
            issue_count += 1
            if verbose:
                print(f'issue in {_i} file {f}')
            continue


    n_ob_errors_per_SRAM = np.array(n_ob_errors_per_SRAM)
    n_ob_errors_per_packet_num = np.array(n_ob_errors_per_packet_num)
    df = pd.DataFrame({'file_num':full_i_file,
                       'timestamp':full_timestamp,
                       'TID':full_tid,
                       'err_packet_word_num':full_err_packet_word_num,
                       'err_packet_num':full_err_packet_num,
                       'n_err':full_n_consec})

    df['duringTID'] = (df.TID>0) & (df.TID<df.TID.values[-1])
    df['ob_sram'] = df.err_packet_word_num%12


    temp_measured = np.array(temp_measured)
    curr_measured = np.array(curr_measured)

    df_tot = pd.DataFrame({'file_num':capture_i_file,
                           'file_name':fname,
                           'timestamp':timestamps,
                           'TID':tid,
                           'len_capture':len_daq_capture,
                           'n_bx_capture':n_bx_capture,
                           'ob_err_total':n_ob_errors,
                           'temp_mean':temp_measured[:,0],
                           'temp_min':temp_measured[:,1],
                           'temp_max':temp_measured[:,2],
                           'temp_first':temp_measured[:,3],
                           'temp_0_mean':temp_measured[:,4],
                           'temp_0_min':temp_measured[:,5],
                           'temp_0_max':temp_measured[:,6],
                           'temp_0_first':temp_measured[:,7],
                           'curr_mean':curr_measured[:,0],
                           'curr_min':curr_measured[:,1],
                           'curr_max':curr_measured[:,2],
                           'curr_first':curr_measured[:,3],
                           'curr_0_mean':curr_measured[:,4],
                           'curr_0_min':curr_measured[:,5],
                           'curr_0_max':curr_measured[:,6],
                           'curr_0_first':curr_measured[:,7],
                          })

    df_tot['n_l1a_capture'] = df_tot.n_bx_capture/3564*67
    for i in range(12):
        df_tot[f'ob_err_sram_{i:02d}'] = n_ob_errors_per_SRAM[:,i]
    df_tot['duringTID'] = (df_tot.TID>0) & (df_tot.TID<df_tot.TID.values[-1])
    for i in range(12):
        df_tot[f'ob_rate_sram_{i:02d}'] = df_tot[f'ob_err_sram_{i:02d}']/df_tot.n_l1a_capture

    df_tot['ob_err_total'] = df_tot[[f'ob_err_sram_{i:02d}' for i in range(12)]].sum(axis=1)
    df_tot['ob_rate_total'] = df_tot[[f'ob_rate_sram_{i:02d}' for i in range(12)]].sum(axis=1)

    return df_tot, df





def parse_single_file(file_name,voltage,verbose=True):
    data = json.load(open(file_name))

    for _t in data['tests']:
        if _t['nodeid']!=f'test_TID.py::test_streamCompareLoop[{voltage}]':
            continue
        t = _t['metadata']
        daq_nl1a    = np.array(t['DAQ_nL1A'])
        print(t['voltage'])
        print(t['Timestamp'][0])

        print('Correct Voltages:', (abs(np.array(t['Voltage']) - t['voltage'])<0.015).all())
        capt_idx67=np.argwhere(daq_nl1a==67).flatten()[0]

        daq_asic    = np.array(np.array(t['DAQ_asic'])[daq_nl1a==67][0])[:,::-1]
        daq_emu     = np.array(np.array(t['DAQ_emu'])[daq_nl1a==67][0])[:,::-1]
        daq_counter = np.array(np.array(t['DAQ_counter'])[daq_nl1a==67][0])
        daq_counter = daq_counter[:,0] + (daq_counter[:,1]<<32)

        new_capture = (daq_counter-np.roll(daq_counter,1))!=1
        n_captures = new_capture.sum()

        mismatch_idx = np.argwhere(~(daq_asic==daq_emu).flatten()).flatten()
        is_ob_err= (((np.roll(mismatch_idx,-1) - mismatch_idx)==12) &
                    ((daq_emu.flatten()[mismatch_idx]>>8)!=0x555555) &
                    ((daq_emu.flatten()[np.roll(mismatch_idx,1)]>>8)!=0x555555) &
                    ((np.roll(mismatch_idx,-2) - mismatch_idx)>=24)
                   )
        print(is_ob_err.sum())

        hasl1a = np.array(t['HasL1A'])
        temperature = np.array(t['Temperature'])
        current = np.array(t['Current'])

        if verbose:
            print('Frequency of errors at different spacings')
            print('    OB should be only at intervals of 12, if others show up it indicates these may be other issues')
            for i in range(25):
                k= (((np.roll(mismatch_idx,-1) - mismatch_idx)==i)
                    & ((daq_emu.flatten()[mismatch_idx]>>8)!=0x555555)
                    & ((daq_emu.flatten()[np.roll(mismatch_idx,1)]>>8)!=0x555555)
                    & ((np.roll(mismatch_idx,-2) - mismatch_idx)>=(2*i))
                   )

                print(i,(k).sum())


        err_idx=(mismatch_idx[is_ob_err]/6).astype(int)
        etx_idx = (mismatch_idx[is_ob_err]%6)

        new_packet=(daq_counter[err_idx]-np.roll(daq_counter[err_idx],1))!=2
        capture_packet_number = (np.cumsum(new_packet))

        n_consec=np.unique(capture_packet_number,return_counts=True)[1]+1
        if len(daq_counter)<4095:
            n_sc_bx_7 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==7][:,1].astype(int)
            n_sc_bx_67 = np.array(t['word_err_count'])[np.array(t['HasL1A'])==67][:,1].astype(int)
            if len(n_sc_bx_7)>0:
                n_bx_capture = n_sc_bx_67[-1] - n_sc_bx_7[-1]
            else:
                n_bx_capture = n_sc_bx_67[-1]
        else:
            n_bx_capture = daq_counter[-1] - daq_counter[0]

        err_packet_index = []
        for i in range(len(err_idx)):
            x=daq_emu[err_idx[i]]
            p=packet_word_index[(full_capture[:,2:]==x[2:]).all(axis=1)][0]
            err_packet_index.append(p[etx_idx[i]])
        err_packet_index = np.array(err_packet_index)

        err_packet_word_num = err_packet_index[new_packet]%1000
        err_packet_num = (err_packet_index[new_packet]/1000).astype(int)

        return t, daq_asic, daq_emu, daq_counter, n_captures, n_bx_capture, mismatch_idx, is_ob_err, err_idx, n_consec, err_packet_index, err_packet_num, err_packet_word_num
