import numpy as np
import glob
import json
import pandas as pd
from datetime import datetime
import os

_file_path = os.path.dirname(os.path.abspath(__file__))
packet_word_index = np.loadtxt(f'{_file_path}/data/packet_word_index.csv',delimiter=',',dtype=int)
full_capture = np.loadtxt(f'{_file_path}/data/full_clean_orbit.csv',delimiter=',',dtype=str)
full_capture = np.vectorize(lambda x: int(x,16))(full_capture)
daq_stream = full_capture.flatten()
isIdle = (daq_stream==0x55555500)
nonMandatoryIdle = (isIdle & np.roll(isIdle,1))
sram_data = daq_stream[~nonMandatoryIdle]

def getBISTresults(fname, verbose=False):
    data = json.load(open(fname))

    found_BIST_data = False
    for _t in data['tests'][::-1]:
        if _t['nodeid']=='test_bist_threshold.py::test_bist_full':
            found_BIST_data = True
            break
        if '::test_bist_full' in _t['nodeid']:
            found_BIST_data = True
            break
        if _t['nodeid']=='test_TID_configurable.py::test_bist':
            found_BIST_data = True
            break
        if _t['nodeid']=='test_TID.py::test_bist':
            found_BIST_data = True
            break
    if not found_BIST_data:
        print('No BIST data found')
        return None

    t = _t['metadata']

    voltages = np.array(t['voltages'])
    try:
        timestamps = np.array(t['timestamps'])
    except:
        timestamps = np.datetime64('2025-01-01')

    try:
        b = np.array(t['bist_results'])
        b_pp = b[:,4:]
        b_ob = b[:,:4]
    except:
        b_pp = np.array(t['ppResults'])
        b_ob = np.array(t['obResults'])

    if (b_pp==4095).any():
        _pp_min = voltages[(b_pp==4095).all(axis=1)].min()
    else:
        _pp_min = 1.5
    if (b_ob==4095).any():
        _ob_min = voltages[(b_ob==4095).all(axis=1)].min()
    else:
        _ob_min = 1.5
    if verbose:
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


def getParsedTables(fname, forceReprocess=False, debug_print=False, drop_last_lc_readout=False):
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
        d_bist = pd.read_csv(fname_bist,index_col='voltages')
    else:
        # print(fname)
        d_tot, df, df_setting = checkErr(fname, debug_print=debug_print, drop_last_lc_readout=drop_last_lc_readout)

        d_bist = getBISTresults(fname, debug_print)
        if not d_bist is None:
            d_tot = d_tot.merge(d_bist[['pass_PP_bist','pass_OB_bist','PP_bist_01','OB_bist_01']],left_index=True,right_index=True,how='left').sort_index()
        else:
            d_tot['pass_PP_bist'] = False
            d_tot['pass_OB_bist'] = False
            d_tot['PP_bist_01'] = 'xxxxxxxxxxxx'
            d_tot['OB_bist_01'] = 'xxxxxxxxxxxx'
        d_tot.timestamp = pd.to_datetime(d_tot.timestamp)

        #if we pass bist at 1.19, fill rest as pass
        if not d_bist is None:
            if d_bist.loc[1.19].pass_PP_bist:
                d_tot.PP_bist_01 = d_tot.PP_bist_01.fillna('111111111111').astype(object)
                d_tot.pass_PP_bist = d_tot.pass_PP_bist.astype(bool).fillna(True)
            if d_bist.loc[1.19].pass_OB_bist:
                d_tot.OB_bist_01 = d_tot.OB_bist_01.fillna('111111111111').astype(object)
                d_tot.pass_OB_bist = d_tot.pass_OB_bist.astype(bool).fillna(True)

        if not df is None:
            df['file'] = fname
        d_tot['file'] = fname
        d_tot.sort_index(inplace=True)
    return d_tot, df, d_bist, df_setting


def checkErr(fname,i=0, drop_last_lc_readout = False, debug_print=False):
    df_packets, df_tot, df_settings=parse_sram_errors_per_packet(fname,sram_data, debug_print=debug_print)
    sum_cols = ['isOBErrors',
                'isSingleError',
                'isBadPacketCRC',
                'isBadPacketHeader',
                'likelyInputCRCError',
                'likelyPPError',
                'isBadOBError',
                'isOtherError',
                'isSingleError_SingleBit',
                'isSingleError_MultiBit',
                'isSingleError_SingleBit_SpecialPackets',
                'isSingleError_MultiBit_SpecialPackets',
                'isOBErrors_SpecialPackets',
               ]


    ###fix that allows dropping the last lc buffer readout from the sums, if there are more than 1
    # y=df_tot.set_index('voltages')[['n_captured_bx','n_captures','n_packets','word_count','error_count','timestamp','current','temperature']]
    if drop_last_lc_readout:
        y=df_tot[['voltages','n_captures','n_packets','n_captured_bx']].groupby('voltages').apply(lambda x: x.iloc[:-1] if len(x)>1 else x, include_groups=False).groupby('voltages').sum()
    else:
        y=df_tot[['voltages','n_captures','n_packets','n_captured_bx']].groupby('voltages').sum()

    y[['word_count','error_count','timestamp','current','temperature']]=df_tot.groupby('voltages')[['word_count','error_count','timestamp','current','temperature']].first()

    if not df_packets is None:
        df_packets['isSpecialPacket'] = df_packets.packet_number.isin([3,4,11,27,32,33,49])
        df_packets['isSingleError_SingleBit'] = df_packets.isSingleError & (df_packets.asic_emu_bit_diff_0==1)
        df_packets['isSingleError_MultiBit'] = df_packets.isSingleError & (df_packets.asic_emu_bit_diff_0>1)

        df_packets['isSingleError_SingleBit_SpecialPackets'] = df_packets.isSingleError_SingleBit & df_packets.isSpecialPacket
        df_packets['isSingleError_MultiBit_SpecialPackets'] = df_packets.isSingleError_MultiBit & df_packets.isSpecialPacket
        df_packets['isOBErrors_SpecialPackets'] = df_packets.isOBErrors & df_packets.isSpecialPacket

        #Bad OB Errors, with multiple errors but not badCRC or packetHeader
        df_packets['isBadOBError'] = (~df_packets.isOBErrors & (df_packets.n_tot_errors>2) & ~df_packets.isBadPacketCRC & ~df_packets.isBadPacketHeader)

        #### likely input CRC errors
        df_packets['likelyInputCRCError'] = (df_packets.isBadPacketCRC & df_packets.isBadPacketHeader & (df_packets.n_tot_errors>=4) & (((df_packets.error_word_emu_2.values>>25))==0b1110000) &
                                            (((((df_packets.error_word_asic_2.values>>25))==0b1100000) & (df_packets.asic_emu_bit_diff_2<5)) |
                                             ((((df_packets.error_word_asic_2.values>>25))==0b1000001) & (df_packets.asic_emu_bit_diff_2<10)))
                                            )

        #### Errors that appear like PP errors (this is not a good classification at the moment)
        df_packets['likelyPPError'] = (df_packets.isBadPacketCRC & (df_packets.n_tot_errors>=2) & ~df_packets.isOBErrors & ~df_packets.likelyInputCRCError)

        df_packets['isOtherError'] = ~(df_packets.isOBErrors | df_packets.isSingleError | df_packets.isBadOBError | df_packets.likelyInputCRCError | df_packets.likelyPPError)


        if drop_last_lc_readout:
            ###fix that allows dropping the last lc buffer readout from the sums, if there are more than 1
            # x=df_packets.groupby('voltages').sum()[sum_cols]
            x=df_packets.groupby(['voltages','lc_number']).sum()[sum_cols].groupby(['voltages']).apply(lambda x: x.iloc[:-1] if len(x)>1 else x, include_groups=False).groupby('voltages').sum()
        else:
            x=df_packets.groupby(['voltages']).sum()[sum_cols]

        z=y.merge(x,left_index=True,right_index=True,how='left').fillna(0)
    else:
        z = y
        z[sum_cols] = 0

    z['error_rate'] = (z.isOBErrors + z.isSingleError)/(z.n_captured_bx/3564*67)
    z['run'] = i
    z = z.astype({c: int for c in sum_cols})
    return z, df_packets, df_settings

### function which merges split-out daq capture data (from January TID runs) back into a single json file
def merge_jsons(fname, old_dir_name='json_files', new_dir_name='merged_jsons'):
    data = json.load(open(fname))
    for t in data['tests']:
        if 'streamCom' in t['nodeid']:
            # if t['outcome']=='error':
            #     continue
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


def parse_packet_errors(daq_asic, daq_emu, daq_count, daq_idx):
    d_asic = daq_asic.flatten()
    d_emu = daq_emu.flatten()
    d_count = daq_count.flatten()
    d_idx = daq_idx.flatten()
    mismatch_idx = np.argwhere(~(d_asic==d_emu)).flatten()
    if len(mismatch_idx)==0:
        # print('Packet Has Zero Errors', d_idx[0])
        return -1,0,0,0,0,0,0,0,0

    try:
        first_mismatch_packet_word_index = packet_word_index[np.argwhere((full_capture[:,2:]==d_emu[20:24]).all(axis=1))[0,0]][mismatch_idx[0]%6]
    except:
        first_mismatch_packet_word_index = -1

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
            if debug_print: print(f'Setup erred out, {tname}')
            continue
        # if _t['outcome']=='error':
        #     if debug_print: print('Test erred out')
        #     continue

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

        #find cases where first current is an outlier/stale data
        #if mean is pulled by first entry, remove first entry
        c = current[hasL1A==nl1a]
        residual = (c-c.mean())/c.std()
        #if residual of first reading is one sign, and all other residuals are opposite sign, first reading is pulling measurement and should be skipped
        if ((residual[0]<0) & (residual[1:]>0).all()) or ((residual[0]>0) & (residual[1:]<0).all()):
            c = current[hasL1A==nl1a][1:]
        mean_current, std_current = c.mean(), c.std()
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
                        this_capt_idx = np.argwhere((full_capture[:,2:]==this_capt_daq_emu_line).all(axis=1))[0][0]
                        last_capt_idx = np.argwhere((full_capture[:,2:]==last_capt_daq_emu_line).all(axis=1))[0][0]
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
                print(count_errors)
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


def getFileList(chip_num):
    #Defining the name of the directory where the JSON files are stored
    idir = f"../Results/ECOND_COB_{chip_num}"

    #Making a list of the files in the directory defined above, which start with the word "report" and ends with .json
    fnames = list(np.sort(glob.glob(f"{idir}/report*.json")))
    return fnames



