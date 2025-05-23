import numpy as np
import pandas as pd


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
