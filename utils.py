import numpy as np
import json

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
    return daq_asic, daq_emu, daq_count, daq_idx
