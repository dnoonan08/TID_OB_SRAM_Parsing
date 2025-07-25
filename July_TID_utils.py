import numpy as np
import datetime

import matplotlib
from matplotlib.dates import num2date
import matplotlib.pyplot as plt

import mplhep as hep
from cycler import cycler

import glob
import pandas as pd

color_cycle =  ['#3f90da','#ffa90e','#bd1f01','#94a4a2','#832db6','#a96b59','#e76300','#b9ac70','#717581','#92dadd']
hep.style.use(["CMS", {"axes.prop_cycle": cycler("color", color_cycle)} ])

_startTime_COB_5Pct_1_1 = np.array('2025-07-17 12')
_startTime_COB_5Pct_1_3 = np.array('2025-07-19 12')
_startTime_COB_10Pct_1_2 = np.array('2025-07-22 12')
_startTime_COB_15Pct_4_4 = np.array('2025-07-23 12')

_xray_times = {
    'COB-5Pct-1-1':{
        'Starttime':np.datetime64('2025-07-17 12:00'),
        'Cooldown' :np.datetime64('2025-07-17 17:50'),
        'Xray 10'  :np.datetime64('2025-07-18 10:27'),
        'Xray 50'  :np.datetime64('2025-07-18 18:29'),
        'Xray Off' :np.datetime64('2025-07-19 11:37'),
        },
    'COB-5Pct-1-3':{
        'Starttime':np.datetime64('2025-07-19 12:00'),
        'Cooldown' :np.datetime64('2025-07-19 14:43'),
        'Xray 10'  :np.datetime64('2025-07-19 16:41'),
        'Xray 50'  :np.datetime64('2025-07-19 22:20'),
        'Xray Off' :np.datetime64('2025-07-22 10:45'),
        },
    'COB-10Pct-1-2':{
        'Starttime':np.datetime64('2025-07-22 12:00'),
        'Cooldown' :np.datetime64('2025-07-22 14:00'),
        'Xray 10'  :np.datetime64('2025-07-22 16:23'),
        'Xray 50'  :np.datetime64('2025-07-22 21:40'),
        'Xray Off' :np.datetime64('2025-07-23 10:13'),
        },
    'COB-15Pct-4-4':{
        'Starttime':np.datetime64('2025-07-23 12:00'),
        'Cooldown' :np.datetime64('2025-07-23 14:00'),
        'Xray 10'  :np.datetime64('2025-07-23 15:43'),
        'Xray 50'  :np.datetime64('2025-07-24 04:38'),
        'Xray Off' :np.datetime64('2025-07-25 09:58'),
        },
    'COB-10Pct-1-1':{
        'Starttime':np.datetime64('2025-07-25 12:00'),
        'Cooldown' :np.datetime64('2025-07-25 13:54'),
        'Xray 10'  :np.datetime64('2025-07-25 16:12'),
        'Xray 50'  :np.datetime64('2025-07-25 21:30'),
        'Xray Off' :np.datetime64('2025-07-28 11:00'),
        },
    }

def mark_TID_times(ax,_COB_,leg_loc=None):
    _xlim = ax.get_xlim()
    _xlim = (num2date(_xlim[0]), num2date(_xlim[1]))
    _ylim = ax.get_ylim()
    _t = _xray_times[_COB_]['Cooldown'].astype(object)
    _chiller_on = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 10'].astype(object)
    _xray_10    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 50'].astype(object)
    _xray_50    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray Off'].astype(object)
    _xray_Off   = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')

    if leg_loc==False:
        return
    if leg_loc is None:
        ax.legend()
    elif 'right' in leg_loc:
        offset = .05
        if '+' in leg_loc:
            _extra = leg_loc.split('+')[-1]
            if _extra=='':
                offset=0.10
            else:
                try:
                    offset = int(_extra)/100.
                except:
                    offset = .10
        ax.legend(loc='upper left', bbox_to_anchor=(1+offset, 1.0))

def mark_TID_times_COB_5Pct_1_1(ax,leg_loc=None):
    _xlim = ax.get_xlim()
    _xlim = (num2date(_xlim[0]), num2date(_xlim[1]))
    _ylim = ax.get_ylim()
    _chiller_on = datetime.datetime(2025, 7, 17, 17, 50, tzinfo=datetime.timezone.utc)
    _xray_10    = datetime.datetime(2025, 7, 18, 10, 27, tzinfo=datetime.timezone.utc)
    _xray_50    = datetime.datetime(2025, 7, 18, 16, 30, tzinfo=datetime.timezone.utc)
    _xray_Off   = datetime.datetime(2025, 7, 19, 11, 37, tzinfo=datetime.timezone.utc)
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')

    if leg_loc==False:
        return
    if leg_loc is None:
        ax.legend()
    elif 'right' in leg_loc:
        offset = .05
        if '+' in leg_loc:
            _extra = leg_loc.split('+')[-1]
            if _extra=='':
                offset=0.10
            else:
                try:
                    offset = int(_extra)/100.
                except:
                    offset = .10
        ax.legend(loc='upper left', bbox_to_anchor=(1+offset, 1.0))

def mark_TID_times_COB_5Pct_1_3(ax,leg_loc=None):
    _xlim = ax.get_xlim()
    _xlim = (num2date(_xlim[0]), num2date(_xlim[1]))
    _ylim = ax.get_ylim()
    _COB_ = 'COB-5Pct-1-3'
    _t = _xray_times[_COB_]['Cooldown'].astype(object)
    _chiller_on = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 10'].astype(object)
    _xray_10    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 50'].astype(object)
    _xray_50    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray Off'].astype(object)
    _xray_Off   = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')

    if leg_loc==False:
        return
    if leg_loc is None:
        ax.legend()
    elif 'right' in leg_loc:
        offset = .05
        if '+' in leg_loc:
            _extra = leg_loc.split('+')[-1]
            if _extra=='':
                offset=0.10
            else:
                try:
                    offset = int(_extra)/100.
                except:
                    offset = .10
        ax.legend(loc='upper left', bbox_to_anchor=(1+offset, 1.0))

def mark_TID_times_COB_10Pct_1_2(ax,leg_loc=None):
    _xlim = ax.get_xlim()
    _xlim = (num2date(_xlim[0]), num2date(_xlim[1]))
    _ylim = ax.get_ylim()
    _chiller_on = datetime.datetime(2025, 7, 22, 14,  0, tzinfo=datetime.timezone.utc)
    _xray_10    = datetime.datetime(2025, 7, 22, 16, 23, tzinfo=datetime.timezone.utc)
    _xray_50    = datetime.datetime(2025, 7, 22, 21, 40, tzinfo=datetime.timezone.utc)
    _xray_Off   = datetime.datetime(2025, 7, 23, 10, 13, tzinfo=datetime.timezone.utc)
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')

    if leg_loc==False:
        return
    if leg_loc is None:
        ax.legend()
    elif 'right' in leg_loc:
        offset = .05
        if '+' in leg_loc:
            _extra = leg_loc.split('+')[-1]
            if _extra=='':
                offset=0.10
            else:
                try:
                    offset = int(_extra)/100.
                except:
                    offset = .10
        ax.legend(loc='upper left', bbox_to_anchor=(1+offset, 1.0))


def mark_TID_times_COB_15Pct_4_4(ax,leg_loc=None):
    _xlim = ax.get_xlim()
    _xlim = (num2date(_xlim[0]), num2date(_xlim[1]))
    _ylim = ax.get_ylim()
    _COB_ = 'COB-15Pct-4-4'
    _t = _xray_times[_COB_]['Cooldown'].astype(object)
    _chiller_on = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 10'].astype(object)
    _xray_10    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray 50'].astype(object)
    _xray_50    = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    _t = _xray_times[_COB_]['Xray Off'].astype(object)
    _xray_Off   = datetime.datetime(_t.year, _t.month, _t.day, _t.hour, _t.minute, tzinfo=datetime.timezone.utc)
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')

    if leg_loc==False:
        return
    if leg_loc is None:
        ax.legend()
    elif 'right' in leg_loc:
        offset = .05
        if '+' in leg_loc:
            _extra = leg_loc.split('+')[-1]
            if _extra=='':
                offset=0.10
            else:
                try:
                    offset = int(_extra)/100.
                except:
                    offset = .10
        ax.legend(loc='upper left', bbox_to_anchor=(1+offset, 1.0))

def plot_error_rate(d_tot,
                    voltages,
                    numerator,
                    denominator,
                    title='Error Rate',
                    ylabel='Error Rate',
                    xlabel='Time',
                    axis=None,
                    logy=False,
                    bist=False,
                    temperature=False,
                    scatterplot=False,
                    markerstyle=None,
                    leg_offset='right+',
                    mark_TID_times=None,
                    xlim = (None,None)):
    if axis is None:
        fig,ax = plt.subplots(1,1)
    else:
        ax = axis
    for v in voltages:
        d = d_tot.loc[v]
        e_rate = d[numerator].sum(axis=1)/d[denominator].sum(axis=1)*100
        if scatterplot:
            ax.scatter(d.timestamp,e_rate, label=f'{v:.02f}V')
        else:
            ax.plot(d.timestamp,e_rate, label=f'{v:.02f}V',marker=markerstyle)
    if bist:
        ax2 = ax.twinx()
        ax2.plot(bist_result.timestamps,bist_result.pp_passing_v,color='black',label='PP Bist',linestyle='dashed')
        ax2.plot(bist_result.timestamps,bist_result.ob_passing_v,color='red',label='OB Bist',linestyle='dashed')
        ax.plot(bist_result.timestamps.iloc[0],.1,color='black',label='PP Bist',linestyle='dashed')
        ax.plot(bist_result.timestamps.iloc[0],.1,color='red',label='OB Bist',linestyle='dashed')
        ax2.set_ylabel('BIST Passing Voltage')

    if temperature:
        ax2 = ax.twinx()
        d = d_tot.loc[1.2]
        ax2.plot(d.timestamp,d.temperature,color='black',label='Temperature',linestyle='dashed')
        # ax2.set_ylim(0.8,None)
        ax.plot(bist_result.timestamps.iloc[0],.1,color='black',label='Temperature',linestyle='dashed')
        ax2.set_ylabel('Temperature')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim[0],xlim[1])
#    plt.xticks(rotation = 45)
    ax.tick_params(labelrotation=45)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(12))
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    if not mark_TID_times is None:
        mark_TID_times(ax,leg_offset)
    if ax is None:
        return fig,ax
    else:
        return ax

def loadData(_COB_,filePath='/eos/user/d/dnoonan/July_2025_TID_Data/parsed_data'):
    d_tot_list = []
    d_bist_list = []
    d_packets_list = []
    d_settings_list = []

    flist = glob.glob(f'{filePath}/report_TID_chip_{_COB_}*totals.csv')
    flist.sort()

    for fname in flist:
        try:
            fname_totals = fname
            fname_packets = fname.replace('_totals','_packets')
            fname_bist = fname.replace('_totals','_bist')
            fname_settings = fname.replace('_totals','_settings')
            d_tot = pd.read_csv(fname_totals,index_col='voltages')
            d_packet = pd.read_csv(fname_packets,index_col=0)
            d_bist = pd.read_csv(fname_bist,index_col='voltages')
            d_settings = pd.read_csv(fname_settings)#,index_col='voltages')

            d_tot_list.append(d_tot)
            d_bist_list.append(d_bist)
            d_packets_list.append(d_packet)
            d_settings_list.append(d_settings)
        except:
            continue
    d_tot = pd.concat(d_tot_list)
    d_packets = pd.concat(d_packets_list)
    d_bist = pd.concat(d_bist_list)
    d_settings = pd.concat(d_settings_list)

    d_tot.timestamp = pd.to_datetime(d_tot.timestamp)
    d_bist.timestamps = pd.to_datetime(d_bist.timestamps)

    column_names = d_tot.columns
    d_tot['n_captured_packets'] = d_tot['n_captured_bx']/3564*67.
    d_tot = d_tot[list(column_names)[:3] + ['n_captured_packets'] + list(column_names)[3:]]

    x = d_bist.copy(deep=True)
    x['pp_passing_v'] = np.where((d_bist[['PPbist_1','PPbist_2','PPbist_3','PPbist_4']]==4095).all(axis=1), d_bist.index, 1.4)
    x['ob_passing_v'] = np.where((d_bist[['OBbist_1','OBbist_2','OBbist_3','OBbist_4']]==4095).all(axis=1), d_bist.index, 1.4)
    x['i2c_drop_v'] = np.where((d_bist[['PPbist_1','PPbist_2','PPbist_3','PPbist_4','OBbist_1','OBbist_2','OBbist_3','OBbist_4']]>=0).all(axis=1),d_bist.index,1.4)
    d_summary = x.groupby('file').min()[['timestamps','pp_passing_v','ob_passing_v','i2c_drop_v']]

    x = d_tot[['file','error_count','word_count']].copy(deep=True)
    x['error_free_voltage'] = np.where(x.error_count==0,x.index,1.35)
    x['last_error_voltage'] = np.where(x.error_count>0,x.index,0)
    x['error_rate_1e4'] = np.where((x.error_count/x.word_count)<1e-4,x.index,1.35)
    x['error_rate_1e6'] = np.where((x.error_count/x.word_count)<1e-6,x.index,1.35)
    x['error_rate_1e8'] = np.where((x.error_count/x.word_count)<1e-8,x.index,1.35)
    d_summary['etx_error_free'] = x.groupby('file').min()[['error_free_voltage']]
    d_summary['etx_last_error'] = x.groupby('file').max()[['last_error_voltage']]
    d_summary['etx_error_1e4'] = x.groupby('file').min()[['error_rate_1e4']]
    d_summary['etx_error_1e6'] = x.groupby('file').min()[['error_rate_1e6']]
    d_summary['etx_error_1e8'] = x.groupby('file').min()[['error_rate_1e8']]

    d_packets.set_index(['file','voltages'],inplace=True)
    return d_tot,d_packets,d_bist,d_settings,d_summary
