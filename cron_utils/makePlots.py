
import sys
sys.path.append('..')

from July_TID_utils import loadData, mark_TID_times, _xray_times

import matplotlib
import matplotlib.pyplot as plt

import mplhep as hep
from cycler import cycler

color_cycle =  ['#3f90da','#ffa90e','#bd1f01','#94a4a2','#832db6','#a96b59','#e76300','#b9ac70','#717581','#92dadd']
hep.style.use(["CMS", {"axes.prop_cycle": cycler("color", color_cycle)} ])

def voltage_summary(d_summary, mark_TID_times, _COB_, xlim=(None,None), ylim=(None,None)):
    fig,ax = plt.subplots(1,1)

    ax.plot(d_summary.timestamps,d_summary.etx_error_free,label='ETX Error Free')
    ax.plot(d_summary.timestamps,d_summary.etx_error_1e8,label='ETX Error < 1e-8')
    ax.plot(d_summary.timestamps,d_summary.etx_error_1e6,label='ETX Error < 1e-6')
    ax.plot(d_summary.timestamps,d_summary.etx_error_1e4,label='ETX Error < 1e-4')
    ax.plot(d_summary.timestamps,d_summary.pp_passing_v,label='PP BIST')
    ax.plot(d_summary.timestamps,d_summary.ob_passing_v,label='OB BIST')
    ax.plot(d_summary.timestamps,d_summary.i2c_drop_v,label='I2C dropout')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(6))
    ax.set_ylabel('Voltage')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_title(_COB_)
    mark_TID_times(ax,_COB_,'right')
    return fig

def temperaturePlot(d_tot,
                    mark_TID_times,
                    _COB_,
                    voltages = [1.08,1.20,1.32],
                    colors = {1.08:'red',1.2:'black',1.32:'green'},
                    xlim = (None,None),
                    ylim = (None,None),
                    bad_temps=[1,2,-1,-2],
                   ):
    fig,ax = plt.subplots(1,1)
    for v in voltages:
        d = d_tot.loc[v]
        d = d[~d.temperature.isin(bad_temps)]
        ax.plot(d.timestamp,d.temperature,color=colors[v],label=f'Temperature {v:.2f} V',linestyle='dashed')


    ax.set_title(f'Temperature Vs Time, {_COB_}')
    ax.set_ylabel('Temperature')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(6))
    mark_TID_times(ax,_COB_)
    
    return fig

def currentPlot(d_tot,
                mark_TID_times,
                _COB_,
                voltages = [1.08,1.20,1.32],
                colors = {1.08:'red',1.2:'black',1.32:'green'},
                xlim = (None,None),
                ylim = (None,None),
               ):
    fig,ax = plt.subplots(1,1)

    for v in voltages:
        d = d_tot.loc[v]
        ax.plot(d.timestamp,d.current,color=colors[v],label=f'{v:.2f} V',linestyle='dashed')

    ax.set_ylabel('Current')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(6))
    ax.set_title(f'Current, {_COB_}')

    mark_TID_times(ax,_COB_)
    return fig

_COB_ = 'COB-15Pct-4-4'

def makeSummaryPlots(_COB_):
    _startTime = _xray_times[_COB_]['Starttime']
    d_tot,d_packets,d_bist,d_settings,d_summary = loadData(_COB_)
    f = voltage_summary(d_summary, mark_TID_times, _COB_);
    f.savefig(f'/eos/user/d/dnoonan/July_2025_TID_Data/plots/{_COB_}/VoltageSummary_{_COB_}.pdf', bbox_inches='tight')

    f = temperaturePlot(d_tot, mark_TID_times, _COB_,xlim=(_startTime, None),bad_temps=[])
    f.savefig(f'/eos/user/d/dnoonan/July_2025_TID_Data/plots/{_COB_}/Temperature_{_COB_}.pdf', bbox_inches='tight')

    f = currentPlot(d_tot, mark_TID_times, _COB_,xlim=(_startTime, None))
    f.savefig(f'/eos/user/d/dnoonan/July_2025_TID_Data/plots/{_COB_}/Current_{_COB_}.pdf', bbox_inches='tight')
