import numpy as np
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import num2date
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore",message="no explicit representation of timezones available for np.datetime64")

import scipy

chiller_on = {'COB122':np.datetime64('2025-01-13 15:19'),
              'COB119':np.datetime64('2025-01-17 15:09'),
              'COB118':np.datetime64('2025-01-20 15:14'),
              'COB114':np.datetime64('2025-01-23 15:45'),
              'COB121':np.datetime64('2025-01-27 13:55'),
              'COB903':np.datetime64('2025-01-29 17:35'),
              'COB901':np.datetime64('2025-01-31 10:45'),
              'COB109':np.datetime64('2025-02-01 12:23'),
             }

xray_10    = {'COB122':np.datetime64('2025-01-13 17:38'),
              'COB119':np.datetime64('2025-01-17 17:38'),
              'COB118':np.datetime64('2025-01-20 17:21'),
              'COB114':np.datetime64('2025-01-23 18:23'),
              'COB121':np.datetime64('2025-01-27 16:32'),
              'COB903':np.datetime64('2025-01-29 19:43'),
              'COB901':np.datetime64('2025-01-31 12:30'),
              'COB109':np.datetime64('2025-02-01 14:23'),
             }

xray_50    = {'COB122':np.datetime64('2025-01-13 21:14'),
              'COB119':np.datetime64('2025-01-18 00:18'),
              'COB118':np.datetime64('2025-01-20 23:19'),
              'COB114':np.datetime64('2025-01-23 22:01'),
              'COB121':np.datetime64('2025-01-27 19:32'),
              'COB903':np.datetime64('2025-01-29 22:50'),
              'COB901':np.datetime64('2025-01-31 15:43'),
              'COB109':np.datetime64('2025-02-01 16:30'),
             }

xray_Off   = {'COB122':np.datetime64('2025-01-17 06:42'),
              'COB119':np.datetime64('2025-01-20 12:32'),
              'COB118':np.datetime64('2025-01-23 12:19'),
              'COB114':np.datetime64('2025-01-26 18:23'),
              'COB121':np.datetime64('2025-01-29 14:38'),
              'COB903':np.datetime64('2025-01-31 09:23'),
              'COB901':np.datetime64('2025-02-01 10:57'),
              'COB109':np.datetime64('2025-02-03 08:12'),
             }

startTime = {'COB122':np.datetime64('2025-01-13 12:00'),
             'COB119':np.datetime64('2025-01-17 12:00'),
             'COB118':np.datetime64('2025-01-20 12:00'),
             'COB114':np.datetime64('2025-01-23 12:00'),
             'COB121':np.datetime64('2025-01-27 10:00'),
             'COB903':np.datetime64('2025-01-29 16:00'),
             'COB901':np.datetime64('2025-01-31 10:00'),
             'COB109':np.datetime64('2025-02-01 10:00'),
            }

zoomTime   = {'COB122':np.datetime64('2025-01-13 22:14'),
              'COB119':np.datetime64('2025-01-18 01:18'),
              'COB118':np.datetime64('2025-01-21 00:19'),
              'COB114':np.datetime64('2025-01-23 23:01'),
              'COB121':np.datetime64('2025-01-27 20:32'),
              'COB903':np.datetime64('2025-01-29 23:50'),
              'COB901':np.datetime64('2025-01-31 16:43'),
              'COB109':np.datetime64('2025-02-01 17:30'),
             }


_d10 = 1.83259677
_d50 = 9.16298385

tid_xlims = {}

for _COB_ in xray_10:
    _start_TID = (startTime[_COB_]-xray_10[_COB_]).astype('timedelta64[m]').astype('float')/60.*_d10
    _tot_10 = (xray_50[_COB_]-xray_10[_COB_]).astype('timedelta64[m]').astype('float')/60.*_d10
    zoomedLim = (zoomTime[_COB_]-xray_50[_COB_]).astype('timedelta64[m]').astype('float')/60.*_d50 + _tot_10
    tid_xlims[_COB_] = (_start_TID, zoomedLim)


def convertTimestampToTID(d_tot,bist_result,_COB_):

    if _COB_=='COB122':
        _xray_10 = xray_10[_COB_]
        _xray_50 = xray_50[_COB_]
        _xray_pause   = np.datetime64('2025-01-13 22:51')
        _xray_restart = np.datetime64('2025-01-14 11:55')
        _xray_Off = xray_Off[_COB_]

        _tot_10 = (_xray_50 - _xray_10).astype('timedelta64[m]').astype('float')/60.*_d10
        _tot_pause = (_xray_pause-_xray_50).astype('timedelta64[m]').astype('float')/60.*_d50
        _tot_50 = (_xray_Off-_xray_restart).astype('timedelta64[m]').astype('float')/60.*_d50

        _t = d_tot.timestamp.values

        d_tot['TID'] = np.where(_t<_xray_50,
                                (_t - _xray_10).astype('timedelta64[s]').astype('float')/3600.*_d10,
                                np.where(_t<_xray_pause,
                                         (_t - xray_50[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10,
                                         np.where(_t<_xray_restart,
                                                  _tot_10 + _tot_pause,
                                                  np.where(_t<_xray_Off,
                                                           (_t - _xray_restart).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10 + _tot_pause,
                                                           _tot_10 + _tot_pause + _tot_50
                                                          )
                                                 )
                                        )
                               )


        d_tot['XRay_current'] = np.where(_t<_xray_10,
                                         0,
                                         np.where(_t<_xray_50,
                                                  10,
                                                  np.where(_t<_xray_pause,
                                                           50,
                                                           np.where(_t<_xray_restart,
                                                                   0,
                                                                   np.where(_t<_xray_Off,
                                                                           50,
                                                                           0)
                                                                   )
                                                          )
                                                 )
                                        )

        d_tot['XRay_On'] = d_tot['XRay_current']>0


        _t = bist_result.timestamps.values

        bist_result['TID'] = np.where(_t<_xray_50,
                                      (_t - _xray_10).astype('timedelta64[s]').astype('float')/3600.*_d10,
                                      np.where(_t<_xray_pause,
                                               (_t - xray_50[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10,
                                               np.where(_t<_xray_restart,
                                                        _tot_10 + _tot_pause,
                                                        np.where(_t<_xray_Off,
                                                                 (_t - _xray_restart).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10 + _tot_pause,
                                                                 _tot_10 + _tot_pause + _tot_50
                                                                )
                                                       )
                                              )
                                     )


        bist_result['XRay_current'] = np.where(_t<_xray_10,
                                               0,
                                               np.where(_t<_xray_50,
                                                        10,
                                                        np.where(_t<_xray_pause,
                                                                 50,
                                                                 np.where(_t<_xray_restart,
                                                                          0,
                                                                          np.where(_t<_xray_Off,
                                                                                   50,
                                                                                   0)
                                                                         )
                                                                )
                                                       )
                                              )

    else:
        _tot_10 = (xray_50[_COB_]-xray_10[_COB_]).astype('timedelta64[m]').astype('float')/60.*_d10
        _tot_50 = (xray_Off[_COB_]-xray_50[_COB_]).astype('timedelta64[m]').astype('float')/60.*_d50

        _t = d_tot.timestamp.values

        d_tot['TID'] = np.where(_t<xray_50[_COB_],
                                (_t - xray_10[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d10,
                                np.where(_t<xray_Off[_COB_],
                                         (_t - xray_50[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10,
                                         _tot_10+_tot_50)
                               )

        d_tot['XRay_current'] = np.where(_t<xray_10[_COB_],
                                         0,
                                         np.where(_t<xray_50[_COB_],
                                                  10,
                                                  np.where(_t<xray_Off[_COB_],
                                                           50,
                                                           0)
                                                 )
                                        )
        d_tot['XRay_On'] = d_tot['XRay_current']>0


        _t = bist_result.timestamps.values

        bist_result['TID'] = np.where(_t<xray_50[_COB_],
                                      (_t - xray_10[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d10,
                                      np.where(_t<xray_Off[_COB_],
                                               (_t - xray_50[_COB_]).astype('timedelta64[s]').astype('float')/3600.*_d50 + _tot_10,
                                               _tot_10+_tot_50)
                                     )
        bist_result['XRay_current'] = np.where(_t<xray_10[_COB_],
                                               0,
                                               np.where(_t<xray_50[_COB_],
                                                        10,
                                                        np.where(_t<xray_Off[_COB_],
                                                                 50,
                                                                 0)
                                                       )
                                              )

def draw_legend(ax,leg_loc):
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

def mark_TID_times(ax,cob,leg_loc=None):
    _xlim = np.array(num2date(ax.get_xlim())).astype(np.datetime64)#(num2date(ax.get_xlim()))
    _ylim = ax.get_ylim()
    _chiller_on = chiller_on[cob]
    _xray_10    = xray_10[cob]
    _xray_50    = xray_50[cob]
    _xray_Off   = xray_Off[cob]
    
    
    if (_xlim[0]<_chiller_on) and (_chiller_on<_xlim[1]):
        ax.vlines(_chiller_on,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')    
    if (_xlim[0]<_xray_10) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')    
    if (_xlim[0]<_xray_50) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')    
    if (_xlim[0]<_xray_Off) and (_xray_Off<_xlim[1]):
        ax.vlines(_xray_Off,_ylim[0], _ylim[1],linestyles='dashed',color='black',linewidth=3,label='X-rays Off')    

    if cob=='COB122' and ((_xlim[0]<np.datetime64('2025-01-13 22:51')) | (_xlim[1]>np.datetime64('2025-01-14 11:55'))):
        ax.add_patch(Rectangle((np.datetime64('2025-01-13 22:51'), _ylim[0]), np.datetime64('2025-01-14 11:55')-np.datetime64('2025-01-13 22:51'), _ylim[1]-_ylim[0],alpha=.2,label='X-rays Paused'))
    
    draw_legend(ax,leg_loc)

def mark_TID_times_TID(ax,cob,leg_loc=None):
    _chiller = (chiller_on[cob]-xray_10[cob]).astype('timedelta64[m]').astype('float')/60.*_d10
    _xray_10 = 0
    _xray_50 = (xray_50[cob]-xray_10[cob]).astype('timedelta64[m]').astype('float')/60.*_d10
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()

    if (_chiller>_xlim[0]) and (_chiller<_xlim[1]):
         ax.vlines(_chiller,_ylim[0], _ylim[1],linestyles='dashed',color='blue',linewidth=3,label='Chiller On')    
    if (_xray_10>_xlim[0]) and (_xray_10<_xlim[1]):
         ax.vlines(_xray_10,_ylim[0], _ylim[1],linestyles='dashed',color='green',linewidth=3,label='X-ray on 10 mA')    
    if (_xray_50>_xlim[0]) and (_xray_50<_xlim[1]):
         ax.vlines(_xray_50,_ylim[0], _ylim[1],linestyles='dashed',color='red',linewidth=3,label='X-ray on 50 mA')    
        
    draw_legend(ax,leg_loc)
    
def plot_error_rate(d_tot,
                    voltages,
                    numerator,
                    denominator,
                    cob,
                    title='Error Rate',
                    ylabel='Error Rate',
                    xlabel='Time',
                    axis=None,
                    logy=False,
                    bist=False,
                    temperature=False,
                    smooth=False,
                    scatterplot=False,
                    markerstyle=None,
                    leg_offset='right+',
                    xlim = None):
    if axis is None:
        fig,ax = plt.subplots(1,1)
    else:
        ax = axis
    for v in voltages:
        d = d_tot.loc[v]
        e_rate = d[numerator].sum(axis=1)/d[denominator].sum(axis=1)
        if smooth:
            e_rate = scipy.signal.savgol_filter(e_rate,5,0)
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
        ax.plot(bist_result.timestamps.iloc[0],.1,color='black',label='Temperature',linestyle='dashed')
        ax2.set_ylabel('Temperature')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim is None:
        ax.set_xlim(startTime[cob])
    else:
        ax.set_xlim(xlim[0],xlim[1])
        
    ax.tick_params(axis='x',labelrotation=45)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(12))
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')

    mark_TID_times(ax,cob,leg_offset)

    if axis is None:
        return fig,ax
    else:
        return ax
    

def plot_error_rate_TID(d_tot,
                        voltages,
                        numerator,
                        denominator,
                        cob,
                        title='Error Rate',
                        ylabel='Error Rate',
                        xlabel='TID (MRad)',
                        axis=None,        
                        logy=False,
                        smooth=False,
                        scatterplot=False,
                        markerstyle=None,
                        leg_offset='right+',
                        xlim = None):
    if axis is None:
        fig,ax = plt.subplots(1,1)
    else:
        ax = axis
    for v in voltages:
        d = d_tot.loc[v]
        d = d.loc[d.timestamp < xray_Off[cob]]
        e_rate = d[numerator].sum(axis=1)/d[denominator].sum(axis=1)
        if smooth:
            e_rate = scipy.signal.savgol_filter(e_rate,5,0)
        if scatterplot:
            ax.scatter(d.TID,e_rate, label=f'{v:.02f}V')
        else:
            ax.plot(d.TID,e_rate, label=f'{v:.02f}V',marker=markerstyle)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if not xlim is None:
        ax.set_xlim(xlim[0],xlim[1])
        
#    ax.tick_params(axis='x',labelrotation=45)

#    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(12))
    ax.set_title(title)
    if logy:
        ax.set_yscale('log')

    mark_TID_times_TID(ax,cob,leg_offset)

    if axis is None:
        return fig,ax
    else:
        return ax
    
def plotCurrentAndTemp(d_tot,_COB_):
    _startTime = startTime[_COB_]
    fig,ax = plt.subplots(1,1)
    d = d_tot.loc[1.08]
    ax.plot(d.timestamp,d.current,color='red',label='1.08 V',linestyle='dashed')
    d = d_tot.loc[1.2]
    ax.plot(d.timestamp,d.current,color='black',label='1.20 V',linestyle='dashed')
    d = d_tot.loc[1.32]
    ax.plot(d.timestamp,d.current,color='green',label='1.32 V',linestyle='dashed')

    ax.set_ylabel('Current')
    ax.set_xlim(_startTime,None)
    ax.set_ylim(0.25,0.425)
    ax.tick_params(labelrotation=45)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(6))
    ax.set_title(f'Current, {_COB_}')

    mark_TID_times(ax,_COB_,'right+2')

    fig.savefig(f'plots/{_COB_}/Currents_{_COB_}.pdf', bbox_inches='tight')

    fig,ax = plt.subplots(1,1)
    d = d_tot.loc[1.08]
    ax.plot(d.timestamp,d.temperature,color='red',label='1.08 V',linestyle='dashed')
    d = d_tot.loc[1.2]
    ax.plot(d.timestamp,d.temperature,color='black',label='1.20 V',linestyle='dashed')
    d = d_tot.loc[1.32]
    ax.plot(d.timestamp,d.temperature,color='green',label='1.32 V',linestyle='dashed')

    ax.set_ylabel('Temperature')
    ax.set_xlim(_startTime,None)
    ax.tick_params(labelrotation=45)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(6))
    ax.set_title(f'Temperature, {_COB_}')
    mark_TID_times(ax,_COB_,'right+2')
    fig.savefig(f'plots/{_COB_}/Temperatures_{_COB_}.pdf', bbox_inches='tight')
    plt.close(fig)
    
def histErrorSources(d_tot,_COB_):
    _max_TID = d_tot[d_tot.XRay_current>0].TID.max()
    for v in d_tot.index.unique():
        d = d_tot.loc[v]
        cut = (d.TID<d.TID.values[-1]) & (d.TID>=0) & (d.XRay_current>0)
        single_single = d.isSingleError_SingleBit
        single_multi = d.isSingleError_MultiBit
        ob = d.isOBErrors
        badOB = d.isBadOBError
        CRC = d.likelyInputCRCError
        PP = d.likelyPPError
        total_p = d.n_packets
        total = d.n_captured_packets
        other = total_p-CRC-PP-badOB-ob-single_multi-single_single
        bad_parsing = ((total_p==1) & (d.n_captured_bx.values==4095)).astype(int)

        tid_bins = np.array(d.TID[cut].values.tolist() + [d.TID.values[-1]])-0.1

        plt.figure()
        # plt.hist(d.TID[cut],weights=total[cut],bins=tid_bins);
        plt.hist(d.TID[cut],weights=((single_single+single_multi+ob+badOB+PP+CRC+other)/total)[cut],bins=tid_bins,label='Other');
        plt.hist(d.TID[cut],weights=((single_single+single_multi+ob+badOB+PP+CRC)/total)[cut],bins=tid_bins,label='Input CRC');
        plt.hist(d.TID[cut],weights=((single_single+single_multi+ob+badOB+PP)/total)[cut],bins=tid_bins,label='PingPong Errors');
        plt.hist(d.TID[cut],weights=((single_single+single_multi+ob+badOB)/total)[cut],bins=tid_bins,label='Bad OB Errors');
        plt.hist(d.TID[cut],weights=((single_single+single_multi+ob)/total)[cut],bins=tid_bins,label='12-word spacing');
        plt.hist(d.TID[cut],weights=((single_single+single_multi)/total)[cut],bins=tid_bins,label='Single Word Multibit');
        plt.hist(d.TID[cut],weights=((single_single)/total)[cut],bins=tid_bins,label='Single Word Singlebit');
        if bad_parsing[cut].sum()>0:
            plt.hist(d.TID[cut],weights=bad_parsing[cut],bins=tid_bins,label='Data Parsing Issues',color='white',hatch='///');

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        plt.title(f'Error Sources, {_COB_}, V={v}')
        plt.ylabel('Error Rate (fraction of packets)')
        plt.xlabel('TID');
        plt.xlim(None,_max_TID*1.05)

        plt.savefig(f'plots/{_COB_}/ErrorSources/ErrorSource_{_COB_}_V_{v:.2f}'.replace('.','_')+'.pdf', bbox_inches='tight')

        plt.ylim(3e-8,1.5)
        plt.yscale('log')
        plt.savefig(f'plots/{_COB_}/ErrorSources/log/ErrorSource_{_COB_}_V_{v:.2f}'.replace('.','_')+'_log.pdf', bbox_inches='tight')    
        plt.close()


def makeStandardPlots(d_tot,voltages,_COB_,_xlim,_zoomed=False, useTID=False, _smoothed=False):
    extra = ''
    if useTID:
        plotFunction = plot_error_rate_TID
        extra += '_TID'
    else:
        plotFunction = plot_error_rate
    if _zoomed:
        extra += '_TurnOn'
    if _smoothed:
        extra += '_Smoothed'
    fig,ax = plotFunction(d_tot,voltages,
                             numerator=['error_count'],denominator=['word_count'],
                             ylabel='Error Rate (% of BX)',
                             title=f'Total Error Rate {_COB_}',
                             logy=True,
                             leg_offset='right',
                             cob=_COB_,
                             xlim=_xlim,
                             smooth=_smoothed,
                            );
    fig.savefig(f'plots/{_COB_}/TotalErrorRate_{_COB_}{extra}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plotFunction(d_tot,voltages,
                             numerator=['n_packets'],denominator=['n_captured_packets'],
                             ylabel='Error Rate (% of Packets)',
                             title=f'Total Packet Errors {_COB_}',
                             logy=True,
                             leg_offset='right',
                             cob=_COB_,
                             xlim=_xlim,
                             smooth=_smoothed,
                            );
    fig.savefig(f'plots/{_COB_}/PacketErrorRate_{_COB_}{extra}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plotFunction(d_tot,voltages,
                             numerator=['isOBErrors'],denominator=['n_captured_packets'],
                             ylabel='Error Rate (% of Packets)',
                             title=f'OB (12-word spacing) Errors {_COB_}',
                             logy=True,
                             leg_offset='right',
                             cob=_COB_,
                             xlim=_xlim,
                             smooth=_smoothed,
                            );
    fig.savefig(f'plots/{_COB_}/OBErrorRate_{_COB_}{extra}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plotFunction(d_tot,voltages,
                             numerator=['isOBErrors','isSingleError'],denominator=['n_captured_packets'],
                             ylabel='Error Rate (% of Packets)',
                             title=f'OB or Single Errors {_COB_}',
                             logy=True,
                             leg_offset='right',
                             cob=_COB_,
                             xlim=_xlim,
                             smooth=_smoothed,
                            );
    fig.savefig(f'plots/{_COB_}/OBorSingleErrorRate_{_COB_}{extra}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plotFunction(d_tot,voltages,
                             numerator=['likelyPPError'],denominator=['n_captured_packets'],
                             ylabel='Error Rate (% of Packets)',
                             title=f'PP Errors {_COB_}',
                             logy=True,
                             leg_offset='right',
                             cob=_COB_,
                             xlim=_xlim,
                             smooth=_smoothed,
                            );
    fig.savefig(f'plots/{_COB_}/PPErrorRate_{_COB_}{extra}.pdf', bbox_inches='tight')
    plt.close(fig)


def plotBist(bist_result,_COB_,_startTime,_zoomTime):
    fig,ax = plt.subplots(1,1)

    ax.plot(bist_result.timestamps,bist_result.pp_passing_v,label='PP BIST',linewidth=2)
    ax.plot(bist_result.timestamps,bist_result.ob_passing_v,label='OB BIST',linewidth=2)
    ax.set_ylabel('Voltage')
    ax.set_xlabel('Time')
    ax.set_title(f'Lowest BIST Passing Voltage {_COB_}')
    ax.tick_params(labelrotation=45)
    ax.set_xlim(_startTime,None)

    mark_TID_times(ax,_COB_,'right+0')

    fig.savefig(f'plots/{_COB_}/BIST_{_COB_}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plt.subplots(1,1)

    ax.plot(bist_result.timestamps,bist_result.pp_passing_v,label='PP BIST',linewidth=2)
    ax.plot(bist_result.timestamps,bist_result.ob_passing_v,label='OB BIST',linewidth=2)
    ax.set_ylabel('Voltage')
    ax.set_xlabel('Time')
    ax.set_title(f'Lowest BIST Passing Voltage {_COB_}')
    ax.tick_params(labelrotation=45)
    ax.set_xlim(_startTime,_zoomTime)

    mark_TID_times(ax,_COB_,'right+0')

    fig.savefig(f'plots/{_COB_}/BIST_{_COB_}_TurnOn.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plt.subplots(1,1)

    ax.plot(bist_result.TID,bist_result.pp_passing_v,label='PP BIST',linewidth=2)
    ax.plot(bist_result.TID,bist_result.ob_passing_v,label='OB BIST',linewidth=2)
    ax.set_ylabel('Voltage')
    ax.set_xlabel('TID (MRad)')
    ax.set_title(f'Lowest BIST Passing Voltage {_COB_}')
    ax.tick_params(labelrotation=45)

    mark_TID_times_TID(ax,_COB_,'right+0')
    fig.savefig(f'plots/{_COB_}/BISTvsTID_{_COB_}.pdf', bbox_inches='tight')
    plt.close(fig)

    fig,ax = plt.subplots(1,1)

    ax.plot(bist_result.TID,bist_result.pp_passing_v,label='PP BIST',linewidth=2)
    ax.plot(bist_result.TID,bist_result.ob_passing_v,label='OB BIST',linewidth=2)
    ax.set_ylabel('Voltage')
    ax.set_xlabel('TID (MRad)')
    ax.set_title(f'Lowest BIST Passing Voltage {_COB_}')
    ax.tick_params(labelrotation=45)
    ax.set_xlim(-2,15)

    mark_TID_times_TID(ax,_COB_,'right+0')
    fig.savefig(f'plots/{_COB_}/BISTvsTID_{_COB_}_TurnOn.pdf', bbox_inches='tight')
    plt.close(fig)

    
def plotErrorFreeVoltages(d_tot, 
                          cob,
                          outputName = None,
                          rates = [0,.1,.01,.001,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                          numerator = ['error_count'], 
                          denominator=['word_count'],
                          title=None,
                          useTID=False,
                          turnOn=False,
                          voltage_smoothing=1,
                          rate_smoothing=1,
                         ):
    
    d=d_tot[['file','timestamp','TID']+numerator+denominator].copy()
    d['error_rate'] = d[numerator].sum(axis=1)/d[denominator].sum(axis=1)

    for v in d.index.unique():
        d.loc[v,'error_rate'] = scipy.signal.savgol_filter(d.loc[v,'error_rate'],rate_smoothing,0)

    d['lowest_error_free']     = np.where(d.error_rate>0   , 99,d.index)
    d['lowest_error_free_1e8'] = np.where(d.error_rate>1e-8, 99,d.index)
    d['lowest_error_free_1e7'] = np.where(d.error_rate>1e-7, 99,d.index)
    d['lowest_error_free_1e6'] = np.where(d.error_rate>1e-6, 99,d.index)
    d['lowest_error_free_1e5'] = np.where(d.error_rate>1e-5, 99,d.index)
    d['lowest_error_free_1e4'] = np.where(d.error_rate>1e-4, 99,d.index)
    d['lowest_error_free_1e3'] = np.where(d.error_rate>0.001,99,d.index)
    d['lowest_error_free_1e2'] = np.where(d.error_rate>0.01, 99,d.index)
    d['lowest_error_free_1e1'] = np.where(d.error_rate>0.1,  99,d.index)
    
    cols = [x for x in d.columns if x.startswith('lowest_error_free')]

    x=d.groupby('file')[cols + ['timestamp','TID']].min()
    for c in cols:
        x.loc[x[c]==99,c] = np.nan
    fig,ax = plt.subplots(1,1)
    
    if useTID:
        x_vals = x.TID
    else:
        x_vals = x.timestamp
        
    if 0 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free       ,voltage_smoothing,0),label='Error Free',linewidth=2)
    if 1e-8 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e8,voltage_smoothing,0),label='Error < $10^{-8}$',linewidth=2)
    if 1e-7 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e7,voltage_smoothing,0),label='Error < $10^{-7}$',linewidth=2)
    if 1e-6 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e6,voltage_smoothing,0),label='Error < $10^{-6}$',linewidth=2)
    if 1e-5 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e5,voltage_smoothing,0),label='Error < $10^{-5}$',linewidth=2)
    if 1e-4 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e4,voltage_smoothing,0),label='Error < $10^{-4}$',linewidth=2)
    if 1e-3 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e3,voltage_smoothing,0),label='Error < $10^{-3}$',linewidth=2)
    if 1e-2 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e2,voltage_smoothing,0),label='Error < $10^{-2}$',linewidth=2)
    if 1e-1 in rates: ax.plot(x_vals,scipy.signal.savgol_filter(x.lowest_error_free_1e1,voltage_smoothing,0),label='Error < $10^{-1}$',linewidth=2)

    if title is None:
        ax.set_title(f'Lowest Error Free Voltage, {cob}')
    else:
        ax.set_title(title)
    ax.set_ylabel('Voltage')

    if useTID:
        ax.set_xlabel('TID')
        if turnOn:
            ax.set_xlim(tid_xlims[cob][0],tid_xlims[cob][1])
        else:
            ax.set_xlim(tid_xlims[cob][0],None)
        mark_TID_times_TID(ax,cob,'right+0')
    else:
        ax.set_xlabel('Time')
        ax.tick_params(labelrotation=45)
        if turnOn:
            ax.set_xlim(startTime[cob],zoomTime[cob])
        else:
            ax.set_xlim(startTime[cob],None)
        mark_TID_times(ax,cob,'right+0')
    
    if outputName:
        fig.savefig(outputName, bbox_inches='tight')
    plt.close(fig)

def makeTemperaturePlots(packets=False):
    rates = [0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
    labels = {0:'Error Free',
              1e-8:'Error < $10^{-8}$',
              1e-7:'Error < $10^{-7}$',
              1e-6:'Error < $10^{-6}$',
              1e-5:'Error < $10^{-5}$',
              1e-4:'Error < $10^{-4}$',
              1e-3:'Error < $10^{-3}$',
              1e-2:'Error < $10^{-2}$',
              1e-1:'Error < $10^{-1}$',
             }
    summary_results = {}
    for _COB_ in startTime.keys():
        summary_results[_COB_] = {}
        plt.figure()
        d_tot = pd.read_csv(f'../../TestData/FullData/totals_{_COB_}.csv',index_col='voltages')
        d_tot.timestamp = pd.to_datetime(d_tot.timestamp)
        voltages = d_tot.index.unique().values
        voltages.sort()
        
        for v in voltages:
            d = d_tot.loc[v]
            cut = (d.TID<0) & (d.timestamp>chiller_on[_COB_]) & (d.temperature>-18)# & (d.temperature<25)
            if packets:
                if (d.n_packets[cut]>0).any():
                    plt.plot(d.temperature[cut],(d.n_packets/d.n_captured_packets)[cut],marker='o',label=f'V={v:.2f}')
            else:
                if (d.error_count[cut]>0).any():
                    plt.plot(d.temperature[cut],(d.error_count/d.word_count)[cut],marker='o',label=f'V={v:.2f}')
        plt.yscale('log')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        plt.ylabel('Error Rate')
        plt.xlabel('Temperature')
        plt.title(f'Error Rate During Cooldown, {_COB_}')
        plt.grid(axis='y')
        plt.savefig(f'plots/{_COB_}/ErrorRateVsTemperature_Packets_{_COB_}.pdf', bbox_inches='tight')
        plt.close()
        
        plt.figure()
        for r in rates:
            if r>0:
                _rate = f"< {r:.0e}".replace('1e-0','$10^{-')+'}$'
            else:
                _rate='Error Free'
            _temperatures = []
            _voltages = []
            for v in voltages:
                d = d_tot.loc[v]
                #get only entries after chiller is on, but before TID has started
                cut = (d.TID<0) & (d.timestamp>chiller_on[_COB_]) & (d.temperature>-18)
                d = d[cut]
                if packets:
                    e_rate = (d.n_packets/d.n_captured_packets).values
                else:
                    e_rate = (d.error_count/d.word_count).values

                temps = d.temperature.values
                #find the point where the line crosses the specific error rate
                # get slope based on points before/after crossing, linearly interpolate between
                x=(e_rate<=r)
                crossing_idx = np.argwhere(x & np.roll(~x,-1)).flatten()
                if len(crossing_idx)>0:
                    i = crossing_idx[0]
                    if i==len(e_rate)-1:
                        continue
                    m=(e_rate[i+1] - e_rate[i])/(temps[i+1]-temps[i])
                    _t = (r-e_rate[i])/m + temps[i]
                    _voltages.append(v)
                    _temperatures.append(_t)
            plt.plot(_temperatures,_voltages,marker='o',label=_rate)
            summary_results[_COB_][r] = {'temperature':_temperatures,'voltages':_voltages}
        plt.legend()
        plt.ylabel('Voltage')
        plt.xlabel('Temperature')
        if packets:
            plt.title(f'{_COB_}, Packet Error Rate')
        else:
            plt.title(f'{_COB_}, Error Rate')
            

        if packets:
            fname = f'plots/{_COB_}/ErrorRateVoltages_Temperature_Packets_{_COB_}.pdf'
        else:
            fname = f'plots/{_COB_}/ErrorRateVoltages_Temperature_{_COB_}.pdf'
        
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
    for c in summary_results:
        for r in summary_results[c]:
            for k in summary_results[c][r]:
                summary_results[c][r][k] = np.array(summary_results[c][r][k])
                
    fit_results = {}
    for r in [1e-6,1e-5,1e-4,1e-3,1e-2]:
        _rate = f"Error Rate < {r:.0e}".replace('1e-0','$10^{-')+'}$'
        _rate_fname = f"{r:.0e}".replace('e-0','e')
        plt.figure()
        for c in summary_results_bx:
            plt.plot(summary_results[c][r]['temperature'],summary_results[c][r]['voltages'],marker='o',label=c)
        plt.ylabel('Voltage')
        plt.xlabel('Temperature')
        plt.title(_rate)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        outputdir = 'Temperature_TotalErrors'
        _extra = ''
        if packets:
            outputdir = 'Temperature_PacketErrors'
            _extra = 'Packets_'
        plt.savefig(f'plots/ErrorRateSummary/{outputdir}/VoltageVsTemperature_{_extra}{_rate_fname}.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()

        fit_results[r] = {}
        for i,c in enumerate(summary_results_bx.keys()):
            x,y=summary_results_bx[c][r]['temperature'],summary_results_bx[c][r]['voltages']

            end=None
            if c=='COB118' and r==1e-6:
                end=-1
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[:end],y[:end])
            fit_results[r][c] = slope

            if np.isnan(slope):
                slopeText = ''
            else:
                slopeText = f', {slope*1000:.2f} mV/˚C'
                plt.plot(_x,_x*slope+intercept,color=color_cycle[i],linestyle='dashed')
            plt.plot(x,y,marker='o',color=color_cycle[i],label=f'{c}{slopeText}')

        plt.ylabel('Voltage')
        plt.xlabel('Temperature')
        plt.title(_rate)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1.0))        
        plt.savefig(f'plots/ErrorRateSummary/{outputdir}/VoltageVsTemperature_Fit_{_extra}{_rate_fname}.pdf', bbox_inches='tight')
        plt.close()

    return summary_results, fit_results