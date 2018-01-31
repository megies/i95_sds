#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle
import numpy as np

from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.imaging.cm import viridis


def _filename_to_nanoseconds_start_of_day(filename):
    year, day_of_year = _filename_to_year_and_day_of_year(filename)
    t = UTCDateTime(year=year, julday=day_of_year)
    return t._ns


def _filename_to_year_and_day_of_year(filename):
    parts = filename.split('.')
    year = int(parts[-3])
    day_of_year = int(parts[-2])
    return (year, day_of_year)


def _filename_to_mpl_day(filename):
    year, day_of_year = _filename_to_year_and_day_of_year(filename)
    return int(date2num(UTCDateTime(year=year, julday=day_of_year).date))


def _nanoseconds_to_mpl_data(times):
    times = [UTCDateTime(ns=t).datetime for t in times]
    return date2num(times)


def _merge_stream_labels(channels):
    if len(channels) == 1:
        return channels[0]
    first = channels[0]
    if all(cha[1:] == first[1:] for cha in channels[1:]):
        return '[{}]{}'.format(
            ''.join(sorted([cha[0] for cha in channels])), first[1:])
    elif all(cha[0] == first[0] for cha in channels[1:]) and \
            all(cha[2:] == first[2:] for cha in channels[1:]):
        return '{}[{}]{}'.format(
            first[0], ''.join(sorted([cha[1] for cha in channels])), first[2:])
    else:
        return '/'.join(channels)


class I95SDSClient(object):
    def __init__(self, sds_root, merge_streams=('HH', 'EH', 'EL')):
        self.sds_root = sds_root
        self.merge_streams = merge_streams
        self.client = SDSClient(self.sds_root)
        self.client.FMTSTR = self.client.FMTSTR + '.npy'
        self.times_of_day_nanoseconds = np.load(
            os.path.join(sds_root, 'times.npy'))
        self.delta_ns = (self.times_of_day_nanoseconds[1] -
                         self.times_of_day_nanoseconds[0])
        self.delta_days = (self.delta_ns / 1e9) / (24 * 3600)
        self.dtype = np.dtype(
            [('time', np.int64), ('i95', np.float32), ('coverage', np.uint8)])
        self.len = len(self.times_of_day_nanoseconds)
        self.no_data_color = 'lightgray'
        # clip at this percentile of all I95 data,
        # to avoid single I95 spikes controlling the whole colormap
        self.vmax_clip_percentile = 98

    def _get_data(self, network, station, location, channel, starttime,
                  endtime):
        if channel[:2] in self.merge_streams:
            channels = [stream + channel[2:] for stream in self.merge_streams]
        else:
            channels = [channel]

        filenames = []
        t = starttime
        while t < endtime:
            filenames.append(
                [(cha, self.client._get_filename(
                    network, station, location, cha, t)) for cha in channels])
            t += 24 * 3600

        # this will contain tuples of (year, day of year) that were processed
        # but have no data availability
        no_data_year_day = []
        no_data = []
        # this will contain all data as a list of numpy arrays
        data = []
        # this is used for temporarily adding up and merging contiguous data
        # blocks
        data_ = []
        # print(filenames)
        used_channels = set()

        for filenames_ in filenames:
            individual_channel_data = []
            for cha, filename in filenames_:
                individual_channel_data.append(
                    (cha, self._load_npy_file(filename)))
            # exactly one channel has data: use it
            if sum([isinstance(d, np.ndarray)
                    for cha, d in individual_channel_data]) == 1:
                # we have data for exactly one channel code (of those that
                # should be combined)
                for cha, d in individual_channel_data:
                    if isinstance(d, np.ndarray):
                        used_channels.add(cha)
                        break
                else:
                    raise NotImplementedError
            # multiple channels have data: merge them
            elif sum([isinstance(d, np.ndarray)
                      for cha, d in individual_channel_data]) > 1:
                # we have data for more than one channel code (of those that
                # should be combined) so we need to combine data..
                ds = []
                for cha, d in individual_channel_data:
                    if isinstance(d, np.ndarray):
                        used_channels.add(cha)
                        ds.append(d)
                best_coverage = np.argmax(np.vstack(ds)['coverage'], axis=0)
                d = np.choose(best_coverage, ds)
            # no channel at all was processed yet.. ignore
            elif all([d is None for cha, d in individual_channel_data]):
                # ..because data was not processed, just skip,
                # leaving an empty spot in the plot
                continue
            # some channels were processed but None has data: add as gap
            elif all([d is False or d is None
                      for cha, d in individual_channel_data]):
                # ..because data was processed, but there was no waveforms,
                # so this is definitely a gap in the waveforms
                no_data_year_day.append(
                    _filename_to_year_and_day_of_year(filename))
                continue
            else:
                raise NotImplementedError

            # we have no data
            if d is None:
                # ..because data was not processed, just skip,
                # leaving an empty spot in the plot
                continue
            elif d is False:
                # ..because data was processed, but there was no waveforms,
                # so this is definitely a gap in the waveforms
                no_data_year_day.append(
                    _filename_to_year_and_day_of_year(filename))
                continue
            # we have data
            if not data_:
                data_.append(d)
                continue
            if d['time'][0] == data_[-1]['time'][-1] + self.delta_ns:
                data_.append(d)
                continue
            data.append(np.concatenate(data_))
            data_ = [d]
        data.append(np.concatenate(data_))
        # setup no data parts
        start = None
        end = None
        seconds_per_day = 24 * 3600
        for year, day in sorted(no_data_year_day):
            t = UTCDateTime(year=year, julday=day)
            if start is None:
                start = t
                end = t + seconds_per_day
                continue
            # extend by one day if contiguous
            if t == end:
                end += seconds_per_day
                continue
            # otherwise start a new timespan
            no_data.append((start, end))
            start = t
            end = t + seconds_per_day
        # add last timespan, if any
        if start is not None:
            no_data.append((start, end))
        # print("nodata: ", no_data)
        # print("nodata tuples: ", no_data_year_day)
        # import pdb; pdb.set_trace()
        return data, no_data, sorted(used_channels)

    def _load_npy_file(self, filename):
        if not os.path.exists(filename):
            return None
        if os.path.getsize(filename) == 1:
            return False
        data_ = np.load(filename)
        data = np.empty(self.len, dtype=self.dtype)
        data['time'] = (self.times_of_day_nanoseconds +
                        _filename_to_nanoseconds_start_of_day(filename))
        for key in ('i95', 'coverage'):
            data[key] = data_[key]
        mask = np.zeros_like(data, dtype=np.bool)
        mask |= np.isnan(data['i95'])
        mask |= data['coverage'] == 0
        data = np.ma.masked_array(data, mask=mask)
        # print(data)
        return data

    def plot_all_data(self, starttime, endtime, cmap=None, show=True,
                      global_norm=False):
        nslc = [('BW', 'MANZ', '', 'HHZ'),
                ('BW', 'MROB', '', 'HHZ'),
                ('BW', 'VIEL', '', 'HHZ')]

        nslc = sorted(nslc)

        all_data = []
        for net, sta, loc, cha in nslc:
            data, no_data, used_channels = self._get_data(
                net, sta, loc, cha, starttime, endtime)
            all_data.append((data, no_data, used_channels))

        # plotting of data parts
        xmin_global = np.inf
        xmax_global = -np.inf
        vmin_global = np.inf
        vmax_global = -np.inf
        fig, axes = plt.subplots(nrows=len(nslc), sharex=True)
        for ax, (net, sta, loc, cha), (data, no_data, used_channels) in zip(
                axes, nslc, all_data):
            vmin, vmax, xmin, xmax, _ = self._plot(
                ax, data, no_data, used_channels, net, sta, loc, cmap=cmap,
                colorbar=not global_norm)
            vmin_global = min(vmin_global, vmin)
            vmax_global = max(vmax_global, vmax)
            xmin_global = min(xmin_global, xmin)
            xmax_global = max(xmax_global, xmax)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.02)
        if global_norm:
            norm = Normalize(vmin_global, vmax_global)
            for ax in axes:
                for im in ax.images:
                    im.set_norm(norm)
            cb = plt.colorbar(mappable=im, ax=axes)
            cb.set_label('I95 [nm/s]')
        fig.canvas.draw_idle()
        if show:
            plt.show()
        return fig, ax

    def _plot(self, ax, data, no_data, used_channels, network, station,
              location, cmap=None, colorbar=True):
        vmin = np.inf
        vmax = -np.inf
        for d in data:
            vmax_ = np.nanmax(d['i95'])
            if not np.isnan(vmax_):
                vmax = max(vmax, vmax_)
            vmin_ = np.nanmin(d['i95'])
            if not np.isnan(vmin_):
                vmin = min(vmin, vmin_)
        # clip vmax at given percentile
        vmax = np.nanpercentile(np.concatenate([d['i95'] for d in data]),
                                q=self.vmax_clip_percentile)
        # print(vmin, vmax)
        cmap = cmap or viridis
        cmap.set_bad(self.no_data_color)
        for d in data:
            self._plot_data(ax, d, vmin, vmax, cmap)
        if colorbar:
            cb = plt.colorbar(mappable=ax.images[-1], ax=ax)
            cb.set_label('I95 [nm/s]')
        else:
            cb = None
        # plotting of "no data" parts
        for start, end in no_data:
            self._plot_no_data(ax, start, end)
        # fig/ax tweaks
        ax.xaxis_date()
        xmin = min(im.get_extent()[0] for im in ax.images)
        xmax = max(im.get_extent()[1] for im in ax.images)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks([])
        channel = _merge_stream_labels(used_channels)
        label = '.'.join([network, station, location, channel])
        ax.set_ylabel(label, fontdict={'family': 'monospace'})
        return vmin, vmax, xmin, xmax, cb

    def plot(self, network, station, location, channel, starttime, endtime,
             cmap=None, show=True):
        data, no_data, used_channels = self._get_data(
            network, station, location, channel, starttime, endtime)

        # plotting of data parts
        fig, ax = plt.subplots()
        self._plot(ax, data, no_data, used_channels, network, station,
                   location, cmap=cmap)
        fig.autofmt_xdate()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _plot_data(self, ax, data, vmin, vmax, cmap):
        half_delta = self.delta_days / 2.0
        start = date2num(UTCDateTime(ns=data['time'][0]).datetime) - half_delta
        end = date2num(UTCDateTime(ns=data['time'][-1]).datetime) + half_delta
        # print(start)
        # print(end)
        # print(half_delta)
        ax.imshow(np.atleast_2d(data['i95']), extent=[start, end, 0, 1],
                  vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest',
                  aspect='auto')

    def _plot_no_data(self, ax, starttime, endtime):
        start = date2num(starttime.datetime)
        delta_days = (endtime - starttime) / (24 * 3600)
        patch = Rectangle((start, 0.0), width=delta_days, height=1,
                          color=self.no_data_color)
        ax.add_patch(patch)

    def _get_availability(self, starttime, endtime, fast=True):
        if not fast:
            raise NotImplementedError

        nslc = self.client.get_all_nslc()
        data = {}

        start_day = int(date2num(starttime.date))
        end_day = int(date2num(endtime.date))
        num_days = end_day - start_day

        for net, sta, loc, cha in nslc:
            filenames = self.client._get_filenames(
                net, sta, loc, cha, starttime, endtime)
            data_ = np.empty(num_days, dtype=np.int8)
            data_.fill(-1)
            for filename in filenames:
                index = _filename_to_mpl_day(filename) - start_day
                try:
                    filesize = os.path.getsize(filename)
                except:
                    # should not happen, as only existing filenames are
                    # returned by obspy SDS Client looks like
                    data_[index] = -1
                else:
                    if filesize > 1:
                        data_[index] = 100
                    else:
                        data_[index] = 0
            data[(net, sta, loc, cha)] = data_
        return start_day, end_day, data

    def plot_availability(self, starttime, endtime, fast=True, show=True):
        if not fast:
            raise NotImplementedError

        start_day, end_day, data = self._get_availability(
            starttime, endtime, fast=fast)

        all_labels = []
        all_data = []
        for (n, s, l, c), data in sorted(data.items()):
            all_labels.append('.'.join([n, s, l, c]))
            all_data.append(data)

        all_data = np.atleast_2d(np.vstack(all_data))
        extent = [start_day, end_day, -0.5, -0.5 + len(all_data)]
        # numpy array gets plotted from bottom to top in imshow
        all_labels = all_labels[::-1]

        # make a color map of fixed colors
        #  -1 in data means: not processed
        #   0 in data means: processed but no waveform data encountered for day
        # 100 in data means: at least partial data encountered for given day
        #                    (actual coverage data is not read in fast mode)
        cmap = ListedColormap(['lightgray', 'red', 'green'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(all_data, extent=extent, interpolation='nearest',
                  aspect='auto', cmap=cmap, norm=norm)
        ax.set_yticks(range(len(all_data)))
        ax.set_yticklabels(all_labels, fontdict={'family': 'monospace'})
        ax.xaxis_date()
        fig.autofmt_xdate()
        fig.tight_layout()

        if show:
            plt.show()
        return fig, ax


if __name__ == '__main__':

    i95_client = I95SDSClient('/bay200/I95_1-20Hz_merged')

    # example that merges data from HHZ and EHZ channels
    start = UTCDateTime(2015, 10, 15)
    end = UTCDateTime(2015, 10, 30)

    fig, ax = i95_client.plot('BW', 'MANZ', '', 'EHZ', start, end, show=False)
    plt.show()

    start = UTCDateTime(2009, 1, 1)
    end = UTCDateTime(2018, 1, 1)
    fig, ax = i95_client.plot_availability(start, end, fast=True, show=False)
    plt.show()

    start = UTCDateTime(2010, 1, 1)
    end = UTCDateTime(2017, 1, 1)

    fig, ax = i95_client.plot('BW', 'VIEL', '', 'EHZ', start, end, show=False)
    plt.show()

    start = UTCDateTime(2016, 1, 1)
    end = UTCDateTime(2017, 1, 1)

    fig, ax = i95_client.plot('BW', 'ALTM', '', 'EHZ', start, end, show=False)
    plt.show()

    # data = i95_client._load_npy_file(
    #     '/bay200/I95_1-20Hz_new/2016/BW/VIEL/EHZ.D/BW.VIEL..EHZ.D.2016.020.npy')

    availability = i95_client.get_availability(
        starttime=UTCDateTime(2010, 1, 1), endtime=UTCDateTime(2018, 1, 1),
        fast=True)

    for (n, s, l, c), data in availability.iteritems():
        print((n, s, l, c), len(data))

    all_labels = []
    all_data = []
    for (n, s, l, c), data in sorted(availability.items()):
        all_labels.append('.'.join([n, s, l, c]))
        all_data.append(data)
    print(all_data)

    # fig, ax = plt.subplots()
    # ax.imshow(np.atleast_2d(np.vstack(all_data)),
    #           extent=[date2num(UTCDateTime(2010, 1, 1).date),
    #                   date2num(UTCDateTime(2018, 1, 1).date),
    #                   -0.5, -0.5 + len(all_data)],
    #           interpolation='nearest',
    #           aspect='auto', cmap=cmap, norm=norm)
    # ax.set_yticks(range(len(all_data)))
    # ax.set_yticklabels(all_labels[::-1], fontdict={'family': 'monospace'})
    # ax.xaxis_date()
    # fig.autofmt_xdate()
    # fig.tight_layout()
    # plt.show()
