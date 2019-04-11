#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import get_cmap
from matplotlib.dates import date2num
from matplotlib.transforms import blended_transform_factory

import numpy as np

from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.imaging.cm import viridis
from obspy.imaging.util import ObsPyAutoDateFormatter


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


def _label_for_used_channels(net, sta, loc, used_channels):
    return '.'.join((net, sta, loc,
                     _merge_stream_labels(used_channels)))


def _merge_stream_labels(channels):
    channels = list(channels)
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
    def __init__(self, sds_root, merge_streams=('HH', 'EH', 'EL'),
                 vmax_clip_percentile=98, smoothing_window_length_hours=None,
                 smoothing_step_hours=None, smoothing_valid_percentage=50,
                 smoothing_percentile=95, smoothing_mean=True):
        """
        :param smoothing_percentile: Defines what percentile of the
            window used for smoothing is used as resulting value of that window
            (values from ``0`` to ``100``). If ``smoothing_mean`` is ``True``,
            then values above this percentile are discarded before calculating
            the mean.
        :param smoothing_mean: Whether or not the mean is calculated as the
            resulting value for each smoothing window. If
            ``smoothing_percentile`` is less then ``100``, then values above
            the specified percentile are excluded from the mean calculation.
        """
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
        self.vmax_clip_percentile = vmax_clip_percentile
        # use uneven number of values, so we have a symmetric stencil and end
        # up on an even time
        self._do_smoothing = False
        self._smooth_wlen = None
        self._smooth_step = None
        self._smoothing_percentile = None
        self._smoothing_valid_count = None
        if (smoothing_window_length_hours is not None or
                smoothing_step_hours is not None):
            if not all((smoothing_window_length_hours is not None,
                        smoothing_step_hours is not None)):
                msg = ("if one of 'smoothing_window_length_hours' and "
                       "'smoothing_step_hours' is set, both must be "
                       "specified.")
                raise ValueError(msg)
            # smoothing parameters stored in units of data array indices
            smoothing_step_ns = int(smoothing_step_hours * 3600 * 1e9)
            smoothing_wlen_ns = int(smoothing_window_length_hours * 3600 * 1e9)
            self._do_smoothing = True
            self._smooth_wlen = smoothing_wlen_ns // self.delta_ns + 1
            self._smooth_step = smoothing_step_ns // self.delta_ns
            self._smoothing_percentile = smoothing_percentile
            self._smoothing_mean = smoothing_mean
            self._smoothing_valid_count = int(
                smoothing_valid_percentage * self._smooth_wlen / 100.0)

    def _smooth(self, data, out=None):
        if not self._do_smoothing:
            return data

        if data.ndim == 1:
            # XXX check usage of out parameter.. looks like _smooth ever is
            # only called with 1d though, so it seems safe
            smoothed, _ = self._smooth_1d(data, out=out)
            return smoothed
        else:
            # see above comment, right now disallow using _smooth() for
            # multiple stations at the same time
            raise NotImplementedError

        smoothed = None
        meta = None
        out = None

        for i, d in enumerate(data):
            if smoothed is not None:
                out = smoothed[i]
            smoothed_, meta = self._smooth_1d(d, meta=meta, out=out)
            if smoothed is None:
                smoothed = np.empty((len(data), len(smoothed_)),
                                    dtype=data.dtype)
            smoothed[i][:] = smoothed_
        return smoothed

    def _smooth_1d(self, data, meta=None, out=None):
        if not self._do_smoothing:
            return data

        if meta is None:
            # idx = (np.arange(self._smooth_wlen) +
            #        np.arange(len(data) + self._smooth_wlen - 1)[:, None] -
            #        (self._smooth_wlen + 1))
            # loosely based on https://stackoverflow.com/a/36703105/3419472
            idx = (
                np.arange(self._smooth_wlen) +
                np.arange(
                    -self._smooth_wlen,
                    len(data) + self._smooth_step,
                    self._smooth_step)[:, None])
            invalid_low = idx < 0
            invalid_high = idx >= len(data)
            invalid = invalid_low | invalid_high
            # somehow can't get the idx array correct right from the start.. so
            # cut off invalid slices now.
            # this is only done once for large datasets, so shouldn't be a big
            # problem
            valid_rows = np.sum(~invalid, axis=1) > self._smoothing_valid_count
            idx = idx[valid_rows]
            # ok now we have the idx array we can really work with..
            invalid_low = idx < 0
            invalid_high = idx >= len(data)
            invalid = invalid_low | invalid_high
            idx_invalid = idx.copy()
            idx[invalid_low] = 0
            idx[invalid_high] = len(data) - 1
            meta = idx, idx_invalid, invalid_low, invalid_high, invalid
        else:
            idx, idx_invalid, invalid_low, invalid_high, invalid = meta

        tmp = data[idx]
        # correct invalid time stamps..
        tmp['time'][invalid_low] += idx_invalid[invalid_low] * self.delta_ns
        tmp['time'][invalid_high] += (
            idx_invalid[invalid_high] - (len(data) - 1)) * self.delta_ns
        tmp['i95'][invalid] = np.nan
        tmp['i95'][invalid] = np.nan
        tmp['coverage'][invalid] = 0
        if out is None:
            # setup out array
            out = np.ma.empty(len(tmp), dtype=data.dtype)
            out.mask = False
        # sanity check on the timestamp array..
        assert np.all(np.diff(tmp['time'], axis=1) == self.delta_ns)
        # mean calculation on nanoseconds in int64 will overflow, so work
        # around by going to milliseconds temporarily
        # seems numpy mean doesn't always respect the specified dtype..
        # so do astype() on top
        out['time'] = np.mean(
            tmp['time'] / 1000000, axis=1,
            dtype=self.dtype['time']).astype(self.dtype['time']) * 1000000
        if self._smoothing_mean:
            tmp_i95 = tmp['i95']
            if self._smoothing_percentile < 100:
                percentiles = np.nanpercentile(
                    tmp_i95, q=self._smoothing_percentile, axis=1)
                for row, percentile in zip(tmp_i95, percentiles):
                    row[row > percentile] = np.nan
            out['i95'] = np.nanmean(tmp_i95, axis=1)
        else:
            out['i95'] = np.nanpercentile(
                tmp['i95'], q=self._smoothing_percentile, axis=1)
        i95_invalid = (
            np.isnan(tmp['i95']).sum(axis=1) >= self._smoothing_valid_count)
        # tmp['i95'] = np.ma.masked_array(z, mask=invalid_rows)
        out['i95'].mask = i95_invalid
        # XXX check what happens with masked values in mean calculation
        out['coverage'] = np.mean(tmp['coverage'], axis=1)
        return out, meta

    def _get_filenames(self, network, station, location, channel, starttime,
                       endtime, merge_streams=False):
        if merge_streams:
            if channel[:2] in self.merge_streams:
                channels = [stream + channel[2:]
                            for stream in self.merge_streams]
            else:
                channels = [channel]
        else:
            channels = [channel]

        num_days = (int(endtime.matplotlib_date) -
                    int(starttime.matplotlib_date) + 1)

        filenames = []
        t = starttime
        while t <= endtime and \
                int(t.matplotlib_date) <= int(endtime.matplotlib_date):
            filenames.append(
                [(cha, self.client._get_filename(
                    network, station, location, cha, t)) for cha in channels])
            t += 24 * 3600

        assert num_days == len(filenames)
        return filenames

    def _merge_streams_in_nslc(self, nslc):
        """
        Removes NSLC combinations that would lead to duplicated data when
        getting/plotting data while merging streams (e.g. combining EH and HH
        data).
        """
        if not self.merge_streams:
            return nslc
        nslc_new = []
        for n, s, l, c in nslc:
            for stream in self.merge_streams:
                if (n, s, l, stream + c[2:]) in nslc_new:
                    break
            else:
                nslc_new.append((n, s, l, c))
        return nslc_new

    def get_data(self, network, station, location, channel, starttime,
                 endtime, out=None, merge_streams=True):
        used_channels = set([channel])

        filenames = self._get_filenames(network, station, location, channel,
                                        starttime, endtime,
                                        merge_streams=merge_streams)
        num_days = len(filenames)

        # if smoothing is used, the provided out array is used for storing the
        # final smoothed array, not the then temporary full array..
        if out is None or self._do_smoothing:
            # prepare array that will be filled with all individual data pieces
            # data = np.zeros(len(filenames) * self.len, dtype=self.dtype)
            data = np.ma.empty(num_days * self.len, dtype=self.dtype)
            data.mask = False
        else:
            data = out
            # XXX do some assertions on dtype and shape and mask!!
        mem_address = data.__array_interface__['data'][0]
        data = data.reshape((num_days, self.len))
        # check that no copying was done in mem by reshaping
        assert mem_address == data.__array_interface__['data'][0]
        # print(filenames)

        for i, filenames_ in enumerate(filenames):
            masked = False
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
            # no channel at all was processed yet.. create dummy data
            elif all([d is None for cha, d in individual_channel_data]):
                # ..because data was not processed.
                # make clear that data was not processed for this time by
                # setting the mask
                d = np.empty(self.len, dtype=self.dtype)
                # XXX should we make coverage an int8 instead, so we can put -1
                # in there for "no data processed yet"?
                d['coverage'] = 0
                d['i95'] = np.nan
                d['time'] = (self.times_of_day_nanoseconds +
                             _filename_to_nanoseconds_start_of_day(filename))
                masked = True
            # some channels were processed but None has data: add as gap
            elif all([d is False or d is None
                      for cha, d in individual_channel_data]):
                # ..because data was processed, but there was no waveforms,
                # so this is definitely a gap in the waveforms
                d = np.empty(self.len, dtype=self.dtype)
                d['coverage'] = 0
                d['i95'] = np.nan
                d['time'] = (self.times_of_day_nanoseconds +
                             _filename_to_nanoseconds_start_of_day(filename))
            else:
                raise NotImplementedError

            # we have data
            data[i][:] = d
            if masked:
                data.mask[i] = True

        data = data.reshape(-1)
        data['time'].mask = False
        # check that no copying was done in mem by reshaping
        assert mem_address == data.__array_interface__['data'][0]

        label = _label_for_used_channels(network, station, location,
                                         used_channels)
        # finally, apply smoothing, if selected
        data = self._smooth(data, out=out)
        return data, sorted(used_channels), label

    @staticmethod
    def _fast_availability_for_filename(filename):
        """
        Quick estimate of availability.

        Returns either -1 (file not present, i.e. data not processed yet), 0
        (stub file present, i.e. data was processed but no waveforms available)
        or 100 (file present, so at least some data is available for that day)
        """
        try:
            filesize = os.path.getsize(filename)
        except:
            # should not happen, as only existing filenames are
            # returned by obspy SDS Client looks like
            return -1
        if filesize == 1:
            return 0
        return 100

    @staticmethod
    def _accurate_availability_for_filename(filename):
        """
        Accurate availability as daily percentage.

        Returns a floating point number between 0 and 100.
        """
        fast_avail = I95SDSClient._fast_availability_for_filename(filename)
        if fast_avail in (-1, 0):
            return fast_avail

        data = np.load(filename)
        return data['coverage'].mean()

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

    def get_data_multiple_nslc(self, nslc, starttime, endtime, merge_streams=True):
        used_channels = []
        labels = []
        # due to smoothing we only really know the shape of the resulting array
        # after the first get_data() call
        data = None
        out = None
        for i, (n, s, l, c) in enumerate(nslc):
            if i > 0:
                out = data[i]
            smoothed_1d, used_channels_, label = self.get_data(
                n, s, l, c, starttime, endtime, out=out,
                merge_streams=merge_streams)
            if i == 0:
                data = np.ma.empty((len(nslc), len(smoothed_1d)),
                                   dtype=self.dtype)
                data.mask = False
                mem_address = data.__array_interface__['data'][0]
                data[i] = smoothed_1d
            labels.append(label)
            used_channels.append(used_channels_)
        data['time'].mask = False
        # check that no copying was done in mem by reshaping or internally in
        # get_data()
        assert mem_address == data.__array_interface__['data'][0]
        return data, used_channels, labels

    def plot_all_data(self, starttime, endtime, type='image', nslc=None,
                      cmap=None, global_norm=False, colorbar=True,
                      merge_streams=True, verbose=False, show=True, ax=None,
                      scale=None, percentiles=None, color=None,
                      violin_kwargs=None):
        """
        :param type: ``'image'``, ``'line'`` or ``'violin'``
        :param scale: ``'nm/s'``, ``'mum/s'``, ``'mm/s'``, ``'m/s'``
        """
        if type not in ('image', 'line', 'violin'):
            msg = "option 'type' must be either 'image', 'line' or 'violin'"
            raise ValueError(msg)

        if nslc is None:
            nslc = self.client.get_all_nslc()

        if merge_streams:
            nslc = self._merge_streams_in_nslc(nslc)

        data, used_channels, labels = self.get_data_multiple_nslc(
            nslc, starttime, endtime, merge_streams=merge_streams)

        # remove ids that have no data in given time range
        valid = [i for i, d in enumerate(data)
                 if not np.all(d['i95'].mask | np.isnan(d['i95']).data)]
        data = data[valid]
        used_channels = [item for i, item in enumerate(used_channels)
                         if i in valid]
        labels = [item for i, item in enumerate(labels) if i in valid]

        if ax:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        if type == 'image':
            # plotting of data parts
            vmin_global = np.nanmin(data['i95'])
            # vmax_global = np.nanmax(data['i95'])
            vmax_global = np.nanpercentile(data['i95'],
                                           q=self.vmax_clip_percentile)
            vmin = None
            vmax = None
            if global_norm:
                vmin = vmin_global
                vmax = vmax_global
            channels = [_merge_stream_labels(chas) for chas in used_channels]
            labels = ['.'.join((n, s, l, c))
                      for (n, s, l, _), c in zip(nslc, channels)]
            self._plot_image(
                ax, data, labels, cmap=cmap,
                colorbar=colorbar, vmin=vmin, vmax=vmax,
                global_norm=global_norm)
        elif type == 'line':
            self._plot_lines(ax, data, labels, scale=scale)
        elif type == 'violin':
            self._plot_violin(ax, data, labels, verbose=verbose, scale=scale,
                              percentiles=percentiles, color=color,
                              violin_kwargs=violin_kwargs)
        else:
            raise ValueError

        fig.autofmt_xdate()
        fig.tight_layout()

        if show:
            plt.show()
        return fig, ax

    def _plot_image(self, ax, data, label, cmap=None, colorbar=True,
                    vmin=None, vmax=None, global_norm=True, scale=None):
        if scale is None:
            scale = data['i95']
        scaling_factor, unit_label = _get_scale(scale)

        if vmin is None:
            vmin = np.nanmin(data['i95'] * scaling_factor)
        if vmax is None:
            # vmax = np.nanmax(data['i95'])
            # clip vmax at given percentile
            # XXX remove this??
            vmax = np.nanpercentile(data['i95'] * scaling_factor,
                                    q=self.vmax_clip_percentile)
            # print vmax
        # print(vmin, vmax)
        cmap = cmap or viridis
        cmap.set_bad(self.no_data_color)

        half_delta = self.delta_days / 2.0
        start = date2num(
            UTCDateTime(ns=data['time'].flat[0]).datetime) - half_delta
        end = date2num(
            UTCDateTime(ns=data['time'].flat[-1]).datetime) + half_delta
        # print(start)
        # print(end)
        # print(half_delta)
        cb = None
        if data.ndim == 1 or global_norm:
            im = ax.imshow(np.atleast_2d(data['i95'] * scaling_factor),
                           extent=[start, end, 0, data.shape[0]], vmin=vmin,
                           vmax=vmax, cmap=cmap, interpolation='nearest',
                           aspect='auto')
            if colorbar:
                cb = plt.colorbar(mappable=im, ax=ax)
                cb.set_label('I95 [nm/s]')
        else:
            for i, data_ in enumerate(data[::-1]):
                vmin = np.nanmin(data_['i95'] * scaling_factor)
                vmax = np.nanpercentile(data_['i95'] * scaling_factor,
                                        q=self.vmax_clip_percentile)
                im = ax.imshow(np.atleast_2d(data_['i95'] * scaling_factor),
                               extent=[start, end, 0 + i, 1 + i], vmin=vmin,
                               vmax=vmax, cmap=cmap, interpolation='nearest',
                               aspect='auto')
            if colorbar:
                # make room for colorbars, shrink axes
                ax_rect = list(ax.get_position().bounds)
                ax_rect[0] += 0.10
                ax_rect[3] = 1.0 - ax_rect[1] - 0.02
                ax_rect[2] -= 0.2
                ax.set_position(ax_rect)
                ax_left, ax_bottom, ax_width, ax_height = ax_rect
                cb_bottom = ax_bottom
                cb_top = cb_bottom + ax_height
                cb_individual_height = (cb_top - cb_bottom) / data.shape[0]
                margin = cb_individual_height * 0.05
                for i, im in enumerate(ax.images):
                    cax_left = ax_left + ax_width + 0.04
                    cax_bottom = ax_bottom + i * cb_individual_height + margin
                    cax_width = 0.02
                    cax_height = cb_individual_height - (2 * margin)
                    cax_rect = [cax_left, cax_bottom, cax_width, cax_height]
                    cax = ax.figure.add_axes(cax_rect)
                    cb = plt.colorbar(mappable=im, cax=cax)
                    cb.set_label('I95 [%s]' % unit_label)

        # fig/ax tweaks
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            ObsPyAutoDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_yticks([])
        fontdict = {'family': 'monospace'}
        if data.ndim > 1:
            ax.set_yticks(np.arange(data.shape[0], dtype=np.float32) + 0.5)
            ax.set_yticklabels(label[::-1], fontdict=fontdict)
            ax.set_ylim(0, data.shape[0])
        else:
            ax.set_ylabel(label, fontdict=fontdict)
            ax.set_ylim(0, 1)
        ax.figure.autofmt_xdate()
        if data.ndim == 1 or global_norm:
            ax.figure.tight_layout()
        ax.figure.canvas.draw_idle()
        return cb

    def _plot_lines(self, ax, data, labels, legend=True, scale=None,
                    color=None):
        if scale is None:
            scale = data['i95']
        scaling_factor, unit_label = _get_scale(scale)

        if data.ndim == 1:
            data = [data]
            labels = [labels]
        for data_, label in zip(data, labels):
            times = date2num([UTCDateTime(ns=t).datetime
                              for t in data_['time']])
            ax.plot(times, data_['i95'] * scaling_factor, label=label, lw=0.5,
                    color=color)
        if legend:
            ax.legend()
        ax.set_ylabel('I95 [%s]' % unit_label)

        # fig/ax tweaks
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            ObsPyAutoDateFormatter(ax.xaxis.get_major_locator()))
        ax.figure.autofmt_xdate()
        ax.figure.canvas.draw_idle()

    def _plot_violin(self, ax, data, labels, verbose=False, percentiles=None,
                     scale=None, color=None, violin_kwargs=None):
        import seaborn as sns

        if violin_kwargs is None:
            violin_kwargs = {}

        if scale is None:
            scale = data['i95']
        scaling_factor, unit_label = _get_scale(scale)

        if data.ndim == 1:
            labels = [labels]
            data = [data['i95'][~data['i95'].mask] * scaling_factor]
        else:
            data = [d['i95'][~d['i95'].mask] * scaling_factor for d in data]

        if verbose:
            for d, label in zip(data, labels):
                print(label)
                for perc in (50, 68, 80, 90, 95, 99):
                    value = np.nanpercentile(d, q=perc)
                    print('  {:d}th percentile: {:6.2f} {}'.format(
                        perc, value, scale))

        # avoid extreme spikes in the plot
        y_min = np.inf
        y_max = -np.inf
        for i, d in enumerate(data):
            value = np.nanpercentile(d, 95)
            y_max = max(y_max, value)
            y_min = min(y_min, np.nanmin(d))

        sns.violinplot(data=data, ax=ax, orient="v", cut=0, gridsize=1000,
                       color=color, **violin_kwargs)
        ax.set_ylim(y_min, y_max)
        ax.set_xticklabels(labels)
        ax.set_ylabel('I95 [%s]' % unit_label)

        if percentiles:
            kwargs = dict(va='bottom', color='k', zorder=5)
            margin = 0.02
            for i, d in enumerate(data):
                for perc in percentiles:
                    value = np.nanpercentile(d, perc)
                    xmin = -0.5 + i
                    xmax = xmin + 1
                    ax.plot([xmin, xmax], [value, value], color='k', zorder=5)
                    ax.text(xmin + margin, value, '%s%%' % perc, ha='left',
                            **kwargs)
                    ax.text(xmax - margin, value, '%#.2g' % value, ha='right',
                            **kwargs)

        # fig/ax tweaks
        ax.figure.canvas.draw_idle()

    def plot(self, network, station, location, channel, starttime, endtime,
             type='image', cmap=None, verbose=False, show=True, ax=None,
             percentiles=None, scale=None, color=None, violin_kwargs=None):
        """
        :param type: ``'image'``, ``'line'`` or ``'violin'``
        :type percentiles: list of float
        :param scale: ``'nm/s'``, ``'mum/s'``, ``'mm/s'``, ``'m/s'``
        """
        data, used_channels, label = self.get_data(
            network, station, location, channel, starttime, endtime)

        if ax:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        if type == 'image':
            self._plot_image(ax, data, label, cmap=cmap, scale=scale)
        elif type == 'line':
            self._plot_lines(ax, data, label, scale=scale, color=color)
        elif type == 'violin':
            self._plot_violin(ax, data, label, verbose=verbose,
                              percentiles=percentiles, scale=scale,
                              color=color, violin_kwargs=violin_kwargs)
        else:
            msg = "option 'type' must be either 'image', 'line' or 'violin'"
            raise ValueError(msg)

        fig.tight_layout()

        if show:
            plt.show()
        return fig, ax

    def _get_availability(self, starttime, endtime, fast=True,
                          merge_streams=False):
        nslc = self.client.get_all_nslc()
        if merge_streams:
            nslc = self._merge_streams_in_nslc(nslc)

        start_day = int(starttime.matplotlib_date)
        end_day = int(endtime.matplotlib_date)
        num_days = end_day - start_day + 1

        if fast:
            dtype = np.int8
        else:
            dtype = np.float32
        data = np.empty((len(nslc), num_days), dtype=dtype)
        data.fill(-1)

        labels = []
        for i, (net, sta, loc, cha) in enumerate(nslc):
            data_ = data[i]
            used_channels = set([cha])
            filenames = self._get_filenames(
                net, sta, loc, cha, starttime, endtime,
                merge_streams=merge_streams)
            for filenames_ in filenames:
                for cha_, filename in filenames_:
                    index = _filename_to_mpl_day(filename) - start_day
                    # make sure we get rid of -1 default entry if at least one
                    # channel was processed for this day
                    data_[index] = max(0, data_[index])
                    if fast:
                        avail_ = self._fast_availability_for_filename(filename)
                    else:
                        avail_ = self._accurate_availability_for_filename(
                            filename)
                    if avail_ >= 0:
                        used_channels.add(cha_)
                    avail_ = min(avail_, 100)
                    # XXX this might not be exact for transition days (days
                    # when e.g. EH and HH have both data for half the day), in
                    # which case we would have to add up both availability
                    # parts. but right now we set only 0% or 100% anyway
                    # (at least with fast=True)
                    # XXX corrected now, adding up availability of merged
                    # streams, above comment obsolete
                    data_[index] += avail_
                    # make sure we don't exceed 100, should only happen with
                    # numerical errors though?!
                    if data_[index] > 100:
                        msg = ('Data coverage for given day adds up to more '
                               'than 100%, merged files: ' + str(filenames))
                        warnings.warn(msg)
                    data_[index] = min(data_[index], 100)
            label = _label_for_used_channels(net, sta, loc, used_channels)
            labels.append(label)
        return data, labels

    def plot_availability(self, starttime, endtime, fast=True,
                          merge_streams=False, show=True, grid=True, ax=None,
                          verbose=False, vmin=0, vmax=100,
                          number_of_colors=None, percentage_in_label=True):
        data, labels = self._get_availability(
            starttime, endtime, fast=fast, merge_streams=merge_streams)

        if verbose:
            print('availability:')
            for data_, label in zip(data, labels):
                print('  %s: %.2f/%.2f/%.2f/%.2f  (min/mean/median/max)' % (
                    label, data_.min(), data_.mean(), np.median(data_),
                    data_.max()))

        start_day = int(starttime.matplotlib_date)
        end_day = int(endtime.matplotlib_date) + 1

        data = np.atleast_2d(data)
        extent = [start_day, end_day, 0, len(data)]

        # make a color map of fixed colors
        #  -1 in data means: not processed
        #   0 in data means: processed but no waveform data encountered for day
        # 100 in data means: at least partial data encountered for given day
        #                    (actual coverage data is not read in fast mode)
        if fast:
            cmap = ListedColormap(['lightgray', 'red', 'green'])
            bounds = [-1.5, -0.5, 0.5, 1.5]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            if number_of_colors is None:
                cmap = 'viridis'
            else:
                cmap = get_cmap('viridis', lut=number_of_colors)
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap.set_bad(color='lightgray')

        if ax:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        # for detailed plot, we mask days that were not even computed, so that
        # they show up gray like in the fast plot
        if not fast:
            data = np.ma.masked_equal(data, -1)

        im = ax.imshow(data, extent=extent, interpolation='nearest',
                       aspect='auto', cmap=cmap, norm=norm)

        if percentage_in_label:
            if fast:
                msg = ("Option 'percentage_in_label' not available together "
                       "with option 'fast'.")
                raise ValueError(msg)
            data = data.filled(0.0)
            for i, d in enumerate(data):
                labels[i] += ' (%#.1f%%)' % d.mean()

        ax.set_yticks(np.arange(len(data)) + 0.5)
        if grid:
            grid_y = np.arange(len(data) + 1)
            grid_kwargs = dict(color='k', lw=1.0, alpha=0.5)
            for y in grid_y[::2]:
                ax.axhline(y, ls='-', **grid_kwargs)
            for y in grid_y[1::2]:
                ax.axhline(y, ls='--', **grid_kwargs)
        # numpy array gets plotted from bottom to top in imshow
        ax.set_yticklabels(labels[::-1], fontdict={'family': 'monospace'})
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            ObsPyAutoDateFormatter(ax.xaxis.get_major_locator()))
        fig.autofmt_xdate()

        if not fast:
            cb = fig.colorbar(mappable=im, ax=ax)
            cb.set_label('Daily data coverage [%]')

        fig.tight_layout()

        if show:
            plt.show()
        return fig, ax


def _get_scale(data, initial_scale=1e9):
    """
    :type data: np.ndarray or str
    :param data: Array with data or one of ``'nm/s'``, ``'mum/s'``, ``'mm/s'``,
        ``'m/s'``
    :param initial_scale: Initial scaling of data (e.g. 1e9 for nm/s data)
    :rtype: (float, str)
    :returns: Scaling factor for data and corresponding units for axis label
    """
    if isinstance(data, np.ndarray):
        if np.ma.is_masked:
            data = data[~data.mask]
        data_max = data.max()
        if data_max < 1e-6 * initial_scale:
            data = 'nm/s'
        elif data_max < 1e-3 * initial_scale:
            data = 'mum/s'
        elif data_max < 1 * initial_scale:
            data = 'mm/s'
        else:
            data = 'm/s'

    if data == 'nm/s':
        scalefac = 1e9
        unit_label = "nm/s"
    elif data == 'mum/s':
        scalefac = 1e6
        unit_label = u"µm/s"
    elif data == 'mm/s':
        scalefac = 1e3
        unit_label = "mm/s"
    elif data == 'm/s':
        scalefac = 1.0
        unit_label = "m/s"
    else:
        raise ValueError()

    return scalefac / initial_scale, unit_label


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

    data, labels = i95_client.get_availability(
        starttime=UTCDateTime(2010, 1, 1), endtime=UTCDateTime(2018, 1, 1),
        fast=True)

    print(data)
    print(labels)
