#!/usr/bin/env python3
"""
This script works on two I95 SDS directory trees. The "update" directory tree
will be used to pull in new data files into the "main" directory tree. Files
will either be overwritten if they exist and the modification timestamp is
newer or they will simply be created, also creating subdirectories as needed.

This program became needed to merge data in that was computed on multicore
worker machines with read access to the waveform archive but no write access to
the main I95 SDS directory. The same could in principle be achieved with a
simple "rsync" command that only copies over newer files, but this is prone to
user error when getting the rsync command syntax wrong and also this would not
prevent merging incompatible I95 data computed with differing processing
parameters.

The program will first check if the I95 SDS directory trees seem compatible,
comparing the processing parameters stored in "parameters.json" and "times.npy"
and stop execution if there is any mismatch, since that would mean merging
together files with differnt processing settings.

All actions are logged to the logfile as specified at the top of the program.

This script was tested before use once, but it is advised to make a backups of
the "main" directory tree first.
"""
import datetime
import json
import logging
import shutil
from pathlib import Path
import numpy as np

# root path of the main I95 SDS directory to merge data into.
# data will be written into this directory and files might get overwritten!
main = Path('/tmp/I95_0.1-5Hz_old')
# root path of the I95 SDS directory to read updated/newer data from.
# data is only read from this directory to update the above directory tree.
update = Path('/tmp/I95_0.1-5Hz_new')


fromtimestamp = datetime.datetime.fromtimestamp

# logfile
now_str = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
logfile = Path(f'/tmp/I95_merge_{now_str}.log')
print(f'Logging to file: {logfile}')


if logfile.exists() or not logfile.parent.is_dir():
    msg = f'Can not write to logfile: {logfile}'
    raise IOError(msg)

logging.basicConfig(
    filename=str(logfile), level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(msg)s')


# first, do some checks on the directories
for path in (main, update):
    # check if both trees appear to really be I95 SDS trees, to avoid copying
    # from/to wrong directory trees
    for name in ("parameters.json", "times.npy"):
        if not (path / name).is_file():
            msg = (f'{(path / name)} is not a file, directory seems not to be '
                   f'an I95 SDS root?')
            logging.critical(msg)
            raise Exception(msg)

# next, check that file contents of these control files are the same,
# otherwise content is likely not compatible
with open(main / "parameters.json", "rt") as fh:
    parameters_main = json.load(fh)
with open(update / "parameters.json", "rt") as fh:
    parameters_update = json.load(fh)
# remove some keys that have nothing to do with processing settings and might
# differ
for key in ('fdsn_base_url', 'sds_root_waveforms'):
    for parameters in (parameters_main, parameters_update):
        parameters.pop(key)
# compare processing settings
if parameters_main != parameters_update:
    msg = (f'Processing parameters seem to differ (parameters.json):\n'
           f'main: {parameters_main}\n'
           f'update: {parameters_update}')
    logging.critical(msg.replace('\n', ' '))
    raise Exception(msg)
# also compare the times.npy file contents
times_main = np.load(main / "times.npy")
times_update = np.load(update / "times.npy")
if not np.array_equal(times_main, times_update):
    msg = (f'Processing parameters seem to differ (times.npy):\n'
           f'main: {times_main}\n'
           f'update: {times_update}')
    logging.critical(msg.replace('\n', ' '))
    raise Exception(msg)


def check_parent(destination):
    """
    Check if destination parent dir exists, and create it if necessary and log
    """
    parent = destination.parent
    if not parent.exists():
        msg = f'creating directory {parent}'
        logging.info(msg)
        parent.mkdir(parents=True)
    else:
        if not parent.is_dir():
            msg = (f'failed to create directory {parent} since it exists but '
                   f'is not a directory')
            logging.critical(msg)
            raise Exception()


def process_source_path(source):
    """
    Copy the file, copy the file metadata too and log
    """
    relative_path = source.relative_to(update)
    year, net, sta, cha, filename = relative_path.parts
    destination = main / year / net / sta / cha / filename
    source_mtime = source.stat().st_mtime
    # check if destination exists
    if destination.exists():
        # check if source is newer than destination or not
        destination_mtime = destination.stat().st_mtime
        if destination_mtime >= source_mtime:
            # our main directory tree file has a newer modification time, skip
            # this file
            msg = (f'skipping {relative_path}: existing data is newer ('
                   f'update file mtime: {fromtimestamp(source_mtime)}, '
                   f'main file mtime: {fromtimestamp(destination_mtime)})')
            logging.info(msg)
            return "outdated"
        else:
            # copy2 copies file and file attributes like modification time
            msg = (f'updating file {destination}: timestamp was '
                   f'{fromtimestamp(destination_mtime)} and now is '
                   f'{fromtimestamp(source_mtime)}')
            logging.info(msg)
            shutil.copy2(source, destination)
            return "updated"
    else:
        check_parent(destination)
        # copy2 copies file and file attributes like modification time
        msg = (f'creating new file {destination}: timestamp is '
               f'{fromtimestamp(source_mtime)}')
        logging.info(msg)
        shutil.copy2(source, destination)
        return "new"


# finally let's copy newer data to the main, processing file by file
count = {'new': 0, 'updated': 0, 'outdated': 0, 'error': 0}
for source in update.glob('20??/*/*/*/*.npy'):
    try:
        result = process_source_path(source)
    except Exception as e:
        exc_msg = str(e).replace("\n", " ")
        msg = f'Unexpected Error processing file {source}: {exc_msg}'
        logging.critical(msg)
        count['error'] += 1
        raise
        continue
    else:
        count[result] += 1

msg = (f'SUMMARY: {count["new"]} new files added, {count["updated"]} files '
       f'updated, {count["outdated"]} older files ignored, {count["error"]} '
       f'errors processing individual files')
logging.info(msg)
