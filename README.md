# I95 SDS

Calculate, store and on demand plot [I95](https://doi.org/10.1111/j.1365-246X.2009.04343.x) noise levels of seismic data stored in a local SDS directory tree. The main idea is to keep it simple and easily automatable to process data on a regular basis in the background to then have an easy way to tap into the pre-computed noise data on demand.

## Usage

There are two files in this project. One is used to calculate I95 values and store them in a [SDS](https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html)(-like) structure and one is used to visualize I95 noise levels based on the previously computed data. Simply download the github repository or `git clone` it (which helps with tracking changes if making any customizations in the code).

### Preparing I95 Data

For this step the file `update_i95_mseed_archive` can be used as an executable that will be run using the current active Python environment with a recent version of [ObsPy](https://github.com/obspy/obspy/) as only dependency. Usage information can be shown like this:

```shell
$ ./update_i95_mseed_archive --help
```

Create a directory to store the computed I95 data:

```shell
$ mkdir /tmp/I95
```

Create a [JSON](https://www.json.org/) file with parameters for the processing. This file **should remain unchanged** after starting to process data, at least in terms of the actual signal processing parameters (filter, deconvolution, ...) because there is **no way to tell later on what settings were in place when an individual daily I95 file was processed if the settings change**. Of course other paramters that do not affect the signal processing (waveform path, FDSN URL, ...) could be changed if needed without changing the signal processing aspect. An example file content to copy/paste can be shown by calling the script with that empty directory as a target:

```shell
$ ./update_i95_mseed_archive --sds-root-i95 /tmp/I95 -t 2024-234 --stream HH
Exception: I95 SDS root directory "/tmp/I95blaa" does not contain mandatory "parameters.json" file. Create file with e.g. following content:

{
    "buffer_length": 120,
    "fdsn_base_url": "https://erde.geophysik.uni-muenchen.de",
[...]
}
```

After this, data can be processed and will get added in the SDS(-like) directory tree (only deviation from strict SDS is the file suffix `.npy`). Note that currently the raw data needs to be in a SDS directory structure (even though the code could easily be extended to use other sources of data available through obspy clients).

What data gets processed depends on two things:
  - command line parameters (SEED conventions are assumed in many places, e.g. three character channel codes)
    - `--stream` (mandatory): this parameter can be specified multiple times, e.g. `--stream HH --stream EH --stream EL`. All channels not starting with any of the given option will be ignored
    - `--component-codes` (optional): this paramter controls what components (third character of channel code) get processed and has a set default if not specified (currently `ZNE123`)
    - `--station-code` (optional): this parameter can be used to restrict processing to certain station codes and can be given multiple times. If not specified all station codes will get processed
  - station metadata fetched at time of processing:
    - if `"fdsn_base_url"` is set with an URL to a FDSN server, the server gets queried for **all** available metadata and only the above filters will get applied otherwise all retrieved stations/channels will get processed. To not fetch station metadata and only rely on user controlled station metadata file(s), set `"fdsn_base_url": null,` in the JSON file.
    - station metadata given in local file(s) readable with ObsPy with JSON parameter `"inventories"`. Again all info found will get processed unless restricted with above command line filters

The program operates in daily chunks and by default, data that was (tried to be) processed at some point will simply be skipped unless `--overwrite` is specified, even if no data could be produced (in which case a "magic" 1-byte file is created to represent a processed day for which no raw data was available).

After processing a single day on a single station the I95 data directory could look like this:

```
.
├── 2024
│   └── BW
│       └── FFB1
│           ├── BH1.D
│           │   └── BW.FFB1..BH1.D.2024.294.npy
│           ├── BH2.D
│           │   └── BW.FFB1..BH2.D.2024.294.npy
│           └── BHZ.D
│               └── BW.FFB1..BHZ.D.2024.294.npy
├── log.txt
├── parameters.json
└── times.npy
```

The data processing can easily be automated in cron jobs, e.g. using `date` command to fill in the `--time` parameter dynamically like this: `date --date="yesterday" '+%Y-%j'`

### Visualizing the I95 Data

For this purpose the file `i95_sds.py` can be used. Just create a short Python script using the provided functionality from there to make custom plots. Easiest way to import from this file is to make a symbolic link to `i95_sds.py` in the directory where the custom plot script is located.

Here is an example with some plot commands:

```python
from i95_sds import I95SDSClient

client = I95SDSClient('/tmp/I95')

client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294')
client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294', type='violin')
client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294', type='line')
client.plot_availability(
    '2024-294', '2024-294', fast=False, percentage_in_label=True)
client.plot_all_data('2024-294', '2024-294')
client.plot_all_data('2024-294', '2024-294', type='violin')
```

Note that violin plots use the `seaborn` module wich can be installed using e.g. `conda` or `pip`.

## Example Plots
