# -*- coding: utf-8 -*-
'''
@author: Michael Schneider, with input from XFEL-CAS and SCS beamline staff

'''

import numpy as np
import xarray as xr
import karabo_data as kd
from karabo_data.read_machinery import find_proposal
import pandas as pd
import os
from glob import glob
from time import strftime, sleep
from tqdm import tqdm


def find_run_dir(proposal, run):
    '''returns the raw data folder for given (integer) proposal and run number'''
    proposal_dir = find_proposal(f'p{proposal:06d}')
    return os.path.join(proposal_dir, f'raw/r{run:04d}')


def load_run_selective(proposal, run_nr, include=None, exclude=None, maxfiles=None):
    '''
    Finds the rundirectory for the given (integer) proposal and run numbers and returns
    a karabo_data.DataCollection instance. Files can be selected or excluded based on
    filenames. Setting any of the optional paramters to None disables that particular
    selection.
    
    Parameters
    ==========
    proposal :: int
        proposal number
    run_nr :: int
        run number
    include :: str
        Load only files that have this string in the filename
    exclude :: str
        Skip files with this string in the filename
    maxfiles :: int
        Load at most this many files
    '''
    rundir = find_run_dir(proposal, run_nr)
    flist = glob(os.path.join(rundir, '*h5'))

    if include is not None:
        flist = [f for f in flist if include in f]

    if exclude is not None:
        flist = [f for f in flist if exclude not in f]
    
    return kd.DataCollection.from_paths(flist[:maxfiles])


def load_scan_variable(run, scan_variable, stepsize=None):
    '''
    Loads the given scan variable and rounds scan positions to integer multiples of "stepsize"
    for consistent grouping. Creates a dummy scan if scan_variable is set to None.
    Parameters:
        run : (karabo_data.DataCollection) RunDirectory instance
        scan_variable : (tuple of str) ("source name", "value path"), examples:
                        ('SCS_ILH_LAS/PHASESHIFTER/DOOCS', 'actualPosition.value')
                        ('SCS_ILH_LAS/DOOCS/PPL_OPT_DELAY', 'actualPosition.value')
                        ('SA3_XTD10_MONO/MDL/PHOTON_ENERGY', 'actualEnergy.value')
                        None creates a dummy file to average over all trains of the run
        stepsize : (float) nominal stepsize of the scan - values of scan_variable will be
                   rounded to integer multiples of this value
    '''
    if scan_variable is not None:
        source, path = scan_variable
        scan = run.get_array(source, path)
        if stepsize is not None:
            scan = stepsize * np.round(scan / stepsize)
    else:
        # dummy scan variable - this will average over all trains
        scan = xr.DataArray(np.ones(len(run.train_ids), dtype=np.int16),
                            dims=['trainId'], coords={'trainId': run.train_ids})
    scan.name = 'scan_variable'
    return scan


def load_xgm(run, print_info=False):
    '''Returns the XGM data from loaded karabo_data.DataCollection'''
    nbunches = run.get_array('SCS_RR_UTC/MDL/BUNCH_DECODER', 'sase3.nPulses.value')
    nbunches = np.unique(nbunches)
    if len(nbunches) == 1:
        nbunches = nbunches[0]
    else:
        warnings.warn('not all trains have same length DSSC data')
        print('nbunches: ', nbunches)
        nbunches = max(nbunches)
    if print_info:
        print('SASE3 bunches per train:', nbunches)
    
    xgm = run.get_array('SCS_BLU_XGM/XGM/DOOCS:output', 'data.intensitySa3TD',
                        roi=kd.by_index[:nbunches], extra_dims=['pulse'])
    return xgm


def load_TIM(run, apd='MCP2apd'):
    '''
    Load TIM traces and match them to SASE3 pulses. "run" is a karabo_data.RunDirectory instance.
    Uses SCS ToolBox.
    '''
    import ToolBox as tb
    
    fields = ["sase1", "sase3", "npulses_sase3", "npulses_sase1", apd, "SCS_SA3", "nrj"]
    timdata = xr.Dataset()
    for f in fields:
        m = tb.mnemonics[f]
        timdata[f] = run.get_array(m['source'], m['key'], extra_dims=m['dim'])

    timdata.attrs['run'] = run
    timdata = tb.matchXgmTimPulseId(timdata)
    return timdata.rename({'sa3_pId': 'pulse'})[apd]


def prepare_module_empty(scan_variable, framepattern):
    '''Create empty (zero-valued) DataArray for a single DSSC module to iteratively add data to'''
    len_scan = len(np.unique(scan_variable))
    dims = ['scan_variable', 'x', 'y']
    coords = {'scan_variable': np.unique(scan_variable)}
    shape = [len_scan, 128, 512]
        
    empty = xr.DataArray(np.zeros(shape, dtype=float), dims=dims, coords=coords)
    module_data = xr.Dataset()
    for name in framepattern:
        module_data[name] = empty.copy()
    
    module_data['sum_count'] = xr.DataArray(np.zeros(len_scan, dtype=int), dims=['scan_variable'])
    return module_data


def load_dssc_info(proposal, run_nr):
    '''Loads the first data file for DSSC module 0 (this is hardcoded) and
    returns the detector_info dictionary'''
    dssc_module = load_run_selective(proposal, run_nr, include='DSSC00', maxfiles=1)
    dssc_info = dssc_module.detector_info('SCS_DET_DSSC1M-1/DET/0CH0:xtdf')
    return dssc_info


def load_chunk_data(sel, sourcename):
    '''Load DSSC data (sel is a DataCollection or a subset of a DataCollection
    obtained by its select_trains() method). The flattened multi-index (trains+pulses)
    is unraveled before returning the data.
    '''
    fpt = sel.detector_info(sourcename)['frames_per_train']
    data = sel.get_array(sourcename, 'image.data', extra_dims=['_empty_', 'x', 'y']).squeeze()
    
    tids = np.unique(data.trainId)
    data = data.rename(dict(trainId='trainId_pulse'))
    
    midx = pd.MultiIndex.from_product([sorted(tids), range(fpt)], names=('trainId', 'pulse'))
    data = xr.DataArray(data, dict(trainId_pulse=midx)).unstack('trainId_pulse')
    data = data.transpose('trainId', 'pulse', 'x', 'y')
    return data


def merge_chunk_data(module_data, chunk_data, framepattern):
    '''Merge chunk data with prepared dataset for entire module.
    Aligns on "scan_variable" and sums values for variables
    ['pumped', 'unpumped', 'sum_count']'''
    where = dict(scan_variable=chunk_data.scan_variable)
    for var in framepattern + ['sum_count']:
        # module_data[var].loc[where] = module_data[var].loc[where] + chunk_data[var]
        # previous line doesn't convert to larger dtypes when necessary
        # the next line concatenates the data along a new dimension ('tmp') and uses
        # the sum() method, which supports automatic conversion
        summed = xr.concat([module_data[var].loc[where], chunk_data[var]], dim='tmp').sum('tmp')
        module_data[var].loc[where] = summed
    return module_data


def split_frames(data, pattern, prefix=''):
    '''Split frames according to "pattern" and average over resulting splits.
    "pattern" is a list of frame names (order matters!). Examples:
        pattern = ['pumped', 'pumped_dark', 'unpumped', 'unpumped_dark']  # 4 DSSC frames, 2 FEL pulses
        pattern = ['pumped', 'unpumped']  # 2 FEL frames, no intermediate darks
        pattern = ['image']  # no splitting, average over all frames
    Returns a dataset with data variables named prefix + framename
    '''
    n = len(pattern)
    dataset = xr.Dataset()
    for i, name in enumerate(pattern):
        dataset[prefix + name] = data.loc[{'pulse': np.s_[i::n]}].mean('pulse')
    return dataset


def calc_xgm_frame_indices(nbunches, framepattern):
    '''
    Returns a coordinate array for XGM data. The coordinates correspond to DSSC
    frame numbers and depend on the number of FEL pulses per train ("nbunches")
    and the framepattern. In framepattern, dark DSSC frame names (i.e., without
    FEL pulse) _must_ include "dark" as a substring.
    '''
    n_frames = len(framepattern)
    n_data_frames = np.sum(['dark' not in p for p in framepattern])
    frame_max = nbunches * n_frames // n_data_frames

    frame_indices = []
    for i, p in enumerate(framepattern):
        if 'dark' not in p:
            frame_indices.append(np.arange(i, frame_max, n_frames))

    return np.sort(np.concatenate(frame_indices))


def process_intra_train(job):
    '''Aggregate DSSC data (chunked, to fit into memory) for a single module.
    Averages over all trains, but keeps all pulses.
    Designed for the multiprocessing module - expects a job dictionary with the following keys:
      proposal : (int) proposal number
      run : (int) run number
      module : (int) DSSC module to process
      chunksize : (int) number of trains to process simultaneously
    '''
    proposal = job['proposal']
    run_nr = job['run_nr']
    module = job['module']
    chunksize = job['chunksize']
    
    sourcename = f'SCS_DET_DSSC1M-1/DET/{module}CH0:xtdf'
    collection = load_run_selective(proposal, run_nr, include=f'DSSC{module:02d}')
    
    fpt = load_dssc_info(proposal, run_nr)['frames_per_train']
    dims = ['pulse', 'x', 'y']
    shape = [fpt, 128, 512]
    module_data = xr.DataArray(np.zeros(shape, dtype=float), dims=dims)
    module_data = module_data.to_dataset(name='image')
    module_data['sum_count'] = xr.DataArray(np.zeros(fpt, dtype=int), dims=['pulse'])
    
    ntrains = len(collection.train_ids)
    chunks = np.arange(ntrains, step=chunksize)
    if module == 15:
        pbar = tqdm(total=len(chunks))
    for start_index in chunks:
        sel = collection.select_trains(kd.by_index[start_index:start_index + chunksize])
        data = load_chunk_data(sel, sourcename)
        data = data.to_dataset(name='image')
        
        data['sum_count'] = xr.full_like(data.trainId, fill_value=1)
        data = data.sum('trainId')

        for var in ['image', 'sum_count']:
            # concatenating and using the sum-method automatically takes care of dtype casting if necessary
            module_data = xr.concat([module_data[var], chunk_data[var]], dim='tmp').sum('tmp')
        if module == 15:
            pbar.update(1)
    
    module_data['image'] = module_data['image'] / module_data.sum_count
    return module_data


def process_dssc_module(job):
    '''Aggregate DSSC data (chunked, to fit into memory) for a single module.
    Groups by "scan_variable" in given scanfile - use dummy scan_variable to average
    over all trains. This implies, that only trains found in the scanfile are considered.
    Designed for the multiprocessing module - expects a job dictionary with the following keys:
      proposal : (int) proposal number
      run : (int) run number
      module : (int) DSSC module to process
      chunksize : (int) number of trains to process simultaneously
      scanfile : (str) name of hdf5 file with xarray.DataArray containing the scan variable and trainIds
      framepattern : (list of str) names for the (possibly repeating) intra-train pulses. See split_dssc_data
      pulsemask : (str) name of hdf5 file with boolean xarray.DataArray to select/reject trains and pulses
    '''
    proposal = job['proposal']
    run_nr = job['run_nr']
    module = job['module']
    chunksize = job['chunksize']
    scanfile = job['scanfile']
    framepattern = job['framepattern']
    maskfile = job.get('maskfile', None)
    
    sourcename = f'SCS_DET_DSSC1M-1/DET/{module}CH0:xtdf'
    
    collection = load_run_selective(proposal, run_nr, include=f'DSSC{module:02d}')
        
    ntrains = len(collection.train_ids)
    
    # read preprocessed scan variable from file - selection and (possibly) rounding already done.
    scan = xr.open_dataarray(scanfile, 'data', autoclose=True)

    # read binary pulse/train mask - e.g. from XGM thresholding
    if maskfile is not None:
        pulsemask = xr.open_dataarray(maskfile, 'data', autoclose=True)
    else:
        pulsemask = None
    
    module_data = prepare_module_empty(scan, framepattern)
    chunks = np.arange(ntrains, step=chunksize)
    if module == 15:
        # quick and dirty progress bar
        pbar = tqdm(total=len(chunks))
    for start_index in chunks:
        sel = collection.select_trains(kd.by_index[start_index:start_index + chunksize])
        data = load_chunk_data(sel, sourcename)
        if pulsemask is not None:
            data = data.where(pulsemask)

        data = split_frames(data, framepattern)
        data['sum_count'] = xr.full_like(data.trainId, fill_value=1)
        data['scan_variable'] = scan  # aligns on trainId, drops non-matching trains 
        data = data.groupby('scan_variable').sum('trainId')
        module_data = merge_chunk_data(module_data, data, framepattern)
        if module == 15:
            pbar.update(1)
    
    for name in framepattern:
        module_data[name] = module_data[name] / module_data.sum_count
    return module_data
        
