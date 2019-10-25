# dssc_process
python notebooks to process DSSC data recorded at the SCS beamline of the European XFEL

## Basic concepts
The purpose of this library is to condense DSSC data following these basic ideas:

### grouping by another variable
The images need to be grouped by some (non-DSSC) variable (`scan_variable`) and subsequently averaged. The scan variable is a function of the trainId. Common examples include the photon energy, the delay in a pump-probe experiment, gas attenuator transmission, or the sample stage position. The scan variable is loaded (and possibly manipulated) in an interactive session and then written out into a temporary file for use during the actual data processing. ___Note that trainIds that are missing in the scan variable file will not be considered during processing!___

In cases where only a single averaged image is required, a dummy scan variable is used (that has the same value for all trainIds).

### frame sorting
Within each train, the DSSC frames fall into a (typically repeating) pattern (`framepattern`). In a pump-probe experiment, this might be "even frames are pumped, odd frames unpumped" (`['pumped', 'unpumped']`). In the most simple case all frames are equal, in which case the frame pattern is of unity length (`['image']`). More complex examples include dark frames, i.e., frames that do not coincide with an x-ray pulse (`['pumped', 'pumped_dark', 'unpumped', 'unpumped_dark']`). Names for such frames _must_ include the substring "dark".

### frame and train rejection
Often, DSSC frames need to be excluded from processing on a per-frame and per-train basis. This may again be decided on the basis of another variable, e.g., x-ray intensity monitors. The rejection is handled by another (optional) temporary file, called `framemask`. This is a boolean array with a shape (trains, frames). Every frame for which the corresponding framemask entry is `False`, will not be considered during processing. ___Similar to the scan variable, this is also the case for frames and trains that are missing in the mask file!___

As a special case, the `maxframes` parameter limits the number of DSSC frames for the entire run. This is useful for runs in which the number of frames recorded per train was misconfigured and doesn't match the XFEL pulse number. This can also be achieved using a frame mask, but requires less computation and is thus faster.


## Data handling concept
The data from individual DSSC modules is processed independently. This allows for easy parallelization on a per-module basis. For performance reasons, all images are loaded in chunks (`chunksize`) of consecutive data (to avoid repeatedly seeking over the data files). All data loading is handled by [karabo_data](https://github.com/European-XFEL/karabo_data) and data manipulation relies on [xarray](http://xarray.pydata.org).

At the beginning of the processing, a zero-valued xarray with the expected final shape (scan_variable, pixel_x, pixel_y) is allocated. The DSSC images are then loaded in chunks, grouped by the scan variable, possibly filtered by a frame mask and finally summed. The summation keeps track of the number of frames that went into each group, so that a correct average can be calculated at the end.
