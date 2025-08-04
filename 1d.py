# -*- coding: utf-8 -*-
"""
Marine Heatwave Detection Script for FESOM2 1D Data
Adapted from original 2D MHW detection script
"""

# Dask imports for parallelization
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
from matplotlib import cm
import pyfesom2 as pf
import netCDF4 as nc
from netCDF4 import Dataset
from tqdm import tqdm
import gc
from scipy import signal
import scipy.ndimage as ndimage
import calendar
import sys

# Load FESOM2 mesh and set data paths
mesh = pf.load_mesh('./data/core2/')
datapath = "./data/"
temp_var_name = "temp" # Variable name for temperature data
start_year = 1850
end_year = 1851
output_path = "./output/" # Path to save output

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Verbosity setting
verbose = True

# Configuration parameters
n_workers = 4  # Number of Dask worker processes (adjust based on your CPU cores)

# Parameters for Marine Heatwave detection
smoothPercentile = True  # Boolean switch to smooth threshold percentile timeseries
smoothPercentileWidth = 31  # Width of smoothing window in timesteps
climatologyPeriod = [1850, 1852]  # Period for climatology calculation (adjust as needed)
pctile = 90  # Threshold percentile for detection of extreme values
maxGap = 2  # Maximum gap between events to be considered as one event
minDuration = 5  # Minimum duration of a MHW event

# Arctic filtering parameters
arctic_lat_threshold = 60.0  # Minimum latitude for Arctic points

# Processing parameters
chunk_size = 3000  # Number of nodes to process in one chunk
# Layers to process (test with 3 first)
depth_layers = [0]  # Indices of depth layers to process

def runavg(ts, w):
    '''
    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.
    Inputs:
      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window
    Outputs:
      ts_smooth     Smoothed time series
    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015
    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts
    
# Compute number of days in year
def days_in_year(year):
    return 366 if calendar.isleap(year) else 365

# Main function for MHW detection
def detect_marine_heatwaves(sst_data, time_array, year_array, month_array, node_indices=None):
    """
    Detect marine heatwaves in time series of SST data (any frequency)
    
    Parameters:
    -----------
    sst_data : numpy.ndarray
        Temperature data with shape (time, nodes)
    time_array : numpy.ndarray
        Array of datetime objects or ordinal dates
    year_array, month_array : numpy.ndarray
        Arrays of years and months (for compatibility)
    node_indices : list or numpy.ndarray, optional
        Indices of nodes to process. If None, process all nodes.
    
    Returns:
    --------
    mhw_metrics : dict
        Dictionary of MHW metrics for each node
    """
    # Basic dimensions
    n_times, n_nodes_all = sst_data.shape
    
    # Subset nodes if specified
    if node_indices is not None:
        sst = sst_data[:, node_indices]
        n_nodes = len(node_indices)
    else:
        sst = sst_data
        n_nodes = n_nodes_all
    
    # Calculate day-of-year for each time step for climatology
    # This works with any time frequency (daily, weekly, monthly, etc.)
    doy_array = np.zeros(n_times)
    for i, t in enumerate(time_array):
        if hasattr(t, 'timetuple'):
            doy_array[i] = t.timetuple().tm_yday
        else:
            # Fallback: use month as proxy for day-of-year
            doy_array[i] = month_array[i] * 30  # Rough approximation
    
    # Use a window around each day-of-year for climatology calculation
    # This is more flexible than monthly grouping
    window_half_width = 15  # days around each day for climatology
    
    # Arrays for results
    mhw_freq = np.full(n_nodes, np.nan)
    mhw_dur = np.full(n_nodes, np.nan)
    mhw_int = np.full(n_nodes, np.nan)
    
    # Process each node
    for n in range(n_nodes):
        if n % 100 == 0 and n > 0:
            print(f"Processed {n}/{n_nodes} nodes")
        
        # Extract time series for this node
        ts = sst[:, n]
        
        # Skip if all data is NaN
        if np.all(np.isnan(ts)):
            continue
        
        # Calculate climatology and threshold for each time step using day-of-year approach
        clim_full = np.full(n_times, np.nan)
        threshold_full = np.full(n_times, np.nan)
        
        # For each time step, calculate climatology using a window around the same day-of-year
        for i in range(n_times):
            current_doy = doy_array[i]
            
            # Find all time steps within the window around this day-of-year
            # Handle year boundaries (e.g., Dec 31 and Jan 1)
            lower_bound = current_doy - window_half_width
            upper_bound = current_doy + window_half_width
            
            if lower_bound < 1:
                # Wrap around to end of year
                window_indices = ((doy_array >= (365 + lower_bound)) | 
                                (doy_array <= upper_bound))
            elif upper_bound > 365:
                # Wrap around to beginning of year
                window_indices = ((doy_array >= lower_bound) | 
                                (doy_array <= (upper_bound - 365)))
            else:
                # Normal case
                window_indices = ((doy_array >= lower_bound) & 
                                (doy_array <= upper_bound))
            
            # Get data for this window
            window_data = ts[window_indices]
            valid_data = window_data[~np.isnan(window_data)]
            
            if len(valid_data) > 0:
                # Calculate climatology (mean) and threshold (percentile)
                clim_full[i] = np.mean(valid_data)
                threshold_full[i] = np.nanpercentile(valid_data, pctile)
        
        # Smooth threshold if needed
        if smoothPercentile and not np.all(np.isnan(threshold_full)):
            # Only smooth non-NaN values
            valid_thresh = ~np.isnan(threshold_full)
            if np.sum(valid_thresh) > smoothPercentileWidth:
                threshold_full[valid_thresh] = runavg(threshold_full[valid_thresh], 
                                                     min(smoothPercentileWidth, np.sum(valid_thresh)))
        
        # Find marine heatwave events
        # A marine heatwave is defined as when temperature exceeds the threshold
        mhw_bool = np.zeros(n_times, dtype=bool)
        mhw_bool[ts > threshold_full] = True
        
        # Account for missing data
        mhw_bool[np.isnan(ts)] = False
        
        # Find contiguous regions (events)
        events, n_events = ndimage.label(mhw_bool)
        
        # If no events found, continue to next node
        if n_events == 0:
            continue
            
        # Store start and end times for each event
        event_starts = []
        event_ends = []
        
        for e in range(1, n_events + 1):
            event_indices = np.where(events == e)[0]
            event_starts.append(event_indices[0])
            event_ends.append(event_indices[-1])
        
        # Link heatwaves that occur before and after a short gap (no longer than maxGap)
        if len(event_starts) > 1:  # Need at least 2 events to have a gap
            # Calculate gap length for each consecutive pair of events
            gaps = np.array(event_starts[1:]) - np.array(event_ends[:-1]) - 1
            
            # While there are gaps shorter than or equal to maxGap
            while len(gaps) > 0 and gaps.min() <= maxGap:
                # Find first short gap
                ev = np.where(gaps <= maxGap)[0][0]
                
                # Extend first MHW to encompass second MHW (including gap)
                event_ends[ev] = event_ends[ev+1]
                
                # Remove second event from record
                del event_starts[ev+1]
                del event_ends[ev+1]
                
                # Recalculate gaps
                if len(event_starts) > 1:
                    gaps = np.array(event_starts[1:]) - np.array(event_ends[:-1]) - 1
                else:
                    gaps = np.array([])
        
        # Process each event after merging
        event_durations = []
        event_intensities = []
        
        # Use the merged events instead of the original ones
        for i in range(len(event_starts)):
            # Get start and end indices for this event
            start_idx = event_starts[i]
            end_idx = event_ends[i]
            event_indices = np.zeros(n_times, dtype=bool)
            event_indices[start_idx:end_idx+1] = True
            
            # Skip events that are too short
            if np.sum(event_indices) < minDuration:
                continue
            
            # Calculate duration and mean intensity
            duration = np.sum(event_indices)
            # Only consider actual heatwave days (not gap days) for intensity calculation
            heatwave_days = event_indices & mhw_bool
            if np.sum(heatwave_days) > 0:  # Make sure we have valid heatwave days
                intensity = np.mean(ts[heatwave_days] - clim_full[heatwave_days])
            else:
                continue  # Skip if no valid heatwave days
            
            event_durations.append(duration)
            event_intensities.append(intensity)
        
        # Calculate metrics across all events
        if len(event_durations) > 0:
            # Frequency: events per year
            n_years = (end_year - start_year + 1)
            mhw_freq[n] = len(event_durations) / n_years
            
            # Average duration
            mhw_dur[n] = np.mean(event_durations)
            
            # Average intensity
            mhw_int[n] = np.mean(event_intensities)
    
    # Create output dictionary
    mhw_metrics = {
        'frequency': mhw_freq,
        'duration': mhw_dur,
        'intensity': mhw_int
    }
    
    return mhw_metrics

# Filter mesh nodes to only include Arctic points
def filter_arctic_nodes(mesh, lat_threshold=60.0):
    """
    Filter mesh nodes to only include points in the Arctic region
    
    Parameters:
    -----------
    mesh : pyfesom2.load_mesh object
        FESOM2 mesh object
    lat_threshold : float
        Minimum latitude to consider as Arctic (default: 60.0°N)
    
    Returns:
    --------
    arctic_nodes : numpy.ndarray
        Boolean array indicating which nodes are in the Arctic
    """
    return mesh.y2 >= lat_threshold

# Process temperature data in chunks of nodes for multiple depth layers
def process_temp_data_in_chunks(mesh, years, arctic_nodes, depth_layers, chunk_size=30, parallel=True, n_workers=4):
    """
    Process temperature data in chunks of nodes for multiple depth layers
    
    Parameters:
    -----------
    mesh : pyfesom2.load_mesh object
        FESOM2 mesh object
    years : list
        List of years to process
    arctic_nodes : numpy.ndarray
        Boolean array indicating which nodes are in the Arctic
    depth_layers : list
        List of depth layer indices to process
    chunk_size : int
        Number of nodes to process in each chunk
        
    Returns:
    --------
    results : dict
        Dictionary containing MHW metrics for each layer and node
    """
    # Get indices of Arctic nodes
    arctic_indices = np.where(arctic_nodes)[0]
    n_arctic_nodes = len(arctic_indices)
    print(f"Found {n_arctic_nodes} Arctic nodes")
    
    # Create a list of chunks
    n_chunks = (n_arctic_nodes + chunk_size - 1) // chunk_size  # Ceiling division
    node_chunks = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_arctic_nodes)
        node_chunks.append(arctic_indices[start_idx:end_idx])
    
    print(f"Processing {n_arctic_nodes} Arctic nodes in {n_chunks} chunks with chunk size {chunk_size}")
    
    # Create 3D array for results (layer, node, metric)
    metrics = ['frequency', 'duration', 'intensity']
    n_nodes_total = len(mesh.x2)  # Total number of nodes in the mesh
    results = {
        'frequency': np.full((len(depth_layers), n_nodes_total), np.nan),
        'duration': np.full((len(depth_layers), n_nodes_total), np.nan),
        'intensity': np.full((len(depth_layers), n_nodes_total), np.nan)
    }
    
    # We'll extract time information directly from the files to respect native frequency
    # Just prepare empty lists that will be populated when loading data
    time_values = []
    year_array = []
    month_array = []
    
    # Process each depth layer
    for layer_idx, depth_idx in enumerate(depth_layers):
        layer_start_time = time.time()
        print(f"\nProcessing depth layer {depth_idx}")
        
        if parallel:
            # Set up Dask client for parallel processing
            print(f"Setting up Dask cluster with {n_workers} workers for parallel chunk processing")
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            print(f"Dask dashboard available at {client.dashboard_link}")
            
            try:
                # Create delayed tasks for each chunk
                delayed_tasks = []
                for chunk_idx, node_chunk in enumerate(node_chunks):
                    # Use dask.delayed to properly wrap the function for distributed execution
                    task = dask.delayed(process_single_chunk)(mesh, years, chunk_idx, node_chunk, depth_idx)
                    delayed_tasks.append((chunk_idx, node_chunk, task))
                
                # Compute all tasks in parallel
                print(f"Submitting {len(delayed_tasks)} tasks to Dask cluster")
                chunk_results = compute(*[task for _, _, task in delayed_tasks])
                
                # Process results
                for i, (chunk_idx, node_chunk, _) in enumerate(delayed_tasks):
                    if chunk_results[i] is not None:  # Skip chunks that failed
                        chunk_metrics = chunk_results[i]
                        # Store results in the main results dictionary
                        for metric in ['frequency', 'duration', 'intensity']:
                            results[metric][layer_idx, node_chunk] = chunk_metrics[metric]
            
            finally:
                # Clean up Dask resources
                client.close()
                cluster.close()
        else:
            # Sequential processing (fallback option)
            for chunk_idx, node_chunk in enumerate(node_chunks):
                chunk_start_time = time.time()
                print(f"\nProcessing chunk {chunk_idx+1}/{n_chunks} (nodes {node_chunk[0]}-{node_chunk[-1]})")
                
                # Load temperature data for this chunk of nodes and all years
                load_start_time = time.time()
                chunk_data = []
                time_values = []
                year_array = []
                month_array = []
                
                for year in years:
                    try:
                        # Construct the file path for this year
                        file_path = f"{datapath}/temp.fesom.{year}.nc"
                        
                        # Open the netCDF file
                        with nc.Dataset(file_path, 'r') as nc_file:
                            # Extract temperature data for this chunk of nodes at the specified depth
                            print(f"Loading data for year {year}, depth {depth_idx}, nodes {len(node_chunk)}")
                            temp_data = nc_file.variables[temp_var_name][:, depth_idx, node_chunk]
                            chunk_data.append(temp_data)
                            
                            # Extract time values from the file
                            time_var = nc_file.variables['time']
                            
                            # Get time units and convert to datetime objects
                            time_units = time_var.units
                            try:
                                # Use netCDF4's built-in date conversion if available
                                times = nc.num2date(time_var[:], time_units)
                                
                                # Extract year and month for each time step
                                for t in times:
                                    time_values.append(t)
                                    year_array.append(t.year)
                                    month_array.append(t.month)
                            except AttributeError:
                                print("Error: Could not convert times automatically. Using time indices.")
                                exit()
                            
                    except Exception as e:
                        print(f"Error loading data for year {year}, depth {depth_idx}: {e}")
                
                if not chunk_data:
                    print(f"No data was loaded for chunk {chunk_idx+1}. Skipping.")
                    continue
                    
                print(f"Data loading for chunk {chunk_idx+1} completed in {time.time() - load_start_time:.2f} seconds")
                
                # Convert to numpy arrays
                time_array = np.array(time_values)
                year_array = np.array(year_array)
                month_array = np.array(month_array)
                
                # Combine all years of data for this chunk
                combined_chunk = np.vstack(chunk_data)
                print(f"Combined chunk data shape: {combined_chunk.shape}")
                print(f"Time array shape: {time_array.shape}")
                
                # Detect marine heatwaves for this chunk
                detect_start_time = time.time()
                print(f"Detecting marine heatwaves for depth {depth_idx}, chunk {chunk_idx+1}...")
                chunk_metrics = detect_marine_heatwaves(
                    combined_chunk,
                    time_array,
                    year_array,
                    month_array
                )
                print(f"Detection for chunk {chunk_idx+1} completed in {time.time() - detect_start_time:.2f} seconds")
                
                # Store results in the main results dictionary
                for metric in metrics:
                    results[metric][layer_idx, node_chunk] = chunk_metrics[metric]
                
                # Clean up memory
                del combined_chunk, chunk_metrics
                gc.collect()
        
        print(f"Processing depth layer {depth_idx} completed in {time.time() - layer_start_time:.2f} seconds")
    
    return results
@dask.delayed
def process_single_chunk(mesh, years, chunk_idx, node_chunk, depth_idx):
    chunk_start_time = time.time()
    print(f"\nProcessing chunk {chunk_idx} (nodes {node_chunk[0]}-{node_chunk[-1]})")
    
    # Load temperature data for this chunk of nodes and all years
    load_start_time = time.time()
    chunk_data = []
    time_values = []
    year_array = []
    month_array = []
    
    for year in years:
        try:
            # Construct the file path for this year
            file_path = f"{datapath}/temp.fesom.{year}.nc"
            
            # Open the netCDF file
            with nc.Dataset(file_path, 'r') as nc_file:
                # Extract temperature data for this chunk of nodes at the specified depth
                print(f"Loading data for year {year}, depth {depth_idx}, nodes {len(node_chunk)}")
                temp_data = nc_file.variables[temp_var_name][:, depth_idx, node_chunk]
                chunk_data.append(temp_data)
                
                # Extract time values from the file
                time_var = nc_file.variables['time']
                
                # Get time units and convert to datetime objects
                time_units = time_var.units
                try:
                    # Use netCDF4's built-in date conversion if available
                    times = nc.num2date(time_var[:], time_units)
                    
                    # Extract year and month for each time step
                    for t in times:
                        time_values.append(t)
                        year_array.append(t.year)
                        month_array.append(t.month)
                except AttributeError:
                    print("Warning: Could not convert times automatically. Using time indices.")
                    exit()
        except Exception as e:
            print(f"Error loading data for year {year}, depth {depth_idx}: {e}")
            return None
        
    if not chunk_data:
        print(f"No data was loaded for chunk {chunk_idx+1}. Skipping.")
        return None
        
    print(f"Data loading for chunk {chunk_idx+1} completed in {time.time() - load_start_time:.2f} seconds")
    
    # Convert to numpy arrays
    time_array = np.array(time_values)
    year_array = np.array(year_array)
    month_array = np.array(month_array)
    
    # Combine all years of data for this chunk
    combined_chunk = np.vstack(chunk_data)
    print(f"Combined chunk data shape: {combined_chunk.shape}")
    print(f"Time array shape: {time_array.shape}")
    
    # Detect marine heatwaves for this chunk
    detect_start_time = time.time()
    print(f"Detecting marine heatwaves for depth {depth_idx}, chunk {chunk_idx+1}...")
    chunk_metrics = detect_marine_heatwaves(
        combined_chunk,
        time_array,
        year_array,
        month_array
    )
    print(f"Detection for chunk {chunk_idx+1} completed in {time.time() - detect_start_time:.2f} seconds")
    
    # Clean up memory to avoid memory leaks
    del combined_chunk
    gc.collect()
    
    return chunk_metrics

# Main script execution
if __name__ == "__main__":
    script_start_time = time.time()
    print("Starting Marine Heatwave detection for FESOM2 data")
    
    # Filter for Arctic nodes
    filter_start_time = time.time()
    print("Filtering for Arctic nodes...")
    arctic_nodes = filter_arctic_nodes(mesh, lat_threshold=arctic_lat_threshold)
    print(f"Arctic filtering completed in {time.time() - filter_start_time:.2f} seconds")
    
    # Use the configured number of workers for parallel processing
    print(f"Will use {n_workers} workers for parallel processing")
    
    # Process temperature data in chunks for multiple depth layers
    processing_start_time = time.time()
    results = process_temp_data_in_chunks(
        mesh=mesh,
        years=list(range(start_year, end_year + 1)),
        arctic_nodes=arctic_nodes,
        depth_layers=depth_layers,
        chunk_size=chunk_size,
        parallel=True,
        n_workers=n_workers
    )

    print(f"Data processing completed in {time.time() - processing_start_time:.2f} seconds")
    
    # Save results
    save_start_time = time.time()
    output_file = f"{output_path}/mhw_metrics_{start_year}_{end_year}_arctic_multilayer.npz"
    print(f"Saving results to {output_file}")
    
    np.savez(
        output_file,
        frequency=results['frequency'],
        duration=results['duration'],
        intensity=results['intensity'],
        arctic_nodes=arctic_nodes,
        node_x=mesh.x2,
        node_y=mesh.y2,
        depth_layers=np.array(depth_layers),
        parameters={
            'pctile': pctile,
            'smoothPercentileWidth': smoothPercentileWidth,
            'minDuration': minDuration,
            'maxGap': maxGap,
            'climatologyPeriod': climatologyPeriod,
            'arctic_lat_threshold': arctic_lat_threshold
        }
    )
    print(f"Results saved in {time.time() - save_start_time:.2f} seconds")
    
    # Generate visualizations for each layer
    try:
        viz_start_time = time.time()
        print("Generating visualizations...")
        
        for layer_idx, depth_idx in enumerate(depth_layers):
            print(f"Creating plots for depth layer {depth_idx}")
            
            # Create data arrays for visualization
            # We need to handle NaN values properly
            freq_data = np.zeros(len(mesh.x2))
            dur_data = np.zeros(len(mesh.x2))
            int_data = np.zeros(len(mesh.x2))
            
            # Find valid nodes (Arctic nodes with data)
            arctic_indices = np.where(arctic_nodes)[0]
            for i, node_idx in enumerate(arctic_indices):
                freq_data[node_idx] = results['frequency'][layer_idx, node_idx]
                dur_data[node_idx] = results['duration'][layer_idx, node_idx]
                int_data[node_idx] = results['intensity'][layer_idx, node_idx]
            
            # Plot frequency - use basic scatter plot with discrete colorbar
            plt.figure(figsize=(12, 8))
            # Mask out points with no data
            valid_indices = ~np.isnan(freq_data)
            x = mesh.x2[valid_indices]
            y = mesh.y2[valid_indices]
            z = freq_data[valid_indices]
            
            # Determine discrete colormap levels for events/year
            # Find the max value and create discrete bins
            max_events = np.ceil(np.max(z)) if len(z) > 0 else 5
            levels = np.arange(0, max_events + 1, 1)  # Integer bins
            norm = plt.cm.colors.BoundaryNorm(levels, plt.cm.get_cmap('viridis').N)
            
            scatter = plt.scatter(x, y, c=z, s=3, cmap='viridis', norm=norm)
            plt.title(f'Marine Heatwave Frequency (events/year) - Depth {depth_idx}')
            cbar = plt.colorbar(scatter, label='Events/year', ticks=levels)
            plt.savefig(f'{output_path}/mhw_freq_{start_year}_{end_year}_depth{depth_idx}.png')
            plt.close()
            
            # Plot duration - use basic scatter plot
            plt.figure(figsize=(12, 8))
            # Mask out points with no data
            valid_indices = ~np.isnan(dur_data)
            x = mesh.x2[valid_indices]
            y = mesh.y2[valid_indices]
            z = dur_data[valid_indices]
            
            plt.scatter(x, y, c=z, s=3, cmap='viridis')
            plt.title(f'Marine Heatwave Average Duration (time steps) - Depth {depth_idx}')
            plt.colorbar(label='Time steps')
            plt.savefig(f'{output_path}/mhw_duration_{start_year}_{end_year}_depth{depth_idx}.png')
            plt.close()
            
            # Plot intensity - use basic scatter plot
            plt.figure(figsize=(12, 8))
            # Mask out points with no data
            valid_indices = ~np.isnan(int_data)
            x = mesh.x2[valid_indices]
            y = mesh.y2[valid_indices]
            z = int_data[valid_indices]
            
            plt.scatter(x, y, c=z, s=3, cmap='viridis')
            plt.title(f'Marine Heatwave Average Intensity (\u00b0C) - Depth {depth_idx}')
            plt.colorbar(label='Temperature (\u00b0C)')
            plt.savefig(f'{output_path}/mhw_intensity_{start_year}_{end_year}_depth{depth_idx}.png')
            plt.close()
        
        print(f"Visualizations saved to output directory in {time.time() - viz_start_time:.2f} seconds")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    total_time = time.time() - script_start_time
    print(f"Marine Heatwave detection complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
