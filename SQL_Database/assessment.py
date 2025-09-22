import os
import glob
import time
import tracemalloc
import sys
sys.path.append('..')  # Allow import from parent directory
from ExonPTMapper import config, mapping

def get_file_size(file):
    size_bytes = os.path.getsize(file)
    size_mb = size_bytes / (1024 * 1024)  # Convert bytes to megabytes
    return size_mb

def get_folder_size(folder_path, file_type = '*', files_to_ignore = []):
    total_size = 0
    for file_path in glob.glob(os.path.join(folder_path, file_type)):
        if os.path.isfile(file_path) and os.path.basename(file_path) not in files_to_ignore:
            total_size += os.path.getsize(file_path)
    total_size_mb = total_size / (1024 * 1024)  # Convert bytes to megabytes
    return total_size_mb

def time_function(func, num_calls = 1000, *args, **kwargs):
    # time function and assesses memory usage
    tracemalloc.start()
    start_time = time.time()
    for i in range(num_calls):
        result = func(*args, **kwargs)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for {num_calls} calls: {elapsed_time} seconds")
    return elapsed_time