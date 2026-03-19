@echo off
set CASTLE_BACKEND=pytorch

if "%1"=="" (
    echo run_discovery v1.1
    echo usage: sh ./run_discovery.sh [algo_name] [hdf5_file_path] [result_dir_path]
    echo   [algo_name]: one of the names used as key in the file 'algorithms.json', '*' or 'all' to use all algorithms
    echo   [hdf5_file_path]: relative or full path of a 'dataset-*-0.hdf5' file
    echo   [result_dir_path]: relative or full path of a directory where the results will be saved
    echo The default are defined in the Python variables ROOT and RESULTS inside the file '2_causal_discovery'
)

rem mkdir log 2>/dev/null

python 2_causal_discovery.py %*

