#!/bin/bash
#SBATCH --account=c.mio                        # Account
#SBATCH --job-name=causal-discovery            # Job name
#SBATCH --partition=only-one-gpu               # Partition name (e.g. only-one-gpu,ulow)
# RESOURCES
#SBATCH --ntasks=1                             # How many tasks
#SBATCH --cpus-per-task=4                      # How many MPI cores per task
#SBATCH --mem=16G                              # Job memory request
#SBATCH --gres=gpu:1                           # How many GPUs (0..1)
#SBATCH --time=23:59:59                        # Time limit hrs:min:sec
# OUTPUT FILES
#SBATCH --output=run_%x_%j.log                 # Standard output and error log, with job name and id
# NOTIFICATION EMAILS
#SBATCH --mail-type=ALL                        # Valid types are: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=corrado.mio@gmail.com
#______ --mail-user=c.mio@unimib.it            # User to receive email notifications

### Definitions
#export BASEDIR="<project>/<experiment>"
#export SHRDIR="/scratch_share/<group_name>/`whoami`"
#export LOCDIR="/scratch_local"
#export TMPDIR=$SHRDIR/$BASEDIR/tmp_$SLURM_JOB_NAME_$SLURM_JOB_ID

export BASEDIR="Projects/article_causal_discovery_bool"
export SHRDIR="$HOME/$BASEDIR"
export LOCDIR="$HOME/$BASEDIR"
export TMPDIR="$HOME/$BASEDIR/tmp_$SLURM_JOB_NAME_$SLURM_JOB_ID"


### File System Setup
#cd $HOME/$BASEDIR                  # use a folder in home directory
#cd $SHRDIR/$BASEDIR                # use a folder in scratch_share
#mkdir -p $TMPDIR                   # create a folder for temporary data
#cp $HOME/<input_data> $TMPDIR      # copy input data to temp folder

cd $HOME/$BASEDIR
mkdir -p $TMPDIR

### Header
pwd; hostname; date    #prints first line of output file

### Software dependencies
#module purge              # unloads every module
#module load <software>    # load dependencies

### Executable script
#
# Your code goes here...
#
export PYTHON=$HOME/miniconda3/envs/causal/bin/python
#export PYTHON=python

export CASTLE_BACKEND=pytorch

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Python=$PYTHON"
echo "Arguments=$*"
echo "Eureka! algo=$1, dataset=$2.hdf5"

$PYTHON 2_causal_discovery.py $1 ../article_causal_discovery_bool_data/datasets/$2.hdf5 ../article_causal_discovery_bool_data/results

### File system cleanup
#cp $TMPDIR/<output_data> $HOME/$BASEDIR/job_logs/    # copy output data to output folder
rm -r $TMPDIR                                         #clean temporary data

### Footer
date    #prints last line of output file
