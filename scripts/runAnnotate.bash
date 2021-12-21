#!/bin/bash

#SBATCH --job-name=annotateDC2
#SBATCH --ntasks=1
#SBATCH -L sps

source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2021_44/loadLSST.bash
setup lsst_distrib

cd /pbs/throng/lsst/users/boutigny/ML/MLClusters
./python/annotateDC2.py
