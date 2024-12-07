#!/bin/bash -l

#PBS -A SEEr-Polaris
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -N OVER1_4
#PBS -m bea
#PBS -M rstrin4@uic.edu
#PBS -V
#PBS -l select=4:ncpus=4:ngpus=4
#PBS -r y

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:/home/tartarughina/libjpeg/lib64:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Number of GPUs per rank --> 1:1
ngpus=1
# Number of ranks per node
nranks=4
# Medium batch size is the only one picked for multinode tests
batch=2048
# Log dir path
log_path="/home/tartarughina/TCBNN/benn_over_log"

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi

NNODES=$(wc -l < $PBS_NODEFILE)
NTOTRANKS=$(( NNODES * nranks ))

cd /home/tartarughina/TCBNN

# Execute 5 times
for i in {1..5}; do
    mpiexec --envall --np "${NTOTRANKS}" --ppn "${nranks}" --hostfile "$PBS_NODEFILE" --cpu-bind list:0,8,16,24 ./benn_nccl -b "${batch}" -u --oversub 1 > "${log_path}/OVER1_gpus_${NTOTRANKS}_batch_${batch}_iter_${i}.txt"
done
