#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun nsys profile --stats=true ./m2 10000 > profile.out
srun ncu -f --set full -o analysis_file ./m2 10000 
{ time srun ./m2 100 > m2_100.out; } 2> time_m2_100.out
{ time srun ./m2 1000 > m2_1000.out; } 2> time_m2_1000.out
{ time srun ./m2 > m2_10000.out; } 2> time_m2_10000.out

# Generate nsys profile information, output will be in profile.out
# srun nsys profile --stats=true ./m2 > profile.out
