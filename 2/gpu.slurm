#!/bin/bash
#SBATCH --job-name=IN5550-Oblig2-PST
#SBATCH --account=ec403
#SBATCH --time=00:30:00      # NOTE: make sure to adjust requested time to actual needs
#SBATCH --partition=accel    # To use the accelerator nodes, i.e., with access to gpus
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=out/test-%j.out

# NB: this script should be run with "sbatch gpu.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

# Loading all the user settings from ~/.bashrc file, just in case
source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# modules suggested in assignment text (section 1)
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-accelerate/1.9.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
# if we need pytorch (unless it is included in one of the other ones?)
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 ${@} # placeholder, we need to define a main script for the assignment
