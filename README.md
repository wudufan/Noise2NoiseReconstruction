# Noise2Noise Reconstruction

This is the code which incorporate Noise2Noise network fine-tuning into iterative reconstruction at testing time. 

## Prerequisite

- Miniconda, the environment is given in env.yml
- https://github.com/wudufan/ReconNet for the reconstruction backend. We are mitigating everything to a new framework https://github.com/wudufan/CTProjector, but please use the old repo at the moment. 

## Structures
### preprocessing
./preprocessing provides codes to generate training/testing dataset from the rebinned projection data from 2016 Low-dose CT Challenge. 

### eval
./eval provides evaluation and plotting scripts. 

### Other
Each folder gives the reconstruction scripts for the corresponding algorithm. 

The Script folder under each gives the bash and slurm scripts to call the python scripts. 