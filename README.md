# VAE Library for Molecular Simulation

A library of tensorflow 2.0 code to support VAEs for molecular systems.
The original code used disentanglement_lib (https://github.com/google-research/disentanglement_lib) as a jumping off point for organization and basic structure of models.
While much of that code remains, the majority of the code here is newly implemented VAE models adding on features like autoregressive decoders, and normalizing flows that can be applied to both the prior and the decoding distribution.
These developments are helpful in working with molecular systems, where enhanced latent space complexity based on a learned prior is helpful, and increased modeling power in terms of capturing correlations between continuous degrees of freedom is often necessary in the decoder.

This code was utilized in the following publications:

Monroe, J. I.; Shen, V. K. Learning Efficient, Collective Monte Carlo Moves with Variational Autoencoders. J. Chem. Theory Comput. 2022, 18 (6), 3622-3636. https://pubs.acs.org/doi/10.1021/acs.jctc.2c00110

Monroe, J. I.; Shen, V. K. Systematic Control of Collective Variables Learned from Variational Autoencoders. J. Chem. Phys. In review.

If you find this code useful plesae consider citing those works as appropriate.


## Dependencies and Installation

See environment.yml for necessary dependencies, as well as the historic_environment.yml for a full conda environment with specific package versions used at the time when this code was used for publications.
To install, simply pull this repository and within the libVAE directory run `conda env create --file environment.yml`.
To switch to this environment once it's installed, run `conda active machinelearning`.
Note that you will also need to make sure that the libVAE directory is in your path and python path.
This can be accomplished by appending the path to your `PATH` and `PYTHONPATH` environment variables.

Note that specific systems (the 2D fluid with a dimer and the Mueller potential) require installation of the deep-boltzmann package: http://doi.org/10.5281/zenodo.3242635
Once the files for that package are pulled, it may be installed into the conda environment crated with the provided environment.yml file by running `pip install` in the directory containing setup.py for the deep-boltzmann package.


## Contact

Please feel free to interact with Jacob Monroe (JIMonroe) through github, or send an email to jacob.monroe@nist.gov.

