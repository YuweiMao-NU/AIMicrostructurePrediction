# An AI-driven framework for time-series processing-microstructure-property prediction
This repository contains the code for performing microstrucutre prediction using AI based on processing parameters as input. The code provides the following functions:

- Train an autoencoder model on a given dataset.
- Simulator code for data generation.
- Predict microstructure using a pre-trained autoencoder model.
- Property calculation and linear bias model correction method for improveing predicted property accuracy.

## Installation Requirements
The basic requirement for using the files is a Python 3.8.19 environment with PyTorch 2.3.0

## Source Files
Here is a brief description of the files and folder content:

- Simulator: folder where includes all files for simulation to generate dataset.
- autoencoder_v0.py: code to train and test autoencoder model.
- linear_add.py: code to calculate property and linear bias model correction method for improveing predicted property accuracy.

## Running the code
To generate the dataset, run process.m located in the Simulator folder. Next, train and test the autoencoder model by executing autoencoder_v0.py. Finally, calculate the property by running linear_add.py.

## Developer Team
The code was developed by Yuwei Mao from the [CUCIS](http://cucis.ece.northwestern.edu/) group at the Electrical and Computer Engineering Department at Northwestern University.

## Disclaimer
The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: yuweimao2019@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support
This work was supported primarily by National Science Foundation (NSF) CMMI awards 2053929/2053840. Partial support from NIST award 70NANB19H005, NSF award OAC-2331329, DOE award DE-SC0021399, and Northwestern Center for Nanocombinatorics is also acknowledged.






