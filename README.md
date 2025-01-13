# An AI-driven framework for time-series processing-microstructure-property prediction
This repository contains the code for performing microstrucutre prediction using AI based on processing parameters as input. The code provides the following functions:

- Train an seq2seq model on a given dataset.
- Simulator code for data generation.
- Predict microstructure using a pre-trained model.

## Installation Requirements
The basic requirement for using the files is a Python 3.8.19 environment with PyTorch 2.3.0

## Source Files
Here is a brief description of the files and folder content:

- Simulator: folder where includes all files for simulation to generate dataset.
- train.py: code to train the seq2seq model.
- test.py: code to test the seq2seq model.
- difH.py: code for analyzing different history H for the seq2seq model.

## Running the code
To generate the dataset, run process.m located in the Simulator folder. Next, train and test the seq2seq model by executing train.py and test.py. Run difH.py if you want to analyze different history H for seq2seq model.

## Developer Team
The code was developed by Yuwei Mao from the [CUCIS](http://cucis.ece.northwestern.edu/) group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication
1. Mao, Yuwei, Mahmudul Hasan, Claire Lee, Muhammed Nur Talha Kilic, Vishu Gupta, Wei-Keng Liao, Alok Choudhary, Pinar Acar, and Ankit Agrawal. "A Deep Learning Framework for Time-Series Processing-Microstructure-Property Prediction." In 2023 International Conference on Machine Learning and Applications (ICMLA), pp. 890-893. IEEE, 2023. [PDF](https://ieeexplore.ieee.org/abstract/document/10459976)
2. 
## Disclaimer
The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: yuweimao2019@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support
This work was supported primarily by National Science Foundation (NSF) CMMI awards 2053929/2053840. Partial support from NIST award 70NANB19H005, NSF award OAC-2331329, DOE award DE-SC0021399, and Northwestern Center for Nanocombinatorics is also acknowledged.






