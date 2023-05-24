# Multicalibration and Game Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository implements multicalibration algorithms.

## Table of Contents

- [Copyright](#copyright)
- [Files](#files)
- [Usage Instructions](#usage-instructions)

## Copyright

MIT License
Copyright (c) 2023 [Redacted].
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## Files

This repository contains the following files:

1. `hedge.py`: Contains the implementation of the Hedge algorithm.
2. `multicalibration.py`: Contains the implementation of the multicalibration algorithm.
3. `benchmark.py`: Runs the multicalibration algorithm on a set of generated data.
4. `calibration.py`: Contains utility functions for calibration.
5. `dataset.py`: Contains dataset wrappers.
6. `analysis.py`: Analyze the results of running `benchmark.py`.

## Usage Instructions

1. Clone the repository.
2. Install the necessary dependencies.
3. Run `benchmark.py` to execute the experiment.
4. Run `analysis.py`.

Warning: This script regularly dumps Pickle checkpoints. It can quickly fill up your disk with hundreds of GBs of these small checkpoint files.