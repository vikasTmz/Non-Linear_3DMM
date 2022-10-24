# README

*The template of this project was inspired from : https://github.com/autonomousvision/texture_fields*

## Installation

````
conda env create -f environment.yaml
conda activate nonlinear3dmm
````

````
python auxiliary/ChamferDistancePytorch/chamfer3D/setup.py install #MIT
cd auxiliary
git clone https://github.com/ThibaultGROUEIX/metro_sources.git
cd metro_sources; python setup.py --build # build metro distance #GPL3
cd ../..
````

Source: https://github.com/ThibaultGROUEIX/AtlasNet


## Usage

**Train**:

````
python train.py config/<choose a config file>

````

**Test**:

Make sure the right model evaluation mode is set in the config file and then run the below.

````
python generate.py config/<choose a config file>

````