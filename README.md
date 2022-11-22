# E-Health - Collaborative Coding
## Cansu - Hasna - Joaquín


\
This repository contains the code for Brain Segmenter, the main product of the company Medvision.

## 1. Pipeline outline

Brain Segementer is a pipeline to segment Brain MRI scans in four different different regions: 
-  White Matter (WM)
-  Gray Matter (WM)
-  Cerebro-Spinal Fluid (CSF)
-  Background + Bone + Extra-Skull-Soft-Tissue

The pipeline general strructure is the following:
- Pre-Processing:
    -  Registering
    -  Intensity Normalization
    -  Voxel Spacing Normalization
    -  Skull Stripping
- Segementation:
    - Reorganizing data
    - Expectation Maximization voxel-classification
- Post-Processing:
    - Brain reconstruction
    - Label matching
    - Image storing

## 2. Detailed pipeline description

Detailed desc

## 3. Instructions for contributers

The presented pipeline can be fully reproduced locally. All the commands below are BASH commands, which can be run in an Unix/Unix-like OS (Mac OS, GNU-Linux). In case you are a windows user, open the Anaconda Prompt and run them there (and consider changing to GNU-Linux, your life will be better).

### 3.1 Setting up the environment

Create the environment
```bash
conda create -n medvision python==3.9.13 anaconda -y &&
conda activate medvision &&
conda update -n medvision conda -y
```

Install requirements
```bash
pip install -r requirements.txt
```
Add current repository path to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
```

### 3.2 Run pipeline an example and reproduce our results
To reproduce our results over one example run the following commands:

#### 3.2.1 Download the example image
```bash
mkdir data &&
cd data/ &&
gdown example_img &&
unzip example_img.zip &&
cd ../
```

#### 3.2.2 Run the pipeline
```bash
mkdir data &&
python brain_segmenter.py -ip [PATH] -op [PATH] &&
```

### 3.2 Run pipeline as developer
#### 3.2.1 Download the database
```bash
mkdir data &&
cd data/ &&
gdown example_img &&
unzip example_img.zip &&
cd ../
```

#### 3.2.2 Run a experiment
1. Modify experiments/train_config.yaml accordingly
2. Run experiments/train.py

#### 3.2.3 Test on the complete test dataset
1. Modify experiments/test_config.yaml accordingly
2. Run experiments/test.py

#### 3.2.4 Get quick analysis of available experiments
1. Run experiments/fast_analysis.py

#### 3.2.4 Recommendations to developers

The code in Medvision is developed following:
- numpy docstring format
- flake8 lintern
- characters per line: 100