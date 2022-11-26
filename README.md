# E-Health - Collaborative Coding
## Cansu - Hasna - Joaquín


\
This repository contains the code for Brain Segmenter, the main product of the company Medvision.

## 1. Pipeline outline

Brain Segmenter is a pipeline to segment Brain MRI scans in four different different regions: 
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
- Segmentation:
    - Reorganizing data
    - Expectation Maximization voxel-classification
- Post-Processing:
    - Brain reconstruction
    - Label matching
    - Image storing


## Benchmarks:
Checkploint file | Model  | CSF | WM | GM 
---------------- | -----  | --- | -- |--- 
checkpoint_def.pkl | EM - init:TM - atlas:mni/after | 0.463 | 0.805  | 0.895

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
ACCESS_TOKEN="ya29.a0AeTM1icvBOlgug4XScuA5WghTwd22DxvMAsox2AKWN0JiTG6yXJCJ8YRqhvdnnLhHewSEGThbc8H1T_oAlXA7aCrCQmlzP3R0lPYrm1YJ_W4FibNPfjF2egK3aKHraIPy0eUCFG-TUr8e98TZ_Z1cRmk-MjmaCgYKAVYSARISFQHWtWOmTKHjja4Xo914ZbiwVkAYdA0163" &&
FILE_ID="1MlypAmSqJIcYi7HRewPFeutDUL3nx2Fx" && 
curl -H "Authorization: Bearer ${ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/${FILE_ID}?alt=media -o example_img.zip &&
unzip example_img.zip &&
rm example_img.zip &&
cd ../
```
If you don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

The directories should be collaborative_coding/data/example_img/[content of example_img zip]

#### 3.2.1 Download the model checkpoints
```bash
cd models/ &&
ACCESS_TOKEN="ya29.a0AeTM1icvBOlgug4XScuA5WghTwd22DxvMAsox2AKWN0JiTG6yXJCJ8YRqhvdnnLhHewSEGThbc8H1T_oAlXA7aCrCQmlzP3R0lPYrm1YJ_W4FibNPfjF2egK3aKHraIPy0eUCFG-TUr8e98TZ_Z1cRmk-MjmaCgYKAVYSARISFQHWtWOmTKHjja4Xo914ZbiwVkAYdA0163" &&
FILE_ID="1bNROHoZSQgwaqF-w0NwWyFHyv3H1tXWx" && 
curl -H "Authorization: Bearer ${ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/${FILE_ID}?alt=media -o checkpoints.zip &&
unzip checkpoints.zip &&
rm example_img.zip &&
cd ../
```
If you don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

The directories should be collaborative_coding/data/models/checkpoints/[content of checkpoints zip]

#### 3.2.2 Run the pipeline
```bash
python brain_segmenter.py -ip [PATH] --chkpt [PATH] -op [PATH]
```

example:

```bash
python brain_segmenter.py --ip /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/example_image/1003/1003.nii.gz --chkpt /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/models/checkpoints/checkpoint_def.pkl --op /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/example_image/1003
```

### 3.2 Run pipeline as developer
#### 3.2.1 Download the database
```bash
mkdir data &&
cd data/ &&
mkdir data &&
cd data/ &&
ACCESS_TOKEN="ya29.a0AeTM1icvBOlgug4XScuA5WghTwd22DxvMAsox2AKWN0JiTG6yXJCJ8YRqhvdnnLhHewSEGThbc8H1T_oAlXA7aCrCQmlzP3R0lPYrm1YJ_W4FibNPfjF2egK3aKHraIPy0eUCFG-TUr8e98TZ_Z1cRmk-MjmaCgYKAVYSARISFQHWtWOmTKHjja4Xo914ZbiwVkAYdA0163" &&
FILE_ID="1-o0pSnkKytqoqaqsW472Ze1wj4ccjD3y" &&
curl -H "Authorization: Bearer ${ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/${FILE_ID}?alt=media -o data.zip &&
unzip data.zip &&
rm data.zip &&
cd ../../
```
If you don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

The directories should be collaborative_coding/data/[content of data zip]

#### 3.2.2 Run a experiment
1. Modify experiments/train_config.yaml accordingly
    Copy the .example file, remove the .example from the name and modify the file
2. Run experiments/train.py
    ```bash
        python experiments/train.py
    ```
#### 3.2.3 Test on the complete test dataset
1. Modify experiments/test_config.yaml accordingly 
    Copy the .example file, remove the .example from the name and modify the file
2. Run experiments/test.py
    ```bash
        python experiments/test.py
    ```

#### 3.2.4 Get quick analysis of available experiments
1. Run experiments/fast_analysis.py

```bash
    python analysis/analysis.py --ft [PATH]/experiments/test_results --op [PATH]/data/comp_imgs --exp [LIST OF EXPERIMENTS] --cases [LIST OF 5 CASES]
```

Example:

```bash
    python analysis/analysis.py --ft /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/experiments/test_results --op /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/comp_imgs --exp try_01 try_02 --cases 1025 1024 1104 1110 1003
```

#### 3.2.4 Recommendations to developers

The code in Medvision is developed following:
- numpy docstring format
- flake8 lintern
- characters per line: 100