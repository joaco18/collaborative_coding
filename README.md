# E-Health - Collaborative Coding
## Cansu - Hasna - Joaquín


\
&emsp;This repository contains the code for Brain Segmenter, the main product of the company Medvision.


---
## 1. Pipeline outline

&emsp; Brain Segmenter is a pipeline to segment Brain MRI scans in four different different regions: 
-  White Matter (WM)
-  Gray Matter (WM)
-  Cerebro-Spinal Fluid (CSF)
-  Background + Bone + Extra-Skull-Soft-Tissue

&emsp;The pipeline general strructure is the following:
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


## 2. Benchmarks:
Checkploint file | Model  | CSF | WM | GM 
---------------- | -----  | --- | -- |--- 
checkpoint_def.pkl | EM - init:TM - atlas:mni/after | 0.463 | 0.805  | 0.895

## 3. Detailed pipeline description

Detailed desc

## 4. Instructions for contributers

&emsp; The presented pipeline can be fully reproduced locally. Below we provide  BASH commands, which can be run in an Unix/Unix-like OS (Mac OS, GNU-Linux) and CMD comands for windows user (¬¬ consider changing to GNU-Linux, your life will be better).

### 4.1 Setting up the environment

- Create the environment

    > Unix:
    ```bash
    conda create -n medvision python==3.9.13 anaconda -y &&
    conda activate medvision
    ```

    >Windows:
    ```bash
    conda create -n medvision python==3.9.13 anaconda -y && conda activate medvision
    ```

- Install requirements
    >Both OS:
    ```bash
    pip install -r requirements.txt
    ```

- Add current repository path to PYTHONPATH

    > Unix:
    ```bash
    export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
    ```

    > Windows:
    ```bash
    set PYTHONPATH=%PYTHONPATH%;/path/to/your/project/
    ```


### 4.2 Run pipeline with an example and reproduce our results
&emsp; To reproduce our results over one example run the following commands:

-  4.2.1 **Download the example image**
    > Unix:
    ```bash
    cd data/ &&
    gdown https://drive.google.com/uc?id=1MlypAmSqJIcYi7HRewPFeutDUL3nx2Fx &&
    unzip example_image.zip &&
    rm example_image.zip &&
    cd ../
    ```

    > Windows:
    ```bash
    cd data/ && gdown https://drive.google.com/uc?id=1MlypAmSqJIcYi7HRewPFeutDUL3nx2Fx && tar -xf example_image.zip && del example_image.zip && cd ..
    ```

    > Alternative:

    &emsp; If you can't or don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

    &emsp; The directories should be collaborative_coding/data/example_img/[content of example_img zip]


- 4.2.2 **Download the model checkpoints**

    > Unix:
    ```bash
    cd models/ &&
    gdown https://drive.google.com/uc?id=1bNROHoZSQgwaqF-w0NwWyFHyv3H1tXWx &&
    unzip checkpoints.zip &&
    rm checkpoints.zip &&
    cd ../
    ```

    > Windows:
    ```bash
    cd models/ && gdown https://drive.google.com/uc?id=1bNROHoZSQgwaqF-w0NwWyFHyv3H1tXWx && tar -xf checkpoints.zip && del checkpoints.zip && cd ..
    ```

    > Alternative:
    
    &emsp; If you can't or don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

    &emsp; The directories should be collaborative_coding/models/checkpoints/[content of checkpoints zip]

- 4.2.3 **Run the pipeline**

    ```bash
    python brain_segmenter.py --ip [PATH] --chkpt [PATH] --op [PATH]
    ```

    Example:

    ```bash
    python brain_segmenter.py --ip /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/example_image/1003/1003.nii.gz --chkpt /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/models/checkpoints/checkpoint_def.pkl --op /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/example_image/1003
    ```

### 4.3 Run pipeline as developer
- 4.3.1 **Download the database**

    > Unix:
    ```bash
    mkdir -p data &&
    cd data/ &&
    gdown https://drive.google.com/uc?id=1-o0pSnkKytqoqaqsW472Ze1wj4ccjD3y &&
    unzip data.zip &&
    rm data.zip &&
    cd ../
    ```
    > Windows:
    ```bash
    mkdir data && cd data/ && gdown https://drive.google.com/uc?id=1-o0pSnkKytqoqaqsW472Ze1wj4ccjD3y && tar -xf data.zip && del data.zip && cd ..
    ```

    > Alternative:

    &emsp; If you don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1pyl_sBiLhhxCFA4bZiXIgZy3_g5hpB78?usp=share_link)

    &emsp; The directories should be collaborative_coding/data/[content of data zip]

- 4.3.2 **Run a experiment**
    1. Modify experiments/train_config.yaml accordingly.
        
        Copy the .example file, remove the .example from the name and modify the file
    2. Run experiments/train.py
        ```bash
            python experiments/train.py
        ```
- 4.3.3 **Test on the complete test dataset**
    1. Modify experiments/test_config.yaml accordingly. 
        
        Copy the .example file, remove the .example from the name and modify the file
    2. Run experiments/test.py
        ```bash
            python experiments/test.py
        ```

- 4.3.4 **Get quick analysis of available experiments**
    1. Run analysis/analysis.py

        ```bash
        python analysis/analysis.py --rf [PATH]/experiments/test_results --op [PATH]/data/comp_imgs --exp [LIST OF EXPERIMENTS] --cases [LIST OF 5 CASES]
        ```

        &emsp; Example:

        ```bash
        python analysis/analysis.py --rf /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/experiments/test_results --op /home/jseia/Desktop/MAIA/classes/spain/ehealth/lab/collaborative_coding/data/comp_imgs --exp try_01 try_02 --cases 1025 1024 1104 1110 1003
        ```

### 4.4 Recommendations to developers

- The code in Medvision is developed following:
    - numpy docstring format
    - flake8 linter
    - characters per line: 100