
<h1 align="center">Tutorial - Activity Guideline</h1>

<h2 align="center"> E-Health - Collaborative Coding </h2>

<h2 align="center">Cansu - Hasna - Joaquín</h2>

---

&emsp; In this file we present you the activity to perform during the tutorial time of our presentation. The general outline of the activity is:
- You will first set up the repository tacking advantage of reproducibility good practices. 
- You will make some minor code modifications. 
- Each group will test a brain segmentation method. 
- The code changes you will have made plus the results of your testing experiment will be merged to dev.
- We will compare together the different results.
- Put the best model "in production" by modifying the example checkpoint file and merging the best model choice to main.

---
## 1. Set up the repository

&emsp; Got the repository page of [collaborative_coding](https://github.com/joaco18/collaborative_coding). Clone the repository to a known location in your laptop:

> Any OS:
```git
git clone https://github.com/joaco18/collaborative_coding.git
```

&emsp; Follow the [README.md](https://github.com/joaco18/collaborative_coding#readme) to set up the working environment and get the data locally in your machine.

## 2. Get familiarized with the project

&emsp; Follow the [README.md](https://github.com/joaco18/collaborative_coding#readme). When the general structure of the project is presented, go roughly trough the code folders, have a general idea of what can be found in each folder/.py file.

&emsp; A working version of the code is provided in the _main_ branch, just in case you want to try sth out.
> Any OS:
```git
git checkout main
```

## 3. Create your working branch from dev

&emsp; Checkout to _dev_ branch, we are going to branch from that one.
> Any OS:
```git
git checkout dev
```

&emsp; Create a new branch with the name "groupN" were N should be replaced with the group number you've been assigned.
> Any OS:
```git
git checkout -b groupN
```

## 3. Fix the code assigned to your team:
&emsp; Please limit yourselves to fix the indicated code pieces so that we can get a toy example of the plausibility of parallel code editing and contribution.

- ### Group 1:
    - There is some missing content in the Kmeans initialization of the EM algorithm.

    - Check the file [models/em.py](../models/em.py). In lines around 135 you will have to complete KMeans models instantiation.
    
    - Using the atributes of the class ExpectationMaximization. You need to set the number of clusters to the number of components we use in EM and you need to fix the random state of KMeans to the defined seed for EM.

- ### Group 2:
    - There is some missing content in the use of an atlas inside EM algorithm in the INTO mode during the expectation phase.

    - Check the file [models/em.py](../models/em.py). In lines around 268 you will have to complete the expectation method of the class ExpectationMaximization.
    
    - Using the atributes of the class ExpectationMaximization. You need to make the poseterior probabilities (weights in MISA lexicon) be equal to the posteriors multiplied by the atlas map probability weights. For the atlas weights, check the atributes of ExpectationMaximization.

- ### Group 3:
    - There is some missing content in the use of an atlas inside EM algorithm in the AFTER mode during the expectation phase.

    - Check the file [models/em.py](../models/em.py). In lines around 270 you will have to complete the expectation method of the class ExpectationMaximization.
    
    - Using the atributes of the class ExpectationMaximization. You need to make the poseterior probabilities (weights in MISA lexicon) be equal to the posteriors multiplied by the atlas map probability weights. For the atlas weights, check the atributes of ExpectationMaximization.

- ### Group 4:
    - There is some missing content in the use of an atlas as initialization for EM algorithm.

    - Check the file [models/em.py](../models/em.py). In lines around 147.
    
    - You need to set the value of the posteriors atribute of class ExpectationMaximization as equal to the atlas probability map values. 
    
    - For the atlas map weights, check the atributes of ExpectationMaximization use the proper atribute.

- ### Group 5:
    - There is some missing content in the analysis function (which is going to mock our company's performances dashbord) that summarizes the results of the run experiments.

    - Check the file [analysis/analysis.py](../analysis/analysis.py). In lines around 55 and 59.
    
    - You need to call two plotting functions which are already defined in [utils/plots.py](../utils/plots.py). Have in mind that the plots module from utils has already been imported and that the funtion arguments are also provided.


## 4. Make sure your changes work by running an experiment

&emsp; Once the code has been modified (in a real case you may have added a new functionality or solved a bug), you need to first check that the general pipeline is working and that is safe to merge your changes to dev.

&emsp; In our case we will do this by running a test experiment. By doing so we will do two things at the same time:

- Be sure that our code is running
- Test one method of having brain segmentations.

&emsp; To do so you will need to check the file [experiments/test_config.yaml.example](../experiments/test_config.yaml.example):

- Copy the indicated file and remove the _.example_ extension from the name.
    > Unix
    ```bash
    cp experiments/test_config.yaml.example experiments/test_config.yaml
    ```
    > Windows
    ```bash
    copy experiments/test_config.yaml.example experiments/test_config.yaml
    ```
    > Alternative

    &emsp; Copy the file _test_config.yaml.example_ in the same directory where it is and when pasting rename it as _test_config.yaml_

&emsp; Before running any experiment, check the content of the configuration file [experiments/test_config.yaml](../experiments/test_config.yaml):

- Understand the configuration variables provided there. Here we provide you some guiding questions you should be able to answer:
    
    - If you want to modify the intialization method, which field would you change?
    - If you want to select a set of cases from the whole dataset to run an experiment, how would you do that?
    - If you want to use an atlas inside EM, how would you indicate that?
    - If you want to use an atlas inside EM, how would you select which atlas do you want to use?

&emsp; Now that we have some idea of how the config file works, lets run the experiment.

- Modify the file [experiments/test_config.yaml.example](../experiments/test_config.yaml.example) according to the experiment assigned to your group:
    -  Group 1:
        
        Run an experiment using Kmeans as initialization for EM. Don't use the atlas in any way. Use the cases: ['1025', '1024', '1104', '1110', '1003']. Modify the experiment name to exp_group1.
    
    -  Group 2:
        
        Run an experiment using tissue models as initialization for EM. Use the Medvision atlas in INTO mode. Use the cases: ['1025', '1024', '1104', '1110', '1003']. Modify the experiment name to exp_group2.

    -  Group 3:
        
        Run an experiment using tissue models as initialization for EM. Use the Medvision atlas in AFTER mode. Use the cases: ['1025', '1024', '1104', '1110', '1003']. Modify the experiment name to exp_group3.

    -  Group 4:
        
        Run an experiment using medvision atlas as initialization for EM. Don't use the atlas inside EM. Use the cases: ['1025', '1024', '1104', '1110', '1003']. Modify the experiment name to exp_group4.
    
    -  Group 5:
        
        Run an experiment using random initialization for EM. Don't use the atlas inside EM. Use the cases: ['1025', '1024', '1104', '1110', '1003']. Modify the experiment name to exp_group5.
    
- Run the experiment:
    > Any OS
    ```bash
    python experiments/test.py
    ```
- Go to the results directory _experiments/test_results/exp_groupN_. Copy the ckeckpoint file of the model you trained inside [models/checkpoints/](../models/checkpoints/) directory under the name _checkpoint_groupN.pkl_

- Update the [README.md](../README.md) in Section 2. Benchmarks. To include your results. Example:

    Checkploint file | Model  | CSF | WM | GM 
    ---------------- | -----  | --- | -- |--- 
    checkpoint_def.pkl | EM - init:TM - atlas:mni/after | 0.463 | 0.805  | 0.895
    checkpoint_groupN.pkl | EM - init:KM - atlas:mv/into | 0.3 | 0.7  | 0.6

## 5. Commit your changes and make a pull request to dev.

&emsp; Now that we are sure that our changes worked. We need to commit the changes locally and then merge them to dev branch.

- Check you changes:
    > Any OS
    ```git
    git status
    ```
- One by one stage your modified files and commit the change with a meaninful commit message:
    > Any OS
    ```git
    git add models/em.py
    ```
    ```git
    git commit -m 'Fixed problems in Kmenas initialization'
    ```

- Once you have all your files commited, make sure you include the latest changes in dev (and potentially solve any conflict) are included in your branch before doing any pull request:
    > Any OS
    ```git
    git checkout dev
    ```
    ```git
    git pull
    ```
    ```git
    git checkout groupN
    ```
    ```git
    git merge dev
    ```

- Push your branch chages to the remote repository:
    > Any OS
    ```git
    git push --set-upstream origin groupN
    ```

- Go to the repository github page and go into the "pull request" tab (link: [collaborative_coding](https://github.com/joaco18/collaborative_coding/pulls)). Click on "New pull request". Choose to do a pull request from _groupN_ branch to _dev_ branch. Click on "Create pull request". In reviewers section add Joaquín as reviewer, and click again on "Create pull request". 

- Well done! You finished a full developing cycle. As a final task, add the [models/checkpoints/checkpoint_groupN.pkl](../models/checkpoints/) file to [this drive folder](https://drive.google.com/drive/folders/1y7xGO4_MhJiiNN8ZOiOSF5zPiXOWv1IJ?usp=share_link) 

## 5. Lets compare toghether the results and put our model in production.
