# ConCon Dataset
This is code for the paper: ```Where is the Truth? The Risk of Getting Confounded in a Continual World```.

The code includes the implementation of our proposed dataset ConCon which is continually confounded. 
The dataset is based on CLEVR which uses [Blender](https://www.blender.org/) for rendering images.

The code for training the models on various schemes is wrtitten in PyTorch.


## Generating Images for disjoint and strict cases

The ```IF-gt.json``` file holds the information for the ground truth rule for each task (we currently have 3 tasks denoted as: 0, 1, 2). 

The ```IF-conf.json``` file holds the information for the confounding rule for each task (we currently have 3 rules for rach tasks denoted as: 0, 1, 2).
The intended usage is specifying a single object confounder for each task. It is possible to support confounders involving multiple objects with minor code modifications.

The ConCon dataset generation script 
```./run_scripts/run_conf_3.sh```

The ```IMG_TYPE``` variable indicates the positive (1) and negative (0) samples.
The ```DATASET_TYPE``` variable indicates the name for the folder where the images will be saved to.
The ```TASK_IDS``` variable specifies the ids of tasks.
The ```conf_class_combos_json``` argument is optional and can be removed to generate unconfounded dataset. 

    cd data_generation/docker/
    docker build -t name_for_docker_cont -f Dockerfile .
    docker run -it --rm  --gpus device=00 -v /localpathto/training_scripts:/workspace/ --name name_for_docker_cont
    source ./run_scripts/run_conf_3.sh


We specify variables in the ```./run_scripts/run_conf_3.sh``` file to generate images.
The file contains self-explanatory variables that can be configured to manage dataset splits
The ```STRICT``` flag indicates the variant of the the dataset that is to be generated. Setting it to ```1``` generates images for ```strict``` variant and setting it to ```0``` genreates images for ```disjoint```.


## Running experiemental evaluations for different training setting

    cd Collect_Baseline/docker/
    docker build -t name_for_docker_cont -f Dockerfile .
    docker run -it --rm  --gpus device=00 -v /localpathto/training_scripts:/workspace/ --name name_for_docker_cont

To train the NN model on the ConCon dataset, run the following bash script:

    ./run_nn.sh {seed} {dataset_type(case_strict or case_disjoint)} {model_name} {path_to_root_directory}

To train the NeSy model on the ConCon dataset, run the following bash script:

    ./run_nesy.sh {seed} {dataset_type(case_strict or case_disjoint)} {model_name} {path_to_root_directory}


The bash scripts contains hyperparameter configurations, training schemes
 have options to train the models as on dataset discussed in the paper.

