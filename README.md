# GNN for Particle Momentum Estimation in the CMS Trigger System

## Task 1 : Electron/photon classification

Use a deep learning method of your choice to achieve the highest possible
classification on this dataset (we ask that you do it both in Keras/Tensorflow and in
PyTorch). Please provide a Jupyter notebook that shows your solution. The model you
submit should have a ROC AUC score of at least 0.80.

Description: 32x32 matrices (two channels - hit energy and time) for two classes of
particles electrons and photons impinging on a calorimeter

Datasets : [photons](https://cernbox.cern.ch/files/public/show/AtBT8y4MiQYFcgc?sort-by=name&sort-dir=asc&items-per-page=100) and [electrons](https://cernbox.cern.ch/files/public/show/FbXw3V4XNyYB3oA?sort-by=name&sort-dir=asc&items-per-page=100)

## Task 2 : Graph Neural Networks

Choose 2 Graph-based architectures of your choice to classify jets as being
quarks or gluons. Provide a description on what considerations you have taken to
project this point-cloud dataset to a set of interconnected nodes and edges.

Discuss the resulting performance of the 2 chosen architectures.

Datasets : [Link](https://zenodo.org/record/3164691#.Yik7G99MHrB)

## requirements.txt

Install the required dependencies. The model in Task-2 is built using pytorch-geometric and energyflow which need to be installed. Specify your torch and cuda version in the speicfied brackets.

```pip
pip install energyflow

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-geometric
```

For example with ```torch==1.13.1``` and ```cuda==11.6```

```pip
pip install energyflow

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-geometric
```

## Description of the files

This repository contains one folder for each task and a requirements.txt file.

### Task 1 <br>
- task1/TASK.ipynb : Notebook containing all the code for task1
- task1/TASK1.ipynb - Colaboratory.pdf : pdf version of the notebook.
- task1/roc-curve.png : Plot of the roc curve of the model

### Task 2 <br>
- task2/TASK2.ipynb : Notebook containing all the code for task2
- task2/TASK2.ipynb - Colaboratory.pdf : pdf version of the notebook.
- task2/roc-curve.png : Plot of the roc curve of the model


## Model Architecture

### Task 1 

Used a modified ResNet-15 model and hyperparameters as explained in the paper [End-to-End Physics Event Classification with CMS Open Data](https://arxiv.org/abs/1807.11916). <br>

Trained both keras and pytorch models achieveing 0.8078 AUC-ROC score.

### Task 2

Trained two GNN architectures for quark-gluon classification. Input feature are a matrix of size (N, M, 4) where,
   - N is the number of jets <br>
   - M is the maximum particles in a jet <br>
   - 4 are the features of each particles denoting (pt, eta, phi, pid) values. <br> 

The graph for each jet is constructed in the following way: 

  - First for each jet we remove all the particles having all features values were zero i.e., remove paddings.
  - Next we calculate the pair-wise euclidean distance between the nodes using this metric,

$$ R = \sqrt{Δη^2 + Δϕ^2} $$

I have set a threshold for R above which we do not consider an edge between the nodes. The chosen value after experimenting is 0.05.

  - Next, edge-index are formed and the reverse edges are also concatenated. 
  - R values are set as the edge-weights and all the 4 feature are set as the node-features.

#### Architecture-1

- 3-layer GCN network with relu for aggregation of node-level features.
- Readput layer as global mean pooling for graph-level embedding.
- A dropout layer followed by linear layer.

#### Architecture-2

- Use of GraphConv layer in-place of GCN layer. It adds skip connections in the network tp preserve central node information and omits neighborhood normalization completely.
- The same readout layer with gloab_mean_pooling is used. I also tried using a combination of global_mean and global_max pool but it lead to decrease in performance.
- This is followed by an additional linear layer with relu. Then a dropout and final linear layer.

## Results

### Task 1

#### Keras-Implementation
- Testing Accuracy 0.737
- F1 score: 0.737
- ROC-AUC: 0.8078

#### PyTorch-Implementation
- Testing Accuracy 0.736
- F1 score score: 0.738
- ROC-AUC score: 0.8058

#### ROC Curve

![roc curve](task1/roc-curve.png)

### Task 2

#### Architecture-1
- Testing Accuracy 0.768
- F1 score: 0.762
- ROC-AUC score: 0.841

#### Architecture-2
- Testing Accuracy 0.792
- F1 score: 0.788
- ROC-AUC score: 0.872

#### ROC Curve

![roc curve](task2/roc-curve.png)







