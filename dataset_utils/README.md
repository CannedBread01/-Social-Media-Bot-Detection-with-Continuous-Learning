# Dataset Utils

provides useful utilities for handling and interacting with datasets

## Datasets 
Multiple different datasets are implemented, but only a subset is actually relevant for the project.
The remaining dataset are not used, as they lack the data the required tweet text information from users.
The following list provides an overview of the relevant datasets.

- Caverlee 11
- Cresci 17
- Cresci 18
- Twibot 20
- Twibot 22
### Datasets: General 
In general all datasets have different splits that can be accessed with `train`, `dev` or `test`. 
Additionally, the label of the samples can be easily be remapped by proving a list for the wanted labels.
Finally, it can be specified to only use a subset of the dataset, there  by reducing its size. 

### Twibot 22
Twibot 22 has two implementations, due to its size.
The raw dataset files are split into multiple JSON files that are each 10GB big and combine into a total dataset size of 125GB.
The 'normal' Twibot22 dataset class just parses and iterates over the JSON files, resulting in a slow and resource intensive process. 
As a preferred alternative the 'Twibot22Improved' class should be used. 
It functions by creating a new reduced dataset from the original dataset and keeps only the necessary information, shrinking the file size to 25GB and speeding up iteration.
To note, the reduced dataset needs to be instantiated once, by calling the function `prepare_dataset()` before the dataset can be used.

## Interleaved Iterable Dataset 
The Purpose of the Interleaved Iterable Dataset is combine multiple iterable dataset into one.
It shuffles the provided datasets either using `RoundRobin` or `Random`.

## Processed Dataset
The Processed Dataset class provides the option to load tensors from a file as a dataset.
It is primarily used to speed up the experiments. 
As the feature vector creation pipeline is unchanged between different classification heads, 
the results of the pipeline are stored to file and are repeatedly accessed with the Processed Dataset object.
This allows us to run the computationally heavy pipeline only once instead of multiple times. 

## Utilities

### Splitting 
The splitting function is used to separate a dataset that does not have a predefined training, development and testing split into these subsets.

### Constants 
The file provides many different functions and utilities that are repeatedly used in the datasets.