# Handwriting Generation
This repository contains a quick modified implementation of Alex Graves' paper: [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) using pytorch.

## How to reproduce the results
1. Run the data preprocessing script to generate required training data 
```
python data_preprocessing.py
```

2. Run training script specifying the task (rand_write or synthesis)
```
python train.py --task 'rand_write'
```

```
python train.py --task 'synthesis'
```
