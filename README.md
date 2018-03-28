# Handwriting Generation
This repository contains a quick modified implementation of Alex Graves' paper: [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) using pytorch.

## How to reproduce the results
1. Run the data preprocessing script to generate required training data 
```
python data_preprocessing.py
```

2. Run training script specifying the task (for unconditional or conditional handwriting generation)
```
python train.py --task 'rand_write'
```

```
python train.py --task 'synthesis'
```

3. The trained models will be saved to the folder 'save'. Change the file name in generate.py and visualize the results on Results.ipynb.
