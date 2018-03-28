# Handwriting Generation
This repository contains a quick modified implementation of Alex Graves' paper: [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) using pytorch.


### The results I obtained
1. Default Architecture: 2-layer LSTM network with skip connections. A window layer is implemented for the handwriting synthesis task (conditional handwriting generation) as adapted from Alex Graves' work.

2. The results shown here are obtained by training using the default configurations below for 50 epochs and 60 epchs respectively for each task. The training is surprisingly stable even with 800 timesteps. Learning rate annealing is implemented but not used by default.

```
    parser.add_argument('--task', type=str, default='rand_write',
                        help='"rand_write" or "synthesis"')
    parser.add_argument('--cell_size', type=int, default=400,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--timesteps', type=int, default=800,
                        help='LSTM sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=8E-4,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='lr decay rate for adam optimizer per epoch')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of attention clusters on text input')
```

3. Some examples of the results I obtained:

#### Unconditional generation
![Alt_text](examples/unconditional_generation.png?raw=true "Unconditional Generation")


#### Conditional generation
![Alt text](examples/conditional_generation.png?raw=true "Conditional Generation")


### How to reproduce the results
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


