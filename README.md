## Real NVP in PyTorch

Implementation of Real NVP in PyTorch. Based on the paper:

  > [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)\
  > Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio\
  > _arXiv:1605.08803_

Training script and hyperparameters designed to match the
CIFAR-10 experiments described in Section 4.1 of the paper.


## Usage

### Environment Setup
  1. Make sure you have [Anaconda or Miniconda](https://conda.io/docs/download.html)
  installed.
  2. Clone repo with `git clone https://github.com/chrischute/real-nvp.git rnvp`.
  3. Go into the cloned repo: `cd rnvp`.
  4. Create the environment: `conda env create -f environment.yml`.
  5. Activate the environment: `source activate rnvp`.

### Train
  1. Make sure you've created and activated the conda environment as described above.
  2. Run `python train.py -h` to see options.
  3. Run `python train.py [FLAGS]` to train. *E.g.,* run
  `python train.py` for the default configuration, or run
  `python train.py --gpu_ids=[0,1] --batch_size=128` to run on
  2 GPUs instead of the default of 1 GPU. 
  4. At the end of each epoch, samples from the model will be saved to
  `samples/epoch_N.png`, where `N` is the epoch number.
 
One epoch takes about 4 minutes when using the default arguments
and running on an NVIDIA Titan Xp card.

## Samples

### Epoch 5

![Samples at Epoch 5](/samples/epoch_5.png?raw=true "Samples at Epoch 5")


### Epoch 10

![Samples at Epoch 10](/samples/epoch_10.png?raw=true "Samples at Epoch 10")


### Epoch 15

![Samples at Epoch 15](/samples/epoch_15.png?raw=true "Samples at Epoch 15")


### Epoch 20

![Samples at Epoch 20](/samples/epoch_20.png?raw=true "Samples at Epoch 20")


### Epoch 25

![Samples at Epoch 25](/samples/epoch_25.png?raw=true "Samples at Epoch 25")



## Results

### Bits per Dimension

| Epoch | Train | Valid |
|-------|-------|-------|
| 5     | 3.97  | 3.98  |
| 10    | 3.76  | 3.76  |
| 15    | 3.69  | 3.74  |
| 20    | 3.65  | 3.70  |
| 25    | 3.62  | 3.74  |
