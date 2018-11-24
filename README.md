## Real NVP in PyTorch

Implementation of Real NVP in PyTorch. Based on the paper: [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.

Model architecture and hyperparameters chosen to match the CIFAR-10
experiments described in Section 4.1 of the paper.

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

## Results

### Epoch 0

![Samples at Epoch 0](/samples/epoch_0.png?raw=true "Samples at Epoch 0")
*Train bits per dimension:* 5.35
*Valid bits per dimension:* 4.54

### Epoch 10

![Samples at Epoch 10](/samples/epoch_10.png?raw=true "Samples at Epoch 10")
*Train bits per dimension:* 3.81
*Valid bits per dimension:* 3.78

### Epoch 20

![Samples at Epoch 20](/samples/epoch_20.png?raw=true "Samples at Epoch 20")
*Train bits per dimension:* 3.69
*Valid bits per dimension:* 3.78

### Epoch 30

![Samples at Epoch 30](/samples/epoch_30.png?raw=true "Samples at Epoch 30")
*Train bits per dimension:* 3.63
*Valid bits per dimension:* 3.72

### Epoch 40

![Samples at Epoch 40](/samples/epoch_40.png?raw=true "Samples at Epoch 40")
*Train bits per dimension:* 3.60
*Valid bits per dimension:* 3.77

### Epoch 50

![Samples at Epoch 50](/samples/epoch_50.png?raw=true "Samples at Epoch 50")
*Train bits per dimension:* 3.61
*Valid bits per dimension:* 3.70
