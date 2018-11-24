## Real-NVP in PyTorch

Implementation of Real-NVP in PyTorch. Based on the paper: [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
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
 
One epoch takes about 4 min, 15 sec when using the default arguments
and running on an NVIDIA Titan Xp card.

## Results

### Epochs 0, 10, 20

![Samples at Epochs 0, 10, 20](/samples/real_nvp_samples_0_10_20.png?raw=true "Samples at Epochs 0, 10, 20")

### Epochs 30, 40, 50

![Samples at Epochs 30, 40, 50](/samples/real_nvp_samples_30_40_50.png?raw=true "Samples at Epochs 30, 40, 50")
