# XSSDense

**Reconstructing electron densities from X-ray solution scattering data using a Variational Autoencoder**

XSSDense is a framework for reconstructing three-dimensional electron density maps from X-ray solution scattering (XSS) data using deep generative models. The workflow consists of voxelisation of structural models, generation of training datasets, training of a variational autoencoder (VAE), extraction of latent-space statistics, and reconstruction of electron densities from scattering profiles.

---

## Workflow

```text
PDB Structures
      │
      ▼
Voxelisation
      │
      ▼
3D Density Grids
      │
      ▼
TFRecord Generation
      │
      ▼
β-VAE Training
      │
      ▼
Latent Space Statistics
      │
      ▼
Genetic Algorithm Reconstruction
      │
      ▼
3D Electron Density Map
```

---

## Quick Start

### 1. Generate TFRecord datasets

Convert voxelized density maps stored as NumPy arrays into TensorFlow TFRecords.

```bash
python voxel_tf_fixed44.py \
    voxel_directory/ \
    tfrecord_output/
```

#### Arguments

| Argument | Description |
|-----------|-------------|
| `voxel_directory` | Directory containing voxelized density maps in `.npy` format. |
| `tfrecord_output` | Directory where TFRecords and metadata are saved. |

---

### 2. Train the VAE

Train a β-VAE on the generated TFRecord dataset.

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    experiment_name \
    late \
    1.0
```

#### Arguments

| Argument | Description |
|-----------|-------------|
| `train.tfrecord` | Training dataset. |
| `test.tfrecord` | Independent test dataset. |
| `experiment_name` | Name used for saved outputs. |
| `mode` | β scheduling mode. Use `constant` for fixed β or `late` for β warm-up scheduling. |
| `beta` | Constant β value or final β value reached during warm-up. |

#### Examples

Fixed β-VAE:

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    beta1 \
    constant \
    1.0
```

β warm-up training:

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    beta10 \
    late \
    10.0
```

Outputs:

```text
_log/
├── encoder_model.keras
├── decoder_model.keras
├── vae_final.weights.h5
└── log.txt
```

---

### 3. Generate latent-space statistics

After training, the latent-space distribution of the training set is analyzed. The mean and standard deviation of each latent dimension are extracted and saved for use during reconstruction.

These statistics define the region of latent space explored by the reconstruction algorithm and improve optimization efficiency.

```bash
python process_training_norms_absolute52.py \
    <model_directory> \
    <tf_record path> \
    <output_directory>
```

Outputs may include:

```text
latent_statistics/
├── latent_means.npy
├── latent_std.npy
├── latent_pca.png
```

These latent statistics are subsequently used by the genetic algorithm during reconstruction.

---

### 4. Reconstruct electron densities

Run reconstruction using the trained VAE and experimental XSS data.

```bash
python main_reconstruction_ga_may25_absolute.py \
    <arguments>
```

#### Key reconstruction parameters

The reconstruction script performs latent-space optimization using a genetic algorithm and requires:

| Parameter | Description |
|-----------|-------------|
| `trained decoder` | Decoder network from a trained VAE. |
| `trained encoder` | Encoder network from a trained VAE. |
| `experimental XSS data` | Experimental scattering curve of the ground state |
| `experimental difference data` | Experimental scattering curve used as reconstruction target. |
| `ground-state voxel` | Reference density map. |
| `latent statistics` | Latent-space normalization statistics obtained from `process_training_norms_absolute52.py`. |
| `output folder` | Directory where reconstruction outputs are written. |
| `target yield` | Excited-state population used during ΔI modelling. |
| `yield weight` | Weight applied to yield optimization. |
| `voxel size` | Voxel spacing in Å. |
| `rho_bulk` | Bulk solvent electron density. |
| `population size` | Number of latent-space candidates maintained by the genetic algorithm. |
| `batch size` | Number of candidates evaluated simultaneously. |
| `max iterations` | Maximum number of optimization rounds. |

#### Example

```bash
python main_reconstruction_ga_may25_absolute.py \
    --model_dir trained_model/ \
    --iq experimental.dat \
    --diff experimental.dat \
    --ground_state dark.npy \
    --latent_stats latent_stats.json \
    --output reconstruction/
```

The reconstruction algorithm uses:

- Experimental absolute and difference scattering data
- Trained encoder and decoder models
- Ground-state density map
- Latent means and standard deviations
---

## Method Overview

### Voxelisation

Before training the VAE, PDB structures must be converted into 3D voxel grids representing the electron density distribution. The voxelisation process transforms atomic coordinates into fixed-size three-dimensional matrices that can be used as input for the neural network.

The voxelisation is performed using atomic form factors, producing standardized volumetric representations suitable for training.

The resulting voxelized density maps are stored as NumPy arrays and subsequently converted into TensorFlow TFRecords using `voxel_tf_fixed44.py`.

---

### Training and Validation

The VAE is trained on voxelised structures represented as 52 × 52 × 52 electron-density grids.

The dataset is initially split into:

- 90% training data
- 10% independent test data

A validation subset is used during training to monitor model performance.

Model performance is monitored throughout training using:

- Training loss
- Validation loss
- Test loss
- Reconstruction errors
- Latent-space visualization
- Principal component analysis of latent representations

The implemented architecture is a β-VAE with configurable latent-space regularization. Two training modes are supported:

- `constant`: fixed β throughout training.
- `late`: β warm-up scheduling, where β is gradually increased during training.

Outputs include trained encoder and decoder networks together with the final VAE weights.

---

### Latent-Space Analysis

Following training, latent representations of the training set are analyzed using the trained encoder.

For each latent dimension, the following quantities are computed:

- Mean latent value
- Standard deviation
- Principal-component projections
- Latent-space visualizations

These statistics define the expected latent-space distribution and are used to initialize and constrain latent-space searches during reconstruction.

---

### Reconstruction

Once the VAE has been trained, the reconstruction pipeline can be used to generate electron-density maps from X-ray solution scattering (XSS) data.

The reconstruction algorithm employs a genetic algorithm operating directly in the latent space of the trained VAE.

Candidate latent vectors are:

1. Generated using latent-space statistics.
2. Decoded into electron-density maps.
3. Converted into theoretical scattering curves.
4. Compared against experimental XSS data.
5. Selected and evolved according to their agreement with experiment.

The optimization seeks the density map whose calculated scattering profile most closely reproduces the measured XSS signal.

The resulting optimized latent representation is decoded by the trained VAE to produce a three-dimensional electron density map consistent with the experimental scattering data.

---
