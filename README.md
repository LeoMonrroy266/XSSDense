# XSSDense

**Reconstructing electron densities from X-ray solution scattering data using a Variational Autoencoder**

XSSDense is a framework for reconstructing three-dimensional electron density maps from X-ray solution scattering (XSS) data using deep generative models. The workflow consists of voxelisation of structural models, generation of TFRecord datasets, training of a variational autoencoder (VAE), and reconstruction of electron densities from scattering profiles.

---

## Quick Start

### 1. Generate TFRecord datasets

Convert voxelized density maps stored as NumPy arrays into TensorFlow TFRecords.

```bash
python voxel_tf_fixed44.py \
    <npy_directory> \
    <reference.npy> \
    <output_directory>
```

### 2. Train the VAE

Train the β-VAE using the generated TFRecords.

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    experiment_name \
    train \
    1.0
```

### 3. Reconstruct electron densities

Run reconstruction using the trained VAE and experimental XSS data.

```bash
python main_reconstruction_ga_may25_absolute.py \
    <arguments>
```

---

## Method Overview

```text
PDB Structures
      │
      ▼
Voxelisation
      │
      ▼
NumPy Density Grids
      │
      ▼
TFRecord Generation
      │
      ▼
Variational Autoencoder
      │
      ▼
Latent Space Representation
      │
      ▼
Genetic-Algorithm Reconstruction
      │
      ▼
3D Electron Density Map
```

---

## Voxelisation

Before training the VAE, PDB structures must be converted into 3D voxel grids representing the electron density distribution. The voxelisation process transforms atomic coordinates into fixed-size three-dimensional matrices that can be used as input for the neural network.

The voxelisation is performed using atomic form factors, producing standardized volumetric representations suitable for training.

The resulting NumPy density maps are converted into TensorFlow TFRecords using:

```bash
python voxel_tf_fixed44.py \
    <npy_directory> \
    <reference.npy> \
    <output_directory>
```

---

## Training and Validation

The VAE is trained on voxelised structures represented as 52×52×52 density grids.

The dataset is initially split into:

- 90% training data
- 10% test data

A validation subset is used during training to monitor model performance.

Model performance is monitored throughout training using:

- Training loss
- Validation loss
- Test loss
- Reconstruction error metrics
- Latent-space visualisation and analysis

The implemented model is a β-VAE with scheduled regularization that balances reconstruction accuracy and latent-space organization.

Example training command:

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    experiment_name \
    train \
    1.0
```

Outputs include:

- Trained encoder model
- Trained decoder model
- Trained VAE weights
- Training logs

---

## Reconstruction

Once the VAE has been trained, the reconstruction pipeline can be used to generate electron-density maps from X-ray solution scattering (XSS) data.

The reconstruction algorithm employs a genetic algorithm to optimize the latent-space representation of the VAE. Candidate latent vectors are decoded into electron-density maps, and corresponding scattering profiles are calculated and compared against experimental XSS measurements.

The optimization aims to identify the density map whose calculated scattering profile best reproduces the experimental data.

The resulting latent representation is decoded by the trained VAE to produce a three-dimensional electron density map consistent with the provided XSS measurements.

Example:

```bash
python main_reconstruction_ga_may25_absolute.py \
    <reconstruction_parameters>
```

---

## Repository Structure

```text
XSSDense/
│
├── voxel_tf_fixed44.py
├── Train_VAE.py
├── main_reconstruction_ga_may25_absolute.py
│
├── data/
├── models/
├── results/
└── README.md
```

---

## Requirements

Install dependencies
