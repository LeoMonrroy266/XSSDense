# XSSDense

**Reconstructing electron densities from X-ray solution scattering data using a Variational Autoencoder**

XSSDense is a framework for reconstructing three-dimensional electron density maps from X-ray solution scattering (XSS) data using deep generative models. The workflow consists of voxelisation of structural models, training of a variational autoencoder (VAE), and reconstruction of electron densities from scattering profiles.

---

## Quick Start

### 1. Voxelisation

Convert structural models into electron density voxel grids.

```bash
python <voxelisation_script.py> \
    --input <input_pdb_directory> \
    --output <output_voxel_directory>
```

### 2. Training

Train the variational autoencoder on the generated voxel grids.

```bash
python <training_script.py> \
    --config <training_configuration>
```

### 3. Reconstruction

Reconstruct an electron density map from XSS data.

```bash
python <reconstruction_script.py> \
    --input <scattering_curve.dat> \
    --model <trained_model>
```

---

## Method Overview

PDB Structures
    |
    v
Voxelisation
    |
    v
3D Density Maps
    |
    v
Variational Autoencoder
    |
    v
Latent Space Representation
    |
    v
Reconstruction from XSS Data
    |
    v
3D Electron Density Map

---

## Voxelisation

Before training the VAE, PDB structures must be converted into 3D voxel grids representing the electron density distribution. The voxelisation process transforms atomic coordinates into fixed-size three-dimensional matrices that can be used as input for the neural network.

The voxelisation is performed using atomic form factors, producing standardized volumetric representations suitable for training.

### Example

```bash
python <voxelisation_script.py> \
    --input example.pdb \
    --output example.npy
```

---

## Training and Validation

The VAE is trained on voxelised structures. The dataset is initially split into 90% training data and 10% test data. During model development, the training subset is further divided into 90% training and 10% validation data.

Model performance is monitored throughout training using:

- Training loss
- Validation loss
- Test loss
- Reconstruction error metrics
- Latent-space visualisation and analysis

These metrics are used to evaluate convergence and detect potential overfitting.

### Example

```bash
python <training_script.py> \
    --train_data <training_dataset> \
    --epochs <number_of_epochs>
```

---

## Reconstruction

Once the VAE has been trained, the reconstruction pipeline can be used to generate electron density maps from X-ray solution scattering (XSS) data.

The reconstruction algorithm optimizes the latent-space representation to identify a density map whose calculated scattering profile best matches the experimental XSS data. Users can specify reconstruction parameters such as optimization settings and input scattering curves.

The resulting latent representation is decoded by the trained VAE to produce a three-dimensional electron density map consistent with the provided XSS measurements.

### Example

```bash
python <reconstruction_script.py> \
    --curve example.dat \
    --model trained_vae.pt \
    --output reconstruction.mrc
```

---

## Repository Structure

```text
XSSDense/
│
├── <voxelisation_scripts>
├── <training_scripts>
├── <reconstruction_scripts>
├── <utility_scripts>
├── data/
├── models/
├── notebooks/
└── README.md
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

---

