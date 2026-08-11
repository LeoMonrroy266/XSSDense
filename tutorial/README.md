# XSSDense Step-by-Step Tutorial

## Overview

This tutorial walks through the complete XSSDense workflow, from a collection of PDB structures to a reconstructed three-dimensional electron density map derived from X-ray solution scattering (XSS) data.

By the end of this tutorial you will have:

1. Generated voxelized electron-density maps.
2. Converted voxel maps into TensorFlow TFRecords.
3. Trained a β-VAE model.
4. Generated latent-space statistics.
5. Reconstructed an electron density map from experimental scattering data.

---

# Workflow

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

# Prerequisites

Before starting, ensure the required dependencies are installed.

Required packages:

```bash
pip install tensorflow numpy scipy matplotlib scikit-learn cupy gemmi tqdm
```

Additional dependencies may be required depending on your installation and cuda version.

---

# Reproducing the Included Example

## Overview

Before applying XSSDense to your own structures and experimental data, we recommend reproducing the included example dataset.

This serves two purposes:

1. Verifies that your installation is working correctly.
2. Demonstrates the complete XSSDense workflow from training through reconstruction.

The repository contains an example dataset and reference outputs that can be used to validate your results.

```text
example/
├── data
      ├── meta
      └── tfrecords
├── voxel_maps.h5
├── ground_state.dat
├── difference_signal.dat
└── reference_results/
```

At the end of this tutorial your outputs should closely resemble the contents of:

```text
example/reference_results/
```

---

## Example Workflow

### Generate TFRecords

Convert the provided voxel dataset into TFRecords:

```bash
python generate_tfrecord.py \
    example/voxel_maps.h5 \
    example_output/
```

Expected output:

```text
example_output/
├── tfrecords
│   ├── train.tfrecord
│   └── test.tfrecord
└── meta
    └── meta.json
```

---

### Train the β-VAE

Train the VAE using the same settings used for the example reconstruction.

```bash
python Train_VAE.py \
    example_output/tfrecords/train.tfrecord \
    example_output/tfrecords/test.tfrecord \
    lov2_model \
    late \
    1
```

Expected output:

```text
lov2_model/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_1.weights.h5
.
.
.
├── vae_final.weights.h5
└── log.txt
```

---

### Generate Latent-Space Statistics

Compute latent-space statistics from the trained model.

```bash
python process_training.py \
    lov2_model/ \
    example_output/tfrecords/train.tfrecord \
    lov2_model/
```

Expected output:

```text
lov2_model/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_1.weights.h5
.
.
.
├── vae_final.weights.h5
├── log.txt
├── latent_pca.png
└── latents_stats.txt
```

---

### Run Reconstruction

Perform latent-space optimization using the included scattering data.

```bash
python reconstruct.py \
    --model_path lov2_model/ \
    --ground_scattering example/ground_state.dat \
    -- output_folder lov2_reconstruction
    -- params example_output/meta/meta.json \
    -- meta_json 2.4 \
    -- batch_size 16 \
    -- max_iter 30 \
    -- target_yield 0.15 \
    -- yield_weight 2 \
    -- latent_size 8 \
    --iq_path example/difference_signal.dat \
    --voxel example/voxel_maps.h5 \
```

Expected output:

```text
lov2_reconstruction/
├── absolute_drho.npy
├── absolute_scale.npy
├── absolute_shell.npy
├── final_rank0_alpha.npy
├── final_rank0_alpha_score.npy
├── final_rank0_deltaI.png
├── final_rank0_diff.ccp4
├── final_rank0_diff_norm.npy
├── final_rank0_drho.npy
├── final_rank0_extrapolated.ccp4
├── final_rank0_extrapolated.npy
├── final_rank0_drho.npy
├── final_rank0_latent.npy
├── final_rank0_score.npy
├── final_rank0_shell.npy
├── final_rank0_drho.npy
.
.
.
├── iteration_0.dat
├── iteration_1.dat
├── iteration_2.dat
.
.
.
├── iteration_29.dat
└── log.txt

```

---

## Comparing with the Reference Results

After reconstruction, compare your outputs against:

```text
example/reference_results/
```

The following should be similar:

- Reconstructed density maps
- Reconstructed scattering curves
- Latent-space solutions
- Optimization convergence behaviour

Small differences between runs are expected because the reconstruction uses stochastic initialization and genetic-algorithm optimization.

---

## Validation Checklist

Before proceeding with your own data, verify that:

- [ ] TFRecords were generated successfully.
- [ ] The VAE completed training.
- [ ] Encoder and decoder models were saved.
- [ ] Latent statistics were generated.
- [ ] Reconstruction completed successfully.
- [ ] Results are similar to the provided reference outputs.

Once the example workflow runs successfully, continue with the remaining sections of this tutorial and substitute your own structures and scattering data.


# Step 1: Voxelise Structural Models

## Purpose

The VAE cannot use atomic coordinates directly.

Each PDB structure must first be converted into a fixed-size electron-density grid.

Input:

```text
pdb_structures/
├── structure1.pdb
├── structure2.pdb
├── structure3.pdb
└── ...
```

Output:

```text
voxel_maps.h5
```

The h5 file contains multiple 52 × 52 × 52 electron-density volumes.

---


# Step 2: Generate TFRecords

## Purpose

Training is performed using TensorFlow TFRecord datasets.

Convert all voxelized density maps to TFRecords using:

```bash
python generate_tfrecord.py \
    voxel_maps.h5 \
    save_dir/
```

---

## Arguments

### Input Directory

H5 file containing voxelized density maps:

```text
voxel_maps.h5
```

### Output Directory

Directory where TFRecords and metadata will be written:

```text
save_dir/
```

---

## Expected Output

```text
save_dir/
├── tfrecords
      ├──train.tfrecords
      └──test.tfrecords
└──meta
      └── meta.json
```

---

## Verify Output

```bash
ls save_dir/tf_records
```

Ensure the training and test TFRecords were created successfully before continuing.

---

# Step 3: Train the β-VAE

## Purpose

The Variational Autoencoder learns a compressed latent-space representation of electron-density maps.

During training:

```text
Density Map
      │
      ▼
Encoder
      │
      ▼
Latent Vector
      │
      ▼
Decoder
      │
      ▼
Reconstructed Density Map
```

---

## Training Arguments

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    save_dir \
    mode \
    beta
```

Where:

- `train.tfrecord` = training dataset
- `test.tfrecord` = independent test dataset
- `save_dir` = output directory 
- `mode` = `constant` or `late`
- `beta` = β value

---
## Training Modes

Two β scheduling options are available.

### Constant β

Uses a fixed β value throughout training.

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    save_dir \
    constant \
    0.5
```

### β Warm-Up

Gradually increases β during training.

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    save_dir \
    late \
    1
```

This is typically recommended for improved latent-space organization.

## Replicating LOV2 results
To reproduce the files in the example folder we will run the following to replicate the result of the LOV2 case:

```bash
python Train_VAE.py \
    train.tfrecord \
    test.tfrecord \
    save_dir \
    late \
    1
```

---

## Monitoring Training

During training monitor:

- Training loss
- Validation loss
- Test loss
- Reconstruction loss
- KL divergence loss

A healthy training process typically shows decreasing reconstruction and validation losses.

---

## Training Outputs

Upon completion:

```text
save_dir/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_1.weights.h5
.
.
.
├── vae_final.weights.h5
└── log.txt
```

Verify these files exist before proceeding.

---

# Step 4: Generate Latent-Space Statistics

## Purpose

The reconstruction algorithm searches the VAE latent space rather than directly searching density maps.

Latent-space statistics provide:

- Mean latent values
- Standard deviations
- PCA projections

These statistics guide the genetic algorithm toward realistic regions of latent space.

---

## Run Analysis

```bash
python process_training_norms_absolute52.py \
    path_to_trained_model/ \
    train.tfrecord \
    latent_statistics/
```

---

## Inputs

```text
path_to_trained_model/
├── encoder_model.keras
├── decoder_model.keras
├── vae_epoch_1.weights.h5
.
.
.
├── vae_final.weights.h5
└── log.txt
```

```text
train.tfrecord
```

---

## Outputs

```text
path_to_trained_model/
├── latent_means.npy
├── latent_std.npy
├── latent_pca.png
```
---

## Verify Statistics

```python
import numpy as np

means = np.load("latent_means.npy")
stds = np.load("latent_std.npy")

print(means.shape)
print(stds.shape)
```

The dimensions should match the VAE latent dimension of 8.

---

# Step 5: Prepare Reconstruction Inputs

Before starting reconstruction, gather all required files.

## Required Inputs

### Trained VAE

```text
trained_model/
├── encoder_model.keras
├── decoder_model.keras
└── vae_final.weights.h5
```

### Latent Statistics

```text
latent_statistics/
├── latent_means.npy
└── latent_std.npy
```

### Experimental Scattering Data

Ground-state scattering:

```text
ground_state.dat
```

Difference scattering:

```text
difference_signal.dat
```

### Ground-State Density

```text
dark.npy
```

---

# Step 6: Run Reconstruction

## Purpose

The reconstruction algorithm searches latent space using a genetic algorithm.

For each candidate:

```text
Latent Vector
      │
      ▼
Decoder
      │
      ▼
Density Map
      │
      ▼
Scattering Calculation
      │
      ▼
Comparison with Experiment
      │
      ▼
Fitness Score
      │
      ▼
Evolution
```

---

## Example Reconstruction Command

```bash
python main_reconstruction_ga_may25_absolute.py \
    --model_dir trained_model/ \
    --iq ground_state.dat \
    --diff difference_signal.dat \
    --ground_state dark.npy \
    --latent_stats latent_statistics/ \
    --output reconstruction/
```

---

# Reconstruction Parameters

Important parameters include:

### Target Yield

Excited-state population used during ΔI modelling.

Example:

```text
0.10
```

corresponds to a 10% excited-state fraction.

### Yield Weight

Controls optimization pressure on yield agreement.

### Population Size

Number of candidate latent vectors.

Increasing population size:

- improves search coverage
- increases runtime

### Batch Size

Number of candidates evaluated simultaneously.

### Maximum Iterations

Maximum number of GA generations.

Higher values typically improve convergence but require additional computational time.

---

# Step 7: Analyze Results

After optimization completes, a reconstruction directory should contain outputs similar to:

```text
reconstruction/
├── best_density.npy
├── best_latent.npy
├── reconstructed_curve.dat
├── fitness_history.csv
└── optimization_log.txt
```

Exact filenames may vary between versions.

---

# Evaluate Reconstruction Quality

## Check Convergence

Inspect:

```text
fitness_history.csv
```

A successful optimization often shows:

- progressively improving fitness
- convergence toward a stable solution
- reduced fitness fluctuations

---

## Compare Scattering Profiles

Plot:

```text
Experimental ΔI(q)
```

against:

```text
Reconstructed ΔI(q)
```

Good overlap indicates successful reconstruction.

---

## Visualize Density Maps

Reconstructed densities can be visualized using:

- ChimeraX
- UCSF Chimera
- VMD
- PyMOL

Load the density map and inspect structural features that explain the observed scattering signal.

---

# Troubleshooting

## TFRecord Generation Fails

Check:

- voxel files exist
- file permissions are correct
- output directory exists

---

## VAE Training Fails

Check:

- TensorFlow installation
- available GPU memory
- TFRecord compatibility

---

## No Reconstruction Convergence

Possible causes include:

- insufficient training data
- poor latent-space organization
- incorrect experimental scaling
- population size too small
- too few optimization iterations

---

# Complete Example Workflow

```bash
# Step 1
voxelise_pdbs.py pdb_structures/ voxel_maps/

# Step 2
python voxel_tf_fixed44.py \
    voxel_maps/ \
    tfrecords/

# Step 3
python Train_VAE.py \
    tfrecords/train.tfrecord \
    tfrecords/test.tfrecord \
    beta10 \
    late \
    10.0

# Step 4
python process_training_norms_absolute52.py \
    trained_model/ \
    tfrecords/train.tfrecord \
    latent_statistics/

# Step 5
python main_reconstruction_ga_may25_absolute.py \
    --model_dir trained_model/ \
    --iq ground_state.dat \
    --diff difference_signal.dat \
    --ground_state dark.npy \
    --latent_stats latent_statistics/ \
    --output reconstruction/
```

---

# Next Steps

After running the complete pipeline you should have:

- A trained β-VAE model
- Latent-space statistics describing the training distribution
- An optimized latent representation
- A reconstructed three-dimensional electron density map consistent with the experimental XSS data

These reconstructed density maps can then be interpreted structurally and compared with candidate molecular models.
