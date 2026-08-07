# XSSDense

**Reconstructing electron densities from X-ray solution scattering data using a Variational Autoencoder**

XSSDense is a framework for reconstructing three-dimensional electron density maps from X-ray solution scattering (XSS) data using deep generative models. The workflow consists of voxelisation of structural models, generation of training datasets, training of a variational autoencoder (VAE), extraction of latent-space statistics, and reconstruction of electron densities from scattering profiles.

---

## Workflow

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

---

See previous provided README content here (full version) ...

## Code Availability

The implementation of XSSDense, including dataset preparation, VAE training, latent-space analysis, and reconstruction workflows, is publicly available at:

https://github.com/LeoMonrroy266/XSSDense
