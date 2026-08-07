# XSSDense

**Reconstructing electron densities from X-ray solution scattering data using a Variational Autoencoder**

## Voxelisation

Before training the VAE, PDB structures must be converted into 3D voxel grids representing the electron density distribution. The voxelisation process transforms atomic coordinates into fixed-size three-dimensional matrices that can be used as input for the neural network.

The voxelisation is performed using atomic form factors producing standardized volumetric representations suitable for training.

## Training and Validation

The VAE is trained on voxelised structures. The dataset is initially split into 90% training data and 10% test data. During model development, the training subset is further divided into 90% training and 10% validation data.

Model performance is monitored throughout training using:

- Training, validation, and test loss curves
- Reconstruction error metrics
- Latent space visualisation and analysis

These metrics are used to evaluate convergence and detect potential overfitting.

## Reconstruction

Once the VAE has been trained, the reconstruction pipeline can be used to generate electron density maps from X-ray solution scattering (XSS) data.

The reconstruction algorithm optimizes the latent-space representation to identify a density map whose calculated scattering profile best matches the experimental XSS data. Users can specify reconstruction parameters such as optimization settings and input scattering curves.

The resulting latent representation is decoded by the trained VAE to produce a three-dimensional electron density map consistent with the provided XSS measurements.
