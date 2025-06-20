# %% [markdown]
# # Multimodal dataset

# %%
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter1d

# %% [markdown]
# ## Notebook

# %% [markdown]
# ### Image sample

# %%
dset_ls = load_dataset("MultimodalUniverse/legacysurvey",
                       streaming=True,
                       split='train')
dset_ls = dset_ls.with_format("numpy")
dset_iterator = iter(dset_ls)

# %%
example = next(dset_iterator)

# %%
# Let's inspect what is contained in an example
example.keys()

# %%
plt.figure(figsize=(12,5))
for i,b in enumerate(example['image']['band']):
  plt.subplot(1,4,i+1)
  plt.title(f'{b}')
  plt.imshow(example['image']['flux'][i], cmap='gray_r')
  plt.axis('off')

# %% [markdown]
# ### Spectra samples

# %%
dset_sdss = load_dataset("MultimodalUniverse/sdss",
                       streaming=True,
                       split='train')
dset_sdss = dset_sdss.with_format("numpy")
dset_iterator = iter(dset_sdss)

# %%
example = next(dset_iterator)

# %%
# Let's inspect what is contained in an example
example.keys()

# %%
# For plotting, we remove the padding values that are recognizable by the -1
m = example['spectrum']['lambda'] > 0

plt.plot(example['spectrum']['lambda'][m],
     example['spectrum']['flux'][m])

# %% [markdown]
# ### Times-series

# %%
dset_plasticc = load_dataset("MultimodalUniverse/plasticc",
                       streaming=True,
                       split='train')
dset_plasticc = dset_plasticc.with_format("numpy")
dset_iterator = iter(dset_plasticc)

# %%
example = next(dset_iterator)
example.keys()

# %%
for b in np.unique(example['lightcurve']['band']):
  m = (example['lightcurve']['flux'] > 0) & (example['lightcurve']['band'] == b)
  plt.plot(example['lightcurve']['time'][m],
      example['lightcurve']['flux'][m],'+', label=b)
  plt.title(example['obj_type'])
plt.legend()

# %% [markdown]
# ### Cross-matching

# %%
from datasets import load_dataset_builder
from mmu.utils import cross_match_datasets

# %%
# Load the dataset descriptions from local copy of the data
sdss = load_dataset_builder("data/MultimodalUniverse/v1/sdss", trust_remote_code=True)
ls = load_dataset_builder("data/MultimodalUniverse/v1/legacysurvey", trust_remote_code=True)

# %%
# Use the cross matching utility to return a new HF dataset, the intersection
# of the parent samples.
dset = cross_match_datasets(sdss, # Left dataset
                            ls,  # Right dataset
                            # matching_radius=1.0, # Distance in arcsec
                            )

# %%
# The resulting dataset contains columns from both parent samples
dset

# %%
dset = dset.with_format("numpy")

# %%
# Extraire un exemple
example = dset[8]
print(example['spectrum'].keys())
print(example['image'].keys())
print(np.shape(example['image']['flux']))

# %%
plt.figure(figsize=[15, 3])

# === SDSS spectrum ===
plt.subplot(1, 5, 1)
plt.ylim(0, 20)

# Filtrage pour lambda > 0 (éviter les valeurs invalides)
m = example['spectrum']['lambda'] > 0

# Tracer le spectre brut + version lissée
plt.plot(example['spectrum']['lambda'][m], example['spectrum']['flux'][m], color='gray')
plt.plot(example['spectrum']['lambda'][m],
         gaussian_filter1d(example['spectrum']['flux'], sigma=5)[m],
         color='k')

plt.title("SDSS spectrum")

# === Legacy Survey images ===
for i in range(4):
    plt.subplot(1, 5, i + 2)

    img = example['image']['flux'][i]
    band = example['image']['band'][i]

    # Affichage en échelle log pour mieux voir les contrastes
    plt.imshow(np.log10(img + 2.0), cmap='gray')
    plt.title(f"Band: {band}")
    plt.axis('off')

plt.tight_layout()
plt.show()
