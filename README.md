# CatalystGen: Diffusion-based Structure Generator for Adsorbate-Specific Catalyst Design

CatalystGen is a generative model designed to create bulk crystal structures and predict their surface orientations (Miller indices) tailored to specific adsorbates. It is inspired by the CDVAE framework and incorporates a conditional VAE with GemNet-based encoder and decoder to handle atomistic structures under periodic boundary conditions.

This repository provides the full pipeline to train, generate, visualize, and evaluate adsorbate-specific catalyst candidates.

---

## Motivation

Designing catalyst structures optimized for particular adsorbates remains a major challenge in heterogeneous catalysis. Existing models lack specificity for adsorbate binding or surface orientation. CatalystGen addresses this gap by:

- Generating bulk crystal structures from a latent distribution conditioned on adsorbate identity.
- Predicting suitable Miller indices to guide surface termination.
- Handling multiple atomic types and lattice geometries.

---

## Key Features

- Conditional structure generation with adsorbate-specific embedding.
- Surface orientation prediction via discrete Miller index classifier.
- PBC-aware coordinate and lattice decoder with Langevin dynamics sampling.
- Evaluation of chemical validity using structure and composition metrics.
- Easy visualization pipeline using CIF and PNG formats.

---

## Installation

```bash
git clone https://github.com/yjunho/CatalystGen.git
cd CatalystGen
conda create -n catalystgen python=3.10
conda activate catalystgen
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Generation
```bash
python generate.py --ads_idx 0 --atom_set 0 --label mygen
```
--ads_idx: index of adsorbate (0: CHCO, 1: CHOCHO, 2: NO2NO2, 3: COCH2O)

--atom_set: predefined atom type set (see code for options)

--label: custom label for output file

Output is saved to generated2/{adsorbate}/gen_{label}.pt.

## Visualization
```bash
python visualize.py --pt_path generated2/CHCO/gen_mygen.pt
```
Output files will be saved under:

generated2/CHCO/cif/

generated2/CHCO/png/


## Evaluation
```bash
python evaluate.py --pt_path generated2/CHCO/gen_mygen.pt
```
Saves results to: generated2/CHCO/validity_results.json


