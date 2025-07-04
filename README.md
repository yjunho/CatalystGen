# CatalystGen: Diffusion-based Structure Generator for Adsorbate-Specific Catalyst Design

**CatalystGen** is a diffusion-based generative model for designing catalyst crystal structures tailored to specific adsorbates. It generates bulk crystal structures and predicts suitable surface orientations (Miller indices) through a conditional generation process.

## Key Features

- Diffusion-based generation of periodic crystal structures  
- Adsorbate-conditioned generation pipeline  
- Miller index prediction for surface orientation  
- End-to-end support for training, generation, visualization, and evaluation

## Model Architecture

<div align="center">
  <img src="assets/model architecture.PNG" alt="Model Architecture" width="400"/>
</div>

## Sample Results

<div align="center">
  <img src="assets/result.PNG" alt="Sample Results" width="400"/>
</div>


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
python generate.py --ads_idx 0 --atom_set 0 --label mygen --ckpt_path weights/all_ads/epoch=16-step=2839.ckpt --num_samples 100 --batch_size 100
```
--ads_idx: index of adsorbate (0: CHCO, 1: CHOCHO, 2: NO2NO2, 3: COCH2O)

--atom_set: predefined atom type set (see code for options)

--label: custom label for output file

--ckpt_path: manually specify which .ckpt checkpoint to use

--num_samples: number of structures to generate (default: 300)

--batch_size: number of samples per diffusion batch (default: 100)

Output is saved to generated2/{adsorbate}/gen_{label}.pt.

## Visualization
```bash
python visualize.py --pt_path generated/CHCO/gen_mygen.pt
```
Output files will be saved under:

generated/CHCO/cif/

generated/CHCO/png/


## Evaluation
Before running evaluation, make sure to generate `.cif` files using `visualize.py`.

```bash
python evaluate.py --pt_path generated/CHCO/gen_mygen.pt
```
Saves results to: generated/CHCO/validity_results.json


