# Code for "Algorithmic Foundations for the Diffraction Limit"

These are scripts for the paper "Algorithmic Foundations for the Diffraction Limit" (<a href="https://arxiv.org/abs/2004.07659">preprint</a>)

## Contents
- `tv.py`: runs numerical integration to produce the data needed to generate Figure 3, dumps this data into `data/tvplot.json`
- `plots.ipynb`: loads data from `data/tvplot.json` and generates Figure 3
- `viz_samples`: draws samples from Airy superpositions to produce the data needed to generate Figure 2, dumps this data into `data/viz_samples_*.json` (NOTE: viz_samples_20000000.json not included in this repo because of upload size constraints)
- `sampl.ipynb`: loads data from `data/viz_samples_*.json` and generates Figure 2
