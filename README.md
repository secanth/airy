# airy
Supporting code for "Lies, Damned Lies, and Statistical Foundations for the Diffraction Limit"

- `tv.py`: runs numerical integration to produce the data needed to generate Figure 3, dumps this data into `data/tvplot.json`
- `plots.ipynb`: loads data from `data/tvplot.json` and generates Figure 3
- `viz_samples`: draws samples from Airy superpositions to produce the data needed to generate Figure 2, dumps this data into `data/viz_samples_*.json`
- `sampl.ipynb`: loads data from `data/viz_samples_*.json` and generates Figure 2
