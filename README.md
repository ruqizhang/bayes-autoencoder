# Bayesian Auto-encoder

## Semi-sup on sentiment classification (MR dataset)
VAE
```
python main_v_semi.py --model vae
```
BVAE
```
python main_v_semi.py --model bvae
```
BAE-S
```
python main_bae_semi.py 
```
BAE-G
```
python main_bae_semi.py
```
Ensemble for test
```
python main_ens.py --model MODEL
```
* ```MODEL``` &mdash;vae, bvae, bae, baeg

`numlabel` controls the number of labelled data (2000 or 3000).
