# CPA - Compositional Perturbation Autoencoder


## What is CPA?
![Screenshot](Figure1.png)

`CPA` is a framework to learn effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug response across different cell types, doses and drug combinations. CPA allows:

* Out-of-distribution predicitons of unseen drug combinations at various doses and among different cell types.
* Learn interpretable drug and cell type latent spaces.
* Estimate dose response curve for each perturbation and their combinations.
* Access the uncertainty of the estimations of the model.

## Package Structure

The repository is centered around the `cpa` module:

* [`cpa.train`](cpa/train.py) contains scripts to train the model.
* [`cpa.api`](cpa/api.py) contains user friendly scripts to interact with the model via scanpy.
* [`cpa.plotting`](cpa/plotting.py) contains scripts to plotting functions.
* [`cpa.model`](cpa/model.py) contains modules of cpa model.
* [`cpa.data`](cpa/data.py) contains data loader, which transforms anndata structure to a class compatible with cpa model.

Additional files and folders:

* [`datasets`](datasets/) contains both versions of the data: raw and pre-processed.
* [`preprocessing`](preprocessing/) contains notebooks to reproduce the datasets pre-processing from raw data.

## Usage

- As a first step, download the contents of `datasets/` and `pretrained_models/` from [this tarball](https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar).


To learn how to use this repository, check 
[`./notebooks/demo.ipynb`](notebooks/demo.ipynb), and the following scripts:


* Note that hyperparameters in the `demo.ipynb` are set as default but might not work work for new datasets.
## Examples and Reproducibility
you can find more example and  hyperparamters tuning scripts and also reproducbility notebooks for the plots in the paper in the [`reproducibility`](https://github.com/theislab/cpa-reproducibility) repo.

## Curation of your own data to train CPA

* To prepare your data to train CPA, you need to add specific fields to adata object and perfrom data split. Examples on how to add 
necessary fields for multiple datasets used in the paper can be found in [`preprocessing/`](/https://github.com/facebookresearch/CPA/tree/master/preprocessing) folder.

## Training a model

There are two ways to train a cpa model:

* Using the command line, e.g.: `python -m cpa.train --data datasets/GSM_new.h5ad  --save_dir /tmp --max_epochs 1 --doser_type sigm`
* From jupyter notebook: example in [`./notebooks/demo.ipynb`](notebooks/demo.ipynb)


## Documentation

Currently you can access the documentation via `help` function in IPython. For example:

```python
from cpa.api import API

help(API)

from cpa.plotting import CPAVisuals

help(CPAVisuals)

```

A separate page with the documentation is coming soon.

## Support and contribute

If you have a question or noticed a problem, you can post an [`issue`](https://github.com/facebookresearch/CPA/issues/new).

## Reference

Please cite the following publication if you find CPA useful in your research.
```
@article{lotfollahi2023predicting,
  title={Predicting cellular responses to complex perturbations in high-throughput screens},
  author={Lotfollahi, Mohammad and Klimovskaia Susmelj, Anna and De Donno, Carlo and Hetzel, Leon and Ji, Yuge and Ibarra, Ignacio L and Srivatsan, Sanjay R and Naghipourfar, Mohsen and Daza, Riza M and Martin, Beth and others},
  journal={Molecular Systems Biology},
  pages={e11517},
  year={2023}
}
```

The paper titled **Predicting cellular responses to complex perturbations in high-throughput screens** can be found [here](https://www.biorxiv.org/content/10.1101/2021.04.14.439903v2](https://www.embopress.org/doi/full/10.15252/msb.202211517).
## License

This source code is released under the MIT license, included [here](LICENSE).


### Personnal modifications:

Modifications are on:

#### cpa/api.py

torch.save((self.model.state_dict(), self.args, self.model.history), filename)
Has been changed to: 
torch.save((self.model.state_dict(), self.args, self.model.history), filename, pickle_protocol=4)

Because the model size is too large and requires a different protocol than the one by default



#### cpa/plotting.py

for split in ["training", "test", "ood"]:

Was changed to:

for split in ["training", "test"]:

Since I did not want to include ood data in the training process. Its supposed to be for final validation.




#### cpa/train.py

The line: 

"ood": evaluate_r2(autoencoder, datasets["ood"], datasets["test"].subset_condition(control=True).genes)

Has been removed from training loop. I use ood data only for final testing after the model finished training 

I've also added pickle_protocol=4 anywhere torch.save() was being called to account for large model size.


 
 #### cpa/helper.py
 
 I noticed that rank_genes_groups tended to look for 50 differentially expressed genes. This hurt the performance of the model
 It was beter to set it to 1000 as a default and then look at the performance f the top 50 genes once training is done. 
 For some reason this worked well and improved performance on the top 50 DE genes.
 
 
 
