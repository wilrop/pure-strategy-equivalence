# Pure Strategy Equivalence
This repository contains the code for the paper: "Bridging the Gap Between Single and Multi Objective Games". The paper describes a new equivalence notion, pure strategy equivalence, that guarantees that pure strategies in a continuous game correspond to mixed strategies in an equivalent MONFG with equal utility.

## Code
The code in utils was taken from [Ramo](https://github.com/wilrop/ramo) as it was not available on PyPI at the time. Additionally, the ```best_response.py```, ```IBR.py```, ```fictitious play.py``` and ```Player.py``` files were also taken from this framework.

The code for the Bertrand pricing game and Polynomial game are in their respective files and handle the full setup. The code which sets up the strategy bijections and identity games necessary to use pure strategy equivalence in our experiments are also in their respective files.

To reproduce the experiments, simply run the ``experiments.py`` file with the default parameters.

## Citation
To cite this paper, you may use the below BibTeX entry for now.

```bibtex
@misc{ropke2023bridging,
  doi = {10.48550/ARXIV.2301.05755},
  
  url = {https://arxiv.org/abs/2301.05755},
  
  author = {Röpke, Willem and Groenland, Carla and Rădulescu, Roxana and Nowé, Ann and Roijers, Diederik M.},
  
  keywords = {Computer Science and Game Theory (cs.GT), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Bridging the Gap Between Single and Multi Objective Games},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```