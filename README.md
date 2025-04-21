# Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks

Official code for the paper [Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks](https://arxiv.org/abs/2410.02596). 

Rui Hu*, Yifan Zhang*, Zhuoran Li, Longbo Huang

This repo is modified from `gflownet-rl` (https://github.com/d-tiapkin/gflownet-rl)

## Installation

- Create conda environment:

```sh
conda create -n gflownet-rl python=3.10
conda activate gflownet-rl
```

- Install PyTorch with CUDA. For our experiments we used the following versions:

```sh
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can change `pytorch-cuda=11.8` with `pytorch-cuda=XX.X` to match your version of `CUDA`.

- Install core dependencies:

```sh
pip install -r requirements.txt
```

```sh
cd torchgfn
pip install -e .
cd ..
```

-*(Optional)* Install dependencies for molecule experiemtns
```sh
pip install -r requirements_mols.txt
```
You can change `requirements_mols.txt` to match your `CUDA` version by replacing `cu118` by `cuXXX`.

## Divergence-based Regression losses

Implemented in `torchgfn/src/gfn/loss.py`

List of available divergence-based losses:
- Linex($\alpha$) loss, in `class G_alpha`, where $\alpha=-1,0,0.5,1,2$ corresponds to reverse-$\chi^2$, reverse-KL, Hellinger's, forward-KL and forward $\chi^2$ divergece, respectively
- Shifted-cosh loss, in `class H_alpha`
- Total variation distance, in `class V_alpha`
- Jenson-Shannon divergence, in `class G_JSD`
- Symmetric-KL divergence, in `class G_SKL`

## Hypergrids

Code for this part heavily utlizes library `torchgfn` (https://github.com/GFNOrg/torchgfn).

Path to configurations (utlizes `ml-collections` library):

- General configuration: `hypergrid/experiments/config/general.py`
- Algorithm: `hypergrid/experiments/config/algo.py`
- Environment: `hypergrid/experiments/config/hypergrid.py`

List of available algorithms:
- Baselines: `fm`, `db`, `tb`, `subtb`

Example of running the experiment on environment with `height=20`, `ndim=4` with `hard6` rewards, seed `101` on the algorithm `fm` and Linex(1) loss:
```bash
python run_hypergrid_exp.py --general experiments/config/general.py:101 --env experiments/config/hypergrid.py:hard6 --algo experiments/config/algo.py:fm --env.height 20 --env.ndim 4 --algo.loss_type G --algo.alpha 1.00
```

## Molecules

The presented experiments actively reuse the existing codebase for molecule generation experiments with GFlowNets (https://github.com/GFNOrg/gflownet/tree/subtb/mols).

Additional requirements for molecule experiments: 
- `pandas rdkit torch_geometric h5py ray hydra` (installation is available in `requirements_mols.txt`)

List of available algorithms:
- Baselines: `fm`, `db`, `tb`, `subtb`

Example of running the experiment with seed `101` on the algorithm `fm` and Linex(1) loss:
```bash
python gflownet.py --objective fm --loss_type G --alpha 1.00 --seed 101
```

## Bit sequences

Code of this part also uses library `FL-GFN` (https://github.com/ling-pan/FL-GFN)

List of available algorithms:
- Baselines: `db`, `tb`, `subtb`

Example of running the experiment with seed `101` on the algorithm `db` and Linex(1) loss:

```bash
python bitseq/run.py --objective db --forward_looking --k 8 --loss_type G --alpha 1.00 --seed 101
```