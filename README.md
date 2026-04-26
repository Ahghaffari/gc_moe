# GC-MoE

**Graph-Conditioned Mixture of Graph Neural Network Experts for Traffic Forecasting**

GC-MoE routes each sensor node to the most suitable frozen pretrained
spatio-temporal GNN expert based on graph topology and current traffic
context. A dual-pathway router fuses static topology descriptors with a
dynamic pathway (temporal attention + spatial message passing) to produce
per-node soft mixture weights, training only a small router on top of
frozen expert backbones.

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for environment setup:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Pretrain experts on the target dataset
python main.py --model gwnet --dataset PEMS04 --device cuda:0 --seed 42
python main.py --model stgcn --dataset PEMS04 --device cuda:0 --seed 42
python main.py --model agcrn --dataset PEMS04 --device cuda:0 --seed 42

# 2. Train the GC-MoE router on top of the frozen experts
python main.py --model moe --dataset PEMS04 --device cuda:0 --seed 42 \
    --frozen_experts --router_type graph_conditioned \
    --load_balance_weight 0.001 --routing_entropy_weight 0.5
```

## Citation

```bibtex
@inproceedings{ghaffari2026gcmoe,
  title={Graph-Conditioned Mixture of Graph Neural Network Experts for Traffic Forecasting},
  author={Ghaffari, Amirhossein and Sheikhi, Saeid and Gilman, Ekaterina},
  booktitle={27th IEEE International Conference on Mobile Data Management (MDM)},
  year={2026},
  organization={IEEE}
}
```

## Acknowledgements

https://github.com/RWLinno/ST-LoRA that helped us with inspirations for ST-LoRA and the code structure design.
