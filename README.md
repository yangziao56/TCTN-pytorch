![TCTN: from observed frames to future frames.](assets/readme-banner.svg)

<p align="center"><strong>A 3D-Temporal Convolutional Transformer Network<br>for Spatiotemporal Predictive Learning</strong></p>
<p align="center">Ziao Yang · Xiangrui Yang · Qifeng Lin</p>
<p align="center"><a href="https://arxiv.org/pdf/2112.01085v1"><strong>Read PDF</strong></a> &nbsp; · &nbsp; <a href="https://arxiv.org/abs/2112.01085v1">Paper</a> &nbsp; · &nbsp; <a href="https://drive.google.com/drive/folders/1goWCEQoSWlcdmhV-UQffnNQLMa1FabeK?usp=sharing">Moving MNIST</a></p>

## Inside TCTN

A PyTorch implementation combining **3D temporal convolution** with **Transformer-based modeling** for spatiotemporal prediction.

```text
Data loaders → TCTN modules → Training & evaluation → Predicted frames
```

| Network | Training | Data & metrics |
| :--- | :--- | :--- |
| [`models/TCTN.py`](models/TCTN.py) | [`run.py`](run.py) | [`data_provider/`](data_provider/) |
| [`TCTN_modules/`](TCTN_modules/) | [`trainer.py`](trainer.py) | [`utils/`](utils/) |

## Set up a sequence

```bash
git clone https://github.com/yangziao56/TCTN-pytorch.git
cd TCTN-pytorch
```

Download Moving MNIST from the link above, then set the training and validation paths in `run.py`. The entry point exposes sequence length, image size, model depth, and training settings.

Questions about another dataset? [Contact Ziao Yang](mailto:ziaoyang@brandeis.edu).

<details>
<summary><strong>Citation</strong></summary>

```bibtex
@misc{yang2021tctn,
  title = {TCTN: A 3D-Temporal Convolutional Transformer Network for Spatiotemporal Predictive Learning},
  author = {Yang, Ziao and Yang, Xiangrui and Lin, Qifeng},
  year = {2021},
  eprint = {2112.01085},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2112.01085v1}
}
```

</details>
