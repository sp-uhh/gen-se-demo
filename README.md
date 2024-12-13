# Diffusion-based Speech Enhancement: Demonstration of Performance and Generalization

<img src="https://raw.githubusercontent.com/sp-uhh/gen-se-demo/main/assets/figure_1.png" width="700" alt="Comparison between SGMSE and Schrödinger bridge.">

This repository contains the interactive demo for the paper [*"Diffusion-based Speech Enhancement: Demonstration of Performance and Generalization"*](https://openreview.net/forum?id=rv5LuElUic) referenced in [1]. The demo contains works from [2], [3], and [4]. 

Please cite these references if you use the code: [[bibtex]](#references)

## Abstract

This demo presents advanced techniques in speech enhancement using deep generative models. It highlights the generalization capabilities of score-based generative models for speech enhancement and compares directly with Schrödinger bridge approaches. The presented methods focus on generating high-quality super-wideband speech at a sampling rate of 48 kHz. Participants will record speech using a single microphone in a noisy environment, such as a conference venue. These recordings will then be enhanced and played back through headphones, demonstrating the model's effectiveness in improving speech quality and intelligibility.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the demo, open the Jupyter notebook `demo.ipynb` and follow the instructions.

## References

```bib
@article{richter2023speech,
  title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351-2364},
  year={2023},
  doi={10.1109/TASLP.2023.3285241}
}
```
```bib
@article{richter2024diffusion,
  title={Diffusion-based Speech Enhancement: Demonstration of Performance and Generalization},
  author={Richter, Julius and Gerkmann, Timo},
  journal={Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation},
  year={2024}
}
```
```bib
@inproceedings{jukic2024schrodinger,
  title={Schr{\"o}dinger Bridge for Generative Speech Enhancement},
  author={Juki{\'c}, Ante and Korostik, Roman and Balam, Jagadeesh and Ginsburg, Boris},
  booktitle={Proceedings of Interspeech},
  pages={1175--1179},
  year={2024}
}
```
```bib
@article{richter2024investigating,
  title={Investigating Training Objectives for Generative Speech Enhancement},
  author={Richter, Julius and de Oliveira, Danilo and Gerkmann, Timo},
  journal={arXiv preprint arXiv:2409.10753},
  year={2024}
}
```

>[1] J. Richter, T. Gerkmann, "Diffusion-based Speech Enhancement: Demonstration of Performance and Generalization", Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation, 2024.
>
>[2] J. Richter, T. Gerkmann, "Speech Enhancement and Dereverberation with Diffusion-based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
>
>[3] A. Jukić, R. Korostik, J. Balam, B. Ginsburg, "Schrödinger Bridge for Generative Speech Enhancement", Proceedings of Interspeech, 2024.
>
>[4] J. Richter, D. de Oliveira, T. Gerkmann, "Investigating Training Objectives for Generative Speech Enhancement", arXiv preprint arXiv:2409.10753, 2024.