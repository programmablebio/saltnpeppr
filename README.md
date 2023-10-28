# SaLT&PepPr

### An Interface-Predicting Language Model for Designing Peptide-Guided Protein Degraders

![saltnpeppr_inference](https://user-images.githubusercontent.com/106272333/196185861-40837a34-2164-4a95-bdf0-30ce9b4b4b9f.png)


Targeted protein degradation of pathogenic proteins represents a powerful new treatment strategy for multiple disease indications. Unfortunately, a sizable portion of these proteins are considered “undruggable” by standard small molecule-based approaches, including PROTACs and molecular glues, largely due to their disordered nature, instability, and lack of binding site accessibility. As a more modular, genetically-encoded strategy, designing functional protein-based degraders to undruggable targets presents a unique opportunity for therapeutic intervention. In this work, we integrate pre-trained protein language models with recently-described joint encoder architectures to devise a unified, sequence-based framework to design target-selective peptide degraders without structural information. By leveraging known experimental binding protein sequences as scaffolds, we create a Structure-agnostic Language Transformer & Peptide Prioritization (SaLT&PepPr) module that efficiently selects peptides for downstream screening.

Please read and cite our [manuscript](https://www.nature.com/articles/s42003-023-05464-z) published in *Communications Biology*!

Model weights are available on [Drive](https://drive.google.com/u/1/uc?id=1JVfVTB2g1yOkpySYsb9nvDTHDI4InGaz&export=download)

All manuscript data is available on [Zenodo](https://zenodo.org/records/10008581)

We have developed a user-friendly [Colab notebook](https://colab.research.google.com/drive/1g-WBPi8_eWqUdD-BWHdPWLIQ8I9V3Log?usp=sharing) for peptide generation with SaLT&PepPr!

Authors: Garyk Brixi, Sophie Vincoff, and Pranam Chatterjee

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

This repository is strictly for non-commercial, academic usage. Any usage of this repository by a private, commercial entity is strictly prohibited. Please contact pranam.chatterjee@duke.edu for licensing information. 
