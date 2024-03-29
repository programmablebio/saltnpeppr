# SaLT&PepPr

### An Interface-Predicting Language Model for Designing Peptide-Guided Protein Degraders

![saltnpeppr_inference](https://user-images.githubusercontent.com/106272333/196185861-40837a34-2164-4a95-bdf0-30ce9b4b4b9f.png)

Targeted protein degradation of pathogenic proteins represents a powerful new treatment strategy for multiple disease indications. Unfortunately, a sizable portion of these proteins are considered “undruggable” by standard small molecule-based approaches, including PROTACs and molecular glues, largely due to their disordered nature, instability, and lack of binding site accessibility. As a more modular, genetically-encoded strategy, designing functional protein-based degraders to undruggable targets presents a unique opportunity for therapeutic intervention. In this work, we integrate pre-trained protein language models with recently-described joint encoder architectures to devise a unified, sequence-based framework to design target-selective peptide degraders without structural information. By leveraging known experimental binding protein sequences as scaffolds, we create a Structure-agnostic Language Transformer & Peptide Prioritization (SaLT&PepPr) module that efficiently selects peptides for downstream screening.

Please read and cite our [manuscript](https://www.nature.com/articles/s42003-023-05464-z) published in *Communications Biology*!

All manuscript data/pipelines, as required/indicated in the manuscript Availability statements, are available on [Zenodo](https://zenodo.org/records/10008581) and in the .zip file within this repo.

Please find training and inference code on [HuggingFace](https://huggingface.co/ubiquitx/saltnpeppr)!  
