<h1 align="center">MMFact</h1>

<p align="center">
  <img src="images/logo.png" alt="MMFact logo" width="240">
</p>

<p align="center">
  <strong>MMFact</strong>: A unified toolbox for <strong><ins>M</ins></strong>ulti<strong><ins>M</ins></strong>odal misinformation detection.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.34162"><img src="https://img.shields.io/badge/arXiv-2610.34162-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="arXiv: 2610.34162" height="20"></a>
  <a href="https://huggingface.co/datasets/MMFact/MMFact"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat&amp;logo=huggingface" alt="Hugging Face: Dataset" height="20"></a>
  <a href="https://huggingface.co/MMFact/MMFact-CKPT"><img src="images/badges/huggingface-ckpt.svg" alt="Hugging Face: CKPT" height="20"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python &gt;=3.10" height="20"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?style=flat&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch &gt;=2.2" height="20"></a>
</p>

## Toolbox

<img src="images/toolbox.png" alt="MMFact toolbox" width="640">

## Quick Start

Requires Python 3.10 or later.

**Installation**

```bash
git clone https://github.com/yuchenmiao04/MMFact.git
cd MMFact
pip install -e ".[hub,models]"
```

**Download Materials and Run**

Request access to the [MMFact dataset](https://huggingface.co/datasets/MMFact/MMFact). After approval, run `hf auth login` with the approved account before downloading materials.

For example, run CLIPMLP on Weibo-17:

```bash
python main.py --model CLIPMLP --dataset weibo17 --download-materials

python main.py --model CLIPMLP --dataset weibo17 \
  --input-mode prepared --materials-root materials
```

Use `--model` and `--dataset` to select other models and datasets. Configuration files are in `configs/`, and results are saved to `runs/`.

Ordinary supervised visual branches default to **ViT-B/16**. Add
`--vision-backbone resnet` to **both** the download and training commands for the
ResNet-50 comparison. Both choices use the same selected experiment parameters;
MAE, CLIP, and dedicated detection or forensic branches retain their own backbones.
See [Visual Backbones](VISION_BACKBONES.md) for feature formats and fine-tuning behavior.

## Datasets

| **Dataset** | **Train** | **Validation** | **Test** | **Images** |
| ----------- | --------: | -------------: | -------: | ---------: |
| Weibo-17 | 5,415 | 842 | 1,465 | 7,717 |
| Weibo-21 | 4,807 | 601 | 601 | 5,877 |
| GossipCop | 8,842 | 1,894 | 1,894 | 12,014 |

## Supported Models

| **Model** | **Paper** | **Conference/Journal** | **Code** |
| --------- | --------- | ---------------------- | -------- |
| Majority |  |  | [majority.py](src/mmfact/models_classification/majority.py) |
| TFIDFLogistic |  |  | [tfidflogistic.py](src/mmfact/models_classification/tfidflogistic.py) |
| TextPLM | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)<br>[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) | NAACL 2019<br>arXiv 2020 | [textplm.py](src/mmfact/models_classification/textplm.py) |
| ImageOnly | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)<br>[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | ICLR 2021<br>CVPR 2016 | [imageonly.py](src/mmfact/models_classification/imageonly.py) |
| ConcatMLP |  |  | [concatmlp.py](src/mmfact/models_classification/concatmlp.py) |
| CLIPLinear |  |  | [cliplinear.py](src/mmfact/models_classification/cliplinear.py) |
| CLIPSVM |  |  | [clipsvm.py](src/mmfact/models_classification/clipsvm.py) |
| CLIPMLP |  |  | [clipmlp.py](src/mmfact/models_classification/clipmlp.py) |
| LateFusion |  |  | [latefusion.py](src/mmfact/models_classification/latefusion.py) |
| att-RNN | [Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs](https://doi.org/10.1145/3123266.3123454) | ACM MM 2017 | [att_rnn.py](src/mmfact/models_classification/att_rnn.py) |
| EANN | [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://doi.org/10.1145/3219819.3219903) | KDD 2018 | [eann.py](src/mmfact/models_classification/eann.py) |
| MVAE | [MVAE: Multimodal Variational Autoencoder for Fake News Detection](https://doi.org/10.1145/3308558.3313552) | WWW 2019 | [mvae.py](src/mmfact/models_classification/mvae.py) |
| SpotFake | [SpotFake: A Multi-modal Framework for Fake News Detection](https://precog.iiit.ac.in/pubs/SpotFake-IEEE_BigMM.pdf) | BigMM 2019 | [spotfake.py](src/mmfact/models_classification/spotfake.py) |
| SpotFake+ | [SpotFake+: A Multimodal Framework for Fake News Detection via Transfer Learning (Student Abstract)](https://ojs.aaai.org/index.php/AAAI/article/view/7230) | AAAI 2020 Student Abstract | [spotfakeplus.py](src/mmfact/models_classification/spotfakeplus.py) |
| SAFE | [SAFE: Similarity-Aware Multi-Modal Fake News Detection](https://arxiv.org/abs/2003.04981) | PAKDD 2020 | [safe.py](src/mmfact/models_classification/safe.py) |
| HMCAN | [Hierarchical Multi-modal Contextual Attention Network for Fake News Detection](https://doi.org/10.1145/3404835.3462871) | SIGIR 2021 | [hmcan.py](src/mmfact/models_classification/hmcan.py) |
| MCAN | [Multimodal Fusion with Co-Attention Networks for Fake News Detection](https://aclanthology.org/2021.findings-acl.226/) | Findings of ACL 2021 | [mcan.py](src/mmfact/models_classification/mcan.py) |
| EM-FEND | [Improving Fake News Detection by Using an Entity-Enhanced Framework to Fuse Diverse Multimodal Clues](https://doi.org/10.1145/3474085.3481548) | ACM MM 2021 Industrial Track | [em_fend.py](src/mmfact/models_classification/em_fend.py) |
| BTIC | [Supervised Contrastive Learning for Multimodal Unreliable News Detection in COVID-19 Pandemic](https://doi.org/10.1145/3459637.3482196) | CIKM 2021 Short | [btic.py](src/mmfact/models_classification/btic.py) |
| CAFE | [Cross-modal Ambiguity Learning for Multimodal Fake News Detection](https://doi.org/10.1145/3485447.3511968) | WWW 2022 | [cafe.py](src/mmfact/models_classification/cafe.py) |
| LIIMR | [Leveraging Intra and Inter Modality Relationship for Multimodal Fake News Detection](https://doi.org/10.1145/3487553.3524650) | WWW Companion 2022 | [liimr.py](src/mmfact/models_classification/liimr.py) |
| CMC | [Cross-Modal Knowledge Distillation in Multi-Modal Fake News Detection](https://doi.org/10.1109/ICASSP43922.2022.9747280) | ICASSP 2022 | [cmc.py](src/mmfact/models_classification/cmc.py) |
| BMR | [Bootstrapping Multi-View Representations for Fake News Detection](https://ojs.aaai.org/index.php/AAAI/article/view/25670) | AAAI 2023 | [bmr.py](src/mmfact/models_classification/bmr.py) |
| MRML | [MRML: Multimodal Rumor Detection by Deep Metric Learning](https://doi.org/10.1109/ICASSP49357.2023.10096188) | ICASSP 2023 | [mrml.py](src/mmfact/models_classification/mrml.py) |
| MMFN | [Multi-modal fake news detection on social media via multi-grained information fusion](https://doi.org/10.1145/3591106.3592271) | ICMR 2023 | [mmfn.py](src/mmfact/models_classification/mmfn.py) |
| HAMMER | [Detecting and Grounding Multi-Modal Media Manipulation](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf) | CVPR 2023 | [hammer.py](src/mmfact/models_classification/hammer.py) |
| LogicDM | [Interpretable Multimodal Misinformation Detection with Logic Reasoning](https://aclanthology.org/2023.findings-acl.620/) | Findings ACL 2023 | [logicdm.py](src/mmfact/models_classification/logicdm.py) |
| FND-CLIP | [Multimodal Fake News Detection via CLIP-Guided Learning](https://doi.org/10.1109/ICME55011.2023.00480) | ICME 2023 | [fnd_clip.py](src/mmfact/models_classification/fnd_clip.py) |
| COOLANT | [Cross-modal Contrastive Learning for Multimodal Fake News Detection](https://doi.org/10.1145/3581783.3613850) | ACM MM 2023 | [coolant.py](src/mmfact/models_classification/coolant.py) |
| CSFND | [Not all fake news is semantically similar: Contextual semantic representation learning for multimodal fake news detection](https://doi.org/10.1016/j.ipm.2023.103564) | IPM 61(1), 2024 | [csfnd.py](src/mmfact/models_classification/csfnd.py) |
| RDCM | [Robust Domain Misinformation Detection via Multi-Modal Feature Alignment](https://doi.org/10.1109/TIFS.2023.3326368) | TIFS 19, 2024 | [rdcm.py](src/mmfact/models_classification/rdcm.py) |
| FSRU | [Frequency Spectrum Is More Effective for Multimodal Representation and Fusion: A Multimodal Spectrum Rumor Detector](https://doi.org/10.1609/aaai.v38i16.29803) | AAAI 2024 | [fsru.py](src/mmfact/models_classification/fsru.py) |
| NSLM | [Unveiling Implicit Deceptive Patterns in Multi-Modal Fake News via Neuro-Symbolic Reasoning](https://doi.org/10.1609/aaai.v38i8.28677) | AAAI 2024 | [nslm.py](src/mmfact/models_classification/nslm.py) |
| MMCAN | [Multimodal matching-aware co-attention networks with mutual knowledge distillation for fake news detection](https://doi.org/10.1016/j.ins.2024.120310) | Information Sciences 664, 2024 | [mmcan.py](src/mmfact/models_classification/mmcan.py) |
| MSACA | [Fake News Detection via Multi-scale Semantic Alignment and Cross-modal Attention](https://doi.org/10.1145/3626772.3657905) | SIGIR 2024 | [msaca.py](src/mmfact/models_classification/msaca.py) |
| NLIN | [Natural Language-centered Inference Network for Multi-modal Fake News Detection](https://www.ijcai.org/proceedings/2024/0281.pdf) | IJCAI 2024 | [nlin.py](src/mmfact/models_classification/nlin.py) |
| Event-Radar | [Event-Radar: Event-driven Multi-View Learning for Multimodal Fake News Detection](https://aclanthology.org/2024.acl-long.316/) | ACL 2024 | [event_radar.py](src/mmfact/models_classification/event_radar.py) |
| MMDFND | [MMDFND: Multi-modal Multi-Domain Fake News Detection](https://doi.org/10.1145/3664647.3681317) | ACM MM 2024 | [mmdfnd.py](src/mmfact/models_classification/mmdfnd.py) |
| AAR | [Unveiling Fake News with Adversarial Arguments Generated by Multimodal Large Language Models](https://aclanthology.org/2025.coling-main.526/) | COLING 2025 | [aar.py](src/mmfact/models_classification/aar.py) |
| ERIC-FND | [External Reliable Information-enhanced Multimodal Contrastive Learning for Fake News Detection](https://doi.org/10.1609/aaai.v39i1.31977) | AAAI 2025 | [eric_fnd.py](src/mmfact/models_classification/eric_fnd.py) |
| DAMMFND | [DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection](https://doi.org/10.1609/aaai.v39i1.32036) | AAAI 2025 | [dammfnd.py](src/mmfact/models_classification/dammfnd.py) |
| GAMED | [GAMED: Knowledge Adaptive Multi-Experts Decoupling for Multimodal Fake News Detection](https://doi.org/10.1145/3701551.3703541) | WSDM 2025 | [gamed.py](src/mmfact/models_classification/gamed.py) |
| MHR | [MHR: A Multi-Modal Hyperbolic Representation Framework for Fake News Detection](https://doi.org/10.1109/TKDE.2025.3528951) | TKDE 37(4), 2025 | [mhr.py](src/mmfact/models_classification/mhr.py) |
| MIMoE | [Modality Interactive Mixture-of-Experts for Fake News Detection](https://doi.org/10.1145/3696410.3714522) | WWW 2025 | [mimoe.py](src/mmfact/models_classification/mimoe.py) |
| C3N | [Improving multimodal fake news detection by leveraging cross-modal content correlation](https://doi.org/10.1016/j.ipm.2025.104120) | IP&M 2025 | [c3n.py](src/mmfact/models_classification/c3n.py) |
| MMRGV | [Toward Multimodal Fake News Detection by Multi-perspective Rationale Generation and Verification](https://ojs.aaai.org/index.php/AAAI/article/view/36965) | AAAI 2026 | [mmrgv.py](src/mmfact/models_classification/mmrgv.py) |
| ADOSE | [Active Multi-source Domain Adaptation for Multimodal Fake News Detection](https://doi.org/10.1609/aaai.v40i24.39125) | AAAI 2026 | [adose.py](src/mmfact/models_classification/adose.py) |

#### Please consider to cite our paper if this framework helps you, thanks:

```bibtex
@misc{miao2026mmfactsimplifyingmultimodalmisinformation,
  title={MMFact: Simplifying Multimodal Misinformation Detection with Shared Resources and Tools},
  author={Yuchen Miao and Zijun Wang and Mingtai Zhang and Xiaoyi Chen and Sihan Deng and Xianglin Ding},
  year={2026},
  eprint={2610.34162},
  archivePrefix={arXiv},
  primaryClass={cs.MM},
  url={https://arxiv.org/abs/2610.34162},
}
```
