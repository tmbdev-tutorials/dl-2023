---
marp: true
headingDivider: 1
paginate: true
footer: Lectures on Self-Supervised Learning - Thomas Breuel - NVIDIA
---
# UN/SEMI-SUPERVISED LEARNING IN VISION

![w:800px h:200px](Figures/gray.png)

# Already covered...

We have already covered (in the OCR case):

- pseudo-labels
- active learning
- data augmentation and prior knowledge of invariances
- EM algorithms

# Pre-Transformer Approaches

Prior attempts to carry over principles for language models and HMMs to the image domain:

- linearize images and apply HMM or LSTM models
- apply VQ to patches and apply syntactic models to the resulting "visual words"
- corrupt images with noise and training a network to restore them
- predict color images from grayscale
- mask parts of images and predict the masked parts (like BERT)
- determine the spatial relations between patches (like entailment)

All of these yield deep learning architectures that are potentially useful for transfer learning, but never beat supervised models.

# Masked Predictions

![](Figures/pathak-context-encoders.png) 

Pathak et al. 2016

# Split-Brain Autoencoder

![](Figures/zhang-split-brain.png) 

Zhang et al. 2016

# Context Encoding

![](Figures/doersch-context-encoding.png)

Doersch et al. 2016

# VISION TRANSFORMERS

# Vision Transformers

![](Figures/dosovitskiy-vit.png)

The NLP transformer architecture carries over directly to images: just change the positional embedding.  Dosovitskiy et al. 2020, arXiv:2010.11929

# BEiT - BERT Pre-Training of Image Transformers

![](Figures/beit-architecture.png) 

BERT-like pretraining carries over directly. Bao et al., 2022

# Masked Autoencoder (MAE)

![](Figures/he-mae-architecture.png) 

MAE uses a simpler architecture and no tokenization. He et al. 2021

# Masked Autoencoder - Reconstructions

![](Figures/mae-reconstructions.png) 

He et al. 2021

# Masked Autoencoder - Transfer Learning

![](Figures/mae-transfer.png)

# Unsupervised Training with Transformers

Transformer architectures make BERT-like masking useful unsupervised pre-training for image-related tasks.

# OTHER APPROACHES

(Not transformer based)

- SimCLR
    - generate two differently augmented version of the same image
    - train a representation that is as similar as possible

- DINO
    - self-DIstillation with NO labels
    - discovers labels / class structure by itself

Tricky to train, highly dependent on chioce of augmentations/parameters.

<!--


# SimCLR - Contrastive Learning

Basic idea:

- generate two differently augmented versions of the same image
- train a representation that is as similar as possible for the same image, different for different images

Details:

- carefully choose augmentations, watch out for trivial solutions (e.g. color)
- separate representation from scoring
- compute "softmax over cosine similarity" over very large batches
- implemented with ResNet

# SimCLR - Contrastive Learning

![](Figures/simclr-architecture.png)

Chen et al., 2020; arXiv:2002.05709

# SimCLR - Augmentations

![](Figures/simclr-augmentations.png)

# SimCLR - Transfer Learning Performance

![](Figures/simclr-transfer-learning.png)

# DINO

- self-DIstillation with NO labels
- discovers labels / class structure by itself
- unsupervised representation learning for images
- impressive semantic segmentation results
- attention map = segmentation map
- vision transformer or ResNet 50 based

# DINO Architecture

![](Figures/dino-architecture.png)

# DINO Results

![](Figures/dino-results.png)

# DINO Segmentation

![](Figures/dino-sematic-seg.png)

- semantic segmentation by attention looks better than from supervised models

# Summary: SimCLR and DINO

High Level:

- SimCLR is a kind of _representation_ or _metric_ learning
- DINO is a kind of _clustering_
- implementations are complex and with lots of hyperparameters

Conclusions:

- tasks like these may be most useful as additional tasks combined with masking (just like BERT)

-->



# COMBINING TEXT AND IMAGE MODELS

# Combining Image and Text Models

- GPT-3, ExT5, etc. show how natural language models can be used for zero short learning
- CLIP
    - use natural language supervision for image recognition (weak supervision)
    - vision: transformer or ResNet, language: transformer
    - permit "prompt engineering" to allow different kinds of NLP tasks
    - uses contrastive pretraining (rather than, say, captioning)

# CLIP

![w:800 h:200](Figures/gray.png)

# CLIP Architecture

![](Figures/clip-architecture.png)

# CLIP Results

![](Figures/clip-results.png)

# CLIP Results

![](Figures/clip-results2.png)

# Discussion

Current and future directions:

- combining text and image
- integrating unsupervised pretrained vision and language models
- integrating video and audio and identifying good self-supervised tasks for these (e.g., VideoMAE, Audio-MAE)
- cross-modal combinations likely also reduce the amount of training data required within each modality

# Joint Embeddings

![h:200 w:800](Figures/gray.png)

# Joint Embeddings

- ViLBERT (Lu et al., 2019): visual question answering tasks
- LXMERT (Tan and Bansal, 2019): visual question answering, image captioning, and visual entailment.
- UNITER (Chen et al., 2020): image-text retrieval, image captioning, and visual question answering
- OSCAR (Li et al., 2020): image-text retrieval, image captioning, and visual question answering
- VILLA (Gupta et al., 2020): video; uses a hierarchical architecture to encode both frame-level features and clip-level features from videos along with textuald descriptions

# Joint Embeddings -- UNITER

- Joint image-text embedding for V+L tasks
- Large-scale pre-training over four datasets
    - COCO, Visual Genome, Conceptual Captions, SBU Captions
- Four main pre-training tasks evaluated
    - Masked Language Modeling (MLM)
    - Image-Text Matching (ITM)
    - Masked Region Modeling (MRM)
    - Masked Object Classification (MOC)

# UNITER

![](Figures/uniter.png)

# Segment Anything (SAM)

![h:200 w:800](Figures/gray.png)

# Segment Anything

![](Figures/sam-1.png)

# Segment Anything

![](Figures/sam-2.png)


# Flamingo

![h:200 w:800](Figures/gray.png)

# Flamingo

![](Figures/flamingo-arch.png)

# Flamingo Architecture

- Flamingo is a VLM with a vision encoder and language decoder.
- Vision encoder maps input to visual embeddings.
- Language decoder generates text from visual embeddings.
- Includes architectural innovations for interleaved data.
- Trained using contrastive training.

# Flamingo - Adapters

- Flamingo is a generalization of adapter modules.
- Adapter modules are small neural networks inserted between layers of a pre-trained model.
- GATED XATTN-DENSE is initialized using an atanh-gating mechanism.
- Unlike adapter modules, Flamingo is designed to add completely new functionality to an existing model.

# MetaLM / KOSMOS-1

![h:200 w:800](Figures/gray.png)

# MetaLM

![h:500](Figures/MetaLM.png)

# MetaLM

![h:500](Figures/MetaLM2.png)

# MetaLM

![h:500](Figures/MetaLM3.png)

# MetaLM Performance

- VQA -- Visual question answering
  - Size: 250k images, 1.2M questions
  - Previous best: 66.7% accuracy
  - New result: 68.4% accuracy

- COCO -- Image captioning
  - Size: 123k images
  - Previous best: 36.2 CIDEr score
  - New result: 40.4 CIDEr score

- StoryCloze -- Given a four-sentence story and two possible endings, choose the correct ending.
   - New result: State-of-the-art performance

# KOSMOS-1

![h:500](Figures/kosmos-1-1.png)

# KOSMOS-1

![h:500](Figures/kosmos-1-2.png)

# SUMMARY

![w:800 h:200](Figures/gray.png)

# SUMMARY

- joint image and text models look very promising
    - text provides zero-shot learning capability to imges
    - images provide additional grounding/semantics to text
- performance is mixed
    - slow inference
    - pure vision models may still be better on specific tasks
