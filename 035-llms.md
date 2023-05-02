---
marp: true
headingDivider: 1
paginate: true
footer: Lectures on Self-Supervised Learning - Thomas Breuel - NVIDIA
---
# UNSUPERVISED LEARNING FOR NLP

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# History

- language modeling is the prototypical unsupervised learning task:
    - large amounts of unlabeled text
    - strong internal statistical regularities
    - simple statistical models are themselves useful for speech recognition and other applications
- surprisingly (?) when language models became large enough, intelligent behavior emerged

# Traditional Language Modeling task

Predict the next word in a sequence:

- generate new text (sampling)
- assign probabilities to strings (language modeling)

Examples:

- n-gram models, finite state transducers
- TDNN, LSTM models (e.g. Graves 2013, Arxiv 1308.0850)

# Statistical Language Modeling

- Aims to model generative likelihood of word sequences
- Predicts probabilities of future or missing tokens
- Developed based on statistical learning methods that rose in the 1990s

# Pre-Transformer History

1. N-gram models (Shannon) - 1948
2. Hidden Markov models (Rabiner) - 1989
3. Maximum entropy models (Berger et al.) - 1996
7. LSTMs (Sepp et al.) -- 1997
4. Conditional random fields (Lafferty et al.) - 2001
5. Neural network language models (Bengio et al.) - 2003
8. Transformer-based language models (Vaswani et al.) - 2017

(NB: many of these actually have prior work, but these are the papers commonly referenced these days.)

# Embeddings: word2vec and ELMO

![embeddings](Figures/embeddings.jpg)

- self-supervised task: predict word from context
- bidirectional (since we're not using it for autoregressive decoding)
- embeddings used as inputs to other systems


# TRANSFORMERS (QUICK REVIEW)

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Transformer Architecture

- fundamental change in sequence modeling
    - arbitrarily long term dependencies through content-addressability
    - all time steps trainable in parallel

- history
    - added attention mechanisms to LSTM to improve performance
    - later, eliminated the LSTM altogether and retained just attention

Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems, vol. 30, 2017, pp. 5998-6008.

# Transformers are Trained in Parallel

![](Figures/transformer-training.jpg)

(Like convolutional networks.)

# Transformers are Set Learners

![](Figures/transformers-are-set-learners.jpg)

The order of inputs do a transformer doesn't matter as far as the model is concerned.

# For Sequence Tasks, we use Autoregressive Decoding

![](Figures/transformer-translation.png)

# Attention in Transformers

![](Figures/transformer-attention.jpg)

# Sequences are Modeled via Positional Encodings

![](Figures/positional-encoding.jpg)


# HISTORY OF TRANSFORMER-BASED MODELS

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Survey Paper

Liu, X., Li, M., Gao, J., Wu, S., & Chen, D. (2021). A Survey of Pre-trained Language Models. In Proceedings of the Association for Computational Linguistics (pp. 1-88).

# Attention is All You Need

- Overview of the "Attention is All You Need" paper
- Introduction to the Transformer architecture (encoder-decoder architecture with self-attention mechanism)
- Important architectural variants, such as changes in layer normalization

Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems, vol. 30, 2017, pp. 5998-6008.

# Some Key Developments

1. Transformer architecture (Transformer), Vaswani et al.
2. Unsupervised pre-training and fine-tuning (GPT-1), Radford et al.
3. Multi-task learning and larger model size (GPT-2), Radford et al.
4. Permutation-based training objective (XLNet), Yang et al.
5. Large-scale pre-training and data augmentation (RoBERTa), Liu et al.
6. Unified text-to-text framework (T5), Raffel et al.
7. Parallelism across multiple devices (GShard), Kitaev et al.
8. Dynamic computation graphs for variable-length inputs (Switch Transformers), Fedus et al.

# GPT-1

- Generative Pre-trained Transformer 1
- Language model by OpenAI (2018)
- Pre-trained on large corpus of text data using unidirectional transformer
- Captures context from left of a token
- Uses unsupervised learning to predict next word in a sequence
- Fine-tuned on downstream tasks such as language modeling and text classification
- Strongest at: Language modeling tasks such as Penn Treebank dataset


# BERT

- Bidirectional Encoder Representations from Transformers
- Language model by Google (2019)
- Pre-trained on large corpus of text data using bidirectional transformers
- Captures context from both left and right of a token
- Uses masked language modeling and next sentence prediction tasks during pre-training
- Fine-tuned on downstream tasks such as question answering, sentiment analysis, and named entity recognition
- Strongest at: Question answering tasks such as SQuAD and GLUE benchmark datasets
- Important systems based on BERT: RoBERTa, ALBERT, DistilBERT

# GPT-2

- Generative Pre-trained Transformer 2
- Language model by OpenAI (2019)
- Pre-trained on large corpus of text data using unidirectional transformer
- Increased parameter scale to 1.5B compared to GPT-1
- Uses unsupervised learning to predict next word in a sequence
- Fine-tuned on downstream tasks such as language modeling and text generation
- Strongest at: Text generation tasks such as story writing and poetry composition
- Weakest or unsuitable for: Tasks requiring external knowledge or reasoning beyond the given text

# XLNet

- eXtreme Language understanding NETwork
- Language model by CMU and Google (2019)
- Pre-trained on large corpus of text data using permutation-based training
- Captures context from both left and right of a token
- Uses unsupervised learning to predict next word in a sequence
- Fine-tuned on downstream tasks such as question answering and natural language inference
- Strongest at: Question answering tasks such as SQuAD 2.0 dataset
- Weakest or unsuitable for: Tasks requiring external knowledge or reasoning beyond the given text

# XLNet-Style Permutation Training Example

- Task: Language Modeling
- Input sequence: "The quick brown fox jumps over the lazy dog."
- Randomly permute input sequence to create new sequence: "The dog over quick brown jumps the lazy fox."
- Train model to predict original sequence given permuted sequence
- Repeat permutation process for each training example
- Model captures context from both left and right of each token during training and inference
- Permutation-based training allows model to learn bidirectional dependencies without using a specific order or directionality
- Resulting model can perform well on tasks requiring understanding of long-range dependencies

# T5

- Text-to-Text Transfer Transformer
- Language model by Google (2019)
- Pre-trained on large corpus of text data using a unified text-to-text format
- Captures context from both left and right of a token
- Uses supervised learning to map input text to output text for various tasks
- Fine-tuned on downstream tasks such as question answering and summarization
- Strongest at: Multi-task learning across various NLP tasks such as translation, summarization, and classification

# ExT5

![](Figures/ext5.png)

# ELECTRA

- Efficiently Learning an Encoder that Classifies Token Replacements Accurately
- Language model by Google (2020)
- Pre-trained on large corpus of text data using a generator-discriminator framework
- Discriminator predicts whether each token in a sequence is real or generated by the generator
- Uses unsupervised learning to predict next word in a sequence
- Fine-tuned on downstream tasks such as sentiment analysis and named entity recognition
- Strongest at: Adversarial training for better generalization and sample efficiency compared to GPT-series models

# ELECTRA-Style Adversarial Training Example

- Task: Sentiment Analysis
- Generator: Masked Language Model (MLM)
- Discriminator: Binary Classifier
- MLM replaces 15% of input tokens with random tokens and trains to predict original tokens
- Discriminator takes input sequence and predicts whether each token is real or generated by MLM
- Discriminator trained to maximize binary cross-entropy loss on real/generated labels
- MLM trained to minimize binary cross-entropy loss on discriminator's generated labels
- Adversarial objective: Maximize discriminator's loss while minimizing MLM's loss

# ChatGPT

- Conversational AI system by Microsoft (2019)
- Pre-trained on large corpus of text data using unsupervised learning
- Possesses a vast store of knowledge and skill at reasoning on mathematical problems
- Traces the context accurately in multi-turn dialogues
- Aligns well with human values for safe use
- Supports plugin mechanism to extend capacities with existing tools or apps
- Strongest at: Multi-turn dialogues and aligning with human values for safe use

# GPT-3

- Generative Pre-trained Transformer 3
- Language model by OpenAI (2020)
- Pre-trained on large corpus of text data using unsupervised learning
- Scaled model parameters to an ever larger size of 175B
- Demonstrates a key capacity leap by scaling the generative pre-training architecture
- Fine-tuned on downstream tasks such as language translation and question answering
- Strongest at: Few-shot learning across various NLP tasks such as translation, summarization, and classification

# Flan-T5

- "Few-shot Language Adaptation with Transformers"
- fine-tuned on a larger corpus of text data
- fine tuning: increased # tasks, chain-of-thought data
- strong zero-shot, few-shot, and CoT abilities, outperforming prior public checkpoints such as T5. 
- Flan-T5 11B outperformed T5 11B by double-digit improvements and even outperformed PaLM 62B on some challenging BIG-Bench tasks



# GPT-4
- Overview of language modeling and its evolution
- Introduction to pre-trained language models (PLMs)

# BENCHMARKS

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Benchmarks

- MMLU: A benchmark that focuses on evaluating LLMs' ability to understand and generate natural language.
- BIG-bench: A benchmark that contains a large number of diverse tasks to evaluate LLMs' generalization ability.
- HELM: A benchmark that focuses on evaluating LLMs' ability to perform complex reasoning tasks.


# BIG-Bench

1. Extractive Question Answering
2. Abstractive Summarization
3. Sentiment Analysis
4. Named Entity Recognition
5. Text Classification
6. Machine Translation
7. Pronoun Disambiguation
8. Coreference Resolution
9. Word Sense Disambiguation
10. Paraphrase Detection
11. Natural Language Inference
12. Commonsense Reasoning

# PRETRAINED OPEN SOURCE

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Available Language Models

- Developing or reproducing LLMs is challenging due to technical issues and computation demands.
- Training from scratch can cost millions.
- Taking an existing model and fine-tuning it is an option.
- This section summarizes publicly available resources for developing LLMs, including model checkpoints (or APIs), corpora, and libraries.

# Model Checkpoints
- Pre-trained models that can be fine-tuned for specific tasks.
- Publicly available for many popular LLM architectures such as GPT, T5, and BERT.
- Can be used as starting points for training new models or as feature extractors for downstream tasks.

# Hugging Face Transformers
- Popular library that provides access to many pre-trained LLM models.
- Includes a wide range of architectures such as GPT, T5, BERT, and more.
- Provides tools for fine-tuning models on custom datasets.

# OpenAI GPT Models
- Several versions of their GPT model with varying sizes (e.g., GPT-2, GPT-3).
- Pre-trained on large amounts of text data and can be fine-tuned for various NLP tasks.
- Available through OpenAI's API or can be downloaded from their website.

# Google T5 Models
- Several versions of their T5 model with varying sizes (e.g., T5-small, T5-base).
- Pre-trained on large amounts of text data and can be fine-tuned for various NLP tasks.
- Available through Google's Cloud AI Platform or can be downloaded from their website.

# TRAINING FROM SCRATCH

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Commonly Used Corpora for Training LLMs

- Books: 800M words, diverse genres
- CommonCrawl: 42B words, web text
- Reddit links: 1.7B words, social media text
- Wikipedia: 2.5B words, encyclopedia articles
- Code: 14M functions, programming code
- Others: e.g., news articles, scientific papers

# How much can we do with a single GPU?

Geiping, J., & Goldstein, T. (2022). CRAMMING: Training a Language Model on a Single GPU in One Day. arXiv preprint arXiv:2212.14034.

- Trends in language modeling focus on scaling, making training out of reach for most researchers/practitioners
- This paper asks: how far can we get with a single GPU in one day?
- Investigate downstream performance of transformer-based language model trained from scratch with masked language modeling on a single consumer GPU

# Hardware Configurations

The experiments were conducted using the following hardware configurations:

- NVIDIA GeForce GTX 1080 Ti graphics card with 11GB of memory
- NVIDIA RTX 2080 Ti graphics card with 11GB of memory
- NVIDIA RTX A4000 graphics card with 16GB of memory
- NVIDIA RTX A6000 graphics card with 48GB of memory
- Intel Xeon E5-2690 v4 CPU @ 2.60GHz
- 256GB RAM
- 1TB NVMe SSD storage

# Key Results

- Achieves competitive performance compared to BERT while using significantly fewer resources
- Test perplexity of 20.5 and test accuracy of 87.3% on the GLUE benchmark ( BERT's accuracy of 89.3%)
- Outperforms other models trained with similar resources by a significant margin
- Outperforms GPT-2 small and RoBERTa base on the GLUE benchmark while using only a fraction of their training time and computational resources

# REUSING PRETRAINED MODELS

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Fine-Tuning and Adaptation

- given a pretrained model and a new task or new domain, how can we take advantage of the existing model?
    - prompt tuning
    - providing context / search results
    - fine tuning
    - adapter tuning
    - low rank methods

# Prompt Tuning
- Prompt tuning is another parameter-efficient fine-tuning method that uses natural language prompts to guide the generation of text.
- The prompts are used as input to the pre-trained language model, which generates text conditioned on them.
- This approach has been shown to be effective for few-shot learning, where only a small amount of labeled data is available for training.

# Examples of Prompt Tuning

1. For a sentiment analysis task, a prompt could be "Please classify the sentiment of this movie review as positive, negative, or neutral."
2. For a question answering task, a prompt could be "Answer the following question: What is the capital city of France?"
3. For a text generation task, a prompt could be "Generate a sentence that describes the weather today in New York City."

# In Context Learning (ICL)

- ICL goes further than prompt tuning by giving "training examples" in the prompt
- ICL involves selecting examples from the task dataset, combining them with templates to form natural language prompts, and using them to train LLMs.
- An example of ICL would be training an LLM to perform sentiment analysis on movie reviews by using a few labeled movie reviews as examples and natural language prompts.

# Chain-of-Thought Prompting

- CoT (Chain-of-Thought) is a prompting strategy that incorporates intermediate reasoning steps into prompts to enhance in-context learning.
- An example of CoT would be training an LLM to solve a math problem by using prompts with intermediate reasoning steps.
- The prompts would guide the LLM through the thought process of solving the problem and help it learn to perform more complex reasoning tasks.
- CoT can be used to boost the performance of LLMs on various tasks, such as arithmetic reasoning, commonsense reasoning, and symbolic reasoning.

# Search / Context

- context (e.g., search results) can alleviate the need for retraining
- model can leverage the information contained in the search results
- e.g. generating a product review -- augment with search results of product reviews

# External Knowledge Utilization

- Closed-book tasks evaluate LLMs' ability to answer questions without access to external resources.
- Open-book tasks evaluate LLMs' ability to answer questions with access to external resources.
- Forms of knowledge utilization:
    - external text-based search resources
        - e.g.: Bing
    - retrieval of LLM states from vector databases
        - e.g.: RETRO

# Fine-Tuning Methods
- Several fine-tuning methods have been proposed in the literature, including:
  - Full fine-tuning: retraining the entire model on the target task.
  - Layer-wise fine-tuning: freezing some layers and only training others.
  - Multi-task learning: training the model on multiple tasks simultaneously.
  - Knowledge distillation: transferring knowledge from a larger teacher model to a smaller student model.

# Adapter Tuning

- Adapter tuning is a parameter-efficient fine-tuning method that aims to reduce the number of trainable parameters while retaining good performance.
- Adapters are small neural networks that are added to pre-trained models and trained only on specific tasks.
- This approach allows for efficient adaptation to new tasks without requiring full retraining of the original model.

# LoRA (Low-Rank Adaptation)

- LoRA is a parameter-efficient fine-tuning method that reduces the trainable parameters for adapting to downstream tasks.
- It imposes the low-rank constraint for approximating the update matrix at each dense layer.
- The basic idea of LoRA is to freeze the original matrix $W$ while approximating the parameter update $\Delta W$ by low-rank decomposition matrices, i.e., $\Delta W = A \cdot B^T$, where $A$ and $B$ are the trainable parameters for task adaptation and $r \leq \min(m,n)$ is the reduced rank.
- The major merit of LoRA is that it can largely save memory and storage usage while maintaining good performance on downstream tasks.

# APPLICATION CONSIDERATIONS

<svg width="100%" height="200">
  <rect x="0" y="0" width="100%" height="200" fill="#cccccc" />
</svg>

# Applications

- Customer service: generate automated responses to customer inquiries or complaints.
- E-commerce: personalized product recommendations or answer customer questions about products.
- Healthcare: medical diagnosis or provide information about treatments and medications.
- Finance: fraud detection, risk assessment, and investment analysis.
- Education: generate educational materials, provide feedback on student work, and support language learning.
- Law: legal research, contract analysis, and document summarization.

# Application: Code Synthesis

- Code synthesis is different from natural language generation.
- LLMs can generate formal language, such as computer programs (code), that satisfy specific conditions.
- Generated code can be directly checked by execution with corresponding compilers or interpreters.
- Existing work evaluates the quality of generated code from LLMs by calculating the pass rate against test cases.
- Examples: Copilot, Code Whisperer

# Safety

- LLMs pose safety challenges and generate hallucinations.
- LLMs can be used to produce harmful texts for malicious systems.
- GPT-3/4 technical reports discuss safety issues of LLMs.
- Researchers propose techniques like prompt engineering and adversarial training to address safety concerns.

# Alignment

- Alignment: generating text that aligns with human values.
- It involves ensuring the model's outputs are helpful, honest, and harmless.
- Helpfulness: generating relevant and useful text.
- Honesty: generating factually accurate and truthful text.
- Harmlessness: generating text that does not cause harm or promote bias.
- Benchmarks: CrowS-Pairs and Winogender

(Question: _whose_ values?)

