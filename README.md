# LightLlama

LightLlama is a lightweight implementation of the Llama architecture from scratch, inspired by Stanford CS336 (2025).  
This project aims to provide a clear and concise codebase for understanding and experimenting with modern large language model components.

## Features

The following components have been implemented:

- **BPE Tokenizer**: Byte Pair Encoding tokenizer for efficient text preprocessing.
- **Llama Model Architecture**:
  - Rotary Position Embeddings (RoPE)
  - Causal Masked Multi-Head Self-Attention
  - Key-Value (KV) Cache for efficient inference
  - SwiGLU activation function
  - RMSNorm normalization
  - Residual connections
  - Pre-Normalization (LayerNorm before sublayers)

<div align="center">

<b>Model Hyperparameters</b>

| Hyperparameter         | Value         |
|-----------------------|--------------|
| Number of Layers      | 32           |
| Hidden Size           | 4096         |
| Number of Attention Heads | 32       |
| Vocabulary Size       | 32000        |
| Intermediate Size     | 11008        |
| Max Sequence Length   | 2048         |
| Dropout               | 0.0          |
| RMSNorm Epsilon       | 1e-5         |

</div>

## Roadmap

- [x] BPE Tokenizer
- [x] Llama Model Architecture (RoPE, Causal Masked Attention, KV Cache, SwiGLU, RMSNorm, Residual, Pre-Norm)
- [ ] Training functionality (coming soon)

## References

- [Stanford CS336: Large Language Models](https://web.stanford.edu/class/cs336/)
- [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
