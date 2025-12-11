# NeMo RL DPO Training & NVIDIA NIM Deployment

This notebook demonstrates an end-to-end workflow for training a language model using Direct Preference Optimization (DPO) with NVIDIA NeMo RL, and deploying it with NVIDIA Inference Microservices (NIM).

## Overview

The workflow covers:
1. **Environment Setup** - Configure caches and storage paths
2. **DPO Training** - Fine-tune Llama 3.2 1B using preference optimization
3. **Model Conversion** - Convert to HuggingFace and SafeTensors formats
4. **NIM Deployment** - Deploy with NVIDIA's optimized inference stack
5. **API Testing** - Validate the deployed model via OpenAI-compatible endpoints

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Minimum 80GB GPU memory recommended
- Sufficient disk space for model checkpoints (~100GB)

### Software Requirements
- Docker with NVIDIA Container Toolkit
- Access to NVIDIA NGC (GPU Cloud)
- Python 3.8+

### API Keys & Tokens
You'll need the following credentials:

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| NVIDIA NGC | `NGC_API_KEY` | Pull NVIDIA containers |
| HuggingFace | HF Token | Download gated models (Llama) |
| Weights & Biases | `WANDB_API_KEY` | Experiment tracking (optional) |

## Quick Start

1. **Set your NGC API key:**
   ```bash
   export NGC_API_KEY="your-ngc-api-key"
   ```

2. **Get a HuggingFace token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a token with read access
   - Accept Llama model license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

3. **Run the notebook:**
   ```bash
   jupyter notebook deploy_nemo-rl.ipynb
   ```

## Notebook Structure

### Phase 1: Environment Setup 
| Step | Description |
|------|-------------|
| NGC Login | Authenticate with NVIDIA container registry |
| Cache Config | Set up ephemeral storage for NIM, Docker, pip, HuggingFace |
| Container Launch | Start NeMo RL v0.4.0 container with GPU support |

### Phase 2: Training 
| Step | Description |
|------|-------------|
| Repository Setup | Clone NeMo RL repo, configure authentication |
| DPO Training | Run preference optimization on Llama 3.2 1B |

### Phase 3: Model Conversion 
| Step | Description |
|------|-------------|
| DCP → HuggingFace | Convert distributed checkpoint to HF format |
| Local Inference Test | Validate converted model works |
| HF → SafeTensors | Convert to secure SafeTensors format |

### Phase 4: Deployment 
| Step | Description |
|------|-------------|
| NIM Configuration | Set up container and model settings |
| Container Launch | Start MultiLLM-NIM with GPU support |
| Health Check | Wait for service readiness |
| API Test | Verify completions endpoint |

## Key Configuration

### DPO Training Parameters
```yaml
cluster.gpus_per_node: 1
dpo.max_num_steps: 10          # Increase for production
policy.model_name: meta-llama/Llama-3.2-1B-Instruct
policy.tokenizer.name: meta-llama/Llama-3.2-1B-Instruct
```

### NIM Deployment Settings
```python
CONTAINER_NAME = "MultiLLM-NIM"
NIM_SERVED_MODEL_NAME = "dpo-llm"
PORT = 8000
```

## Output Artifacts

After running the notebook, you'll have:

```
./results/dpo/step_10/
├── config.yaml           # Training configuration
├── policy/
│   └── weights/          # DCP checkpoint
├── hf/                   # HuggingFace format model
└── hf_st/                # SafeTensors format model
```

## API Usage

Once deployed, the model is accessible via OpenAI-compatible endpoints:

### Completions API
```bash
curl -X POST 'http://localhost:8000/v1/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dpo-llm",
    "prompt": "The sky appears blue because",
    "max_tokens": 64
  }'
```

### Health Check
```bash
curl http://localhost:8000/v1/health/ready
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| NGC login fails | Verify `NGC_API_KEY` is set correctly |
| HuggingFace download fails | Check token has read access, model license accepted |
| Docker permission denied | Run with sudo or add user to docker group |
| GPU not detected | Ensure NVIDIA Container Toolkit is installed |
| NIM health check timeout | Increase wait time, check GPU memory availability |

### Useful Commands
```bash
# Check running containers
docker ps

# View container logs
docker logs nemo-rl
docker logs MultiLLM-NIM

# Check GPU usage
nvidia-smi

# Stop all containers
docker stop nemo-rl MultiLLM-NIM
```

## References

- [NVIDIA NeMo RL Documentation](https://github.com/NVIDIA-NeMo/RL)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [DPO Paper - Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## License

This notebook is provided for educational and demonstration purposes. Please ensure compliance with:
- NVIDIA NGC Terms of Service
- Meta Llama License Agreement
- HuggingFace Terms of Service
