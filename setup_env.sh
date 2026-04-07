#!/bin/bash
set -e

# Setup the conda environment for LLM-Based Clustering (CLUSPRO)
echo "Creating conda environment 'llm_cluspro'..."
conda create -n llm_cluspro python=3.10 -y

# Initialize conda in script
eval "$(conda shell.bash hook)"
conda activate llm_cluspro

echo "Installing dependencies..."
# Install PyTorch with CUDA 11.8 (compatible with RTX 3080 and vLLM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM for high-throughput LLM inference and other tools
pip install vllm transformers pandas numpy scipy tqdm ftfy regex

# Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

echo "Environment 'llm_cluspro' is ready."
echo "To run the clustering script, use:"
echo "  conda activate llm_cluspro"
echo "  python llm_clustering.py --dataset mit-states --num_clusters 10 --model meta-llama/Meta-Llama-3-8B-Instruct --tp_size 8"
echo "To run the training script, use:"
echo "  python train_llm_cluspro.py --dataset mit-states --llm_clusters_path llm_clusters.json"
