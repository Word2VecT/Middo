<p align="center">
<h1 align="center">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2508.21589"><img src="https://img.shields.io/badge/📄-Paper-red"></a>
    <a href="https://github.com/Word2VecT/Middo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Word2VecT/Middo"></a>
    <a href="https://huggingface.co/collections/QizhiPei/mathfusion-67d92b8e505635db1baf20bb"><img src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
</p>

🎉🎉: Middo is accepted EMNLP 2025 (main).

## 🎯 Quick Start
Install the dependencies

### 💻 Test System Information

- System: CentOS Linux 7 (Core), no root permission
- Conda: Miniconda 25.7.0
- GNU C Library: ldd (GNU libc) 2.17
- CUDA: release 12.4, V12.4.131

```bash
# Install LLaMA-Factory for training
cd LLaMA-Factory
conda create -n lf python=3.10 -y
conda activate lf
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyarrow==20.0.0 av==14.4.0 soxr==0.5.0.post1 Cython scikit-build-core setuptools_scm
pip install -e ".[torch,metrics]" --no-build-isolation
pip install vllm
# Install Liger-Kernel and FlashAttention for acclerating
pip install liger-kernel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install OpenCompass for evaluation
cd ../opencompass
conda deactivate
conda create -n oc python=3.10 -y
conda activate oc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyarrow==20.0.0
pip install -e "."
pip install vllm

# Install DataDreamer for data synthetic
cd ../datadreamer
conda deactivate
conda create -n dd python=3.11 -y
conda activate dd
pip install -r requirements.txt
pip install datadreamer.dev
```