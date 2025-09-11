<p align="center">
<h1 align="center">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2508.21589"><img src="https://img.shields.io/badge/📄-Paper-red"></a>
    <a href="https://github.com/Word2VecT/Middo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Word2VecT/Middo"></a>
    <a href="https://huggingface.co/collections/Word2Li/middo-68c27d3b42f79febf7f6312c"><img src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
</p>

🎉🎉: Middo is accepted EMNLP 2025 (main).

We introduce **Middo**, a self-evolving **M**odel-**i**nformed **d**ynamic **d**ata **o**ptimization framework that uses model-aware data selection and context-preserving data refinement. Middo establishes a closed-loop optimization system:

1. A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - *loss patterns (complexity)*, *embedding cluster dynamics (diversity)*, and *self-alignment scores (quality)*;
2. An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity;
3. This optimization process continuously evolves with model capability through dynamic learning principles.

![Middo](imgs/Middo.png)

Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by $7.15%$ on average while maintaining the original dataset scale.

![Result](imgs/result.png)

We release six optimized datasets and corresponing models fine-tuned on these dataset.

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
pip install pyarrow==20.0.0 av==14.4.0 deepspeed==0.16.9 soxr==0.5.0.post1 Cython scikit-build-core setuptools_scm
pip install -e ".[torch,metrics]" --no-build-isolation
pip install vllm
# Install Liger-Kernel and FlashAttention for acclerating
pip install liger-kernel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install opencompass for evaluation
cd ../opencompass
conda deactivate
conda create -n oc python=3.10 -y
conda activate oc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pyarrow==20.0.0
pip install -e "."
pip install vllm flashinfer-python==0.2.8

# Install DataDreamer for data synthetic
cd ../datadreamer
conda deactivate
conda create -n dd python=3.10 -y
conda activate dd
pip install -r requirements.txt
pip install datadreamer.dev
```

## 📚 Data

View data from huggingface [Word2Li/MiddOptimized](https://huggingface.co/datasets/Word2Li/MiddOptimized) then convert each split to `.json` or `.jsonl` file. Edit [`dataset_info.jsonl`](LLaMA-Factory/data/dataset_info.json) file and training yaml script (e.g. `train_llama.yaml`) file according to LLaMA-Factory [Data Preparation](https://github.com/hiyouga/LLaMA-Factory#data-preparation).

## 🤖 Training

Our training codes depend on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
cd LLaMA-Factory
conda activate lf
llamafactory-cli train train_llama.yaml
```

## 📊 Evaluation

Our training codes depend on [OpenCompass](https://github.com/hiyouga/LLaMA-Factory). You need to first download the model from HuggingFace, or SFT the model on your own.

```bash
cd opencompass
conda activate oc

```

## 🙏 Acknowledgements

Many thanks to

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main)
- [OpenCompass](https://github.com/open-compass/opencompass)
- [DataDreamer](https://github.com/datadreamer-dev/DataDreamer)

## Citation

If you find our code, model, or data are useful, please kindly cite our [paper](https://arxiv.org/abs/2508.21589):

```bibtex
@misc{tang2025middomodelinformeddynamicdata,
      title={Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning}, 
      author={Zinan Tang and Xin Gao and Qizhi Pei and Zhuoshi Pan and Mengzhang Cai and Jiang Wu and Conghui He and Lijun Wu},
      year={2025},
      eprint={2508.21589},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.21589}, 
}
```
