# Depth Any Video with Scalable Synthetic Data

**Depth Any Video** introduces a **scalable synthetic data** pipeline, capturing **40,000** video clips from diverse games, and leverages powerful priors of generative **video diffusion models** to advance video depth estimation. By incorporating rotary position encoding, flow matching, and a mixed-duration training strategy, it robustly handles **varying video lengths and frame rates**. Additionally, a novel depth interpolation method enables **high-resolution depth inference**, achieving superior spatial accuracy and temporal consistency over previous models.

This repository is the official implementation of the paper:
<div align='center'>

[**Depth Any Video with Scalable Synthetic Data**](http://arxiv.org/abs/2410.10815)

[*Honghui Yang**](https://hhyangcs.github.io/),
[*Di Huang**](https://dihuang.me/),
[*Wei Yin*](https://scholar.google.com/citations?user=ZIf_rtcAAAAJ),
[*Chunhua Shen*](https://scholar.google.com/citations?user=Ljk2BvIAAAAJ),
[*Haifeng Liu*](https://scholar.google.com/citations?user=oW108fUAAAAJ),
[*Xiaofei He*](https://scholar.google.com/citations?user=QLLFowsAAAAJ),
[*Binbin Lin+*](https://scholar.google.com/citations?user=Zmvq4KYAAAAJ),
[*Wanli Ouyang*](https://scholar.google.com/citations?user=pw_0Z_UAAAAJ),
[*Tong He+*](https://scholar.google.com/citations?user=kWADCMUAAAAJ)

<a href='https://arxiv.org/abs/2410.10815'><img src='https://img.shields.io/badge/arXiv-2410.10815-b31b1b.svg'></a> &nbsp;
<a href='https://depthanyvideo.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://huggingface.co/spaces/hhyangcs/depth-any-video'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;
 
</div>

![teaser](assets/teaser.png)

## News

[2024-10-20] The Replicate Demo and API is added here [![Replicate](https://replicate.com/chenxwh/depth-any-video/badge)](https://replicate.com/chenxwh/depth-any-video).

[2024-10-20] The Hugging Face online demo is live [here](https://huggingface.co/spaces/hhyangcs/depth-any-video).

[2024-10-15] The arXiv submission is available [here](https://arxiv.org/abs/2410.10815).

## Installation

Setting up the environment with conda. With support for the app.

```bash
git clone https://github.com/Nightmare-n/DepthAnyVideo
cd DepthAnyVideo

# create env using conda
conda create -n dav python==3.10
conda activate dav
pip install -r requirements.txt
pip install gradio
```

## Inference
- To run inference on an image, use the following command:
```bash
python run_infer.py --data_path ./demos/arch_2.jpg --output_dir ./outputs/ --max_resolution 2048
```

- To run inference on a video, use the following command:
```bash
python run_infer.py --data_path ./demos/wooly_mammoth.mp4 --output_dir ./outputs/ --max_resolution 960
```

## Citation

If you find our work useful, please cite:

```bibtex
@article{yang2024depthanyvideo,
  author    = {Honghui Yang and Di Huang and Wei Yin and Chunhua Shen and Haifeng Liu and Xiaofei He and Binbin Lin and Wanli Ouyang and Tong He},
  title     = {Depth Any Video with Scalable Synthetic Data},
  journal   = {arXiv preprint arXiv:2410.10815},
  year      = {2024}
}
```
