[中文阅读](README_zh_cn.md)

<p align="center">
  <img src="assets/teaser.png">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HunyuanWorld-1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2507.21809 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>

<p align="center">
  "To see a World in a Grain of Sand, and a Heaven in a Wild Flower"
</p>

https://github.com/user-attachments/assets/747b3e41-df9c-4cd2-b1d1-c0dce63f63ef

## 🔥 News
- October 22, 2025: 🤗 We release [HunyuanWorld-1.1 (WorldMirror)](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), supporting 3D world creation from videos or multi-view images!
- October 16, 2025: 🤗 We recently propose  [FlashWorld](https://github.com/imlixinyang/FlashWorld), enabling 3DGS world generation in 5~10 seconds on a single GPU!
- September 2, 2025: 🤗 We release our RGB-D Video Diffusion model  [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager/), which supports 3D-consistency world exploration and fast 3D reconstruction!
- August 15, 2025: 🤗 We release the quantization version of HunyuanWorld-1.0 (HunyuanWorld-1.0-lite), which now supports running on Consumer-grade GPUs such as 4090!
- July 26, 2025: 👋 We present the [technical report](https://arxiv.org/abs/2507.21809) of HunyuanWorld-1.0, please check out the details and spark some discussion!
- July 26, 2025: 🤗 We release the first open-source, simulation-capable, immersive 3D world generation model, HunyuanWorld-1.0!

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 

## ☯️ **HunyuanWorld 1.0**

### Abstract
Creating immersive and playable 3D worlds from texts or images remains a fundamental challenge in computer vision and graphics. Existing world generation approaches typically fall into two categories: video-based methods that offer rich diversity but lack 3D consistency and rendering efficiency, and 3D-based methods that provide geometric consistency but struggle with limited training data and memory-inefficient representations. To address these limitations, we present HunyuanWorld 1.0, a novel framework that combines the best of both sides for generating immersive, explorable, and interactive 3D worlds from text and image conditions. Our approach features three key advantages: 1) 360° immersive experiences via panoramic world proxies; 2) mesh export capabilities for seamless compatibility with existing computer graphics pipelines; 3) disentangled object representations for augmented interactivity. The core of our framework is a semantically layered 3D mesh representation that leverages panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction, enabling the generation of diverse 3D worlds. Extensive experiments demonstrate that our method achieves state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds while enabling versatile applications in virtual reality, physical simulation, game development, and interactive content creation.

<p align="center">
  <img src="assets/application.png">
</p>

### Architecture
Tencent HunyuanWorld-1.0's generation architecture integrates panoramic proxy generation, semantic layering, and hierarchical 3D reconstruction to achieve   high-quality scene-scale 360° 3D world generation, supporting both text and image inputs.

<p align="left">
  <img src="assets/arch.jpg">
</p>

### Performance

We have evaluated HunyuanWorld 1.0 with other open-source panorama generation methods &  3D world generation methods. The numerical results indicate that HunyuanWorld 1.0 surpasses baselines in  visual quality and geometric consistency.

Text-to-panorama generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 69.5                  | 7.5                | 1.8                 | 20.9               |
| MVDiffusion      | 47.9                  | 7.1                | 2.4                 | 21.5               |
| PanFusion        | 56.6                  | 7.6                | 2.2                 | 21.0               |
| LayerPano3D      | 49.6                  | 6.5                | 3.7                 | 21.5               |
| HunyuanWorld 1.0 | **40.8**              | **5.8**            | **4.4**             | **24.3**           |

Image-to-panorama generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Diffusion360     | 71.4                  | 7.8                | 1.9                 | 73.9               |
| MVDiffusion      | 47.7                  | 7.0                | 2.7                 | 80.8               |
| HunyuanWorld 1.0 | **45.2**              | **5.8**            | **4.3**             | **85.1**           |

Text-to-world generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-T(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| Director3D       | 49.8                  | 7.5                | 3.2                 | 23.5               |
| LayerPano3D      | 35.3                  | 4.8                | 3.9                 | 22.0               |
| HunyuanWorld 1.0 | **34.6**              | **4.3**            | **4.2**             | **24.0**           |

Image-to-world generation:

| Method           | BRISQUE(⬇) | NIQE(⬇) | Q-Align(⬆) | CLIP-I(⬆) |
| ---------------- | --------------------- | ------------------ | ------------------- | ------------------ |
| WonderJourney    | 51.8                  | 7.3                | 3.2                 | 81.5               |
| DimensionX       | 45.2                  | 6.3                | 3.5                 | 83.3               |
| HunyuanWorld 1.0 | **36.2**              | **4.6**            | **3.9**             | **84.5**           |

### Visual Results

360 ° immersive and explorable 3D worlds generated by HunyuanWorld 1.0:

<p align="left">
  <img src="assets/panorama1.gif">
</p>

 <p align="left">
  <img src="assets/panorama2.gif">
</p> 

<p align="left">
  <img src="assets/roaming_world.gif">
</p>

## 🎁 Models Zoo
The open-source version of HY World 1.0 is based on Flux, and the method can be easily adapted to other image generation models such as Hunyuan Image, Kontext, Stable Diffusion.

| Model                          | Description                 | Date       | Size  | Huggingface                                                                                        |
|--------------------------------|-----------------------------|------------|-------|----------------------------------------------------------------------------------------------------| 
| HunyuanWorld-PanoDiT-Text      | Text to Panorama Model      | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text)      |
| HunyuanWorld-PanoDiT-Image     | Image to Panorama Model     | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image)     |
| HunyuanWorld-PanoInpaint-Scene | PanoInpaint Model for scene | 2025-07-26 | 478MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene) |
| HunyuanWorld-PanoInpaint-Sky   | PanoInpaint Model for sky   | 2025-07-26 | 120MB | [Download](https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky)   |

## 🤗 Get Started with HunyuanWorld 1.0

You may follow the next steps to use Hunyuan3D World 1.0 via:

### Environment construction
We test our model with Python 3.10 and PyTorch 2.5.0+cu124.

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
cd HunyuanWorld-1.0
conda env create -f docker/HunyuanWorld.yaml

# real-esrgan install
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr-fixed
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# zim anything install & download ckpt from ZIM project page
cd ..
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

# TO export draco format, you should install draco first
cd ../..
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install

# login your own hugging face account
cd ../..
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Code Usage
For Image to World generation, you can use the following code:
```python
# First, generate a Panorama image with An Image.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2/panorama.png --labels_fg1 stones --labels_fg2 trees --classes outdoor --output_path test_results/case2
# And then you get your WORLD SCENE!!
```

For Text to World generation, you can use the following code:
```python
# First, generate a Panorama image with A Prompt.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7
# Second, using this Panorama image, to create a World Scene with HunyuanWorld 1.0
# You can indicate the foreground objects labels you want to layer out by using params labels_fg1 & labels_fg2
# such as --labels_fg1 sculptures flowers --labels_fg2 tree mountains
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7/panorama.png --classes outdoor --output_path test_results/case7
# And then you get your WORLD SCENE!!
```

### Quantization & Cache Usage
For Image to World generation, you can use the following code with quantization/cache:
```python
# Step 1:
# To optimize memory usage and speed up inference, quantization is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "" --image_path examples/case2/input.png --output_path test_results/case2_cache --cache
# Step 2:
# To optimize memory usage and speed up inference, quantization is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_quant/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case2_cache/panorama.png --labels_fg1 stones --labels_fg2 trees  --classes outdoor --output_path test_results/case2_cache --cache
```

For Text to World generation, you can use the following code with quantization/cache:
```python
# Step 1:
# To optimize memory usage and speed up inference, quantization is a practical solution.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
python3 demo_panogen.py --prompt "At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary" --output_path test_results/case7_cache --cache
# Step 2:
# To optimize memory usage and speed up inference, quantization is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_quant/panorama.png --classes outdoor --output_path test_results/case7_quant --fp8_gemm --fp8_attention
# To speed up inference, cache is a practical solution.
CUDA_VISIBLE_DEVICES=0 python3 demo_scenegen.py --image_path test_results/case7_cache/panorama.png --classes outdoor --output_path test_results/case7_cache --cache
```

### Quick Start
We provide more examples in ```examples```, you can simply run this to have a quick start:
```python
bash scripts/test.sh
```

###  3D World Viewer

We provide a ModelViewer tool to enable quick visualization of your own generated 3D WORLD in the Web browser.

Just open ```modelviewer.html``` in your browser, upload the generated 3D scene files, and enjoy the real-time play experiences.

<p align="left">
  <img src="assets/quick_look.gif">
</p>

Due to hardware limitations, certain scenes may fail to load.

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [x] Technical Report
- [x] Lite Version
- [x] Voyager (RGBD Video Diffusion)

## 🔗 BibTeX
```
@article{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Team HunyuanWorld},
    year={2025},
    journal={arXiv preprint}
}
```


## Contact
Please send emails to tengfeiwang12@gmail.com if there is any question

## Acknowledgements
We would like to thank the contributors to the [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [ZIM](https://github.com/naver-ai/ZIM), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [MoGe](https://github.com/microsoft/moge), [Worldsheet](https://worldsheet.github.io/), [WorldGen](https://github.com/ZiYang-xie/WorldGen) repositories, for their open research.
