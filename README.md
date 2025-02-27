# 🚀 Multimodal Attributed Graph Benchmark  ![](https://img.shields.io/badge/license-CCY4.0-blue)


<p>
    <img src="Figure/Logo.jpg" width="200" align="left" style="margin-right: 20px;"/>
</p>

<p>
    💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-vl/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-VL/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2502.13923">Paper</a>&nbsp&nbsp
</p>

[//]: # (# MAGB  ![]&#40;https://img.shields.io/badge/license-CCY4.0-blue&#41;)
 MAGB is a project to share the public multimodal attributed graph &#40;MAG&#41; datasets and benchmark the performance of the different baseline methods.

 MAGB is a project to share the public multimodal attributed graph &#40;MAG&#41; datasets and benchmark the performance of the different baseline methods.
 

(We welcome more to share datasets that are valuable for MAGs research.)

 MAGB is a project to share the public multimodal attributed graph &#40;MAG&#41; datasets and benchmark the performance of the different baseline methods.

[//]: # ()
[//]: # (## Datasets 🔔)

[//]: # (We collect and construct 5 MAG datasets from Amazon and Reddit.)

[//]: # ()

[//]: # (A comprehensive benchmark for evaluating multimodal attributed graphs.  )

[//]: # (This repository provides datasets, feature extraction methods, evaluation metrics, and baseline models for research in multimodal attributed graphs.)

---

## 📖 Table of Contents  
- [📖 Introduction](#-introduction)  
- [💻 Installation](#-installation)  
- [⚙️ Configuration](#-configuration)  
- [🚀 Usage](#-usage)  
- [📊 Results](#-results)  
- [🤝 Contributing](#-contributing)  
- [❓ FAQ](#-faq)  

---

## 📖 Introduction  
Multimodal attributed graphs (MAGs) incorporate multiple data types (e.g., text, images, numerical features) into graph structures, enabling more powerful learning and inference capabilities.  
This benchmark provides:  
✅ **Standardized datasets** with multimodal attributes.  
✅ **Feature extraction pipelines** for different modalities.  
✅ **Evaluation metrics** to compare different models.  
✅ **Baselines and benchmarks** to accelerate research.  

---

## 💻 Installation  
Ensure you have the required dependencies installed before running the benchmark.  

```bash
# Clone the repository
git clone https://github.com/your_repo/MAG-Benchmark.git
cd MAG-Benchmark

# Install dependencies
pip install -r requirements.txt


[//]: # (Now you can go to the 'Files and version' in [MAG]&#40;https://huggingface.co/Sherirto/MAG&#41; to find the datasets we upload!)

[//]: # (You can use the node initial feature we created, and you also can extract the node feature from our code. )

[//]: # (For a more detailed and clear process, please [clik there.😎]&#40;FeatureExtractor/README.md&#41;)
[//]: # (In each dataset folder, you can find the **csv** file &#40;which save the text attribute of the dataset&#41;, **pt** file &#40;which represent the dgl graph file&#41;, and the **Feature** folder &#40;which save the text embedding we extract from the PLM&#41;.)

## Todo