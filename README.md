# MAGB![](https://img.shields.io/badge/license-CC%20BY%204.0-blue)![](https://img.shields.io/github/stars/sktsherlock/MAGB?style=social)![](https://img.shields.io/github/forks/sktsherlock/MAGB?style=social)![](https://img.shields.io/github/languages/top/sktsherlock/MAGB)


<p>
    <img src="Figure/Logo.jpg" width="190" align="left" style="margin-right: 20px;"/>
</p>

<p>
    <b>MAGB: A Comprehensive Benchmark for Multimodal Attributed Graphs</b>
</p>


In many real-world scenarios, graph nodes are associated with multimodal attributes, such as texts and images, resulting in **Multimodal Attributed Graphs (MAGs)**.

MAGB first provide 5 dataset from E-Commerce and Social Networks. And we evaluate two major paradigms: _**GNN-as Predictor**_  and **_VLM-as-Predictor_** . The datasets are publicly available:

<p>
     🤗 <a href="https://huggingface.co/datasets/Sherirto/MAGB">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">FeatureExtract</a>&nbsp&nbsp  | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2410.09132">Paper</a>&nbsp&nbsp
</p>



## 📖 Table of Contents  
- [📖 Introduction](#-introduction)  
- [💻 Installation](#-installation)
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
git clone https://github.com/sktsherlock/MAGB.git
cd MAGB

# Install dependencies
pip install -r requirements.txt
```
## 🚀 Usage

### 1. Download the datasets from [MAGB](https://huggingface.co/datasets/Sherirto/MAGB). 👐

```bash
cd Data/
sudo apt-get update && sudo apt-get install git-lfs && git clone https://huggingface.co/datasets/Sherirto/MAGB .
ls
```
Now, you can see the **Movies**, **Toys**, **Grocery**, **Reddit-S** and **Reddit-M** under the **''Data''** folder. 

<p align="center">
    <img src="Figure/Dataset.jpg" width="900"/>
<p>

Each dataset consists of several parts shown in the image below, including:

- Graph Data (*.pt): Stores the graph structure, including adjacency information and node labels. It can be loaded using DGL.
- Node Textual Metadata (*.csv): Contains node textual descriptions, neighborhood relationships, and category labels.
- Text, Image, and Multimodal Features (TextFeature/, ImageFeature/, MMFeature/): Pre-extracted embeddings from the MAGB paper for different modalities.
- Raw Images (*.tar.gz): A compressed folder containing images named by node IDs. It needs to be extracted before use.

Because of the Reddit-M dataset is too large, you may need to follow the below scripts to unzip the dataset.
```bash
cd MAGB/Data/
cat RedditMImages_parta RedditMImages_partb RedditMImages_partc > RedditMImages.tar.gz
tar -xvzf RedditMImages.tar.gz
```

### 2. GNN-as-Predictor 


