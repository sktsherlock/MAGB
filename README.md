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