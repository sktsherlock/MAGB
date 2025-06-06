from transformers import pipeline, AutoTokenizer, set_seed
import torch
from datasets import load_dataset
import os
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import pandas as pd
import deepspeed
import argparse
import warnings
warnings.filterwarnings("ignore")
# Casual LLM for extracting the keywords from the raw text file
# facebook/opt-66b; mosaicml/mpt-30b-instruct; mosaicml/mpt-30b ; meta-llama/Llama-2-7b-hf;  meta-llama/Llama-2-70b-hf  ; tiiuae/falcon-40b-instruct ;
# Summnarization: facebook/bart-large-cnn;

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Path to the config file')
parser.add_argument('--num', type=int, default=1, help='The prompt numbers')
args = parser.parse_args()

if args.config_name == 'facebook/opt-30b':
    from Config import OPT_30b as config
elif args.config_name == 'mosaicml/mpt-30b':
    from Config import MPT_30b as config
elif args.config_name == 'meta-llama/Llama-2-70b-hf':
    from Config import LLAMA2_70b as config
elif args.config_name == 'meta-llama/Llama-2-7b-hf':
    from Config import LLAMA2_7b as config
elif args.config_name == 'tiiuae/falcon-40b-instruct':
    from Config import FLACON_40b as config
elif args.config_name == 'mosaicml/mpt-7b':
    from Config import MPT_7b as config
elif args.config_name == 'facebook/opt-6.7b':
    from Config import OPT_6b as config
else:
    raise ValueError

# 加载token
access_token = os.getenv('ACCESS_TOKEN', None)

world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

# 解析命令行参数
model_name = config.model_name

tokenizer_name = config.tokenizer_name

Text_path = config.path

if not os.path.exists(Text_path):
    os.makedirs(Text_path)

output_file = Text_path + 'Keywords_' + model_name.split('/')[-1].replace("-", "_") + f"_{args.num}_shot.csv"
print(output_file)

# Set seed before initializing model.
set_seed(config.seed)

# 加载数据集
# Loading a dataset from your local files. CSV training and evaluation files are needed.
csv_file = config.csv_file
root_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(root_dir.rstrip('/'))

data_files = os.path.join(base_dir, csv_file)

dataset = load_dataset(
    "csv",
    data_files=data_files,
)

# 加载模型和分词器
if tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=access_token)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)

# model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True,
#                                                   token=access_token)

pipe = pipeline(
    config.task_name,
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    token=access_token,
    trust_remote_code=True,
    device_map="auto",
)

if config.speed:
    pipe.model = deepspeed.init_inference(pipe.model,
                                          max_out_tokens=4096,
                                          tensor_parallel={'tp_size': world_size},
                                          dtype=torch.half,
                                          replace_with_kernel_inject=True)
pipe.model.eval()

Demonstration = """The mechanistic basis of data dependence and abrupt learning in an in-context classification task. Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with traditional in-weights learning of query-output relationships. What aspects of the training data distribution and architecture favor in-context vs in-weights learning? Recent work has shown that specific distributional properties inherent in language, such as burstiness, large dictionaries and skewed rank-frequency distributions, control the trade-off or simultaneous appearance of these two forms of learning. We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. By identifying progress measures that precede in-context learning and targeted experiments, we construct a two-parameter model of an induction head which emulates the full data distributional dependencies displayed by the attention-based network. A phenomenological model of induction head formation traces its abrupt emergence to the sequential learning of three nested logits enabled by an intrinsic curriculum. We propose that the sharp transitions in attention-based networks arise due to a specific chain of multi-layer operations necessary to achieve ICL, which is implemented by nested nonlinearities sequentially learned during training.
Summarise the keywords from the above text.
Keywords:
in-context learning, mechanistic interpretability, language models, induction heads."""

Three_Demonstration = """
The mechanistic basis of data dependence and abrupt learning in an in-context classification task. Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with traditional in-weights learning of query-output relationships. What aspects of the training data distribution and architecture favor in-context vs in-weights learning? Recent work has shown that specific distributional properties inherent in language, such as burstiness, large dictionaries and skewed rank-frequency distributions, control the trade-off or simultaneous appearance of these two forms of learning. We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. By identifying progress measures that precede in-context learning and targeted experiments, we construct a two-parameter model of an induction head which emulates the full data distributional dependencies displayed by the attention-based network. A phenomenological model of induction head formation traces its abrupt emergence to the sequential learning of three nested logits enabled by an intrinsic curriculum. We propose that the sharp transitions in attention-based networks arise due to a specific chain of multi-layer operations necessary to achieve ICL, which is implemented by nested nonlinearities sequentially learned during training.
Summarise the keywords from the above text.
Keywords:
in-context learning, mechanistic interpretability, language models, induction heads.

LRM: Large Reconstruction Model for Single Image to 3D. We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs including real-world in-the-wild captures and images from generative models. Video demos and interactable 3D meshes can be found on this anonymous website: https://scalei3d.github.io/LRM.
Summarise the keywords from the above text.
Keywords:
3D Reconstruction, Large-Scale, Transformers.

Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis. One-shot 3D talking portrait generation aims to reconstruct a 3D avatar from an unseen image, and then animate it with a reference video or audio to generate a talking portrait video. The existing methods fail to simultaneously achieve the goals of accurate 3D avatar reconstruction and stable talking face animation. Besides, while the existing works mainly focus on synthesizing the head part, it is also vital to generate natural torso and background segments to obtain a realistic talking portrait video. To address these limitations, we present Real3D-Potrait, a framework that (1) improves the one-shot 3D reconstruction power with a large image-to-plane model that distills 3D prior knowledge from a 3D face generative model; (2) facilitates accurate motion-conditioned animation with an efficient motion adapter; (3) synthesizes realistic video with natural torso movement and switchable background using a head-torso-background super-resolution model; and (4) supports one-shot audio-driven talking face generation with a generalizable audio-to-motion model. Extensive experiments show that Real3D-Portrait generalizes well to unseen identities and generates more realistic talking portrait videos compared to previous methods. Video samples are available at https://real3dportrait.github.io.
Summarise the keywords from the above text.
Keywords:
Neural Radiance Field, One-shot Talking Face Generation.
"""

Five_Demonstration = """
The mechanistic basis of data dependence and abrupt learning in an in-context classification task. Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with traditional in-weights learning of query-output relationships. What aspects of the training data distribution and architecture favor in-context vs in-weights learning? Recent work has shown that specific distributional properties inherent in language, such as burstiness, large dictionaries and skewed rank-frequency distributions, control the trade-off or simultaneous appearance of these two forms of learning. We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. By identifying progress measures that precede in-context learning and targeted experiments, we construct a two-parameter model of an induction head which emulates the full data distributional dependencies displayed by the attention-based network. A phenomenological model of induction head formation traces its abrupt emergence to the sequential learning of three nested logits enabled by an intrinsic curriculum. We propose that the sharp transitions in attention-based networks arise due to a specific chain of multi-layer operations necessary to achieve ICL, which is implemented by nested nonlinearities sequentially learned during training.
Summarise the keywords from the above text.
Keywords:
in-context learning, mechanistic interpretability, language models, induction heads.

LRM: Large Reconstruction Model for Single Image to 3D. We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs including real-world in-the-wild captures and images from generative models. Video demos and interactable 3D meshes can be found on this anonymous website: https://scalei3d.github.io/LRM.
Summarise the keywords from the above text.
Keywords:
3D Reconstruction, Large-Scale, Transformers.

Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis. One-shot 3D talking portrait generation aims to reconstruct a 3D avatar from an unseen image, and then animate it with a reference video or audio to generate a talking portrait video. The existing methods fail to simultaneously achieve the goals of accurate 3D avatar reconstruction and stable talking face animation. Besides, while the existing works mainly focus on synthesizing the head part, it is also vital to generate natural torso and background segments to obtain a realistic talking portrait video. To address these limitations, we present Real3D-Potrait, a framework that (1) improves the one-shot 3D reconstruction power with a large image-to-plane model that distills 3D prior knowledge from a 3D face generative model; (2) facilitates accurate motion-conditioned animation with an efficient motion adapter; (3) synthesizes realistic video with natural torso movement and switchable background using a head-torso-background super-resolution model; and (4) supports one-shot audio-driven talking face generation with a generalizable audio-to-motion model. Extensive experiments show that Real3D-Portrait generalizes well to unseen identities and generates more realistic talking portrait videos compared to previous methods. Video samples are available at https://real3dportrait.github.io.
Summarise the keywords from the above text.
Keywords:
Neural Radiance Field, One-shot Talking Face Generation.

Entropic Neural Optimal Transport via Diffusion Processes. We propose a novel neural algorithm for the fundamental problem of computing the entropic optimal transport (EOT) plan between probability distributions which are accessible by samples. Our algorithm is based on the saddle point reformulation of the dynamic version of EOT which is known as the Schrödinger Bridge problem. In contrast to the prior methods for large-scale EOT, our algorithm is end-to-end and consists of a single learning step, has fast inference procedure, and allows handling small values of the entropy regularization coefficient which is of particular importance in some applied problems. Empirically, we show the performance of the method on several large-scale EOT tasks. The code for the ENOT solver can be found at https://github.com/ngushchin/EntropicNeuralOptimalTransport
Summarise the keywords from the above text.
Keywords:
Optimal transport, Schrödinger Bridge, Entropy regularized OT, Neural Networks, Unpaired Learning.

Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models. Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.
Summarise the keywords from the above text.
Keywords:
model editing, transfer learning, neural tangent kernel, vision-language pre-training, deep learning science.
"""

# Summary
Keywords_prompt = """Summarise the keywords from the above text.
Keywords:
"""

Summary_prompt = """Please summarise the above description from a paper on the arxiv CS field.
Summary:
"""


def add_keywords_prompt(example, column_name=config.text_column, num=args.num):
    if num == 5:
        example[f"{column_name}"] = f"{Five_Demonstration}\n{example[f'{column_name}']}\n{Keywords_prompt}"
    elif num == 1:
        example[f"{column_name}"] = f"{Demonstration}\n{example[f'{column_name}']}\n{Keywords_prompt}"
    elif num == 3:
        example[f"{column_name}"] = f"{Three_Demonstration}\n{example[f'{column_name}']}\n{Keywords_prompt}"
    elif num == 0:
        example[f"{column_name}"] = f"{example[f'{column_name}']}\n{Keywords_prompt}"
    else:
        raise ValueError
    return example


def add_summary_prompt(example, column_name='TA'):
    example[f"{column_name}"] = f"{example[f'{column_name}']}\n{Summary_prompt}"
    return example


if config.prompt == 'keywords':
    prompt_dataset = dataset.map(add_keywords_prompt)
else:
    prompt_dataset = dataset.map(add_summary_prompt)

# 打开CSV文件并创建写入器
generated_text_list = []  # 创建一个列表用于存储生成的文本


batch_size = 1000  # 每次生成的批次大小

# Pipe 的方式生成并保存文本
for i, out in enumerate(tqdm(pipe(KeyDataset(prompt_dataset['train'], config.text_column), do_sample=True,
                                  max_new_tokens=config.max_new_tokens, use_cache=True,
                                  repetition_penalty=2.5,
                                  top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
                                  return_full_text=config.return_full_text))):
    generated_text = out[0]['generated_text'] if config.task_name == "text-generation" else out[0]['summary_text']
    generated_text_list.append(generated_text)

    # 每生成一个批次后，保存到CSV文件
    if (i + 1) % batch_size == 0:
        df = pd.DataFrame({'Keywords': generated_text_list})
        df.to_csv(output_file, mode='a', index=False, header=not i)  # 追加到CSV文件中
        generated_text_list = []  # 清空列表以存储下一个批次的生成文本

# 生成完成后，将剩余的文本保存到CSV文件
if generated_text_list:
    df = pd.DataFrame({'Keywords': generated_text_list})
    df.to_csv(output_file, mode='a', index=False, header=False)  # 追加到CSV文件中


print("CSV file has been generated successfully.")

"""
CUDA_VISIBLE_DEVICES=1 python CASUAL.py --csv_file /dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/OGBN_ARXIV.csv --model_name  mosaicml/mpt-30b-instruct --num 0
CUDA_VISIBLE_DEVICES=2 python CASUAL.py --csv_file /dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/OGBN_ARXIV.csv --model_name  mosaicml/mpt-7b --num 0
CUDA_VISIBLE_DEVICES=5 python CASUAL.py --config_name  mosaicml/mpt-7b 
"""
