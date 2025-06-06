import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import dgl
import torch as th
from matplotlib.colors import LinearSegmentedColormap
import os
import argparse
from huggingface_hub import upload_file
import random
from collections import defaultdict


def split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    if fewshots is not None:
        train_ids = []

        unique_labels = np.unique(labels)  # 获取唯一的类别标签
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]  # 获取属于当前类别的样本索引
            np.random.shuffle(label_indices)  # 对当前类别的样本索引进行随机排序

            fewshot_indices = label_indices[:fewshots]  # 选择指定数量的few-shot样本
            train_ids.extend(fewshot_indices)

        remaining_indices = np.setdiff1d(indices, train_ids)  # 获取剩余的样本索引
        np.random.shuffle(remaining_indices)  # 对剩余样本索引进行随机排序

        val_size = int(len(remaining_indices) * val_ratio)  # 计算验证集大小

        val_ids = remaining_indices[:val_size]  # 划分验证集
        test_ids = remaining_indices[val_size:]  # 划分测试集

    else:

        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_ids = indices[:train_size]
        val_ids = indices[train_size:train_size + val_size]
        test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


def load_data(graph_path, train_ratio=0.6, val_ratio=0.2, fewshots=None,):

    # load the graph from local path
    graph = dgl.load_graphs(graph_path)[0][0]
    labels = graph.ndata['label']
    train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio, labels, fewshots=fewshots)
    train_idx = th.tensor(train_idx)
    val_idx = th.tensor(val_idx)
    test_idx = th.tensor(test_idx)

    return graph, labels, train_idx, val_idx, test_idx


argparser = argparse.ArgumentParser(
    "TSNE Config",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
argparser.add_argument(
    "--graph_path", type=str, default='/home/aiscuser/ATG/Data/Reddit/RedditGraph.pt', help="The datasets to be implemented."
)
argparser.add_argument(
    "--save_path", type=str, default='./',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--sample", type=int, default=300,
    help="The sample size of test idx."
)
argparser.add_argument(
    "--dataname", type=str, default='Reddit',
    help="The datasets name."
)
argparser.add_argument(
    "--feat1", type=str,
    default='/home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_LLAMA8B_CLIP.npy',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--feat2", type=str,
    default='/home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_Llama-3.2-11B-Vision-Instruct_tv.npy',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--label1", type=str, default='LLaMA+CLIP', help="The datasets to be implemented."
)
argparser.add_argument(
    "--label2", type=str, default='LLaMA3.2-11B-Vision', help="The datasets to be implemented."
)
args = argparser.parse_args()

graph, labels, train_idx, val_idx, test_idx = load_data(graph_path=args.graph_path)
# 根据test_idx 进行采样
sample = min(args.sample, len(test_idx))  # 确保不超过列表长度
sample = max(sample, 0)  # 确保是正数
test_idx = test_idx[:sample]


def visualize(feat1, feat2, path, label, sample_size=1000, label1='Reddit_LLAMA8B_CLIP', label2='Reddit_Llama-3.2-11B', dataname=None):
    # 对 PLM_feat 进行采样和获取标签
    # feat1_sample = feat1[:sample_size]
    # print(feat1_sample)
    # 创建自定义颜色渐变
    colors = ['#0081a7', '#00afb9', '#fdfcdc', '#fed9b7', '#f07167']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)


    feat1_test = feat1[test_idx]
    print(feat1_test.shape)
    # 对 LLM_feat 进行采样和获取标签
    feat2_test = feat2[test_idx]
    label_list = label[test_idx]

    # 对 feat1 进行 TSNE 降维
    tsne_feat1 = TSNE(n_components=2).fit_transform(feat1_test)

    # 对 feat2 进行 TSNE 降维
    tsne_feat2 = TSNE(n_components=2).fit_transform(feat2_test)

    # 绘制 t-SNE 可视化结果并保存
    # plt.scatter(tsne_feat1[:, 0], tsne_feat1[:, 1], c=label_list, marker='*', label=label1, cmap='viridis')
    scatter1 = plt.scatter(tsne_feat1[:, 0], tsne_feat1[:, 1], c=label_list, marker='*', label=label1, cmap=custom_cmap)

    # 设置横纵坐标刻度大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.title(f'T-SNE for {label1} on {dataname}')
    plt.legend(fontsize='large')
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Classes', fontsize=14)  # 放大 colorbar 标签
    cbar.ax.tick_params(labelsize=14)  # colorbar 刻度字体加大
    # plt.colorbar(scatter1, label='Classes')


    save_path_feat1 = os.path.join(path, f'{label1}_tsne_BR.svg')
    plt.savefig(save_path_feat1, format='svg', dpi=600, bbox_inches='tight')
    plt.close()

    scatter2 = plt.scatter(tsne_feat2[:, 0], tsne_feat2[:, 1], c=label_list, marker='*', label=label2, cmap=custom_cmap)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.scatter(tsne_feat2[:, 0], tsne_feat2[:, 1], c=label_list, marker='*', label=label2, cmap='viridis')
    # plt.title(f'T-SNE Visualization for {label2}')
    plt.legend(fontsize='large')
    cbar = plt.colorbar(scatter2)
    cbar.set_label('Classes', fontsize=14)  # 放大 colorbar 标签
    cbar.ax.tick_params(labelsize=14)  # colorbar 刻度字体加大
    save_path_feat2 = os.path.join(path, f'{label2}_tsne_BR.svg')
    plt.savefig(save_path_feat2, format='svg', dpi=600, bbox_inches='tight')
    plt.close()

    # 绘制 t-SNE 可视化结果
    plt.scatter(tsne_feat1[:, 0], tsne_feat1[:, 1], c=label_list, marker='*', label=label1, cmap='viridis', s=50)
    plt.scatter(tsne_feat2[:, 0], tsne_feat2[:, 1], c=label_list, marker='o', label=label2, cmap='coolwarm', s=25)

    # plt.scatter(tsne_result[:sample_size, 0], tsne_result[:sample_size, 1], cmap='viridis', label=label1)
    # plt.scatter(tsne_result[sample_size:, 0], tsne_result[sample_size:, 1], cmap='viridis', label=label2)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title(f'T-SNE Visualization between {label1} and {label2}')
    plt.legend()
    save_path = os.path.join(path, f'{label1}_{label2}_combined_tsne.pdf')
    plt.savefig(save_path)
    plt.show()

    print(label_list.shape)

    # 计算轮廓系数
    from sklearn.metrics import silhouette_score
    silhouette1 = silhouette_score(tsne_feat1, label_list)
    silhouette2 = silhouette_score(tsne_feat2, label_list)
    print(f"Silhouette Coefficient for {label1}: {silhouette1:.2f}")
    print(f"Silhouette Coefficient for {label2}: {silhouette2:.2f}")

    # 计算 Calinski-Harabasz Index
    ch1 = calinski_harabasz_score(tsne_feat1, label_list)
    ch2 = calinski_harabasz_score(tsne_feat2, label_list)
    print(f"Calinski-Harabasz Index for {label1}: {ch1:.2f}")
    print(f"Calinski-Harabasz Index for {label2}: {ch2:.2f}")

    # 计算 Davies-Bouldin Index
    db1 = davies_bouldin_score(tsne_feat1, label_list)
    db2 = davies_bouldin_score(tsne_feat2, label_list)
    print(f"Davies-Bouldin Index for {label1}: {db1:.2f}")
    print(f"Davies-Bouldin Index for {label2}: {db2:.2f}")



Feature1 = th.from_numpy(np.load(args.feat1).astype(np.float32))

Feature2 = th.from_numpy(np.load(args.feat2).astype(np.float32))



visualize(Feature1, Feature2, args.save_path, labels, label1=args.label1, label2=args.label2, dataname=args.dataname)
print('Finished TSNE')


upload_file(path_or_fileobj=f'{args.label1}_tsne_BR.svg', path_in_repo=f'{args.dataname}/TSNE/{args.label1}_tsne.svg', repo_id="Sherirto/MAG")
upload_file(path_or_fileobj=f'{args.label2}_tsne_BR.svg', path_in_repo=f'{args.dataname}/TSNE/{args.label2}_tsne.svg', repo_id="Sherirto/MAG")



# python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_Qwen2-VL-7B-Instruct_tv.npy --feat2 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_openai_clip-vit-large-patch14.npy  --label1 Qwen2-VL-7B --label2 CLIP