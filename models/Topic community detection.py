import pandas as pd
import jieba
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import logging
import time
import igraph as ig
import leidenalg as la
import numpy as np
import re
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

stopwords_path = r'D:\Python-work\2750stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stop_words = set(f.read().splitlines())

file_path = r'D:\Python-work\G2021（2021.7.18.0时-8.15.0时）.xlsx'
sheet_name = 'Sheet1'

logging.info("开始读取Excel文件")
start_time = time.time()
data = pd.read_excel(file_path, sheet_name=sheet_name)
logging.info(f"Excel文件读取完成，耗时 {time.time() - start_time:.2f} 秒")

content_column = data['content'].astype(str).fillna('')


def emoji_to_text(text):
    if not isinstance(text, str):
        text = str(text)
    return emoji.demojize(text, delimiters=("", ""))


def remove_urls(text):
    url_pattern = re.compile(r'http\S+|www\S+')
    return url_pattern.sub(r'', text)


def preprocess_text(text):
    text = emoji_to_text(text)
    text = remove_urls(text)
    words = jieba.lcut(text)
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)


logging.info("开始应用预处理函数")
start_time = time.time()
data['processed_content'] = content_column.apply(preprocess_text)
logging.info(f"预处理函数应用完成，耗时 {time.time() - start_time:.2f} 秒")

data['processed_content'].replace('', '空', inplace=True)
data.dropna(subset=['processed_content'], inplace=True)

logging.info(f"预处理后的数据样本:\n{data['processed_content'].head()}")

output_file = '2021年河南暴雨事件text_analysis_results.xlsx'

try:
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Processed_Content', index=False)
        logging.info("预处理结果已保存到 text_analysis_results.xlsx 的 Processed_Content 表")

    if data.shape[0] < 10000:
        tfidf_threshold = 0.001
        df_threshold = 5
        resolution_parameter = 0.8
        alpha = 0.00001
    else:
        tfidf_threshold = 0.001
        df_threshold = 10
        resolution_parameter = 1.0
        alpha = 0.000001

    logging.info("开始计算TF-IDF")
    start_time = time.time()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_content'])
    terms = tfidf_vectorizer.get_feature_names_out()
    logging.info(f"TF-IDF计算完成，耗时 {time.time() - start_time:.2f} 秒")

    logging.info(f"TF-IDF矩阵大小: {tfidf_matrix.shape}")
    logging.info(f"词项示例: {terms[:10]}")

    logging.info("开始计算词频（DF）")
    start_time = time.time()
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(data['processed_content'])
    count_terms = count_vectorizer.get_feature_names_out()
    term_frequencies = count_matrix.sum(axis=0).A1
    logging.info(f"词频计算完成，耗时 {time.time() - start_time:.2f} 秒")

    logging.info(f"词频矩阵大小: {count_matrix.shape}")
    logging.info(f"词频示例: {count_terms[:10]}")

    logging.info("开始筛选关键词")
    start_time = time.time()


    def is_meaningful(word):
        if word.isdigit():
            return False
        num_digits = sum(c.isdigit() for c in word)
        if num_digits / len(word) > 0.5:
            return False
        return True


    keywords = [terms[i] for i in range(len(terms)) if
                tfidf_matrix[:, i].mean() > tfidf_threshold and term_frequencies[i] > df_threshold and is_meaningful(
                    terms[i])]
    logging.info(f"关键词筛选完成，耗时 {time.time() - start_time:.2f} 秒")
    logging.info(f"关键词数量: {len(keywords)}")

    keywords_df = pd.DataFrame(keywords, columns=['keyword'])
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        keywords_df.to_excel(writer, sheet_name='Keywords', index=False)
        logging.info("关键词已保存到 text_analysis_results.xlsx 的 Keywords 表")

    logging.info("开始构建共现矩阵")
    start_time = time.time()
    cooccurrence = defaultdict(int)

    window_size = 8

    for text in data['processed_content']:
        words = text.split()
        for i in range(len(words)):
            if words[i] in keywords:
                for j in range(i + 1, min(i + window_size, len(words))):
                    if words[j] in keywords:
                        cooccurrence[(words[i], words[j])] += 1
                        cooccurrence[(words[j], words[i])] += 1

    logging.info(f"共现矩阵构建完成，耗时 {time.time() - start_time:.2f} 秒")

    logging.info(f"共现关系示例: {list(cooccurrence.items())[:10]}")

    cooccurrence_df = pd.DataFrame(list(cooccurrence.items()), columns=['word_pair', 'frequency'])
    cooccurrence_df[['word1', 'word2']] = pd.DataFrame(cooccurrence_df['word_pair'].tolist(),
                                                       index=cooccurrence_df.index)
    cooccurrence_matrix = cooccurrence_df.pivot(index='word1', columns='word2', values='frequency').fillna(0)

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        cooccurrence_df.to_excel(writer, sheet_name='Cooccurrence_Matrix', index=False)
        logging.info("共现矩阵已保存到 text_analysis_results.xlsx 的 Cooccurrence_Matrix 表")


    def double_stochastic_residualization(matrix):
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
        norm_matrix = matrix / np.outer(row_sums, col_sums)
        return norm_matrix


    def strong_connected_component_extraction(matrix, alpha):
        norm_matrix = double_stochastic_residualization(matrix.values)
        H = nx.Graph()
        nodes = matrix.index.tolist()
        for node in nodes:
            H.add_node(node)
        edges = []
        rows, cols = np.where(norm_matrix > alpha)

        sorted_indices = np.argsort(norm_matrix[rows, cols])[::-1]
        for idx in sorted_indices:
            row = rows[idx]
            col = cols[idx]
            if row != col:
                H.add_edge(nodes[row], nodes[col], weight=norm_matrix[row, col])
        return H


    logging.info("开始双随机矩阵的两阶段主干网络提取")
    start_time = time.time()
    G = strong_connected_component_extraction(cooccurrence_matrix, alpha)
    logging.info(f"主干网络提取完成，耗时 {time.time() - start_time:.2f} 秒")
    logging.info(f"主干网络节点数量: {G.number_of_nodes()}")
    logging.info(f"主干网络边数量: {G.number_of_edges()}")
    logging.info(f"主干网络提取完成，耗时 {time.time() - start_time:.2f} 秒")
    logging.info("开始绘制主干网络图")
    start_time = time.time()


    def plot_backbone_network(G, output_file):
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(15, 15))
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, font_color='black', node_color='skyblue',
                edge_color='gray')
        plt.title('Backbone Network')
        plt.savefig(output_file)
        plt.close()


    plot_backbone_network(G, 'backbone_network.png')
    logging.info(f"主干网络图绘制完成，耗时 {time.time() - start_time:.2f} 秒")
    logging.info("开始使用Leiden算法检测社团")

    start_time = time.time()
    G_igraph = ig.Graph.from_networkx(G)
    partition = la.find_partition(G_igraph, la.CPMVertexPartition, resolution_parameter=resolution_parameter)
    communities = partition.membership
    logging.info(f"Leiden算法社团检测完成，发现 {len(set(communities))} 个社团，耗时 {time.time() - start_time:.2f} 秒")

    node_index_map = {node: idx for idx, node in enumerate(G.nodes())}
    sorted_nodes = sorted(G.nodes(), key=lambda x: node_index_map[x])
    sorted_communities = [communities[node_index_map[node]] for node in sorted_nodes]

    partition_df = pd.DataFrame({'node': sorted_nodes, 'community': sorted_communities})
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        partition_df.to_excel(writer, sheet_name='Community_Detection_Leiden', index=False)
        logging.info("社团检测结果已保存到 text_analysis_results.xlsx 的 Community_Detection_Leiden 表")

    logging.info("开始计算社团的各种中心性")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500)
    pagerank = nx.pagerank(G)

    centrality_df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree_centrality': pd.Series(degree_centrality),
        'betweenness_centrality': pd.Series(betweenness_centrality),
        'closeness_centrality': pd.Series(closeness_centrality),
        'eigenvector_centrality': pd.Series(eigenvector_centrality),
        'pagerank': pd.Series(pagerank)
    })

    partition_df = partition_df.merge(centrality_df, on='node')

    logging.info("开始保存每个社团的关键词及其中心性值")
    community_keywords = []
    for community in partition_df['community'].unique():
        community_nodes = partition_df[partition_df['community'] == community]
        community_keywords.extend(community_nodes.to_dict('records'))

    community_keywords_df = pd.DataFrame(community_keywords)
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        community_keywords_df.to_excel(writer, sheet_name='Community_Keywords_Centrality', index=False)
        logging.info(
            "每个社团的关键词及其中心性值已保存到 text_analysis_results.xlsx 的 Community_Keywords_Centrality 表")

    logging.info("开始进行层次聚类以合并社团")
    start_time = time.time()

    community_vectors = np.zeros((len(set(communities)), len(terms)))
    for i, comm in enumerate(sorted(set(communities))):
        community_indices = [idx for idx, c in enumerate(communities) if c == comm]
        community_vectors[i, :] = tfidf_matrix[community_indices, :].sum(axis=0).A1
    community_vectors = community_vectors / np.maximum(community_vectors.sum(axis=1, keepdims=True), 1)

    distance_matrix = pdist(community_vectors, metric='cosine')
    linkage_matrix = linkage(distance_matrix, method='average')
    cluster_labels = fcluster(linkage_matrix, t=0.988, criterion='distance')

    logging.info(f"层次聚类完成，耗时 {time.time() - start_time:.2f} 秒")
    logging.info(f"聚类后的社团数量: {len(set(cluster_labels))}")

    community_map = {comm: label for comm, label in zip(sorted(set(communities)), cluster_labels)}
    final_communities = np.array([community_map[comm] for comm in communities])

    merged_community_keywords = []
    for community in set(final_communities):
        community_indices = np.where(final_communities == community)[0]
        for idx in community_indices:
            node = sorted_nodes[idx]
            community_id = final_communities[idx]
            row = partition_df.loc[partition_df['node'] == node].iloc[0].to_dict()
            row['community'] = community_id
            merged_community_keywords.append(row)

    merged_community_keywords_df = pd.DataFrame(merged_community_keywords)

    formatted_rows = []
    for community_id in merged_community_keywords_df['community'].unique():
        formatted_rows.append({'community': f'社团编号: {community_id}', 'node': '', 'degree_centrality': '',
                               'betweenness_centrality': '', 'closeness_centrality': '', 'eigenvector_centrality': '',
                               'pagerank': ''})
        community_df = merged_community_keywords_df[merged_community_keywords_df['community'] == community_id]
        formatted_rows.extend(community_df.to_dict('records'))

    formatted_community_keywords_df = pd.DataFrame(formatted_rows)

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        formatted_community_keywords_df.to_excel(writer, sheet_name='Merged_Community_Key_Results', index=False)
        logging.info(
            "合并后的社团及其关键词和中心性值已保存到 text_analysis_results.xlsx 的 Merged_Community_Key_Results 表")

    logging.info("对包含超过50个关键词的社团进行进一步分割")


    def split_large_community(community_keywords, max_keywords=50, min_keywords=15):
        community_vector_matrix = tfidf_vectorizer.transform(community_keywords).toarray()
        G = nx.Graph()
        for i, word in enumerate(community_keywords):
            G.add_node(word, vector=community_vector_matrix[i])
        for i in range(len(community_keywords)):
            for j in range(i + 1, len(community_keywords)):
                similarity = np.dot(community_vector_matrix[i], community_vector_matrix[j])
                if similarity > 0:
                    G.add_edge(community_keywords[i], community_keywords[j], weight=similarity)
        partition = la.find_partition(ig.Graph.from_networkx(G), la.CPMVertexPartition, resolution_parameter=1.0)
        sub_communities = defaultdict(list)
        for word, community in zip(community_keywords, partition.membership):
            sub_communities[community].append(word)
        small_communities = {k: v for k, v in sub_communities.items() if len(v) < min_keywords}
        for small_key in small_communities:
            if small_communities[small_key]:
                for key, value in sub_communities.items():
                    if key != small_key and len(value) + len(small_communities[small_key]) <= max_keywords:
                        sub_communities[key].extend(small_communities[small_key])
                        sub_communities.pop(small_key)
                        break
        return sub_communities


    final_split_communities = {}
    community_id_offset = max(final_communities) + 1

    for community_id in set(final_communities):
        community_indices = np.where(final_communities == community_id)[0]
        community_keywords = [sorted_nodes[idx] for idx in community_indices if sorted_nodes[idx] in keywords]
        if len(community_keywords) > 50:
            sub_communities = split_large_community(community_keywords, max_keywords=50)
            for sub_id, sub_keywords in sub_communities.items():
                final_split_communities[community_id_offset + sub_id] = sub_keywords
            community_id_offset += len(sub_communities)
        else:
            final_split_communities[community_id] = community_keywords
    logging.info(f"分割后社团数量: {len(final_split_communities)}")

    logging.info("合并过小的社团")
    merged_small_communities = defaultdict(list)
    for community_id, keywords in final_split_communities.items():
        if len(keywords) < 15:
            closest_community = min(final_split_communities.keys(), key=lambda k: np.mean(
                [np.dot(tfidf_vectorizer.transform([kw]).toarray(), tfidf_vectorizer.transform([kw2]).toarray().T)
                 for kw in keywords for kw2 in final_split_communities[k]]))
            merged_small_communities[closest_community].extend(keywords)
            if len(merged_small_communities[closest_community]) < 15:
                continue  # 确保合并后关键词数量不低于15个
        else:
            merged_small_communities[community_id].extend(keywords)

    final_split_communities = {k: v for k, v in merged_small_communities.items() if len(v) >= 15}
    logging.info(f"合并后社团数量: {len(final_split_communities)}")

    logging.info("按照中心性值排序社团中的关键词")
    final_formatted_rows = []
    for community_id in final_split_communities:
        sorted_keywords = sorted(final_split_communities[community_id],
                                 key=lambda kw: partition_df.loc[partition_df['node'] == kw, 'pagerank'].values[0],
                                 reverse=True)
        final_formatted_rows.append({'community': f'社团编号: {community_id}', 'node': '', 'degree_centrality': '',
                                     'betweenness_centrality': '', 'closeness_centrality': '',
                                     'eigenvector_centrality': '', 'pagerank': ''})
        for keyword in sorted_keywords:
            row = partition_df.loc[partition_df['node'] == keyword].iloc[0].to_dict()
            row['community'] = community_id
            final_formatted_rows.append(row)

    final_formatted_communities_df = pd.DataFrame(final_formatted_rows)

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        final_formatted_communities_df.to_excel(writer, sheet_name='Final_Communities', index=False)
        logging.info("最终社团结果已保存到 text_analysis_results.xlsx 的 Final_Communities 表")


    def plot_term_frequencies(term_frequencies, terms, output_file):
        freq_df = pd.DataFrame({'term': terms, 'frequency': term_frequencies})
        freq_df = freq_df.sort_values(by='frequency', ascending=False).head(50)
        plt.figure(figsize=(15, 10))
        sns.barplot(x='frequency', y='term', data=freq_df)
        plt.title('Top 50 Term Frequencies')
        plt.xlabel('Frequency')
        plt.ylabel('Term')
        plt.savefig(output_file)
        plt.close()


    plot_term_frequencies(term_frequencies, terms, 'term_frequencies.png')


    def plot_cooccurrence_matrix(cooccurrence_matrix, output_file):
        plt.figure(figsize=(15, 15))
        sns.heatmap(cooccurrence_matrix, cmap='viridis')
        plt.title('Co-occurrence Matrix Heatmap')
        plt.xlabel('Terms')
        plt.ylabel('Terms')
        plt.savefig(output_file)
        plt.close()


    plot_cooccurrence_matrix(cooccurrence_matrix, 'cooccurrence_matrix.png')


    def plot_community_structure(G, partition, output_file):
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(15, 15))
        cmap = plt.get_cmap('viridis', max(partition) + 1)
        nx.draw_networkx_nodes(G, pos, node_size=50, cmap=cmap, node_color=partition)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.title('Community Structure')
        plt.savefig(output_file)
        plt.close()


    plot_community_structure(G, final_communities, 'community_structure.png')

except Exception as e:
    logging.error(f"处理过程出错: {e}")