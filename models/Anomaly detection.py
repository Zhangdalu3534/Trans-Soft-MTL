import os
import pandas as pd
import numpy as np
import logging
from collections import defaultdict, Counter
from statsmodels.tsa.seasonal import STL
from networkx import degree_centrality, closeness_centrality, betweenness_centrality, eigenvector_centrality, pagerank
import networkx as nx
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_valid_filename(s):
    return "".join([c if c.isalnum() or c in (' ', '.') else '_' for c in s]).rstrip()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

file_path = r'D:\Python-work\G2021text_analysis_results.xlsx'
logging.info("开始加载数据")
final_communities_df = pd.read_excel(file_path, sheet_name='Final_Communities')
data = pd.read_excel(file_path, sheet_name='Processed_Content')
logging.info("数据加载完成")
logging.info("开始转换时间字段")
try:
    data['time'] = pd.to_datetime(data['time'], format='%Y年%m月%d日 %H:%M', errors='coerce')
except Exception as e:
    logging.error(f"时间转换出错: {e}")
    data['time'] = pd.to_datetime(data['time'], format='mixed', errors='coerce')
logging.info("时间字段转换完成")

invalid_times = data[data['time'].isna()]
if not invalid_times.empty:
    logging.warning(f"无法解析的时间数据: {invalid_times}")

data = data.dropna(subset=['time'])
logging.info("过滤无法解析的时间数据完成")

data = data.sort_values(by='time')

def calculate_topic_vector(doc_words, communities):
    topic_vector = np.zeros(len(communities))
    for i, community in enumerate(communities):
        community_words = set(community)
        overlap = len(doc_words & community_words)
        topic_vector[i] = overlap / len(doc_words)
    return topic_vector

def clean_community_id(community_id):
    if isinstance(community_id, str):
        return int(community_id.replace('社团编号:', '').strip())
    return community_id

logging.info("开始计算社区词列表")
communities = []
final_communities_df['cleaned_community'] = final_communities_df['community'].apply(clean_community_id)
community_ids = final_communities_df['cleaned_community'].unique()
for community_id in community_ids:
    community_words = set(final_communities_df[final_communities_df['cleaned_community'] == community_id]['node'])
    communities.append(community_words)
logging.info("社区词列表计算完成")
logging.info(f"Unique community IDs: {community_ids}")
logging.info("开始计算文档的隶属度向量")
document_topic_vectors = []
for doc in data['processed_content']:  # 使用分词后的内容
    doc_words_set = set(doc.split())
    topic_vector = calculate_topic_vector(doc_words_set, communities)
    document_topic_vectors.append(topic_vector)
logging.info("文档的隶属度向量计算完成")

def calculate_popularity(document_topic_vectors, post_times, interval_hours=3):
    topic_popularity = defaultdict(lambda: np.zeros(len(communities)))
    interval = pd.Timedelta(hours=interval_hours)
    for topic_vector, post_time in zip(document_topic_vectors, post_times):
        rounded_time = post_time - pd.Timedelta(minutes=post_time.minute % interval.seconds // 60,
                                                seconds=post_time.second,
                                                microseconds=post_time.microsecond)
        topic_popularity[rounded_time] += topic_vector
    return topic_popularity

unique_times = sorted(set(data['time']))

logging.info("开始计算话题热度")
topic_popularity = calculate_popularity(document_topic_vectors, data['time'])
logging.info("话题热度计算完成")

total_popularity = sum(topic_popularity.values())

logging.info("开始计算每个时点的话题热度占比")
if np.any(total_popularity == 0):
    logging.warning("某些总热度值为零，可能导致计算无效。")
    total_popularity[total_popularity == 0] = 1  # 避免除以零

topic_popularity_percentage = {time: popularity / total_popularity for time, popularity in topic_popularity.items()}
logging.info("每个时点的话题热度占比计算完成")

topic_popularity_df = pd.DataFrame(topic_popularity_percentage).T.sort_index()
topic_popularity_df.columns = [f'Topic_{i}' for i in range(len(communities))]

output_dir = r'D:\Python-work\G2021异常点检测及分析'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'topic_popularity.xlsx')
logging.info(f"开始保存结果到 {output_file}")
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    topic_popularity_df.to_excel(writer, sheet_name='Topic_Popularity', index=True)
logging.info("结果保存完成")

time_span = (data['time'].max() - data['time'].min()).total_seconds()
interval_hours = 3
period = int(time_span // (interval_hours * 3600))

color_list = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

anomalies_info = []
anomaly_segments = []

logging.info("开始使用STL分解话题热度时间序列")
for topic in topic_popularity_df.columns:
    series = topic_popularity_df[topic]
    stl = STL(series, period=period)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    try:
        plt.figure(figsize=(10, 8))
        plt.subplot(411)
        plt.plot(series, label='Original')
        plt.title(f'Topic {topic} - Original')
        plt.legend(loc='upper right')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.title('Trend')
        plt.legend(loc='upper right')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal')
        plt.title('Seasonal')
        plt.legend(loc='upper right')
        plt.subplot(414)
        plt.plot(residual, label='Residual')
        plt.title('Residual')
        plt.legend(loc='upper right')
        plt.tight_layout()

        stl_image_file = os.path.join(output_dir, f'stl_decomposition_{topic}.png')
        plt.savefig(stl_image_file)
        logging.info(f"STL分解图已保存到 {stl_image_file}")
        plt.close()
    except Exception as e:
        logging.error(f"绘制或保存STL分解图时出错: {e}")

    logging.info("开始进行异常检测")
    b_value = 3

    residual_mean = residual.mean()
    residual_std = residual.std()
    threshold = residual_mean + b_value * residual_std

    anomalies = residual[(residual > threshold)].index

    consecutive_anomalies = []
    isolated_anomalies = []
    segment_start = None

    for i in range(len(anomalies) - 1):
        if (anomalies[i + 1] - anomalies[i]).total_seconds() <= 24 * 3600:
            consecutive_anomalies.append(anomalies[i])
            consecutive_anomalies.append(anomalies[i + 1])
            if segment_start is None:
                segment_start = anomalies[i]
        else:
            if anomalies[i] not in consecutive_anomalies:
                isolated_anomalies.append(anomalies[i])
            if segment_start is not None:
                anomaly_segments.append((segment_start, anomalies[i], topic))
                segment_start = None
            if i == len(anomalies) - 2 and anomalies[i + 1] not in consecutive_anomalies:
                isolated_anomalies.append(anomalies[i + 1])

    consecutive_anomalies = sorted(set(consecutive_anomalies))
    isolated_anomalies = sorted(set(isolated_anomalies))

    if segment_start is not None:
        anomaly_segments.append((segment_start, anomalies[-1], topic))

    for anomaly in consecutive_anomalies:
        anomalies_info.append((anomaly, topic, 'consecutive'))
    for anomaly in isolated_anomalies:
        anomalies_info.append((anomaly, topic, 'isolated'))

    logging.debug(
        f"Topic {topic}: Consecutive anomalies {consecutive_anomalies}, Isolated anomalies {isolated_anomalies}")

logging.info("异常检测完成")

for topic in topic_popularity_df.columns:
    try:
        plt.figure(figsize=(10, 8))

        plt.subplot(211)
        plt.plot(topic_popularity_df[topic], label=topic)
        for idx, (start, end, t) in enumerate(anomaly_segments):
            if t == topic:
                color = color_list[idx % len(color_list)]
                plt.axvspan(start, end, color=color, alpha=0.3, label=f'Consecutive Anomalies {idx + 1}')
        plt.scatter([a[0] for a in anomalies_info if a[1] == topic and a[2] == 'isolated'],
                    topic_popularity_df[topic][[a[0] for a in anomalies_info if a[1] == topic and a[2] == 'isolated']],
                    color='orange', label='Isolated Anomalies')
        plt.title(f'Topic {topic} - Popularity with Anomalies (b={b_value})')
        plt.legend(loc='upper right')
        plt.subplot(212)
        plt.plot(residual)
        plt.axhline(residual_mean, color='black', linestyle='--', label='Mean')
        plt.axhline(threshold, color='blue', linestyle='--', label='μ + bσ')
        for idx, (start, end, t) in enumerate(anomaly_segments):
            if t == topic:
                color = color_list[idx % len(color_list)]
                plt.axvspan(start, end, color=color, alpha=0.3, label=f'Consecutive Anomalies {idx + 1}')
        plt.scatter([a[0] for a in anomalies_info if a[1] == topic and a[2] == 'isolated'],
                    residual[[a[0] for a in anomalies_info if a[1] == topic and a[2] == 'isolated']],
                    color='orange', label='Isolated Anomalies')
        plt.title(f'Residual with Anomalies (b={b_value})')
        plt.legend(loc='upper right')
        plt.tight_layout()

        anomaly_image_file = os.path.join(output_dir, f'anomaly_detection_{topic}_b{b_value}.png')
        plt.savefig(anomaly_image_file)
        logging.info(f"异常检测图已保存到 {anomaly_image_file}")
        plt.close()  # 关闭图形，避免显示阻塞
    except Exception as e:
        logging.error(f"绘制或保存异常检测图时出错: {e}")

plt.figure(figsize=(14, 8))
for topic in topic_popularity_df.columns:
    plt.plot(topic_popularity_df.index, topic_popularity_df[topic], label=topic)
plt.xlabel('Time')
plt.ylabel('Popularity')
plt.title('Topic Popularity Over Time')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
all_topics_image_file = os.path.join(output_dir, 'all_topics_popularity_trend.png')
plt.savefig(all_topics_image_file)
plt.close()
logging.info(f"所有话题的热度趋势图已保存到 {all_topics_image_file}")
logging.info("开始分析情感极性变化和关键词")
sentiment_analysis = []
keyword_analysis = []

for segment in anomaly_segments:
    try:
        start, end, topic = segment
        period_data = data[(data['time'] >= start) & (data['time'] <= end)]
        if period_data.empty:
            logging.warning(f"No data found for segment: {segment}")
            continue

        sentiments = period_data['processed_content'].apply(lambda x: TextBlob(x).sentiment.polarity)
        avg_sentiment = sentiments.mean()
        sentiment_analysis.append((start, end, topic, avg_sentiment))

        words = period_data['processed_content'].str.cat(sep=' ').split()
        word_freq = Counter(words)
        common_words = word_freq.most_common(10)
        keyword_analysis.append((start, end, topic, common_words))

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(period_data['time'], sentiments, label='Sentiment Polarity')
            plt.title(
                f'Sentiment Polarity from {start.strftime("%Y-%m-%d %H:%M:%S")} to {end.strftime("%Y-%m-%d %H:%M:%S")} for Topic {topic}')
            plt.xlabel('Time')
            plt.ylabel('Sentiment Polarity')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.gcf().autofmt_xdate()
            plt.legend(loc='upper right')
            plt.tight_layout()
            valid_filename = create_valid_filename(
                f'sentiment_polarity_{start.strftime("%Y-%m-%d_%H-%M-%S")}_{end.strftime("%Y-%m-%d_%H-%M-%S")}_topic_{topic}.png')
            plt.savefig(os.path.join(output_dir, valid_filename))
            plt.close()
            plt.figure(figsize=(10, 6))
            plt.plot(period_data['time'], period_data['popularity'], label='Popularity')
            plt.title(f'Popularity from {start.strftime("%Y-%m-%d %H:%M:%S")} to {end.strftime("%Y-%m-%d %H:%M:%S")} for Topic {topic}')
            plt.xlabel('Time')
            plt.ylabel('Popularity')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.gcf().autofmt_xdate()  # 自动旋转日期标签
            plt.legend(loc='upper right')
            plt.tight_layout()
            valid_filename = create_valid_filename(
                f'popularity_{start.strftime("%Y-%m-%d_%H-%M-%S")}_{end.strftime("%Y-%m-%d_%H-%M-%S")}_topic_{topic}.png')
            plt.savefig(os.path.join(output_dir, valid_filename))
            plt.close()
        except Exception as e:
            logging.error(f"绘制或保存情感极性或热度变化图时出错: {e}")
    except ValueError as ve:
        logging.error(f"解包时出错: {ve}, segment: {segment}")
        continue

sentiment_df = pd.DataFrame(sentiment_analysis, columns=['Start', 'End', 'Topic', 'Average_Sentiment'])
keyword_df = pd.DataFrame(keyword_analysis, columns=['Start', 'End', 'Topic', 'Common_Words'])

logging.info("开始分析单独异常点的中心性")
G = nx.Graph()
for idx, row in data.iterrows():
    words = row['processed_content'].split()
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)):
            G.add_edge(word, words[j])

centrality_analysis = []
for anomaly in [a[0] for a in anomalies_info if a[2] == 'isolated']:
    try:
        period_data = data[data['time'] == anomaly]
        for _, row in period_data.iterrows():
            words = row['processed_content'].split()
            subgraph = G.subgraph(words)
            if len(subgraph) > 0:
                degree = degree_centrality(subgraph)
                closeness = closeness_centrality(subgraph)
                betweenness = betweenness_centrality(subgraph)
                eigenvector = eigenvector_centrality(subgraph, max_iter=500)
                pr = pagerank(subgraph)
                centrality_analysis.append((anomaly, degree, closeness, betweenness, eigenvector, pr))
            else:
                centrality_analysis.append((anomaly, None, None, None, None, None))
    except Exception as e:
        logging.error(f"分析中心性时出错: {e}, anomaly: {anomaly}")

centrality_df = pd.DataFrame(centrality_analysis,
                             columns=['Anomaly_Time', 'Degree', 'Closeness', 'Betweenness', 'Eigenvector', 'PageRank'])

with pd.ExcelWriter(os.path.join(output_dir, 'anomaly_analysis_results.xlsx')) as writer:
    sentiment_df.to_excel(writer, sheet_name='Sentiment_Analysis', index=False)
    keyword_df.to_excel(writer, sheet_name='Keyword_Analysis', index=False)
    centrality_df.to_excel(writer, sheet_name='Centrality_Analysis', index=False)

logging.info("情感极性、关键词和中心性分析完成")