import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import re
import xgboost as xgb
from catboost import CatBoostRegressor
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import matplotlib.font_manager as fm

file_path = 'F:/Disaster_Case_Analysis.xlsx'
xls = pd.ExcelFile(file_path)
sheets = ['Geological', 'Earthquake', 'Freezing', 'Storm', 'Forest Fire', 'Typhoon']

dataframes = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheets}

def parse_user_distribution(user_str):
    try:
        if isinstance(user_str, str):
            topics = re.findall(r'#(\d+):\[(.*?)\]', user_str)
            topic_distributions = {}
            for topic_id, dist in topics:
                proportions = list(map(float, dist.split(',')))
                topic_distributions[int(topic_id)] = proportions
            return topic_distributions
        else:
            return None
    except (SyntaxError, ValueError):
        print(f"Error parsing user distribution: {user_str}")
        return None

def parse_and_classify_topic_path(topic_paths_str):
    parsed_topics = []
    if pd.isna(topic_paths_str) or not isinstance(topic_paths_str, str) or topic_paths_str.strip() == "":
        return []
    try:

        topics = re.findall(r'#\d+:\[([^,]+),([^,]+)->([^,]+),([0-9.]+)\]', topic_paths_str)
        for category, start, end, importance in topics:
            parsed_topics.append({
                'Category': category.strip(),
                'Path': f"{start.strip()}->{end.strip()}",
                'Importance': float(importance)
            })
    except Exception as e:
        print(f"Error parsing topic paths: {e}")
    return parsed_topics

def summarize_topic_paths(df):
    topic_features = []
    for index, row in df.iterrows():
        topic_paths_str = row.get('Hot Topic Path', '')
        parsed_topics = parse_and_classify_topic_path(topic_paths_str)

        for topic in parsed_topics:
            topic_features.append({
                'case ID': row.get('case ID', ''),
                'Stage': row.get('Stage', ''),
                'Duration': row.get('Duration (hours)', 0),
                'Category': topic['Category'],
                'Path': topic['Path'],
                'Importance': topic['Importance'],
                'User Type Distribution': row.get('User Type Distribution', '')
            })

    return pd.DataFrame(topic_features)

def apply_xgboost(df, sheet_name, stage_name):
    print(f"Applying XGBoost for sheet: {sheet_name}, stage: {stage_name}")

    le_category = LabelEncoder()
    le_path = LabelEncoder()

    df = df.copy()
    df['Category_Encoded'] = le_category.fit_transform(df['Category'].astype(str))
    df['Path_Encoded'] = le_path.fit_transform(df['Path'].astype(str))

    X_user_dist = df['User Type Distribution'].apply(lambda x: parse_user_distribution(x))
    X_user_dist = X_user_dist.dropna()

    X_features = []
    for dist in X_user_dist:
        if dist and len(list(dist.values())) > 0:
            all_distributions = [np.array(proportions) for proportions in dist.values() if len(proportions) == 5]
            if all_distributions:
                avg_distribution = np.mean(all_distributions, axis=0)
                X_features.append(avg_distribution)
            else:
                X_features.append([0, 0, 0, 0, 0])
        else:
            X_features.append([0, 0, 0, 0, 0])

    X_features = np.array(X_features)
    df_valid = df.loc[X_user_dist.index]

    df_valid = df_valid.dropna(subset=['Duration', 'Category', 'Importance'])
    valid_idx = df_valid.index

    X_features = X_features[:len(valid_idx)]

    category_encoded = df_valid['Category_Encoded'].values.reshape(-1, 1)
    path_encoded = df_valid['Path_Encoded'].values.reshape(-1, 1)

    X_combined = np.column_stack((df_valid['Duration'], X_features, category_encoded, path_encoded))

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    y = df_valid['Importance']

    xgboost_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgboost_model.fit(X_combined_scaled, y)

    y_pred = xgboost_model.predict(X_combined_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"XGBoost MSE for stage {stage_name}: {mse}")
    print(f"XGBoost R² for stage {stage_name}: {r2}")

    importances = xgboost_model.feature_importances_
    important_features_indices = np.argsort(importances)[::-1][:len(importances)]
    print("Selected features by XGBoost Importance:", important_features_indices)

    return important_features_indices, importances, mse, r2

def apply_catboost(df, sheet_name, stage_name):
    print(f"Applying CatBoost for sheet: {sheet_name}, stage: {stage_name}")

    le_category = LabelEncoder()
    le_path = LabelEncoder()

    df = df.copy()

    df['Category_Encoded'] = le_category.fit_transform(df['Category'].astype(str))
    df['Path_Encoded'] = le_path.fit_transform(df['Path'].astype(str))

    X_user_dist = df['User Type Distribution'].apply(lambda x: parse_user_distribution(x))
    X_user_dist = X_user_dist.dropna()

    X_features = []
    for dist in X_user_dist:
        if dist and len(list(dist.values())) > 0:
            all_distributions = [np.array(proportions) for proportions in dist.values() if len(proportions) == 5]
            if all_distributions:
                avg_distribution = np.mean(all_distributions, axis=0)
                X_features.append(avg_distribution)
            else:
                X_features.append([0, 0, 0, 0, 0])
        else:
            X_features.append([0, 0, 0, 0, 0])

    X_features = np.array(X_features)
    df_valid = df.loc[X_user_dist.index]

    df_valid = df_valid.dropna(subset=['Duration', 'Category', 'Importance'])
    valid_idx = df_valid.index

    X_features = X_features[:len(valid_idx)]

    category_encoded = df_valid['Category_Encoded'].values.reshape(-1, 1)
    path_encoded = df_valid['Path_Encoded'].values.reshape(-1, 1)

    X_combined = np.column_stack((df_valid['Duration'], X_features, category_encoded, path_encoded))

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    y = df_valid['Importance']

    catboost_model = CatBoostRegressor(n_estimators=100, depth=6, random_state=42, verbose=0)
    catboost_model.fit(X_combined_scaled, y)

    y_pred = catboost_model.predict(X_combined_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"CatBoost MSE for stage {stage_name}: {mse}")
    print(f"CatBoost R² for stage {stage_name}: {r2}")
    catboost_importances = catboost_model.get_feature_importance()
    important_features_indices = np.argsort(catboost_importances)[::-1][:len(catboost_importances)]
    print("Selected features by CatBoost Importance:", important_features_indices)

    return important_features_indices, catboost_importances, mse, r2

results = []

for sheet, df in dataframes.items():
    print(f"Analyzing features for disaster type: {sheet}")
    df_summarized = summarize_topic_paths(df)
    stages = df_summarized['Stage'].unique()
    for stage in stages:
        stage_df = df_summarized[df_summarized['Stage'] == stage]
        xgboost_features, xgboost_importances, xgboost_mse, xgboost_r2 = apply_xgboost(stage_df, sheet, stage)
        catboost_features, catboost_importances, catboost_mse, catboost_r2 = apply_catboost(stage_df, sheet, stage)

        for feature_index in xgboost_features:
            results.append({
                'Disaster Type': sheet,
                'Stage': stage,
                'Model': 'XGBoost',
                'Feature Index': feature_index,
                'Importance': xgboost_importances[feature_index],
                'MSE': xgboost_mse,
                'R²': xgboost_r2
            })

        for feature_index in catboost_features:
            results.append({
                'Disaster Type': sheet,
                'Stage': stage,
                'Model': 'CatBoost',
                'Feature Index': feature_index,
                'Importance': catboost_importances[feature_index],
                'MSE': catboost_mse,
                'R²': catboost_r2
            })

results_df = pd.DataFrame(results)
results_df.to_excel('XGBoost_CatBoost_Feature_Selection_MSE_R2_Results.xlsx', index=False)
print("Results saved to XGBoost_CatBoost_Feature_Selection_MSE_R2_Results.xlsx")

color_map = {
    'Geological': 'blue',
    'Earthquake': 'green',
    'Freezing': 'red',
    'Storm': 'orange',
    'Forest Fire': 'purple',
    'Typhoon': 'cyan'
}

disaster_map = {disaster: idx for idx, disaster in enumerate(results_df['Disaster Type'].unique())}

font_path = r'C:\Users\Lenovo\Desktop\SimHei.ttf'
zh_font = fm.FontProperties(fname=font_path)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

disasters = results_df['Disaster Type'].unique()
stages = ['潜伏期', '爆发期', '扩散期', '衰退期', '长尾期']

stage_map = {stage: idx for idx, stage in enumerate(stages)}

for disaster in disasters:
    for model in ['XGBoost', 'CatBoost']:
        disaster_data = results_df[(results_df['Disaster Type'] == disaster) & (results_df['Model'] == model)].copy()
        disaster_data.loc[:, 'Stage_Numeric'] = disaster_data['Stage'].map(stage_map)
        disaster_data = disaster_data.sort_values('Stage_Numeric')

        stage_numeric = disaster_data['Stage_Numeric'].values
        importance = disaster_data['Importance'].values
        _, unique_indices = np.unique(stage_numeric, return_index=True)
        stage_numeric = stage_numeric[unique_indices]
        importance = importance[unique_indices]

        if model == 'CatBoost':
            importance = 2 * importance / 100

        interp_points = 100
        spline = make_interp_spline(stage_numeric, importance, k=3)
        smooth_x = np.linspace(stage_numeric.min(), stage_numeric.max(), interp_points)
        smooth_importance = spline(smooth_x)

        disaster_numeric = disaster_map[disaster]
        line_style = '--' if model == 'CatBoost' else '-'
        ax.plot(smooth_x, [disaster_numeric] * len(smooth_x), smooth_importance, label=f'{disaster} - {model}',
                linestyle=line_style, color=color_map[disaster], linewidth=2)

ax.set_xlabel('Stage', fontsize=12)
ax.set_ylabel('Disaster Type', fontsize=12)
ax.set_zlabel('Importance', fontsize=12)
ax.set_xticks(list(stage_map.values()))
ax.set_xticklabels(list(stage_map.keys()), fontsize=10, fontproperties=zh_font, rotation=45)
ax.set_yticks(list(disaster_map.values()))
ax.set_yticklabels(list(disaster_map.keys()), fontsize=10)
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10)
ax.view_init(elev=20, azim=40)
plt.title("不同灾害类型的舆情演化平滑轨迹 (XGBoost vs CatBoost)", fontsize=14)

output_dir = '舆情演化平滑轨迹.png'
plt.savefig(output_dir, bbox_inches='tight', dpi=800, pad_inches=0.2)
plt.show()
print(f"图形已保存为 {output_dir}")