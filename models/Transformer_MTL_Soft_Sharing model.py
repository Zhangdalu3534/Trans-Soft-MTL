import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os
from bayes_opt import BayesianOptimization
import random

file_path = 'F:\Disaster_Case_Anomaly Analysis - Four Categories.xlsx'
sheet_names = ['Geological anomaly', 'Earthquake anomaly', 'Freezing and Fire anomaly', 'Typhoon and Storm anomaly']
stages = ['潜伏期', '爆发期', '扩散期', '衰退期', '长尾期']

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_data(file_path, sheet_names, stages):
    scaler = MinMaxScaler()
    all_data, all_targets, all_task_indices, task_feature_names = [], [], [], []

    # 收集所有 Topic 和 Stage 列
    all_topic_columns, all_stage_columns = set(), set()
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if 'Consecutive Anomaly Region Topic' in df.columns:
            all_topic_columns.update(pd.get_dummies(df['Consecutive Anomaly Region Topic'], prefix='Topic').columns)
        if 'Stage' in df.columns:
            all_stage_columns.update(pd.get_dummies(df['Stage'], prefix='Stage').columns)

    all_topic_columns, all_stage_columns = sorted(all_topic_columns), sorted(all_stage_columns)

    for task_idx, sheet in enumerate(sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet)
        feature_columns = ['Number of Hot Topics', 'Anomaly Duration (hours)']

        encoded_stages = pd.get_dummies(df['Stage'], prefix='Stage').reindex(columns=all_stage_columns, fill_value=0)

        encoded_topics = pd.get_dummies(df['Consecutive Anomaly Region Topic'], prefix='Topic').reindex(
            columns=all_topic_columns, fill_value=0)

        df[feature_columns] = scaler.fit_transform(df[feature_columns].fillna(0))

        df['Crisis_Degree'] = df['Average Sentiment*24'].apply(lambda x: np.exp(abs(x)) if x < 0 else x)
        df[['Crisis_Degree']] = MinMaxScaler().fit_transform(df[['Crisis_Degree']])

        task_features = np.hstack([df[feature_columns].values, encoded_stages.values, encoded_topics.values])
        X_aug, y_aug = augment_data(task_features.astype(np.float32), df['Crisis_Degree'].values.astype(np.float32))

        all_data.append(X_aug)
        all_targets.append(y_aug)
        all_task_indices.extend([task_idx] * len(X_aug))
        task_feature_names.append(feature_columns + all_stage_columns + all_topic_columns)

    return np.vstack(all_data), np.hstack(all_targets), np.array(all_task_indices), task_feature_names

def augment_data(X, y, factor=6, noise_level=0.01):
    X_augmented, y_augmented = [], []
    for _ in range(factor):
        noise = np.random.normal(scale=noise_level, size=X.shape)
        X_augmented.append(X + noise)
        y_augmented.append(y)
    return np.vstack(X_augmented), np.hstack(y_augmented)

class SoftSharingMultiTaskModel(nn.Module):
    def __init__(self, input_size, shared_hidden_size, task_hidden_size, num_tasks, num_heads=4, num_layers=2):
        super().__init__()
        self.shared_layer = nn.Linear(input_size, shared_hidden_size)
        self.task_layers = nn.ModuleList([nn.Linear(shared_hidden_size, task_hidden_size) for _ in range(num_tasks)])
        self.output_layers = nn.ModuleList([nn.Linear(task_hidden_size, 1) for _ in range(num_tasks)])

        # Attention Layer to calculate attention scores for tasks
        self.attention_layer = nn.Linear(shared_hidden_size, num_tasks)

        # Transformer Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=shared_hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x, task_idx):
        # Shared Layer Output
        shared_output = torch.relu(self.shared_layer(x))  # (batch_size, shared_hidden_size)

        # Task-Specific Output with Attention Weights
        attention_weights = torch.softmax(self.attention_layer(shared_output), dim=1)  # (batch_size, num_tasks)

        # Apply self-attention mechanism on the shared output
        shared_output = shared_output.unsqueeze(1)  # (batch_size, 1, shared_hidden_size)
        attention_output, _ = self.self_attention(shared_output, shared_output, shared_output)  # (batch_size, 1, shared_hidden_size)
        attention_output = attention_output.squeeze(1)  # (batch_size, shared_hidden_size)

        # Get the task-specific output by applying attention weight
        task_output = torch.relu(self.task_layers[task_idx](attention_output) * attention_weights[:, task_idx].unsqueeze(1))

        # Return final output for the task
        return self.output_layers[task_idx](task_output)

def optimize_hyperparameters(data, targets, task_indices, num_tasks, device):
    def objective(shared_hidden_size, task_hidden_size, learning_rate, num_heads, num_layers):
        model = SoftSharingMultiTaskModel(
            input_size=data.shape[1],
            shared_hidden_size=int(shared_hidden_size),
            task_hidden_size=int(task_hidden_size),
            num_tasks=num_tasks,
            num_heads=int(num_heads),
            num_layers=int(num_layers)
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        val_losses = np.zeros(num_tasks)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(data.cpu().numpy(), task_indices.cpu().numpy())):
            for epoch in range(400):
                train_loss_per_task = train_model(model, optimizer, data[train_idx], targets[train_idx],
                                                  task_indices[train_idx], num_tasks, device)
                val_loss_per_task = train_model(model, optimizer, data[val_idx], targets[val_idx], task_indices[val_idx],
                                                num_tasks, device)
                for task_idx in range(num_tasks):
                    val_losses[task_idx] += val_loss_per_task[task_idx]

        return np.mean(val_losses)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'shared_hidden_size': (16, 128),
            'task_hidden_size': (8, 64),
            'learning_rate': (1e-5, 1e-2),
            'num_heads': (2, 8),
            'num_layers': (1, 4)
        },
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=10)

    print("Best Hyperparameters:", optimizer.max)
    return optimizer.max['params']

def train_model(model, optimizer, data, targets, task_indices, num_tasks, device):
    model.train()
    total_losses = [0] * num_tasks
    task_counts = [0] * num_tasks

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
    if isinstance(task_indices, np.ndarray):
        task_indices = torch.tensor(task_indices, dtype=torch.int64, device=device)

    for task_idx in range(num_tasks):
        task_data = data[task_indices == task_idx].to(device)
        task_targets = targets[task_indices == task_idx].unsqueeze(1).to(device)

        if len(task_data) == 0:
            continue

        optimizer.zero_grad()
        outputs = model(task_data, task_idx)
        loss = nn.MSELoss()(outputs, task_targets)
        loss.backward()
        optimizer.step()

        total_losses[task_idx] += loss.item()
        task_counts[task_idx] += 1

    avg_losses = [total_losses[i] / task_counts[i] if task_counts[i] > 0 else 0 for i in range(num_tasks)]
    return avg_losses

def plot_loss_curve(train_losses, val_losses, task_colors, output_dir="Loss_Curves"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 8))
    for task_idx in range(len(train_losses)):
        plt.plot(
            range(len(train_losses[task_idx])),
            train_losses[task_idx],
            label=f"Task {task_idx} Train Loss",
            marker='o',
            color=task_colors[task_idx]
        )
        plt.plot(
            range(len(val_losses[task_idx])),
            val_losses[task_idx],
            label=f"Task {task_idx} Val Loss",
            linestyle='--',
            marker='s',
            color=task_colors[task_idx]
        )
    plt.title("Loss Curves for All Tasks")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid()

    overall_curve_file = os.path.join(output_dir, "All_Tasks_Loss_Curves.png")
    plt.savefig(overall_curve_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"All tasks' loss curves saved to {overall_curve_file}")

    for task_idx in range(len(train_losses)):
        plt.figure(figsize=(12, 8))
        plt.plot(
            range(len(train_losses[task_idx])),
            train_losses[task_idx],
            label=f"Task {task_idx} Train Loss",
            marker='o',
            color=task_colors[task_idx]
        )
        plt.plot(
            range(len(val_losses[task_idx])),
            val_losses[task_idx],
            label=f"Task {task_idx} Val Loss",
            linestyle='--',
            marker='s',
            color=task_colors[task_idx]
        )

        plt.title(f"Loss Curve for Task {task_idx}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.grid()
        task_curve_file = os.path.join(output_dir, f"Task_{task_idx}_Loss_Curve.png")
        plt.savefig(task_curve_file, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Task {task_idx} loss curve saved as {task_curve_file}")

def plot_learning_curve(model_class, data, targets, task_indices, num_tasks, input_size, params, stratified_kfold, device, output_dir="Learning_Curves"):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_losses_per_task = [[] for _ in range(num_tasks)]
    val_losses_per_task = [[] for _ in range(num_tasks)]

    for train_size in train_sizes:
        fold_train_losses = [[] for _ in range(num_tasks)]
        fold_val_losses = [[] for _ in range(num_tasks)]

        for train_idx, val_idx in stratified_kfold.split(data, task_indices):
            subset_train_idx = train_idx[:int(len(train_idx) * train_size)]

            train_data = torch.tensor(data[subset_train_idx], dtype=torch.float32, device=device)
            val_data = torch.tensor(data[val_idx], dtype=torch.float32, device=device)
            train_targets = torch.tensor(targets[subset_train_idx], dtype=torch.float32, device=device)
            val_targets = torch.tensor(targets[val_idx], dtype=torch.float32, device=device)
            train_task_indices = torch.tensor(task_indices[subset_train_idx], dtype=torch.int64, device=device)
            val_task_indices = torch.tensor(task_indices[val_idx], dtype=torch.int64, device=device)

            model = model_class(
                input_size=input_size,
                shared_hidden_size=params['shared_hidden_size'],
                task_hidden_size=params['task_hidden_size'],
                num_tasks=num_tasks
            )
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

            train_loss_per_task = train_model(model, optimizer, train_data, train_targets, train_task_indices, num_tasks, device)
            val_loss_per_task = train_model(model, optimizer, val_data, val_targets, val_task_indices, num_tasks, device)

            for task_idx in range(num_tasks):
                fold_train_losses[task_idx].append(train_loss_per_task[task_idx])
                fold_val_losses[task_idx].append(val_loss_per_task[task_idx])

        for task_idx in range(num_tasks):
            train_losses_per_task[task_idx].append(np.mean(fold_train_losses[task_idx]))
            val_losses_per_task[task_idx].append(np.mean(fold_val_losses[task_idx]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for task_idx in range(num_tasks):
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_losses_per_task[task_idx], label="Train Loss", marker='o')
        plt.plot(train_sizes, val_losses_per_task[task_idx], label="Validation Loss", linestyle='--', marker='s')
        plt.title(f"Learning Curve for {sheet_names[task_idx]}")
        plt.xlabel("Training Set Proportion")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"Task_{task_idx}_Learning_Curve.png"))
        plt.close()

    plt.figure(figsize=(12, 8))
    for task_idx in range(num_tasks):
        plt.plot(train_sizes, train_losses_per_task[task_idx], label=f"Task {task_idx} Train", linestyle="-")
        plt.plot(train_sizes, val_losses_per_task[task_idx], label=f"Task {task_idx} Val", linestyle="--")
    plt.title("Overall Learning Curves for All Tasks")
    plt.xlabel("Training Set Proportion")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "All_Tasks_Learning_Curve.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    for task_idx in range(num_tasks):
        plt.plot(train_sizes, train_losses_per_task[task_idx], label=f"Task {task_idx} Train", linestyle="-")
        plt.plot(train_sizes, val_losses_per_task[task_idx], label=f"Task {task_idx} Val", linestyle="--")
    plt.title("Overall Learning Curves for All Tasks")
    plt.xlabel("Training Set Proportion")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "All_Tasks_Learning_Curve.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    for task_idx in range(num_tasks):
        plt.plot(train_sizes, train_losses_per_task[task_idx], label=f"Task {task_idx} Train", linestyle="-")
        plt.plot(train_sizes, val_losses_per_task[task_idx], label=f"Task {task_idx} Val", linestyle="--")
    plt.title("Overall Learning Curves for All Tasks")
    plt.xlabel("Training Set Proportion")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "All_Tasks_Learning_Curve.png"))
    plt.close()

def main():
    set_random_seed(42)
    data, targets, task_indices, task_feature_names = preprocess_data(file_path, sheet_names, stages)
    data, targets, task_indices = (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(task_indices, dtype=torch.int64),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, targets, task_indices = data.to(device), targets.to(device), task_indices.to(device)

    model = SoftSharingMultiTaskModel(
        input_size=data.shape[1],
        shared_hidden_size=64,
        task_hidden_size=32,
        num_tasks=len(sheet_names),
        num_heads=4,
        num_layers=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    task_colors = ["blue", "red", "pink", "cyan"]
    train_losses, val_losses = [[] for _ in range(len(sheet_names))], [[] for _ in range(len(sheet_names))]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data.cpu().numpy(), task_indices.cpu().numpy())):
        for epoch in range(400):
            train_loss_per_task = train_model(model, optimizer, data[train_idx], targets[train_idx],
                                              task_indices[train_idx], len(sheet_names), device)
            val_loss_per_task = train_model(model, optimizer, data[val_idx], targets[val_idx], task_indices[val_idx],
                                            len(sheet_names), device)

            for task_idx in range(len(sheet_names)):
                train_losses[task_idx].append(train_loss_per_task[task_idx])
                val_losses[task_idx].append(val_loss_per_task[task_idx])

        plot_loss_curve(train_losses, val_losses, task_colors)

    if not os.path.exists("Scatter_Plots"):
        os.makedirs("Scatter_Plots")

    def evaluate_model(model, data, targets, task_indices, num_tasks, device, sheet_names, output_dir="Scatter_Plots"):
        model.eval()
        results = []
        overall_predictions = []
        overall_true_values = []
        total_mse, total_mae, total_rmse, total_mape = 0, 0, 0, 0
        total_samples = 0

        with torch.no_grad():
            for task_idx in range(num_tasks):
                task_data = data[task_indices == task_idx].to(device)
                task_targets = targets[task_indices == task_idx].cpu().numpy()
                outputs = model(task_data, task_idx=task_idx).cpu().numpy()

                mse = mean_squared_error(task_targets, outputs.flatten())
                r2 = r2_score(task_targets, outputs.flatten())

                mae = np.mean(np.abs(task_targets - outputs))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((task_targets - outputs) / task_targets)) * 100

                total_mse += mse * len(task_targets)
                total_mae += mae * len(task_targets)
                total_rmse += rmse * len(task_targets)
                total_mape += mape * len(task_targets)
                total_samples += len(task_targets)

                results.append({
                    "Task": sheet_names[task_idx],
                    "MSE": mse,
                    "R²": r2,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape
                })

                overall_predictions.extend(outputs.flatten())
                overall_true_values.extend(task_targets)

                plt.figure(figsize=(8, 6))
                plt.scatter(task_targets, outputs.flatten(), alpha=0.5)
                plt.plot([min(task_targets), max(task_targets)], [min(task_targets), max(task_targets)], 'k--',
                         lw=2)
                plt.xlabel('The real data')
                plt.ylabel('The predicted data')
                plt.title(f'{sheet_names[task_idx]} (R² = {r2:.4f})')
                plt.savefig(os.path.join(output_dir, f'Task_{task_idx}_Scatter_Plot.png'), dpi=600, bbox_inches="tight")
                plt.close()

        amse = total_mse / total_samples
        amae = total_mae / total_samples
        armse = total_rmse / total_samples
        amape = total_mape / total_samples
        overall_mse = mean_squared_error(overall_true_values, overall_predictions)
        overall_r2 = r2_score(overall_true_values, overall_predictions)

        print("\nModel Performance for Each Task:")
        print(f"{'Task':<30}{'MSE':<15}{'R²':<15}{'MAE':<15}{'RMSE':<15}{'MAPE':<15}")
        print("-" * 90)
        for result in results:
            print(
                f"{result['Task']:<30}{result['MSE']:<15.4f}{result['R²']:<15.4f}{result['MAE']:<15.4f}{result['RMSE']:<15.4f}{result['MAPE']:<15.4f}")

        print("\nModel Performance (Overall):")
        print(f"Mean Squared Error (MSE): {overall_mse:.4f}")
        print(f"R²: {overall_r2:.4f}")
        print(f"AMSE: {amse:.4f}")
        print(f"AMAE: {amae:.4f}")
        print(f"ARMSE: {armse:.4f}")
        print(f"AMAPE: {amape:.4f}%")

        df = pd.DataFrame(results)
        df.to_excel('task_performance_metrics.xlsx', index=False)

        return results, overall_mse, overall_r2, amse, amae, armse, amape


    task_metrics, overall_mse, overall_r2, amse, amae, armse, amape = evaluate_model(
        model, data, targets, task_indices, len(sheet_names), device, sheet_names, output_dir="Scatter_Plots")

    shap_results = []
    for i in range(len(sheet_names)):
        explainer = shap.Explainer(
            lambda x: model(torch.tensor(x, dtype=torch.float32, device=device), i).detach().cpu().numpy(),
            data.cpu().numpy()
        )
        shap_values = explainer(data.cpu().numpy())
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        shap_results.extend([{
            "Task": sheet_names[i],
            "Feature": feature,
            "Mean SHAP": val
        } for feature, val in zip(task_feature_names[i], mean_shap_values)])

    df_shap = pd.DataFrame(shap_results)
    df_metrics = pd.DataFrame(task_metrics)

    with pd.ExcelWriter("MTL_Soft_Sharing_SHAP_Results.xlsx") as writer:
        df_shap.to_excel(writer, sheet_name="SHAP Values", index=False)
        df_metrics.to_excel(writer, sheet_name="Task Metrics", index=False)

    print("Results saved to MTL_Soft_Sharing_SHAP_Results.xlsx")

if __name__ == "__main__":
    main()
