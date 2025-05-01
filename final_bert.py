import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AdamW
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score,classification_report, f1_score, accuracy_score
import utils
from data_loader import load_train_data, load_test_data
import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import math



# ------------------ Dataset & Model ------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'inputs': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

class TimeSeriesBERTFineTune(nn.Module):
    def __init__(
        self,
        input_dim=39,
        hidden_size=768,
        seq_len=50,
        output_dim=39,
        freeze_until_layer=12       
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (a) projection
        self.embedding = nn.Linear(input_dim, hidden_size)

        # (b) sinusoidal PE được lưu dưới buffer
        pe = self.sinusoidal_positional_embedding(seq_len + 1, hidden_size)
        self.register_buffer("pos_embedding", pe)     # không rơi vào .parameters()

        # (c) learnable CLS, khởi tạo truncated‑normal std=0.02
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # (d) load SecBERT
        # self.bert = BertModel.from_pretrained("jackaduma/SecBERT")
        self.bert = BertModel.from_pretrained("distilbert-base-uncased")

        # (e) freeze phần dưới
        # for layer in self.bert.encoder.layer[:freeze_until_layer]:
        #     for p in layer.parameters():
        #         p.requires_grad = False

        # (f) head
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.to(self.device)

    def sinusoidal_positional_embedding(self, seq_len, hidden_dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        x shape: (B, seq_len, input_dim)
        Return: (B, output_dim)
        """
        batch_size = x.size(0)

        # 1) Embedding
        x = self.embedding(x)  # (B, seq_len, hidden_size)

        # 2) Thêm [CLS] vào đầu
        cls_expanded = self.cls_token.expand(batch_size, 1, -1)  # (B, 1, hidden_size)
        x = torch.cat([cls_expanded, x], dim=1)  # (B, seq_len + 1, hidden_size)

        # 3) Thêm positional embedding
        # pos_embedding shape: (1, seq_len+1, hidden_size)
        # x shape: (B, seq_len+1, hidden_size)
        x = x + self.pos_embedding[:, : x.size(1)] 

        # 4) Attention mask
        attention_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=self.device)

        # 5) Forward qua BERT
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)

        # 6) Lấy hidden state của [CLS] => index 0
        cls_state = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)

        # 7) Qua output_layer
        prediction = self.output_layer(cls_state)  # (B, output_dim)

        return prediction
    def compute_loss(self, predictions, targets, lambda_smooth=0.1):
        mse_loss = nn.MSELoss()(predictions, targets)
        # Smoothness regularization (differences between consecutive predictions)
        loss_smoothness = torch.mean((predictions[:, 1:] - predictions[:, :-1]) ** 2)
        total_loss = mse_loss + lambda_smooth * loss_smoothness
        return total_loss
    @torch.no_grad()
    def reconstruction_errors(self, X, history=100, device='cuda', batch_size=512):
        """
        Tính reconstruction error theo kiểu sliding window:
          - Tương tự logic 'sequence' bạn đã làm, i.e. X[i-history:i] -> dự đoán X[i]
          - Trả về mảng (num_samples, output_dim) chứa MSE per-feature ở mỗi timestep.
        
        X: np.array shape (N, input_dim) 
           => N phải >= history
        Return: errors shape (N - history, output_dim),
                mỗi dòng là (pred - ground_truth)^2 per-feature
        """
        self.eval()  # model eval
        all_errors = []

        # Duyệt theo batch để tránh OOM
        num_samples = len(X)
        start_idx = history+1
        end_idx   = num_samples

        # Chạy vòng lặp chunk
        idx = start_idx
        while idx < end_idx:
            # Lấy batch từ idx -> idx+batch_size
            batch_end = min(idx + batch_size, end_idx)
            # Lấy input sequences
            seq_list = []
            for i in range(idx, batch_end):
                seq = X[i-history:i]  # shape (history, input_dim)
                seq_list.append(seq)
            seq_batch = np.stack(seq_list, axis=0)  # (batch_size, history, input_dim)

            seq_batch_torch = torch.tensor(seq_batch, dtype=torch.float32, device=device)
            preds = self.forward(seq_batch_torch)  # (batch_size, output_dim)

            # ground truth = X[idx->batch_end], shape (batch_size, input_dim)
            # so sánh => error
            ground_truth = X[idx:batch_end]  # (batch_size, input_dim)
            ground_truth_torch = torch.tensor(ground_truth, dtype=torch.float32, device=device)

            # MSE mỗi feature
            # shape (batch_size, output_dim)
            squared_error = (preds - ground_truth_torch) ** 2

            all_errors.append(squared_error.cpu().numpy())
            idx = batch_end

        # ghép tất cả batch lại
        all_errors = np.concatenate(all_errors, axis=0)  # shape ((N-history), output_dim)
        return all_errors
    def reconstruction_errors_by_idxs(self, Xfull, idxs, history, bs=128, device='cuda'):
        self.eval()
        full_errors = np.zeros((len(idxs), Xfull.shape[1]))  # (num_samples, num_features)

        for start in range(0, len(idxs), bs):
            end = min(start + bs, len(idxs))
            batch_idxs = idxs[start:end]

            # Chuẩn bị batch input và target
            Xbatch = np.array([Xfull[i - history:i] for i in batch_idxs])  # shape (B, history, input_dim)
            Ybatch = np.array([Xfull[i+1] for i in batch_idxs])              # shape (B, input_dim)

            # Đưa vào PyTorch
            X_tensor = torch.tensor(Xbatch, dtype=torch.float32).to(device)
            Y_tensor = torch.tensor(Ybatch, dtype=torch.float32).to(device)

            # Dự đoán
            preds = self.forward(X_tensor)  # self là model

            # Tính lỗi bình phương
            squared_error = (preds - Y_tensor) ** 2  # shape (B, output_dim)

            # Gán vào full_errors
            full_errors[start:end] = squared_error.detach().cpu().numpy()


        return full_errors
    def cached_detect(self, instance_errors, theta, window=1):
        """
        instance_errors: shape (num_samples, output_dim),
                         thường là KQ của reconstruction_errors(...)

        - Tính trung bình lỗi trên các feature => instance_errors.mean(axis=1) => 1-D
        - So sánh với threshold => 0/1
        - Áp dụng window (nếu window>1, check consecutive)
        """
        # Lấy MSE trung bình mỗi timestep
        detection = instance_errors > theta

        # If window exceeds one, look for consective detections
        if window > 1:

            detection = np.convolve(detection, np.ones(window), 'same') // window

            # clement: Removing this behavior
            # Backfill the windows (e.g. if idx 255 is 1, all of 255-window:255 should be filled)
            # fill_idxs = np.where(detection)
            # fill_detection = detection.copy()
            # for idx in fill_idxs[0]:
            #     fill_detection[idx - window : idx] = 1
            # return fill_detection

        return detection
    def save_detection_params(self, best_theta, best_window):
        self.best_theta = best_theta
        self.best_window = best_window
        print(f"Storing best parameters as: theta={best_theta}, window={best_window}")

    def best_cached_detect(self, instance_errors):
        return self.cached_detect(instance_errors, theta=self.best_theta, window=self.best_window)
# ------------------ Training & Evaluation ------------------
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=5e-5,
    weight_decay=1e-4,
    use_scheduler=False,
    early_stopping_patience=5
):
    """
    - weight_decay: L2 regularization (AdamW)
    - use_scheduler: dùng ReduceLROnPlateau nếu muốn
    - early_stopping_patience: dừng sớm nếu val_loss ko cải thiện
    """

    # Dùng AdamW thay vì Adam (thường khuyến nghị cho transformer)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.to(device)

    train_losses, val_losses = [], []
    val_maes, val_r2s, val_mses = [], [], []

    best_val_loss = float('inf')
    no_improve_count = 0

    # Scheduler giảm lr khi val_loss không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    ) if use_scheduler else None

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs  = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(inputs)
            loss    = model.compute_loss(outputs, targets, lambda_smooth=0.1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✂️
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        predictions, true_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predictions.append(outputs.cpu().numpy())
                true_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        preds = np.vstack(predictions)
        trues = np.vstack(true_targets)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
        )

        # Update scheduler
        if scheduler:
            scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses, val_mses, val_maes, val_r2s

def plot_mse_per_feature(mse_per_feature, save_path="mse_per_feature.png", feature_names=None):
    num_features = len(mse_per_feature)
    x = np.arange(num_features)

    plt.figure(figsize=(12, 6))
    plt.bar(x, mse_per_feature, color='skyblue')
    if feature_names:
        plt.xticks(x, feature_names, rotation=90)
    else:
        plt.xticks(x, [f"F{i}" for i in x], rotation=90)
    plt.ylabel("MSE")
    plt.title("Mean Squared Error per Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ MSE plot saved to {save_path}")

def evaluate_model(model, data_loader, device, save_dir="results", model_name="my_model", feature_names=None):
    model.eval()
    predictions, targets = [], []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['inputs'].to(device)
            target = batch['targets'].to(device)
            output = model(inputs)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    # Ép kiểu float để tránh lỗi json dump
    mse_total = float(mean_squared_error(targets, predictions))
    mae_total = float(mean_absolute_error(targets, predictions))
    r2_total = float(r2_score(targets, predictions))

    mse_per_feature = [
        float(mean_squared_error(targets[:, i], predictions[:, i]))
        for i in range(targets.shape[1])
    ]
    mae_per_feature = [
        float(mean_absolute_error(targets[:, i], predictions[:, i]))
        for i in range(targets.shape[1])
    ]
    r2_per_feature  = [
        float(r2_score(targets[:, i], predictions[:, i]))
        for i in range(targets.shape[1])
    ]

    print(f"\n🔍 Evaluation Summary:")
    print(f"Total MSE: {mse_total:.4f}, MAE: {mae_total:.4f}, R²: {r2_total:.4f}")

    # Save metrics to JSON
    metrics = {
        "overall": {
            "mse": mse_total,
            "mae": mae_total,
            "r2": r2_total
        },
        "per_feature": {
            "mse": mse_per_feature,
            "mae": mae_per_feature,
            "r2": r2_per_feature
        }
    }

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save model weights
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved to {model_path}")

    # Plot MSE bar chart
    mse_plot_path = os.path.join(save_dir, f"{model_name}_mse_barplot.png")
    plot_mse_per_feature(mse_per_feature, save_path=mse_plot_path, feature_names=feature_names)

    return predictions, targets, metrics

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainable parameters: {trainable:,} / {total:,} ({(trainable/total)*100:.2f}%)")



def hyperparameter_search(event_detector,config, model_type, Xval, Xtest, Ytest, dataset_name, val_idxs=None, test_split=0.7, run_name='results', verbose=1):

    # model_name= "Distil_Bert"
    model_name= "Final_Secbert"
    do_batches = False

    Ytest = Ytest.astype(int)
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False)

   
    history = 100

        # Clip the prediction to match LSTM prediction window
    Ytest_test = Ytest_test[history + 1:]
    Ytest_val = Ytest_val[history + 1:]
    # Ytest_test = Ytest_test[history :]
    # Ytest_val = Ytest_val[history :]
    do_batches = True   

    ##### Cross Validation
    # if val_idxs is None:
    #     validation_errors = event_detector.reconstruction_errors(Xval)
    
    
    validation_errors = event_detector.reconstruction_errors_by_idxs(Xfull, val_idxs, history=history)

    

    test_errors = event_detector.reconstruction_errors(Xtest_val)
    test_instance_errors = test_errors.mean(axis=1)

    # Default to empty dict. Will still do F1 for window=1, ptile=95%
    grid_config = config.get('grid_search', dict())

    cutoffs = grid_config.get('percentile', [0.95])
    windows = grid_config.get('window', [1])
    print(grid_config)
    eval_metrics = grid_config.get('metrics', ['F1'])
    print(eval_metrics)

    firstPlotsError = True
    firstNpysError = True

    for metric in eval_metrics:

        # FPR is a negative metric (lower is better)
        negative_metric = (metric == 'false_positive_rate')

        # FPR is a negative metric (lower is better)
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        print("metric: ", metric)
        metric_func = metrics.get("F1")
        print("metric_fun: ",metric_func )

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(validation_errors.mean(axis = 1), percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]

                #Yhat = event_detector.detect(Xtest, theta = theta, window = window, batches=True)
                Yhat = event_detector.cached_detect(test_instance_errors, theta = theta, window = window)
                Yhat = Yhat[window-1:].astype(int)

                Yhat, Ytest_val = utils.normalize_array_length(Yhat, Ytest_val)
                # choice_value = metric_func(Yhat, Ytest_val)
                # chinh sua lai su dung f1 skeatleatn
                choice_value = f1_score(Ytest_val, Yhat, average='micro')

                if verbose > 0:
                    print("{} is {:.3f} at theta={:.3f}, percentile={:.4f}, window={}".format(metric, choice_value, theta, percentile, window))

                # FPR is a negative metric (lower is better)
                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

                if grid_config.get('save-metric-info', False):
                    metric_vals[percentile_idx, window_idx] = choice_value

                if grid_config.get('detection-plots', False):

                    fig_detect, ax_detect = plt.subplots(figsize=(20, 4))

                    ax_detect.plot(Yhat, color = '0.1', label = 'predicted state')
                    ax_detect.plot(Ytest_val, color = 'r', alpha = 0.75, lw = 2, label = 'real state')
                    ax_detect.fill_between(np.arange(len(Yhat)), 0, Yhat.astype(int), color = '0.1')
                    ax_detect.set_title(
                        'Detection trajectory on test dataset, {}, percentile={:.3f}, window={}'
                        .format(model_type, percentile, window), fontsize = 14)
                    ax_detect.set_yticks([0,1])
                    ax_detect.set_yticklabels(['NO ATTACK','ATTACK'])
                    ax_detect.legend(fontsize = 12, loc = 2)
                    try:
                        fig_detect.savefig(f'plots/{run_name}/{model_name}-{percentile}-{window}.pdf')
                        if firstPlotsError:
                            print(f"Saving plots for model {model_name} to plots/{run_name}")
                            firstPlotsError = False
                    except FileNotFoundError:
                        fig_detect.savefig(f'plots/results/{model_name}-{percentile}-{window}.pdf')
                        if firstPlotsError:
                            print(f"Directory plots/{run_name}/ not found, saving plots for model {model_name} to plots/results/ instead")
                            firstPlotsError = False
                    plt.close(fig_detect)

            if grid_config.get('save-theta', False):
                try:
                    pickle.dump(theta, open(f'models/{run_name}/{model_name}-{percentile}-theta.pkl', 'wb'))
                    print(f'Saved theta to models/{run_name}/{model_name}-{percentile}-theta.pkl')
                except FileNotFoundError:
                    pickle.dump(theta, open(f'models/results/{model_name}-{percentile}-theta.pkl', 'wb'))
                    print(f"Directory models/{run_name}/ not found, saved theta to models/results/{model_name}-{percentile}-theta.pkl instead")

        print("Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(metric, best_metric, best_percentile, best_window))

        # Final test performance
        final_test_errors = event_detector.reconstruction_errors(Xtest_test)
        final_test_instance_errors = final_test_errors.mean(axis=1)

        best_theta = np.quantile(validation_errors.mean(axis = 1), best_percentile)
        event_detector.save_detection_params(best_theta=best_theta, best_window=best_window)

        final_Yhat = event_detector.best_cached_detect(final_test_instance_errors)
        final_Yhat = final_Yhat[best_window-1:].astype(int)

        metric_func = metrics.get(metric)
        final_Yhat, Ytest_test = utils.normalize_array_length(final_Yhat, Ytest_test)
        final_value = metric_func(Ytest_test,final_Yhat)
        final_value =final_value = f1_score(Ytest_test,final_Yhat, average= "micro")
        print("Final {} is {:.3f} at percentile={:.5f}, window {}".format(metric, final_value, best_percentile, best_window))

        if grid_config.get('save-metric-info', False):
            try:
                np.save(f'npys/{run_name}/{model_name}-{metric}.npy', metric_vals)
                print(f'Saved metric at npys/{run_name}/{model_name}-{metric}.npy')
            except FileNotFoundError:
                np.save(f'npys/results/{model_name}-{metric}.npy', metric_vals)
                if firstPlotsError:
                    print(f"Directory npys/{run_name}/ not found, saved metric {model_name}-{metric}.npy to npys/results/ instead")
                    firstPlotsError = False
                    
        # ****************
        # luu de ve bieu do ROC
        np.save(f'results/{model_name}_y_true.npy', Ytest_test)
        np.save(f'results/{model_name}_y_score.npy', final_Yhat)
        f1_micro = f1_score(Ytest_test, final_Yhat, average="micro")
        f1_macro = f1_score(Ytest_test, final_Yhat, average="macro")
        precision_micro = precision_score(Ytest_test, final_Yhat, average="micro")
        precision_macro = precision_score(Ytest_test, final_Yhat, average="macro")
        recall_micro = recall_score(Ytest_test, final_Yhat, average="micro")
        recall_macro = recall_score(Ytest_test, final_Yhat, average="macro")

        # output f1 score
        report = classification_report(Ytest_test, final_Yhat, digits=4)
        # roc curve
        fpr, tpr, _ = roc_curve(Ytest_test, final_Yhat)
        # precision-recall curve
        precision, recall, _ = precision_recall_curve(Ytest_test, final_Yhat)
        cm = confusion_matrix(Ytest_test, final_Yhat)
        df_cm = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        # output this to a file
        with open(f"results/{model_name}-classification_report.txt", "w") as f:
            # f.write(f"test_errors: {test_errors}\n")
            f.write(report+"\n")
            f.write(f"F1 Score (micro): {f1_micro}\n")
            f.write(f"F1 Score (macro): {f1_macro}\n")
            f.write(f"Precision (micro): {precision_micro}\n")
            f.write(f"Precision (macro): {precision_macro}\n")
            f.write(f"Recall (micro): {recall_micro}\n")
            f.write(f"Recall (macro): {recall_macro}\n")
            f.write(f"Final {metric}: {final_value}\n")
            f.write(f"Best {metric}: {best_metric}\n")
            f.write(f"Best percentile: {best_percentile}\n")
            f.write(f"Best window: {best_window}\n")
            f.write(f"Confusion matrix:\n")
            f.write(f"{df_cm}\n")
            f.write(f"ROC Curve:\n")
            f.write(f"False Positive Rate: {fpr}\n")
            f.write(f"True Positive Rate: {tpr}\n")
            f.write(f"Precision-Recall Curve:\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
        print("Classification report đã được lưu vào 'classification_report.txt'")
    return event_detector

# ------------------ MAIN ------------------
if __name__ == "__main__":
    import random
    model_type = "Final_secBert"
    # model_type= "SEC_Bert"
    dataset_name = "BATADAL"
    history = 100
    test_ratio = 0.2

    config = {
        'grid_search': {
            # Mặc định muốn quét threshold ở mức percentile=0.95
            'percentile': [ 
                0.90, 0.91,0.92, 0.93, 0.94, 
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
            0.991,
            0.992,
            0.993,
            0.994,
            0.995,
            0.996,
            0.997,
            0.998,
            0.999,
            0.9995,
            0.99995,], 

            # Mặc định chỉ kiểm tra window=1 (không dùng smoothing window)
            'window': [1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],         

            # Mặc định dùng metric F1
            'metrics': ['F1'],     

            'pr-plot': False,
            'detection-plots': False,
            'save-metric-info': True,
            'save-theta': True
        }
    }
   
    # 1) Load data
    Xfull, _ = load_train_data(dataset_name)
    Xtest, Ytest, _ = load_test_data(dataset_name)
    # Xfull = Xfull[:1000]
    print("Xshape: ", Xfull.shape)
    print("X[0]: ", Xfull[0])

    # -- (A) Chuẩn hóa X (khuyến nghị) --
   

    # 2) Chia random 80-20
    # X_train, X_val, _, _  = train_test_split(Xfull, Xfull, test_size=0.2, random_state=42, shuffle=False)
    train_idxs, val_idxs = utils.train_val_history_idx_split(Xfull, history)
    
    # print("After split, X_train:", X_train.shape, "X_val:", X_val.shape)

    # ## test code
    # X_train= X_train[0: 300] 
    
    # 3) Tạo chuỗi (sequence) và target cho TRAIN
    train_seq = [Xfull[i - history:i] for i in train_idxs]
    train_tgt = [Xfull[i+1] for i in train_idxs]

    # 4) Tạo chuỗi (sequence) và target cho VAL
    val_seq = [Xfull[i - history:i] for i in val_idxs]
    val_tgt = [Xfull[i+1] for i in val_idxs]

    print("train_seq len:", len(train_seq), "val_seq len:", len(val_seq))

    # 5) Dataset, DataLoader
    train_ds = TimeSeriesDataset(train_seq, train_tgt)
    val_ds   = TimeSeriesDataset(val_seq, val_tgt)
    train_loader = DataLoader(train_ds, batch_size=64,
                          shuffle=True, num_workers=4,
                          drop_last=False, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=4,
                            drop_last=False, pin_memory=True)

    # 6) Khởi tạo model (với freeze layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesBERTFineTune(
        input_dim=39,
        output_dim=39,
        seq_len=history,
        freeze_until_layer=12  # Freeze encoder.layer.0 đến encoder.layer.7
    )

    count_trainable_params(model)
    
  
    # model.load_state_dict(torch.load("eval_results/bert_sec_finetune.pt"))
    # model.to(device)
    # model.eval()


     

    # 7) Huấn luyện (thử weight_decay=1e-4, use_scheduler=True, early_stopping_patience=5)
    train_losses, val_losses, val_mses, val_maes, val_r2s = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=20,
        lr=5e-5,
        
        weight_decay=1e-4,
        use_scheduler=True,
        early_stopping_patience=5
    )
    
    # Ve do thi lose
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=300)  # hoặc .pdf để đưa vào paper
    # plt.show()
    plt.close()
        
    loss_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_mse': val_mses,
        'val_mae': val_maes,
        'val_r2': val_r2s
    })

    loss_df.to_csv('results/training_logs.csv', index=False)
    ######################


    # 8) Evaluate + plot
    preds, trues, _ = evaluate_model(
        model,
        val_loader,
        device,
        save_dir="eval_results",
        model_name="distilbertfinetune",
        feature_names=None
    )
    validation_errors = model.reconstruction_errors_by_idxs(Xfull, val_idxs, history=history)
    val_errors =validation_errors.mean(axis=1)
    print("validation: ",validation_errors.mean(axis=1))
    with open(f"{model_type}- validation", "w") as f:
        f.write("validation per timestep:\n")
        for val in val_errors:
            f.write(f"{val}\n")

   
    
    hyperparameter_search(model,config, model_type, Xfull, Xtest, Ytest,dataset_name,val_idxs= val_idxs,verbose=0)
