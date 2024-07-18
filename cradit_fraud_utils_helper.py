import os
import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_checkpoint(model, epoch, checkpoint_dir='models/focal_loss_checkpoints', title=''):
    checkpoint_dir = checkpoint_dir + title
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch

def save_model_comparison(model_comparison, path):
    model_comparison = pd.DataFrame(model_comparison).T
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(model_comparison, annot=True, cmap='viridis', cbar=True, annot_kws={"size": 12}, fmt='.2f')
    plt.title('Model Performance Comparison', fontsize=20, pad=20)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()