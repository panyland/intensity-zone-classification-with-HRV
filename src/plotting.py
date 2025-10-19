import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_subject_data(data, output_dir='plots/descriptive'):
    os.makedirs(output_dir, exist_ok=True)

    for id_value, group in data.groupby('ID'):
        vt1_powers = group.loc[group['vt1_marker'] == 1, 'power'].unique()
        vt2_powers = group.loc[group['vt2_marker'] == 1, 'power'].unique()

        # RR and Power
        fig, ax1 = plt.subplots()
        ax1.plot(group['time'], group['RR'], color='tab:blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RR (ms)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(group['time'], group['power'], color='tab:red')
        ax2.set_ylabel('Power (W)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        for vt in vt1_powers:
            ax2.axhline(y=vt, color='tab:purple', linestyle='--', linewidth=1.5)
        for vt in vt2_powers:
            ax2.axhline(y=vt, color='tab:orange', linestyle='--', linewidth=1.5)

        plt.title(f'RR and Power for ID {id_value}')
        fig.tight_layout()
        plt.savefig(f'{output_dir}/plot_RR_power_{id_value}.png')
        plt.close()

        # VO2 and Power
        fig, ax1 = plt.subplots()
        ax1.plot(group['time'], group['VO2'], color='tab:green')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('VO2 (ml/min)', color='tab:green')
        ax1.tick_params(axis='y', labelcolor='tab:green')

        ax2 = ax1.twinx()
        ax2.plot(group['time'], group['power'], color='tab:red')
        ax2.set_ylabel('Power (W)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        for vt in vt1_powers:
            ax2.axhline(y=vt, color='tab:purple', linestyle='--', linewidth=1.5)
        for vt in vt2_powers:
            ax2.axhline(y=vt, color='tab:orange', linestyle='--', linewidth=1.5)

        plt.title(f'VO2 and Power for ID {id_value}')
        fig.tight_layout()
        plt.savefig(f'{output_dir}/plot_VO2_power_{id_value}.png')
        plt.close()


def plot_confusion_matrix(cm, y_test, output_dir='plots/results'):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test)
    )
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.title('Random Forest Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
    plt.close()


def plot_importances(im, output_dir='plots/results'):
    indices = ['sdnn','rmssd','sampen', 'lf','hf','lf_hf','total_power']

    plt.figure(figsize=(10, 4))
    plt.bar(indices, im, color='skyblue')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Gini importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    plt.close()


def plot_patterns_by_class(X, y, output_dir='plots/results'):
    classes = np.unique(y)
    plt.figure(figsize=(10, 4))
    for c in classes:
        seqs = X[y == c] 
        mean_seq = np.mean(seqs, axis=0)
        std_seq = np.std(seqs, axis=0)
        plt.plot(mean_seq, label=f'{c}')
        plt.fill_between(range(len(mean_seq)), mean_seq - std_seq, mean_seq + std_seq, alpha=0.2)

    plt.xlabel('RR position in sequence', fontsize=12)
    plt.ylabel('RR interval (ms)', fontsize=12)
    plt.title('Mean RR sequences ± 1 SD per class', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rr_sequence_patterns.png', dpi=300)
    plt.close()