import matplotlib.pyplot as plt
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