import numpy as np 
import datetime 
import os 
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.dates as mdates

class Journal:
    # ANSI color codes
    COLORS = {
        'RED': '\033[91m',
        'YELLOW': '\033[93m',
        'GREEN': '\033[92m',
        'BLUE': '\033[94m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }

    def __init__(self, dir_name, experiment_name):
        # Create directory relative to current working directory
        file_dir = os.path.join(os.getcwd(), dir_name)
        
        # Create directory if it doesn't exist
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
            print(f"{self.COLORS['GREEN']}Directory created successfully{self.COLORS['RESET']}")
            
        self.file_dir = file_dir
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        self.save_path = os.path.join(file_dir, experiment_name + ".txt")
        self.box_width = 70
        self.separator = "=" * self.box_width
    
    def format_line(self, content: str) -> str:
        return f"|{content.ljust(self.box_width - 4)}|"

    def _get_datainfo(self, data, name: str) -> None:
        # For console output (with colors)
        colored_lines = [
            self.separator,
            self.format_line(f"{self.COLORS['BLUE']}Data Statistics {name}{self.COLORS['RESET']}"),
            self.format_line(f"length: {len(data)}"),
            self.format_line(f"max: {np.max(data)}"),
            self.format_line(f"min: {np.min(data)}"),
            self.format_line(f"mean: {np.mean(data):.4f}"),
            self.format_line(f"std: {np.std(data):.4f}"),
            self.separator + "\n"
        ]
        
        # Print colored version to console
        for line in colored_lines:
            print(line)
        
        # For file output (without colors)
        clean_lines = [line for line in colored_lines]
        for color in self.COLORS.values():
            clean_lines = [line.replace(color, '') for line in clean_lines]
        
        with open(self.save_path, "a") as f:
            f.write("\n".join(clean_lines))

    def _get_warning(self, e) -> None:
        # Console output (with colors)
        warning_msg = f"{self.COLORS['YELLOW']}{self.COLORS['BOLD']}WARNING! {e}{self.COLORS['RESET']}"
        formatted_msg = self.format_line(warning_msg)
        print(formatted_msg)
        
        # File output (without colors)
        clean_msg = self.format_line(f"WARNING! {e}")
        with open(self.save_path, "a") as f:
            f.write(clean_msg + "\n")

    def _get_error(self, e) -> None:
        # Console output (with colors)
        error_msg = f"{self.COLORS['RED']}{self.COLORS['BOLD']}ERROR! {e}{self.COLORS['RESET']}"
        formatted_msg = self.format_line(error_msg)
        print(formatted_msg)
        
        # File output (without colors)
        clean_msg = self.format_line(f"ERROR! {e}")
        with open(self.save_path, "a") as f:
            f.write(clean_msg + "\n")

    def _get_data_stats(self, stats) -> None:
        # For console output (with colors)
        identifier_line = self.format_line("Energy Data Statistics")
        colored_lines = [self.separator, identifier_line] + [
            self.format_line(f"{self.COLORS['BLUE']} {key}: {stats.get(key)}{self.COLORS['RESET']}")
            for key in stats
        ]
        
        # Print colored version to console
        for line in colored_lines:
            print(line)

        # For file output (without colors)
        clean_lines = [self.separator, identifier_line] + [
            self.format_line(f" {key}: {stats.get(key)}")
            for key in stats
        ]
        
        # Write clean version to file
        with open(self.save_path, "a") as f:
            f.write("\n".join(clean_lines) + "\n")

    
    def _process_smoothly(self, tex):

        tex1 = f"{self.COLORS['GREEN']}{self.COLORS['BOLD']}Process! {tex}{self.COLORS['RESET']}"
        formatted_msg = self.format_line(tex1)
        print(formatted_msg)
        
        # File output (without colors)
        clean_msg = self.format_line(f"Process! {tex}")
        with open(self.save_path, "a") as f:
            f.write(clean_msg + "\n")


    def _plot_energy(self, dataframe):
        # Save data to CSV first
        self._save_energy_data(dataframe)
        
        # Create figure with more height and proper spacing
        fig, ax = plt.subplots(3, 1, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.4)  # Add space between subplots
        
        # Ensure datetime is set as index
        if 'datetime' in dataframe.columns:
            df_plot = dataframe.set_index('datetime')
        else:
            df_plot = dataframe
        
        # Plot Solar Generation
        ax[0].plot(df_plot.index, df_plot["solar"], 
                   label="Solar Generation", 
                   color='orange', 
                   linewidth=2)
        ax[0].set_title('Solar Power Generation', fontsize=12, pad=10)
        ax[0].set_ylabel('Power (kW)', fontsize=10)
        ax[0].grid(True, linestyle='--', alpha=0.7)
        ax[0].legend(fontsize=10)
        
        # Plot Load
        ax[1].plot(df_plot.index, df_plot["load"], 
                   label="Load Demand", 
                   color='blue', 
                   linewidth=2)
        ax[1].set_title('Load Demand', fontsize=12, pad=10)
        ax[1].set_ylabel('Power (kW)', fontsize=10)
        ax[1].grid(True, linestyle='--', alpha=0.7)
        ax[1].legend(fontsize=10)
        
        # Plot Net Power (Load - Solar)
        net_power = df_plot["load"] - df_plot["solar"]
        ax[2].plot(df_plot.index, net_power, 
                   label="Net Power", 
                   color='green', 
                   linewidth=2)
        ax[2].set_title('Net Power (Load - Solar)', fontsize=12, pad=10)
        ax[2].set_ylabel('Power (kW)', fontsize=10)
        ax[2].set_xlabel('Time', fontsize=10)
        ax[2].grid(True, linestyle='--', alpha=0.7)
        ax[2].legend(fontsize=10)
        
        # Format x-axis for all subplots
        for a in ax:
            # Rotate and align the tick labels so they look better
            plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Use AutoDateFormatter for smart date formatting
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)  # Adjust number of ticks
            formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')  # Custom date format
            a.xaxis.set_major_locator(locator)
            a.xaxis.set_major_formatter(formatter)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Create figures directory if it doesn't exist
        save_path = os.path.join("Code/results/", 'figures')
        os.makedirs(save_path, exist_ok=True)
        
        # Save the figure
        save_path = os.path.join(save_path, 'energy_profiles.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log the save operation
        self._process_smoothly(f"Energy profiles plot saved to {save_path}")

    def _save_energy_data(self, dataframe):
        """Save energy data to CSV with datetime index"""
        
        # Create csv directory if it doesn't exist
        save_dir = os.path.join("Code/results/", 'csv')
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare dataframe for saving
        if 'datetime' in dataframe.columns:
            df_to_save = dataframe.set_index('datetime')
        else:
            df_to_save = dataframe.copy()
        
        # Add net power column
        df_to_save['net_power'] = df_to_save['load'] - df_to_save['solar']
        
        # Generate filename with timestamp
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'energy_data.csv'
        save_path = os.path.join(save_dir, filename)
        
        # Save to CSV
        df_to_save.to_csv(save_path)
        
        # Log the save operation
        self._process_smoothly(f"Energy data saved to {save_path}")

    def _save_training_data(self, episode_data, formatted_time):
        """Save training episode data to CSV"""
        # Create csv directory if it doesn't exist
        save_dir = os.path.join("Code/results/", 'csv')
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert episode data to DataFrame
        df = pd.DataFrame(episode_data)
        df['episode'] = range(len(df))
        
        # Save to CSV
        filename = f'training_data_energy.csv'
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
        
        self._process_smoothly(f"Training data saved to {save_path}")
        return df

    def _plot_training_results(self, step_df, episode_df, formatted_time):
        """Create plots for both step-level and episode-level data"""
        save_dir = os.path.join("Code/results/", 'figures')
        os.makedirs(save_dir, exist_ok=True)
        
        # Step-level plots
        step_plots = {
            'step_rewards': ['reward'],
            'step_battery': ['soc', 'current'],
            'step_losses': ['p_loss', 'q_loss', 'q_loss_cumulative'],
            'step_violations': ['soc_violation', 'action_violation']
        }
        
        # Episode-level plots
        episode_plots = {
            'episode_rewards': ['total_reward'],
            'episode_soc': ['mean_soc', 'min_soc', 'max_soc'],
            'episode_performance': ['soc_violations', 'mean_current'],
            'episode_losses': ['total_p_loss', 'total_q_loss', 'final_q_loss_cumulative']
        }
        
        # Create step-level plots
        for name, metrics in step_plots.items():
            plt.figure(figsize=(12, 6))
            for metric in metrics:
                if metric in step_df.columns:
                    plt.plot(step_df['step'], step_df[metric], label=metric)
            plt.title(f'{name.replace("_", " ").title()} per Step')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{name}_{formatted_time}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            self._process_smoothly(f"{name} plot saved")
        
        # Create episode-level plots
        for name, metrics in episode_plots.items():
            plt.figure(figsize=(12, 6))
            for metric in metrics:
                if metric in episode_df.columns:
                    plt.plot(episode_df['episode'], episode_df[metric], label=metric)
            plt.title(f'{name.replace("_", " ").title()} per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{name}_{formatted_time}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            self._process_smoothly(f"{name} plot saved")


