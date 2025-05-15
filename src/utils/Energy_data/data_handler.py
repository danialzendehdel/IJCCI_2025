import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.utils.journal import Journal

def load_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)

    print(f'df.keys(): {df.keys()}')

    # Convert datetime and set as index
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Ensure data is at 15-minute intervals
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()
    expected_times = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    # Reindex to include all expected timestamps
    df = df.set_index('datetime').reindex(expected_times)
    
    # Forward fill small gaps (up to 1 hour)
    df = df.fillna(method='ffill', limit=4)
    
    # Reset index to make datetime a column again
    df = df.reset_index()
    df = df.rename(columns={'index': 'datetime'})

    # Drop any remaining NaN values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["solar"] = df["solar"].clip(lower=0)
    # calculate hour
    df["hour"] = df["datetime"].apply(lambda x: x.hour + x.minute/60)
    df["day_of_week"] = df["datetime"].apply(lambda x: x.weekday())

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Number of days: {(df['datetime'].max() - df['datetime'].min()).days + 1}")

    # Verify hour calculation
    sample_times = df["datetime"].head()
    sample_hours = df["hour"].head()
    print("\nSample time and calculated hours:")
    for t, h in zip(sample_times, sample_hours):
        print(f"Time: {t}, Calculated hour: {h:.2f}")
    
    solar_max, solar_min, solar_mean = df["solar"].max(), df["solar"].min(), df["solar"].mean()
    load_max, load_min, load_mean = df["load"].max(), df["load"].min(), df["load"].mean()
    print(f"\nsolar max: {solar_max}, solar min: {solar_min}, solar mean: {solar_mean}")
    print(f"load max: {load_max}, load min: {load_min}, load mean: {load_mean}")

    if solar_min < 0:
        assert False, "solar min value is negative"


    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Solar generation distribution
    ax1.hist(df["solar"], bins=50, alpha=0.7, color='orange', label="Solar")
    ax1.set_title("Distribution of Solar Generation")
    ax1.set_xlabel("Power (kW)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    # Weekly data plot (first week as example)
    week_data = df.iloc[:7*24*4] # First week (7 days * 24 hours * 4 quarters)
    ax2.plot(week_data["datetime"], week_data["solar"], label="Solar")
    ax2.plot(week_data["datetime"], week_data["load"], label="Load") 
    ax2.set_title("Weekly Solar Generation and Load")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power (kW)")
    ax2.legend()
    
    # Daily data plot (first day as example)
    day_data = df.iloc[:24*4] # First day (24 hours * 4 quarters)
    ax3.plot(day_data["datetime"], day_data["solar"], label="Solar")
    ax3.plot(day_data["datetime"], day_data["load"], label="Load")
    ax3.set_title("Daily Solar Generation and Load")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Power (kW)")
    ax3.legend()
    
    # Load distribution
    ax4.hist(df["load"], bins=50, alpha=0.7, color='blue', label="Load")
    ax4.set_title("Distribution of Load")
    ax4.set_xlabel("Power (kW)")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    save_path = "/home/danial/Documents/Codes/RL-BMS2/outputs/plots/"
    plt.savefig(f"{save_path.split('/')[-1]}_plots.png")

    return df, {
        "solar_max": solar_max, 
        "solar_min": solar_min, 
        "solar_mean": solar_mean, 
        "load_max": load_max, 
        "load_min": load_min, 
        "load_mean": load_mean
    }


class DataHandler(Dataset):
    def __init__(self, file_path):
        # Load and preprocess data
        df = pd.read_csv(file_path)
        print(f'df.keys(): {df.keys()}')
        print(f'Initial rows: {len(df)}')
        # print(f'NaN values before dropping:\n{df.isna().sum()}')

        # Convert datetime with UTC and set as index
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
        
        # Drop NaN values
        df.dropna(inplace=True)
        # print(f'Rows after dropping NaN: {len(df)}')
        
        # Reset index to make datetime a column again
        df = df.reset_index()
        
        # Sort by datetime and remove duplicates
        df = df.sort_values('datetime').drop_duplicates('datetime')
        
        # Clip solar values
        df["solar"] = df["solar"].clip(lower=0)
        
        # Calculate hour and day_of_week
        df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute/60
        df["day_of_week"] = df["datetime"].dt.weekday
        
        # print(f'Final rows: {len(df)}')
        # print(f'NaN values after processing:\n{df.isna().sum()}')
        
        # Store statistics
        self.stats = {
            "solar_max": df["solar"].max(), 
            "solar_min": df["solar"].min(), 
            "solar_mean": df["solar"].mean(),
            "load_max": df["load"].max(), 
            "load_min": df["load"].min(), 
            "load_mean": df["load"].mean(),
            "number of points": len(df),
            "start_date": df["datetime"].min(),
            "end_date": df["datetime"].max()
        }
        
        self.df = df
        self.current_index = 0

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.df.iloc[index]
    
    @staticmethod
    def sync_nan_drop(data_handlers):
        """Synchronize NaN dropping across multiple DataHandler instances"""
        # Get all dataframes
        dfs = [handler.df for handler in data_handlers]
        
        # Get all unique timestamps
        all_times = pd.concat([df['datetime'] for df in dfs]).unique()
        
        # Find timestamps that have NaN in any dataset
        nan_times = set()
        for df in dfs:
            df_nan_times = df[df.isna().any(axis=1)]['datetime']
            nan_times.update(df_nan_times)
        
        # Remove rows with these timestamps from all datasets
        for handler in data_handlers:
            handler.df = handler.df[~handler.df['datetime'].isin(nan_times)]
            handler.df.reset_index(drop=True, inplace=True)
        
        return data_handlers





if __name__ == "__main__":
    file_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/data/processed_data_661.csv"
    # df, stats = load_data(file_path)

    data_handler = DataHandler(file_path)
    print(f"Total rows: {len(data_handler)}")

    # Create Journal instance first
    journaling = Journal("results1", "energy_stats")
    journaling._get_data_stats(data_handler.stats)
    print(len(data_handler))
    print(data_handler[1000]["datetime"].weekday())







