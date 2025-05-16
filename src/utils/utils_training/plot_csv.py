import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


def plot_results(csv_path, formatted_time):
    """
    Create plots from the CSV data
    """
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return
        
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(csv_path), "plots", formatted_time)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # 1. Energy plot (P_l, P_g, P_batt)
    plt.figure(figsize=(12, 6))
    plt.plot(df['P_l'], label='Load Power (P_l)')
    plt.plot(df['P_g'], label='Generation Power (P_g)')
    plt.plot(df['P_batt'], label='Battery Power (P_batt)')
    plt.xlabel('Steps')
    plt.ylabel('Power (kW)')
    plt.title('Power Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'energy_power.png'))
    
    # 2. Energy exchange (E_sell, E_buy)
    plt.figure(figsize=(12, 6))
    plt.plot(df['E_sell'], label='Energy Sold')
    plt.plot(df['E_buy'], label='Energy Bought')
    plt.xlabel('Steps')
    plt.ylabel('Energy (kWh)')
    plt.title('Energy Exchange with Grid')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'energy_exchange.png'))

    plt.figure(figsize=(12, 6))
    plt.plot(df['Cost_with_batt'], label='Cost with Battery')
    plt.plot(df['cost_without_batt'], label='Cost without Battery')
    
    
    # 3. Economic data
    plt.figure(figsize=(12, 6))
    # plt.plot(df['Cost_with_batt'], label='Cost with Battery')
    # plt.plot(df['cost_without_batt'], label='Cost without Battery')
    plt.plot(df['batt_wear_cost'], label='Battery Wear Cost')
    plt.plot(df['reward'], label='Reward')
    # plt.plot(df['accumulated_cost'], label='Accumulated Cost')
    plt.xlabel('Steps')
    plt.ylabel('Cost (€)')
    plt.title('Economic Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'economic.png'))

    # 3.1. Economic data
    plt.figure(figsize=(12, 6))
    plt.plot(df['accumulated_cost'], label='Accumulated Cost')
    plt.plot(df['accumulated_battery_wear_cost'], label='Accumulated Battery Wear Cost')
    plt.plot(df['accumulated_reward'], label='Accumulated Reward')
    plt.xlabel('Steps')
    plt.ylabel('Cost (€)')
    plt.title('Accumulated Cost and Battery Wear Cost')
    plt.legend()
    
    # 4. SOC (State of Charge)
    plt.figure(figsize=(12, 6))
    plt.plot(df['SoC'], label='State of Charge')
    plt.xlabel('Steps')
    plt.ylabel('SOC (%)')
    plt.title('Battery State of Charge')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'soc.png'))
    
    # 5. Current
    plt.figure(figsize=(12, 6))
    plt.plot(df['current'], label='Battery Current')
    plt.xlabel('Steps')
    plt.ylabel('Current (A)')
    plt.title('Battery Current')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'current.png'))
    
    print(f"Plots saved to {plots_dir}")