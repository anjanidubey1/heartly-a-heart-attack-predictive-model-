import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_clean_data

def perform_eda(df: pd.DataFrame):
    """
    Performs Exploratory Data Analysis on the new medical dataset and
    saves the generated plots as images.

    Args:
        df (pd.DataFrame): The cleaned dataframe to analyze.
    """
    if df is None:
        print("DataFrame is empty. Skipping EDA.")
        return
        
    print("Starting Exploratory Data Analysis for Medical Dataset...")

    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df, palette='pastel')
    plt.title('Distribution of Heart Attack Result (1 = positive, 0 = negative)')
    plt.xlabel('Result')
    plt.ylabel('Patient Count')
    plt.savefig('target_distribution.png')
    plt.close()
    print("Saved target_distribution.png")

    
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Medical Features')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Saved correlation_heatmap.png")

    
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=df, x='heart_rate', hue='target', kde=True, bins=30, palette='bright')
    plt.title('Heart Rate Distribution by Result')
    plt.xlabel('Heart Rate')
    plt.ylabel('Frequency')
    plt.savefig('heart_rate_distribution.png')
    plt.close()
    print("Saved heart_rate_distribution.png")
    
    print("EDA complete. All plots have been saved.")


if __name__ == '__main__':
    
    df, _, _ = load_and_clean_data('Medicaldataset.csv')
   
    perform_eda(df)