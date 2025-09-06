import pandas as pd  
import numpy as np  
import yfinance as yf 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta 
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

currency_pairs = ['USDINR=X', 'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X'] 
start_date = '2023-01-01' 
end_date = '2024-12-31'  

print("=== Currency Exchange Rate Analysis ===")
print(f"Analyzing currency pairs: {currency_pairs}")
print(f"Period: {start_date} to {end_date}\n")

print("Step 1: Downloading currency data...")
currency_data = {} 

for pair in currency_pairs: 
    try:        
        ticker = yf.Ticker(pair)  
        data = ticker.history(start=start_date, end=end_date) 
        currency_data[pair] = data['Close'] 
        print(f"Downloaded {pair}: {len(data)} data points ")
    except Exception as e: 
        print(f"Error downloading {pair}: {e}")

print("\nStep 2: Creating combined dataset...")
df = pd.DataFrame(currency_data)  
df.fillna(method='ffill', inplace=True)  
print(f"Combined dataset shape: {df.shape}")  
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")  

print("\nStep 3: Calculating daily returns...")
returns = df.pct_change() * 100  
returns = returns.dropna() 
print("Daily returns calculated for all currency pairs")

print("\nStep 4: Calculating key statistics...")
stats = pd.DataFrame()  
for pair in currency_pairs:  
    if pair in df.columns: 
        pair_stats = {
            'Mean_Daily_Return': returns[pair].mean(), 
            'Std_Daily_Return': returns[pair].std(),   
            'Min_Daily_Return': returns[pair].min(),   
            'Max_Daily_Return': returns[pair].max(),   
            'Total_Return': ((df[pair].iloc[-1] / df[pair].iloc[0]) - 1) * 100,  
            'Volatility_Annualized': returns[pair].std() * np.sqrt(252) 
        }
        stats[pair] = pair_stats 
stats = stats.round(4) 
print("Key statistics calculated")

print("\nStep 5: Calculating moving averages for trend analysis...")
ma_short = 20 
ma_long = 50  
for pair in currency_pairs:
    if pair in df.columns:
        df[f'{pair}_MA_{ma_short}'] = df[pair].rolling(window=ma_short).mean() 
        df[f'{pair}_MA_{ma_long}'] = df[pair].rolling(window=ma_long).mean()  

print(f"Moving averages calculated: {ma_short}-day and {ma_long}-day")

print("\nStep 6: Generating trading signals...")
signals = pd.DataFrame(index=df.index)  

for pair in currency_pairs:
    if pair in df.columns:
        short_ma = df[f'{pair}_MA_{ma_short}'] 
        long_ma = df[f'{pair}_MA_{ma_long}']   
        
        signals[pair] = np.where(short_ma > long_ma, 1, -1) 
        signals[pair] = signals[pair].fillna(0)  
print("Trading signals generated based on moving average crossover")
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

print("\nKEY STATISTICS:")
print(stats.T) 

print(f"\nRECENT PRICES (Last 5 days):")
recent_data = df[currency_pairs].tail() 
print(recent_data)

print(f"\nCURRENT TRADING SIGNALS:")
current_signals = signals[currency_pairs].iloc[-1] 
for pair in currency_pairs:
    if pair in current_signals.index:
        signal = current_signals[pair]
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        print(f"{pair}: {signal_text}")
        
print("\nStep 8: Creating visualizations...")

for pair in currency_pairs:
    if pair in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[pair], label=f'{pair}', linewidth=1, alpha=0.7)

        if f'{pair}_MA_{ma_short}' in df.columns:
            plt.plot(df.index, df[f'{pair}_MA_{ma_short}'], 
                     label=f'{ma_short}-day MA', linewidth=2, alpha=0.8)
        if f'{pair}_MA_{ma_long}' in df.columns:
            plt.plot(df.index, df[f'{pair}_MA_{ma_long}'], 
                     label=f'{ma_long}-day MA', linewidth=2, alpha=0.8)

        plt.title(f'{pair} Exchange Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


print(f"\nStep 9: Volatility Analysis...")
plt.figure(figsize=(12, 6))  


for pair in currency_pairs:
    if pair in returns.columns:
        rolling_vol = returns[pair].rolling(window=30).std() * np.sqrt(252) 
        plt.plot(returns.index, rolling_vol, label=f'{pair} Volatility', linewidth=2) 

plt.title('30-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold') 
plt.xlabel('Date') 
plt.ylabel('Volatility (%)')  
plt.legend() 
plt.grid(True, alpha=0.3)  
plt.tight_layout()  
plt.show()  


print(f"\nStep 10: Correlation Analysis...")
correlation_matrix = returns.corr() 
print("Correlation Matrix of Daily Returns:")
print(correlation_matrix.round(3))  


plt.figure(figsize=(10, 8)) 
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto') 
plt.colorbar(label='Correlation') 
plt.title('Currency Pairs Correlation Matrix', fontsize=14, fontweight='bold') 


for i in range(len(currency_pairs)):
    for j in range(len(currency_pairs)):
        if currency_pairs[i] in correlation_matrix.index and currency_pairs[j] in correlation_matrix.columns:
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')  

plt.xticks(range(len(currency_pairs)), currency_pairs, rotation=45)  
plt.yticks(range(len(currency_pairs)), currency_pairs)  
plt.tight_layout()  
plt.show() 

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nKey Insights:")
print("1. Check the statistics table for risk-return profiles")
print("2. Higher volatility indicates more risk but potentially higher returns")
print("3. Trading signals based on moving average crossovers")
print("4. Correlation matrix shows how currencies move together")
print("5. Use this analysis as a foundation for developing trading strategies")


print(f"\nStep 11: Saving results...")
try:
    df.to_csv('currency_prices_and_ma.csv')  
    stats.T.to_csv('currency_statistics.csv') 
    returns.to_csv('currency_returns.csv') 
    signals.to_csv('trading_signals.csv') 
    correlation_matrix.to_csv('correlation_matrix.csv') 
    print("All results saved to CSV files")
except Exception as e: 
    print(f"Error saving files: {e}")

print("\nFiles saved:")
print("- currency_prices_and_ma.csv: Historical prices and moving averages")
print("- currency_statistics.csv: Key statistics for each currency pair")
print("- currency_returns.csv: Daily percentage returns")
print("- trading_signals.csv: Buy/sell signals based on moving averages")
print("- correlation_matrix.csv: Correlation between currency pairs")