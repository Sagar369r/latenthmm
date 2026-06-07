import argparse

def download_data():
    print("=" * 60)
    print("Latent Diffusion-HMM: Data Acquisition")
    print("=" * 60)
    print("This engine uses Dukascopy 1-minute bid data as the golden source.")
    print("Yahoo Finance data is explicitly excluded due to tick-volume inaccuracy.\n")
    print("To download the required data, use the dukascopy-node CLI tool:\n")
    print("  npx dukascopy-node -i xauusd -timeframe m1 -from 2022-01-01 -to 2024-12-31 -f csv -dir data")
    print("  npx dukascopy-node -i eurusd -timeframe m1 -from 2022-01-01 -to 2024-12-31 -f csv -dir data")
    print("\nOr using the Python dukascopy package:")
    print("  pip install dukascopy")
    print("  from dukascopy import fetch_ohlcv")
    print("  df = fetch_ohlcv('XAUUSD', '1m', '2022-01-01', '2024-12-31')")
    print("  df.to_csv('data/xauusd_2022_2024_1m.csv')")
    print("=" * 60)
    
if __name__ == "__main__":
    download_data()
