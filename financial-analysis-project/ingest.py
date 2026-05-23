import yfinance as yf
import pandas as pd

data = yf.download('AAPL', start='2010-01-01', end='2022-02-26')
pd.DataFrame(data).to_csv('./data/aapl.csv', index=True)