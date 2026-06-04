import io
import requests
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
response.raise_for_status()

df = pd.read_html(io.StringIO(response.text))[0]
tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()

pd.Series(tickers).to_csv("sp500_tickers.csv", index=False, header=["ticker"])
