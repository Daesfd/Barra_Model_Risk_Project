import yfinance as yf
import pandas as pd
import numpy as np
import requests
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import time
import re

tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOG', 'TSLA', 'BRK-B', 'JPM',
           'ORCL', 'LLY', 'NFLX', 'MA', 'XOM', 'INTC']

headers = {
    "User-Agent": "Edson Ishizu Junior Financial Data Scrap (edsonicizojunior@gmail.com)"
}


def get_shares_outstanding(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {
        "User-Agent": "Edson Ishizu Junior Financial Data Scrap (edsonicizojunior@gmail.com)"
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        dei_facts = data["facts"]["dei"]

        # Primeiro tenta obter as ações em circulação (shares)
        if "EntityCommonStockSharesOutstanding" in dei_facts:
            facts = dei_facts["EntityCommonStockSharesOutstanding"]["units"]["shares"]
            df = pd.DataFrame(facts)
            df['Date'] = pd.to_datetime(df['end'])
            df = df[['Date', 'val']]
            df = df.sort_values('Date')
            df['Source'] = 'SharesOutstanding'
            return df

        # Se não houver, tenta pegar o valor de mercado das ações públicas
        elif "EntityPublicFloat" in dei_facts:
            facts = dei_facts["EntityPublicFloat"]["units"]["USD"]
            df = pd.DataFrame(facts)
            df['Date'] = pd.to_datetime(df['end'])
            df = df[['Date', 'val']]
            df = df.sort_values('Date')
            df['Source'] = 'PublicFloat'
            return df

        else:
            print(f"CIK {cik}: Nenhum dos dados disponíveis (Shares ou Public Float)")
            return pd.DataFrame(columns=['Date', 'val'])

    except Exception as e:
        print(f"Erro ao processar o CIK {cik}: {e}")
        return pd.DataFrame(columns=['Date', 'val'])


def calculate_marketcap(ticker, cik):
    df_shares = get_shares_outstanding(cik)
    
    if df_shares.empty:
        print(f"Sem dados de ações para {ticker}")
        return None

    try:
        df_price = yf.Ticker(ticker).history(start="2020-01-01", end="2025-01-01", interval="1d")
        df_price = df_price.reset_index()
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
        df_price = df_price[['Date', 'Close']]

        df_merged = pd.merge_asof(df_price.sort_values('Date'),
                                  df_shares.sort_values('Date'),
                                  on='Date', direction='backward')

        df_merged['Ticker'] = ticker
        df_merged['CIK'] = cik

        # Cálculo de marketcap depende da origem dos dados
        if df_merged['Source'].iloc[0] == 'SharesOutstanding':
            df_merged['MarketCap'] = df_merged['Close'] * df_merged['val']
        elif df_merged['Source'].iloc[0] == 'PublicFloat':
            # Nesse caso o "val" já é o valor em dólares
            df_merged['MarketCap'] = df_merged['val']

        return df_merged

    except Exception as e:
        print(f"Erro ao processar o ticker {ticker}: {e}")
        return None


def get_cik_from_ticker(ticker):
    # consulta a página "getcompany" do EDGAR
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        return None
    text = resp.text
    # procura por "CIK: 0000123456" ou "CIK&nbsp;#: 0000123456"
    m = re.search(r"CIK[#\s\:]*\s*0*([0-9]{1,10})", text, re.IGNORECASE)
    if m:
        cik = m.group(1).lstrip("0") or "0"
        return cik
    # alternativa: buscar pattern "CIK=0000123456"
    m2 = re.search(r"CIK=(\d{1,10})", text)
    if m2:
        return m2.group(1).lstrip("0") or "0"
    return None

# Com o nome dos tickers acima, obtém-se os códigos de cada ticker e coloca-os em df_cik.
results = {}
for t in tickers:
    cik = get_cik_from_ticker(t)
    results[t] = cik
    print(f"{t} -> {cik}")
    time.sleep(0.2)  # respeitar o servidor

# Exibir com zero-padding 10 dígitos (útil para usar em /submissions/CIK#########.json)
data = []
for t, cik in results.items():
    if cik:
        cik_10 = cik.zfill(10)
        data.append({"Ticker": t, "CIK": cik_10})

# Criar DataFrame
df_cik = pd.DataFrame(data)

# Com o cik de cada ticker, calcula-se o marketcap, usando dados de preços dos ativos e o número de ações
# que cada ativo teve até a data da consulta.

marketcap_data = []

for _, row in df_cik.iterrows():
    ticker = row['Ticker']
    cik = row['CIK']
    df_result = calculate_marketcap(ticker, cik)
    if df_result is not None:
        marketcap_data.append(df_result)

# Junta tudo em um DataFrame só
df_marketcaps = pd.concat(marketcap_data, ignore_index=True)
df_pivot = df_marketcaps.pivot_table(index='Date', columns='Ticker', values='MarketCap')
df_pivot.to_csv('./dados/dados_mcap.csv')

# Para a segunda parte do código, usando requests, pega o nome de todas as ações
# do S&P500 e, do mesmo modo que na primeira parte, obtem-se os dados de marketcap
# de todas as ações do mercado.

## Scrapping da wikipedia para pegar os nomes
url = "https://en.wikipedia.org/wiki/List_of_S&P_500_companies"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
resp = requests.get(url, headers=headers)
resp.raise_for_status()

tables = pd.read_html(resp.text)
df_sp500 = tables[0]

df_sp500 = df_sp500[['Symbol', 'Security']]
df_sp500 = df_sp500.rename(columns={'Symbol': 'Ticker', 'Security': 'Company'})

# Obtem-se o cik de cada ação.

results = {}
for t in df_sp500['Ticker']:
    cik = get_cik_from_ticker(t)
    results[t] = cik
    print(f"{t} -> {cik}")
    time.sleep(0.2)  # respeitar o servidor

# Exibir com zero-padding 10 dígitos (útil para usar em /submissions/CIK#########.json)
data = []
for t, cik in results.items():
    if cik:
        cik_10 = cik.zfill(10)
        data.append({"Ticker": t, "CIK": cik_10})

# Criar DataFrame
df_cik_total = pd.DataFrame(data)

# Obtem o valor de marketcap de cada ação usando o preço e o número de ações no mercado.

marketcap_data_total = []

for _, row in df_cik_total.iterrows():
    ticker = row['Ticker']
    cik = row['CIK']
    df_result = calculate_marketcap(ticker, cik)
    if df_result is not None:
        marketcap_data_total.append(df_result)

# Junta tudo em um DataFrame só
marketcap_data_total = pd.concat(marketcap_data_total, ignore_index=True)
df_total_marketcap = (
    marketcap_data_total
    .groupby('Date')['MarketCap']
    .sum()
    .reset_index()
    .rename(columns={'MarketCap': 'SP500_TotalMarketCap'})
)

df_total_marketcap
df_total_marketcap = df_total_marketcap.set_index('Date')

# Usa-se apenas as datas comuns entre os dfs para pegar o peso das ações escolhidas.

common_dates = df_pivot.index.intersection(df_total_marketcap.index)

df_pivot = df_pivot.loc[common_dates]
df_total_marketcap = df_total_marketcap.loc[common_dates]

df_weights = df_pivot.div(df_total_marketcap['SP500_TotalMarketCap'], axis=0)

df_weights.to_csv('./dados/df_pesos_historicos.csv')