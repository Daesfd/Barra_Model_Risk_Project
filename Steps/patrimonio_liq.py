import pandas as pd
import numpy as np
import yfinance as yf
import requests

import re
import time
from tqdm import tqdm

tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOG', 'TSLA', 'BRK-B', 'JPM',
           'ORCL', 'LLY', 'NFLX', 'MA', 'XOM', 'INTC']

headers = {
    "User-Agent": "Edson Ishizu Junior Financial Data Scrap (edsonicizojunior@gmail.com)"
}

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


def get_book_value(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {
        "User-Agent": "Edson Ishizu Junior Financial Data Scrap (edsonicizojunior@gmail.com)"
    }
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        gap_facts = data["facts"]["us-gaap"]

        possible_tags = [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "CommonStockholdersEquity"
        ]

        for tag in possible_tags:
            if tag in gap_facts:
                facts = gap_facts[tag]["units"]["USD"]
                df = pd.DataFrame(facts)
                df['Date'] = pd.to_datetime(df['end'])
                df = df[['Date', 'val']]
                df = df.sort_values('Date')
                df['Source'] = tag
                return df

        print(f"CIK {cik}: Nenhuma tag de patrimônio líquido encontrada.")
        return pd.DataFrame(columns=['Date', 'val'])

    except Exception as e:
        print(f"Erro ao processar o CIK {cik}: {e}")
        return pd.DataFrame(columns=['Date', 'val'])


def get_all_book_values(tickers_ciks):
    df_list = []

    for _, row in tqdm(tickers_ciks.iterrows(), desc="Processando CIKs"):
        ticker = row['Ticker']
        cik = row['CIK']
        df_book = get_book_value(cik)
        if not df_book.empty:
            df_book['Ticker'] = ticker
            df_book['CIK'] = cik
            df_list.append(df_book)
    
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        df_all = df_all.sort_values(['Ticker', 'Date'])
        return df_all
    else:
        print("Nenhum dado retornado.")
        return pd.DataFrame(columns=['Date', 'val', 'Source', 'Ticker', 'CIK'])
        

# Dado os nomes dos tickers, pega o cik de cada um e armazena-os em um df.
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

# Pega os dados de patrimonio líquido de cada ticker escolhido
df_pl = get_all_book_values(df_cik)
df_book_pivot = df_pl.pivot_table(index='Date', columns='Ticker', values='val')
full_dates = pd.date_range(start='2019-01-01', end='2025-01-01', freq='D')
df_book_full = df_book_pivot.reindex(full_dates)  # reindexa para todas as datas
df_book_full = df_book_full.ffill()               # preenche NaN com o último valor válido
df_book_full.index.name = 'Date'                 # opcional, renomeia o índice

df_book_full = df_book_full.iloc[365:]
df_book_full.to_csv('./dados/dados_pl.csv')