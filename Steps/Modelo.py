import pandas as pd
import numpy as np
import yfinance as yf
import requests

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm

def classificar_bm(row):
    # Calcula os quantis do B/M do dia
    low = row.quantile(0.3)
    high = row.quantile(0.7)
    
    # Classifica cada ativo
    return pd.Series(np.where(
        row <= low, 'Low',
        np.where(row >= high, 'High', 'Medium')
    ), index=row.index)


def calc_hml_simple(row_ret, row_class):
    # Alinhar índices antes de filtrar
    row_ret, row_class = row_ret.align(row_class, join='inner')
    
    high = row_ret[row_class == 'High'].mean()
    low = row_ret[row_class == 'Low'].mean()
    
    return high - low


def classificar_mom(row):
    low = row.quantile(0.3)
    high = row.quantile(0.7)
    return pd.Series(np.where(
        row <= low, 'Loser',
        np.where(row >= high, 'Winner', 'Neutral')
    ), index=row.index)


def calc_mom(row_ret, row_class):
    row_ret, row_class = row_ret.align(row_class, join='inner')
    win = row_ret[row_class == 'Winner'].mean()
    lose = row_ret[row_class == 'Loser'].mean()
    return win - lose


# Obtenção dos dados que serão usados no modelo
tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOG', 'TSLA', 'BRK-B', 'JPM',
           'ORCL', 'LLY', 'NFLX', 'MA', 'XOM', 'INTC']
ind = "^GSPC"

df_modelo = pd.DataFrame()

df_pesos = pd.read_csv('./dados/df_pesos_historicos.csv', index_col='Date')
df_pesos = df_pesos.div(df_pesos.sum(axis=1), axis=0)
df_pesos = df_pesos.iloc[1:]

retornos = dados.pct_change().dropna()
retornos['^GSPC'] = retornos['^GSPC']- 0.0001

df_pl = pd.read_csv('./dados/dados_pl.csv', index_col='Date')
df_pl.index = pd.to_datetime(df_pl.index)
df_pl = df_pl[df_pl.index.isin(retornos.index)]

dados_mcap = pd.read_csv('./dados/dados_mcap.csv', index_col='Date')
dados_mcap.index = pd.to_datetime(dados_mcap.index)
dados_mcap = dados_mcap.iloc[1:]

dados = yf.download(tickers + [ind], start="2020-01-01", end="2025-01-01")['Close']
dados = dados.iloc[1:]

# Fator SmallxBig

## O fator é escolhido pelo marketcap das ações, sendo o small menor que a mediana
## e o big, maior que a mediana
## Após, é calculado o retorno de cada classificação e coloca-os, subtraindo, em uma lista

smb = []
dates = dados_mcap.index

for date in dates:
    mkt_caps = dados_mcap.loc[date].dropna()
    rets = retornos.loc[date].dropna()

    small = mkt_caps[mkt_caps <= np.median(mkt_caps.values)].index
    big = mkt_caps[mkt_caps >= np.median(mkt_caps.values)].index

    small_ret = rets[small].mean()
    big_ret = rets[big].mean()

    smb.append(small_ret - big_ret)
smb_series = pd.Series(smb, index=dates, name="SMB")

# Cria um df usando dados de patrimônio líquido e marketcap, para conseguir
# o índice book-to-market.
# Assim, calcula-se o fator hml e transforma-os em uma série.

df_bm = df_pl / dados_mcap

df_class = df_bm.apply(classificar_bm, axis=1)

HML = [calc_hml_simple(retornos.loc[date], df_class.loc[date]) for date in retornos.index]
HML = pd.Series(HML, index=retornos.index, name='HML')

# Cria-se um df com os dados de momentum de cada ação para obter o fator momentum.

dados_mom = yf.download(tickers, start="2019-01-01", end="2025-01-01")['Close']
retornos_mom = dados_mom.pct_change().dropna()

retornos_mom = retornos_mom.sort_index()
retornos_mom.index = pd.to_datetime(retornos_mom.index)

# Calcular retorno acumulado de 252 dias anteriores (excluindo o mês atual)
ret_acum = (1 + retornos_mom).shift(1).rolling(window=252, min_periods=252).apply(lambda x: x.prod() - 1)
ret_acum = ret_acum.iloc[252:]


MOM = [calc_mom(retornos.loc[date], df_class_mom.loc[date]) for date in retornos.index]
MOM = pd.Series(MOM, index=retornos.index, name='MOM')

# Cria um df apenas com os fatores, para uma maior facilidade de manejamento dos dados

df_fatores = pd.concat([retornos['^GSPC'], smb_series, HML, MOM], axis=1)

df_modelo = retornos.join(df_fatores, how='inner')

fatores = df_modelo[['^GSPC', 'SMB', 'HML', 'MOM']]
fatores = sm.add_constant(fatores)

# Faz regressões para cada ticker, considerando os fatores de mercado
# Após isso, pega os betas de cada fator, desconsiderando o beta1

resultados = {}
betas = {}
for ticker in tickers:
    y = df_modelo[ticker]
    modelo = sm.OLS(y, fatores, missing='drop').fit()
    resultados[ticker] = modelo


betas = pd.DataFrame({t: resultados[t].params for t in tickers}).T
betas = betas.drop(columns='const')

for ativo in tickers:
    print(f"=== {ativo} ===")
    print(resultados[ativo].summary())
    print()

# Nesta parte, para obter o valor do risco de portfólio, é necessário obter:
# A covariância dos fatores, a variância dos resíduos das regressões e
# os betas das regressões.
# Assim, consegue-se obter o sigma_total, que, com o peso normalizado de cada ação, obtem-se
# o risco do portfólio.

correlacoes = df_fatores.corr()

sigma_f = df_fatores.cov().values

resid = pd.DataFrame({t: resultados[t].resid for t in tickers})
specific_var = np.diag(resid.var())

sig_total = betas.values @ sigma_f @ betas.values.T + specific_var

w = df_pesos.iloc[-1:].to_numpy().flatten()

port_risk = np.sqrt(w.T @ sig_total @ w)

# Por fim, com uma confiança de 95%, obtem-se o VaR do portfólio.

conf = 0.95
VaR = norm.ppf(1-conf) * port_risk

retornos = retornos.drop(columns='^GSPC')
ret_port = (retornos @ w).to_frame('retorno_port')

df_modelo.to_csv('./dados_modelo.csv')