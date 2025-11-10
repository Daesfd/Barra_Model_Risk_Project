# Modelo de Fatores e Risco de Portfólio (Python)

Este projeto tem como objetivo construir um modelo multifatorial de risco para portfólios de ações do mercado norte-americano, utilizando dados públicos da **SEC (EDGAR)** e da **Yahoo Finance**.  
O pipeline coleta, processa e integra informações de **valor de mercado**, **patrimônio líquido**, e **retornos históricos** para calcular fatores como **SMB**, **HML** e **MOM**, além de estimar o **VaR (Value at Risk)** do portfólio.

---
## Ordem

python patrimonio_liq.py
python mcap.py
python Modelo.py

## Funcionalidades dos Scripts

### `mcap.py`
- Faz o *scraping* dos **CIKs** (identificadores oficiais da SEC) para os tickers selecionados.  
- Coleta o número de ações em circulação (*shares outstanding*) e o preço diário das ações via **SEC API** e **Yahoo Finance**.  
- Calcula o **market capitalization (valor de mercado)** de cada empresa e do S&P 500 total.  
- Gera os arquivos:
  - `dados_mcap.csv` → histórico de market cap das ações.
  - `df_pesos_historicos.csv` → peso de cada ação no portfólio em relação ao S&P500.

---

### `patrimonio_liq.py`
- Obtém o **patrimônio líquido (book value)** das empresas a partir das demonstrações financeiras enviadas à **SEC (EDGAR)**.  
- Identifica as tags contábeis adequadas (`StockholdersEquity`, `CommonStockholdersEquity`, etc.) e normaliza as datas.  
- Gera o arquivo:
  - `dados_pl.csv` → série temporal do patrimônio líquido das empresas analisadas.

---

### `Modelo.py`
- Utiliza os dados de **market cap** e **patrimônio líquido** para construir fatores de risco:
  - **SMB (Small Minus Big)** — diferença de retorno entre empresas pequenas e grandes.  
  - **HML (High Minus Low)** — diferença de retorno entre empresas com alto e baixo índice book-to-market.  
  - **MOM (Momentum)** — diferença de retorno entre ações vencedoras e perdedoras no passado recente.
- Regressões lineares múltiplas com **OLS (Statsmodels)** para estimar os betas fatoriais de cada ativo.  
- Calcula:
  - **Matriz de covariância total** (com fatores + resíduos),
  - **Risco total do portfólio**,  
  - **Value at Risk (VaR)** com 95% de confiança.
- Salva os resultados em:
  - `dados_modelo.csv`

---

## 📊 Principais Saídas

| Arquivo | Descrição |
|----------|------------|
| `dados_mcap.csv` | Valor de mercado diário das ações analisadas |
| `dados_pl.csv` | Patrimônio líquido (book value) das empresas |
| `df_pesos_historicos.csv` | Pesos relativos no portfólio |
| `dados_modelo.csv` | Dados finais com fatores e retornos para regressão |

---

## 🧠 Principais Bibliotecas
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `statsmodels`, `scipy`, `pypfopt`
- `yfinance`, `requests`, `re`, `tqdm`
