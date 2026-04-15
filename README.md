# TCC — SVI e Volatilidade de Ações da B3

Trabalho de Conclusão de Curso em Economia.

**Questão de pesquisa:** O Search Volume Index (SVI) do Google Trends melhora a previsão de volatilidade de ações listadas na B3?

---

## Metodologia

O SVI é usado como variável exógena num modelo **GARCH-X(1,1)**, que compete contra um GARCH(1,1) padrão e um passeio aleatório. A avaliação é feita com a função de perda **QLIKE** e o teste formal de **Model Confidence Set (MCS)** de Hansen, Lunde & Nason (2011), em janela móvel out-of-sample de 3 anos.

| Modelo | Especificação |
|--------|--------------|
| M0 — Random Walk | σ²ₜ = RV_{t-1} |
| M1 — GARCH(1,1) | σ²ₜ = ω + α·ε²_{t-1} + β·σ²_{t-1} |
| M2 — GARCH-X(1,1) | σ²ₜ = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·ΔSVI_{t-1} |

---

## Estrutura do repositório

```
.
├── POC_TCC_Google_Trends_v2.ipynb   # Notebook principal
├── requirements.txt
└── data/
    ├── acoes-listadas-b3.csv         # Lista de tickers da B3
    ├── acoes_elegiveis_v3.csv        # Tickers que passaram nos filtros de liquidez/histórico
    ├── precos_semanais_v3.csv        # Preços e volatilidade realizada semanal por ticker
    ├── google_trends_svi.csv         # SVI semanal por ticker (Google Trends)
    ├── log_coleta_svi.csv            # Log da coleta do Google Trends
    ├── log_adf_svi.csv               # Resultados do teste ADF por ticker
    ├── base_final_tcc.csv            # Base combinada (preços + SVI), usada na modelagem
    ├── previsoes_consolidadas.csv    # Previsões out-of-sample dos três modelos
    ├── resumo_parametros.csv         # Parâmetros médios por ticker (α, β, γ, p-valor)
    ├── qlike_por_ticker.csv          # QLIKE médio por ticker e modelo
    └── mcs_resultado.csv             # Resultado do Model Confidence Set
```

---

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Abrir o notebook

```bash
jupyter notebook POC_TCC_Google_Trends_v2.ipynb
```

### 3. Rodar as células

- **Seção "Bases salvas"** — carrega todos os dados da pasta `data/`, sem necessidade de re-coletar
- **Seção "Coleta dos dados"** — re-executa a coleta do Yahoo Finance e Google Trends (demora ~1h devido ao rate limit da API do Trends)
- **Seção "Texto Principal"** — avalia os modelos nos cenários de filtro ≥ 70% e ≥ 50% de zeros no SVI
- **Seção "Apêndice"** — avalia sem filtro, com explicação das distorções

---

## Dados

Os dados de preços são coletados via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance).  
O SVI é coletado via [pytrends](https://github.com/GeneralMills/pytrends) (Google Trends), usando o ticker como termo de busca, geo=BR, frequência semanal.

A volatilidade realizada é calculada como RV = Σ rᵢ² (soma dos retornos logarítmicos diários ao quadrado dentro de cada semana).

---

## Dependências principais

- `arch` — estimação e previsão GARCH/GARCH-X
- `statsmodels` — teste ADF de estacionariedade
- `pytrends` — coleta do Google Trends
- `yfinance` — dados financeiros do Yahoo Finance
