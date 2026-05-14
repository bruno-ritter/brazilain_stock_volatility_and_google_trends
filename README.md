# TCC — SVI e Volatilidade de Ações da B3

Trabalho de Conclusão de Curso em Economia.

**Questão de pesquisa:** O Search Volume Index (SVI) do Google Trends melhora a previsão de volatilidade de ações listadas na B3?

---

## Metodologia

### Variáveis

**Volatilidade Realizada (RV)**  
Proxy semanal de volatilidade calculada como a soma dos quadrados dos retornos logarítmicos diários dentro de cada semana:

$$RV_t = \sum_{d \in t} r_d^2, \quad r_d = \ln\left(\frac{P_d}{P_{d-1}}\right)$$

**Search Volume Index (ΔSVI)**  
O SVI é o índice semanal normalizado (0–100) do Google Trends para o termo de busca correspondente ao ticker, com geo=BR. Como a série em nível apresenta raiz unitária em parte dos tickers, utiliza-se a **primeira diferença** (ΔSVI) como variável exógena, confirmada estacionária pelo teste ADF em 323 dos 324 tickers elegíveis.

---

### Modelos Competidores

| # | Modelo | Equação da variância condicional |
|---|--------|----------------------------------|
| M0 | Random Walk | $\hat{\sigma}^2_t = RV_{t-1}$ |
| M1 | GARCH(1,1) | $\sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1} + \beta\,\sigma^2_{t-1}$ |
| M3 | GJR-GARCH(1,1) | $\sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1} + \gamma_{GJR}\,\varepsilon^2_{t-1}\,\mathbb{I}(\varepsilon_{t-1}<0) + \beta\,\sigma^2_{t-1}$ |
| M2 | GJR-GARCH-X(1,1) | $\sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1} + \gamma_{GJR}\,\varepsilon^2_{t-1}\,\mathbb{I}(\varepsilon_{t-1}<0) + \beta\,\sigma^2_{t-1} + \delta\,\Delta\text{SVI}_{t-1}$ |

M0 é o benchmark ingênuo. M1 é o modelo padrão de volatilidade condicional. M3 adiciona o **efeito de assimetria (GJR)**: choques negativos podem impactar a volatilidade mais que choques positivos do mesmo módulo. M2 estende M3 com o **SVI na equação da variância** — o coeficiente $\delta$ captura se o interesse de busca eleva a volatilidade futura.

Todos os modelos GARCH (M1, M2, M3) usam **inovações Student-t** — escolha imposta pela biblioteca `gjr-garch-x` (única que estima exógenos na equação da variância) e replicada no M1 por consistência distribucional.

**Decomposição do efeito do SVI:** a comparação **M3 vs M2** isola o ganho **puro** do SVI condicional à assimetria já modelada; **M1 vs M3** isola o ganho da assimetria; **M1 vs M2** mede o efeito conjunto. Sem o M3 não seria possível atribuir corretamente as melhorias do M2 sobre o M1.

---

### Estimação em Janela Móvel (*Rolling Window*)

Os modelos são estimados e avaliados **fora da amostra** (*out-of-sample*) por janela móvel de tamanho fixo:

- **Janela de treino:** 156 semanas (3 anos)
- **Horizonte de previsão:** 1 passo à frente (semana seguinte)
- **Procedimento:** a cada semana *t*, estima-se o modelo nas 156 observações anteriores e gera-se uma previsão para *t*; a janela avança uma semana e repete-se o processo

Isso evita *look-ahead bias* e reproduz o ambiente real de previsão.

**Implementação:**
- **M1** é estimado via `arch.arch_model` com `vol="GARCH"`, `dist="t"` e `mean="Constant"`. O forecast 1-passo usa `fit.forecast(horizon=1)`.
- **M2 e M3** são estimados via [`gjr-garch-x`](https://github.com/studiofarzulla/gjr-garch-x), única biblioteca Python que aceita regressores na equação da variância. M3 chama `estimate_gjr_garch_x(returns, None)` (sem exógeno); M2 chama com `exog_vars=pd.DataFrame({"svi_diff": svi_diff.shift(1)})` — o **shift(1)** garante que o que entra na variância no tempo *t* é $\Delta\text{SVI}_{t-1}$, evitando look-ahead.
- A biblioteca `gjr-garch-x` **não tem método `.forecast()`**. A previsão 1-passo é calculada manualmente pela equação fechada: $\hat{\sigma}^2_t = \omega + \alpha\hat{\varepsilon}^2_{t-1} + \gamma_{GJR}\hat{\varepsilon}^2_{t-1}\mathbb{I}(\hat{\varepsilon}_{t-1}<0) + \beta\hat{\sigma}^2_{t-1} + \delta\,\Delta\text{SVI}_{t-1}$, usando os parâmetros estimados na janela e os últimos valores de resíduo e variância condicional (`res.residuals.iloc[-1]`, `res.volatility.iloc[-1]**2`).
- Retornos são multiplicados por 100 antes da estimação (estabilidade numérica) e a previsão de variância é dividida por 10.000 antes de gravar no CSV.

---

### Função de Perda — QLIKE

A avaliação pontual utiliza a função de perda **QLIKE** (Quasi-Likelihood):

$$\text{QLIKE}(RV_t, \hat{\sigma}^2_t) = \frac{RV_t}{\hat{\sigma}^2_t} - \ln\frac{RV_t}{\hat{\sigma}^2_t} - 1$$

QLIKE é assimétrica e penaliza mais fortemente previsões subestimadas, sendo considerada robusta para comparação de modelos de volatilidade (Patton, 2011). O valor médio de QLIKE por ticker e por modelo é armazenado em `data/qlike_por_ticker.csv`.

---

### Inferência Formal — Model Confidence Set (MCS)

A comparação entre os quatro modelos é feita pelo **Model Confidence Set** de Hansen, Lunde & Nason (2011), implementado via `arch.bootstrap.MCS`. O MCS é um teste de hipótese sequencial que, a partir de um conjunto inicial de modelos, elimina iterativamente o pior modelo enquanto ele for significativamente inferior aos demais (α = 10%). O conjunto final contém todos os modelos que *não podem ser rejeitados* como igualmente bons ao melhor.

Resultado salvo em `data/mcs_resultado.csv`.

---

### Filtro de Zeros no SVI

Tickers com alta proporção de semanas com SVI = 0 indicam que o Google Trends não registrou buscas suficientes para o papel — o sinal é essencialmente ruído. Isso causa dois problemas:

1. **Matriz singular:** ΔSVI sem variação impossibilita a estimação do GARCH-X (erro numérico na otimização)
2. **Viés favorável ao M2:** previsões baseadas em ruído puro podem, por acaso, parecer melhores num subconjunto pequeno de janelas

O repositório avalia **quatro estratégias de filtro** comparativamente (diagnóstico salvo em `data/diagnostico_filtros.txt`):

| Estratégia | Descrição |
|------------|-----------|
| Apenas singular | Remove somente tickers que falharam com erro de matriz singular |
| ≥ 70% zeros | Remove tickers com 70% ou mais das semanas com SVI = 0 |
| ≥ 50% zeros | Remove tickers com 50% ou mais das semanas com SVI = 0 |
| Sem filtro | Inclui todos os tickers (inclusive os com ruído alto) |

A análise principal usa o limiar de **≥ 50% de zeros** como critério de elegibilidade.

---

## Estrutura do repositório

```
.
├── 01_coleta_dados.ipynb        # Coleta e preparo dos dados
├── 02_modelos.ipynb             # Estimação, previsão e diagnóstico de filtros
├── 03_analise_resultados.ipynb  # Análise exploratória (filtro ≥50% zeros)
├── requirements.txt
└── data/
    ├── acoes-listadas-b3.csv        # Lista de tickers da B3
    ├── acoes_elegiveis.csv          # Tickers que passaram nos filtros de liquidez/histórico
    ├── precos_semanais.csv          # Preços e volatilidade realizada semanal por ticker
    ├── google_trends_svi.csv        # SVI semanal por ticker (Google Trends)
    ├── log_coleta_svi.csv           # Log da coleta do Google Trends
    ├── log_adf_svi.csv              # Resultados do teste ADF por ticker
    ├── base_final_tcc.csv           # Base combinada (preços + SVI), usada na modelagem
    ├── previsoes_consolidadas.csv   # Previsões out-of-sample dos 4 modelos (M0, M1, M2, M3) + parâmetros estimados por janela
    ├── resumo_parametros.csv        # Parâmetros médios por ticker (α, β, γ_GJR, δ_SVI, p-valores, persistência)
    ├── qlike_por_ticker.csv         # QLIKE médio por ticker e modelo
    ├── mcs_resultado.csv            # Resultado do Model Confidence Set
    └── diagnostico_filtros.txt      # Comparação das quatro estratégias de filtro
```

---

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Rodar os notebooks em ordem

```bash
jupyter notebook
```

| Notebook | O que faz | Tempo estimado |
|----------|-----------|----------------|
| `01_coleta_dados.ipynb` | Coleta preços (Yahoo Finance), calcula RV, coleta SVI (Google Trends), testa ADF, salva CSVs | ~1h (rate limit do Google Trends) |
| `02_modelos.ipynb` | Estima M0/M1/M2/M3 em janela móvel, calcula QLIKE, roda MCS, compara estratégias de filtro | ~1–2h (M2 e M3 usam otimizador SLSQP da `scipy`, mais lento que `arch`) |
| `03_analise_resultados.ipynb` | Análise exploratória dos resultados (filtro ≥50%) | instantâneo |

> **Atalho:** os dados já estão em `data/`. Para pular a coleta, basta executar a seção **"Bases salvas"** do `02_modelos.ipynb` diretamente.

---

## Dados

| Fonte | Biblioteca | Detalhes |
|-------|-----------|----------|
| Yahoo Finance | `yfinance` | Preços diários de fechamento ajustado; tickers B3 com sufixo `.SA` |
| Google Trends | `pytrends` | SVI semanal, geo=BR, termo de busca = ticker sem `.SA`; pausa de 12s entre requisições |

**Filtros de elegibilidade aplicados na coleta:**
- Volume financeiro médio diário ≥ R$ 1 milhão
- Pelo menos 5 anos de histórico de preços disponíveis
- Série de SVI coletada com sucesso

---

## Dependências

```
arch          # estimação GARCH (M1) e Model Confidence Set
gjr-garch-x   # estimação GJR-GARCH com exógeno na variância (M2, M3)
statsmodels   # teste ADF de estacionariedade
pytrends      # coleta do Google Trends
yfinance      # dados financeiros do Yahoo Finance
pandas
numpy
matplotlib
seaborn
tqdm
```

---

## Referências

- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
