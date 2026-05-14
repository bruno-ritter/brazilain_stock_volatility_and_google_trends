# ==============================================================================
# 02_modelos.R
# TCC — Previsão de Volatilidade com SVI (Google Trends)
# Modelos: M0 (passeio aleatório), M1 (GARCH 1,1), M2 (GARCH-X 1,1)
# Janela móvel: 156 semanas (52 × 3); avaliação OOS via QLIKE + MCS
# ==============================================================================

# ── 0. Bibliotecas e constantes ───────────────────────────────────────────────
suppressPackageStartupMessages({
  library(rugarch)
  library(MCS)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lubridate)
  library(parallel)
})

CAMINHO      <- "./data/"
JANELA       <- 52L * 3L       # 156 semanas de treino
MIN_OBS      <- JANELA + 1L    # mínimo para produzir ≥1 previsão
USE_PARALLEL <- TRUE
N_CORES      <- max(1L, parallel::detectCores() - 1L)
set.seed(42)

cat("=== 02_modelos.R ===\n")
cat(sprintf("Janela de treino : %d semanas\n", JANELA))
cat(sprintf("Núcleos usados   : %d\n", if (USE_PARALLEL) N_CORES else 1L))

# ── 1. Carrega dados ──────────────────────────────────────────────────────────
df_final <- read_csv(
  paste0(CAMINHO, "base_final_tcc.csv"),
  col_types = cols(semana = col_date(format = "%Y-%m-%d")),
  show_col_types = FALSE
)

df_trends <- read_csv(
  paste0(CAMINHO, "google_trends_svi.csv"),
  col_types = cols(semana = col_date(format = "%Y-%m-%d")),
  show_col_types = FALSE
)

cat(sprintf("\nbase_final_tcc   : %d linhas, %d tickers\n",
            nrow(df_final), n_distinct(df_final$ticker_b3)))

# Percentual de semanas com SVI = 0 por ticker (calculado do CSV de trends)
zeros_pct <- df_trends %>%
  group_by(ticker_b3) %>%
  summarise(pct_zeros = mean(svi == 0, na.rm = TRUE), .groups = "drop")

# ── 2. Funções auxiliares ─────────────────────────────────────────────────────

# M0: passeio aleatório — rv_pred = rv do período anterior
calc_m0 <- function(df_ticker) {
  df_ticker %>%
    arrange(semana) %>%
    transmute(semana, rv_pred_m0 = lag(rv))
}

# QLIKE: rv/pred - log(rv/pred) - 1  (NA onde condições não satisfeitas)
calc_qlike <- function(rv, pred) {
  mask   <- !is.na(rv) & !is.na(pred) & rv > 0 & pred > 0
  result <- rep(NA_real_, length(rv))
  r            <- rv[mask] / pred[mask]
  result[mask] <- r - log(r) - 1
  result
}

# M1: GARCH(1,1) rolling — retorna tibble com 1 linha por janela OOS
calc_m1 <- function(df_ticker, janela = JANELA) {
  df_t <- df_ticker %>% arrange(semana)
  n    <- nrow(df_t)
  if (n <= janela) return(tibble())

  spec_m1 <- ugarchspec(
    variance.model    = list(model = "sGARCH", garchOrder = c(1L, 1L)),
    mean.model        = list(armaOrder = c(0L, 0L), include.mean = TRUE),
    distribution.model = "norm"
  )

  resultados <- vector("list", n - janela)

  for (t in seq(janela + 1L, n)) {
    train_ret <- df_t$ret_semanal[(t - janela):(t - 1L)] * 100

    linha <- list(
      semana      = df_t$semana[t],
      rv          = df_t$rv[t],
      ret_semanal = df_t$ret_semanal[t],
      rv_pred_m1  = NA_real_,
      omega       = NA_real_,
      alpha       = NA_real_,
      beta        = NA_real_
    )

    fit_res <- tryCatch({
      fit <- ugarchfit(spec_m1, data = train_ret, solver = "hybrid",
                       fit.control = list(stationarity = 1),
                       solver.control = list(trace = 0))
      fc  <- ugarchforecast(fit, n.ahead = 1L)
      list(ok = TRUE, fit = fit, fc = fc)
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))

    if (fit_res$ok) {
      mc              <- fit_res$fit@fit$matcoef
      linha$rv_pred_m1 <- as.numeric(sigma(fit_res$fc))^2 / 10000
      linha$omega      <- mc["omega",  1L]
      linha$alpha      <- mc["alpha1", 1L]
      linha$beta       <- mc["beta1",  1L]
    }

    resultados[[t - janela]] <- linha
  }

  bind_rows(resultados)
}

# M2: GARCH-X(1,1) com svi_diff como regressor externo na variância
calc_m2 <- function(df_ticker, janela = JANELA) {
  df_t <- df_ticker %>%
    arrange(semana) %>%
    filter(!is.na(svi_diff))   # remove NAs gerados pelo diff() (primeira linha)

  n <- nrow(df_t)
  if (n <= janela) return(tibble())

  resultados <- vector("list", n - janela)

  for (t in seq(janela + 1L, n)) {
    train_ret <- df_t$ret_semanal[(t - janela):(t - 1L)] * 100
    train_x   <- matrix(df_t$svi_diff[(t - janela):(t - 1L)], ncol = 1L)
    # Espelha Python: usa svi_diff[t] (contemporâneo ao período previsto)
    fc_x      <- matrix(df_t$svi_diff[t], nrow = 1L, ncol = 1L)

    linha <- list(
      semana     = df_t$semana[t],
      rv         = df_t$rv[t],
      rv_pred_m2 = NA_real_,
      alpha_m2   = NA_real_,
      beta_m2    = NA_real_,
      gamma      = NA_real_,
      gamma_pval = NA_real_,
      status_m2  = NA_character_
    )

    # spec_m2 deve ser criado dentro do loop (external.regressors varia com t)
    fit_res <- tryCatch({
      spec_m2 <- ugarchspec(
        variance.model    = list(model = "sGARCH", garchOrder = c(1L, 1L),
                                 external.regressors = train_x),
        mean.model        = list(armaOrder = c(0L, 0L), include.mean = TRUE),
        distribution.model = "norm"
      )
      fit <- ugarchfit(spec_m2, data = train_ret, solver = "hybrid",
                       fit.control = list(stationarity = 1),
                       solver.control = list(trace = 0))
      fc  <- ugarchforecast(fit, n.ahead = 1L,
                            external.forecasts = list(vxreg = fc_x))
      list(ok = TRUE, fit = fit, fc = fc)
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))

    if (fit_res$ok) {
      mc               <- fit_res$fit@fit$matcoef
      linha$rv_pred_m2  <- as.numeric(sigma(fit_res$fc))^2 / 10000
      linha$alpha_m2    <- mc["alpha1", 1L]
      linha$beta_m2     <- mc["beta1",  1L]
      linha$gamma       <- mc["vxreg1", 1L]
      linha$gamma_pval  <- mc["vxreg1", 4L]
      linha$status_m2   <- "ok"
    } else {
      linha$status_m2 <- fit_res$msg
    }

    resultados[[t - janela]] <- linha
  }

  bind_rows(resultados)
}

# ── 3. Loop principal por ticker ──────────────────────────────────────────────

processar_ticker <- function(i) {
  ticker <- tickers[i]
  n_total <- length(tickers)
  cat(sprintf("  [%d/%d] %s\n", i, n_total, ticker))
  grupo <- df_final %>% filter(ticker_b3 == ticker) %>% arrange(semana)
  n     <- nrow(grupo)

  if (n <= JANELA) return(NULL)

  semana_corte <- grupo$semana[JANELA + 1L]

  df_m0 <- calc_m0(grupo)
  df_m1 <- calc_m1(grupo)
  df_m2 <- calc_m2(grupo)

  if (nrow(df_m1) == 0L || nrow(df_m2) == 0L) return(NULL)

  # M1 é a base; M0 e M2 são left-joined por semana
  df_tick <- df_m1 %>%
    left_join(df_m0 %>% select(semana, rv_pred_m0), by = "semana") %>%
    left_join(
      df_m2 %>% select(semana, rv_pred_m2, alpha_m2, beta_m2,
                        gamma, gamma_pval, status_m2),
      by = "semana"
    ) %>%
    mutate(
      ticker_b3 = ticker,
      split     = if_else(semana >= semana_corte, "out-of-sample", "in-sample")
    )

  # Resumo de parâmetros M2 para este ticker
  n_previsoes <- nrow(df_m2)
  resumo <- tibble(
    ticker_b3        = ticker,
    n_previsoes      = n_previsoes,
    gamma_medio      = mean(df_m2$gamma,      na.rm = TRUE),
    gamma_pval_medio = mean(df_m2$gamma_pval, na.rm = TRUE),
    pct_signif_5pct  = mean(df_m2$gamma_pval < 0.05, na.rm = TRUE) * 100,
    persistencia_m1  = mean(df_m1$alpha + df_m1$beta,         na.rm = TRUE),
    persistencia_m2  = mean(df_m2$alpha_m2 + df_m2$beta_m2,   na.rm = TRUE)
  )

  list(previsoes = df_tick, resumo = resumo)
}

tickers <- sort(unique(df_final$ticker_b3))
cat(sprintf("\nProcessando %d tickers...\n", length(tickers)))

indices <- seq_along(tickers)

if (USE_PARALLEL && N_CORES > 1L) {
  cl <- parallel::makeCluster(N_CORES, type = "PSOCK")
  parallel::clusterExport(cl, varlist = c(
    "df_final", "JANELA", "MIN_OBS", "tickers",
    "processar_ticker", "calc_m0", "calc_m1", "calc_m2"
  ))
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(rugarch)
      library(dplyr)
    })
  })
  results_list <- tryCatch(
    parallel::parLapply(cl, indices, processar_ticker),
    finally = parallel::stopCluster(cl)
  )
} else {
  results_list <- lapply(indices, processar_ticker)
}

# ── 4. Merge e salva previsoes_consolidadas.csv ───────────────────────────────
results_ok       <- Filter(Negate(is.null), results_list)
df_previsoes_raw <- bind_rows(lapply(results_ok, `[[`, "previsoes"))
df_resumo_params <- bind_rows(lapply(results_ok, `[[`, "resumo"))

# Ordem de colunas compatível com o notebook Python
col_order <- c(
  "semana", "rv", "ret_semanal",
  "rv_pred_m1", "omega", "alpha", "beta",
  "rv_pred_m0",
  "rv_pred_m2", "gamma", "gamma_pval", "status_m2",
  "ticker_b3", "split"
)
df_previsoes_raw <- df_previsoes_raw %>%
  select(any_of(col_order))

write_csv(df_previsoes_raw, paste0(CAMINHO, "previsoes_consolidadas.csv"))
cat(sprintf("\nprevisoes_consolidadas.csv : %d linhas, %d tickers\n",
            nrow(df_previsoes_raw), n_distinct(df_previsoes_raw$ticker_b3)))

# ── 5. Salva resumo_parametros.csv ────────────────────────────────────────────
write_csv(df_resumo_params, paste0(CAMINHO, "resumo_parametros.csv"))
cat(sprintf("resumo_parametros.csv      : %d tickers\n", nrow(df_resumo_params)))

