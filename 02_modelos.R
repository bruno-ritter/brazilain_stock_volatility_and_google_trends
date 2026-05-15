# ==============================================================================
# 02_modelos.R
# TCC — Previsão de Volatilidade com SVI (Google Trends)
# Modelos: M0 (RW), M1 (GARCH), M2 (GARCH-X ΔSVI), M3 (GARCH-X log-dev SVI)
# Janela móvel: 156 semanas; painel com mesmas semanas por ticker (sem lacunas)
# ==============================================================================

# ── 0. Bibliotecas e constantes ───────────────────────────────────────────────
suppressPackageStartupMessages({
  library(rugarch)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lubridate)
  library(parallel)
})

CAMINHO      <- "./data/"
JANELA       <- 52L * 3L       # 156 semanas de treino
T_START      <- JANELA + 9L   # warm-up svi_log_dev (8) + 1
USE_PARALLEL <- TRUE
N_CORES      <- max(1L, parallel::detectCores() - 1L)
set.seed(42)

cat("=== 02_modelos.R ===\n")
cat(sprintf("Janela de treino : %d semanas\n", JANELA))
cat(sprintf("Primeira prev.   : indice t = %d\n", T_START))
cat(sprintf("Núcleos usados   : %d\n", if (USE_PARALLEL) N_CORES else 1L))

# ── 1. Carrega dados ──────────────────────────────────────────────────────────
df_final <- read_csv(
  paste0(CAMINHO, "base_final_tcc.csv"),
  col_types = cols(semana = col_date(format = "%Y-%m-%d")),
  show_col_types = FALSE
)

cat(sprintf("\nbase_final_tcc   : %d linhas, %d tickers\n",
            nrow(df_final), n_distinct(df_final$ticker_b3)))


# ── 2. Funções auxiliares ─────────────────────────────────────────────────────

calc_svi_log_dev <- function(svi) {
  n <- length(svi)
  out <- rep(NA_real_, n)
  for (i in 9L:n) {
    med <- median(svi[(i - 8L):(i - 1L)], na.rm = TRUE)
    out[i] <- log1p(svi[i]) - log1p(med)
  }
  out
}

janela_ok <- function(df_t, t, janela = JANELA) {
  w <- (t - janela):(t - 1L)
  all(!is.na(df_t$ret_semanal[w])) &&
    all(!is.na(df_t$svi_diff[w])) && !is.na(df_t$svi_diff[t - 1L]) &&
    all(!is.na(df_t$svi_log_dev[w])) && !is.na(df_t$svi_log_dev[t - 1L]) &&
    !is.na(df_t$rv[t - 1L])
}

# M0: passeio aleatório — rv_pred = rv_{t-1} nas semanas idx
calc_m0 <- function(df_t, idx) {
  tibble(
    semana     = df_t$semana[idx],
    rv_pred_m0 = df_t$rv[idx - 1L]
  )
}

# M1: GARCH(1,1) rolling nas semanas idx
calc_m1 <- function(df_t, idx, janela = JANELA) {
  if (length(idx) == 0L) return(tibble())

  spec_m1 <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(1L, 1L)),
    mean.model         = list(armaOrder = c(0L, 0L), include.mean = TRUE),
    distribution.model = "norm"
  )

  resultados <- vector("list", length(idx))

  for (k in seq_along(idx)) {
    t <- idx[k]
    train_ret <- df_t$ret_semanal[(t - janela):(t - 1L)] * 100

    linha <- list(
      semana      = df_t$semana[t],
      rv          = df_t$rv[t],
      ret_semanal = df_t$ret_semanal[t],
      rv_pred_m1  = NA_real_,
      omega       = NA_real_,
      alpha       = NA_real_,
      beta        = NA_real_,
      status_m1   = NA_character_
    )

    fit_res <- tryCatch({
      fit <- ugarchfit(spec_m1, data = train_ret, solver = "hybrid",
                       fit.control = list(stationarity = 1),
                       solver.control = list(trace = 0))
      fc  <- ugarchforecast(fit, n.ahead = 1L)
      list(ok = TRUE, fit = fit, fc = fc)
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))

    if (fit_res$ok) {
      mc               <- fit_res$fit@fit$matcoef
      linha$rv_pred_m1 <- as.numeric(sigma(fit_res$fc))^2 / 10000
      linha$omega      <- mc["omega",  1L]
      linha$alpha      <- mc["alpha1", 1L]
      linha$beta       <- mc["beta1",  1L]
      linha$status_m1  <- "ok"
    } else {
      linha$status_m1 <- fit_res$msg
    }

    resultados[[k]] <- linha
  }

  bind_rows(resultados)
}

# GARCH-X(1,1) — M2 (svi_diff) ou M3 (svi_log_dev)
calc_garchx <- function(df_t, x_col, idx, prefix, janela = JANELA) {
  if (length(idx) == 0L) return(tibble())

  is_m2 <- prefix == "m2"
  resultados <- vector("list", length(idx))

  for (k in seq_along(idx)) {
    t <- idx[k]
    x_vec     <- df_t[[x_col]]
    train_ret <- df_t$ret_semanal[(t - janela):(t - 1L)] * 100
    train_x   <- matrix(x_vec[(t - janela):(t - 1L)], ncol = 1L)
    fc_x      <- matrix(x_vec[t - 1L], nrow = 1L, ncol = 1L)

    linha <- list(semana = df_t$semana[t], rv = df_t$rv[t])

    if (is_m2) {
      linha$rv_pred_m2 <- NA_real_
      linha$alpha_m2   <- NA_real_
      linha$beta_m2    <- NA_real_
      linha$gamma      <- NA_real_
      linha$gamma_pval <- NA_real_
      linha$status_m2  <- NA_character_
    } else {
      linha$rv_pred_m3     <- NA_real_
      linha$alpha_m3       <- NA_real_
      linha$beta_m3        <- NA_real_
      linha$gamma_m3       <- NA_real_
      linha$gamma_m3_pval  <- NA_real_
      linha$status_m3      <- NA_character_
    }

    fit_res <- tryCatch({
      spec_x <- ugarchspec(
        variance.model     = list(model = "sGARCH", garchOrder = c(1L, 1L),
                                  external.regressors = train_x),
        mean.model         = list(armaOrder = c(0L, 0L), include.mean = TRUE),
        distribution.model = "norm"
      )
      fit <- ugarchfit(spec_x, data = train_ret, solver = "hybrid",
                       fit.control = list(stationarity = 1),
                       solver.control = list(trace = 0))
      fc  <- ugarchforecast(fit, n.ahead = 1L,
                            external.forecasts = list(vxreg = fc_x))
      list(ok = TRUE, fit = fit, fc = fc)
    }, error = function(e) list(ok = FALSE, msg = conditionMessage(e)))

    if (fit_res$ok) {
      mc <- fit_res$fit@fit$matcoef
      rv_p <- as.numeric(sigma(fit_res$fc))^2 / 10000
      if (is_m2) {
        linha$rv_pred_m2 <- rv_p
        linha$alpha_m2   <- mc["alpha1", 1L]
        linha$beta_m2    <- mc["beta1",  1L]
        linha$gamma      <- mc["vxreg1", 1L]
        linha$gamma_pval <- mc["vxreg1", 4L]
        linha$status_m2  <- "ok"
      } else {
        linha$rv_pred_m3    <- rv_p
        linha$alpha_m3      <- mc["alpha1", 1L]
        linha$beta_m3       <- mc["beta1",  1L]
        linha$gamma_m3      <- mc["vxreg1", 1L]
        linha$gamma_m3_pval <- mc["vxreg1", 4L]
        linha$status_m3     <- "ok"
      }
    } else {
      if (is_m2) linha$status_m2 <- fit_res$msg else linha$status_m3 <- fit_res$msg
    }

    resultados[[k]] <- linha
  }

  bind_rows(resultados)
}


# ── 3. Pré-voo: calendário global e tickers elegíveis ─────────────────────────

df_final <- df_final %>%
  group_by(ticker_b3) %>%
  mutate(svi_log_dev = calc_svi_log_dev(svi)) %>%
  ungroup()

tickers_all <- sort(unique(df_final$ticker_b3))
grupo_ref <- df_final %>%
  filter(ticker_b3 == tickers_all[1L], !is.na(svi_diff)) %>%
  arrange(semana)
n_ref <- nrow(grupo_ref)

if (n_ref <= T_START) {
  stop(sprintf("Serie de referencia muito curta (n=%d, T_START=%d)", n_ref, T_START))
}

idx_global        <- seq(T_START, n_ref)
semanas_previsao  <- grupo_ref$semana[idx_global]
semana_corte_global <- semanas_previsao[1L]
n_semanas_previsao  <- length(idx_global)

tickers_elegiveis <- character(0)
for (tk in tickers_all) {
  g <- df_final %>%
    filter(ticker_b3 == tk, !is.na(svi_diff)) %>%
    arrange(semana)
  if (nrow(g) != n_ref) next
  if (!identical(g$semana, grupo_ref$semana)) next
  if (all(vapply(idx_global, janela_ok, logical(1L), df_t = g))) {
    tickers_elegiveis <- c(tickers_elegiveis, tk)
  }
}

cat(sprintf("\nPre-voo:\n"))
cat(sprintf("  Semanas de previsao (contiguas) : %d\n", n_semanas_previsao))
cat(sprintf("  De %s ate %s\n",
            min(semanas_previsao), max(semanas_previsao)))
cat(sprintf("  Tickers elegiveis             : %d / %d\n",
            length(tickers_elegiveis), length(tickers_all)))
cat(sprintf("  Excluidos no pre-voo          : %d\n",
            length(tickers_all) - length(tickers_elegiveis)))

if (length(tickers_elegiveis) == 0L) {
  stop("Nenhum ticker passou no pre-voo.")
}


# ── 4. Loop principal por ticker ──────────────────────────────────────────────

processar_ticker <- function(i) {
  ticker <- tickers_elegiveis[i]
  n_total <- length(tickers_elegiveis)
  cat(sprintf("  [%d/%d] %s\n", i, n_total, ticker))

  grupo <- df_final %>%
    filter(ticker_b3 == ticker, !is.na(svi_diff)) %>%
    arrange(semana)

  df_m1 <- calc_m1(grupo, idx_global)
  df_m2 <- calc_garchx(grupo, "svi_diff", idx_global, "m2")
  df_m3 <- calc_garchx(grupo, "svi_log_dev", idx_global, "m3")
  df_m0 <- calc_m0(grupo, idx_global)

  stopifnot(
    identical(df_m1$semana, df_m2$semana),
    identical(df_m1$semana, df_m3$semana)
  )

  df_tick <- df_m1 %>%
    mutate(rv_pred_m0 = df_m0$rv_pred_m0) %>%
    left_join(
      df_m2 %>% select(semana, rv_pred_m2, alpha_m2, beta_m2,
                       gamma, gamma_pval, status_m2),
      by = "semana"
    ) %>%
    left_join(
      df_m3 %>% select(semana, rv_pred_m3, alpha_m3, beta_m3,
                       gamma_m3, gamma_m3_pval, status_m3),
      by = "semana"
    ) %>%
    mutate(
      ticker_b3 = ticker,
      split     = if_else(semana >= semana_corte_global,
                          "out-of-sample", "in-sample")
    )

  resumo <- tibble(
    ticker_b3           = ticker,
    n_previsoes         = nrow(df_m1),
    gamma_medio         = mean(df_m2$gamma, na.rm = TRUE),
    gamma_pval_medio    = mean(df_m2$gamma_pval, na.rm = TRUE),
    pct_signif_5pct     = mean(df_m2$gamma_pval < 0.05, na.rm = TRUE) * 100,
    gamma_m3_medio      = mean(df_m3$gamma_m3, na.rm = TRUE),
    gamma_m3_pval_medio = mean(df_m3$gamma_m3_pval, na.rm = TRUE),
    pct_signif_5pct_m3  = mean(df_m3$gamma_m3_pval < 0.05, na.rm = TRUE) * 100,
    persistencia_m1     = mean(df_m1$alpha + df_m1$beta, na.rm = TRUE),
    persistencia_m2     = mean(df_m2$alpha_m2 + df_m2$beta_m2, na.rm = TRUE),
    persistencia_m3     = mean(df_m3$alpha_m3 + df_m3$beta_m3, na.rm = TRUE)
  )

  list(previsoes = df_tick, resumo = resumo)
}

cat(sprintf("\nProcessando %d tickers elegiveis...\n", length(tickers_elegiveis)))

indices <- seq_along(tickers_elegiveis)

if (USE_PARALLEL && N_CORES > 1L) {
  cl <- parallel::makeCluster(N_CORES, type = "PSOCK")
  parallel::clusterExport(cl, varlist = c(
    "df_final", "JANELA", "T_START", "idx_global", "semana_corte_global",
    "tickers_elegiveis",
    "processar_ticker", "calc_m0", "calc_m1", "calc_garchx",
    "calc_svi_log_dev", "janela_ok"
  ), envir = environment())
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

# ── 5. Merge, validação e salva CSVs ──────────────────────────────────────────
results_ok       <- Filter(Negate(is.null), results_list)
df_previsoes_raw <- bind_rows(lapply(results_ok, `[[`, "previsoes"))
df_resumo_params <- bind_rows(lapply(results_ok, `[[`, "resumo"))

n_por_ticker <- df_previsoes_raw %>% count(ticker_b3)
if (length(unique(n_por_ticker$n)) != 1L) {
  stop("Painel desbalanceado: tickers com numero diferente de semanas.")
}
if (unique(n_por_ticker$n) != n_semanas_previsao) {
  stop(sprintf("Esperado %d semanas/ticker, obtido %d.",
               n_semanas_previsao, unique(n_por_ticker$n)))
}

cal_ok <- df_previsoes_raw %>%
  distinct(ticker_b3, semana) %>%
  count(ticker_b3)
if (length(unique(cal_ok$n)) != 1L) {
  stop("Calendario inconsistente entre tickers.")
}

col_order <- c(
  "semana", "rv", "ret_semanal",
  "rv_pred_m1", "omega", "alpha", "beta", "status_m1",
  "rv_pred_m0",
  "rv_pred_m2", "gamma", "gamma_pval", "status_m2",
  "rv_pred_m3", "gamma_m3", "gamma_m3_pval", "alpha_m3", "beta_m3", "status_m3",
  "ticker_b3", "split"
)
df_previsoes_raw <- df_previsoes_raw %>%
  select(any_of(col_order))

write_csv(df_previsoes_raw, paste0(CAMINHO, "previsoes_consolidadas.csv"))
cat(sprintf("\nprevisoes_consolidadas.csv : %d linhas, %d tickers, %d semanas/ticker\n",
            nrow(df_previsoes_raw), n_distinct(df_previsoes_raw$ticker_b3),
            unique(n_por_ticker$n)))

write_csv(df_resumo_params, paste0(CAMINHO, "resumo_parametros.csv"))
cat(sprintf("resumo_parametros.csv      : %d tickers\n", nrow(df_resumo_params)))
