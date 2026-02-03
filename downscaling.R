muse <- FALSE
if (muse) {
  # Get the muse folder
  folder_muse <- "/home/serrec/work_rainstsimu/downscaling"
  setwd(folder_muse)
}

source("load_libraries.R")
source("pinnEV.R")
library(sf)
library(sp)
library(geosphere)
library(reticulate)
library(lubridate)
Sys.unsetenv("RETICULATE_PYTHON")
py_version <- "3.9.18"
path <- paste0(reticulate::virtualenv_root(), "/pinnEV_env/bin/python")
# Sys.setenv(RETICULATE_PYTHON = path) # Set Python interpreter
Sys.setenv(RETICULATE_LOG_LEVEL = "DEBUG")
tf_version = "2.13.1"
reticulate::use_virtualenv("pinnEV_env", required = T)

library(keras)
library(tensorflow)
library(dplyr)

##### EGPD mean parameters from stations ##############################################

filename_egpd <- paste0(data_folder, "../thesis/resources/images/EGPD/OMSEV/2019_2024/egpd_results.csv")
egpd_params <- read.csv(filename_egpd)

kappa_vect <- egpd_params$kappa
xi_vect <- egpd_params$xi
sigma_vect <- egpd_params$sigma

mean_kappa <- mean(kappa_vect, na.rm = TRUE)
mean_xi <- mean(xi_vect, na.rm = TRUE)
mean_sigma <- mean(sigma_vect, na.rm = TRUE)

# or
median_kappa <- median(kappa_vect, na.rm=TRUE)
median_xi    <- median(xi_vect, na.rm=TRUE)
median_sigma <- median(sigma_vect, na.rm=TRUE)


# LOAD DOWNSCALING TABLE #######################################################
# and add variables ############################################################

output_file <- paste0(data_folder, "downscaling/downscaling_table.csv")

df_Y_X_raw <- read.csv(output_file, sep = ";", header = TRUE)

# first line
df_Y_X_raw[1,]
df_Y_X <- df_Y_X_raw

# keep only case where Y_obs is not NA nor 0
df_Y_X <- df_Y_X[!is.na(df_Y_X$Y_obs), ]
df_Y_X <- df_Y_X[df_Y_X$Y_obs > 0, ]
length(df_Y_X$Y_obs)
# Keep time as a POSIXct object
df_Y_X$time <- as.POSIXct(df_Y_X$time,
                         format = "%Y-%m-%d %H:%M:%S", tz = "GMT")

df_Y_X$hour <- hour(df_Y_X$time)
df_Y_X$minute <- minute(df_Y_X$time)
df_Y_X$day <- day(df_Y_X$time)
df_Y_X$month <- month(df_Y_X$time)
df_Y_X$year <- year(df_Y_X$time)
head(df_Y_X)

# transform with sin and cos to have cyclical variables ie 0 and 23 hours are close
df_Y_X$hour_sin <- sin(2 * pi * df_Y_X$hour / 24)
df_Y_X$hour_cos <- cos(2 * pi * df_Y_X$hour / 24)
df_Y_X$minute_sin <- sin(2 * pi * df_Y_X$minute / 60)
df_Y_X$minute_cos <- cos(2 * pi * df_Y_X$minute / 60)
df_Y_X$day_sin <- sin(2 * pi * df_Y_X$day / 31)  # Max 31 days
df_Y_X$day_cos <- cos(2 * pi * df_Y_X$day / 31)
df_Y_X$month_sin <- sin(2 * pi * (df_Y_X$month-1) / 12)
df_Y_X$month_cos <- cos(2 * pi * (df_Y_X$month-1) / 12)


# flag to know if episodes are corresponding or not between X and Y
df_Y_X$corres <- 1
# when all X1 to X27 are 0 put corres to 0
for (i in 1:nrow(df_Y_X)) {
  if (all(df_Y_X[i, paste0("X", 1:27)] == 0)) {
    df_Y_X$corres[i] <- 0
  }
}

count(df_Y_X$corres == 1)
count(df_Y_X$corres == 0)

# do spatial summaries of the cube X1 to X27
df_Y_X$X_mean <- rowMeans(df_Y_X[, paste0("X", 1:27)], na.rm = TRUE)
df_Y_X$X_sd <- apply(df_Y_X[, paste0("X", 1:27)], 1, sd, na.rm = TRUE)
df_Y_X$X_max <- apply(df_Y_X[, paste0("X", 1:27)], 1, max, na.rm = TRUE)
df_Y_X$X_min <- apply(df_Y_X[, paste0("X", 1:27)], 1, min, na.rm = TRUE)

# spatial offset between X and Y
df_Y_X$spatial_offset <- sqrt((df_Y_X$lon_X - df_Y_X$lon_Y)^2 +
                               (df_Y_X$lat_X - df_Y_X$lat_Y)^2)

# remove useless columns
df_Y_X <- df_Y_X[, !names(df_Y_X) %in% c("lon_X", "lat_X", "lon_Y", "lat_Y", paste0("X", 1:27),
                                         "hour", "minute", "day", "month", "year")]

colnames(df_Y_X)
head(df_Y_X)

######################################################################################
# DOWNSCALING WITH NEURAL NETWORKS ##################################################
######################################################################################
# Load downscaling table
output_file <- paste0(data_folder, "downscaling/downscaling_table.csv")
df_raw <- read.csv(output_file, sep=";", header=TRUE)

# Get needed features for modeling
df_model <- df_raw %>%
    filter(!is.na(Y_obs), Y_obs > 0) %>%
    mutate(
      time = as.POSIXct(time, format = "%Y-%m-%d %H:%M:%S", tz = "GMT"),
      hour = as.integer(format(time, "%H")),
      minute = as.integer(format(time, "%M")),
      day = as.integer(format(time, "%d")),
      month = as.integer(format(time, "%m")),
      year = as.integer(format(time, "%Y")),
      # cyclical time
      hour_sin = sin(2*pi*hour/24),   hour_cos = cos(2*pi*hour/24),
      month_sin = sin(2*pi*(month-1)/12), month_cos = cos(2*pi*(month-1)/12)
    )

# corres based on raw X1..X27
x_cols27 <- paste0("X", 1:27)
# flag to know if episodes are corresponding in X and Y
df_model$corres <- as.integer(apply(df_model[, x_cols27], 1, function(r) !all(r == 0)))

df_model <- df_model %>%
mutate(
    X_mean = rowMeans(across(all_of(x_cols27)), na.rm = TRUE),
    X_sd   = apply(select(., all_of(x_cols27)), 1, sd,  na.rm = TRUE),
    X_max  = apply(select(., all_of(x_cols27)), 1, max, na.rm = TRUE),
    X_min  = apply(select(., all_of(x_cols27)), 1, min, na.rm = TRUE),
    spatial_offset = sqrt((lon_X - lon_Y)^2 + (lat_X - lat_Y)^2),
    spatial_dist_XY = distHaversine(matrix(c(lon_X, lat_X), ncol=2),
                                           matrix(c(lon_Y, lat_Y), ncol=2))
)

  # keep only useful columns (keep time + station for split/encoding)
df_model <- df_model %>%
    select(time, station, Y_obs,
           hour_sin, hour_cos, month_sin, month_cos,
           X_mean, X_sd, X_max, X_min, spatial_offset, spatial_dist_XY,
           corres)

df_model$station <- factor(df_model$station)
head(df_model)

get_dist_mat <- function(df_lonlat) {
  # df_lonlat must have cols Longitude, Latitude
  n <- nrow(df_lonlat)
  out <- matrix(0, n, n)
  for (i in 1:n) {
    out[i, ] <- geosphere::distHaversine(
      cbind(df_lonlat$Longitude[i], df_lonlat$Latitude[i]),
      cbind(df_lonlat$Longitude, df_lonlat$Latitude)
    )
  }
  out
}

# temporal split function
make_time_split <- function(df, train_frac = 0.8, seed = 123) {
  set.seed(seed)
  times <- sort(unique(df$time))
  train_times <- sample(times, size = floor(train_frac * length(times)))
  train_idx <- which(df$time %in% train_times)
  valid_idx <- setdiff(seq_len(nrow(df)), train_idx)
  list(train_idx = train_idx, valid_idx = valid_idx)
}

split <- make_time_split(df_model, train_frac = 0.8, seed = 123)


# standardization function based on training set only
standardize_train_only <- function(df, train_idx, scale_cols) {
  mu  <- colMeans(df[train_idx, scale_cols, drop = FALSE], na.rm = TRUE)
  sdv <- apply(df[train_idx, scale_cols, drop = FALSE], 2, sd, na.rm = TRUE)
  sdv[sdv == 0] <- 1

  df[, scale_cols] <- sweep(df[, scale_cols, drop=FALSE], 2, mu, "-")
  df[, scale_cols] <- sweep(df[, scale_cols, drop=FALSE], 2, sdv, "/")

  list(df = df, mu = mu, sdv = sdv)
}


build_xy <- function(df, x_cols, train_idx, valid_idx) {
  X_mat <- data.matrix(df[, x_cols])
  n_obs <- nrow(df); p <- ncol(X_mat)
  X_all <- array(X_mat, dim = c(n_obs, 1, p))

  Y_vec <- as.numeric(df$Y_obs)
  Y.train <- Y_vec
  Y.valid <- Y_vec
  Y.train[valid_idx] <- -1e10
  Y.valid[train_idx] <- -1e10

  Y.train <- array(Y.train, dim = c(n_obs, 1))
  Y.valid <- array(Y.valid, dim = c(n_obs, 1))

  list(
    X.s = list(X.nn.s = X_all),
    X.k = list(X.nn.k = X_all),
    Y.train = Y.train,
    Y.valid = Y.valid
  )
}


evaluate_model <- function(df, valid_idx, fit, X.s, X.k, probs = c(0.95, 0.99)) {
  out <- eGPD.NN.predict(X.s = X.s, X.k = X.k, fit$model)

  # out$pred.sigma etc are typically arrays (n_obs,1) or vectors; coerce
  sigma_hat <- as.vector(out$pred.sigma)
  kappa_hat <- as.vector(out$pred.kappa)
  xi_hat    <- as.vector(out$pred.xi)

  y_valid <- df$Y_obs[valid_idx]

  res <- list(
    train_loss = as.numeric(fit$`Training loss`["loss"]),
    valid_loss = as.numeric(fit$`Validation loss`["loss"])
  )

  # quantile calibration if qeGPD is available
  if (exists("qeGPD", mode = "function")) {
    for (p in probs) {
      qhat <- qeGPD(p, sigma = sigma_hat, kappa = kappa_hat, xi = xi_hat)
      exc_rate <- mean(y_valid > qhat[valid_idx])
      res[[paste0("exc_rate_", p)]] <- exc_rate

      # RMSE on quantile exceedances indicator
      res[[paste0("rmse_ind_", p)]] <- sqrt(mean((as.numeric(y_valid > qhat[valid_idx]) - (1 - p))^2))
    }
  }

  res
}


make_config <- function(name, use_corres = FALSE, use_station_oh = FALSE,
                        widths = c(8, 4), init = c("mean", "median")) {
  list(name = name,
       use_corres = use_corres,
       use_station_oh = use_station_oh,
       widths = widths,
       init = match.arg(init))
}


run_one_config <- function(df_model, split, cfg,
                           mean_sigma, mean_kappa, mean_xi,
                           median_sigma, median_kappa, median_xi,
                           n_ep = 30, batch_size = 64, seed = 1) {

  df <- df_model

  # station one-hot if requested
  if (cfg$use_station_oh) {
    station_oh <- as.data.frame(model.matrix(~ station - 1, df))
    df <- bind_cols(df %>% select(-station), station_oh)
    station_cols <- colnames(station_oh)
  } else {
    station_cols <- character(0)
    df$station <- factor(df$station) # keep as factor (not in X)
  }

  # choose x_cols
  time_cols  <- c("hour_sin","hour_cos","month_sin","month_cos")
  radar_cols <- c("X_mean","X_sd","X_max","X_min","spatial_offset")
  x_cols <- c(time_cols, radar_cols)

  if (cfg$use_corres) x_cols <- c(x_cols, "corres")
  x_cols <- c(x_cols, station_cols)

  # standardize continuous columns (NOT sin/cos, NOT one-hot, NOT corres)
  scale_cols <- radar_cols
  st <- standardize_train_only(df, split$train_idx, scale_cols)
  df <- st$df

  # build tensors
  xy <- build_xy(df, x_cols, split$train_idx, split$valid_idx)

  # init
  if (cfg$init == "mean") {
    init.scale <- mean_sigma; init.kappa <- mean_kappa; init.xi <- mean_xi
  } else {
    init.scale <- median_sigma; init.kappa <- median_kappa; init.xi <- median_xi
  }

  # train
  fit <- eGPD.NN.train(
    xy$Y.train, xy$Y.valid,
    xy$X.s, xy$X.k,
    type = "MLP",
    n.ep = n_ep, batch.size = batch_size,
    init.scale = init.scale,
    init.kappa = init.kappa,
    init.xi = init.xi,
    widths = cfg$widths,
    seed = seed
  )

  # evaluate
  metrics <- evaluate_model(df, split$valid_idx, fit, xy$X.s, xy$X.k)

  data.frame(
    model = cfg$name,
    init = cfg$init,
    widths = paste(cfg$widths, collapse = "-"),
    use_corres = cfg$use_corres,
    use_station_oh = cfg$use_station_oh,
    train_loss = metrics$train_loss,
    valid_loss = metrics$valid_loss,
    exc_rate_0.95 = if (!is.null(metrics$exc_rate_0.95)) metrics$exc_rate_0.95 else NA_real_,
    exc_rate_0.99 = if (!is.null(metrics$exc_rate_0.99)) metrics$exc_rate_0.99 else NA_real_,
    stringsAsFactors = FALSE
  )
}


# Configs to compare
configs <- list(
  make_config("baseline_9_mean_w84", use_corres=FALSE, use_station_oh=FALSE, widths=c(8,4), init="mean"),
  make_config("baseline_9_median_w84", use_corres=FALSE, use_station_oh=FALSE, widths=c(8,4), init="median"),

#   make_config("plus_corres_mean_w84", use_corres=TRUE,  use_station_oh=FALSE, widths=c(8,4), init="mean"),
#   make_config("plus_station_mean_w84",use_corres=FALSE, use_station_oh=TRUE,  widths=c(8,4), init="mean"),

#   make_config("station_corres_mean_w84",use_corres=TRUE, use_station_oh=TRUE, widths=c(8,4), init="mean"),

#   make_config("baseline_9_mean_w63", use_corres=FALSE, use_station_oh=FALSE, widths=c(6,3), init="mean"),
#   make_config("baseline_9_mean_w42", use_corres=FALSE, use_station_oh=FALSE, widths=c(4,2), init="mean")
)


results <- dplyr::bind_rows(lapply(configs, function(cfg) {
  message("Running: ", cfg$name)
  run_one_config(
    df_base, split, cfg,
    mean_sigma, mean_kappa, mean_xi,
    median_sigma, median_kappa, median_xi,
    n_ep = 30, batch_size = 64, seed = 1
  )
}))

print(results %>% arrange(valid_loss))
write.csv(results, file = paste0(data_folder, "downscaling/model_comparison.csv"), row.names = FALSE)


library(ggplot2)

ggplot(results, aes(x=reorder(model, valid_loss), y=valid_loss)) +
  geom_col() +
  coord_flip() +
  labs(x="Model", y="Validation loss", title="eGPD-NN model comparison")
