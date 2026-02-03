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

