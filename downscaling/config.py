XI_INIT = 0.24
SIGMA_INIT = 0.57
KAPPA_INIT = 0.28

CENSOR_THRESHOLD = 0.22
BUCKET_RESOLUTION = 0.2153

TIME_COLS = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
SPATIAL_COLS = ["lat_Y", "lon_Y", "lat_X", "lon_X"]

ALLOWED_VARIANTS = {"both", "sigma_only", "kappa_only"}

SINGLE_COV_COL = "X_p01_dt0h"

BUCKET_RESOLUTION = 0.2153
RAIN_THRESHOLD_POSITIVE = 0.0

TUNING_PRESET = "large"
