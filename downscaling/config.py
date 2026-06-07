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

SEED = 2026
DEVICE = None

REFERENCE_MODEL = "Stationary EGPD"
MODEL_ORDER = ["Stationary EGPD", "GLM", "GAM", "NN"]
MODEL_ORDER_NO_REF = ["GLM", "GAM", "NN"]

SIGMA_INIT = 0.58
KAPPA_INIT = 0.27
XI_INIT = 0.26

KAPPA_LIMIT = 2.0
LAMBDA_PROP_KAPPA_GT2 = 0.05
LAMBDA_EXCESS_KAPPA = 0.01

STATION_COL_CANDIDATES = ["site", "station", "station_name", "gauge", "name_Y", "site_Y"]

QUANTILES_FOR_DIAGNOSTICS = (
    0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
    0.90, 0.95, 0.99, 0.995, 0.999,
)

