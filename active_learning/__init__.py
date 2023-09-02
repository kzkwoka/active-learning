
__app_name__ = "active_learning"
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    PARAM_ERROR,
    DEVICE_ERROR,
    MODEL_ERROR,
    DATASET_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    PARAM_ERROR: "parameter value error",
    DEVICE_ERROR: "error loading specified device",
    MODEL_ERROR: "error loading model. invalid model or weights name",
    DATASET_ERROR: "error loading dataset"
    
}