from collections.abc import Sequence
from pathlib import Path

class GELOSDataSet(GeoDataSet):
    """
    Dataset intended for embedding extraction and exploration.
    Contains Sentinel 1 and 2 data, DEM, and Landsat 8 and 9 data.

    Dataset Format:

    .tif files for Sentinel 1, Sentinel 2, DEM, and Landsat 8 and 9 data
    .csv chip tracker with chip-level land cover classification

    Dataset Features:
    TBD Dataset Size
    4 time steps for each land cover chip
    """
 
    S2_BAND_NAMES = [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
        # "WATER_VAPOR",
        "CIRRUS",
        "THEMRAL_INFRARED_1",
      ] 
    S1_BAND_NAMES = [ 
        "VV",
        "VH",
        "ASC_VV",
        "ASC_VH",
        "DSC_VV",
        "DSC_VH",
        "VV_VH",
      ]
    LANDSAT_BAND_NAMES = [
        "coastal",    # Coastal/Aerosol (Band 1)
        "blue",      # Blue (Band 2)
        "green",      # Green (Band 3)
        "red",        # Red (Band 4)
        "nir08",      # Near Infrared (NIR, Band 5)
        "swir16",    # Shortwave Infrared 1 (SWIR1, Band 6)
        "swir22",     # Shortwave Infrared 2 (SWIR2, Band 7)
      ]
    DEM_BAND_NAMES = [
        "dem"
      ]
    all_band_names = {
        "S1": S1_BAND_NAMES,
        "S2": S2_BAND_NAMES,
        "Landsat": LANDSAT_BAND_NAMES,
        "DEM": DEM_BAND_NAMES,
    }

    rgb_bands = {
        "S1": [],
        "S2": ["RED", "GREEN", "BLUE"],
        "Landsat": ["red", "green", "blue"],
        "DEM": [],
    }

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    default_metadata_filename = "chip_tracker.csv"
    
    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, List[str]] = BAND_SETS["all"],
        transform: A.Compose | None = None,
    ) -> None:
        """
        Initializes an instance of GELOS.
        
        Args:
        data_root (str | Path): root directory where the dataset can be found
        bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
        transform (A.compose, optional): transform to apply. Defaults to ToTensorV2.
        """