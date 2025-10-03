from pathlib import Path
from typing import List
from torchgeo.datasets import GeoDataSet
import geopandas as gpd
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
        self.data_root = data_root
        self.bands = bands

        assert set(self.bands.keys()).issubset(
            set(self.all_band_names.keys())
        ), f"Please choose a subset of valid sensors: {self.all_band_names.keys()}"

        self.band_indices = {
            sens: [self.all_band_names[sens].index(band) for band in self.bands[sens]]
            for sens in self.bands.keys()
        }
        
        self.df = gpd.read_file(self.data_root / self.metadata_filename)

        # Adjust transforms based on the number of sensors
        if len(self.bands.keys()) == 1:
            self.transform = transform if transform else default_transform
        elif transform is None:
            self.transform = MultimodalToTensor(self.bands.keys())
        else:
            transform = {
                s: transform[s] if s in transform else default_transform
                for s in self.bands.keys()
            }
            self.transform = MultimodalTransforms(transform, shared=False)

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        sample_row = self.df.iloc[index]

        output = {}

        for sensor in self.bands.keys():
            sensor_filepaths = sample_row[sensor]
            image = self._load_sensor_images(sensor_filepaths)
            output[sensor] = image.astype(np.float32)

        if len(output.keys()) == 1:
            # Rename the single sensor key to "image"
            sensor = list(output.keys())[0]
            output["image"] = output.pop(sensor)
        if self.transform:
          output = self.transform(output)

        if self.concat_bands:
            # Concatenate bands of all image modalities
            data = [output.pop(m) for m in self.image_modalities if m in output]
            output["image"] = torch.cat(data, dim=1 if self.data_with_sample_dim else 0)
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.modalities if m in output}

        output["filename"] = self.samples[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None) -> np.Array

        data = rioxarray.open_rasterio(path, masked=True).to_numpy()
        if nan_replace is not None:
            data = np.nan_to_num(data, nan=nan_replace)

        return data

    def _load_sensor_images(self, sensor_filepaths: List(int)) -> np.Array

        sensor_images = [self._load_file(sensor_filepaths, self.nan_replace) for path in sensor_filepaths]
        sensor_image = np.stack(sensor_images, dim=0)

        return sensor_image

    def _process_metadata_df(self):
        for modality in self.bands.keys():
            
            # for each modality, construct file paths
            # if the modality has multiple dates, construct them from the dates column
            # otherwise, for single time step variables, construct from chip id
      


