from collections.abc import Sequence
from typing import Any

import albumentations as A
from torch.utils.data import DataLoader

from terratorch.datamodules.generic_multimodal_data_module import MultimodalNormalize, wrap_in_compose_is_list
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize

from terratorch.datasets import GELOSDataset
from torchgeo.datamodules import GeoDataModule
from kournia.augmentation import AugementationSequential

MEANS = {
    "S2": { 
        "COASTAL_AEROSOL": 0.0,
        "BLUE": 0.0,
        "GREEN": 0.0,
        "RED": 0.0,
        "RED_EDGE_1": 0.0,
        "RED_EDGE_2": 0.0,
        "RED_EDGE_3": 0.0,
        "NIR_BROAD": 0.0,
        "NIR_NARROW": 0.0,
        "SWIR_1": 0.0,
        "SWIR_2": 0.0,
        # "WATER_VAPOR": 0.0,
        "CIRRUS": 0.0,
        "THEMRAL_INFRARED_1": 0.0,
    }, 
    "S1": { 
        "VV": 0.0,
        "VH": 0.0,
        "ASC_VV": 0.0,
        "ASC_VH": 0.0,
        "DSC_VV": 0.0,
        "DSC_VH": 0.0,
        "VV_VH": 0.0,
    }, 
    "Landsat": {
        "coastal": 0.0,    # Coastal/Aerosol (Band 1)
        "blue": 0.0,      # Blue (Band 2)
        "green": 0.0,      # Green (Band 3)
        "red": 0.0,        # Red (Band 4)
        "nir08": 0.0,      # Near Infrared (NIR, Band 5)
        "swir16": 0.0,    # Shortwave Infrared 1 (SWIR1, Band 6)
        "swir22": 0.0,     # Shortwave Infrared 2 (SWIR2, Band 7)
    },
    "DEM": {
        "dem": 0.0,
      },
    }

STDS = {
    "S2": { 
        "COASTAL_AEROSOL": 1.0,
        "BLUE": 1.0,
        "GREEN": 1.0,
        "RED": 1.0,
        "RED_EDGE_1": 1.0,
        "RED_EDGE_2": 1.0,
        "RED_EDGE_3": 1.0,
        "NIR_BROAD": 1.0,
        "NIR_NARROW": 1.0,
        "SWIR_1": 1.0,
        "SWIR_2": 1.0,
        # "WATER_VAPOR": 1.0,
        "CIRRUS": 1.0,
        "THEMRAL_INFRARED_1": 1.0,
    }, 
    "S1": { 
        "VV": 1.0,
        "VH": 1.0,
        "ASC_VV": 1.0,
        "ASC_VH": 1.0,
        "DSC_VV": 1.0,
        "DSC_VH": 1.0,
        "VV_VH": 1.0,
    }, 
    "Landsat": {
        "coastal": 1.0,    # Coastal/Aerosol (Band 1)
        "blue": 1.0,      # Blue (Band 2)
        "green": 1.0,      # Green (Band 3)
        "red": 1.0,        # Red (Band 4)
        "nir08": 1.0,      # Near Infrared (NIR, Band 5)
        "swir16": 1.0,    # Shortwave Infrared 1 (SWIR1, Band 6)
        "swir22": 1.0,     # Shortwave Infrared 2 (SWIR2, Band 7)
    },
    "DEM": {
        "dem": 1.0,
      },
    }
 
# instantiate GELOS datamodule class
class GELOSDataModule(GeoDataModule):
    """
    This is the datamodule for Geospatial Exploration of Latent Observation Space (GELOS)
    """
    
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            data_root: str | Path,
            bands: dict[str] = GELOS.all_band_names,
            transform: A.Compose | None | list[A.BasicTransform] = None, 
            aug: AugmentationSequential = None,
            metadata_filename: str = "chip_tracker.csv",
            **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule for GELOS.
        
        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            data_root (str | Path): Root directory for dataset.
            bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
            transform (A.Compose, optional): Transforms for data, defaults to ToTensorV2.
            aug (AugmentationSequential, optional): Augmentation or normalization to apply. Defaults to normalization if not provided.
            metadata_filename: (str, optional): Filename for chip tracker
            **kwargs: Additional keyword arguments.
            """
        super().__init__(GELOSDataSet, batch_size, num_workers, **kwargs)
        
        self.data_root = data_root
        self.bands = bands
        self.modalities = self.bands.keys()
        for modality in self.modalities:
            self.means[modality] = [MEANS[modality][band] for band in self.bands[modality]]
            self.stds[modality] = [STDS[modality][band] for band in self.bands[modality]]
        self.transform = wrap_in_compose_is_list(transform)
        if len(self.bands.keys) == 1:
            self.aug = Normalize(self.means[self.modalities[0]], self.stds[self.modalities[0]]) if aug is None else aug
        else:
            MultimodalNormalize(self.means, self.stds) if aug is None else aug
