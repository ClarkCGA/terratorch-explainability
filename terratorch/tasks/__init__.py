import logging
from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.tasks.regression_tasks import PixelwiseRegressionTask
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask
from terratorch.tasks.multilabel_classification_tasks import MultiLabelClassificationTask
from terratorch.tasks.reconstruction_tasks import ReconstructionTask
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.object_detection_task import ObjectDetectionTask
from terratorch.tasks.embedding_generation import EmbeddingGeneration

try:
    wxc_present = True
    from terratorch.tasks.wxc_downscaling_task import WxCDownscalingTask 
    from terratorch.tasks.wxc_task import WxCTask
    logging.getLogger('terratorch').debug('wxc_downscaling found.')
except ImportError as e:
    import logging
    logging.getLogger('terratorch').debug('wxc_downscaling not installed')
    wxc_present = False


__all__ = (
    "SemanticSegmentationTask",
    "PixelwiseRegressionTask",
    "ClassificationTask",
    "MultiLabelClassificationTask"
    "ReconstructionTask",
    "ObjectDetectionTask",
    "EmbeddingGeneration",
)

if wxc_present:
    __all__.__add__(("WxCDownscalingTask", "WxCTask",))
