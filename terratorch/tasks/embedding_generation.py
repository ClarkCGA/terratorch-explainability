import os
import torch
import rasterio
import logging
from torchgeo.trainers import BaseTask
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models.utils import TemporalWrapper

logger = logging.getLogger("terratorch")

class EmbeddingGeneration(BaseTask):
    """
    Task that runs inference once over datamodule to generate and save embeddings.
    """

    def __init__(
        self,
        model: str,
        model_args: dict,
        output_dir: str,
        use_temporal: bool = False,
        temporal_pooling: str = "mean",
        concat: bool = False,
        n_timestamps: int = 4,
    ) -> None:
        """
        Args:
            model (str): Name of ModelFactory class to be used to instantiate the model.
            model_args (Dict): Arguments passed to the model factory.
            output_dir (str): Directory to save embeddings in.
        """
        super().__init__()
        self.save_hyperparameters()        
        
    def configure_callbacks(self):
        return []

    def configure_models(self):
        self.model = BACKBONE_REGISTRY.build(self.hparams.model, **self.hparams.model_args)
        if self.hparams.use_temporal:
            self.model = TemporalWrapper(
                self.model, 
                pooling=self.hparams.temporal_pooling, 
                concat=self.hparams.concat, 
                n_timestamps=self.hparams.n_timestamps
            )
        self.model.eval()
        os.makedirs(self.hparams.output_dir, exist_ok=True)

    def training_step(self, *args, **kwargs): pass
    def validation_step(self, *args, **kwargs): pass
    def on_train_epoch_end(self): pass
    def on_validation_epoch_end(self): pass

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        x = batch["image"]
        file_names = x['file_id']
        x.pop("file_id", None)    
        batch_id = batch["filename"] 
        
        emb = self.model.get_embedding(x)

        # Handle torch.Tensor embedding
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu()
            B, T, H, W = emb.shape
            out_dir = os.path.join(self.hparams.output_dir, 'embeddings')
            os.makedirs(out_dir, exist_ok=True)

            for i in range(B):
                for t in range(T):
                    arr = emb[i, t].numpy()
                    fname = file_names[i][t]
                    out_tiff = os.path.join(out_dir, f"{fname}.tif")
                    os.makedirs(os.path.dirname(out_tiff), exist_ok=True)
                    with rasterio.open(
                        out_tiff,
                        "w",
                        driver="GTiff",
                        height=H,
                        width=W,
                        count=1,
                        dtype=arr.dtype,
                    ) as dst:
                        dst.write(arr, 1)
        

        # Handle dict embedding (multimodal data)
        elif isinstance(emb, dict):
            for modality, emb_mod in emb.items():
                emb_mod = emb_mod.detach().cpu()
                B, T, H, W = emb_mod.shape
                out_dir = os.path.join(self.hparams.output_dir, modality)
                os.makedirs(out_dir, exist_ok=True)

                for i in range(B):
                    for t in range(T):
                        arr = emb_mod[i, t].numpy()
                        fname = file_names[i][t]
                        out_tiff = os.path.join(out_dir, f"{fname}.tif")
                        os.makedirs(os.path.dirname(out_tiff), exist_ok=True)
                        with rasterio.open(
                            out_tiff,
                            "w",
                            driver="GTiff",
                            height=H,
                            width=W,
                            count=1,
                            dtype=arr.dtype,
                        ) as dst:
                            dst.write(arr, 1)
            print(f"Saved {out_tiff}")
        else:
            raise ValueError("Embedding must be a torch.Tensor or dict of tensors.")