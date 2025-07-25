import os
import torch
import rasterio
import logging
from torchgeo.trainers import BaseTask
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models.utils import TemporalWrapper
from terratorch.models.model import ModelOutput
from torch import nn


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
            model_factory (str): Name of ModelFactory class to be used to instantiate the model.
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
        if self.hparams.model_args.necks:
            self.neck: nn.Module = nn.Sequential(*self.hparams.model_args.necks)
        self.model.eval()
        # for k, v in self.model.named_parameters():
        #     print(k, v.shape)
        os.makedirs(self.hparams.output_dir, exist_ok=True)
    

    def training_step(self, *args, **kwargs): pass
    def validation_step(self, *args, **kwargs): pass
    def on_train_epoch_end(self): pass
    def on_validation_epoch_end(self): pass

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        
        x = batch["image"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        filename = batch["filename"]
        
        rest = {k: batch[k] for k in other_keys}

        if self.hparams.use_temporal:
            model_output = self.model.get_embedding(x, **rest)
        else:
            model_output = self.model.forward(x, **rest)

        emb = self.neck(model_output) if self.neck else model_output

        # handle options from config
        #TODO: call a function (e.g. unpatchify, for prithvi output) on outputs
         
        # Handle torch.Tensor embedding
        if isinstance(emb, torch.Tensor):
            emb['image'] = emb

        for modality, emb_mod in emb.items():
            emb_mod = emb_mod.detach().cpu()
            
            out_dir = os.path.join(self.hparams.output_dir, modality)
            os.makedirs(out_dir, exist_ok=True)

            for i in range(emb_mod[0]): # for each in batch
                emb_mod_fname = os.path.join(out_dir, f"{filename[i]}_embedding.pt")
                torch.save(emb_mod[i,1:,:],emb_mod_fname)
                cls_fname = os.path.join(out_dir, f"{filename[i]}_cls.pt")
                torch.save(emb_mod[i,:1,:],cls_fname)
            return