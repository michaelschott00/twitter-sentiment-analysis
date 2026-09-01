import lightning.pytorch as pl
import torch

from twitter import data, models, modules

batch_size = 4
model_name = "distilbert-base-cased"
encoder = models.TransformerEncoder(name=model_name)
module = modules.SupervisedConstrastiveLearningModule(encoder=encoder)
data_module = data.TwitterDataModule(
    root_dir="data/splits",
    features="contrast",
    labels="clf",
    encoder_name=model_name,
    batch_size=batch_size
)

# auto-select accelerator (gpu if available, else cpu)
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

# fast dev run
trainer = pl.Trainer(accelerator=accelerator, fast_dev_run=True)
trainer.fit(module, data_module)

# overfit a small batch
trainer = pl.Trainer(accelerator=accelerator, overfit_batches=0.01)
trainer.fit(module, data_module)
