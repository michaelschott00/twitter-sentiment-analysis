from twitter import modules, models, data
import lightning.pytorch as pl


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

# fast dev run
trainer = pl.Trainer(accelerator="gpu", fast_dev_run=True)
trainer.fit(module, data_module)

# overfit a small batch
trainer = pl.Trainer(accelerator="gpu", overfit_batches=0.01)
trainer.fit(module, data_module)
