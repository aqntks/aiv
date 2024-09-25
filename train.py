

import os
import argparse
import datetime

from sconf import Config
from pathlib import Path
from os.path import basename

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from aiv.dataset import DonutDataset
from aiv.augmentation import get_transforms
from aiv.module import LightningModelModule, LightningDataModule


def train(config):

    pl.utilities.seed.seed_everything(config.seed, workers=True)

    model_module = LightningModelModule(config)
    data_module = LightningDataModule(config)

    transforms = get_transforms(config)

    datasets = {"train": [], "validation": []}
    for i, dataset_path in enumerate(config.dataset_paths):
        task_name = os.path.basename(dataset_path) 
        

        for split in ["train", "validation"]:
            datasets[split].append(
                DonutDataset(
                    dataset_name_or_path=dataset_path,
                    donut_model=model_module.model,
                    max_length=config.max_length,
                    transforms=transforms,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"<s_{task_name}>",
                    prompt_end_token=f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key
                )
            )

    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=config.precision,
        num_sanity_val_steps=config.num_sanity_val_steps,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    train(config)
