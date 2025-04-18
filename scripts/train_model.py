import importlib
import logging
import yaml
import math

import torch
from torch.optim import AdamW
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy

from torch.utils.data import DataLoader

from src.constants import PAD_IDX
from src.midi_processor import MIDIProcessor
from src.midi_dataset_preprocessed import MIDIDatasetPreprocessed

def load_config(path="config/config.yaml"):
    with open(path,'r') as f:
        return yaml.safe_load(f)

class MidiLightningModule(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        inp = torch.cat((x[:, :1], y), dim=1)
        loss = self.model(inp)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inp = torch.cat((x[:, :1], y), dim=1)
        loss = self.model(inp)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        t = self.cfg['training']
        opt_cfg = t['optimizer']
        name = opt_cfg['name']
        params = opt_cfg['params']

        # dynamic optimizer lookup
        if name == 'FusedAdam':
            OptimClass = importlib.import_module('deepspeed.ops.adam').FusedAdam
        else:
            OptimClass = getattr(torch.optim, name, AdamW)

        optimizer = OptimClass(self.parameters(), **params)

        # Scheduler
        sched_cfg = t.get('scheduler', {})
        scheduler = None
        if sched_cfg:
            s_name = sched_cfg['name']
            s_params = sched_cfg['params'].copy()

            # compute total_steps if null
            if s_params.get('total_steps') is None:
                # total_batches per epoch
                data_size = len(self.trainer.datamodule.train_dataloader()) \
                            if hasattr(self.trainer, 'datamodule') else None
                # fallback: compute from loader attr
                data_size = data_size or len(self.trainer.train_dataloader)
                total_batches = math.ceil(data_size / t['grad_accumulation_steps'])
                s_params['total_steps'] = total_batches * t['max_epochs']

            if s_name == 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=s_params['total_steps']
                )
            elif s_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=s_params['total_steps'],
                    eta_min=params['lr'] * s_params.get('min_lr_ratio', 0.0)
                )
            elif s_name == 'linear_cosine':
                # warmup then cosine
                warm = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=s_params['warmup_steps']
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=s_params['total_steps'] - s_params['warmup_steps'],
                    eta_min=params['lr'] * s_params.get('min_lr_ratio', 0.0)
                )
                scheduler = {
                    'scheduler': torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warm, cosine],
                        milestones=[s_params['warmup_steps']]
                    ),
                    'interval': 'step'
                }
            else:
                raise ValueError(f"Unknown scheduler: {s_name}")

        return {'optimizer': optimizer, 'lr_scheduler': scheduler} if scheduler else optimizer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    
    strat_cfg = cfg['training'].get('strategy', {})
    strat_name = strat_cfg.get('name')
    strat_params = strat_cfg.get('params', {})

    if strat_name == 'deepspeed':
        strategy = DeepSpeedStrategy(**strat_params)
    elif strat_name == 'ddp':
        strategy = DDPStrategy(**strat_params)
    else:
        strategy = None

    # Data
    ds = cfg['dataset']
    proc = MIDIProcessor(**cfg['processor'])
    dataset = MIDIDatasetPreprocessed(
        sequence_length=ds['sequence_length'],
        preprocessed_dir=cfg['preprocessed_dir'],
        pad_idx=PAD_IDX,
        augmentation_shift=ds.get('augmentation_shift',0)
    )
    loader = DataLoader(dataset,
                        batch_size=cfg['training']['batch_size_per_gpu'],
                        num_workers=cfg['training']['num_workers'],
                        shuffle=True)

    # Dynamic Model + Wrapper
    mcfg = cfg['model']
    mod = importlib.import_module(mcfg['module'])
    ModelClass = getattr(mod, mcfg['class'])
    decoder = ModelClass(**mcfg.get('params', {}))

    wcfg = cfg.get('wrapper', {})
    wmod = importlib.import_module(wcfg.get('module','x_transformers'))
    WrapperClass = getattr(wmod, wcfg.get('class','AutoregressiveWrapper'))
    model = WrapperClass(decoder, **wcfg.get('params', {}))

    # LightningModule
    lit = MidiLightningModule(model, cfg)

    # Logger & checkpoint
    wandb = WandbLogger(project=cfg['wandb']['project_name'],
                        log_model=cfg['wandb'].get('log_model'))
    ckpt = ModelCheckpoint(dirpath=cfg['checkpoint_dir'],
                          save_top_k=1,
                          monitor='val_loss',
                          mode='min')

    # Trainer
    trainer = Trainer(
        logger=wandb,
        callbacks=[ckpt],
        strategy=strategy,
        max_epochs=cfg['training']['max_epochs'],
        precision=cfg['training']['precision'],
        accumulate_grad_batches=cfg['training']['grad_accumulation_steps'],
        gradient_clip_val=cfg['training']['gradient_clip_val']
    )
    trainer.fit(lit, loader)

if __name__=="__main__":
    main()