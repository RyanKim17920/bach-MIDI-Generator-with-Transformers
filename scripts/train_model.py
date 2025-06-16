import importlib # Keep for optimizer loading
import logging
import yaml
import math
import os # Added for path joining

import torch
from torch.optim import AdamW
# Import random_split
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from src.callbacks import ValidationMIDIGenerationCallback


# --- Add direct imports for the fixed model structure ---
from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper

from src.constants import PAD_IDX
from src.midi_dataset_preprocessed import MIDIDatasetPreprocessed

def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        # Basic interpolation for dataset.sequence_length
        # A more robust solution would use OmegaConf or similar
        content = f.read()
        cfg_raw = yaml.safe_load(content)
        seq_len = cfg_raw.get('dataset', {}).get('sequence_length', 2048) # Default if not found
        content = content.replace('${dataset.sequence_length}', str(seq_len))
        return yaml.safe_load(content)
    
class MidiLightningModule(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        # Avoid saving the large model dict itself in hyperparameters
        hparams_to_save = {k: v for k, v in cfg.items() if k != 'model'}
        hparams_to_save['model_config'] = cfg.get('model', {}) # Save model config separately
        self.save_hyperparameters(hparams_to_save)


    def forward(self, x, **kwargs):
        # AutoregressiveWrapper handles the forward pass for training/generation
        # Pass kwargs for generation parameters like temperature, filter_logits_fn etc.
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        # AutoregressiveWrapper expects the target sequence
        # It internally shifts the sequence for prediction and calculates loss
        _, y = batch # Use target sequence y (x is context, not needed here for loss calculation)
        loss = self.model(y) # Pass target sequence, get loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch # Use target sequence y
        loss = self.model(y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        t = self.cfg['training']
        opt_cfg = t['optimizer']
        name = opt_cfg['name']
        params = opt_cfg['params']

        # --- FIX: Check if LR is None ---
        if params.get('lr') is None:
            raise ValueError("Learning rate 'lr' is not defined or is None in training.optimizer.params config.")
        # Ensure LR is float for calculations later
        lr_value = float(params['lr'])
        eps_value = params.get('eps', 1e-8) # Default epsilon value for AdamW
        params['eps'] = float(eps_value) # Update params dict just in case
        params['lr'] = lr_value # Update params dict just in case

        # --- Check if using DeepSpeed ---
        is_deepspeed = isinstance(self.trainer.strategy, DeepSpeedStrategy) if self.trainer else False

        # dynamic optimizer lookup
        if name == 'FusedAdam':
            try:
                OptimClass = importlib.import_module('deepspeed.ops.adam').FusedAdam
                logging.info("Using FusedAdam optimizer.")
            except ImportError:
                logging.warning("FusedAdam not available, falling back to AdamW.")
                OptimClass = AdamW
        else:
            OptimClass = getattr(torch.optim, name, AdamW)
            logging.info(f"Using {OptimClass.__name__} optimizer.")

        # Initialize optimizer with checked params
        optimizer = OptimClass(self.parameters(), **params)

        # --- Scheduler Configuration ---
        if is_deepspeed:
            logging.info("DeepSpeed strategy detected. Optimizer configured, DeepSpeed will handle LR scheduling based on its configuration.")
            return optimizer

        sched_cfg = t.get('scheduler', {})
        scheduler_dict = None
        if sched_cfg and sched_cfg.get('name'):
            s_name = sched_cfg['name']
            s_params = sched_cfg['params'].copy()

            if s_params.get('total_steps') is None and s_name in ['cosine', 'linear_cosine']:
                 raise ValueError(f"Scheduler '{s_name}' requires 'total_steps'. It should have been calculated in the main script.")

            if s_name == 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=s_params['warmup_steps']
                )
                scheduler_dict = {'scheduler': scheduler, 'interval': 'step'}
            elif s_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=s_params['total_steps'],
                    eta_min=lr_value * s_params.get('min_lr_ratio', 0.0) # Use checked lr_value
                )
                scheduler_dict = {'scheduler': scheduler, 'interval': 'step'}
            elif s_name == 'linear_cosine':
                warmup_steps = s_params['warmup_steps']
                total_steps = s_params['total_steps']
                min_lr_ratio = s_params.get('min_lr_ratio', 0.0)

                if warmup_steps <= 0:
                     logging.info("Warmup steps is 0, using CosineAnnealingLR directly.")
                     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                         optimizer,
                         T_max=total_steps,
                         eta_min=lr_value * min_lr_ratio # Use checked lr_value
                     )
                elif warmup_steps >= total_steps:
                     logging.warning(f"Warmup steps ({warmup_steps}) >= total steps ({total_steps}). Using linear warmup for the entire duration.")
                     scheduler = torch.optim.lr_scheduler.LinearLR(
                         optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
                     )
                else:
                    warm = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-8,
                        end_factor=1.0,
                        total_iters=warmup_steps
                    )
                    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_steps - warmup_steps,
                        eta_min=lr_value * min_lr_ratio # Use checked lr_value
                    )
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warm, cosine],
                        milestones=[warmup_steps]
                    )
                scheduler_dict = {'scheduler': scheduler, 'interval': 'step'}
            else:
                raise ValueError(f"Unknown scheduler: {s_name}")

            logging.info(f"Configured PyTorch Lightning LR Scheduler: {s_name}")
            return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        else:
            logging.info("No LR scheduler configured in YAML.")
            return optimizer

def main():
    # Load config and set log level from YAML verbose flag
    cfg = load_config()
    
    # New: Get logging level from config string
    log_level_str = cfg.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid
    
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    t_cfg = cfg['training'] # Training config section

    # --- Strategy Setup ---
    strat_cfg = t_cfg.get('strategy', {})
    strat_name = strat_cfg.get('name')
    strat_params = strat_cfg.get('params', {})
    strategy = None
    is_deepspeed = False

    # Only initialize strategy if accelerator is not CPU
    accelerator = t_cfg.get('accelerator', 'auto')
    if accelerator != 'cpu' and strat_name:
        if strat_name == 'deepspeed':
             try:
                 strategy = DeepSpeedStrategy(**strat_params)
                 logging.info("Using DeepSpeed strategy.")
                 is_deepspeed = True
             except ImportError:
                 logging.warning("DeepSpeed strategy is not available. Falling back to default strategy.")
                 strategy = "auto"
             except Exception as e:
                 logging.error(f"Error initializing DeepSpeedStrategy: {e}. Falling back to default.")
                 strategy = "auto"
        elif strat_name == 'ddp':
             strategy = DDPStrategy(**strat_params)
             logging.info("Using DDP strategy.")
        else:
             logging.warning(f"Strategy '{strat_name}' requested but not recognized or supported. Using default.")
             strategy = "auto"
    elif accelerator == 'cpu':
        logging.info("Accelerator set to CPU. Ignoring strategy configuration.")
        strategy = "auto" # Let PL handle CPU case
    else: # No strategy name provided
         strategy = "auto"
         logging.info("No strategy specified or accelerator is CPU, using PyTorch Lightning default.")


    # --- Data Setup ---
    ds_cfg = cfg['dataset']
    # ... (Dataset loading and splitting remains the same) ...
    full_dataset = MIDIDatasetPreprocessed(
        sequence_length=ds_cfg['sequence_length'],
        preprocessed_dir=cfg['preprocessed_dir'],
        pad_idx=PAD_IDX,
        augmentation_shift=ds_cfg.get('augmentation_shift', 0)
    )

    # --- Split Dataset ---
    val_split_perc = ds_cfg.get('validation_split_percentage', 0)
    if not (0 <= val_split_perc < 100):
        logging.error("validation_split_percentage must be between 0 and 99. Setting to 0.")
        val_split_perc = 0

    if val_split_perc > 0:
        total_size = len(full_dataset)
        val_size = int(total_size * (val_split_perc / 100.0))
        train_size = total_size - val_size

        if train_size == 0 or val_size == 0:
             logging.warning(f"Dataset split resulted in zero samples for train ({train_size}) or validation ({val_size}). Disabling validation split.")
             train_dataset = full_dataset
             val_dataset = None # No validation set
        else:
            logging.info(f"Splitting dataset: {train_size} train, {val_size} validation samples.")
            generator = torch.Generator().manual_seed(42) # Example seed
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    else:
        logging.info("No validation split requested (validation_split_percentage is 0 or invalid). Using full dataset for training.")
        train_dataset = full_dataset
        val_dataset = None

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=t_cfg['batch_size_per_gpu'],
        num_workers=t_cfg['num_workers'],
        shuffle=True,
        pin_memory=(accelerator != 'cpu') # Pin memory only useful for GPU
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=t_cfg['batch_size_per_gpu'],
            num_workers=t_cfg['num_workers'],
            shuffle=False,
            pin_memory=(accelerator != 'cpu')
        )
        logging.info(f"Train loader size: {len(train_loader)} batches")
        logging.info(f"Validation loader size: {len(val_loader)} batches")
    else:
        logging.info(f"Train loader size: {len(train_loader)} batches (No validation loader)")


    # --- Calculate total_steps for scheduler if needed ---
    scheduler_needs_total_steps = (
        not is_deepspeed and
        t_cfg.get('scheduler', {}).get('name') in ['cosine', 'linear_cosine'] and
        t_cfg.get('scheduler', {}).get('params', {}).get('total_steps') is None
    )

    if scheduler_needs_total_steps:
        if t_cfg['max_epochs'] is None or t_cfg['max_epochs'] <= 0:
             raise ValueError("Cannot compute total_steps: training.max_epochs is not set or invalid.")
        if len(train_loader) == 0:
             raise ValueError("Cannot compute total_steps: train_loader has zero length.")
        steps_per_epoch = math.ceil(len(train_loader) / t_cfg['grad_accumulation_steps'])
        total_steps = steps_per_epoch * t_cfg['max_epochs']
        logging.info(f"Calculated total_steps for PL scheduler: {total_steps}")
        if 'scheduler' not in t_cfg:
            t_cfg['scheduler'] = {}
        if 'params' not in t_cfg['scheduler']:
            t_cfg['scheduler']['params'] = {}
        t_cfg['scheduler']['params']['total_steps'] = total_steps
    elif is_deepspeed:
        logging.info("Using DeepSpeed. Total steps for LR scheduling should be configured within the DeepSpeed config if needed.")


    # --- Model Instantiation ---
    # ... (Model instantiation remains the same) ...
    mcfg = cfg['model']
    decoder_params = mcfg.get('decoder_params', {})
    tw_params = mcfg.get('transformer_wrapper_params', {})
    aw_params = mcfg.get('autoregressive_wrapper_params', {})

    aw_params['pad_value'] = PAD_IDX
    aw_params['ignore_index'] = PAD_IDX

    logging.info(f"Decoder Params: {decoder_params}")
    logging.info(f"TransformerWrapper Params: {tw_params}")
    logging.info(f"AutoregressiveWrapper Params: {aw_params}")

    try:
        decoder = Decoder(**decoder_params)
        transformer_wrapper = TransformerWrapper(
            attn_layers =decoder, # Corrected param name
            **tw_params
        )
        model = AutoregressiveWrapper(
            transformer_wrapper,
            **aw_params
        )
        logging.info("Model created successfully.")
    except Exception as e:
         logging.error(f"Error creating model: {e}")
         raise

    # --- LightningModule ---
    lit = MidiLightningModule(model, cfg) # Pass potentially updated cfg

    # --- Logger & Checkpoint ---
    # Conditional WandB Logger
    logger = None
    if cfg.get('wandb', {}).get('enable', False):
        try:
            import wandb
            if 'api_key' in cfg['wandb']:
                wandb.login(key=cfg['wandb']['api_key'])
            run_name = cfg['wandb'].get('run_name', 'default_midi_run')
            logger = WandbLogger(
                project=cfg['wandb']['project_name'],
                name=run_name,
                log_model=cfg['wandb'].get('log_model', 'all'),
                save_dir=os.path.join(cfg['wandb']['save_dir'], run_name) if cfg['wandb'].get('save_dir') else None,
            )
            logging.info(f"WandB logging enabled for project '{cfg['wandb']['project_name']}', run '{run_name}'.")
        except ImportError:
            logging.warning("WandB logger specified but `wandb` package not found. Install with `pip install wandb`. Disabling logger.")
        except Exception as e:
            logging.error(f"Error initializing WandB logger: {e}. Disabling logger.")
    else:
        logging.info("WandB logging disabled.")

    # Checkpoint callback
    run_name_for_ckpt = cfg.get('wandb', {}).get('run_name', 'default_midi_run') # Use run name for consistency
    ckpt_filename = f'{run_name_for_ckpt}-{{epoch:02d}}-{{val_loss:.3f}}' if val_loader else f'{run_name_for_ckpt}-{{epoch:02d}}'
    monitor_metric = 'val_loss' if val_loader else None

    ckpt = ModelCheckpoint(
        dirpath=cfg['checkpoint_dir'],
        filename=ckpt_filename,
        save_top_k=1,
        monitor=monitor_metric,
        mode='min',
        save_last=True
    )
    callbacks = [ckpt]
    val_gen_cfg = t_cfg.get('validation_generation', {})
    if val_gen_cfg.get('enable', False) and val_loader is not None:
        # Get the new parameter from config, with a default if not specified
        generation_interval = val_gen_cfg.get('generation_epoch_interval', 0.25) 

        val_gen_callback = ValidationMIDIGenerationCallback(
            generation_dir=cfg.get('generation_dir', 'outputs/generated'),
            processor_config=cfg.get('processor', {}),
            num_samples=val_gen_cfg.get('num_samples', 2),
            temperature=val_gen_cfg.get('temperature', 1.0),
            prompt_tokens=val_gen_cfg.get('prompt_tokens', [1]),
            generate_length=val_gen_cfg.get('generate_length', 2048),
            max_generations=val_gen_cfg.get('max_generations', 5),
            generation_epoch_interval=generation_interval # Pass the new parameter
        )
        callbacks.append(val_gen_callback)
        logging.info(f"Validation MIDI generation enabled: {val_gen_cfg.get('num_samples')} samples per validation, interval: {generation_interval} epochs.")
    

    resume_ckpt_path = t_cfg.get('resume_from_checkpoint', None)
    if resume_ckpt_path and not os.path.exists(resume_ckpt_path):
        logging.warning(f"Checkpoint path specified for resume ({resume_ckpt_path}) does not exist. Starting new training.")
        resume_ckpt_path = None
    elif resume_ckpt_path:
        logging.info(f"Resuming training from checkpoint: {resume_ckpt_path}")

    trainer = Trainer(
        logger=logger, # Pass the potentially None logger
        callbacks=callbacks,
        accelerator=accelerator, # Use configured accelerator
        strategy=strategy, # Use configured strategy (or auto)
        max_epochs=t_cfg['max_epochs'],
        precision=t_cfg['precision'] if accelerator != 'cpu' else '32-true', # Precision ignored on CPU, set default
        accumulate_grad_batches=t_cfg['grad_accumulation_steps'],
        gradient_clip_val=t_cfg['gradient_clip_val'],
        val_check_interval=t_cfg.get('val_check_interval', 1.0), # Use float for fraction/int for epochs
        num_sanity_val_steps=t_cfg.get('num_sanity_val_steps', 2) if val_loader else 0,
        log_every_n_steps=t_cfg.get('log_every_n_steps', 50),
        # deterministic=t_cfg.get('deterministic', False)
        
    )

    # --- Refined total_steps calculation (if needed after trainer init) ---
    # This part is generally fine, but ensure it only runs if PL scheduler is used
    if scheduler_needs_total_steps:
         # Recalculate using trainer info if necessary, though previous calc might be sufficient
         steps_per_epoch = math.ceil(len(train_loader) / t_cfg['grad_accumulation_steps']) # Batches per device / accum
         # If using DDP, len(train_loader) is already per device.
         # If not using DDP, num_devices is 1. So this calculation is likely okay.
         total_steps = steps_per_epoch * t_cfg['max_epochs']
         if cfg['training']['scheduler']['params']['total_steps'] != total_steps:
              logging.info(f"Refined total_steps calculation for scheduler: {total_steps}")
              cfg['training']['scheduler']['params']['total_steps'] = total_steps


    logging.info("Starting training...")
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt_path)
    logging.info("Training finished.")


if __name__=="__main__":
    main()