import os
import torch
import logging
from pytorch_lightning.callbacks import Callback
import time
from src.midi_processor import MIDIProcessor

class ValidationMIDIGenerationCallback(Callback):
    def __init__(self, 
                 generation_dir, 
                 processor_config,
                 num_samples=2, 
                 temperature=1.0,
                 prompt_tokens=None, 
                 generate_length=2048,
                 max_generations=5,
                 generation_epoch_interval=0.25): # New parameter
        """
        Callback for generating MIDI samples during validation at specified epoch intervals.
        
        Args:
            generation_dir (str): Directory to save generated MIDI files
            processor_config (dict): Configuration for MIDIProcessor
            num_samples (int): Number of samples to generate per validation check
            temperature (float): Temperature for sampling
            prompt_tokens (list): Tokens to use as prompt for generation
            generate_length (int): Length of sequence to generate
            max_generations (int): Maximum number of validation generations to perform during training
            generation_epoch_interval (float): Interval in epochs for MIDI generation (e.g., 0.25 for every quarter epoch)
        """
        super().__init__()
        self.generation_dir = os.path.join(generation_dir, "validation_samples")
        self.processor_config = processor_config
        self.num_samples = num_samples
        self.temperature = temperature
        self.prompt_tokens = prompt_tokens if prompt_tokens else [1]  # Default to START token
        self.generate_length = generate_length
        self.max_generations = max_generations
        self.generation_count = 0
        
        self.generation_epoch_interval = generation_epoch_interval
        if self.generation_epoch_interval <= 0:
            logging.warning(
                f"ValidationMIDIGenerationCallback: generation_epoch_interval ({self.generation_epoch_interval}) "
                "is not positive. Interval-based MIDI generation will be disabled."
            )
            # Set to infinity to effectively disable interval-based generation
            self.generation_epoch_interval = float('inf') 

        self.next_generation_epoch_target = 0.0  # Will be properly set in on_fit_start
        self.num_optimizer_steps_per_epoch = 0
        
        os.makedirs(self.generation_dir, exist_ok=True)
        
        try:
            self.processor = MIDIProcessor(**self.processor_config)
        except Exception as e:
            logging.error(f"Failed to initialize MIDIProcessor: {e}")
            self.processor = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize settings at the start of training."""
        if self.generation_epoch_interval == float('inf') or not self.processor:
            # Interval generation is disabled or processor failed
            return

        if hasattr(trainer, 'num_training_batches'):
            self.num_optimizer_steps_per_epoch = trainer.num_training_batches
        else: # Fallback for older PyTorch Lightning or different setups
            if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None and \
               hasattr(trainer, 'accumulate_grad_batches') and len(trainer.train_dataloader) > 0:
                self.num_optimizer_steps_per_epoch = len(trainer.train_dataloader) // trainer.accumulate_grad_batches
            else:
                self.num_optimizer_steps_per_epoch = 0
        
        if self.num_optimizer_steps_per_epoch == 0:
            logging.warning(
                "ValidationMIDIGenerationCallback: Could not determine num_optimizer_steps_per_epoch. "
                "Interval-based MIDI generation will be disabled."
            )
            self.generation_epoch_interval = float('inf') # Disable
            return
        
        # Set the first target for generation after the first interval of actual training
        self.next_generation_epoch_target = self.generation_epoch_interval
        logging.info(
            f"ValidationMIDIGenerationCallback initialized. MIDI will be generated every "
            f"{self.generation_epoch_interval} epochs. First generation target at epoch progress: "
            f"{self.next_generation_epoch_target:.2f}"
        )

    def _generate_samples(self, trainer, pl_module, current_epoch_exact_progress):
        """Helper method to perform the actual MIDI generation."""
        current_epoch_int = trainer.current_epoch
        global_step_int = trainer.global_step

        # Create subdirectory for this generation event
        # Include exact progress in folder name for clarity
        val_dir_name = f"epoch_{current_epoch_int}_step_{global_step_int}_progress_{current_epoch_exact_progress:.3f}"
        val_dir = os.path.join(self.generation_dir, val_dir_name)
        os.makedirs(val_dir, exist_ok=True)
        
        logging.info(
            f"Generating {self.num_samples} validation samples (Epoch Progress ~{current_epoch_exact_progress:.3f}, "
            f"Generation {self.generation_count + 1}/{self.max_generations}). Saving to: {val_dir}"
        )
        
        device = pl_module.device
        prompt = torch.tensor(self.prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        original_training_state = pl_module.model.training
        pl_module.model.eval()
        
        with torch.no_grad():
            for i in range(self.num_samples):
                try:
                    start_time = time.time()
                    generated = pl_module.model.generate(
                        prompts=prompt,
                        seq_len=self.generate_length,
                        temperature=self.temperature
                    )
                    tokens = generated[0].cpu().numpy()
                    
                    midi_filename = f"sample_{i+1}_epoch_{current_epoch_int}_step_{global_step_int}.mid"
                    midi_path = os.path.join(val_dir, midi_filename)
                    self.processor.tokens_to_midi(tokens, save_path=midi_path)
                    
                    generation_time = time.time() - start_time
                    logging.info(f"Generated validation sample {i+1}/{self.num_samples} in {generation_time:.2f}s - {midi_path}")
                except Exception as e:
                    logging.error(f"Error generating validation sample {i+1}: {e}")
                    continue
        
        if original_training_state: # Restore original training state
            pl_module.model.train()

    def on_validation_end(self, trainer, pl_module):
        """Generate MIDI samples at the end of validation if interval is met."""
        if trainer.sanity_checking:
            logging.debug("Skipping MIDI generation during sanity check.")
            return
        
        if not self.processor or self.generation_epoch_interval == float('inf'):
            # Callback not properly initialized or interval generation disabled
            return

        if self.generation_count >= self.max_generations:
            # Max generations already reached
            if self.generation_count == self.max_generations and not hasattr(self, '_max_gen_logged'):
                logging.info(f"Max generations ({self.max_generations}) reached. No more validation MIDI will be generated.")
                self._max_gen_logged = True # Log only once
            return
            
        if self.num_optimizer_steps_per_epoch == 0:
            # This should have been caught in on_fit_start, but as a safeguard
            return

        current_epoch_exact_progress = trainer.global_step / self.num_optimizer_steps_per_epoch
        
        generated_in_this_call = False
        while current_epoch_exact_progress >= self.next_generation_epoch_target:
            if self.generation_count >= self.max_generations:
                break 

            self._generate_samples(trainer, pl_module, current_epoch_exact_progress)
            
            self.generation_count += 1
            self.next_generation_epoch_target += self.generation_epoch_interval
            generated_in_this_call = True

            if self.generation_count >= self.max_generations:
                logging.info(f"Max generations ({self.max_generations}) reached after current generation.")
                break 
        
        if generated_in_this_call:
            logging.debug(f"Next MIDI generation target epoch progress: {self.next_generation_epoch_target:.3f}")
