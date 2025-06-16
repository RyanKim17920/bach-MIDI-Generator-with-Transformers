#!/usr/bin/env python3
import os
# Remove importlib as it's no longer needed for dynamic loading
# import importlib
import logging
import yaml
import torch

# --- Add direct imports for the fixed model structure ---
from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper

from src.midi_processor import MIDIProcessor
from src.constants import PAD_IDX # Import PAD_IDX if needed for model setup (though likely handled by wrapper)

def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        # Basic interpolation for dataset.sequence_length
        content = f.read()
        cfg_raw = yaml.safe_load(content) # Load once to get seq_len
        # Use a default if dataset or sequence_length is missing
        seq_len = cfg_raw.get('dataset', {}).get('sequence_length', 2048)
        content = content.replace('${dataset.sequence_length}', str(seq_len))
        # Load again with interpolation applied
        return yaml.safe_load(content)


def main():
    cfg = load_config()
    # New: Get logging level from config string
    log_level_str = cfg.get('logging_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    gen_cfg = cfg.get('generation', {})
    # Determine checkpoint path
    ckpt_path = gen_cfg.get('checkpoint_path') # Renamed variable for clarity
    if not ckpt_path:
        ckpt_dir = cfg.get('checkpoint_dir')
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
             logging.error(f"Checkpoint directory '{ckpt_dir}' not found or not specified.")
             return
        # Filter for .ckpt files and sort (e.g., by modification time or name)
        files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not files:
            logging.error("No .ckpt checkpoint files found in %s", ckpt_dir)
            return
        # Example: sort by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        ckpt_path = files[0] # Use the most recent checkpoint
        logging.info(f"No specific checkpoint path provided, using the latest found: {ckpt_path}")
    elif not os.path.exists(ckpt_path):
        logging.error(f"Specified checkpoint path does not exist: {ckpt_path}")
        return

    logging.info(f"Using checkpoint: {ckpt_path}")

    # --- Simplified Model Instantiation ---
    mcfg = cfg['model']
    decoder_params = mcfg.get('decoder_params', {})
    tw_params = mcfg.get('transformer_wrapper_params', {})
    aw_params = mcfg.get('autoregressive_wrapper_params', {})

    # Set pad_value for AutoregressiveWrapper if needed (might not be strictly necessary for generation)
    aw_params['pad_value'] = PAD_IDX

    logging.info(f"Decoder Params: {decoder_params}")
    logging.info(f"TransformerWrapper Params: {tw_params}")
    logging.info(f"AutoregressiveWrapper Params: {aw_params}")

    # Instantiate the nested model structure directly
    try:
        decoder = Decoder(**decoder_params)
        transformer_wrapper = TransformerWrapper(
            attn_layers=decoder,
            **tw_params
        )
        model = AutoregressiveWrapper(
            transformer_wrapper,
            **aw_params
        )
        logging.info("Model structure created successfully.")
    except Exception as e:
         logging.error(f"Error creating model structure: {e}")
         raise

    # --- End Simplified Model Instantiation ---

    # Load state_dict from Lightning checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading checkpoint on device: {device}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract the model's state_dict (it's nested under 'model.' prefix)
    # Check if 'state_dict' key exists (standard for PL checkpoints)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Filter keys to remove the 'model.' prefix
        model_state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items() if k.startswith('model.')}

        if not model_state_dict:
             logging.error("Could not find keys starting with 'model.' in the checkpoint's state_dict. Checkpoint structure might be different.")
             # Fallback: Try loading the whole state_dict directly if no 'model.' keys found
             logging.warning("Attempting to load the entire state_dict directly into the model.")
             model_state_dict = state_dict # Use the raw state_dict as a fallback

        try:
            model.load_state_dict(model_state_dict, strict=True) # Use strict=True to catch mismatches
            logging.info("Successfully loaded model state_dict from checkpoint.")
        except RuntimeError as e:
            logging.error(f"Error loading state_dict: {e}")
            logging.warning("Attempting to load with strict=False. Some weights might be missing or unexpected.")
            # Try loading non-strictly as a last resort
            model.load_state_dict(model_state_dict, strict=False)

    else:
        # If 'state_dict' key is missing, assume the checkpoint *is* the state_dict
        logging.warning("Checkpoint does not contain a 'state_dict' key. Assuming the file is the state_dict itself.")
        model.load_state_dict(checkpoint) # Try loading directly

    model.eval().to(device)
    logging.info("Model loaded and set to evaluation mode.")

    # Initialize processor
    try:
        proc = MIDIProcessor(**cfg['processor'])
        # Assuming START token ID is accessible like this, adjust if needed
        start_token_id = proc.start_token_id
    except AttributeError:
        logging.error("Could not find 'start_token_id' in MIDIProcessor. Please ensure it's defined.")
        # Fallback or default if necessary, though using the processor's constant is best
        start_token_id = 1 # Example fallback, adjust as needed
        logging.warning(f"Using default START token ID: {start_token_id}")
    except Exception as e:
        logging.error(f"Error initializing MIDIProcessor: {e}")
        return


    # Prepare output directory
    out_dir = cfg.get('generation_dir', 'outputs/generated')
    os.makedirs(out_dir, exist_ok=True)

    # Prepare prompt sequence
    prompt_tokens = None
    if gen_cfg.get('prompt_midi_file'):
        prompt_midi_path = gen_cfg['prompt_midi_file']
        if os.path.exists(prompt_midi_path):
            logging.info(f"Processing prompt MIDI file: {prompt_midi_path}")
            res = proc.process_midi_file(prompt_midi_path)
            if not res or 'tokens' not in res:
                logging.error("Failed to process prompt MIDI file or extract tokens: %s", prompt_midi_path)
                return
            prompt_tokens = res['tokens']
            logging.info(f"Using prompt tokens from MIDI file (length: {len(prompt_tokens)})")
        else:
            logging.error(f"Prompt MIDI file not found: {prompt_midi_path}")
            return
    else:
        # Use configured prompt tokens or default to START token
        prompt_tokens = gen_cfg.get('prompt_tokens', [start_token_id])
        logging.info(f"Using configured/default prompt tokens: {prompt_tokens}")

    # Ensure prompt is not empty
    if not prompt_tokens:
        logging.error("Prompt tokens are empty. Cannot generate.")
        return

    prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    num_samples = gen_cfg.get('num_samples', 1)
    # generate_length in config likely means *total* sequence length
    total_seq_len = gen_cfg.get('generate_length', cfg.get('dataset', {}).get('sequence_length', 1024))
    temperature = gen_cfg.get('temperature', 1.0)
    # Add filter_thres if needed, example:
    # filter_thres = gen_cfg.get('filter_thres', 0.9)

    # Generate samples
    for i in range(num_samples):
        logging.info(f"Generating sample {i+1}/{num_samples} (Total length: {total_seq_len}, Temp: {temperature})...")
        try:
            # AutoregressiveWrapper.generate expects seq_len to be the *total* length
            generated_output = model.generate(
                prompts=prompt,
                seq_len=total_seq_len,
                temperature=temperature,
                # filter_logits_fn='top_k', # Example filter function
                # filter_thres=filter_thres, # Example threshold
                # Add other generation args as needed from AutoregressiveWrapper documentation
            )
            # Output shape is typically (batch_size, sequence_length)
            tokens = generated_output[0].cpu().numpy() # Get the first sample in the batch

            # Ensure the output directory exists before saving
            os.makedirs(out_dir, exist_ok=True)
            midi_filename = f"sample_{i+1}_len{total_seq_len}_temp{temperature:.2f}.mid"
            midi_path = os.path.join(out_dir, midi_filename)

            logging.info(f"Converting {len(tokens)} tokens to MIDI...")
            proc.tokens_to_midi(tokens, save_path=midi_path)
            logging.info(f"Saved generated MIDI to {midi_path}")

        except Exception as e:
            logging.error(f"Error during generation or saving sample {i+1}: {e}")
            # Continue to the next sample if one fails
            continue


if __name__ == '__main__':
    main()