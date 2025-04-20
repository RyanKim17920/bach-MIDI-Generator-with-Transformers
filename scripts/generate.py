#!/usr/bin/env python3
import os
import importlib
import logging
import yaml
import torch

from src.midi_processor import MIDIProcessor


def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    # Set log level from YAML verbose flag
    log_level = logging.DEBUG if cfg.get('verbose', False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    gen_cfg = cfg.get('generation', {})
    # Determine checkpoint path
    ckpt = gen_cfg.get('checkpoint_path')
    if not ckpt:
        ckpt_dir = cfg.get('checkpoint_dir')
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') or f.endswith('.pt')]
        if not files:
            logging.error("No checkpoint files found in %s", ckpt_dir)
            return
        files.sort()
        ckpt = os.path.join(ckpt_dir, files[0])
    logging.info(f"Using checkpoint: {ckpt}")

    # Load model dynamically
    mcfg = cfg['model']
    mod = importlib.import_module(mcfg['module'])
    ModelClass = getattr(mod, mcfg['class'])
    decoder = ModelClass(**mcfg.get('params', {}))
    wcfg = cfg.get('wrapper', {})
    wmod = importlib.import_module(wcfg.get('module', 'x_transformers'))
    WrapperClass = getattr(wmod, wcfg.get('class', 'AutoregressiveWrapper'))
    model = WrapperClass(decoder, **wcfg.get('params', {}))

    # Load state_dict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval().to(device)

    # Initialize processor
    proc = MIDIProcessor(**cfg['processor'])

    # Prepare output directory
    out_dir = cfg.get('generation_dir', 'outputs/generated')
    os.makedirs(out_dir, exist_ok=True)

    # Prepare prompt sequence
    if gen_cfg.get('prompt_midi_file'):
        res = proc.process_midi_file(gen_cfg['prompt_midi_file'])
        if not res:
            logging.error("Failed to process prompt MIDI file: %s", gen_cfg['prompt_midi_file'])
            return
        prompt_tokens = res['tokens']
    else:
        prompt_tokens = gen_cfg.get('prompt_tokens', [proc.START])
    prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    num_samples = gen_cfg.get('num_samples', 1)
    seq_len = gen_cfg.get('generate_length', cfg['dataset']['sequence_length'])
    temperature = gen_cfg.get('temperature', 1.0)

    # Generate samples
    for i in range(num_samples):
        logging.info(f"Generating sample {i+1}/{num_samples}...")
        generated = model.generate(prompts=prompt, seq_len=seq_len, temperature=temperature)
        tokens = generated[0].cpu().numpy()
        midi_path = os.path.join(out_dir, f"sample_{i+1}.mid")
        proc.tokens_to_midi(tokens, save_path=midi_path)
        logging.info(f"Saved generated MIDI to {midi_path}")


if __name__ == '__main__':
    main()