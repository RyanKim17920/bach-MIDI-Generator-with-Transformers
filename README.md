# MIDI Generator with Transformers

**A work‑in‑progress**: convert MIDI → token sequences → train a Transformer → regenerate MIDI.

## Features

- Flexible tokenization with `MIDIProcessor`
- On‑the‑fly or preprocessed PyTorch `Dataset` (`MIDIDataset`, `MIDIDatasetPreprocessed`)
- Config‑driven training with PyTorch Lightning + Deepspeed / DDP
- Full round‑trip: MIDI → tokens → MIDI
- Unit tests with `pytest`

## Requirements

```bash
pip install -r requirements.txt
```

## Quickstart

1. Edit config/config.yaml (set raw_midi_dir, preprocessed_dir, model & training params).
2. Preprocess:

```bash
python scripts/preprocess_data.py
```

3. Train:

```bash
python scripts/train_model.py
```

4. Check reconstruction:

```bash
python scripts/check_reconstruction.py
```

## Testing

Testing

```bash
pytest tests/
```