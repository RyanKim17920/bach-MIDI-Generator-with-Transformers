from setuptools import setup, find_packages

setup(
    name="bach_midi_generator",
    version="0.1.0",
    description="MIDI Generator with Transformers",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    scripts=[
        "scripts/preprocess_data.py",
        "scripts/train_model.py",
        "scripts/generate.py",
        "scripts/check_reconstruction.py",
    ],
    install_requires=[
        # core dependencies
        "pretty_midi>=0.2.10",
        "numpy>=1.23.0",
        "torch>=1.13.0",
        "pytorch-lightning>=2.0.0",
        "deepspeed>=0.9.0",
        "x-transformers>=0.37.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "torchmetrics>=0.11.0",
        "wandb>=0.15.0",
    ],
)
