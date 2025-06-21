# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PESTO (Pitch Estimation with Self-supervised Transposition-equivariant Objective) is a Python library for fast and accurate pitch estimation in audio files. It uses PyTorch-based deep learning models for real-time pitch detection.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv package manager
uv sync

# Install in development mode
uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_cli.py
uv run pytest tests/test_performances.py
uv run pytest tests/test_stream.py

# Run with verbose output
uv run pytest -v
```

### Running the Application
```bash
# CLI usage
pesto my_file.wav

# Or using Python module
uv run python -m pesto my_file.wav

# Real-time processing examples
uv run python realtime/test.py
uv run python realtime/benchmark_kernels.py
```

## Architecture

### Core Components (`/pesto/`)
- **`core.py`**: Main inference functions (`predict`, `predict_from_files`)
- **`model.py`**: Neural network architecture (Resnet1d with Toeplitz layers)
- **`loader.py`**: Model loading utilities
- **`weights/`**: Pre-trained checkpoints (`mir-1k.ckpt`, `mir-1k_g7.ckpt`)

### Signal Processing (`/pesto/utils/`)
- **`hcqt.py`**: Harmonic Constant-Q Transform implementation
- **`cached_conv.py`**: Optimized convolution operations
- **`reduce_activations.py`**: Post-processing for pitch predictions

### Real-time Processing (`/realtime/`)
- **`export_jit.py`**: JIT model compilation for reduced latency
- **`speed.py`**: Performance benchmarking utilities
- **Streaming implementation**: `StreamingVQT` for real-time audio processing

## Key Technical Details

- **Input**: Audio files processed via Harmonic Constant-Q Transform (HCQT)
- **Model**: CNN-based architecture with residual connections
- **Output**: Pitch probabilities converted to frequencies using weighted averaging
- **Performance**: ~12x faster than real-time processing with GPU acceleration
- **Streaming**: Circular buffer implementation for continuous audio processing

## Testing Structure

- **`tests/test_cli.py`**: Command-line interface testing
- **`tests/test_performances.py`**: Performance benchmarks
- **`tests/test_stream.py`**: Streaming functionality
- **`tests/test_shape.py`**: Output shape validation
- **`tests/audios/example.wav`**: Sample audio for testing

## Dependencies

- Uses **uv** for dependency management (see `uv.lock`)
- Core: PyTorch, torchaudio, numpy, scipy
- Testing: pytest (>=8.3.5)
- Optional: matplotlib for visualization