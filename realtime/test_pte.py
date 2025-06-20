import torch
import torchaudio
import numpy as np
import argparse
import os
from pathlib import Path

try:
    MATPLOTLIB_AVAILABLE = True
    import matplotlib.pyplot as plt
except (IndexError, ModuleNotFoundError):
    MATPLOTLIB_AVAILABLE = False

try:
    from executorch.runtime import Runtime
    EXECUTORCH_AVAILABLE = True
except ImportError:
    EXECUTORCH_AVAILABLE = False
    print("Warning: ExecuTorch runtime not available. Install with: pip install executorch")


def load_and_process_audio(audio_file: str, sampling_rate: int, chunk_size: int):
    """Load audio file and split into chunks for processing."""
    print(f"Loading audio file: {audio_file}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_file)
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sampling_rate:
        print(f"Resampling from {sr}Hz to {sampling_rate}Hz")
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        waveform = resampler(waveform)
    
    # Normalize to [-1, 1]
    waveform = waveform.clamp(-1, 1)
    
    # Split into chunks
    num_samples = waveform.size(1)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    print(f"Audio length: {num_samples} samples ({num_samples/sampling_rate:.2f}s)")
    print(f"Processing in {num_chunks} chunks of {chunk_size} samples")
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        
        chunk = waveform[:, start_idx:end_idx]
        
        # Pad last chunk if needed
        if chunk.size(1) < chunk_size:
            padding = chunk_size - chunk.size(1)
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        
        chunks.append(chunk)
    
    return chunks, sampling_rate


def run_executorch_inference(model_path: str, audio_chunks: list, chunk_size: int):
    """Run inference using ExecuTorch model."""
    if not EXECUTORCH_AVAILABLE:
        raise RuntimeError("ExecuTorch runtime not available")
    
    print(f"Loading ExecuTorch model: {model_path}")
    
    # Load ExecuTorch model
    runtime = Runtime.get()
    
    program = runtime.load_program(model_path)
    method = program.load_method("forward")
    
    all_predictions = []
    all_confidence = []
    all_volume = []
    all_activations = []
    
    print("Running inference...")
    for i, chunk in enumerate(audio_chunks):
        if i % 10 == 0:
            print(f"Processing chunk {i+1}/{len(audio_chunks)}")
        
        # Run inference - ExecuTorch outputs are already on CPU
        outputs = method.execute([chunk])
        
        # Extract outputs (pred, conf, vol, act)
        # Note: ExecuTorch outputs are EValues, need to convert to numpy
        if len(outputs) > 0 and outputs[0] is not None:
            pred_np = outputs[0].to_numpy() if hasattr(outputs[0], 'to_numpy') else outputs[0].numpy()
            all_predictions.append(pred_np)
            
        if len(outputs) > 1 and outputs[1] is not None:
            conf_np = outputs[1].to_numpy() if hasattr(outputs[1], 'to_numpy') else outputs[1].numpy()
            all_confidence.append(conf_np)
            
        if len(outputs) > 2 and outputs[2] is not None:
            vol_np = outputs[2].to_numpy() if hasattr(outputs[2], 'to_numpy') else outputs[2].numpy()
            all_volume.append(vol_np)
            
        if len(outputs) > 3 and outputs[3] is not None:
            act_np = outputs[3].to_numpy() if hasattr(outputs[3], 'to_numpy') else outputs[3].numpy()
            all_activations.append(act_np)
    
    # Concatenate results
    results = {}
    if all_predictions:
        results['predictions'] = np.concatenate(all_predictions, axis=-1)
    if all_confidence:
        results['confidence'] = np.concatenate(all_confidence, axis=-1)
    if all_volume:
        results['volume'] = np.concatenate(all_volume, axis=-1)
    if all_activations:
        results['activations'] = np.concatenate(all_activations, axis=1)
    
    return results


def create_visualization(results: dict, output_path: str, sampling_rate: int, chunk_size: int):
    """Create PNG visualization of the results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Matplotlib not available. Skipping PNG generation.")
        return
    
    # Calculate time steps
    step_size_ms = 1000 * chunk_size / sampling_rate
    if 'predictions' in results:
        num_frames = results['predictions'].shape[-1]
        timesteps = np.arange(num_frames) * step_size_ms
    else:
        print("No predictions found for visualization")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Pitch predictions
    if 'predictions' in results:
        predictions = results['predictions'].squeeze()
        axes[0].plot(timesteps / 1000, predictions, 'b-', linewidth=1)
        axes[0].set_ylabel('Pitch (Hz)')
        axes[0].set_title('Pitch Predictions')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence
    if 'confidence' in results:
        confidence = results['confidence'].squeeze()
        axes[1].plot(timesteps / 1000, confidence, 'g-', linewidth=1)
        axes[1].set_ylabel('Confidence')
        axes[1].set_title('Voiced/Unvoiced Confidence')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Activations heatmap
    if 'activations' in results and 'confidence' in results:
        activations = results['activations'].squeeze()
        confidence = results['confidence'].squeeze()
        
        # Apply confidence weighting
        weighted_activations = activations * confidence[:, None]
        
        # Crop to reasonable pitch range (C1 to A8, MIDI 24-108)
        bps = activations.shape[1] // 128  # bins per semitone
        lims = (21, 109)  # semitone range
        cropped_activations = weighted_activations[:, bps*lims[0]:bps*lims[1]]
        
        im = axes[2].imshow(cropped_activations.T,
                           aspect='auto', origin='lower', cmap='inferno',
                           extent=(timesteps[0] / 1000, timesteps[-1] / 1000, lims[0], lims[1]))
        axes[2].set_ylabel('Pitch (semitones)')
        axes[2].set_title('Pitch Activations')
        plt.colorbar(im, ax=axes[2])
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test ExecuTorch PESTO model on audio file")
    parser.add_argument("model_path", type=str, help="Path to .pte model file")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG path")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
    
    # Set default output path
    if args.output is None:
        audio_name = Path(args.audio_file).stem
        model_name = Path(args.model_path).stem
        args.output = f"{audio_name}_{model_name}_pitch.png"
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and process audio
        audio_chunks, sr = load_and_process_audio(args.audio_file, args.sampling_rate, args.chunk_size)
        
        # Run inference
        results = run_executorch_inference(args.model_path, audio_chunks, args.chunk_size)
        
        # Print results summary
        print("\nResults summary:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                if key == 'predictions':
                    valid_preds = value[value > 0]  # Filter out silence/unvoiced
                    if len(valid_preds) > 0:
                        print(f"    Valid pitch range: {valid_preds.min():.1f} - {valid_preds.max():.1f} Hz")
        
        # Create visualization
        create_visualization(results, args.output, args.sampling_rate, args.chunk_size)
        
        # Save raw results
        npz_path = args.output.replace('.png', '.npz')
        np.savez(npz_path, **results)
        print(f"Raw results saved to: {npz_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()