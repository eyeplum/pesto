#!/usr/bin/env python3
"""
Test script to plot pitch contour and confidence using ExecuTorch (.pte) model for inference.
"""

import argparse
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from realtime.export_pte import export_model
from pesto.loader import load_model


def create_executorch_runtime(pte_model_path):
    """Create ExecuTorch runtime and load program."""
    try:
        from executorch.runtime import Runtime
        
        runtime = Runtime.get()
        program = runtime.load_program(pte_model_path)
        method = program.load_method("forward")
        return method
    except ImportError:
        raise ImportError("ExecuTorch runtime not available. Install executorch runtime.")


def plot_pitch_contour_comparison(audio_path, pte_model_path=None, checkpoint_name="mir-1k_g7", 
                                 output_path=None, sampling_rate=48000, chunk_size=960):
    """
    Plot pitch contour and confidence comparing ExecuTorch and Torch model inference.
    
    Args:
        audio_path: Path to input audio file
        pte_model_path: Path to .pte model file (if None, exports from checkpoint)
        checkpoint_name: Checkpoint name to export if pte_model_path is None
        output_path: Path to save plot (if None, displays plot)
        sampling_rate: Target sampling rate for processing
        chunk_size: Chunk size for processing
    """
    print(f"Loading audio: {audio_path}")
    
    # Load audio file
    try:
        waveform, original_sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)  # Remove channel dimension if mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)  # Convert to mono if stereo
        print(f"Audio loaded: {len(waveform)} samples at {original_sr} Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, None
    
    # Setup ExecuTorch model
    if pte_model_path is None or not os.path.exists(pte_model_path):
        print(f"Exporting ExecuTorch model from checkpoint: {checkpoint_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pte_path = os.path.join(temp_dir, "temp_model.pte")
            try:
                _, pte_path = export_model(checkpoint_name, sampling_rate, chunk_size, temp_pte_path)
                method = create_executorch_runtime(pte_path)
            except Exception as e:
                print(f"Error exporting ExecuTorch model: {e}")
                return None, None, None
    else:
        print(f"Loading ExecuTorch model: {pte_model_path}")
        try:
            method = create_executorch_runtime(pte_model_path)
        except Exception as e:
            print(f"Error loading ExecuTorch model: {e}")
            return None, None, None
    
    # Resample audio if needed
    if original_sr != sampling_rate:
        print(f"Resampling from {original_sr} Hz to {sampling_rate} Hz")
        resampler = torchaudio.transforms.Resample(original_sr, sampling_rate)
        waveform = resampler(waveform)
    
    # Process audio in chunks with overlap
    hop_size = chunk_size // 2  # 50% overlap
    num_chunks = max(1, (len(waveform) - chunk_size) // hop_size + 1)
    
    print(f"Processing {num_chunks} chunks of size {chunk_size}")
    
    # ExecuTorch predictions
    pitch_predictions_pte = []
    confidence_predictions_pte = []
    volume_predictions_pte = []
    time_stamps = []
    
    # Torch predictions - process in streaming chunks (same as ExecuTorch)
    print("Running Torch model inference...")
    step_size = 1000 * chunk_size / sampling_rate
    torch_model = load_model(
        checkpoint_name, 
        step_size=step_size, 
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=1
    )
    torch_model.eval()
    
    # Process Torch model in same chunks as ExecuTorch
    pitch_predictions_torch = []
    confidence_predictions_torch = []
    volume_predictions_torch = []
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * hop_size
            end_idx = start_idx + chunk_size
            
            # Extract chunk with padding if necessary (same as ExecuTorch)
            if end_idx > len(waveform):
                chunk = torch.zeros(chunk_size)
                available_samples = len(waveform) - start_idx
                if available_samples > 0:
                    chunk[:available_samples] = waveform[start_idx:]
            else:
                chunk = waveform[start_idx:end_idx]
            
            # Prepare input for Torch model (add batch dimension)
            chunk_input = chunk.unsqueeze(0)
            
            # Run Torch inference
            pred, conf, vol, act = torch_model(chunk_input)
            
            # Extract outputs (remove batch dimension)
            pitch_predictions_torch.append(pred[0].item())
            confidence_predictions_torch.append(conf[0].item())
            volume_predictions_torch.append(vol[0].item() if vol.numel() > 0 else 0.0)
    
    # Convert to numpy arrays
    torch_pitch_predictions = np.array(pitch_predictions_torch)
    torch_confidence_predictions = np.array(confidence_predictions_torch)
    torch_time_stamps = np.array(time_stamps)  # Use same timestamps as ExecuTorch
    
    print(f"Torch model generated {len(torch_pitch_predictions)} predictions")
    print("Running ExecuTorch model inference...")
    
    for i in range(num_chunks):
        start_idx = i * hop_size
        end_idx = start_idx + chunk_size
        
        # Extract chunk with padding if necessary
        if end_idx > len(waveform):
            chunk = torch.zeros(chunk_size)
            available_samples = len(waveform) - start_idx
            if available_samples > 0:
                chunk[:available_samples] = waveform[start_idx:]
        else:
            chunk = waveform[start_idx:end_idx]
        
        # Prepare input for ExecuTorch model (add batch dimension)
        chunk_input = chunk.unsqueeze(0)
        
        try:
            # Run ExecuTorch inference
            outputs = method.execute([chunk_input])
            
            # Extract outputs: [pred, conf, vol, act]
            pred = outputs[0][0].item()  # Remove batch dimension and convert to scalar
            conf = outputs[1][0].item()
            vol = outputs[2][0].item() if len(outputs) > 2 else 0.0
            
            pitch_predictions_pte.append(pred)
            confidence_predictions_pte.append(conf)
            volume_predictions_pte.append(vol)
            
            # Calculate time stamp for this chunk center
            time_stamp = (start_idx + chunk_size // 2) / sampling_rate
            time_stamps.append(time_stamp)
            
        except Exception as e:
            print(f"Error during inference at chunk {i}: {e}")
            break
    
    if not pitch_predictions_pte:
        print("No ExecuTorch predictions generated")
        return None, None, None
    
    # Convert ExecuTorch predictions to numpy arrays
    pitch_predictions_pte = np.array(pitch_predictions_pte)
    confidence_predictions_pte = np.array(confidence_predictions_pte)
    volume_predictions_pte = np.array(volume_predictions_pte)
    time_stamps = np.array(time_stamps)
    
    print(f"ExecuTorch model generated {len(pitch_predictions_pte)} predictions")
    print(f"ExecuTorch Pitch range: {pitch_predictions_pte.min():.1f} - {pitch_predictions_pte.max():.1f} MIDI")
    print(f"ExecuTorch Average confidence: {confidence_predictions_pte.mean():.3f}")
    print(f"Torch Pitch range: {torch_pitch_predictions.min():.1f} - {torch_pitch_predictions.max():.1f} MIDI")
    print(f"Torch Average confidence: {torch_confidence_predictions.mean():.3f}")
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Combine pitch ranges for consistent y-axis
    all_pitch = np.concatenate([pitch_predictions_pte.flatten(), torch_pitch_predictions.flatten()])
    pitch_min, pitch_max = all_pitch.min() - 5, all_pitch.max() + 5
    
    # Plot pitch contour comparison
    axes[0].plot(range(len(pitch_predictions_pte)), pitch_predictions_pte, 'g-', linewidth=1.0, alpha=0.8, label='ExecuTorch Model')
    axes[0].plot(range(len(torch_pitch_predictions)), torch_pitch_predictions, 'r-', linewidth=1.0, alpha=0.8, label='Torch Model')
    axes[0].set_ylabel('Pitch (MIDI)')
    axes[0].set_title(f'Pitch Contour Comparison - {os.path.basename(audio_path)}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(pitch_min, pitch_max)
    axes[0].legend()
    
    # Plot confidence comparison
    axes[1].plot(range(len(confidence_predictions_pte)), confidence_predictions_pte, 'g-', linewidth=1.0, alpha=0.8, label='ExecuTorch Model')
    axes[1].plot(range(len(torch_confidence_predictions)), torch_confidence_predictions, 'r-', linewidth=1.0, alpha=0.8, label='Torch Model')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Pitch Confidence Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    
    # Set xlabel for the last visible plot
    axes[1].set_xlabel('Frame Index')
    
    plt.tight_layout()
    
    # Save or display plot
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()
    else:
        print("Displaying plot...")
        plt.show()
    
    return (pitch_predictions_pte, confidence_predictions_pte, time_stamps, 
            torch_pitch_predictions, torch_confidence_predictions, torch_time_stamps)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Plot pitch contour and confidence using ExecuTorch model inference"
    )
    
    parser.add_argument(
        "audio_file", 
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "-m", "--model", 
        default=None,
        help="Path to .pte model file (if not provided, will export from checkpoint)"
    )
    
    parser.add_argument(
        "-c", "--checkpoint", 
        default="mir-1k_g7",
        help="Checkpoint name to use if exporting ExecuTorch model (default: mir-1k_g7)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Output path for plot image (if not provided, will display plot)"
    )
    
    parser.add_argument(
        "-r", "--sampling_rate", 
        type=int, 
        default=48000,
        help="Target sampling rate for processing (default: 48000)"
    )
    
    parser.add_argument(
        "-s", "--chunk_size", 
        type=int, 
        default=960,
        help="Chunk size for processing (default: 960)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Validate .pte model file if provided
    if args.model and not os.path.exists(args.model):
        print(f"Warning: .pte model file not found: {args.model}")
        print("Will export from checkpoint instead")
        args.model = None
    
    try:
        # Generate plot
        results = plot_pitch_contour_comparison(
            audio_path=args.audio_file,
            pte_model_path=args.model,
            checkpoint_name=args.checkpoint,
            output_path=args.output,
            sampling_rate=args.sampling_rate,
            chunk_size=args.chunk_size
        )
        
        if results[0] is not None:
            print("Success!")
            return 0
        else:
            print("Failed to generate plot")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())