#!/usr/bin/env python3
"""
Test script to plot pitch contour and confidence using ONNX model for inference.
"""

import argparse
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from realtime.export_onnx import export_model, create_ort_session
from pesto.core import predict
from pesto.loader import load_model


def plot_pitch_contour_comparison(audio_path, onnx_model_path=None, checkpoint_name="mir-1k_g7", 
                                 output_path=None, sampling_rate=48000, chunk_size=960):
    """
    Plot pitch contour and confidence comparing ONNX and Torch model inference.
    
    Args:
        audio_path: Path to input audio file
        onnx_model_path: Path to ONNX model file (if None, exports from checkpoint)
        checkpoint_name: Checkpoint name to export if onnx_model_path is None
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
    
    # Setup ONNX model
    if onnx_model_path is None or not os.path.exists(onnx_model_path):
        print(f"Exporting ONNX model from checkpoint: {checkpoint_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_onnx_path = os.path.join(temp_dir, "temp_model.onnx")
            try:
                _, onnx_path = export_model(checkpoint_name, sampling_rate, chunk_size, temp_onnx_path)
                session = create_ort_session(onnx_path, 'CPUExecutionProvider')
            except Exception as e:
                print(f"Error exporting ONNX model: {e}")
                return None, None, None
    else:
        print(f"Loading ONNX model: {onnx_model_path}")
        try:
            session = create_ort_session(onnx_model_path, 'CPUExecutionProvider')
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
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
    
    # ONNX predictions
    pitch_predictions_onnx = []
    confidence_predictions_onnx = []
    volume_predictions_onnx = []
    time_stamps = []
    
    # Torch predictions - process full audio
    print("Running Torch model inference...")
    torch_model = load_model(checkpoint_name, step_size=10.0, sampling_rate=sampling_rate)
    torch_model.eval()
    
    with torch.no_grad():
        torch_timesteps, torch_pitch, torch_confidence, torch_activations = predict(
            waveform, sampling_rate, model_name=checkpoint_name, convert_to_freq=False
        )
    
    # Convert torch timesteps to seconds
    torch_time_stamps = torch_timesteps.numpy() / sampling_rate
    torch_pitch_predictions = torch_pitch.squeeze().numpy() if torch_pitch.dim() > 1 else torch_pitch.numpy()
    torch_confidence_predictions = torch_confidence.squeeze().numpy() if torch_confidence.dim() > 1 else torch_confidence.numpy()
    
    print(f"Torch model generated {len(torch_pitch_predictions)} predictions")
    print("Running ONNX model inference...")
    
    input_name = session.get_inputs()[0].name
    
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
        
        # Prepare input for ONNX model (add batch dimension)
        chunk_input = chunk.unsqueeze(0).numpy().astype(np.float32)
        
        try:
            # Run ONNX inference
            outputs = session.run(None, {input_name: chunk_input})
            
            # Extract outputs: [pred, conf, vol, act]
            pred = outputs[0][0]  # Remove batch dimension
            conf = outputs[1][0]
            vol = outputs[2][0] if len(outputs) > 2 else 0.0
            
            pitch_predictions_onnx.append(pred)
            confidence_predictions_onnx.append(conf)
            volume_predictions_onnx.append(vol)
            
            # Calculate time stamp for this chunk center
            time_stamp = (start_idx + chunk_size // 2) / sampling_rate
            time_stamps.append(time_stamp)
            
        except Exception as e:
            print(f"Error during inference at chunk {i}: {e}")
            break
    
    if not pitch_predictions_onnx:
        print("No ONNX predictions generated")
        return None, None, None
    
    # Convert ONNX predictions to numpy arrays
    pitch_predictions_onnx = np.array(pitch_predictions_onnx)
    confidence_predictions_onnx = np.array(confidence_predictions_onnx)
    volume_predictions_onnx = np.array(volume_predictions_onnx)
    time_stamps = np.array(time_stamps)
    
    print(f"ONNX model generated {len(pitch_predictions_onnx)} predictions")
    print(f"ONNX Pitch range: {pitch_predictions_onnx.min():.1f} - {pitch_predictions_onnx.max():.1f} MIDI")
    print(f"ONNX Average confidence: {confidence_predictions_onnx.mean():.3f}")
    print(f"Torch Pitch range: {torch_pitch_predictions.min():.1f} - {torch_pitch_predictions.max():.1f} MIDI")
    print(f"Torch Average confidence: {torch_confidence_predictions.mean():.3f}")
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Combine pitch ranges for consistent y-axis
    all_pitch = np.concatenate([pitch_predictions_onnx.flatten(), torch_pitch_predictions.flatten()])
    pitch_min, pitch_max = all_pitch.min() - 5, all_pitch.max() + 5
    
    # Plot pitch contour comparison
    axes[0].plot(range(len(pitch_predictions_onnx)), pitch_predictions_onnx, 'b-', linewidth=1.0, alpha=0.8, label='ONNX Model')
    axes[0].plot(range(len(torch_pitch_predictions)), torch_pitch_predictions, 'r-', linewidth=1.0, alpha=0.8, label='Torch Model')
    axes[0].set_ylabel('Pitch (MIDI)')
    axes[0].set_title(f'Pitch Contour Comparison - {os.path.basename(audio_path)}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(pitch_min, pitch_max)
    axes[0].legend()
    
    # Plot confidence comparison
    axes[1].plot(range(len(confidence_predictions_onnx)), confidence_predictions_onnx, 'b-', linewidth=1.0, alpha=0.8, label='ONNX Model')
    axes[1].plot(range(len(torch_confidence_predictions)), torch_confidence_predictions, 'r-', linewidth=1.0, alpha=0.8, label='Torch Model')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Pitch Confidence Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    
    # Plot volume (ONNX only) - hidden
    # axes[2].plot(time_stamps, volume_predictions_onnx, 'g-', linewidth=1.0, alpha=0.8)
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_ylabel('Volume')
    # axes[2].set_title('Volume (ONNX Model Only)')
    # axes[2].grid(True, alpha=0.3)
    
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
    
    return (pitch_predictions_onnx, confidence_predictions_onnx, time_stamps, 
            torch_pitch_predictions, torch_confidence_predictions, torch_time_stamps)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Plot pitch contour and confidence using ONNX model inference"
    )
    
    parser.add_argument(
        "audio_file", 
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "-m", "--model", 
        default=None,
        help="Path to ONNX model file (if not provided, will export from checkpoint)"
    )
    
    parser.add_argument(
        "-c", "--checkpoint", 
        default="mir-1k_g7",
        help="Checkpoint name to use if exporting ONNX model (default: mir-1k_g7)"
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
    
    # Validate ONNX model file if provided
    if args.model and not os.path.exists(args.model):
        print(f"Warning: ONNX model file not found: {args.model}")
        print("Will export from checkpoint instead")
        args.model = None
    
    try:
        # Generate plot
        results = plot_pitch_contour_comparison(
            audio_path=args.audio_file,
            onnx_model_path=args.model,
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