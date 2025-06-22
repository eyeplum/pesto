import argparse
import os
import time
import platform

import torch
import onnxruntime as ort
import numpy as np
from pesto import load_model


def export_model(checkpoint_name, sampling_rate, chunk_size, onnx_name):
    """Exports a model to ONNX format and saves it to a file."""
    step_size = 1000 * chunk_size / sampling_rate
    batch_size = 1

    print("Chunk size:", chunk_size)

    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=4
    )
    model.eval()  # Set the model to evaluation mode

    # Create output directory
    output_dir = "onnx-export"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add directory to filename if not already included
    if not os.path.dirname(onnx_name):
        onnx_path = os.path.join(output_dir, onnx_name)
    else:
        onnx_path = onnx_name

    # Example input for export
    example_input = torch.randn(batch_size, chunk_size).clip(-1, 1)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred', 'conf', 'vol', 'act'],
    )
    print(f"Model successfully exported as '{onnx_path}'")

    return model, onnx_path


def create_ort_session(onnx_name, execution_provider='CPUExecutionProvider'):
    """Creates an ONNX Runtime session with the specified execution provider."""
    providers = []
    
    if execution_provider == 'CoreMLExecutionProvider':
        # Check if we're on macOS
        if platform.system() != 'Darwin':
            print("Warning: CoreML execution provider only available on macOS. Falling back to CPU.")
            execution_provider = 'CPUExecutionProvider'
        else:
            # CoreML provider options
            provider_options = {
                'MLComputeUnits': 'CPUOnly',  # Use all available compute units
                'ModelFormat': 'MLProgram'  # Use MLProgram format for better performance
            }
            providers.append(('CoreMLExecutionProvider', provider_options))
    
    # Always add CPU as fallback
    providers.append('CPUExecutionProvider')
    
    try:
        session = ort.InferenceSession(onnx_name, providers=providers)
        print(f"ONNX Runtime session created with providers: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"Failed to create session with {execution_provider}, falling back to CPU: {e}")
        return ort.InferenceSession(onnx_name, providers=['CPUExecutionProvider'])


def validate_model(original_model, onnx_name, chunk_size, execution_provider='CPUExecutionProvider'):
    """Loads the exported ONNX model and validates its output."""
    batch_size = 1
    
    # Load ONNX model
    ort_session = create_ort_session(onnx_name, execution_provider)
    
    example_input = torch.randn(batch_size, chunk_size).clip(-1, 1)

    # Run the original PyTorch model
    with torch.no_grad():
        original_output = original_model(example_input)

    # Run the ONNX model
    onnx_input = {ort_session.get_inputs()[0].name: example_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)

    # Compare outputs
    output_names = ["pred", "conf", "vol", "act"]
    for i, name in enumerate(output_names):
        torch_out = original_output[i].numpy()
        onnx_out = onnx_output[i]
        
        if np.allclose(torch_out, onnx_out, atol=1e-4):
            print(name, ":", torch_out.shape, "\n", "Test passed: Outputs are close.\n")
        else:
            print(name, "Test failed: Significant difference in outputs.")
            print(f"Max difference: {np.max(np.abs(torch_out - onnx_out))}")


def validate_performance(original_model, onnx_name, chunk_size, sampling_rate=48000, num_iterations=1000, execution_provider='CPUExecutionProvider'):
    """Validates and compares the performance between PyTorch and ONNX models."""
    batch_size = 1
    
    # Load ONNX model
    ort_session = create_ort_session(onnx_name, execution_provider)
    
    example_input = torch.randn(batch_size, chunk_size).clip(-1, 1)
    
    # Benchmark PyTorch model
    torch_times = []
    with torch.no_grad():
        # Warm-up runs
        for _ in range(10):
            _ = original_model(example_input)
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            iter_start = time.time()
            _ = original_model(example_input)
            torch_times.append(time.time() - iter_start)
        total_pytorch_time = time.time() - start_time
    
    # Benchmark ONNX model
    onnx_input = {ort_session.get_inputs()[0].name: example_input.numpy()}
    onnx_times = []
    
    # Warm-up runs
    for _ in range(10):
        _ = ort_session.run(None, onnx_input)
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        iter_start = time.time()
        _ = ort_session.run(None, onnx_input)
        onnx_times.append(time.time() - iter_start)
    total_onnx_time = time.time() - start_time
    
    # Calculate statistics
    torch_times = np.array(torch_times) * 1000  # Convert to ms
    onnx_times = np.array(onnx_times) * 1000   # Convert to ms
    
    print("\n=== Performance Validation Results ===")
    print(f"Number of iterations: {num_iterations}")
    print(f"Chunk size: {chunk_size}")
    
    print(f"\nPyTorch Model:")
    print(f"  Average time: {torch_times.mean():.3f} ± {torch_times.std():.3f} ms")
    print(f"  Min time: {torch_times.min():.3f} ms")
    print(f"  Max time: {torch_times.max():.3f} ms")
    print(f"  Total time: {total_pytorch_time*1000:.1f} ms")
    
    print(f"\nONNX Model ({execution_provider}):")
    print(f"  Average time: {onnx_times.mean():.3f} ± {onnx_times.std():.3f} ms")
    print(f"  Min time: {onnx_times.min():.3f} ms")
    print(f"  Max time: {onnx_times.max():.3f} ms")
    print(f"  Total time: {total_onnx_time*1000:.1f} ms")
    
    # Performance comparison
    speedup = torch_times.mean() / onnx_times.mean()
    print(f"\nSpeedup: {speedup:.2f}x ({'ONNX faster' if speedup > 1 else 'PyTorch faster'})")
    
    # Real-time processing analysis
    chunk_duration_ms = 1000 * chunk_size / sampling_rate
    pytorch_realtime_factor = torch_times.mean() / chunk_duration_ms
    onnx_realtime_factor = onnx_times.mean() / chunk_duration_ms
    
    print(f"\nReal-time processing capability:")
    print(f"  Chunk duration: {chunk_duration_ms:.1f} ms")
    print(f"  PyTorch real-time factor: {pytorch_realtime_factor:.3f} ({'✓' if pytorch_realtime_factor < 1 else '✗'})")
    print(f"  ONNX real-time factor: {onnx_realtime_factor:.3f} ({'✓' if onnx_realtime_factor < 1 else '✗'})")
    
    return {
        'pytorch_avg_ms': torch_times.mean(),
        'onnx_avg_ms': onnx_times.mean(),
        'speedup': speedup,
        'pytorch_realtime_capable': pytorch_realtime_factor < 1,
        'onnx_realtime_capable': onnx_realtime_factor < 1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model to ONNX format and validate its outputs.")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name for loading the model.")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate of the model.")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing.")
    parser.add_argument("-o", "--onnx_name", type=str, default=None, help="Optional custom ONNX filename.")
    parser.add_argument("-e", "--execution_provider", type=str,
                        choices=['CPUExecutionProvider', 'CoreMLExecutionProvider'],
                        default='CPUExecutionProvider',
                        help="ONNX Runtime execution provider to use.")

    args = parser.parse_args()

    # Construct default ONNX name if not provided
    if args.onnx_name is None:
        args.onnx_name = f"sr{args.sampling_rate//1000}k_h{args.chunk_size}.onnx"

    model, onnx_name = export_model(args.checkpoint_name, args.sampling_rate, args.chunk_size, args.onnx_name)
    validate_model(model, onnx_name, args.chunk_size, args.execution_provider)
    validate_performance(model, onnx_name, args.chunk_size, args.sampling_rate, execution_provider=args.execution_provider)