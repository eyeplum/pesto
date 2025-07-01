import argparse
import datetime
import os
import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

from pesto import load_model


def export_model(checkpoint_name, sampling_rate, chunk_size, pte_name):
    """Exports a model using ExecuTorch with XNNPACK backend and saves it to a .pte file."""
    step_size = 1000 * chunk_size / sampling_rate

    print("Chunk size:", chunk_size)

    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=1
    )
    model.eval()  # Set the model to evaluation mode

    # Example input for export (batch_size=1 to match max_batch_size)
    # Use C4 sine wave (261.63 Hz) for deterministic testing
    t = torch.linspace(0, chunk_size / sampling_rate, chunk_size)
    c4_freq = 261.63
    example_input = (torch.sin(2 * torch.pi * c4_freq * t).unsqueeze(0),)

    # Export the model using torch.export
    print("Exporting model with torch.export...")
    exported_program = export(model, example_input)

    # Transform and lower with XNNPACK backend
    print("Applying XNNPACK partitioner...")
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    # Save the ExecuTorch program
    print(f"Saving to {pte_name}...")
    os.makedirs(os.path.dirname(pte_name), exist_ok=True)
    with open(pte_name, "wb") as file:
        file.write(executorch_program.buffer)
    
    print(f"Model successfully exported as '{pte_name}'")

    return model, pte_name


def validate_model(original_model, pte_name, chunk_size):
    """Loads the exported model and validates its output using ExecuTorch runtime."""
    try:
        from executorch.runtime import Runtime
        
        runtime = Runtime.get()
        program = runtime.load_program(pte_name)
        
        # Use same C4 sine wave for validation
        t = torch.linspace(0, chunk_size / 48000, chunk_size)  # Assume 48kHz for validation
        c4_freq = 261.63
        example_input = torch.sin(2 * torch.pi * c4_freq * t).unsqueeze(0)

        # Run the original model
        with torch.no_grad():
            original_output = original_model(example_input)

        # Run the ExecuTorch model
        method = program.load_method("forward")
        executorch_output = method.execute([example_input])

        # Compare outputs
        for name, x1, x2 in zip(["pred", "conf", "vol", "act"], original_output, executorch_output):
            print(f"  Original range: [{x1.min():.6f}, {x1.max():.6f}]")
            print(f"  ExecuTorch range: [{x2.min():.6f}, {x2.max():.6f}]")
            print(f"  Max diff: {torch.abs(x1 - x2).max():.6f}")
            if torch.allclose(x1, x2, rtol=1e-3, atol=1e-3):
                print(name, ":", x1.shape, "\n", "Test passed: Outputs are close.\n")
            else:
                print(f"{name} Test failed: Significant difference in outputs.")

    except ImportError:
        print("ExecuTorch runtime not available for validation.")
        print("Install executorch runtime to validate exported model.")
    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model using ExecuTorch with XNNPACK and validate its outputs.")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name for loading the model.")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate of the model.")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing.")
    parser.add_argument("-p", "--pte_name", type=str, default=None, help="Optional custom .pte filename.")

    args = parser.parse_args()

    # Construct default pte name if not provided
    if args.pte_name is None:
        date_str = datetime.datetime.now().strftime("%y%m%d")
        args.pte_name = f"pte-export/{date_str}_sr{args.sampling_rate//1000}k_h{args.chunk_size}_xnnpack.pte"

    model, pte_name = export_model(args.checkpoint_name, args.sampling_rate, args.chunk_size, args.pte_name)
    validate_model(model, pte_name, args.chunk_size)