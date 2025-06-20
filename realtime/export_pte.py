import torch
import argparse
import datetime
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

from pesto import load_model


def export_model_to_executorch(checkpoint_name, sampling_rate, chunk_size, export_path, backend="xnnpack"):
    """Exports a PESTO model to ExecuTorch format."""
    step_size = 1000 * chunk_size / sampling_rate
    
    print(f"Chunk size: {chunk_size}")
    print(f"Step size: {step_size}ms")
    print(f"Backend: {backend}")
    
    # Load model in streaming mode for real-time inference
    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=4
    )
    model.eval()
    
    # Example input for export - fixed shape for ExecuTorch
    example_input = (torch.randn(1, chunk_size).clip(-1, 1),)
    
    # Export the model using torch.export with no_grad for inference
    with torch.no_grad():
        exported_program = torch.export.export(model, example_input)
    
    # Choose partitioner based on backend
    if backend == "xnnpack":
        partitioner = [XnnpackPartitioner()]
    elif backend == "mps":
        partitioner = [MPSPartitioner([])]
    else:
        partitioner = []
    
    # Configure edge compilation to handle complex number operations and dtype mismatches
    compile_config = EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten.view_as_complex.default,
            torch.ops.aten.view_as_real.default,
            torch.ops.aten.view_as_complex_copy.default,
            torch.ops.aten.permute_copy.default,
            torch.ops.aten.abs.default,
        ],
        _check_ir_validity=False  # Skip dtype validation for complex operations
    )
    
    # Convert to ExecuTorch format
    et_program = to_edge_transform_and_lower(
        exported_program, 
        partitioner=partitioner,
        compile_config=compile_config
    ).to_executorch()
    
    # Save the ExecuTorch program
    with open(export_path, "wb") as f:
        f.write(et_program.buffer)
    
    print(f"Model successfully exported to '{export_path}'")
    return model, export_path


def validate_model_outputs(original_model, export_path, chunk_size):
    """Validates ExecuTorch model outputs against original model."""
    from executorch.runtime import Runtime
    
    # Load ExecuTorch model
    runtime = Runtime()
    program = runtime.load_program(export_path)
    method = program.load_method("forward")
    
    # Create test input
    test_input = torch.randn(1, chunk_size).clip(-1, 1)
    
    # Run original model
    with torch.no_grad():
        original_output = original_model(test_input)
    
    # Run ExecuTorch model
    et_output = method.execute([test_input])
    
    # Compare outputs
    output_names = ["pred", "conf", "vol", "act"] if len(original_output) == 4 else ["pred", "conf", "vol"]
    
    for i, name in enumerate(output_names[:len(et_output)]):
        if torch.allclose(original_output[i], et_output[i], atol=1e-5):
            print(f"{name}: {original_output[i].shape} - Test passed: Outputs are close.")
        else:
            print(f"{name}: Test failed: Significant difference in outputs.")
            print(f"  Max diff: {torch.max(torch.abs(original_output[i] - et_output[i]))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PESTO model to ExecuTorch format.")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name for loading the model.")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate of the model.")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing.")
    parser.add_argument("-e", "--export_path", type=str, default=None, help="Optional custom export path.")
    parser.add_argument("-b", "--backend", type=str, default="xnnpack", 
                       choices=["xnnpack", "mps", "none"], help="Backend for acceleration.")
    parser.add_argument("--validate", action="store_true", help="Validate exported model outputs.")
    
    args = parser.parse_args()
    
    # Construct default export path if not provided
    if args.export_path is None:
        date_str = datetime.datetime.now().strftime("%y%m%d")
        backend_suffix = f"_{args.backend}" if args.backend != "none" else ""
        args.export_path = f"{date_str}_sr{args.sampling_rate//1000}k_h{args.chunk_size}{backend_suffix}.pte"
    
    # Export model
    model, export_path = export_model_to_executorch(
        args.checkpoint_name, 
        args.sampling_rate, 
        args.chunk_size, 
        args.export_path,
        args.backend
    )
    
    # Validate if requested
    if args.validate:
        print("\nValidating exported model...")
        validate_model_outputs(model, export_path, args.chunk_size)