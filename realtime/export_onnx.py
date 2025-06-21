import argparse
import datetime

import torch
import onnxruntime as ort
import numpy as np
from pesto import load_model


def export_model(checkpoint_name, sampling_rate, chunk_size, onnx_name):
    """Exports a model to ONNX format and saves it to a file."""
    step_size = 1000 * chunk_size / sampling_rate

    print("Chunk size:", chunk_size)

    model = load_model(
        checkpoint_name,
        step_size=step_size,
        sampling_rate=sampling_rate,
        streaming=True,
        max_batch_size=4
    )
    model.eval()  # Set the model to evaluation mode

    # Example input for export
    example_input = torch.randn(3, chunk_size).clip(-1, 1)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        example_input,
        onnx_name,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred', 'conf', 'vol', 'act'],
    )
    print(f"Model successfully exported as '{onnx_name}'")

    return model, onnx_name


def validate_model(original_model, onnx_name, chunk_size):
    """Loads the exported ONNX model and validates its output."""
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_name)
    
    example_input = torch.randn(3, chunk_size).clip(-1, 1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model to ONNX format and validate its outputs.")
    parser.add_argument("checkpoint_name", type=str, help="Checkpoint name for loading the model.")
    parser.add_argument("-r", "--sampling_rate", type=int, default=48000, help="Sampling rate of the model.")
    parser.add_argument("-c", "--chunk_size", type=int, default=960, help="Chunk size for processing.")
    parser.add_argument("-o", "--onnx_name", type=str, default=None, help="Optional custom ONNX filename.")

    args = parser.parse_args()

    # Construct default ONNX name if not provided
    if args.onnx_name is None:
        date_str = datetime.datetime.now().strftime("%y%m%d")
        args.onnx_name = f"{date_str}_sr{args.sampling_rate//1000}k_h{args.chunk_size}.onnx"

    model, onnx_name = export_model(args.checkpoint_name, args.sampling_rate, args.chunk_size, args.onnx_name)
    validate_model(model, onnx_name, args.chunk_size)