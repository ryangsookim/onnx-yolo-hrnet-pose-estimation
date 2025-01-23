import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import timeit
import os

# Configuration options
default_config = {
    "model_path": "../models",
    "model_name": "hrnet",
    "input_name": "input.1",
    "input_shape": (1, 3, 256, 192),
    "output_name": "output",
    "output_shape": (1, 17, 64, 48),
    "min_input_shape": (1, 3, 256, 192),
    "opt_input_shape": (5, 3, 256, 192),
    "max_input_shape": (5, 3, 256, 192),
    "num_iterations": 300,
    "mode": "compile",  # Default mode: "compile" or "infer"
    "use_fp16": True,  # Enable or disable FP16 precision
    "use_execute_v3": True
}

# Function to load configuration
def load_config():
    # User can modify this dictionary to adjust settings
    return default_config

config = load_config()

# Extract configuration variables
model_name = config["model_name"]
model_path = os.path.join(config["model_path"], model_name + ".onnx")
precision_suffix = "FP16" if config["use_fp16"] else "FP32"
engine_path = os.path.join("../models", model_name + f"_{precision_suffix}.engine")

input_name = config["input_name"]
output_name = config["output_name"]
input_shape = config["input_shape"]
output_shape = config["output_shape"]
min_input_shape = config["min_input_shape"]
opt_input_shape = config["opt_input_shape"]
max_input_shape = config["max_input_shape"]
num_iterations = config["num_iterations"]
mode = config["mode"]
use_fp16 = config["use_fp16"]
use_execute_v3 = config["use_execute_v3"]

# Compile mode
def compile_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Load ONNX model
    with open(model_path, "rb") as f:
        onnx_model = f.read()

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(model_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise RuntimeError("Failed to parse ONNX model")

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_input_shape, opt_input_shape, max_input_shape)
    # network.get_input(0).dtype = trt.float16  # Set input tensor to FP16
    # network.get_output(0).dtype = trt.float16  # Set output tensor to FP16

    config = builder.create_builder_config()
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 if configured
    config.add_optimization_profile(profile)

    # Build and serialize the engine
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("TensorRT engine has been successfully compiled and saved as", engine_path)

# Inference mode
def run_inference():
    """
    Run inference using TensorRT engine.

    Parameters:
    - use_execute_v3: If True, use execute_async_v3. Otherwise, use execute_v2.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Load TensorRT engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    context.set_input_shape(input_name, input_shape)

    # Allocate Pinned Memory for input and output
    host_input = cuda.pagelocked_empty(int(np.prod(input_shape)), dtype=np.float32).reshape(input_shape)
    host_output = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float32).reshape(output_shape)

    # Fill input data
    host_input[:] = np.random.random_sample(input_shape).astype(np.float32)

    # Allocate GPU memory for input and output
    d_input = cuda.mem_alloc(host_input.nbytes)
    d_output = cuda.mem_alloc(host_output.nbytes)

    # Bindings
    bindings = [int(d_input), int(d_output)]

    # Create a CUDA stream
    stream = cuda.Stream()

    # Set tensor addresses if using execute_async_v3
    if use_execute_v3:
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

    # Perform inference and measure latency
    latencies = []

    for _ in range(num_iterations):
        timings = {}  # Dictionary to store timings for each step

        tic = timeit.default_timer()

        # Asynchronous data transfer
        h2d_start = timeit.default_timer()
        cuda.memcpy_htod_async(d_input, host_input, stream)
        timings['memcpy_htod'] = (timeit.default_timer() - h2d_start) * 1000

        # Inference execution
        exec_start = timeit.default_timer()
        if use_execute_v3:
            context.execute_async_v3(stream_handle=stream.handle)
        else:
            context.execute_v2(bindings=bindings)
        timings['execute_inference'] = (timeit.default_timer() - exec_start) * 1000

        # Asynchronous data transfer back
        d2h_start = timeit.default_timer()
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
        timings['memcpy_dtoh'] = (timeit.default_timer() - d2h_start) * 1000

        toc = timeit.default_timer()
        timings['total'] = (toc - tic) * 1000  # Convert to milliseconds

        latencies.append(timings['total'])

        # Print detailed timings for this iteration
        for step, time_ms in timings.items():
            print(f"Step '{step}': {time_ms:.2f} ms")

    average_latency = sum(latencies) / len(latencies)
    print(f"Average inference latency over {num_iterations} iterations: {average_latency:.1f} ms")
    print("Inference result shape:", np.asarray(host_output).shape)
    print(host_output)


# Main logic
if __name__ == "__main__":
    if mode == "compile":
        compile_engine()
        run_inference()  # Automatically run inference after compilation
    elif mode == "infer":
        run_inference()
    else:
        raise ValueError("Invalid mode specified in configuration. Use 'compile' or 'infer'.")
