import torch
from torch.nn import Module 
import time
import numpy
from src.inference import gpu_utils as utils

def measure_inference_time(
    network: Module, 
    batch_size: int, 
    input_shape: list,
    warmup_steps: int,
    total_iterations: int,
    inference_device: str = 'cpu',
    gpu_speed: int = None,
):
    """
    Measures inference time of the network
    using a batch of images of 'input_shape'
    on N total iterations.

    Parameters:
    ----------
        network - nn.Module neural network with 'forward' and 'backward' method initialized
        batch_size - number of images in a single batch of data
        input_shape - shape of the image in a batch
        inference_device - device to use during inference
    """
    if 'cuda' in inference_device:
        utils.set_gpu_speed(gpu_speed)

    device_data = torch.randn(
        size=[batch_size]+input_shape, 
        dtype=torch.float32
    ).to(inference_device)

    # running warmup steps

    for _ in range(warmup_steps):
        _ = network.forward(device_data)

    if 'cuda' in inference_device:
        start_events = [torch.cuda.Event() for _ in range(total_iterations)]
        end_events = [torch.cuda.Event() for _ in range(total_iterations)]

    times = []

    for idx in range(total_iterations):

        if 'cpu' in inference_device:
            start = time.time()
            predictions = network.to(inference_device).forward(device_data)
            end = time.time()
            elapsed_time = end - start
            del predictions

        if 'cuda' in inference_device:
            
            start_events[idx].record()
            predictions = network.to(inference_device).forward(device_data)
            end_events[idx].record()

            predictions._zero()

            elapsed_time = end_events[idx].elapsed_time(start_events[idx])
            torch.cuda.synchronize()

        utils.flush_cache()
    
    times.append(elapsed_time)

    if 'cuda' in inference_device:
        utils.reset_gpu_speed()

    return numpy.mean(times)
