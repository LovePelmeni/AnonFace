import subprocess 
import torch 
import gc 
import torch 

def flush_cache():
    """
    Flushes both CPU and GPU
    cached data between measuring epochs
    """
    torch.cuda.empty_cache()
    gc.collect()

def set_gpu_clock_speed(gpu_speed: int, device: torch.device):
    """
    Fixates NVIDIA GPU clock speed
    to be equal to a specific value.
    NOTE:
        this method does not always work 
        as expected, because predominant portion of
        NVIDIA GPUs have blockers to prevent hardware 
        damage. Make sure to pick clock speed slightly
        below maximum to see the right results.

        Additionally, method requires nvidia-smi interface
        to be installed on the machine.

    Parameters:
    -----------
         - gpu_speed (int) - clock speed of the GPU to set
         - device - GPU device as a torch.Device interface
    """
    device_props = torch.cuda.get_device_properties(device)
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {device_props.name}",shell=True)
    process = subprocess.run(f"sudo nvidia-smi -lgc {gpu_speed} -i {device_props.name}",shell=True)
    print(stdout)

def reset_gpu_clock_speed(device: torch.device):
    """
    Resets clock speed of the GPU back to the default
    state.s
    NOTE:
        works only for NVIDIA GPU cards,
        because method uses the 'nvidia-smi' interface
        for applying configuration updates.
    
    Parameters:
    -----------
        - device (torch.device) - NVIDIA GPU via torch.Device interface
    """
    device_props = torch.cuda.get_device_properties(device=device)
    if len(device_props) == 0:
        raise RuntimeError("Invalid GPU device. Failed to parse configuration properties")

    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {device_props.name}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {device_props.name}", shell=True)