import platform
import psutil
import GPUtil

def get_cpu_info():
    cpu_info = {}
    cpu_info['CPU Count'] = psutil.cpu_count(logical=True)  # Logical CPU count
    cpu_info['CPU Frequency'] = f"{psutil.cpu_freq().current:.2f} MHz"  # Current CPU frequency
    cpu_info['CPU Usage'] = f"{psutil.cpu_percent()}%"  # CPU usage
    return cpu_info

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'GPU ID': gpu.id,
            'GPU Name': gpu.name,
            'GPU Memory Total': f"{gpu.memoryTotal} MB",
            'GPU Memory Free': f"{gpu.memoryFree} MB",
            'GPU Memory Used': f"{gpu.memoryUsed} MB",
            'GPU Load': f"{gpu.load * 100}%",  # Convert to percentage
        })
    return gpu_info

def get_system_specifications():
    system_info = {}
    system_info['Operating System'] = platform.system()
    system_info['OS Version'] = platform.version()
    system_info['Architecture'] = platform.architecture()
    system_info['Processor'] = platform.processor()
    system_info['Node Name'] = platform.node()
    system_info['CPU Info'] = get_cpu_info()
    system_info['GPU Info'] = get_gpu_info()
    
    return system_info

def save_specifications_to_file(specifications, filename='spec.txt'):
    with open(filename, 'w') as f:
        for key, value in specifications.items():
            if isinstance(value, dict):  # If the value is a dictionary
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            elif isinstance(value, list):  # If the value is a list (for GPUs)
                f.write(f"{key}:\n")
                for idx, gpu in enumerate(value):
                    f.write(f"  GPU {idx}:\n")
                    for sub_key, sub_value in gpu.items():
                        f.write(f"    {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
if __name__ == "__main__":
    specifications = get_system_specifications()
    save_specifications_to_file(specifications)
    print("System specifications saved to spec.txt.")
