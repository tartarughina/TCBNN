import os
import subprocess

def get_mode(mode: int) -> str:
    if mode == 0:
        return "noUM"
    elif mode == 1:
        return "UM"
    elif mode == 2:
        return "UMT"
    else:
        return ""

def main():
    batch_sizes = [1024, 2048, 4096]
    iterations = 5
    output_dir = "log"
    gpus = 4

    os.makedirs(output_dir, exist_ok=True)
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/tartarughina/libjpeg/lib64"

    # 0: normal memory, 1: managed memory, 2: managed memory tuning
    for mode in range(3):
        for batch in batch_sizes:
            for i in range(1, iterations+1):
                command = f"mpirun -n {gpus} ./benn_nccl -b {batch}"

                if mode == 1:
                    command += " -m"
                elif mode == 2:
                    command += " -m -t"

                result_file = os.path.join(output_dir, f"{get_mode(mode)}_gpus_{gpus}_batch_{batch}_iter_{i}.txt")

                print(f"Running command: {command}")

                # Run the command and capture the output
                with open(result_file, "w") as output_file:
                    subprocess.run(command, shell=True, stdout=output_file, stderr=subprocess.STDOUT)

                print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()
