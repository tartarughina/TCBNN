import os
import subprocess

def get_oversub(oversub: int) -> str:
    if oversub == 1:
        return "OVER1"
    elif oversub == 2:
        return "OVER2"
    else:
        return ""

def main():
    batch_sizes = [1024, 2048, 4096]
    iterations = 5
    output_dir = "benn_over_log"
    gpus = 4

    os.makedirs(output_dir, exist_ok=True)
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/tartarughina/libjpeg/lib64"

    # 0: normal memory, 1: managed memory, 2: managed memory tuning
    for oversub in range(1,3):
        for batch in batch_sizes:
            for i in range(1, iterations+1):
                command = f"mpirun -n {gpus} ./benn_nccl -b {batch} -u --oversub {oversub}"

                result_file = os.path.join(output_dir, f"{get_oversub(oversub)}_gpus_{gpus}_batch_{batch}_iter_{i}.txt")

                print(f"Running command: {command}")

                # Run the command and capture the output
                with open(result_file, "w") as output_file:
                    subprocess.run(command, shell=True, stdout=output_file, stderr=subprocess.STDOUT)

                print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()
