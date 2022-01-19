import sys, os
import numpy as np

if __name__ == "__main__":

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    for fil in os.listdir(input_dir):
        if fil.startswith("Run") and fil.endswith(".dat"):
            # print(
            #     f"python3 ./plot_run_pipeline.py {os.path.join(input_dir, fil)} 4 3 {output_dir} True"
            # )
            os.system(
                f"python ./plot_run_pipeline.py {os.path.join(input_dir, fil)} 4 3 '{output_dir}' True"
            )
    print("done.")
