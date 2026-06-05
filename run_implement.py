from multiprocessing import Pool
import subprocess


configs = []
torch_seed = [1337, 2337, 3337]
window_size = [8]
low_threshold = [0.96, 0.97, 0.98]
date_class = [0, 1]
devices = ["cuda:0", "cuda:1"]


if __name__ == "__main__":
    ########## Plan 1 (bug has been fixed) ##########
    # However, too slow.
    # def run(config):
    #     subprocess.run(["python",
    #     "Ablation_angle_withoutKbest.py",
    #     *config])
    # for seed in torch_seed:
    #     for window in window_size:
    #         for low in low_threshold:
    #             for date in date_class:
    #                 for device in devices:
    #                     config = ["--device", device,
    #                             "--torch_seed", str(seed),
    #                             "--window_size", str(window),
    #                             "--low_threshold", str(low),
    #                             "--date_class", str(date)]
    #                     configs.append(config)
    # with Pool(processes=4) as pool:
    #     pool.map(run, configs)
    
    ########## Plan 2 ##########
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--torch_seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")

    args, _ = parser.parse_known_args()
    for seed in torch_seed:
        for window in window_size:
            for low in low_threshold:
                for date in date_class:
                    config = ["--device",args.device,
                              "--torch_seed", str(seed),
                              "--window_size", str(window),
                              "--low_threshold", str(low),
                              "--date_class", str(date)]
                    subprocess.run(["python", "Implement_code.py", *config])
