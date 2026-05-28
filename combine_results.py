import csv

# GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,sort_period,repeat,agent_density,mean_message_count,s_rtc,s_simulation,s_init,s_exit,s_step_mean

FLAME_DIR = "./build/FLAME-GPU-2-results"
CUPY_DIR  = "/home/erkoning/ABM-GPU-examples/models/circles/cupy"
OUT_DIR = "./build/combined-FLAME-cupy-results"

#LABELS = ["fixed-density", "variable-density", "comm-radius", "sort-period", "dimensions", "high-density", "model-type", "grid-stride", "block-size"]
#LABELS = ["sort-period"]
LABELS = ["fixed-density"]
# LABELS = []

FLAME_SIM_EXT = "_perSimulationCSV.csv"
CUPY_SIM_EXT = "_perSimulation_CSV.csv"
OUT_EXT = FLAME_SIM_EXT

for LABEL in LABELS:
    try:
        with open(f"{FLAME_DIR}/{LABEL}{FLAME_SIM_EXT}") as flame_input:
            flame_reader = csv.reader(flame_input, delimiter=',', quotechar='|')

            with open(f"{CUPY_DIR}/{LABEL}{CUPY_SIM_EXT}") as cupy_input:
                cupy_reader = csv.DictReader(cupy_input)

                with open(f"{OUT_DIR}/{LABEL}{OUT_EXT}", 'w') as fout:
                    for row in flame_reader:
                        print(', '.join(row), file=fout)
                    for row in cupy_reader:
                        print(f"{row['GPU']}, 1, 0, cupy-{row['model']}, {row['steps']}, {row['agent_count']}, {row['env_width']}, {row['comm_radius']}, {row['sort_period']}, {row['repeat']}, {row['agent_density']}, 0, 0, {row['s_simulation']}, {row['s_init']}, {row['s_exit']}, {row['s_step_mean']}", file=fout)

    except FileNotFoundError:
        with open(f"{CUPY_DIR}/{LABEL}{CUPY_SIM_EXT}") as cupy_input:
            cupy_reader = csv.DictReader(cupy_input)
            with open(f"{OUT_DIR}/{LABEL}{OUT_EXT}", 'w') as fout:
                if LABEL == "grid-stride" or LABEL == "block-size":
                    print("GPU, release_mode, seatbelts_on, model, steps, agent_count, env_width, comm_radius, sort_period, block_size, max_threads, repeat, agent_density, mean_message_count, s_rtc, s_simulation, s_init, s_exit, s_step_mean", file=fout)
                    for row in cupy_reader:
                        print(f"{row['GPU']}, 1, 0, cupy-{row['model']}, {row['steps']}, {row['agent_count']}, {row['env_width']}, {row['comm_radius']}, {row['sort_period']}, {row['block_size']}, {row['max_threads']}, {row['repeat']}, {row['agent_density']}, 0, 0, {row['s_simulation']}, {row['s_init']}, {row['s_exit']}, {row['s_step_mean']}", file=fout)
                else:
                    print("GPU, release_mode, seatbelts_on, model, steps, agent_count, env_width, comm_radius, sort_period, repeat, agent_density, mean_message_count, s_rtc, s_simulation, s_init, s_exit, s_step_mean", file=fout)
                if LABEL == "dimensions":
                    for row in cupy_reader:
                        print(f"{row['GPU']}, 1, 0, cupy-{row['model']} {row['dimensions']}D, {row['steps']}, {row['agent_count']}, {row['env_width']}, {row['comm_radius']}, {row['sort_period']}, {row['repeat']}, {row['agent_density']}, 0, 0, {row['s_simulation']}, {row['s_init']}, {row['s_exit']}, {row['s_step_mean']}", file=fout)
                elif LABEL == "data-type":
                    for row in cupy_reader:
                        if row['dtype'] == "<class 'numpy.float32'>":
                            row['dtype'] = "float32"
                        elif row['dtype'] == "<class 'numpy.float64'>":
                            row['dtype'] = "float64"
                        print(f"{row['GPU']}, 1, 0, cupy-{row['model']} {row['dtype']}, {row['steps']}, {row['agent_count']}, {row['env_width']}, {row['comm_radius']}, {row['sort_period']}, {row['repeat']}, {row['agent_density']}, 0, 0, {row['s_simulation']}, {row['s_init']}, {row['s_exit']}, {row['s_step_mean']}", file=fout)
                else:
                    for row in cupy_reader:
                        print(f"{row['GPU']}, 1, 0, cupy-{row['model']}, {row['steps']}, {row['agent_count']}, {row['env_width']}, {row['comm_radius']}, {row['sort_period']}, {row['repeat']}, {row['agent_density']}, 0, 0, {row['s_simulation']}, {row['s_init']}, {row['s_exit']}, {row['s_step_mean']}", file=fout)

DIFF_LABELS = ["fixed-density"]

for LABEL in DIFF_LABELS:
    try:
        with open(f"{FLAME_DIR}/{LABEL}{FLAME_SIM_EXT}") as flame_input:
            flame_reader = csv.DictReader(flame_input)

            with open(f"{CUPY_DIR}/{LABEL}{CUPY_SIM_EXT}") as cupy_input:
                cupy_reader = csv.DictReader(cupy_input)

                with open(f"{OUT_DIR}/{LABEL}_diff_{OUT_EXT}", 'w') as fout:
                    cupy_results = dict()
                    flame_results = dict()
                    GPU = None
                    flame_row_count = 0
                    for row in flame_reader:
                        flame_row_count += 1
                        if row['model'] == 'circles_spatial3D':
                            if GPU is None:
                                GPU = row['GPU']
                            elif GPU != row['GPU']:
                                print(f"WARNING: Expected GPU {GPU} and found GPU {row['GPU']}")
                            key = f"{row['steps']}, {row['agent_count']}, {float(row['env_width'])}, {float(row['comm_radius'])}, {int(row['sort_period'])}, 0, 0, 0, {float(row['agent_density'])}"
                            if key in flame_results:
                                flame_results[key].append(float(row['s_step_mean']))
                            else:
                                flame_results[key] = [float(row['s_step_mean'])]
                    for row in cupy_reader:
                        if GPU is None:
                            GPU = row['GPU']
                        elif GPU != row['GPU']:
                            print(f"WARNING: Expected GPU {GPU} and found GPU {row['GPU']}")
                        if row['model'] == 'grid':
                            key = f"{row['steps']}, {row['agent_count']}, {float(row['env_width'])}, {float(row['comm_radius'])}, {int(row['sort_period'])}, 0, 0, 0, {float(row['agent_density'])}"
                            if key in cupy_results:
                                cupy_results[key].append(float(row['s_step_mean']))
                            else:
                                cupy_results[key] = [float(row['s_step_mean'])]
                    print("GPU, release_mode, seatbelts_on, model, steps, agent_count, env_width, comm_radius, sort_period, block_size, max_threads, repeat, agent_density, mean_message_count, s_rtc, s_simulation, s_init, s_exit, s_step_mean", file=fout)
                    for key in cupy_results.keys():
                        flame_avg = sum(flame_results[key])/len(flame_results[key])
                        cupy_avg = sum(cupy_results[key])/len(cupy_results[key])
                        print(f"{GPU}, 1, 0, speedup, {key}, 0, 0, 0, 0, 0, {cupy_avg/flame_avg}", file=fout)
                        print(f"{GPU}, 1, 0, diff, {key}, 0, 0, 0, 0, 0, {cupy_avg-flame_avg}", file=fout)

    except FileNotFoundError:
        print(f"Unable to calculate differences for {LABEL}")
