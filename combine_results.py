import csv
import re

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

DEFAULT_HEADER = ["GPU", "release_mode", "seatbelts_on", "model", "steps", "agent_count", "env_width", "comm_radius", "sort_period", "repeat", "agent_density", "mean_message_count", "s_rtc", "s_simulation", "s_init", "s_exit", "s_step_mean"]
# Experiments with extra columns, inserted after sort_period when there are no FLAME results to take the header from
EXTRA_COLUMNS = {
    "grid-stride": ["block_size", "max_threads"],
    "block-size": ["block_size", "max_threads"],
}

def clean_dtype(dtype):
    # e.g. "<class 'numpy.float32'>-<class 'numpy.int64'>" -> "float32-int64"
    return re.sub(r"<class 'numpy\.(\w+)'>", r"\1", dtype)

def cupy_model_name(label, row):
    name = f"cupy-{row['model']}"
    if label == "dimensions":
        name += f" {row['dimensions']}D"
    elif label == "data-type":
        name += f" {clean_dtype(row['dtype'])}"
    return name

def cupy_output_row(label, row, header):
    # Values FLAME records that cupy doesn't; any other column missing from the cupy CSV is also filled with 0
    values = dict(row, release_mode=1, seatbelts_on=0, mean_message_count=0, s_rtc=0)
    values['model'] = cupy_model_name(label, row)
    return [str(values.get(column, 0)) for column in header]

for LABEL in LABELS:
    try:
        cupy_input = open(f"{CUPY_DIR}/{LABEL}{CUPY_SIM_EXT}")
    except FileNotFoundError:
        print(f"No cupy results for {LABEL}, skipping")
        continue

    with cupy_input, open(f"{OUT_DIR}/{LABEL}{OUT_EXT}", 'w') as fout:
        try:
            with open(f"{FLAME_DIR}/{LABEL}{FLAME_SIM_EXT}") as flame_input:
                flame_reader = csv.reader(flame_input, delimiter=',', quotechar='|')
                header = [column.strip() for column in next(flame_reader)]
                print(', '.join(header), file=fout)
                for row in flame_reader:
                    print(', '.join(row), file=fout)
        except FileNotFoundError:
            extra = EXTRA_COLUMNS.get(LABEL, [])
            split = DEFAULT_HEADER.index("sort_period") + 1
            header = DEFAULT_HEADER[:split] + extra + DEFAULT_HEADER[split:]
            print(', '.join(header), file=fout)

        for row in csv.DictReader(cupy_input):
            print(', '.join(cupy_output_row(LABEL, row, header)), file=fout)

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
