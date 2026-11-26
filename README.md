# CIARA

Code and data of the paper "CIARA: Causal Intervention-Aware Resource Allocation for Cloud-Edge LLM Inference in IoT".

CIARA is a controller for cloud-edge collaborative inference, where small language models at the edge serve most IoT requests and a cloud model handles the remainder. CIARA estimates the deployed effect of each edge configuration from safe randomized interventions on the live system, bounds every effect with a confidence radius that remains valid at every look at the data and under coupling through the shared cloud, and returns the policy to deploy from a pessimistic linear program. The deployed policy satisfies every latency, deadline, energy and cloud budget with a specified confidence.

## What the repository reproduces

| Level | What it shows | Needs | Time |
| --- | --- | --- | --- |
| 1 | The certificate of the paper from the published campaign data, and every figure and table of the paper and the supplement | Python with numpy and matplotlib | under two minutes |
| 2 | The numerical checks of Lemma 4 and of Propositions 2 and 5(a), and the simulations of the supplement | Python with numpy | minutes for the quick runs |
| 3 | The serving system of the evaluation: edge gateway, cloud gateway, campaign controller, IoT client emulator, outcome pipeline | GPUs, vLLM and the models of the model table | hours to days |

### Level 1: the certificate, the figures and the tables

```bash
python3 scripts/certify_campaign.py --check
```

The script reads the node-blocks of the certification campaign (`data/campaign_node_blocks.csv`), evaluates Algorithm 2 at the two looks of the campaign, and checks the published decisions: at day 4 (230 rounds) no policy is certifiable and the incumbent is retained, and at day 7 (403 rounds) the certified policy with weights (0.106, 0.548, 0.346) is deployed with a bound of 0.084 on the quality cost.

```bash
bash scripts/make_all.sh
```

The script rebuilds every data-driven figure and table of the paper and the supplement from `data/` into `out/`. Each generator `scripts/make_*.py` reads only files of `data/` and writes one figure or table.

### Level 2: the simulation

```bash
cd simulation
python3 check_lemma4.py
```

The command checks the bound of Lemma 4 exactly. The other statements are reproduced by the following quick runs, each of a few minutes:

```bash
python3 run_experiments.py --R 20 --out /tmp/est estimators && python3 summarize_bias.py /tmp/est/estimators.csv
python3 run_balanced.py --mode campaign --R 250 --chunk 250 --procs 4 --out /tmp/bal && python3 analyze_balanced.py /tmp/bal /tmp/bal_tables
python3 run_coupled.py --scen K3X1 null --t 5 --N 5 --R 250 --chunk 250 --procs 4 --out /tmp/cpl && python3 analyze_coupled.py --dir /tmp/cpl
python3 check_radius.py && python3 check_coupled.py && python3 check_targets.py && python3 cor6_values.py
```

The first line gives the bias of per-epoch randomization against the block design (Proposition 2). The second line gives the balanced allocation against independent draws, and the third line the coverage of the radii under coupling (Proposition 5(a)). The last line runs the exact checks of the radius and the values of the cost of certification. `simulation/results/` holds the tables of the full runs.

### Level 3: the serving system

A device sends a request to the edge gateway of its node (`ciara_edge.gateway`), which admits it, serves it on the edge engine or escalates it to the cloud gateway (`ciara_control.cloud_gateway`). The engine manager (`ciara_edge.manager`) keeps the models of a node resident and switches the configuration. The controller (`ciara_control.controller`) assigns a configuration to every node per block from a run configuration in `configs/controller/`, the client emulator (`ciara_clients`) generates the devices, links and arrivals from `configs/clients/`, and the outcome pipeline (`ciara_analysis.pipeline`) turns the request logs into the node-block outcomes. Each module is started with `python3 -m <module> --help` after

```bash
export PYTHONPATH=src/edge:src/controller:src/clients:src/analysis:src/monitor:baselines:algorithms
export CIARA_PILOT_ROOT=workload
```

The unit tests run without a GPU.

## Layout

| Folder | Content | Paper |
| --- | --- | --- |
| `algorithms/` | The package `ciara`: campaign design, anytime-valid radius, carryover band, pessimistic linear program with the share constraint, the gate and the cost bound, screening, and the measurements of Stage 1 | Section V, Algorithms 1 and 2, Lemma 4, Proposition 5 |
| `baselines/` | The five published controllers as reimplemented on the same configurations: the profile-driven selector, RACE, Mix-or-Split, RecServe and a constrained bandit | Section VI, Baselines and Metrics |
| `src/` | Edge gateway with admission and escalation, engine manager, cloud gateway, campaign controller, IoT client emulator, outcome pipeline, checks on the campaign data | Sections III and VI |
| `workload/` | Suite loaders and scorers of the two request families, the model table, the quantization recipe | Section VI, supplementary Section S-IV |
| `configs/` | Example configurations of the controller and the clients, link profiles, energy coefficients | Section VI, supplementary Sections S-III and S-IV |
| `simulation/` | Synthetic system and the numerical checks of the lemma and the propositions | Section V, supplementary Section S-II |
| `data/` | The data behind every figure and table, with a column dictionary and checksums | Section VI, supplementary Sections S-IV to S-VI |
| `datasets/` | The workload: the sensor-interpretation suite, the selection and build scripts of the tool-call suite, the fitted arrival parameters | Section VI, Testbed and Workload |
| `scripts/` | The certificate script and the generators of the figures and tables | Section VI |
| `tests/` | Unit tests of the method, the baselines and the serving system | |

## Tests

```bash
export PYTHONPATH=algorithms:baselines:src/edge:src/edge/tools:src/controller:src/clients:src/analysis:src/monitor
for d in tests/algorithms tests/baselines tests/src/edge tests/src/controller tests/src/clients tests/src/analysis tests/src/monitor; do python3 -B -m unittest discover -s $d; done
```

The tests of `tests/algorithms` and `tests/baselines` need numpy only. The tests of `tests/src` need the packages of `src/requirements-src.txt`. Tests that read the tool-call suite are skipped until the suite is built from its public source with `datasets/iot_toolcall/build/freeze_iot.py` (source and revision in `datasets/manifest.json`).

## Requirements

Python 3.11 or newer. `requirements.txt` covers Levels 1 and 2 and the tests of the method and the baselines. `src/requirements-src.txt` covers the serving system.

## Data

`data/` holds tidy CSV files: one row per node-block, per request, or per arm and outcome. `data/columns.md` defines every column of every file, and `data/SHA256SUMS` holds the checksums. The raw request and engine logs of the runs (approximately 92 GB) are not part of the repository; the reduced request logs in `data/` carry the fields that the figures use. Model weights are not part of the repository: `workload/models/model_table.py` names every checkpoint, and `workload/quantization/build.py` is the recipe of the INT8 and INT4 builds.

## Terms

| Code | Paper |
| --- | --- |
| `I1-E` | The incumbent: the profile-driven selector that repeats its selection in every epoch |
| `I1-S`, `I1-oracle` | The selector that selects once; the selector with a profile measured under co-location |
| `I2`, `I3` | RACE; Mix-or-Split |
| `f1prio`, `fifo` | Priority admission (the priority arm); FIFO admission (the FIFO arm) |
| `c32`, `kappaC`, `cascade` | Token-budget escalation with a budget of 32 tokens |
| `Y_Q`, `Y_L_f1`, `Y_L_f2`, `Y_D`, `Y_E_A`, `Y_E_B`, `Y_E_C`, `Y_C` | Quality, tail exceedance of families 1 and 2, deadline misses, device energy of classes A, B and C, cloud share |
| `e4`, `e1e5`, `e2`, `e8v9` | Held-out deployment; comparison run; reference levels of the validity analysis; capacity contexts |
| `eta_dep`, `zeta`, `t_bar`, `tau` | Deployment slack; coupling slack; mixing bound; switch transient |

## License

The code is released under the MIT License (`LICENSE`). The data in `data/` and the sensor-interpretation suite in `datasets/sensor_interp/` are released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`). The public sets that the workload builds on keep their own licenses, which `datasets/manifest.json` records.
