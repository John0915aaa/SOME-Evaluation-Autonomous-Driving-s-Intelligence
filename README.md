# AV Intelligence Evaluation Lightweight Reproduction

This repository provides a lightweight reproduction package for an autonomous vehicle intelligence evaluation workflow. It includes example interaction indexes, a prepared training table, and a lightweight Waymo/trajdata cache, so users can run the main workflow locally without downloading the full Waymo or InterHub dataset.

This repository uses **705 Waymo interaction samples** to enable fast reproduction of the complete evaluation workflow. For access to the full dataset, please visit https://github.com/zxc-tju/InterHub to download the complete Waymo dataset, and replace the lightweight index files with `data/waymo_data index_all.csv` and `data/dlmm_interaction_index_all.csv`.

The project contains three modules:

1. `1-objective_metrics`: extracts objective metrics from indexed Waymo interaction segments.
2. `2-subjective_evaluation`: generates DLMM prompts and optionally performs LLM-based subjective evaluation.
3. `3-evaluation_model`: trains the final evaluation model using objective metrics and subjective scores.

## Repository Structure

```text
Code and Data/
  data/
    waymo_data index.csv
    ...
    train_data.csv
    waymo_lightweight_cache/
      waymo_0-299/
      ...
  1-objective_metrics/
    safety_metrics.py
    ...
    src/
    output/
  2-subjective_evaluation/
    dlmm_model.py
    ...
    src/
    prompt/
    output/
  3-evaluation_model/
    train_evaluation_model.py
    src/
    output/
  requirement.txt
  README.md
```

## Included Data

`data/waymo_data index.csv` is the interaction index used by the objective metric module. Each row corresponds to one interaction segment to be analyzed. Important columns include:

- `index`: interaction segment ID.
- `dataset`: dataset name, such as `waymo_train`.
- `folder`: cache group, such as `waymo_0-299`.
- `scenario_idx`: Waymo scenario ID.
- `track_id`: IDs of the interacting agents or ego vehicle.
- `start` / `end`: start and end timesteps of the interaction segment.
- `ego_type`: role type of the autonomous vehicle in this segment.
- `key_agents`: key interacting agents.
- `path_relationship`: interaction type, such as CP or MP.

`data/dlmm_interaction_index.csv` is the interaction index and feature table used by the subjective evaluation module. It refers to the same lightweight Waymo scenes, but also includes fields required for prompt generation, such as prompt case, turning labels, priority labels, lane-change information, and interaction descriptions.

`data/train_data.csv` is the prepared table used by the evaluation model training module. It already contains the model-ready features and target score, including columns such as `TTC`, `PET`, `a_p`, `a_l`, `jerk`, `yaw_rate`, `task_time`, `avg_delay`, `IO`, `impact`, and `score`.

For convenience, `train_data.csv` already provides the normalized objective-metric results for all **705 interaction samples**, together with the final subjective score (`score`), which is calculated as the average of **three LLM evaluation results**. Users who intend to train the model using other interaction data may normalize the extracted objective metrics according to the method described in the paper before constructing a new training table.

`data/waymo_lightweight_cache` is the lightweight Waymo/trajdata cache extracted from the full cache. The code uses the `folder` and `scenario_idx` fields in the CSV files to locate the required scene directories, map caches, and `scenes_list.dill` files. The cache path is resolved dynamically relative to the repository root, so the whole `Code and Data` folder can be moved to another location as long as its internal structure is preserved.

## Environment Setup

This lightweight reproduction package is intended to run on Ubuntu only. Ubuntu 20.04 is recommended.

Use conda to create an isolated Python environment, then install the dependencies from `requirement.txt`.

A typical setup is:

```bash
cd "Code and Data"
conda create -n av_intelligence_eval python=3.8 -y
conda activate av_intelligence_eval
pip install -r requirement.txt
```

## Data Path Mechanism

For normal lightweight reproduction, no environment variable is required.

If you need to override the cache location manually, the following environment variables are supported:

- `WAYMO_CACHE_ROOT`: a shared cache root. The code appends `<folder>` automatically.
- `WAYMO_CACHE_0_299`: path for `waymo_0-299`.
- `WAYMO_CACHE_300_499`: path for `waymo_300-499`.
- `WAYMO_CACHE_500_799`: path for `waymo_500-799`.
- `WAYMO_CACHE_800_999`: path for `waymo_800-999`.

## Reproduction Workflow

The complete workflow has three stages:

1. Read indexed interaction segments from the lightweight Waymo cache and compute objective metrics.
2. Generate subjective evaluation prompts and optionally call GPT-4o for subjective scoring.
3. Train the final evaluation model using the prepared training table.

If you only want to verify that the package runs locally, you can run Stage 1 and Stage 3 first and skip the online LLM calls.

## Stage 1: Objective Metric Extraction

Enter the objective metric module:

```bash
cd "Code and Data/1-objective_metrics"
```

Run the metric scripts:

```bash
python safety_metrics.py
python comfort_metrics.py
python efficiency_metrics.py
python interaction_metrics.py
python impact_metrics.py
```

Outputs are saved under:

```text
1-objective_metrics/output/
```

Script descriptions:

- `safety_metrics.py`: computes safety-related metrics. The current script entry computes TTC by default; the PET computation function is retained in the code.
- `comfort_metrics.py`: computes longitudinal acceleration, lateral acceleration, speed, jerk, and yaw rate.
- `efficiency_metrics.py`: computes task-time efficiency and background-vehicle mean speed.
- `interaction_metrics.py`: computes the social interaction metric IO.
- `impact_metrics.py`: computes the impact on surrounding traffic flow.

The interaction rows are read from:

```text
data/waymo_data index.csv
```

The trajectory and map cache data are read from:

```text
data/waymo_lightweight_cache
```

## Stage 2: Subjective Evaluation

Enter the subjective evaluation module:

```bash
cd "Code and Data/2-subjective_evaluation"
```

Generate DLMM prompts:

```bash
python dlmm_model.py
```

Generated prompts are saved to:

```text
2-subjective_evaluation/prompt/prompts.csv
```

To call GPT-4o for subjective evaluation, set an OpenAI API key first:

```bash
export OPENAI_API_KEY="your_api_key"
```

Then run:

```bash
python LLM_evaluate.py
```

LLM responses are saved to:

```text
2-subjective_evaluation/output/evaluation_results.csv
```

Extract numeric scores from the LLM responses:

```bash
python extract_score.py
```

Extracted scores are saved to:

```text
2-subjective_evaluation/output/evaluation_score.csv
```

If you do not have an OpenAI API key, you can run only `dlmm_model.py` and skip `LLM_evaluate.py` and `extract_score.py`.

## Stage 3: Evaluation Model Training

Enter the model training module:

```bash
cd "Code and Data/3-evaluation_model"
```

Run:

```bash
python train_evaluation_model.py
```

By default, the script reads:

```text
data/train_data.csv
```

The included training table already contains normalized objective-metric values and the averaged score from three LLM evaluations for the 705 lightweight interaction samples. Therefore, users can directly run the model training script without repeating the subjective evaluation process.

For training with a custom dataset, users should first extract the objective metrics, normalize them following the procedure described in the paper, and prepare a training table with the same feature structure as `train_data.csv`.

Outputs are written to:

```text
3-evaluation_model/output/
```

Output files:

- `evaModel_lightweight.pth`: trained lightweight evaluation model weights.
- `loss.png`: training and validation loss curves.
- `attention_matrix.png`: attention-based feature relationship matrix.

## Recommended First Run

For a first-time user, the recommended order is:

1. Check that `data/waymo_lightweight_cache` exists and contains the four cache groups.
2. Install the required dependencies and confirm that trajdata Waymo support is available.
3. Run the scripts in `1-objective_metrics` and check whether CSV files are generated under `output`.
4. Run `2-subjective_evaluation/dlmm_model.py` and check `prompt/prompts.csv`.
5. If an OpenAI API key is available, run `LLM_evaluate.py` and `extract_score.py`.
6. Run `3-evaluation_model/train_evaluation_model.py` and check the model weights and figure outputs.
