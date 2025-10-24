# RecipeMind: Guiding Ingredient Choices from Food Pairing to Recipe Completion using Cascaded Set Transformer
* [paper - CIKM 2022 Proceedings](https://dl.acm.org/doi/abs/10.1145/3511808.3557092)
* [demo - http://recipemind.korea.ac.kr](http://recipemind.korea.ac.kr)

## Abstract

We propose a computational approach for recipe ideation, a down-stream task that helps users select and gather ingredients for creating dishes. To perform this task, we developed RecipeMind, a food affinity score prediction model that quantifies the suitability of adding an ingredient to set of other ingredients. We constructed a large-scale dataset containing ingredient co-occurrence based scores to train and evaluate RecipeMind on food affinity score prediction. Deployed in recipe ideation, RecipeMind helps the user expand an initial set of ingredients by suggesting additional ingredients. Experiments and qualitative analysis show RecipeMind’s potential in fulfilling its assistive role in cuisine domain.

## Overview of Recipe Ideation

![img](./figures/0_task.png)

## Overview of RecipeMind

![img](./figures/1_model.png)

## Prerequisites for running RecipeMind

- Python 3.8.12
- CUDA: 11.X
- Download and extract data.tar.gz ([link](https://drive.google.com/file/d/1xZa4fPQvoxWBX_fvcFtmZjWZj0Fa7pFj/view?usp=sharing), 388.4MB) at directory **./data**. These files are the datasets containing ingredient n-tuplets with food affinity scores and ingredient word embeddings.
- Download and extract saved.tar.gz ([link](https://drive.google.com/file/d/1D_PQcf82-0b4qW3EUGQWV_cQnKezt2Yc/view?usp=sharing), 115.3MB) at directory **./saved**. These files are the model checkpoints for each random seed (1001 ~ 1005).

## Installing the Python (3.8.12) Conda Environment

```
conda env create -f recipemind.yml
conda activate recipemind
```

## Training RecipeMind

Run the following code,
```
./train_script.sh {your_session_name} {random_seed_integer}
```

## Testing RecipeMind in all ingredient set sizes from 2 to 7

Run the following code,
```
./test_script.sh {your_session_name} {random_seed_integer}
```

## Analyzing RecipeMind

The jupyter notebook **RecipeMind Post Analysis.ipynb** contains the source code for deploying the trained RecipeModel model in recipe ideation scenarios starting with any number of ingredients. We provided example cells that output the ideation results and attention heatmaps for interpretation purposes. The example heatmaps are the following,

### Case Study 1: Starting with Carrots and Onions

![img](./figures/2_attnmaps1.png)

### Case Study 2: Starting with Buttermilk and Flour

![img](./figures/3_attnmaps2.png)


## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Mogan Gim&dagger;</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>akim@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Donghee Choi&dagger;</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>choidonghee@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Kana Maruyama</td>		
		<td>Sony AI, Tokyo, Japan</td>
		<td>Kana.Maruyama@sony.com</td>
	</tr>
	<tr>
		<td>Jihun Choi</td>		
		<td>Sony AI, Tokyo, Japan</td>
		<td>Jihun.A.Choi@sony.com</td>
	</tr>
	<tr>
		<td>Donghyeon Park*</td>		
		<td>Food & Nutrition AI Lab,<br>Sejong University, Seoul, South Korea</td>
		<td>parkdh@sejong.ac.kr</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>

</table>



- &dagger;: *Equal Contributors*
- &ast;: *Corresponding Authors*

## Citation
```bibtex
@inproceedings{gim2022recipemind,
  title={RecipeMind: Guiding Ingredient Choices from Food Pairing to Recipe Completion using Cascaded Set Transformer},
  author={Gim, Mogan and Choi, Donghee and Maruyama, Kana and Choi, Jihun and Kim, Hajung and Park, Donghyeon and Kang, Jaewoo},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={3092--3102},
  year={2022}
}
```


Generating Paper Figures & Console Tables

This repo includes a standalone script to generate HerbMind manuscript figures and print tables.

Run

pip install -r requirements.txt
python Mypaperfiguretable.py


Tables (dataset stats, ablations) are printed to console (no CSV export).

Figures are saved as PNG under figures/:

fig1_model_architecture.png

fig2_spmir_distribution.png

fig3_accuracy_comparison.png

fig4_attention_heatmap.png

fig5_qualitative_examples.png


## Quick Start (HerbMind figures & tables)

Put your KIOM CSVs under ./data/

Expected columns (auto-detected): 처방아이디 (prescription id), 약재명 or 약재한글명 (herb name).

Run (one command):

```bash
bash run.sh
```

A local virtualenv (.venv) will be created and dependencies installed (compatible with macOS Homebrew Python).

Tables print to console (no CSV export).

Figures are saved under ./figures/:

- fig1_model_architecture.png
- fig2_spmir_distribution.png
- fig3_accuracy_comparison.png
- fig4_attention_heatmap.png
- fig5_qualitative_examples.png

If you're on Apple Silicon with Homebrew Python (PEP 668), the script handles a local venv automatically.

## Publication-quality figures (600 dpi)

```bash
bash run.sh --pro
```

Uses `Mypaperfiguretable_Pro.py` (Helvetica/Arial fonts, unified palette).

Reads optional `outputs/metrics.json` for real results; otherwise uses placeholders.

## Stepwise application figures (RecipeMind Fig.7/8 style)

```bash
bash run.sh --cases
```

- Generates `figures/fig_caseA_stepwise.png` and `figures/fig_caseB_stepwise.png`.
- Edit `cases.yaml` to change seed herbs or step count.
- Uses model scores if `outputs/model_scores.pkl` is available; otherwise falls back to sPMIr computed from `data/*.csv`.

## RecipeMind-style, layout-matched figures

```bash
bash run.sh --match
```

Produces the publication-layout counterparts:

- `fig1_overview_matched.png`: split panel (User vs. HerbMind, step capsules)
- `fig2_spmir_kde_matched.png`: KDE curves with mean markers
- `fig3_cascaded_arch_matched.png`: cascaded SAB/PMX diagram
- `fig5_baselines_heat_matched.png` and `fig6_ablation_heat_matched.png`: heat tables with per-cell annotations
- `fig7_case*_stepwise_matched.png`: dual tables with highlighted seeds and per-step top-3
- `fig9_case*_attn_matched.png`: triangular attention heatmaps with `?` placeholders for masked cells

Figures are saved under `./figures/`; all scripts respect optional metrics in `outputs/metrics.json` when present.
