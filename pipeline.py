import os
import yaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from herbmind.config import parser
from herbmind.data import StepwiseDataset
from herbmind.models.models import HerbMind
from herbmind.pipeline.trainer import Trainer


def set_korean_font():
    """
    Sets the matplotlib font to a Korean font if available.
    """
    try:
        font_name = None
        for font in fm.fontManager.ttflist:
            if 'AppleGothic' in font.name or 'Malgun Gothic' in font.name or 'NanumGothic' in font.name:
                font_name = font.name
                break

        if font_name:
            plt.rc('font', family=font_name)
            print(f"Korean font set to: {font_name}")
        else:
            print("Warning: No common Korean fonts found (AppleGothic, Malgun Gothic, NanumGothic). Trying default sans-serif.")
            plt.rc('font', family='sans-serif')
    except Exception as e:
        print(f"Warning: Could not set Korean font. {e}")
        plt.rc('font', family='sans-serif')


def get_recommendations(model, query_set, dataset, k=3):
    """
    Gets the Top-k recommendations for the current query set.
    """
    batch = dataset.get_inference_batch(query_set)
    batch = model.prep_batch(batch)
    with torch.no_grad():
        pred = model(batch)

    pred = pred.view(-1)
    scores, indices = torch.topk(pred, k)

    recs = [batch['item_ids'][i.item()] for i in indices]
    scores_list = scores.detach().cpu().numpy().squeeze().tolist()

    if not isinstance(recs, list):
        recs = [recs]
    if isinstance(scores_list, float):
        scores_list = [scores_list]

    return recs, scores_list


def plot_cross_attention_barchart(pmx_attn, query_set, best_rec, step, fpath):
    """
    [Figure 5] Plots a bar chart of cross-attention weights.
    Shows who the 'new herb' (best_rec) pays attention to in the 'existing set' (query_set).
    """
    plt.clf()

    if pmx_attn is None:
        print("  [Plot Warning] Cross-attention map unavailable for this step.")
        return

    weights = pmx_attn.mean(0).detach().cpu().numpy().squeeze()
    query_list = list(query_set)

    if len(weights) != len(query_list):
        print(f"  [Plot Error] Cross-attention weight/query mismatch: {len(weights)} vs {len(query_list)}")
        return

    plt.figure(figsize=(max(len(query_list) * 0.8, 5), 4))
    sns.barplot(x=query_list, y=weights, palette="viridis")

    title = f"Step {step+1}: AI가 '{best_rec}'를 추가할 때\n기존 묶음에 부여한 주목도 (교차-어텐션)"
    plt.title(title, fontsize=12)
    plt.ylabel("Attention Weight (주목도)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = f"{fpath}_step{step+1}_cross_attn_barchart.png"
    plt.savefig(save_path)
    print(f"    ... [Figure 5] 교차-어텐션 바 차트 저장: {save_path}")


def plot_self_attention_heatmap(sab_attn, query_set, step, fpath):
    """
    [Figure 4] Plots a heatmap of self-attention weights.
    Shows how the 'existing set' (query_set) members pay attention to each other.
    """
    plt.clf()

    if sab_attn is None:
        print("  [Plot Warning] Self-attention map unavailable for this step.")
        return

    weights = sab_attn.mean(0).detach().cpu().numpy().squeeze()
    query_list = list(query_set)

    if weights.shape[0] != len(query_list) or weights.shape[1] != len(query_list):
        print(f"  [Plot Error] Self-attention matrix/query mismatch: {weights.shape} vs {len(query_list)}")
        return

    plt.figure(figsize=(max(len(query_list) * 0.7, 5), max(len(query_list) * 0.7, 5)))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="rocket_r",
                xticklabels=query_list, yticklabels=query_list)

    title = f"Step {step+1}: 기존 묶음 {query_list}의\n내부 상호작용 분석 (자가-어텐션)"
    plt.title(title, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = f"{fpath}_step{step+1}_self_attn_heatmap.png"
    plt.savefig(save_path)
    print(f"    ... [Figure 4] 자가-어텐션 히트맵 저장: {save_path}")


if __name__ == '__main__':
    args = parser.parse_args()

    set_korean_font()
    sns.set_theme(style="whitegrid")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    dataset = StepwiseDataset(args)
    args.n_items = dataset.n_items

    model = HerbMind(args).to(device)

    trainer = Trainer(args)
    trainer.load_model(args.model_path)
    model = trainer.model
    model.eval()

    with open(args.cases, 'r', encoding='utf-8') as f:
        cases = yaml.safe_load(f)

    all_case_results = []

    for idx, (case_id, case) in enumerate(cases.items()):
        if args.cases_limit is not None and idx >= args.cases_limit:
            break
        print(f"\n===== {case.get('title_ko', case_id)} 사례 연구 시작 =====")
        query_set = set(case['seeds'])
        n_steps = case.get('steps', 8)

        case_study_data = []

        for i in range(n_steps):
            print(f"--- Step {i+1} / {n_steps} ---")
            print(f"  현재 묶음: {query_set}")

            recs, scores = get_recommendations(model, query_set, dataset, k=3)
            best_rec = recs[0]

            step_data = {'Step': i+1, 'Query Set': ', '.join(sorted(list(query_set)))}
            for k_idx in range(len(recs)):
                step_data[f'Top {k_idx+1} Rec'] = recs[k_idx]
                step_data[f'Top {k_idx+1} Score'] = f"{scores[k_idx]:.4f}" if isinstance(scores[k_idx], float) else scores[k_idx]
            case_study_data.append(step_data)

            batch = dataset.get_inference_batch(query_set, best_rec)
            batch = model.prep_batch(batch)

            pmx_attn, sab_attn = model.get_attnmaps(batch)

            fpath_base = os.path.join(args.plot_dir, f"{case_id}")

            plot_cross_attention_barchart(pmx_attn, query_set, best_rec, i, fpath_base)

            if len(query_set) > 1:
                plot_self_attention_heatmap(sab_attn, query_set, i, fpath_base)

            query_set.add(best_rec)

        df = pd.DataFrame(case_study_data)
        table_fpath = os.path.join(args.plot_dir, f"{case_id}_recommendation_table.csv")
        df.to_csv(table_fpath, index=False, encoding='utf-8-sig')
        print(f"===== {case_id} 사례 연구 [Table 1] 저장: {table_fpath} =====")
        all_case_results.append((case_id, df))

    excel_fpath = os.path.join(args.plot_dir, "all_cases_summary.xlsx")
    with pd.ExcelWriter(excel_fpath) as writer:
        for case_id, df in all_case_results:
            df.to_excel(writer, sheet_name=case_id, index=False)
    print(f"\n모든 사례 연구 요약본 Excel 저장: {excel_fpath}")
