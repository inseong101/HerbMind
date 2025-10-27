import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from collections import Counter

from herbmind.data import StepwiseDataset
from herbmind.pipeline.trainer import Trainer
from herbmind.config import parser


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
            print("Warning: No common Korean fonts found. Trying default sans-serif.")
            plt.rc('font', family='sans-serif')
    except Exception as e:
        print(f"Warning: Could not set Korean font. {e}")
        plt.rc('font', family='sans-serif')


def generate_dataset_statistics(dataset, output_dir):
    """
    [Table 1] Generates statistics about the herb dataset.
    """
    print("--- [Table 1] 데이터셋 통계 생성 중 ---")
    prescriptions = dataset.prescriptions
    n_prescriptions = len(prescriptions)
    n_herbs = dataset.n_items

    herb_counts_per_prescription = [len(h) for h in prescriptions.values()]

    all_herbs_list = [herb for herbs in prescriptions.values() for herb in herbs]
    herb_frequency = Counter(all_herbs_list)

    stats = {
        '총 처방 개수 (Unique)': n_prescriptions,
        '총 약재 개수 (Unique)': n_herbs,
        '처방당 평균 약재 수': sum(herb_counts_per_prescription) / n_prescriptions if n_prescriptions else 0,
        '처방당 최소 약재 수': min(herb_counts_per_prescription) if herb_counts_per_prescription else 0,
        '처방당 최대 약재 수': max(herb_counts_per_prescription) if herb_counts_per_prescription else 0,
    }

    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ['통계치']

    top_10_herbs = herb_frequency.most_common(10)
    top_10_df = pd.DataFrame(top_10_herbs, columns=['약재명', '등장 횟수'])

    table_path = os.path.join(output_dir, "dataset_statistics.xlsx")
    with pd.ExcelWriter(table_path) as writer:
        stats_df.to_excel(writer, sheet_name='데이터셋 요약')
        top_10_df.to_excel(writer, sheet_name='약재 사용 빈도 Top 10')

    print(f"[Table 1] 데이터셋 통계 저장 완료: {table_path}")


def _extract_embeddings(model, dataset):
    if hasattr(model, 'item_embed'):
        weights = model.item_embed.weight.detach().cpu().numpy()
    else:
        print("Warning: Model does not expose item embeddings; using identity vectors for t-SNE.")
        weights = torch.eye(dataset.n_items).cpu().numpy()
    return weights


def plot_embedding_tsne(model, dataset, output_dir, target_herbs):
    """
    [Figure 3] Plots the learned herb embeddings using t-SNE.
    """
    print("--- [Figure 3] 약재 관계 지도 (t-SNE) 생성 중 ---")
    set_korean_font()

    idx2item = {v: k for k, v in dataset.item2idx.items()}
    embeddings = _extract_embeddings(model, dataset)

    print("  t-SNE 계산 시작... (시간이 걸릴 수 있습니다)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca')
    tsne_results = tsne.fit_transform(embeddings)
    print("  t-SNE 계산 완료.")

    plt.figure(figsize=(20, 20))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.3, color='blue')

    count = 0
    for i in range(len(idx2item)):
        herb_name = idx2item[i]
        if herb_name in target_herbs:
            plt.annotate(
                herb_name,
                (tsne_results[i, 0], tsne_results[i, 1]),
                fontsize=10,
                color='red',
                fontweight='bold'
            )
            count += 1

    plt.title("학습된 약재 관계 지도 (t-SNE 시각화)")
    plt.xlabel("t-SNE 1번 축")
    plt.ylabel("t-SNE 2번 축")
    plt.grid(True, linestyle='--', alpha=0.5)

    output_path = os.path.join(output_dir, "figure_3_embedding_tsne_map.png")
    plt.savefig(output_path)
    print(f"[Figure 3] 약재 관계 지도 저장 완료 (총 {count}개 약재 레이블): {output_path}")


if __name__ == '__main__':
    args = parser.parse_args()

    args.data_dir = './data/'
    args.model_path = './model_best.pth'
    args.plot_dir = './plots/'
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    args.device = torch.device('cpu')
    dataset = StepwiseDataset(args)
    args.n_items = dataset.n_items

    generate_dataset_statistics(dataset, args.plot_dir)

    trainer = Trainer(args)
    trainer.load_model(args.model_path)
    model = trainer.model
    model.eval()

    target_herbs = ['숙지황', '山茱萸', '산약', '택사', '목단피', '복령', '麻黃', '계지', '행인', '감초']

    plot_embedding_tsne(model, dataset, args.plot_dir, target_herbs)
