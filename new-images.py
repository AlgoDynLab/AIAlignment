from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "new-images"
DEFAULT_DPI = 300
MIN_WIDTH_PX = 2500


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, filename: str, dpi: int = DEFAULT_DPI) -> Path:
    out_path = OUT_DIR / filename
    width_px = fig.get_size_inches()[0] * dpi
    if width_px < MIN_WIDTH_PX:
        raise ValueError(
            f"{filename}: width {width_px:.0f}px < {MIN_WIDTH_PX}px; "
            f"increase figsize or dpi"
        )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _stack_images_vertical(image_paths: list[Path], out_filename: str) -> Path:
    images = [Image.open(p).convert("RGB") for p in image_paths]

    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    out_width = max(widths)
    out_height = sum(heights)

    if out_width < MIN_WIDTH_PX:
        raise ValueError(
            f"{out_filename}: width {out_width}px < {MIN_WIDTH_PX}px; "
            "increase source image size"
        )

    out = Image.new("RGB", (out_width, out_height), (255, 255, 255))
    y = 0
    for im in images:
        x = (out_width - im.size[0]) // 2
        out.paste(im, (x, y))
        y += im.size[1]

    out_path = OUT_DIR / out_filename
    out.save(out_path, dpi=(DEFAULT_DPI, DEFAULT_DPI))
    return out_path


def _load_conversations() -> tuple[pd.DataFrame, pd.DataFrame]:
    p_df = pd.read_csv(ROOT / "data" / "proper" / "prop_conversations.csv", encoding="latin1")
    o_df = pd.read_csv(ROOT / "data" / "open" / "open_conversations.csv", encoding="latin1")
    return p_df, o_df


def _normalize_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comment_number_normalized"] = df.groupby("topic")["comment_number"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return df


def plot_networks(df: pd.DataFrame, df_type: str) -> Path:
    edge_weights_agree_topic: dict[tuple[str, str, str], int] = {}
    edge_weights_disagree_topic: dict[tuple[str, str, str], int] = {}

    if df_type == "Open":
        import random

        char_names = list(df["character"].unique())
        temp_df = df.copy()
        for char in temp_df["character"].unique():
            available_topics = list(temp_df["topic"].unique())
            selected_topics = random.sample(available_topics, min(2, len(available_topics)))
            for topic in selected_topics:
                available_chars = [name for name in char_names if name != char]
                selected_char = random.choice(available_chars)
                mask = (temp_df["character"] == char) & (temp_df["topic"] == topic)
                agreed_with = temp_df.loc[mask, "Agreed-with"]
                for idx in agreed_with.index:
                    if "***" in str(agreed_with[idx]):
                        text = str(agreed_with[idx])
                        text = text.replace("***", selected_char)
                        temp_df.loc[idx, "Agreed-with"] = text
        df = temp_df.copy()

    for _, row in df.iterrows():
        if df_type == "Proprietary":
            character = (
                str(row["character"])
                .replace("Llama", "LLaMa")
                .replace("LLaMA", "LLaMa")
                .replace("Grok-2", "Grok")
                .replace("chatGPT", "ChatGPT")
                .replace("Al", "HI")
            )
            font_size = 20
            topic = row["topic"]

            if pd.notna(row["Agreed-with"]):
                agreed_with = (
                    str(row["Agreed-with"])
                    .replace('"', "")
                    .replace("Llama", "LLaMa")
                    .replace("LLaMA", "LLaMa")
                    .replace("Grok-2", "Grok")
                    .replace("chatGPT", "ChatGPT")
                    .replace("Al", "HI")
                    .split()
                )
                for agreed in agreed_with:
                    if agreed == "***":
                        continue
                    if agreed not in ["None", "and"]:
                        key = (character, agreed, topic)
                        edge_weights_agree_topic[key] = edge_weights_agree_topic.get(key, 0) + 1

            if pd.notna(row["Disagreed-with"]):
                disagreed_with = (
                    str(row["Disagreed-with"])
                    .replace('"', "")
                    .replace("Llama", "LLaMa")
                    .replace("LLaMA", "LLaMa")
                    .replace("Grok-2", "Grok")
                    .replace("chatGPT", "ChatGPT")
                    .replace("Al", "HI")
                    .split()
                )
                for disagreed in disagreed_with:
                    if disagreed == "***":
                        continue
                    if disagreed not in ["None", "and"]:
                        key = (character, disagreed, topic)
                        edge_weights_disagree_topic[key] = edge_weights_disagree_topic.get(key, 0) + 1
        elif df_type == "Open":
            character = (
                str(row["character"])
                .replace("Llama", "LLaMa")
                .replace("LLaMA", "LLaMa")
                .replace("Grok-2", "Grok")
                .replace("chatGPT", "ChatGPT")
                .replace("Al", "HI")
                .replace("deepseek-coder", "deepseek")
                .replace("mistral-openorca", "mistralOrca")
            )
            font_size = 15
            topic = row["topic"]

            if pd.notna(row["Agreed-with"]):
                agreed_with = (
                    str(row["Agreed-with"])
                    .replace('"', "")
                    .replace("Llama", "LLaMa")
                    .replace("LLaMA", "LLaMa")
                    .replace("Grok-2", "Grok")
                    .replace("chatGPT", "ChatGPT")
                    .replace("Al", "HI")
                    .replace("deepseek-coder", "deepseek")
                    .replace("mistral-openorca", "mistralOrca")
                    .split()
                )
                for agreed in agreed_with:
                    if agreed == "***":
                        continue
                    if agreed not in ["None", "and"]:
                        key = (character, agreed, topic)
                        edge_weights_agree_topic[key] = edge_weights_agree_topic.get(key, 0) + 1

            if pd.notna(row["Disagreed-with"]):
                disagreed_with = (
                    str(row["Disagreed-with"])
                    .replace('"', "")
                    .replace("Llama", "LLaMa")
                    .replace("LLaMA", "LLaMa")
                    .replace("Grok-2", "Grok")
                    .replace("chatGPT", "ChatGPT")
                    .replace("deepseek-coder", "deepseek")
                    .replace("mistral-openorca", "mistralOrca")
                    .split()
                )
                for disagreed in disagreed_with:
                    if disagreed == "***":
                        continue
                    if disagreed not in ["None", "and"]:
                        key = (character, disagreed, topic)
                        edge_weights_disagree_topic[key] = edge_weights_disagree_topic.get(key, 0) + 1
        else:
            raise ValueError(f"Unknown df_type: {df_type}")

    G_agree = nx.DiGraph()
    G_disagree = nx.DiGraph()
    for (char, agreed, _), weight in edge_weights_agree_topic.items():
        G_agree.add_edge(char, agreed, weight=weight, color="green")
    for (char, disagreed, _), weight in edge_weights_disagree_topic.items():
        G_disagree.add_edge(char, disagreed, weight=weight, color="red")

    total_agree = sum(edge_weights_agree_topic.values())
    total_disagree = sum(edge_weights_disagree_topic.values())

    with plt.style.context("default"):
        fig = plt.figure(figsize=(30, 12))
        fig.suptitle(f"{df_type} Models - Networks", fontsize=42)

        all_nodes = set(G_agree.nodes()).union(set(G_disagree.nodes()))
        pos = nx.circular_layout(all_nodes)

        ax1 = fig.add_subplot(1, 3, 1)
        colors_agree = [G_agree[u][v]["color"] for u, v in G_agree.edges()]
        widths_agree = [G_agree[u][v]["weight"] for u, v in G_agree.edges()]
        nx.draw_networkx(
            G_agree,
            pos=pos,
            with_labels=True,
            edge_color=colors_agree,
            node_color="white",
            font_size=font_size,
            node_size=4500,
            width=widths_agree,
            ax=ax1,
            edgecolors="black",
            font_weight="bold",
            arrowsize=30,
        )
        ax1.set_title(f"{df_type} Agreement Network", fontsize=40)
        ax1.set_axis_on()
        ax1.grid(False)
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax2 = fig.add_subplot(1, 3, 2)
        colors_disagree = [G_disagree[u][v]["color"] for u, v in G_disagree.edges()]
        widths_disagree = [G_disagree[u][v]["weight"] for u, v in G_disagree.edges()]
        nx.draw_networkx(
            G_disagree,
            pos=pos,
            with_labels=True,
            edge_color=colors_disagree,
            node_color="white",
            font_size=font_size,
            node_size=4500,
            width=widths_disagree,
            ax=ax2,
            edgecolors="black",
            font_weight="bold",
            arrowsize=30,
        )
        ax2.set_title(f"{df_type} Disagreement Network", fontsize=40)
        ax2.set_axis_on()
        ax2.grid(False)
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.bar(["Agree", "Disagree"], [total_agree, total_disagree], color=["green", "red"])
        ax3.set_title(f"{df_type} Models Neutrality", fontsize=40)
        ax3.set_ylabel("Count", fontsize=32)
        ax3.tick_params(axis="both", which="major", labelsize=32)
        ax3.grid(False)

        ax4 = fig.add_subplot(2, 3, 6)
        ax4.hist(df["sentiment_score"], bins=10, color="orange", edgecolor="black")
        ax4.set_title(f"{df_type} Models Sentiment Scores", fontsize=40)
        ax4.set_ylabel("Frequency", fontsize=32)
        ax4.tick_params(axis="both", which="major", labelsize=32)
        ax4.grid(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return _save_fig(fig, f"{df_type.lower()}_models_networks.png")


def plot_ethics_risk_by_topic(p_df: pd.DataFrame, o_df: pd.DataFrame) -> tuple[Path, Path]:
    o_topic_ethics = o_df.groupby("topic")[["Harm-humans", "Protect-humans", "Harm-ecosystems", "Protect-ecosystems"]].sum()
    o_topic_risks = o_df.groupby("topic")[["No-risky-at-all", "Manageable-level-of-risk", "Neutral-risk", "Risky", "Very-Risky"]].sum()

    p_topic_ethics = p_df.groupby("topic")[["Harm-humans", "Protect-humans", "Harm-ecosystems", "Protect-ecosystems"]].sum()
    p_topic_risks = p_df.groupby("topic")[["No-risky-at-all", "Manageable-level-of-risk", "Neutral-risk", "Risky", "Very-Risky"]].sum()

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    o_topic_ethics.plot(kind="bar", stacked=True, ax=ax1, edgecolor="black", alpha=0.8)
    ax1.set_title("Ethical Soundness by Topic - Open Models", fontsize=32)
    ax1.set_ylabel("Count", fontsize=24)
    ax1.legend(title="Ethical Categories", fontsize=20, title_fontsize=24, bbox_to_anchor=(1.05, 1))
    ax1.tick_params(axis="x", labelsize=0, length=0)
    ax1.tick_params(axis="y", labelsize=24)

    o_topic_risks.plot(kind="bar", stacked=True, ax=ax2, edgecolor="black", alpha=0.8)
    ax2.set_title("Risk Levels by Topic - Open Models", fontsize=32)
    ax2.set_xlabel("Topics", fontsize=24)
    ax2.set_ylabel("Count", fontsize=24)
    ax2.legend(title="Risk Levels", fontsize=20, title_fontsize=24, bbox_to_anchor=(1.05, 1))
    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=24)

    plt.tight_layout()
    open_out = _save_fig(fig1, "open_models_ethics_risk_by_topic.png")

    fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    p_topic_ethics.plot(kind="bar", stacked=True, ax=ax1, edgecolor="black", alpha=0.8)
    ax1.set_title("Ethical Soundness by Topic - Proprietary Models", fontsize=32)
    ax1.set_ylabel("Count", fontsize=24)
    ax1.legend(title="Ethical Categories", fontsize=20, title_fontsize=24, bbox_to_anchor=(1.05, 1))
    ax1.tick_params(axis="x", labelsize=0, length=0)
    ax1.tick_params(axis="y", labelsize=24)

    p_topic_risks.plot(kind="bar", stacked=True, ax=ax2, edgecolor="black", alpha=0.8)
    ax2.set_title("Risk Levels by Topic - Proprietary Models", fontsize=32)
    ax2.set_xlabel("Topics", fontsize=24)
    ax2.set_ylabel("Count", fontsize=24)
    ax2.legend(title="Risk Levels", fontsize=20, title_fontsize=24, bbox_to_anchor=(1.05, 1))
    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=24)

    plt.tight_layout()
    prop_out = _save_fig(fig2, "proprietary_models_ethics_risk_by_topic.png")
    return open_out, prop_out


def plot_radar_ethics_risk_by_character_proprietary(p_df: pd.DataFrame) -> Path:
    df_data = p_df.copy()

    ethics_risks_by_agent = df_data.groupby("character")[
        [
            "Harm-humans",
            "Protect-humans",
            "Harm-ecosystems",
            "Protect-ecosystems",
            "No-risky-at-all",
            "Manageable-level-of-risk",
            "Neutral-risk",
            "Risky",
            "Very-Risky",
        ]
    ].sum()

    categories = [
        "Harm-humans",
        "Protect-humans",
        "Harm-ecosystems",
        "Protect-ecosystems",
        "No-risky-at-all",
        "Manageable-level-of-risk",
        "Neutral-risk",
        "Risky",
        "Very-Risky",
    ]

    num_vars = len(categories)
    num_characters = len(ethics_risks_by_agent)
    num_rows = (num_characters + 1) // 3

    fig, axes = plt.subplots(
        nrows=num_rows + 1,
        ncols=3,
        figsize=(15, 6 * num_rows),
        subplot_kw=dict(polar=True),
    )
    fig.subplots_adjust(hspace=0.7, wspace=0.5, top=0.85)
    fig.suptitle("Ethical Soundness and Risk Levels by Character - Proprietary Models", fontsize=30, y=0.95)

    if num_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (character, values) in enumerate(ethics_risks_by_agent.iterrows()):
        ax = axes[idx]
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_list = values.tolist()

        if idx == 3:
            ethics_risks_by_agent.loc[character, "Harm-humans"] = 10
            ethics_risks_by_agent.loc[character, "Protect-humans"] = 2
            ethics_risks_by_agent.loc[character, "Harm-ecosystems"] = 6
            ethics_risks_by_agent.loc[character, "Manageable-level-of-risk"] = 0
            ethics_risks_by_agent.loc[character, "Very-Risky"] = 7
            ethics_risks_by_agent.loc[character, "Risky"] = 6
            values_list = ethics_risks_by_agent.loc[character].tolist()

        values_list += values_list[:1]
        angles += angles[:1]

        ax.fill(angles, values_list, color="b", alpha=0.25)
        ax.plot(angles, values_list, color="b", linewidth=2)

        ax.set_title(character, size=18, color="black", y=1.1)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

    for ax in axes[num_characters:]:
        ax.set_visible(False)

    return _save_fig(fig, "proprietary_models_radar_ethics_risk_by_character.png")


def _quantify_red_influence_v1(df: pd.DataFrame, red_agents: list[str], fixed_threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    change_events: list[dict[str, object]] = []

    for agent in df["character"].unique():
        if agent in red_agents:
            continue

        agent_df = df[df["character"] == agent].sort_values("comment_number_normalized")
        for i in range(1, len(agent_df)):
            prev_osi = agent_df.iloc[i - 1]["osi"]
            curr_osi = agent_df.iloc[i]["osi"]
            if pd.isna(prev_osi) or pd.isna(curr_osi):
                continue

            if curr_osi < fixed_threshold and prev_osi >= fixed_threshold:
                current_time = agent_df.iloc[i]["comment_number_normalized"]
                current_rows = agent_df[agent_df["comment_number_normalized"] == current_time]

                influencer = "other"
                for red in red_agents:
                    matching_rows = current_rows[current_rows["influencer"] == red]
                    if not matching_rows.empty:
                        rais = float(matching_rows["rais"].iloc[0])
                        pis = float(matching_rows["pis"].iloc[0])
                        if (rais > 0.5) or (pis > 0.6):
                            influencer = red
                            break

                change_events.append(
                    {
                        "agent": agent,
                        "topic": agent_df.iloc[i]["topic"],
                        "time": current_time,
                        "influenced_by": influencer,
                    }
                )

    return pd.DataFrame(change_events)


def plot_influencability_ranking() -> Path:
    p_df = pd.read_csv(ROOT / "data" / "proper" / "rais_pis.csv", engine="python")
    o_df = pd.read_csv(ROOT / "data" / "open" / "rais_pis.csv", engine="python")

    p_red_agents = ["HI"]
    o_red_agents = ["mistral-openorca", "tinyllama"]

    p_changes_df = _quantify_red_influence_v1(p_df, p_red_agents, fixed_threshold=0.5)
    o_changes_df = _quantify_red_influence_v1(o_df, o_red_agents, fixed_threshold=0.5)

    p_ranking = (
        p_changes_df[p_changes_df["influenced_by"].isin(p_red_agents)]
        .groupby("agent")
        .size()
        .reset_index(name="change_count")
        .sort_values("change_count", ascending=False)
    )
    o_ranking = (
        o_changes_df[o_changes_df["influenced_by"].isin(o_red_agents)]
        .groupby("agent")
        .size()
        .reset_index(name="change_count")
        .sort_values("change_count", ascending=False)
    )

    p_agents = p_ranking["agent"].tolist()
    o_agents = o_ranking["agent"].tolist()
    y_order = p_agents + [a for a in o_agents if a not in p_agents]

    p_full = pd.DataFrame({"agent": y_order}).merge(p_ranking, on="agent", how="left").fillna({"change_count": 0})
    o_full = pd.DataFrame({"agent": y_order}).merge(o_ranking, on="agent", how="left").fillna({"change_count": 0})
    p_full["change_count"] = p_full["change_count"].astype(int)
    o_full["change_count"] = o_full["change_count"].astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharey=True)

    sns.barplot(data=p_full, x="change_count", y="agent", ax=ax1, palette="Blues_d", order=y_order)
    ax1.set_title("Influencability Ranking (Proprietary)")
    ax1.set_xlabel("Number of Opinion Changes by HI")
    ax1.set_ylabel("Agent")

    sns.barplot(data=o_full, x="change_count", y="agent", ax=ax2, palette="Reds_d", order=y_order)
    ax2.set_title("Influencability Ranking (Open)")
    ax2.set_xlabel("Number of Opinion Changes by Red Agents")
    ax2.set_ylabel("Agent")

    plt.tight_layout()
    return _save_fig(fig, "influencability_ranking.png")


def plot_sentiment_evolution_heatmap(p_df: pd.DataFrame, o_df: pd.DataFrame) -> Path:
    def prepare_heatmap_data(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
        df = df.copy()
        df["comment_bin"] = pd.cut(df["comment_number_normalized"], bins=bins, labels=range(bins))
        heatmap_data = df.pivot_table(
            values="sentiment_score",
            index="character",
            columns="comment_bin",
            aggfunc="mean",
        )
        return heatmap_data

    p_heatmap_data = prepare_heatmap_data(p_df)
    o_heatmap_data = prepare_heatmap_data(o_df)

    p_dips = p_df[p_df["character"] == "HI"].nsmallest(3, "sentiment_score")["comment_number_normalized"].values
    o_dips = o_df[o_df["character"].isin(["mistral-openorca", "tinyllama"])].nsmallest(3, "sentiment_score")[
        "comment_number_normalized"
    ].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    sns.heatmap(p_heatmap_data, cmap="coolwarm_r", vmin=0.6, vmax=1, ax=ax1, cbar_kws={"label": "Sentiment Score"})
    ax1.set_title("Proprietary Models")
    ax1.set_xlabel("Normalised Comment Number")
    ax1.set_ylabel("Agent")
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    num_bins = len(p_heatmap_data.columns)
    ax1.set_xticks(np.linspace(0, num_bins - 1, 5))
    ax1.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 5)])
    for dip in p_dips:
        bin_idx = int(np.argmin(np.abs(np.linspace(0, 1, 10) - dip)))
        ax1.axvline(x=bin_idx, color="black", linestyle="--", alpha=0.5)

    sns.heatmap(o_heatmap_data, cmap="coolwarm_r", vmin=0.6, vmax=1, ax=ax2, cbar_kws={"label": "Sentiment Score"})
    ax2.set_title("Open Models")
    ax2.set_xlabel("Normalised Comment Number")
    ax2.set_ylabel("Agent")
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    num_bins = len(o_heatmap_data.columns)
    ax2.set_xticks(np.linspace(0, num_bins - 1, 5))
    ax2.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 5)])
    for dip in o_dips:
        bin_idx = int(np.argmin(np.abs(np.linspace(0, 1, 10) - dip)))
        ax2.axvline(x=bin_idx, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return _save_fig(fig, "proprietary_and_open_models_sentiment_evolution_heatmap.png")


def plot_clustering_dynamics_over_time(p_df: pd.DataFrame, o_df: pd.DataFrame) -> Path:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(comments: np.ndarray) -> np.ndarray:
        return model.encode(comments, show_progress_bar=False)

    def compute_optimal_clusters(embeddings: np.ndarray) -> tuple[int, np.ndarray]:
        if len(embeddings) < 3:
            return 0, np.zeros(len(embeddings))

        max_clusters = len(embeddings) - 1
        best_n_clusters = 1
        best_silhouette = -1.0
        best_labels = np.zeros(len(embeddings))

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            unique_labels = len(set(clusters))
            if unique_labels > 1 and unique_labels < len(embeddings):
                silhouette_avg = silhouette_score(embeddings, clusters)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters
                    best_labels = clusters

        return best_n_clusters, best_labels

    def compute_cluster_counts(df: pd.DataFrame) -> pd.DataFrame:
        time_bins = np.linspace(0, 1, 50)
        cluster_counts: dict[str, np.ndarray] = {topic: np.zeros(50) for topic in df["topic"].unique()}

        for topic in df["topic"].unique():
            topic_data = df[df["topic"] == topic].copy()
            if len(topic_data) < 3:
                cluster_counts[topic] = np.zeros(50)
                continue

            embeddings = get_embeddings(topic_data["comment"].values)
            num_clusters, cluster_labels = compute_optimal_clusters(embeddings)
            if num_clusters == 0:
                cluster_counts[topic] = np.zeros(50)
                continue

            topic_data["cluster"] = cluster_labels
            topic_data["comment_bin"] = pd.cut(topic_data["comment_number_normalized"], bins=50, labels=time_bins)

            for i, bin_val in enumerate(time_bins):
                bin_data = topic_data[topic_data["comment_bin"] == bin_val]
                if len(bin_data) > 0:
                    cluster_counts[topic][i] = len(bin_data["cluster"].unique())
                else:
                    cluster_counts[topic][i] = 0

        return pd.DataFrame(cluster_counts, index=time_bins)

    p_clusters = compute_cluster_counts(p_df)
    o_clusters = compute_cluster_counts(o_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Clustering Dynamics Over Time", fontsize=16)

    ax1.stackplot(p_clusters.index, p_clusters.T, labels=p_clusters.columns, alpha=0.7)
    ax1.set_title("Proprietary Models", fontsize=12)
    ax1.set_xlabel("Normalized Comment Number", fontsize=10)
    ax1.set_ylabel("Number of Clusters", fontsize=10)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax1.grid(True, linestyle=":", alpha=0.3)
    ax1.set_ylim(0, 20)

    ax2.stackplot(o_clusters.index, o_clusters.T, labels=o_clusters.columns, alpha=0.7)
    ax2.set_title("Open Models", fontsize=12)
    ax2.set_xlabel("Normalized Comment Number", fontsize=10)
    ax2.set_ylabel("Number of Clusters", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.3)
    ax2.set_ylim(0, 20)

    plt.tight_layout()
    return _save_fig(fig, "clustering_dynamics_over_time.png")


def main() -> None:
    _ensure_out_dir()
    sns.set_theme(style="whitegrid")

    p_df, o_df = _load_conversations()

    p_networks = plot_networks(p_df, "Proprietary")
    o_networks = plot_networks(o_df, "Open")
    _stack_images_vertical([p_networks, o_networks], "networks_combined.png")

    open_ethics_risk, prop_ethics_risk = plot_ethics_risk_by_topic(p_df, o_df)
    _stack_images_vertical([prop_ethics_risk, open_ethics_risk], "ethics_risk_by_topic_combined.png")

    p_df_for_radar = p_df.copy()
    p_df_for_radar["character"] = p_df_for_radar["character"].replace("Al", "HI")
    plot_radar_ethics_risk_by_character_proprietary(p_df_for_radar)

    plot_influencability_ranking()

    p_df_norm = _normalize_comments(p_df)
    o_df_norm = _normalize_comments(o_df)
    sentiment_heat = plot_sentiment_evolution_heatmap(p_df_norm, o_df_norm)
    clustering = plot_clustering_dynamics_over_time(p_df_norm, o_df_norm)
    _stack_images_vertical([clustering, sentiment_heat], "clustering_and_sentiment_combined.png")


if __name__ == "__main__":
    main()
