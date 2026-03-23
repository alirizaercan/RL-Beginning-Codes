#!/usr/bin/env python3
"""LunarLander action veri seti icin sunuma uygun bir HTML dashboard uret."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STATE_RE = re.compile(
    r"State:\s*\["
    r"x=(?P<x>-?\d+(?:\.\d+)?),\s*"
    r"y=(?P<y>-?\d+(?:\.\d+)?),\s*"
    r"vx=(?P<vx>-?\d+(?:\.\d+)?),\s*"
    r"vy=(?P<vy>-?\d+(?:\.\d+)?),\s*"
    r"angle=(?P<angle>-?\d+(?:\.\d+)?),\s*"
    r"angular_vel=(?P<angular_vel>-?\d+(?:\.\d+)?),\s*"
    r"left_leg=(?P<left_leg>-?\d+(?:\.\d+)?),\s*"
    r"right_leg=(?P<right_leg>-?\d+(?:\.\d+)?)\]"
)
ACTION_RE = re.compile(r"Action:\s*(?P<action_id>\d+)\s*\((?P<action_label>[^)]+)\)")

HF_DATASET_ID = "Ali2023kosemen/lunar_lander_270_reward"
DEFAULT_OUTPUT_PATH = Path("fine-tune/reports/lunar_lander_270_reward_dashboard.html")

ACTION_META = {
    0: {"label": "Action 0", "name": "Do nothing", "color": "#3b82f6"},
    1: {"label": "Action 1", "name": "Fire left engine", "color": "#ef4444"},
    2: {"label": "Action 2", "name": "Fire main engine", "color": "#10b981"},
    3: {"label": "Action 3", "name": "Fire right engine", "color": "#f59e0b"},
}


def load_dataset() -> list[dict]:
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(HF_DATASET_ID, split="train")
    return [dict(row) for row in dataset]


def parse_conversation(sample: dict) -> dict:
    conversations = sample.get("conversations", [])
    if len(conversations) < 2:
        raise ValueError("Sample must contain at least 2 turns.")

    human_text = conversations[0].get("value", "")
    gpt_text = conversations[1].get("value", "")

    state_match = STATE_RE.search(human_text)
    action_match = ACTION_RE.search(gpt_text)
    if not state_match:
        raise ValueError(f"Could not parse state from: {human_text}")
    if not action_match:
        raise ValueError(f"Could not parse action from: {gpt_text}")

    row = {key: float(value) for key, value in state_match.groupdict().items()}
    action_id = int(action_match.group("action_id"))

    row["action_id"] = action_id
    row["action_label"] = ACTION_META[action_id]["label"]
    row["action_name"] = ACTION_META[action_id]["name"]
    row["action_display"] = f"{ACTION_META[action_id]['label']} - {ACTION_META[action_id]['name']}"
    row["left_leg_contact"] = int(row["left_leg"] > 0)
    row["right_leg_contact"] = int(row["right_leg"] > 0)
    row["both_legs_contact"] = int(row["left_leg_contact"] and row["right_leg_contact"])
    row["speed"] = math.sqrt(row["vx"] ** 2 + row["vy"] ** 2)
    return row


def build_dataframe(samples: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(parse_conversation(sample) for sample in samples)
    df["action_id"] = df["action_id"].astype(int)
    df["action_display"] = pd.Categorical(
        df["action_display"],
        categories=[f"{ACTION_META[i]['label']} - {ACTION_META[i]['name']}" for i in range(4)],
        ordered=True,
    )
    return df


def color_map() -> dict[str, str]:
    return {f"{ACTION_META[i]['label']} - {ACTION_META[i]['name']}": ACTION_META[i]["color"] for i in range(4)}


def base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left"},
        height=height,
        margin=dict(l=40, r=24, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#0f172a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1.0,
            title="",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def make_action_count_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["action_display"]
        .value_counts()
        .reindex(df["action_display"].cat.categories)
        .reset_index()
    )
    counts.columns = ["action_display", "count"]
    fig = px.bar(
        counts,
        x="count",
        y="action_display",
        orientation="h",
        color="action_display",
        color_discrete_map=color_map(),
        text="count",
    )
    fig.update_traces(textposition="outside", hovertemplate="%{y}<br>Adet: %{x:,}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(list(df["action_display"].cat.categories))))
    return base_layout(fig, "Action Dagilimi")


def make_action_share_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["action_display"]
        .value_counts()
        .reindex(df["action_display"].cat.categories)
        .reset_index()
    )
    counts.columns = ["action_display", "count"]
    fig = px.pie(
        counts,
        names="action_display",
        values="count",
        hole=0.58,
        color="action_display",
        color_discrete_map=color_map(),
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate="%{label}<br>Oran: %{percent}<br>Adet: %{value:,}<extra></extra>",
    )
    fig.update_layout(
        title={"text": "Action Oranlari", "x": 0.03, "xanchor": "left"},
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#0f172a"),
        showlegend=False,
    )
    return fig


def make_xy_scatter(df: pd.DataFrame) -> go.Figure:
    sampled = pd.concat(
        [
            group.sample(n=min(1800, len(group)), random_state=42).copy()
            for _, group in df.groupby("action_display", observed=False)
        ],
        ignore_index=True,
    )
    fig = px.scatter(
        sampled,
        x="x",
        y="y",
        color="action_display",
        facet_col="action_display",
        facet_col_wrap=2,
        color_discrete_map=color_map(),
        opacity=0.5,
        render_mode="webgl",
        hover_data=["vx", "vy", "angle", "angular_vel"],
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)), showlegend=False)
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=12), yshift=-8)
    )
    fig.update_xaxes(matches=None, title_text="x")
    fig.update_yaxes(matches=None, title_text="y")
    fig.update_layout(
        title={"text": "x / y", "x": 0.03, "xanchor": "left"},
        height=620,
        margin=dict(l=40, r=24, t=92, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#0f172a"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def make_velocity_scatter(df: pd.DataFrame) -> go.Figure:
    sampled = pd.concat(
        [
            group.sample(n=min(1800, len(group)), random_state=24).copy()
            for _, group in df.groupby("action_display", observed=False)
        ],
        ignore_index=True,
    )
    fig = px.scatter(
        sampled,
        x="vx",
        y="vy",
        color="action_display",
        facet_col="action_display",
        facet_col_wrap=2,
        color_discrete_map=color_map(),
        opacity=0.5,
        render_mode="webgl",
        hover_data=["x", "y", "speed"],
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)), showlegend=False)
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=12), yshift=-8)
    )
    fig.update_xaxes(matches=None, title_text="vx")
    fig.update_yaxes(matches=None, title_text="vy")
    fig.update_layout(
        title={"text": "vx / vy", "x": 0.03, "xanchor": "left"},
        height=620,
        margin=dict(l=40, r=24, t=92, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#0f172a"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def make_xy_overlay(df: pd.DataFrame) -> go.Figure:
    sampled = pd.concat(
        [
            group.sample(n=min(2200, len(group)), random_state=11).copy()
            for _, group in df.groupby("action_display", observed=False)
        ],
        ignore_index=True,
    )
    fig = px.scatter(
        sampled,
        x="x",
        y="y",
        color="action_display",
        color_discrete_map=color_map(),
        opacity=0.28,
        render_mode="webgl",
        hover_data=["vx", "vy", "angle", "angular_vel"],
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)), showlegend=False)
    fig = base_layout(fig, "Tum Action'lar Ayni x / y Grafiginde", height=520)
    fig.update_layout(showlegend=False, legend_title_text="")
    return fig


def make_action_by_altitude_chart(df: pd.DataFrame) -> go.Figure:
    altitude = df.copy()
    altitude["y_band"] = pd.cut(altitude["y"], bins=10, include_lowest=True)
    summary = (
        altitude.groupby(["y_band", "action_display"], observed=False)
        .size()
        .reset_index(name="count")
    )
    total_per_band = summary.groupby("y_band", observed=False)["count"].transform("sum")
    summary["ratio"] = summary["count"] / total_per_band
    summary["y_mid"] = summary["y_band"].apply(lambda interval: f"{float(interval.mid):.2f}")
    summary["y_band_label"] = summary["y_mid"]
    fig = px.bar(
        summary,
        x="y_band_label",
        y="ratio",
        color="action_display",
        barmode="stack",
        color_discrete_map=color_map(),
    )
    fig.update_traces(hovertemplate="y band: %{x}<br>Oran: %{y:.2%}<extra></extra>")
    fig.update_yaxes(title="Action orani", tickformat=".0%")
    fig.update_xaxes(title="y merkezi", tickangle=0)
    fig = base_layout(fig, "Yukseklik Bandina Gore Action Oranlari")
    fig.update_layout(showlegend=False, legend_title_text="")
    return fig


def make_action_feature_profile(df: pd.DataFrame) -> go.Figure:
    features = ["x", "y", "vx", "vy", "angle", "angular_vel", "speed"]
    grouped = df.groupby("action_display", observed=False)[features].mean()
    normalized = (grouped - df[features].mean()) / df[features].std()
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized.values,
            x=features,
            y=list(normalized.index),
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.5, "#f8fafc"],
                [1.0, "#14532d"],
            ],
            zmid=0,
            text=[[f"{value:.2f}" for value in row] for row in normalized.values],
            texttemplate="%{text}",
            hovertemplate="Action: %{y}<br>Feature: %{x}<br>Z-score: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, "Action Bazinda Ortalama Ozellik Profili", height=520)


def make_action_by_angle_chart(df: pd.DataFrame) -> go.Figure:
    angle_df = df.copy()
    angle_df["angle_band"] = pd.cut(angle_df["angle"], bins=10, include_lowest=True)
    summary = (
        angle_df.groupby(["angle_band", "action_display"], observed=False)
        .size()
        .reset_index(name="count")
    )
    total_per_band = summary.groupby("angle_band", observed=False)["count"].transform("sum")
    summary["ratio"] = summary["count"] / total_per_band
    summary["angle_mid"] = summary["angle_band"].apply(lambda interval: f"{float(interval.mid):.2f}")
    fig = px.bar(
        summary,
        x="angle_mid",
        y="ratio",
        color="action_display",
        barmode="stack",
        color_discrete_map=color_map(),
    )
    fig.update_traces(hovertemplate="angle band: %{x}<br>Oran: %{y:.2%}<extra></extra>")
    fig.update_yaxes(title="Action orani", tickformat=".0%")
    fig.update_xaxes(title="angle merkezi", tickangle=0)
    fig = base_layout(fig, "Angle Bandina Gore Action Oranlari")
    fig.update_layout(showlegend=False, legend_title_text="")
    return fig


def make_speed_histogram(df: pd.DataFrame) -> go.Figure:
    sampled = pd.concat(
        [
            group.sample(n=min(4000, len(group)), random_state=7).copy()
            for _, group in df.groupby("action_display", observed=False)
        ],
        ignore_index=True,
    )
    fig = px.histogram(
        sampled,
        x="speed",
        color="action_display",
        nbins=28,
        opacity=0.8,
        barmode="overlay",
        color_discrete_map=color_map(),
    )
    fig.update_yaxes(title="Adet")
    fig.update_xaxes(title="speed")
    fig = base_layout(fig, "Speed Dagilimi ve Action")
    fig.update_layout(showlegend=False, legend_title_text="")
    return fig


def make_feature_boxplot(df: pd.DataFrame, feature: str, title: str, yaxis_title: str) -> go.Figure:
    fig = px.box(
        df,
        x="action_display",
        y=feature,
        color="action_display",
        color_discrete_map=color_map(),
        points=False,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title=yaxis_title)
    return base_layout(fig, title)


def make_leg_contact_chart(df: pd.DataFrame) -> go.Figure:
    counts = pd.DataFrame(
        [
            {"metric": "Left leg contact", "count": int(df["left_leg_contact"].sum()), "color": "#14b8a6"},
            {"metric": "Right leg contact", "count": int(df["right_leg_contact"].sum()), "color": "#8b5cf6"},
            {"metric": "Both legs contact", "count": int(df["both_legs_contact"].sum()), "color": "#f97316"},
        ]
    )
    fig = px.bar(
        counts,
        x="metric",
        y="count",
        color="metric",
        color_discrete_sequence=list(counts["color"]),
        text="count",
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>Adet: %{y:,}<extra></extra>")
    fig.update_layout(showlegend=False)
    return base_layout(fig, "Inis Ayagi Temas Sikligi")


def make_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr_cols = ["x", "y", "vx", "vy", "angle", "angular_vel", "speed", "action_id"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.5, "#f8fafc"],
                [1.0, "#14532d"],
            ],
            zmid=0,
            text=[[f"{value:.2f}" for value in row] for row in corr.values],
            texttemplate="%{text}",
            hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, "Ozellik Korelasyon Matrisi")


def metric_card(title: str, value: str, detail: str, accent: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-accent" style="background:{accent};"></div>
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-detail">{detail}</div>
    </div>
    """


def build_summary_cards(df: pd.DataFrame) -> str:
    top_action = df["action_display"].value_counts().idxmax()
    both_legs_ratio = df["both_legs_contact"].mean() * 100
    cards = [
        metric_card("Toplam Ornek", f"{len(df):,}", "Fine-tuning icin kullanilan conversation pair sayisi", "#2563eb"),
        metric_card("En Sik Action", top_action.split(" - ")[0], top_action.split(" - ")[1], "#10b981"),
        metric_card("Ortalama Speed", f"{df['speed'].mean():.3f}", "vx ve vy uzerinden hesaplandi", "#f59e0b"),
        metric_card("Both legs contact", f"{both_legs_ratio:.2f}%", "Iki ayagin birden yere temas ettigi ornekler", "#8b5cf6"),
    ]
    return "\n".join(cards)


def build_findings(df: pd.DataFrame) -> str:
    counts = df["action_display"].value_counts().reindex(df["action_display"].cat.categories)
    max_share = counts.max() / len(df) * 100
    min_share = counts.min() / len(df) * 100
    findings = [
        f"Veri setinde en baskin action sinifi %<b>{max_share:.1f}</b> paya sahip; en dusuk sinif ise %<b>{min_share:.1f}</b> seviyesinde.",
        "x / y ve vx / vy uzaylarinda siniflar belirgin bicimde ust uste biniyor; bu da yalnizca metin tabanli kucuk modeller icin ayrimi zorlastirabilir.",
        "Ozellikle Action 2 veri setinde daha yaygin; bu durum modelin main engine cevaplarina fazla kaymasina yol acabilir.",
        "Leg contact sinyalleri gorece az ise, iniş anina yakin kararlar model tarafinda daha zayif ogrenilmis olabilir.",
    ]
    items = "".join(f"<li>{item}</li>" for item in findings)
    return f"""
    <div class="findings-card">
      <h3>Ana Bulgular</h3>
      <ul>{items}</ul>
    </div>
    """


def build_action_legend() -> str:
    items = []
    for action_id in range(4):
        meta = ACTION_META[action_id]
        items.append(
            f"""
            <div class="legend-item">
              <span class="legend-swatch" style="background:{meta['color']};"></span>
              <span class="legend-text">{meta['label']} - {meta['name']}</span>
            </div>
            """
        )
    return f"""
    <div class="legend-card">
      <h3>Renk Anahtari</h3>
      <div class="legend-grid">
        {''.join(items)}
      </div>
    </div>
    """


def build_overview_table(df: pd.DataFrame) -> str:
    rows = []
    action_counts = df["action_display"].value_counts().reindex(df["action_display"].cat.categories)
    for action, count in action_counts.items():
        ratio = count / len(df) * 100
        rows.append(
            f"""
            <tr>
              <td>{action}</td>
              <td>{count:,}</td>
              <td>{ratio:.2f}%</td>
            </tr>
            """
        )
    return f"""
    <div class="overview-card">
      <h3>Veri Seti Ozeti</h3>
      <p>
        Asagidaki dashboard, action secimini surukleyen state dagilimini ozetler.
        Scatter grafiklerinde okunabilirlik icin ornekleme kullanildi; adet bazli grafikler tum veri setini kullanir.
      </p>
      <table>
        <thead>
          <tr>
            <th>Action</th>
            <th>Adet</th>
            <th>Oran</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      <div class="range-grid">
        <div><b>x range</b><span>{df['x'].min():.4f} to {df['x'].max():.4f}</span></div>
        <div><b>y range</b><span>{df['y'].min():.4f} to {df['y'].max():.4f}</span></div>
        <div><b>vx range</b><span>{df['vx'].min():.4f} to {df['vx'].max():.4f}</span></div>
        <div><b>vy range</b><span>{df['vy'].min():.4f} to {df['vy'].max():.4f}</span></div>
        <div><b>angle range</b><span>{df['angle'].min():.4f} to {df['angle'].max():.4f}</span></div>
        <div><b>angular velocity range</b><span>{df['angular_vel'].min():.4f} to {df['angular_vel'].max():.4f}</span></div>
      </div>
    </div>
    """


def write_dashboard(df: pd.DataFrame, figures: dict[str, go.Figure], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_html = {
        name: fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, (name, fig) in enumerate(figures.items())
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>LunarLander Veri Seti Dashboard</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 26%),
        var(--bg);
      color: var(--ink);
    }}
    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .hero-card, .overview-card, .panel, .findings-card, .legend-card {{
      background: var(--card);
      border: 1px solid rgba(226,232,240,0.9);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      padding: 28px;
    }}
    .eyebrow {{
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #2563eb;
      background: rgba(37,99,235,0.08);
      padding: 8px 12px;
      border-radius: 999px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.08;
    }}
    .hero-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      position: relative;
      background: linear-gradient(180deg, #fff, #f8fbff);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 18px 16px;
      overflow: hidden;
    }}
    .metric-accent {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 4px;
    }}
    .metric-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
      font-weight: 600;
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .metric-detail {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .overview-card {{
      padding: 24px 28px;
      margin-bottom: 24px;
    }}
    .legend-card {{
      padding: 20px 28px;
      margin-bottom: 24px;
    }}
    .legend-card h3 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 16px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      flex: 0 0 14px;
    }}
    .legend-text {{
      font-size: 14px;
      font-weight: 600;
    }}
    .findings-card {{
      padding: 24px 28px;
      margin-bottom: 24px;
      background: linear-gradient(135deg, #0f172a, #1e293b);
      color: #e2e8f0;
    }}
    .findings-card h3 {{
      margin: 0 0 14px;
      color: #ffffff;
      font-size: 22px;
    }}
    .findings-card ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.7;
    }}
    .findings-card li {{
      margin-bottom: 10px;
    }}
    .overview-card h3 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    .overview-card p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.65;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 18px;
      overflow: hidden;
      border-radius: 14px;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      background: #f8fafc;
      font-weight: 700;
    }}
    .range-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .range-grid div {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .range-grid b {{
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
    }}
    .range-grid span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .section-title {{
      margin: 36px 0 14px;
      font-size: 22px;
      font-weight: 800;
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }}
    .grid-one {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .panel {{
      padding: 10px 10px 2px;
    }}
    @media (max-width: 1100px) {{
      .hero, .grid-two, .range-grid {{
        grid-template-columns: 1fr;
      }}
      .metrics {{
        grid-template-columns: 1fr 1fr;
      }}
      .legend-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .container {{
        padding: 20px 14px 32px;
      }}
      .metrics {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">LunarLander Veri Seti Incelemesi (Schema: conversations)</div>
        <h1>Action Dagilimi ve State Kapsami Dashboard'u</h1>
        <p>
          Bu ekran, `Ali2023kosemen/lunar_lander_270_reward` veri setinin mevcut schema'sindaki `conversations` alanini okuyarak LunarLander state'lerini dort ayrik action arasinda nasil dagittigini ozetler.
          Dashboard sunum odaklidir: ozet metrikler, temiz dagilim grafikleri ve toplantida rahat okunabilen orneklemeli state-space gorselleri.
        </p>
      </div>
      <div class="metrics">
        {build_summary_cards(df)}
      </div>
    </section>

    {build_overview_table(df)}
    {build_action_legend()}
    {build_findings(df)}

    <h2 class="section-title">Action Dengesi</h2>
    <div class="grid-two">
      <div class="panel">{chart_html["action_counts"]}</div>
      <div class="panel">{chart_html["action_share"]}</div>
    </div>

    <h2 class="section-title">State Uzayi</h2>
    <div class="grid-two">
      <div class="panel">{chart_html["xy_overlay"]}</div>
      <div class="panel">{chart_html["altitude_action"]}</div>
    </div>
    <div class="grid-two">
      <div class="panel">{chart_html["xy_scatter"]}</div>
      <div class="panel">{chart_html["velocity_scatter"]}</div>
    </div>

    <h2 class="section-title">Ozellik Dagilimlari</h2>
    <div class="grid-two">
      <div class="panel">{chart_html["angle_box"]}</div>
      <div class="panel">{chart_html["angular_vel_box"]}</div>
    </div>

    <h2 class="section-title">Temas ve Korelasyon</h2>
    <div class="grid-one">
      <div class="panel">{chart_html["corr_heatmap"]}</div>
    </div>
    <div class="grid-two">
      <div class="panel">{chart_html["angle_action"]}</div>
      <div class="panel">{chart_html["speed_hist"]}</div>
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output HTML file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataframe(load_dataset())
    figures = {
        "action_counts": make_action_count_chart(df),
        "action_share": make_action_share_chart(df),
        "xy_overlay": make_xy_overlay(df),
        "altitude_action": make_action_by_altitude_chart(df),
        "xy_scatter": make_xy_scatter(df),
        "velocity_scatter": make_velocity_scatter(df),
        "angle_box": make_feature_boxplot(df, "angle", "Action Bazinda Angle Dagilimi", "Angle"),
        "angular_vel_box": make_feature_boxplot(
            df, "angular_vel", "Action Bazinda Angular velocity Dagilimi", "Angular velocity"
        ),
        "leg_contact": make_leg_contact_chart(df),
        "feature_profile": make_action_feature_profile(df),
        "corr_heatmap": make_correlation_heatmap(df),
        "angle_action": make_action_by_angle_chart(df),
        "speed_hist": make_speed_histogram(df),
    }
    write_dashboard(df, figures, args.output)
    print(f"{len(df):,} ornek yuklendi: {HF_DATASET_ID}")
    print(f"Dashboard olusturuldu: {args.output}")


if __name__ == "__main__":
    main()
