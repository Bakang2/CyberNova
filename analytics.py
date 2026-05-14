import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Branding palette
PRIMARY    = "#1B3A6B"   # deep navy
ACCENT     = "#00B4D8"   # electric cyan
HIGHLIGHT  = "#F77F00"   # vibrant amber
LIGHT_GREY = "#F4F6F9"
MID_GREY   = "#9BA4B5"
WHITE      = "#FFFFFF"

def section(title):
    bar = "─" * 65
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")

# PHASE 1 — IMPORT, CLEAN & VALIDATE
section("PHASE 1 — Data Import & Cleaning")

df_raw = pd.read_csv("data/cybernova_iis_logs.csv", parse_dates=["datetime"])
print(f"  Rows loaded          : {len(df_raw):,}")
print(f"  Columns              : {list(df_raw.columns)}")

# Cleaning steps
df = df_raw.copy()

# 1. Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"  Duplicates removed   : {before - len(df)}")

# 2. Drop rows with null critical fields
critical = ["datetime", "cs_uri_stem", "country_code", "sc_status"]
df = df.dropna(subset=critical)

# 3. Filter only valid HTTP methods
df = df[df["cs_method"].isin(["GET", "POST", "PUT", "DELETE"])]

# 4. Filter valid status codes
df = df[df["sc_status"].between(100, 599)]

# 5. Parse extra time features
df["date"]    = pd.to_datetime(df["date"])
df["month"]   = df["date"].dt.to_period("M").astype(str)
df["week"]    = df["date"].dt.to_period("W").astype(str)
df["hour"]    = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
df["weekday"] = df["date"].dt.day_name()
df["is_demo"] = df["cs_uri_stem"] == "/schedule-demo"
df["is_ai"]   = df["cs_uri_stem"] == "/ai-assistant"

print(f"  Clean rows remaining : {len(df):,}")
print(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Unique countries     : {df['country_code'].nunique()}")
print(f"  Unique services      : {df['service'].nunique()}")

# PHASE 2 — SUMMARY STATISTICS
section("PHASE 2 — Summary Statistics")

total_requests = len(df)
total_demo     = df["is_demo"].sum()
total_ai       = df["is_ai"].sum()
conversion_rate = (total_demo / total_requests) * 100

print(f"\n  ── Overall Metrics ──────────────────────────────────────")
print(f"  Total Requests          : {total_requests:,}")
print(f"  Schedule-Demo Requests  : {total_demo:,}  ({conversion_rate:.1f}% of traffic)")
print(f"  AI-Assistant Requests   : {total_ai:,}")
print(f"  Unique Countries        : {df['country_code'].nunique()}")
print(f"  Mean Bytes / Request    : {df['sc_bytes'].mean():,.0f}")
print(f"  Std Dev Bytes           : {df['sc_bytes'].std():,.0f}")
print(f"  Mean Time Taken (ms)    : {df['time_taken'].mean():.0f}")
print(f"  Std Dev Time Taken (ms) : {df['time_taken'].std():.0f}")

# Status code distribution
print(f"\n  ── HTTP Status Distribution ──────────────────────────────")
status_dist = df.groupby("sc_status").size().reset_index(name="count")
status_dist["pct"] = (status_dist["count"] / total_requests * 100).round(1)
for _, row in status_dist.iterrows():
    print(f"    {int(row.sc_status)}  → {row['count']:>5,} requests  ({row.pct:>5.1f}%)")

# Top countries
print(f"\n  ── Top 8 Countries by Requests ──────────────────────────")
top_countries = (
    df.groupby("country")
      .agg(requests=("cs_uri_stem","count"),
           demos=("is_demo","sum"))
      .sort_values("requests", ascending=False)
      .head(8)
)
top_countries["demo_rate"] = (top_countries["demos"] / top_countries["requests"] * 100).round(1)
print(top_countries.to_string())

# Top services
print(f"\n  ── Service Popularity ───────────────────────────────────")
service_dist = (
    df.groupby("service")
      .size()
      .sort_values(ascending=False)
      .reset_index(name="count")
)
service_dist["pct"] = (service_dist["count"] / total_requests * 100).round(1)
print(service_dist.to_string(index=False))

# Monthly summary
print(f"\n  ── Monthly Request Volume ───────────────────────────────")
monthly = df.groupby("month").size().reset_index(name="requests")
print(monthly.to_string(index=False))

# PHASE 3 — VISUALISATIONS
section("PHASE 3 — Generating Visualisations")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    LIGHT_GREY,
    "figure.facecolor":  WHITE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        WHITE,
    "grid.linewidth":    0.8,
    "axes.labelcolor":   PRIMARY,
    "xtick.color":       PRIMARY,
    "ytick.color":       PRIMARY,
    "text.color":        PRIMARY,
})

def styled_title(ax, title, subtitle=""):
    ax.set_title(title, fontsize=14, fontweight="bold", color=PRIMARY, pad=12)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.01), xycoords="axes fraction",
                    ha="center", fontsize=9, color=MID_GREY)

def watermark(fig):
    fig.text(0.99, 0.01, "CyberNova Solutions Ltd. | Analytics Prototype v1.0",
             ha="right", va="bottom", fontsize=7, color=MID_GREY)


# Chart 1: Request Volume Over Time (monthly)
print("  [1/5] Request volume over time...")
fig, ax = plt.subplots(figsize=(11, 4.5))
monthly_plot = df.groupby("month").size().reset_index(name="requests")
ax.fill_between(monthly_plot["month"], monthly_plot["requests"],
                color=ACCENT, alpha=0.25)
ax.plot(monthly_plot["month"], monthly_plot["requests"],
        color=PRIMARY, linewidth=2.5, marker="o", markersize=7, markerfacecolor=ACCENT)

# Annotate each point
for _, row in monthly_plot.iterrows():
    ax.annotate(f"{row.requests:,}", (row["month"], row["requests"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=PRIMARY, fontweight="bold")

styled_title(ax, "Monthly Request Volume", "Sept 2024 – Mar 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Requests")
ax.set_ylim(0, monthly_plot["requests"].max() * 1.25)
plt.xticks(rotation=30, ha="right")
watermark(fig)
plt.tight_layout()
plt.savefig("outputs/chart1_request_volume_over_time.png", dpi=150)
plt.close()


# Chart 2: Top Countries by Request Volume
print("  [2/5] Top countries bar chart...")
fig, ax = plt.subplots(figsize=(10, 5.5))
top10 = (df.groupby("country").size()
           .sort_values(ascending=True)
           .tail(10))
colors = [ACCENT if i >= 7 else PRIMARY for i in range(len(top10))]
bars = ax.barh(top10.index, top10.values, color=colors, edgecolor=WHITE, linewidth=0.5)

for bar, val in zip(bars, top10.values):
    ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9, color=PRIMARY, fontweight="bold")

styled_title(ax, "Top 10 Countries by Request Volume",
             "Weighted heavily toward Southern Africa")
ax.set_xlabel("Total Requests")
ax.set_xlim(0, top10.max() * 1.18)
legend_elems = [
    mpatches.Patch(color=ACCENT, label="Top 3 Regions"),
    mpatches.Patch(color=PRIMARY, label="Other Regions"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
watermark(fig)
plt.tight_layout()
plt.savefig("outputs/chart2_top_countries.png", dpi=150)
plt.close()


# ── Chart 3: Service Type Distribution (Pie) ─────────────────────────────────
print("  [3/5] Service type pie chart...")
fig, ax = plt.subplots(figsize=(9, 7))
svc = df.groupby("service").size().sort_values(ascending=False)
palette = [PRIMARY, ACCENT, HIGHLIGHT, "#2EC4B6", "#E71D36", "#FF9F1C", "#8338EC", "#3A86FF"]
wedges, texts, autotexts = ax.pie(
    svc.values,
    labels=svc.index,
    autopct="%1.1f%%",
    colors=palette[:len(svc)],
    startangle=140,
    wedgeprops={"edgecolor": WHITE, "linewidth": 1.5},
    pctdistance=0.82,
)
for t in texts:
    t.set_fontsize(10)
    t.set_color(PRIMARY)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_color(WHITE)
    at.set_fontweight("bold")
styled_title(ax, "Service Type Distribution",
             "Share of requests by CyberNova product/service")
watermark(fig)
plt.tight_layout()
plt.savefig("outputs/chart3_service_distribution.png", dpi=150)
plt.close()


# Chart 4: Traffic vs Demo Conversion by Country (Scatterplot)
print("  [4/5] Traffic vs demo-conversion scatterplot...")
country_stats = (
    df.groupby("country")
      .agg(total_requests=("cs_uri_stem", "count"),
           demo_requests=("is_demo", "sum"))
      .reset_index()
)
country_stats["demo_rate"] = country_stats["demo_requests"] / country_stats["total_requests"] * 100
country_stats = country_stats[country_stats["total_requests"] >= 30]  # exclude tiny samples

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    country_stats["total_requests"],
    country_stats["demo_rate"],
    s=country_stats["total_requests"] / 3,
    c=country_stats["demo_rate"],
    cmap="coolwarm",
    edgecolors=PRIMARY, linewidths=0.7, alpha=0.85
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Demo Conversion Rate (%)", color=PRIMARY)

for _, row in country_stats.iterrows():
    ax.annotate(
        row["country"].split()[0],   # first word only to avoid clutter
        (row["total_requests"], row["demo_rate"]),
        textcoords="offset points", xytext=(6, 4),
        fontsize=8, color=PRIMARY
    )

# Quadrant lines
med_req  = country_stats["total_requests"].median()
med_rate = country_stats["demo_rate"].median()
ax.axvline(med_req,  color=MID_GREY, linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(med_rate, color=MID_GREY, linestyle="--", linewidth=1, alpha=0.7)
ax.text(med_req + 5, ax.get_ylim()[1] * 0.97,
        "← Low traffic | High traffic →", fontsize=7.5, color=MID_GREY)
ax.text(ax.get_xlim()[0], med_rate + 0.1,
        "High conversion ↑", fontsize=7.5, color=MID_GREY)

styled_title(ax,
    "Traffic Volume vs Demo Conversion Rate by Country",
    "Bubble size = request volume  |  Quadrants identify priority targets")
ax.set_xlabel("Total Requests")
ax.set_ylabel("Schedule-Demo Rate (%)")
watermark(fig)
plt.tight_layout()
plt.savefig("outputs/chart4_traffic_vs_conversion.png", dpi=150)
plt.close()


# Chart 5: Hourly Heatmap (Hour × Weekday)
print("  [5/5] Hourly traffic heatmap...")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heat = (
    df.groupby(["weekday", "hour"])
      .size()
      .unstack(fill_value=0)
)
heat = heat.reindex([d for d in day_order if d in heat.index])

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Number of Requests")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, fontsize=9)
styled_title(ax, "Hourly Traffic Heatmap by Day of Week",
             "Identifies peak engagement windows for CyberNova's marketing team")
ax.set_xlabel("Hour of Day (UTC)")
watermark(fig)
plt.tight_layout()
plt.savefig("outputs/chart5_hourly_heatmap.png", dpi=150)
plt.close()

print("  All 5 charts saved to outputs/")


# PHASE 4 — AUTOMATED REPORT
section("PHASE 4 — Generating Analytics Report")

top_country    = top_countries.index[0]
top_service    = service_dist.iloc[0]["service"]
top_svc_pct    = service_dist.iloc[0]["pct"]
best_conversion = country_stats.sort_values("demo_rate", ascending=False).iloc[0]
high_traffic_low_demo = country_stats[
    (country_stats["total_requests"] > country_stats["total_requests"].median()) &
    (country_stats["demo_rate"] < country_stats["demo_rate"].median())
]

report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         CyberNova Solutions Ltd. — Web Analytics Executive Summary           ║
║                    Reporting Period: Sept 2024 – Mar 2025                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Generated by: CyberNova IIS Analytics System v1.0
Prepared by : Bakang Unashe Kaisara | Business Intelligence & Data Analytics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   During the 7-month period, CyberNova's web properties received a total of
   {total_requests:,} validated requests from {df['country'].nunique()} countries, with a
   mean response payload of {df['sc_bytes'].mean():,.0f} bytes (σ = {df['sc_bytes'].std():,.0f} bytes)
   and a mean server response time of {df['time_taken'].mean():.0f} ms (σ = {df['time_taken'].std():.0f} ms).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. HIGH-PERFORMING REGIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   The leading market is {top_country}, which generated the highest absolute
   request volume during the period. The full ranking is:

"""
for i, (country, row) in enumerate(top_countries.iterrows(), 1):
    report += f"   {i:2}. {country:<20}  {int(row.requests):>5,} requests  |  Demo rate: {row.demo_rate:.1f}%\n"

report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. MOST-REQUESTED SERVICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   The "{top_service}" service was the most accessed, accounting for {top_svc_pct:.1f}%
   of all traffic. The Schedule-Demo feature specifically attracted {total_demo:,}
   requests ({conversion_rate:.1f}% of overall traffic), indicating strong pipeline
   interest.

   Full service breakdown:
"""
for _, row in service_dist.iterrows():
    bar_len = int(row.pct / 1.5)
    report += f"   {'█' * bar_len:<20}  {row.pct:>5.1f}%  {row['service']}\n"

report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. HIGH TRAFFIC, LOW CONVERSION (PRIORITY TARGETS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   The following countries have above-median traffic but below-median demo
   conversion rates — representing the highest-opportunity markets for the
   sales team to target:

"""
if len(high_traffic_low_demo) > 0:
    for _, row in high_traffic_low_demo.iterrows():
        report += (f"   • {row['country']:<22}  {int(row.total_requests):>5,} requests  |"
                   f"  {row.demo_rate:.1f}% demo rate  ← OPPORTUNITY\n")
else:
    report += "   No clear high-traffic/low-conversion markets identified.\n"

report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. BEST CONVERSION MARKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   {best_conversion['country']} achieved the highest demo conversion rate at
   {best_conversion['demo_rate']:.1f}% from {int(best_conversion['total_requests']):,} requests.
   This suggests strong product-market fit in this region.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   1. Focus marketing spend on high-traffic, low-conversion countries identified
      in Section 4 — these represent untapped pipeline potential.
   2. Replicate the engagement strategy used in {best_conversion['country']} across
      lower-performing Southern African markets.
   3. Invest further in the AI Assistant and Schedule-Demo features, as these
      attract the highest combined traffic share.
   4. Schedule marketing campaigns during peak traffic hours (see heatmap chart)
      to maximise impressions and conversion opportunities.
   5. Monitor monthly trend data for seasonal demand shifts and adjust campaign
      timing accordingly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Visualisations: outputs/chart1_request_volume_over_time.png
                   outputs/chart2_top_countries.png
                   outputs/chart3_service_distribution.png
                   outputs/chart4_traffic_vs_conversion.png
                   outputs/chart5_hourly_heatmap.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(report)

with open("outputs/cybernova_analytics_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("  Report saved → outputs/cybernova_analytics_report.txt")

import json
dashboard_data = {
    "metrics": {
        "total_requests": int(total_requests),
        "demo_conversion": float(conversion_rate),
        "unique_countries": int(df['country_code'].nunique()),
        "avg_time": float(df['time_taken'].mean())
    },
    "monthly_volume": {
        "labels": monthly_plot["month"].tolist(),
        "values": monthly_plot["requests"].tolist()
    },
    "top_countries": {
        "labels": top10.index.tolist(),
        "values": top10.tolist()
    },
    "service_distribution": {
        "labels": svc.index.tolist(),
        "values": svc.tolist()
    },
    "traffic_vs_conversion": {
        "countries": country_stats["country"].tolist(),
        "requests": country_stats["total_requests"].tolist(),
        "demo_rate": country_stats["demo_rate"].tolist()
    },
    "hourly_heatmap": {
        "days": list(heat.index),
        "hours": list(range(24)),
        "data": heat.values.tolist()
    }
}

with open("outputs/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(dashboard_data, f)
print("  JSON data saved → outputs/dashboard_data.json")

section("✓ Analytics Complete — all outputs in outputs/")
