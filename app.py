import dash
import dash_auth
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os

# 1. Load and Clean Data
df_raw = pd.read_csv("data/cybernova_iis_logs.csv", parse_dates=["datetime"])
critical = ["datetime", "cs_uri_stem", "country_code", "sc_status"]
df = df_raw.dropna(subset=critical).copy()
df = df[df["cs_method"].isin(["GET", "POST", "PUT", "DELETE"])]
df = df[df["sc_status"].between(100, 599)]

df["month"]   = df["datetime"].dt.to_period("M").astype(str)
df["week"]    = df["datetime"].dt.to_period("W").astype(str)
df["hour"]    = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.day_name()
df["is_demo"] = df["cs_uri_stem"] == "/schedule-demo"
df["is_ai"]   = df["cs_uri_stem"] == "/ai-assistant"
df["is_job"]  = df["cs_uri_stem"].isin(["/ai-assistant", "/prototype"])

df["service"] = df["cs_uri_stem"].map({
    "/": "Home", "/schedule-demo": "Schedule Demo", "/ai-assistant": "AI Assistant",
    "/prototype": "Prototype", "/events": "Events", "/pricing": "Pricing",
    "/about": "About", "/contact": "Contact"
}).fillna(df["cs_uri_stem"])


months    = sorted(df["month"].unique())
countries = sorted(df["country"].dropna().unique())
services  = sorted(df["service"].dropna().unique())

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# 3. App Setup
VALID_USERS = {"admin": "CyberNova26"}
app = dash.Dash(__name__, title="CyberNova Analytics", suppress_callback_exceptions=True)
server = app.server
auth = dash_auth.BasicAuth(app, VALID_USERS)
template     = "simple_white"
chart_colors = ["#00B4D8","#1B3A6B","#F77F00","#48CAE4","#023E8A","#0096C7","#90E0EF","#FFB703",
                "#E71D36","#2EC4B6","#8338EC","#3A86FF","#FF006E","#FFBE0B","#FB5607","#06D6A0"]

def apply_layout(fig):
    fig.update_layout(
        template=template,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Inter, sans-serif", size=14, color="#000000"),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Inter")
    )
    return fig

PAGES = [
    ("/",           "fa-house",         "Overview"),
    ("/volume",     "fa-chart-line",    "Traffic & Volume"),
    ("/geographic", "fa-earth-africa",  "Geographic"),
    ("/services",   "fa-layer-group",   "Services"),
    ("/conversions","fa-bullseye",      "Conversions"),
    ("/heatmap",    "fa-fire",          "Hourly Heatmap"),
    ("/jobs",       "fa-briefcase",     "Jobs Placed"),
    ("/forecast",   "fa-wand-sparkles", "Predictive Forecast"),
    ("/logs",       "fa-table-list",    "Log Explorer"),
]

# 4. Layout
app.layout = html.Div(id="app-wrapper", children=[
    dcc.Location(id="url", refresh=False),

    html.Div(className="sidebar", children=[
        html.Div(className="brand", children=[
            html.Img(src="/assets/logo.png", className="brand-logo"),
            html.Div("CyberNova", className="brand-name")
        ]),
        html.Div(className="nav-menu", children=[
            dcc.Link([html.I(className=f"fa-solid {icon}"), label],
                     href=path, id=f"link-{path.strip('/') or 'overview'}", className="nav-link")
            for path, icon, label in PAGES
        ]),
        html.Div(className="filters-area", children=[
            html.Div("Global Filters", className="filter-label",
                     style={"textTransform":"uppercase","marginBottom":"16px","letterSpacing":"1px","fontSize":"11px"}),
            *[html.Div(className="filter-group", children=[
                html.Div(label, className="filter-label"),
                dcc.Dropdown(id=fid, options=opts, value="ALL", clearable=False,
                             style={"color":"#1E293B","fontSize":"13px"})
            ]) for label, fid, opts in [
                ("Month",       "filter-month",   [{"label":"All Months","value":"ALL"}]  + [{"label":m,"value":m} for m in months]),
                ("Country",     "filter-country", [{"label":"All Countries","value":"ALL"}]+[{"label":c,"value":c} for c in countries]),
                ("Service",     "filter-service", [{"label":"All Services","value":"ALL"}]+[{"label":s,"value":s} for s in services]),
                ("HTTP Method", "filter-method",  [{"label":"All Methods","value":"ALL"},{"label":"GET","value":"GET"},
                                                   {"label":"POST","value":"POST"},{"label":"PUT","value":"PUT"},{"label":"DELETE","value":"DELETE"}]),
                ("Status Code", "filter-status",  [{"label":"All Statuses","value":"ALL"},{"label":"2xx Success","value":"2xx"},
                                                   {"label":"3xx Redirect","value":"3xx"},{"label":"4xx Client Error","value":"4xx"},
                                                   {"label":"5xx Server Error","value":"5xx"}]),
            ]],
        ]),
        html.Div("© 2026 CyberNova Analytics", className="sidebar-footer")
    ]),

    html.Div(className="main-content", id="page-content")
])

# 5. Main Callback
link_ids = [f"link-{p.strip('/') or 'overview'}" for p, _, _ in PAGES]

@app.callback(
    [Output("page-content", "children")] + [Output(lid, "className") for lid in link_ids],
    [Input("url", "pathname"),
     Input("filter-month",   "value"),
     Input("filter-country", "value"),
     Input("filter-service", "value"),
     Input("filter-method",  "value"),
     Input("filter-status",  "value")]
)
def render_page(pathname, month, country, service, method, status):
    # Filter
    fdf = df.copy()
    if month   != "ALL": fdf = fdf[fdf["month"]      == month]
    if country != "ALL": fdf = fdf[fdf["country"]    == country]
    if service != "ALL": fdf = fdf[fdf["service"]    == service]
    if method  != "ALL": fdf = fdf[fdf["cs_method"]  == method]
    if status  != "ALL":
        code = int(status[0]) * 100
        fdf = fdf[fdf["sc_status"].between(code, code + 99)]

    # Nav classes
    pn = pathname or "/"
    if pn == "": pn = "/"
    classes = {p: ("nav-link nav-link-active" if p == pn else "nav-link") for p, _, _ in PAGES}

    def nav_classes():
        return [classes[p] for p, _, _ in PAGES]

    def page_header(title, subtitle=""):
        return html.Div(className="page-header", children=[
            html.Div(title,    className="page-title"),
            html.Div(subtitle, className="page-subtitle")
        ])

    def chart_card(fig, height="100%"):
        return html.Div(className="chart-container",
                        children=[dcc.Graph(figure=fig, style={"height": height, "width": "100%"})])

    if len(fdf) == 0:
        content = html.Div([page_header("No Data", "Adjust filters — no records match.")])
        return [content] + nav_classes()

    # OVERVIEW
    if pn == "/":
        total   = len(fdf)
        demo_rt = fdf["is_demo"].sum() / total * 100
        jobs    = int(fdf["is_job"].sum())
        avg_t   = fdf["time_taken"].mean()

        def fmt(v, d=0, s=""): return f"{v:,.{d}f}{s}" if pd.notnull(v) else "N/A"

        content = html.Div([
            page_header("Executive Overview", "Key performance indicators based on current filters"),
            html.Div(className="kpi-grid", children=[
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Total Requests",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-server")])]),
                    html.Div(f"{total:,}", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Unique Countries",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-globe")])]),
                    html.Div(f"{fdf['country'].nunique()}", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Demo Conversion",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-bullseye")])]),
                    html.Div(f"{demo_rt:.1f}%", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Jobs Placed",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-briefcase")])]),
                    html.Div(f"{jobs:,}", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Avg Response Time",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-bolt")])]),
                    html.Div(f"{avg_t:.0f} ms", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Success Rate",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-circle-check")])]),
                    html.Div(fmt(fdf["sc_status"].between(200,299).sum()/total*100,1,"%"), className="kpi-value")]),
            ]),
            html.Div(className="stats-row", children=[
                html.Div(className="stats-section", children=[
                    html.Div(className="stats-section-title", children=[html.I(className="fa-solid fa-database"), " Response Payload (Bytes)"]),
                    html.Div(className="stats-metrics", children=[
                        html.Div(className="stat-item", children=[html.Div("Mean",className="stat-label"),   html.Div(fmt(fdf["sc_bytes"].mean()),  className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Std Dev",className="stat-label"),html.Div(fmt(fdf["sc_bytes"].std()),   className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Min",className="stat-label"),    html.Div(fmt(fdf["sc_bytes"].min()),   className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Max",className="stat-label"),    html.Div(fmt(fdf["sc_bytes"].max()),   className="stat-value")]),
                    ])
                ]),
                html.Div(className="stats-section", children=[
                    html.Div(className="stats-section-title", children=[html.I(className="fa-solid fa-stopwatch"), " Response Time (ms)"]),
                    html.Div(className="stats-metrics", children=[
                        html.Div(className="stat-item", children=[html.Div("Mean",className="stat-label"),   html.Div(fmt(fdf["time_taken"].mean()), className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Std Dev",className="stat-label"),html.Div(fmt(fdf["time_taken"].std()),  className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Min",className="stat-label"),    html.Div(fmt(fdf["time_taken"].min()),  className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Max",className="stat-label"),    html.Div(fmt(fdf["time_taken"].max()),  className="stat-value")]),
                    ])
                ]),
                html.Div(className="stats-section", children=[
                    html.Div(className="stats-section-title", children=[html.I(className="fa-solid fa-briefcase"), " Jobs & Engagement"]),
                    html.Div(className="stats-metrics", children=[
                        html.Div(className="stat-item", children=[html.Div("Jobs Placed",className="stat-label"),    html.Div(f"{jobs:,}",className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("AI Requests",className="stat-label"),    html.Div(f"{int(fdf['is_ai'].sum()):,}",className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Demo Requests",className="stat-label"),  html.Div(f"{int(fdf['is_demo'].sum()):,}",className="stat-value")]),
                        html.Div(className="stat-item", children=[html.Div("Error Rate",className="stat-label"),     html.Div(fmt(fdf["sc_status"].between(400,599).sum()/total*100,1,"%"),className="stat-value")]),
                    ])
                ]),
            ]),
            html.Div(className="overview-charts-grid", children=[
                html.Div(className="overview-chart-card", children=[
                    html.Div("Monthly Volume Overview", className="overview-chart-title"),
                    dcc.Graph(figure=apply_layout(go.Figure([go.Scatter(
                        x=fdf.groupby("month").size().index,
                        y=fdf.groupby("month").size().values,
                        mode="lines+markers", line=dict(color="#1B3A6B",width=3), marker=dict(color="#00B4D8",size=8)
                    )]).update_layout(xaxis_title="", yaxis_title="Requests", margin=dict(l=20,r=20,t=10,b=20))),
                    style={"height":"100%","flexGrow":"1"})
                ]),
                html.Div(className="overview-chart-card", children=[
                    html.Div("Service Distribution", className="overview-chart-title"),
                    dcc.Graph(figure=apply_layout(go.Figure([go.Pie(
                        labels=fdf.groupby("service").size().index,
                        values=fdf.groupby("service").size().values,
                        hole=0.7, marker=dict(colors=chart_colors, line=dict(color="#FFFFFF",width=2)), textinfo="none"
                    )]).update_layout(showlegend=True, margin=dict(l=10,r=10,t=10,b=10))),
                    style={"height":"100%","flexGrow":"1"})
                ]),
            ])
        ])

    # TRAFFIC & VOLUME
    elif pn == "/volume":
        monthly = fdf.groupby("month").size().reset_index(name="count")
        weekly  = fdf.groupby("week").size().reset_index(name="count")

        fig_m = go.Figure([go.Scatter(x=monthly["month"], y=monthly["count"],
            mode="lines+markers", line=dict(color="#1B3A6B",width=3),
            marker=dict(color="#00B4D8",size=8,line=dict(color="white",width=2)))])
        apply_layout(fig_m)
        fig_m.update_layout(xaxis_title="Month", yaxis_title="Requests", hovermode="x unified")

        fig_w = go.Figure([go.Bar(x=weekly["week"], y=weekly["count"],
            marker_color="#00B4D8", opacity=0.85)])
        apply_layout(fig_w)
        fig_w.update_layout(xaxis_title="Week", yaxis_title="Requests",
                             xaxis_tickangle=-45)

        content = html.Div([
            page_header("Traffic & Volume", "Monthly and weekly request volume trends"),
            html.Div(style={"display":"flex","flexDirection":"column","gap":"24px","padding":"0 24px 24px"}, children=[
                html.Div(style={"background":"#fff","borderRadius":"16px","border":"1px solid #E2E8F0",
                                "boxShadow":"0 4px 20px rgba(0,0,0,0.05)","padding":"20px"}, children=[
                    html.Div("Monthly Request Volume", style={"fontWeight":"700","fontSize":"16px","marginBottom":"12px","color":"#1B3A6B"}),
                    dcc.Graph(figure=fig_m, style={"height":"300px"})
                ]),
                html.Div(style={"background":"#fff","borderRadius":"16px","border":"1px solid #E2E8F0",
                                "boxShadow":"0 4px 20px rgba(0,0,0,0.05)","padding":"20px"}, children=[
                    html.Div("Weekly Request Volume", style={"fontWeight":"700","fontSize":"16px","marginBottom":"12px","color":"#1B3A6B"}),
                    dcc.Graph(figure=fig_w, style={"height":"300px"})
                ]),
            ])
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # GEOGRAPHIC
    elif pn == "/geographic":
        c_data = fdf.groupby("country").size().nlargest(10).reset_index(name="count")
        fig = go.Figure([go.Bar(y=c_data["country"][::-1], x=c_data["count"][::-1],
            orientation="h", marker_color="#1B3A6B", opacity=0.85,
            text=c_data["count"][::-1], textposition="outside")])
        apply_layout(fig)
        fig.update_layout(xaxis_title="Total Requests", yaxis_title="")
        content = html.Div([
            page_header("Geographic Distribution", "Top 10 countries by total web server requests"),
            chart_card(fig)
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # SERVICES
    elif pn == "/services":
        svc = fdf.groupby("service").size().reset_index(name="count")
        svc["pct"] = (svc["count"] / svc["count"].sum() * 100).round(1)
        fig = go.Figure([go.Pie(labels=svc["service"], values=svc["count"],
            hole=0.55, marker=dict(colors=chart_colors, line=dict(color="#FFFFFF",width=2)),
            textinfo="percent+label", textfont_size=13,
            hovertemplate="<b>%{label}</b><br>Requests: %{value:,}<br>Share: %{percent}<extra></extra>")])
        apply_layout(fig)
        fig.update_layout(showlegend=True)
        content = html.Div([
            page_header("Service Distribution", "Share of traffic by CyberNova product/service page"),
            chart_card(fig)
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # CONVERSIONS
    elif pn == "/conversions":
        cs = fdf.groupby("country").agg(requests=("cs_uri_stem","count"), demos=("is_demo","sum")).reset_index()
        cs["demo_rate"] = cs["demos"] / cs["requests"] * 100
        cs = cs[cs["requests"] >= 10]
        med_r = cs["requests"].median()
        med_d = cs["demo_rate"].median()

        fig = go.Figure()
        if len(cs) > 0:
            fig.add_trace(go.Scatter(
                x=cs["requests"], y=cs["demo_rate"], mode="markers", text=cs["country"],
                marker=dict(size=[min(r/50+10,50) for r in cs["requests"]],
                            color=cs["demo_rate"], colorscale="Blues", showscale=True,
                            colorbar=dict(title="Demo Rate %"),
                            line=dict(width=1,color="#FFFFFF")),
                hovertemplate="<b>%{text}</b><br>Requests: %{x:,}<br>Demo Rate: %{y:.1f}%<extra></extra>"
            ))
            fig.add_vline(x=med_r, line_dash="dash", line_color="#9BA4B5", annotation_text="Median traffic")
            fig.add_hline(y=med_d, line_dash="dash", line_color="#9BA4B5", annotation_text="Median conversion")
        apply_layout(fig)
        fig.update_layout(xaxis_title="Total Requests", yaxis_title="Schedule-Demo Conversion Rate (%)")
        content = html.Div([
            page_header("Traffic vs Conversions", "Country traffic volume vs schedule-demo conversion rate — bubble size = request volume"),
            chart_card(fig)
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # HOURLY HEATMAP
    elif pn == "/heatmap":
        heat = fdf.groupby(["weekday","hour"]).size().unstack(fill_value=0)
        heat = heat.reindex([d for d in DAY_ORDER if d in heat.index])
        hours = [f"{h:02d}:00" for h in range(24)]

        fig = go.Figure(go.Heatmap(
            z=heat.values,
            x=hours,
            y=list(heat.index),
            colorscale="YlOrRd",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Requests: %{z:,}<extra></extra>",
            colorbar=dict(title="Requests")
        ))
        apply_layout(fig)
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            xaxis=dict(tickangle=-45)
        )
        content = html.Div([
            page_header("Hourly Traffic Heatmap", "Request volume by day of week and hour — identifies peak engagement windows"),
            chart_card(fig, height="500px")
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # JOBS PLACED
    elif pn == "/jobs":
        jobs_df = fdf[fdf["is_job"]].groupby(["service","month"]).size().reset_index(name="count")
        total_jobs = int(fdf["is_job"].sum())
        ai_jobs    = int(fdf["is_ai"].sum())
        proto_jobs = int((fdf["cs_uri_stem"] == "/prototype").sum())

        fig = go.Figure()
        for svc, color in [("AI Assistant","#00B4D8"),("Prototype","#1B3A6B")]:
            sub = jobs_df[jobs_df["service"] == svc]
            fig.add_trace(go.Bar(x=sub["month"], y=sub["count"], name=svc,
                                 marker_color=color, opacity=0.85))
        apply_layout(fig)
        fig.update_layout(barmode="group", xaxis_title="Month",
                          yaxis_title="Jobs Placed", hovermode="x unified")

        country_jobs = fdf[fdf["is_job"]].groupby("country").size().nlargest(10).reset_index(name="jobs")
        fig2 = go.Figure([go.Bar(y=country_jobs["country"][::-1], x=country_jobs["jobs"][::-1],
            orientation="h", marker_color="#F77F00", opacity=0.85,
            text=country_jobs["jobs"][::-1], textposition="outside")])
        apply_layout(fig2)
        fig2.update_layout(xaxis_title="Jobs Placed", yaxis_title="")

        content = html.Div([
            page_header("Jobs Placed Analysis", "Number of jobs placed for AI Assistant and Prototyping services"),
            html.Div(className="kpi-grid", style={"padding":"0 24px 16px"}, children=[
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Total Jobs Placed",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-briefcase")])]),
                    html.Div(f"{total_jobs:,}", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("AI Assistant Jobs",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-robot")])]),
                    html.Div(f"{ai_jobs:,}", className="kpi-value")]),
                html.Div(className="kpi-card", children=[
                    html.Div(className="kpi-card-header", children=[html.Div("Prototype Jobs",className="kpi-title"), html.Div(className="kpi-icon",children=[html.I(className="fa-solid fa-code")])]),
                    html.Div(f"{proto_jobs:,}", className="kpi-value")]),
            ]),
            html.Div(style={"display":"flex","gap":"24px","padding":"0 24px 24px"}, children=[
                html.Div(style={"flex":"2","background":"#fff","borderRadius":"16px","border":"1px solid #E2E8F0",
                                "boxShadow":"0 4px 20px rgba(0,0,0,0.05)","padding":"20px"}, children=[
                    html.Div("Monthly Jobs by Service", style={"fontWeight":"700","fontSize":"16px","marginBottom":"12px","color":"#1B3A6B"}),
                    dcc.Graph(figure=fig, style={"height":"350px"})
                ]),
                html.Div(style={"flex":"1","background":"#fff","borderRadius":"16px","border":"1px solid #E2E8F0",
                                "boxShadow":"0 4px 20px rgba(0,0,0,0.05)","padding":"20px"}, children=[
                    html.Div("Top Countries by Jobs", style={"fontWeight":"700","fontSize":"16px","marginBottom":"12px","color":"#1B3A6B"}),
                    dcc.Graph(figure=fig2, style={"height":"350px"})
                ]),
            ])
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # PREDICTIVE FORECAST
    elif pn == "/forecast":
        demo_df = fdf[fdf["is_demo"]].copy()
        demo_df["date_only"] = demo_df["datetime"].dt.date
        daily = demo_df.groupby("date_only").size().reset_index(name="count")
        daily["date_only"] = pd.to_datetime(daily["date_only"])

        fig = go.Figure()
        if not daily.empty:
            x = np.arange(len(daily))
            y = daily["count"].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            daily["trend"] = p(x)
            last = daily["date_only"].max()
            f_dates = [last + pd.Timedelta(days=i) for i in range(1,31)]
            f_vals  = p(np.arange(len(daily), len(daily)+30))

            fig.add_trace(go.Scatter(x=daily["date_only"], y=daily["count"],
                name="Historical", mode="lines", line=dict(color="#00B4D8",width=2)))
            fig.add_trace(go.Scatter(x=daily["date_only"], y=daily["trend"],
                name="Trend", mode="lines", line=dict(color="#1B3A6B",dash="dash")))
            fig.add_trace(go.Scatter(x=f_dates, y=f_vals,
                name="30-Day Forecast", mode="lines",
                line=dict(color="#F77F00",width=3)))
        apply_layout(fig)
        fig.update_layout(xaxis_title="Date", yaxis_title="Demo Bookings", hovermode="x unified")
        content = html.Div([
            page_header("Predictive Forecast", "30-day linear forecast of Schedule-Demo demand"),
            chart_card(fig)
        ], style={"height":"100%","display":"flex","flexDirection":"column"})

    # LOG EXPLORER
    elif pn == "/logs":
        disp = fdf[["date","time","c_ip","cs_method","service","sc_status","time_taken","country"]].reset_index(drop=True)
        total_rows  = len(disp)
        page_size   = 20
        total_pages = max(1,(total_rows+page_size-1)//page_size)
        content = html.Div([
            page_header("Log Explorer", f"Raw IIS web server logs — {total_rows:,} records matching current filters"),
            dcc.Store(id="log-store", data=disp.to_dict("records")),
            dcc.Store(id="log-page",  data=0),
            html.Div(style={"background":"#fff","borderRadius":"16px","border":"1px solid #E2E8F0",
                            "boxShadow":"0 4px 20px rgba(0,0,0,0.05)","overflow":"hidden","margin":"0 24px 24px"}, children=[
                html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                                "padding":"16px 24px","borderBottom":"1px solid #E2E8F0","backgroundColor":"#f8fafc"}, children=[
                    html.Div(id="log-page-info", style={"fontSize":"14px","color":"#64748B","fontWeight":"500"}),
                    html.Div(style={"display":"flex","gap":"10px"}, children=[
                        html.Button([html.I(className="fa-solid fa-chevron-left",style={"marginRight":"6px"}),"Previous"],
                            id="log-prev", n_clicks=0,
                            style={"background":"#0B192C","color":"white","border":"none","padding":"10px 20px",
                                   "borderRadius":"8px","cursor":"pointer","fontFamily":"Inter, sans-serif","fontSize":"14px","fontWeight":"600"}),
                        html.Button(["Next",html.I(className="fa-solid fa-chevron-right",style={"marginLeft":"6px"})],
                            id="log-next", n_clicks=0,
                            style={"background":"#00B4D8","color":"white","border":"none","padding":"10px 20px",
                                   "borderRadius":"8px","cursor":"pointer","fontFamily":"Inter, sans-serif","fontSize":"14px","fontWeight":"600"}),
                    ])
                ]),
                html.Div(id="log-table-wrapper", style={"overflowX":"auto"})
            ])
        ])
    else:
        content = html.Div([page_header("Page Not Found", "Use the sidebar to navigate.")])

    return [content] + nav_classes()


# 6. Log Pagination Callback
PAGE_SIZE = 20

@app.callback(
    [Output("log-page","data"),
     Output("log-table-wrapper","children"),
     Output("log-page-info","children")],
    [Input("log-prev","n_clicks"), Input("log-next","n_clicks")],
    [State("log-page","data"),    State("log-store","data")],
    prevent_initial_call=False
)
def paginate_logs(prev_clicks, next_clicks, current_page, stored_data):
    if stored_data is None:
        return 0, html.Div("No data"), ""
    total_rows  = len(stored_data)
    total_pages = max(1,(total_rows+PAGE_SIZE-1)//PAGE_SIZE)
    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"]
        if trigger == "log-next.n_clicks":
            current_page = min(current_page+1, total_pages-1)
        elif trigger == "log-prev.n_clicks":
            current_page = max(current_page-1, 0)
    start     = current_page * PAGE_SIZE
    end       = min(start+PAGE_SIZE, total_rows)
    page_data = stored_data[start:end]
    columns   = list(page_data[0].keys()) if page_data else []
    table = dash_table.DataTable(
        data=page_data,
        columns=[{"name":c.replace("_"," ").title(),"id":c} for c in columns],
        page_action="none",
        style_table={"overflowX":"auto"},
        style_header={"backgroundColor":"#f8fafc","fontWeight":"700","borderBottom":"2px solid #E2E8F0",
                      "textAlign":"left","padding":"14px 16px","fontFamily":"Inter, sans-serif",
                      "fontSize":"12px","textTransform":"uppercase","letterSpacing":"0.5px","color":"#64748B"},
        style_cell={"textAlign":"left","padding":"13px 16px","fontFamily":"Inter, sans-serif",
                    "fontSize":"13px","borderBottom":"1px solid #f1f5f9","color":"#1E293B"},
        style_data_conditional=[
            {"if":{"row_index":"odd"},"backgroundColor":"#f8fafc"},
            {"if":{"filter_query":"{sc_status} >= 200 && {sc_status} < 300"},"color":"#16a34a","fontWeight":"600"},
            {"if":{"filter_query":"{sc_status} >= 300 && {sc_status} < 400"},"color":"#0284c7"},
            {"if":{"filter_query":"{sc_status} >= 400 && {sc_status} < 500"},"color":"#dc2626","fontWeight":"600"},
            {"if":{"filter_query":"{sc_status} >= 500"},"color":"#9333ea","fontWeight":"600"},
        ]
    )
    page_info = f"Showing rows {start+1}–{end} of {total_rows:,}  |  Page {current_page+1} of {total_pages}"
    return current_page, table, page_info


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
