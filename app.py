import csv
import io
from datetime import datetime, date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
import streamlit as st

st.set_page_config(page_title="IBKR Portfolio Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Dark / Light mode toggle
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default to dark

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode, key="dark_toggle", on_change=toggle_theme)

dark = st.session_state.dark_mode

# Theme colours
if dark:
    BG = "#0E1117"
    TEXT = "#FAFAFA"
    TABLE_HEADER_BORDER = "#555"
    TABLE_ROW_BORDER = "#333"
    PLOT_BG = "#0E1117"
    PLOT_PAPER = "#0E1117"
    PLOT_FONT = "#FAFAFA"
else:
    BG = "#FFFFFF"
    TEXT = "#1A1A1A"
    TABLE_HEADER_BORDER = "#999"
    TABLE_ROW_BORDER = "#DDD"
    PLOT_BG = "#FFFFFF"
    PLOT_PAPER = "#FFFFFF"
    PLOT_FONT = "#1A1A1A"

# Inject global CSS for the chosen theme
SIDEBAR_BG = "#1A1D23" if dark else "#F0F2F6"
st.markdown(f"""
<style>
    /* ---- Main app background & text ---- */
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    .stApp header, .stApp [data-testid="stHeader"] {{
        background-color: {BG};
    }}

    /* ---- Force ALL text elements to theme colour ---- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp li, .stApp td, .stApp th,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp [class*="stText"],
    .stApp [class*="stMarkdown"] {{
        color: {TEXT} !important;
    }}

    /* ---- File uploader ---- */
    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stFileUploader"] span,
    .stApp [data-testid="stFileUploader"] p,
    .stApp [data-testid="stFileUploader"] div,
    .stApp [data-testid="stFileUploader"] small {{
        color: {TEXT} !important;
    }}
    /* Dropzone — force dark bg on every possible element */
    .stApp [data-testid="stFileUploader"] section,
    .stApp [data-testid="stFileUploader"] section > *,
    .stApp [data-testid="stFileUploader"] section div,
    .stApp [data-testid="stFileUploader"] section span,
    .stApp [data-testid="stFileUploadDropzone"],
    .stApp [data-testid="stFileUploadDropzone"] > *,
    .stApp [data-testid="stFileUploadDropzone"] div {{
        background-color: {"#1A1D23" if dark else "#FAFAFA"} !important;
        color: {"#A0A4B0" if dark else "#666"} !important;
    }}
    .stApp [data-testid="stFileUploader"] section {{
        border: 1px dashed {"#4A4F5C" if dark else "#CCC"} !important;
        border-radius: 0.5rem !important;
    }}
    .stApp [data-testid="stFileUploadDropzone"] svg {{
        fill: {"#A0A4B0" if dark else "#666"} !important;
    }}
    .stApp [data-testid="stFileUploadDropzone"] button,
    .stApp [data-testid="stBaseButton-secondary"] {{
        background-color: {"#2A2D35" if dark else "#FFFFFF"} !important;
        color: {TEXT} !important;
        border-color: {"#4A4F5C" if dark else "#CCC"} !important;
    }}

    /* ---- Info / Warning / Error alert boxes ---- */
    .stApp [data-testid="stNotification"],
    .stApp .stAlert,
    .stApp [data-testid="stAlert"] {{
        background-color: {"#1A2332" if dark else "#EFF6FF"} !important;
        border-color: {"#2A4A6B" if dark else "#B3D4FC"} !important;
        color: {TEXT} !important;
    }}
    .stApp [data-testid="stNotification"] p,
    .stApp [data-testid="stNotification"] span,
    .stApp .stAlert p, .stApp .stAlert span,
    .stApp [data-testid="stAlert"] p,
    .stApp [data-testid="stAlert"] span {{
        color: {TEXT} !important;
    }}

    /* ---- Sidebar ---- */
    .stSidebar, .stSidebar [data-testid="stSidebarContent"] {{
        background-color: {SIDEBAR_BG};
        color: {TEXT};
    }}
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar p, .stSidebar span, .stSidebar label, .stSidebar div {{
        color: {TEXT} !important;
    }}
    .stSidebar [data-testid="stWidgetLabel"] p,
    .stSidebar [data-testid="stWidgetLabel"] span {{
        color: {TEXT} !important;
    }}

    /* ---- Checkbox labels in sidebar ---- */
    .stSidebar .stCheckbox label span {{
        color: {TEXT} !important;
    }}

    /* ---- Tab separator styling ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {"#444" if dark else "#CCC"} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 8px 20px;
        border: 1px solid {"#444" if dark else "#CCC"} !important;
        border-bottom: none !important;
        border-radius: 6px 6px 0 0;
        background-color: {"#1A1D23" if dark else "#F0F2F6"} !important;
        color: {TEXT} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BG} !important;
        border-bottom: 2px solid {BG} !important;
        margin-bottom: -2px;
    }}

    /* ---- Expander (details/summary) dark mode fix ---- */
    details, details summary, details div,
    [data-testid="stExpander"], [data-testid="stExpander"] *,
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("IBKR Portfolio Analyzer")


# ---------------------------------------------------------------------------
# Colour palette — smooth gradient from blue through purple, pink, red,
# orange, yellow, to green (matching the reference chart style)
# ---------------------------------------------------------------------------
PIE_COLORS = [
    "#7AB8F5",  # light blue
    "#A78BDB",  # lavender / purple
    "#C87EC5",  # mauve / pink-purple
    "#D46BA8",  # rose
    "#DC607E",  # pink-red
    "#D4605A",  # salmon
    "#D47B4A",  # burnt orange
    "#E0A035",  # amber / orange
    "#E8C840",  # gold / yellow
    "#C8D44A",  # yellow-green
    "#8BC34A",  # lime green
    "#5DAF5D",  # green
    "#3D9970",  # teal-green
    "#2E8B8B",  # teal
    "#4A90B8",  # steel blue
    "#6A7FCC",  # periwinkle
    "#8877CC",  # violet
    "#AA66BB",  # orchid
    "#CC6699",  # hot pink
    "#B85050",  # brick red
]


# ---------------------------------------------------------------------------
# CSV Parser — extracts a specific section from the IBKR PortfolioAnalyst CSV
# ---------------------------------------------------------------------------
def parse_section(raw_text: str, section_name: str) -> pd.DataFrame:
    """Parse a named section from the IBKR multi-section CSV.

    Each row in the CSV starts with:  SectionName, RowType, ...fields
    RowType is one of: MetaInfo, Header, Data, Total

    Returns a DataFrame with proper column names built from the Header row,
    containing only the Data rows (Total rows excluded).
    """
    columns = None
    rows = []

    reader = csv.reader(io.StringIO(raw_text))
    for parts in reader:
        if len(parts) < 3:
            continue
        row_section = parts[0].strip()
        row_type = parts[1].strip()

        if row_section != section_name:
            continue

        # The actual fields start at index 2
        fields = parts[2:]

        if row_type == "Header" and columns is None:
            columns = [f.strip() for f in fields]
        elif row_type == "Data" and columns is not None:
            # Skip subtotal rows (Date column == "Total")
            if fields and fields[0].strip() == "Total":
                continue
            # Pad or trim fields to match column count
            cleaned = [f.strip() for f in fields]
            if len(cleaned) < len(columns):
                cleaned += [""] * (len(columns) - len(cleaned))
            elif len(cleaned) > len(columns):
                cleaned = cleaned[:len(columns)]
            rows.append(cleaned)

    if columns is None or len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=columns)
    return df


# ---------------------------------------------------------------------------
# Open Position Summary analysis
# ---------------------------------------------------------------------------
def parse_concentration_holdings(raw_text: str) -> pd.DataFrame:
    """Parse Concentration > Holdings top-level rows from the IBKR CSV.

    Uses csv.reader to correctly handle commas inside quoted fields.
    Filters: section=Concentration, row_type=Data, subsection=Holdings.
    Excludes indented sub-breakdown rows (symbol starts with space) and Total.
    """
    columns = None
    rows = []

    reader = csv.reader(io.StringIO(raw_text))
    for parts in reader:
        if len(parts) < 3:
            continue
        if parts[0].strip() != "Concentration":
            continue

        row_type = parts[1].strip()
        fields = parts[2:]

        # Grab the first Header row for the Holdings subsection
        if row_type == "Header" and columns is None:
            columns = [f.strip() for f in fields]
            continue

        if row_type != "Data" or columns is None:
            continue

        # SubSection must be "Holdings"
        if fields[0].strip() != "Holdings":
            continue

        # Symbol is field[1] — exclude indented rows and "Total"
        raw_symbol = fields[1]
        if raw_symbol.startswith(" ") or raw_symbol.strip() == "Total":
            continue

        # Pad or trim to match column count
        cleaned = [f.strip() for f in fields]
        if len(cleaned) < len(columns):
            cleaned.extend([""] * (len(columns) - len(cleaned)))
        elif len(cleaned) > len(columns):
            cleaned = cleaned[:len(columns)]

        rows.append(cleaned)

    if columns is None or len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns
    for col in ["LongValue", "ShortValue", "NetValue",
                "LongParsedWeight", "ShortParsedWeight", "NetParsedWeight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the Open Position Summary data."""
    # Convert numeric columns
    numeric_cols = ["Quantity", "ClosePrice", "Value", "Cost Basis",
                    "UnrealizedP&L", "FXRateToBase"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute value in base currency (SGD)
    df["ValueInBase"] = df["Value"] * df["FXRateToBase"]

    # Friendly label for the chart
    df["Label"] = df["Symbol"] + " (" + df["Description"] + ")"

    return df


# ---------------------------------------------------------------------------
# Main app flow
# ---------------------------------------------------------------------------
st.markdown("""
**How to download your CSV from IBKR:**
1. Log in to your **IBKR** account
2. Click **Performance & Reports** in the top navigation
3. Click the **Reports** tab in PortfolioAnalyst
4. Find the **Since Inception** row under Default Reports
5. Click the **CSV icon** (rightmost icon) to download
""")

uploaded_file = st.file_uploader(
    "Upload your IBKR PortfolioAnalyst CSV", type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload an IBKR PortfolioAnalyst CSV file to get started.")
    st.stop()

# Read the raw text (handle encoding)
raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

# --- Parse data for both sections before tabs ------------------------------
df_positions = parse_section(raw_text, "Open Position Summary")

if df_positions.empty:
    st.warning("No 'Open Position Summary' data found in the uploaded file.")
    st.stop()

df_positions = prepare_open_positions(df_positions)

# Sidebar — filter by FinancialInstrument using checkboxes
st.sidebar.header("Financial Instruments")
instruments = sorted(df_positions["FinancialInstrument"].unique().tolist())
selected = []
for inst in instruments:
    if st.sidebar.checkbox(inst, value=True, key=f"inst_{inst}"):
        selected.append(inst)

if not selected:
    st.warning("Select at least one Financial Instrument from the sidebar.")
    st.stop()

df_filtered = df_positions[df_positions["FinancialInstrument"].isin(selected)]
df_filtered = df_filtered.sort_values("ValueInBase", ascending=False).reset_index(drop=True)

df_conc = parse_concentration_holdings(raw_text)
df_alloc = parse_section(raw_text, "Allocation by Financial Instrument")
df_cashflows = parse_section(raw_text, "Deposits And Withdrawals")
df_nav = parse_section(raw_text, "Allocation by Asset Class")

# Extract the actual report end date from Key Statistics → Analysis Period
report_end_date = None
for line in raw_text.splitlines():
    parts = line.split(",")
    if len(parts) >= 4 and parts[0].strip() == "Key Statistics" and parts[1].strip() == "MetaInfo":
        # Format: "September 14, 2017 - February 13, 2026"
        period_str = ",".join(parts[2:]).strip().strip('"')
        if " - " in period_str:
            end_str = period_str.split(" - ")[1].strip()
            try:
                report_end_date = pd.to_datetime(end_str, format="%B %d, %Y")
            except Exception:
                report_end_date = pd.to_datetime(end_str)
        break

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Open Position Summary", "Concentration", "Allocation", "Performance", "Return"])

# === Tab 1: Open Position Summary ==========================================
with tab1:
    # Assign colours from the palette in rank order (largest slice gets first colour)
    n = len(df_filtered)
    colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(n)]

    fig = go.Figure(data=[go.Pie(
        labels=df_filtered["Symbol"],
        values=df_filtered["ValueInBase"],
        marker=dict(colors=colors),
        hole=0.4,
        textposition="inside",
        textinfo="none",
        texttemplate="%{label}<br>%{percent:.1%}",
        textfont=dict(color="white", size=12),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>",
        customdata=df_filtered[["Description"]].values,
        sort=False,
        direction="clockwise",
        rotation=0,
    )])
    fig.update_layout(
        title=dict(text="Portfolio Allocation (Base Currency)", font=dict(color=PLOT_FONT, size=20)),
        legend=dict(
            title=dict(text="Position", font=dict(color=PLOT_FONT, size=14)),
            font=dict(color=PLOT_FONT, size=13),
        ),
        height=800,
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font_color=PLOT_FONT,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table with formatted columns
    st.subheader("Position Details")

    df_display = df_filtered.copy()
    df_display["Quantity"] = df_display["Quantity"].apply(lambda x: f"{x:,.0f}")
    df_display["ClosePrice"] = df_display["ClosePrice"].apply(lambda x: f"{x:,.2f}")
    df_display["Value"] = df_display["Value"].apply(lambda x: f"{x:,.0f}")
    df_display["ValueInBase"] = df_display["ValueInBase"].apply(lambda x: f"{x:,.0f}")

    display_cols = ["FinancialInstrument", "Symbol", "Description", "Sector",
                    "Currency", "Quantity", "ClosePrice", "Value",
                    "ValueInBase"]

    num_cols = {"Quantity", "ClosePrice", "Value", "ValueInBase"}

    header_html = "".join(f"<th style='padding:8px 12px; border-bottom:2px solid {TABLE_HEADER_BORDER}; color:{TEXT};'>{c}</th>" for c in display_cols)

    rows_html = ""
    for _, row in df_display[display_cols].iterrows():
        cells = ""
        for col in display_cols:
            align = "right" if col in num_cols else "left"
            cells += f"<td style='padding:6px 12px; text-align:{align}; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{TEXT};'>{row[col]}</td>"
        rows_html += f"<tr>{cells}</tr>"

    table_html = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:14px; background-color:{BG};">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

# === Tab 2: Concentration — Holdings =======================================
with tab2:
    if df_conc.empty:
        st.warning("No 'Concentration' Holdings data found in the uploaded file.")
    else:
        df_conc = df_conc.sort_values("NetValue", ascending=False).reset_index(drop=True)

        TOP_N = 12
        if len(df_conc) > TOP_N:
            df_top = df_conc.head(TOP_N).copy()
            others_value = df_conc.iloc[TOP_N:]["NetValue"].sum()
            others_row = pd.DataFrame([{
                "Symbol": "Others",
                "Description": f"{len(df_conc) - TOP_N} other holdings",
                "Sector": "",
                "NetValue": others_value,
                "LongValue": df_conc.iloc[TOP_N:]["LongValue"].sum(),
                "ShortValue": df_conc.iloc[TOP_N:]["ShortValue"].sum(),
                "NetParsedWeight": df_conc.iloc[TOP_N:]["NetParsedWeight"].sum(),
            }])
            df_conc_chart = pd.concat([df_top, others_row], ignore_index=True)
        else:
            df_conc_chart = df_conc.copy()

        n_conc = len(df_conc_chart)
        colors_conc = []
        for i in range(n_conc):
            if df_conc_chart.iloc[i]["Symbol"] == "Others":
                colors_conc.append("#B0B0B0")
            else:
                colors_conc.append(PIE_COLORS[i % len(PIE_COLORS)])

        fig_conc = go.Figure(data=[go.Pie(
            labels=df_conc_chart["Symbol"],
            values=df_conc_chart["NetValue"],
            marker=dict(colors=colors_conc),
            hole=0.4,
            textposition="inside",
            textinfo="none",
            texttemplate="%{label}<br>%{percent:.1%}",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>Net Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>",
            customdata=df_conc_chart[["Description"]].values,
            sort=False,
            direction="clockwise",
            rotation=0,
        )])
        fig_conc.update_layout(
            title=dict(text="Concentration by Holding (Net Value) — Top 12", font=dict(color=PLOT_FONT, size=20)),
            legend=dict(
                title=dict(text="Holding", font=dict(color=PLOT_FONT, size=14)),
                font=dict(color=PLOT_FONT, size=13),
            ),
            height=800,
            paper_bgcolor=PLOT_PAPER,
            plot_bgcolor=PLOT_BG,
            font_color=PLOT_FONT,
        )

        st.plotly_chart(fig_conc, use_container_width=True)

        # Concentration summary table
        st.subheader("Concentration Details")

        df_conc_display = df_conc_chart.copy()
        df_conc_display["NetValue"] = df_conc_display["NetValue"].apply(lambda x: f"{x:,.0f}")
        df_conc_display["% of Total"] = df_conc_display["NetParsedWeight"].apply(lambda x: f"{x:.2f}%")

        conc_display_cols = ["Symbol", "Description", "Sector", "NetValue", "% of Total"]
        conc_num_cols = {"NetValue", "% of Total"}

        conc_header = "".join(
            f"<th style='padding:8px 12px; border-bottom:2px solid {TABLE_HEADER_BORDER}; color:{TEXT};'>{c}</th>"
            for c in conc_display_cols
        )
        conc_rows = ""
        for _, row in df_conc_display[conc_display_cols].iterrows():
            cells = ""
            for col in conc_display_cols:
                align = "right" if col in conc_num_cols else "left"
                cells += f"<td style='padding:6px 12px; text-align:{align}; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{TEXT};'>{row[col]}</td>"
            conc_rows += f"<tr>{cells}</tr>"

        conc_table = f"""
        <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:14px; background-color:{BG};">
        <thead><tr>{conc_header}</tr></thead>
        <tbody>{conc_rows}</tbody>
        </table>
        </div>
        """
        st.markdown(conc_table, unsafe_allow_html=True)

# === Tab 3: Allocation by Financial Instrument =============================
INSTRUMENT_COLORS = {
    "Stocks":  "#636EFA",   # vivid indigo blue
    "ETFs":    "#00CC96",   # bright emerald
    "Options": "#FFA15A",   # warm amber
    "Cash":    "#AB63FA",   # rich violet
}

with tab3:
    if df_alloc.empty:
        st.warning("No 'Allocation by Financial Instrument' data found in the uploaded file.")
    else:
        # Convert numeric columns (everything except Date)
        for col in df_alloc.columns[1:]:
            df_alloc[col] = pd.to_numeric(df_alloc[col], errors="coerce")

        # Instrument columns (exclude NAV — that's the total)
        instrument_cols = [c for c in df_alloc.columns[1:] if c != "NAV"]

        # Parse Date column into proper datetime for a smooth x-axis
        df_alloc["DateParsed"] = pd.to_datetime(
            df_alloc["Date"].astype(str), format="%Y%m"
        )
        df_alloc["DateLabel"] = df_alloc["DateParsed"].dt.strftime("%Y/%m")

        # --- Vertical stacked bar chart ---
        fig_alloc = go.Figure()

        shown = [inst for inst in instrument_cols if inst in selected]

        for inst in shown:
            color = INSTRUMENT_COLORS.get(inst, PIE_COLORS[instrument_cols.index(inst) % len(PIE_COLORS)])
            fig_alloc.add_trace(go.Bar(
                name=inst,
                x=df_alloc["DateLabel"],
                y=df_alloc[inst],
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"{inst}: %{{y:,.0f}}<extra></extra>",
            ))

        fig_alloc.update_layout(
            barmode="stack",
            title=dict(
                text="Allocation by Financial Instrument",
                font=dict(color=PLOT_FONT, size=20),
            ),
            legend=dict(
                title=dict(text="Instrument", font=dict(color=PLOT_FONT, size=14)),
                font=dict(color=PLOT_FONT, size=13),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            height=900,
            paper_bgcolor=PLOT_PAPER,
            plot_bgcolor=PLOT_BG,
            font_color=PLOT_FONT,
            hovermode="x unified",
            xaxis=dict(
                title="",
                type="category",
                showgrid=False,
                tickfont=dict(color=PLOT_FONT, size=11),
                tickvals=[lbl for lbl in df_alloc["DateLabel"] if lbl.endswith("/01")],
                ticktext=[lbl[:4] for lbl in df_alloc["DateLabel"] if lbl.endswith("/01")],
                showline=True,
                linecolor="#444" if dark else "#CCC",
            ),
            yaxis=dict(
                title="Value (Base Currency)",
                showgrid=True,
                gridcolor="#222" if dark else "#EEE",
                gridwidth=1,
                tickfont=dict(color=PLOT_FONT),
                tickformat=",.0f",
                showline=True,
                linecolor="#444" if dark else "#CCC",
            ),
            bargap=0.15,
            margin=dict(t=80, b=40),
        )

        st.plotly_chart(fig_alloc, use_container_width=True)

        # --- Summary table (latest month) ---
        latest = df_alloc.iloc[-1]
        st.subheader(f"Latest Month — {latest['DateLabel']}")

        shown_instruments = [inst for inst in instrument_cols if inst in selected]
        total_val = sum(abs(latest[inst]) for inst in shown_instruments)

        rows_data = []
        for inst in shown_instruments:
            val = abs(latest[inst])
            pct = val / total_val * 100 if total_val else 0
            rows_data.append({
                "Instrument": inst,
                "Value": f"{val:,.0f}",
                "% of Total": f"{pct:.1f}%",
            })

        alloc_display_cols = ["Instrument", "Value", "% of Total"]
        alloc_num_cols = {"Value", "% of Total"}

        alloc_header = "".join(
            f"<th style='padding:8px 12px; border-bottom:2px solid {TABLE_HEADER_BORDER}; color:{TEXT};'>{c}</th>"
            for c in alloc_display_cols
        )
        alloc_rows_html = ""
        for rd in rows_data:
            cells = ""
            for col in alloc_display_cols:
                align = "right" if col in alloc_num_cols else "left"
                cells += f"<td style='padding:6px 12px; text-align:{align}; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{TEXT};'>{rd[col]}</td>"
            alloc_rows_html += f"<tr>{cells}</tr>"

        alloc_table = f"""
        <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:14px; background-color:{BG};">
        <thead><tr>{alloc_header}</tr></thead>
        <tbody>{alloc_rows_html}</tbody>
        </table>
        </div>
        """
        st.markdown(alloc_table, unsafe_allow_html=True)

# === Tab 4: Performance =====================================================

def xirr(cashflows):
    """Calculate XIRR given a list of (date, amount) tuples.

    Convention: negative = money going in (investment), positive = money coming out.
    Returns the annualised rate of return.
    """
    if len(cashflows) < 2:
        return None
    d0 = min(cf[0] for cf in cashflows)

    def npv(rate):
        return sum(
            amt / (1 + rate) ** ((d - d0).days / 365)
            for d, amt in cashflows
        )

    try:
        return brentq(npv, -0.999, 100.0, maxiter=10000)
    except Exception:
        return None


with tab4:
    if df_cashflows.empty or df_nav.empty or "NAV" not in df_nav.columns:
        st.warning("Cash flow or NAV data not found in the uploaded file.")
    else:
        # --- Prepare cash flow data ---
        df_cf = df_cashflows.copy()
        df_cf["DateParsed"] = pd.to_datetime(df_cf["Date"], format="%m/%d/%y")
        df_cf["Amount"] = pd.to_numeric(df_cf["Amount"], errors="coerce").fillna(0)
        df_cf["Year"] = df_cf["DateParsed"].dt.year

        # --- Prepare NAV data ---
        df_n = df_nav[["Date", "NAV"]].copy()
        df_n["NAV"] = pd.to_numeric(df_n["NAV"], errors="coerce")
        df_n["DateParsed"] = pd.to_datetime(df_n["Date"].astype(str), format="%Y%m")
        # Use last day of month for NAV date
        df_n["DateParsed"] = df_n["DateParsed"] + pd.offsets.MonthEnd(0)
        df_n = df_n.sort_values("DateParsed").reset_index(drop=True)
        df_n["Year"] = df_n["DateParsed"].dt.year

        inception_year = df_cf["Year"].min()
        latest_nav_date = df_n["DateParsed"].max()
        current_year = latest_nav_date.year
        years = list(range(inception_year, current_year + 1))

        # Build a lookup: last NAV for each year
        nav_by_year = {}
        for yr in years:
            yr_navs = df_n[df_n["Year"] == yr]
            if not yr_navs.empty:
                row = yr_navs.iloc[-1]
                nav_date = row["DateParsed"]
                # For the current year, use actual report end date instead of month-end
                if yr == current_year and report_end_date is not None:
                    nav_date = report_end_date
                nav_by_year[yr] = (nav_date, row["NAV"])

        # --- Calculate MWR for each year and cumulative ---
        results = []
        for yr in years:
            if yr not in nav_by_year:
                continue

            end_date, end_nav = nav_by_year[yr]

            # --- Yearly MWR ---
            # Beginning NAV = end of previous year's NAV (0 for inception year)
            if yr == inception_year:
                begin_nav = 0.0
                begin_date = df_cf["DateParsed"].min()
            else:
                prev = nav_by_year.get(yr - 1)
                if prev is None:
                    continue
                begin_date = prev[0]
                begin_nav = prev[1]

            # Cash flows for this year
            yr_cf = df_cf[df_cf["Year"] == yr]

            # Build XIRR cash flow list (investor perspective)
            # Negative = money invested, positive = money received back
            yearly_cfs = []
            if begin_nav > 0:
                yearly_cfs.append((begin_date, -begin_nav))
            for _, row in yr_cf.iterrows():
                yearly_cfs.append((row["DateParsed"], -row["Amount"]))
            yearly_cfs.append((end_date, end_nav))

            yearly_mwr = xirr(yearly_cfs)

            # De-annualise YTD return for current year
            if yr == current_year and yearly_mwr is not None:
                t = (end_date - begin_date).days / 365.25
                yearly_mwr = (1 + yearly_mwr) ** t - 1

            # --- Cumulative MWR (since inception) ---
            all_cf = df_cf[df_cf["DateParsed"] <= end_date]
            cum_cfs = []
            for _, row in all_cf.iterrows():
                cum_cfs.append((row["DateParsed"], -row["Amount"]))
            cum_cfs.append((end_date, end_nav))

            cum_mwr = xirr(cum_cfs)

            # Label
            if yr == current_year:
                label = f"{yr} YTD"
            else:
                label = str(yr)

            results.append({
                "Year": label,
                "Yearly MWR": yearly_mwr,
                "Cumulative MWR": cum_mwr,
            })

        # --- Display chart and table ---
        if results:
            st.subheader("Money Weighted Return (MWR)")

            # --- Bar chart ---
            labels = [r["Year"] for r in results]
            yearly_vals = [r["Yearly MWR"] for r in results]
            cum_vals = [r["Cumulative MWR"] for r in results]

            # Bar colors: green for positive, red for negative
            yearly_colors = ["#4CAF50" if v is not None and v >= 0 else "#EF5350" for v in yearly_vals]

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                name="Yearly MWR",
                x=labels,
                y=[v * 100 if v is not None else 0 for v in yearly_vals],
                marker=dict(color=yearly_colors, line=dict(width=0)),
                hovertemplate="%{x}: %{y:.2f}%<extra>Yearly MWR</extra>",
            ))
            fig_perf.add_trace(go.Scatter(
                name="Cumulative MWR",
                x=labels,
                y=[v * 100 if v is not None else 0 for v in cum_vals],
                mode="lines+markers",
                line=dict(color="#FFA726", width=2.5),
                marker=dict(size=7, color="#FFA726"),
                hovertemplate="%{x}: %{y:.2f}%<extra>Cumulative MWR</extra>",
            ))
            fig_perf.update_layout(
                height=500,
                paper_bgcolor=PLOT_PAPER,
                plot_bgcolor=PLOT_BG,
                font_color=PLOT_FONT,
                hovermode="x unified",
                legend=dict(
                    font=dict(color=PLOT_FONT, size=13),
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                ),
                xaxis=dict(
                    type="category",
                    tickfont=dict(color=PLOT_FONT, size=12),
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                ),
                yaxis=dict(
                    title="Return (%)",
                    showgrid=True,
                    gridcolor="#222" if dark else "#EEE",
                    tickfont=dict(color=PLOT_FONT),
                    ticksuffix="%",
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                    zeroline=True,
                    zerolinecolor="#666" if dark else "#AAA",
                    zerolinewidth=1,
                ),
                bargap=0.3,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            perf_header = ""
            for col in ["Year", "Yearly MWR", "Cumulative MWR"]:
                perf_header += f"<th style='padding:8px 12px; text-align:{'left' if col == 'Year' else 'right'}; border-bottom:2px solid {TABLE_HEADER_BORDER}; color:{TEXT};'>{col}</th>"

            perf_rows_html = ""
            for r in results:
                yr_val = r["Yearly MWR"]
                cum_val = r["Cumulative MWR"]
                yr_str = f"{yr_val:.2%}" if yr_val is not None else "N/A"
                cum_str = f"{cum_val:.2%}" if cum_val is not None else "N/A"

                # Color: green for positive, red for negative
                yr_color = "#4CAF50" if yr_val is not None and yr_val >= 0 else "#EF5350"
                cum_color = "#4CAF50" if cum_val is not None and cum_val >= 0 else "#EF5350"

                perf_rows_html += f"""<tr>
                    <td style='padding:6px 12px; text-align:left; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{TEXT};'>{r['Year']}</td>
                    <td style='padding:6px 12px; text-align:right; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{yr_color}; font-weight:600;'>{yr_str}</td>
                    <td style='padding:6px 12px; text-align:right; border-bottom:1px solid {TABLE_ROW_BORDER}; color:{cum_color}; font-weight:600;'>{cum_str}</td>
                </tr>"""

            perf_table = f"""
            <div style="overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:15px; background-color:{BG};">
            <thead><tr>{perf_header}</tr></thead>
            <tbody>{perf_rows_html}</tbody>
            </table>
            </div>
            """
            st.markdown(perf_table, unsafe_allow_html=True)
        else:
            st.warning("Could not calculate performance — insufficient data.")

# === Tab 5: Return ===========================================================
with tab5:
    if df_cashflows.empty or df_nav.empty or "NAV" not in df_nav.columns:
        st.warning("Cash flow or NAV data not found in the uploaded file.")
    else:
        # --- Prepare NAV data (monthly, end-of-month) ---
        df_nav_ret = df_nav[["Date", "NAV"]].copy()
        df_nav_ret["NAV"] = pd.to_numeric(df_nav_ret["NAV"], errors="coerce")
        df_nav_ret["DateParsed"] = pd.to_datetime(df_nav_ret["Date"].astype(str), format="%Y%m")
        df_nav_ret["DateParsed"] = df_nav_ret["DateParsed"] + pd.offsets.MonthEnd(0)
        df_nav_ret = df_nav_ret.sort_values("DateParsed").reset_index(drop=True)

        # --- Prepare deposit data ---
        df_dep = df_cashflows.copy()
        df_dep["DateParsed"] = pd.to_datetime(df_dep["Date"], format="%m/%d/%y")
        df_dep["Amount"] = pd.to_numeric(df_dep["Amount"], errors="coerce").fillna(0)
        df_dep["YearMonth"] = df_dep["DateParsed"].dt.to_period("M")

        # ---------------------------------------------------------------
        # Monthly Return
        # ---------------------------------------------------------------
        monthly_results = []
        for i in range(len(df_nav_ret)):
            end_date = df_nav_ret.iloc[i]["DateParsed"]
            end_nav = df_nav_ret.iloc[i]["NAV"]
            period = end_date.to_period("M")

            # Beginning NAV = previous month's ending NAV (0 for first month)
            begin_nav = df_nav_ret.iloc[i - 1]["NAV"] if i > 0 else 0.0

            # Total deposits within this month
            month_deps = df_dep[df_dep["YearMonth"] == period]["Amount"].sum()

            monthly_return = end_nav - begin_nav - month_deps
            monthly_results.append({
                "Month": end_date.strftime("%Y/%m"),
                "Beginning NAV": begin_nav,
                "Deposits": month_deps,
                "Ending NAV": end_nav,
                "Return": monthly_return,
            })

        # ---------------------------------------------------------------
        # Annual Return
        # ---------------------------------------------------------------
        df_nav_ret["Year"] = df_nav_ret["DateParsed"].dt.year
        df_dep["Year"] = df_dep["DateParsed"].dt.year
        years_ret = sorted(df_nav_ret["Year"].unique())

        annual_results = []
        for yr in years_ret:
            yr_navs = df_nav_ret[df_nav_ret["Year"] == yr]
            if yr_navs.empty:
                continue

            end_nav = yr_navs.iloc[-1]["NAV"]

            # Beginning NAV = end of previous year (0 for first year)
            prev_yr_navs = df_nav_ret[df_nav_ret["Year"] == yr - 1]
            begin_nav = prev_yr_navs.iloc[-1]["NAV"] if not prev_yr_navs.empty else 0.0

            # Total deposits for the year
            year_deps = df_dep[df_dep["Year"] == yr]["Amount"].sum()

            annual_return = end_nav - begin_nav - year_deps

            label = f"{yr} YTD" if yr == years_ret[-1] else str(yr)
            annual_results.append({
                "Year": label,
                "Beginning NAV": begin_nav,
                "Deposits": year_deps,
                "Ending NAV": end_nav,
                "Return": annual_return,
            })

        # ---------------------------------------------------------------
        # Display Monthly Return (first)
        # ---------------------------------------------------------------
        if monthly_results:
            st.subheader("Monthly Return")

            mon_labels = [r["Month"] for r in monthly_results]
            mon_vals = [r["Return"] for r in monthly_results]
            mon_colors = ["#4CAF50" if v >= 0 else "#EF5350" for v in mon_vals]

            fig_mon = go.Figure()
            fig_mon.add_trace(go.Bar(
                x=mon_labels,
                y=mon_vals,
                marker=dict(color=mon_colors, line=dict(width=0)),
                hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
            ))
            fig_mon.update_layout(
                height=450,
                paper_bgcolor=PLOT_PAPER,
                plot_bgcolor=PLOT_BG,
                font_color=PLOT_FONT,
                xaxis=dict(
                    type="category",
                    tickfont=dict(color=PLOT_FONT, size=10),
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                    tickvals=[lbl for lbl in mon_labels if lbl.endswith("/01")],
                    ticktext=[lbl[:4] for lbl in mon_labels if lbl.endswith("/01")],
                ),
                yaxis=dict(
                    title="Return (Base Currency)",
                    showgrid=True,
                    gridcolor="#222" if dark else "#EEE",
                    tickfont=dict(color=PLOT_FONT),
                    tickformat=",.0f",
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                    zeroline=True,
                    zerolinecolor="#666" if dark else "#AAA",
                    zerolinewidth=1,
                ),
                bargap=0.15,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_mon, use_container_width=True)

        # ---------------------------------------------------------------
        # Display Annual Return (second)
        # ---------------------------------------------------------------
        if annual_results:
            st.subheader("Annual Return")

            ann_labels = [r["Year"] for r in annual_results]
            ann_vals = [r["Return"] for r in annual_results]
            ann_colors = ["#4CAF50" if v >= 0 else "#EF5350" for v in ann_vals]

            fig_ann = go.Figure()
            fig_ann.add_trace(go.Bar(
                x=ann_labels,
                y=ann_vals,
                marker=dict(color=ann_colors, line=dict(width=0)),
                hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
            ))
            fig_ann.update_layout(
                height=450,
                paper_bgcolor=PLOT_PAPER,
                plot_bgcolor=PLOT_BG,
                font_color=PLOT_FONT,
                xaxis=dict(
                    type="category",
                    tickfont=dict(color=PLOT_FONT, size=12),
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                ),
                yaxis=dict(
                    title="Return (Base Currency)",
                    showgrid=True,
                    gridcolor="#222" if dark else "#EEE",
                    tickfont=dict(color=PLOT_FONT),
                    tickformat=",.0f",
                    showline=True,
                    linecolor="#444" if dark else "#CCC",
                    zeroline=True,
                    zerolinecolor="#666" if dark else "#AAA",
                    zerolinewidth=1,
                ),
                bargap=0.3,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_ann, use_container_width=True)
