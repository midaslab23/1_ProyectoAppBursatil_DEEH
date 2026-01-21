import streamlit as st

# streamlit_app_finanzas.py
# Aplicación Streamlit modular para análisis financiero
# Módulo inicial: Selección de acciones + rango de fechas + periodicidad + validación de datos
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# -------------------- CONFIG / Parámetros globales (fácil de editar) --------------------
CONFIG = {
    "DEFAULT_TICKERS": ["NVDA.MX", "META.MX", "GM.MX"],
    "MIN_ROWS_ABSOLUTE": 30,  # mínimo absoluto de filas para ejecutar cálculos
    "PERIODS_PER_YEAR": {"Daily": 252, "Weekly": 52, "Monthly": 12},
    "USER_THEME": {
        "bg_color": "#0b0f14",
        "card_bg": "#0f1720",
        "accent_blue": "#2EA3FF",
        "accent_neon": "#39FF14",
        "text": "#E6EEF3"
    }
}

# -------------------- Estilos CSS (tema negro / azul / verde-neon) --------------------
def local_css():
    t = CONFIG['USER_THEME']
    css = f"""
    <style>
    /* Background and base colors */
    .stApp {{ background-color: {t['bg_color']}; color: {t['text']}; }}
    .css-1d391kg {{ background-color: {t['bg_color']}; }} /* root container (may vary) */

    /* Sidebar */
    .stSidebar {{ background-color: {t['card_bg']}; border-right: 1px solid rgba(255,255,255,0.03); }}

    /* Cards / containers */
    .stButton>button, .stDownloadButton>button {{ background: linear-gradient(90deg, {t['accent_blue']} 0%, {t['accent_neon']} 100%); color: #001; border-radius: 8px; padding: 8px 12px; font-weight:600; }}
    .stTextInput>div>div>input, .stDateInput>div>div>input, .stSelectbox>div>div>div {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); color: {t['text']}; border-radius:6px; padding:6px; }}

    /* Headers */
    h1, h2, h3 {{ color: {t['accent_blue']}; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }}

    /* Make dataframes easier to read */
    .stDataFrame table {{ background-color: transparent !important; color: {t['text']} !important; }}

    /* Small tweaks */
    .element-container {{ background-color: {t['card_bg']}; border-radius: 10px; padding: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

#
st.set_page_config(
    page_title="Finanzas - App Modular",
    layout="wide",
    initial_sidebar_state="expanded"
)
#

# -------------------- Caching: descarga de datos (por ticker) --------------------
@st.cache_data(ttl=60*60)  # cache por 1 hora
def download_ticker(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Descarga un ticker individualmente y devuelve un df con índice datetime y columnas OHLCV.
    interval aceptado: '1d','1wk','1mo'"""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, interval=interval, auto_adjust=False)
        if df is None:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return pd.DataFrame()

# -------------------- Utilidades de validación --------------------

def required_min_rows(years_for_seasonality: float, periodicity: str) -> int:
    ppy = CONFIG['PERIODS_PER_YEAR'].get(periodicity, 252)
    # requerimos al menos el 60% de los períodos esperados por los años pedidos, y mínimo absoluto
    expected = int(years_for_seasonality * ppy)
    return max(CONFIG['MIN_ROWS_ABSOLUTE'], int(0.6 * expected))




# -------------------- Estructura de la app / Sidebar modules --------------------

def main():
    local_css()

   

    st.sidebar.title("Módulos")
    module = st.sidebar.selectbox("Selecciona módulo:", ["Análisis técnico", "Optimización + Simulación", "Pronóstico ML"])

    st.sidebar.markdown("---")

    # Parámetros globales visibles en sidebar
    st.sidebar.header("Parámetros generales")
    default_tickers = st.sidebar.text_input("Tickers por defecto (coma-separados)", ",".join(CONFIG['DEFAULT_TICKERS']))
    ticker_list_input = st.sidebar.text_input("Ingresar tickers (coma-separados)", ",".join(CONFIG['DEFAULT_TICKERS']))
    st.sidebar.write("Tip: usar símbolos de Yahoo Finance, p.ej. AMXL.MX, WALMEX.MX")

    # Guardamos en session_state para que otros módulos puedan leerlo si se implementan
    if 'global_params' not in st.session_state:
        st.session_state['global_params'] = {}
    st.session_state['global_params']['available_tickers'] = [t.strip().upper() for t in ticker_list_input.split(',') if t.strip()]

    # Rango de fechas y periodicidad comunes
    today = datetime.today().date()
    default_start = today.replace(year=today.year - 1)
    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", today)
    periodicity = st.sidebar.selectbox("Periodicidad", ["Daily", "Weekly", "Monthly"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Interfaz modular, parámetros principales arriba. Cada módulo es independiente y usa parámetros comunes.")

    # Main area: renderizamos según módulo
    if module == "Análisis técnico":
        render_technical_module(start_date, end_date, periodicity)
    elif module == "Optimización + Simulación":
        st.info("Módulo de optimización y simulación todavía no implementado. ¿Quieres que lo haga ahora?")
    elif module == "Pronóstico ML":
        st.info("Módulo de pronóstico ML no implementado (próximamente). Puedo crear plantillas: features, train/test etc.")

# -------------------- Módulo: Análisis técnico --------------------



def render_technical_module(start_date, end_date, periodicity):
    st.title("Análisis técnico")
    st.write("Selecciona las acciones que quieres analizar, el rango de fechas y la periodicidad. La app validará si hay suficientes datos.")

    available = st.session_state['global_params']['available_tickers'] if 'global_params' in st.session_state else CONFIG['DEFAULT_TICKERS']

    # Selección de tickers
    tickers = st.multiselect("Selecciona tickers:", options=available, default=available[:3])

    # Parámetros propios del módulo
    years_for_seasonality = st.number_input(
        "Años para estacionalidad (para validación de datos)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.5
    )

    run = st.button("Descargar y mostrar precios")

    if not run:
        st.info("Configura los parámetros y presiona **Descargar y mostrar precios**.")
        return

    if not tickers:
        st.warning("No seleccionaste ningún ticker.")
        return

    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    interval = interval_map.get(periodicity, "1d")

    start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    min_rows_needed = required_min_rows(years_for_seasonality, periodicity)

    errors = {}
    good_data = {}

    progress = st.progress(0)
    for i, t in enumerate(tickers):
        progress.progress(int((i + 1) / len(tickers) * 100))
        df = download_ticker(t, start=start, end=end, interval=interval)

        if df.empty or 'Close' not in df.columns:
            errors[t] = "no_data"
            continue

        df = df.sort_index()
        df = df[df['Close'].notna()]

        if df.shape[0] < min_rows_needed:
            errors[t] = f"pocas filas ({df.shape[0]} < {min_rows_needed})"
            continue

        good_data[t] = df

    progress.empty()

    if errors:
        st.warning("Algunos tickers no pudieron procesarse:")
        for t, msg in errors.items():
            st.write(f"• **{t}** → {msg}")

    if not good_data:
        st.error("No hay tickers con datos suficientes para mostrar gráficos.")
        return

    st.success(f"Mostrando gráficos de precio para {len(good_data)} tickers.")

    # ------------------ GRÁFICOS DE PRECIO (Close) ------------------
    
    for t, df in good_data.items():
        st.subheader(f"{t} — Precio de cierre")

        Close_series = df['Close'].squeeze()

        last = float(Close_series.iloc[-1])
        mean = float(Close_series.mean())
        std = float(Close_series.std())

        st.markdown(
            f"**Último Close:** {last:.4f} &nbsp;&nbsp; "
            f"**Media:** {mean:.4f} &nbsp;&nbsp; "
            f"**Std:** {std:.4f}"
        )

        # --- Matplotlib dark style (from tecnicos.py) ---
        plt.style.use('dark_background')
        rc = {
        'axes.facecolor': '#222222',
        'figure.facecolor':'#222222',
        'axes.edgecolor': '#444444',
        'grid.color': '#333333',
        'xtick.color': 'white',
        'ytick.color': 'white',
        }
        plt.rcParams.update(rc)


        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(Close_series.index, Close_series.values, linewidth=1.6, alpha=0.95)
        ax.set_title(f"{t} — Precio de cierre (Close)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio Close")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()


        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


        with st.expander("Ver datos (primeras filas)"):
            st.dataframe(df.head())
        
        # ---------------- ANÁLISIS TÉCNICO (robusto: asegurar 1-D y tipos) ----------------
        st.subheader(f"{t} — Análisis técnico")

        # ---------- Helpers: normalizar columnas OHLCV (robusto) ----------
        # Helper: normalizar a Open/High/Low/Close/Volume (Title case)
        def ensure_ohlcv_columns(df):
            df = df.copy()
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

            # Flatten MultiIndex si aplica
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    last = df.columns.get_level_values(-1)
                    last_names = [str(x) for x in last]
                    if any('close' in x.lower() or 'open' in x.lower() for x in last_names):
                        df.columns = last_names
                    else:
                        df.columns = ['_'.join([str(c) for c in col if c is not None and str(c) != '']) for col in df.columns]
                except Exception:
                    df.columns = ['_'.join([str(c) for c in col if c is not None and str(c) != '']) for col in df.columns]

            # Forzar strings
            df.columns = [str(c) for c in df.columns]

            # Mapear variantes a Title case
            mapping = {}
            used = set()
            for c in df.columns:
                low = c.lower()
                if 'open' in low and 'Open' not in used:
                    mapping[c] = 'Open'; used.add('Open')
                elif 'high' in low and 'High' not in used:
                    mapping[c] = 'High'; used.add('High')
                elif 'low' in low and 'Low' not in used:
                    mapping[c] = 'Low'; used.add('Low')
                elif 'adj close' in low and 'Close' not in used:
                    mapping[c] = 'Close'; used.add('Close')
                elif 'close' in low and 'Close' not in used:
                    mapping[c] = 'Close'; used.add('Close')
                elif 'volume' in low and 'Volume' not in used:
                    mapping[c] = 'Volume'; used.add('Volume')

            if mapping:
                df = df.rename(columns=mapping)

            # heurística final: si falta Close y hay una sola columna numérica -> usarla
            if 'Close' not in df.columns:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric_cols) == 1:
                    df = df.rename(columns={numeric_cols[0]: 'Close'})
                else:
                    candidates = [c for c in df.columns if ('price' in c.lower() or 'last' in c.lower())]
                    if candidates:
                        df = df.rename(columns={candidates[0]: 'Close'})

            return df




        # ---------- Preparación y normalización ----------
        dft = df.copy()
        dft.index = pd.to_datetime(dft.index)
        dft.sort_index(inplace=True)
        dft = ensure_ohlcv_columns(dft)

        # DEBUG temporal
        st.write(f"Columnas normalizadas para {t}:", dft.columns.tolist())
        st.dataframe(dft.head())

        if 'Close' not in dft.columns:
            st.warning(f"{t}: no se encontró la columna 'Close' tras normalizar. Columnas detectadas: {dft.columns.tolist()}")
            continue

        # Forzar tipos numéricos en Title case
        for col in ['Open','High','Low','Close','Volume']:
            if col in dft.columns:
                dft[col] = pd.to_numeric(dft[col].astype(object), errors='coerce').astype(float)

        # eliminar filas sin OHLC esenciales (usar las mismas mayúsculas)
        required = [c for c in ['Open','High','Low','Close'] if c in dft.columns]
        dft.dropna(subset=required, inplace=True)
        if dft.empty or len(dft) < 3:
            st.warning(f"{t}: tras limpiar OHLC no quedan filas suficientes. Se salta.")
            continue

        # Asegurar índice datetime y orden
        dft.index = pd.to_datetime(dft.index)
        dft.sort_index(inplace=True)

        # Serie de cierre para indicadores
        close_series = dft['Close']

        # Parámetros
        sma_short, sma_med, sma_long = 21, 55, 200

        # Calcular indicadores usando close_series
        df_ind = dft.copy()
        df_ind[f"SMA_{sma_short}"] = SMAIndicator(close_series, window=sma_short).sma_indicator()
        df_ind[f"SMA_{sma_med}"]   = SMAIndicator(close_series, window=sma_med).sma_indicator()
        df_ind[f"SMA_{sma_long}"]  = SMAIndicator(close_series, window=sma_long).sma_indicator()

        macd = MACD(close_series, window_slow=26, window_fast=12, window_sign=9)
        df_ind["MACD"] = macd.macd()
        df_ind["MACD_signal"] = macd.macd_signal()
        df_ind["MACD_diff"] = macd.macd_diff()

        df_ind["RSI"] = RSIIndicator(close_series, window=14).rsi()

        # Preparar addplots (usar df_ind index alineado con dft)
        apds = [
            mpf.make_addplot(df_ind[f"SMA_{sma_short}"], color='#EDF67D', width=0.8),
            mpf.make_addplot(df_ind[f"SMA_{sma_med}"],   color='#CA7DF9', width=0.8),
            mpf.make_addplot(df_ind[f"SMA_{sma_long}"],  color='#40F7DF', width=0.8),

            mpf.make_addplot(df_ind["MACD"], panel=1, color='fuchsia', width=0.9),
            mpf.make_addplot(df_ind["MACD_signal"], panel=1, color='gold', width=0.9),
            mpf.make_addplot(df_ind["MACD_diff"], panel=1, type='bar', color='gray', alpha=0.5),

            mpf.make_addplot(df_ind["RSI"], panel=2, color='yellow', width=0.9),
        ]

        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'figure.facecolor':'#222222'})

        try:
            fig, _ = mpf.plot(
                dft,
                type='candle',
                style=mpf_style,
                addplot=apds,
                volume='Volume' in dft.columns,
                panel_ratios=(6,2,2),
                figsize=(14,9),
                title=f"{t} — Candles + SMAs + MACD + RSI",
                tight_layout=True,
                returnfig=True
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"{t}: error al graficar con mplfinance: {e}")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(close_series.index, close_series.values)
            ax.set_title(f"{t} — Precio (fallback)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)



        with st.expander("Ver datos (primeras filas)"):
            st.dataframe(dft.head())

# -------------------- Entrypoint --------------------
if __name__ == '__main__':
    main()