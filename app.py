import streamlit as st

# streamlit_app_finanzas.py
# Aplicación Streamlit modular para análisis financiero
# Módulo inicial: Selección de acciones + rango de fechas + periodicidad + validación de datos
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, date
import math

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
    h1, h2, h3 {{ color: {t['accent_blue']}; font-family: Merriweather, Playfair Display, Lato, Open Sans; }}

    /* Make dataframes easier to read */
    .stDataFrame table {{ background-color: transparent !important; color: {t['text']} !important; }}

    /* Small tweaks */
    .element-container {{ background-color: {t['card_bg']}; border-radius: 10px; padding: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

#
st.set_page_config(
    page_title="App Financiera - Proyecto Midas",
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

    st.sidebar.markdown("""
**Elaborado por:**<br>
**Diego Eduardo Enríquez Hernández**<br><br>
<a href="https://www.linkedin.com/in/diegoeduardoenriquezhernandez/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="26" style="vertical-align:middle"/>
    &nbsp; Ver perfil en LinkedIn
</a><br><br>
**Economics · ML/Data Science · Finance**
""", unsafe_allow_html=True)
    

    


    # Main area: renderizamos según módulo
    if module == "Análisis técnico":
        render_technical_module(start_date, end_date, periodicity)
    elif module == "Optimización + Simulación":
        render_optimization_module()
    elif module == "Pronóstico ML":
        st.info("Módulo de pronóstico ML aún no implementado (próximamente)")

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

                # ------------------ Añadir Ichimoku, ADX y Stochastic + mejorar leyendas ------------------
        # Asegurarnos de series necesarias
        high = dft['High']
        low = dft['Low']
        vol = dft['Volume'] if 'Volume' in dft.columns else None

        # Parámetros Ichimoku / otros
        tenkan_window, kijun_window, senkou_b_window, senkou_shift = 9, 26, 52, 26
        stoch_window, stoch_smooth = 14, 3
        cci_window = 20
        adx_window = 14

        # Calcular Ichimoku
        tenkan = (high.rolling(window=tenkan_window).max() + low.rolling(window=tenkan_window).min()) / 2
        kijun  = (high.rolling(window=kijun_window).max() + low.rolling(window=kijun_window).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(senkou_shift)
        senkou_b = ((high.rolling(window=senkou_b_window).max() + low.rolling(window=senkou_b_window).min()) / 2).shift(senkou_shift)
        chikou = close_series.shift(-(senkou_shift))

        df_ind["ICH_Tenkan"] = tenkan
        df_ind["ICH_Kijun"]  = kijun
        df_ind["ICH_Senkou_A"] = senkou_a
        df_ind["ICH_Senkou_B"] = senkou_b
        df_ind["ICH_Chikou"] = chikou

        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close_series,
                                    window=stoch_window, smooth_window=stoch_smooth)
        df_ind["STOCH_k"] = stoch.stoch()
        df_ind["STOCH_d"] = stoch.stoch_signal()

        # CCI, ADX, OBV (si aún no calculaste)
        df_ind["CCI"] = CCIIndicator(high=high, low=low, close=close_series, window=cci_window).cci()
        df_ind["ADX"] = ADXIndicator(high=high, low=low, close=close_series, window=adx_window).adx()
        #if vol is not None:
        df_ind["OBV"] = OnBalanceVolumeIndicator(close=close_series, volume=vol).on_balance_volume()

        #RECOMENDACIONES-----------------
        # ------------------ Señales simples (reglas condicionales, UX) ------------------
        def generate_simple_signals(df_ind, close_series, sma_short, sma_med, sma_long):
            """
            Devuelve (signal_label, confidence_text, bullets_list)
            signal_label: 'buy'|'sell'|'neutral'
            confidence_text: 'Alta / Media / Baja' (heurística)
            bullets_list: lista de strings con explicaciones cortas
            """
            bulls = 0
            bears = 0
            notes = []

            # último precio
            try:
                price = float(close_series.iloc[-1])
            except Exception:
                return "neutral", "Sin datos", ["No hay precio válido para generar señales."]

            # SMA checks
            for name, window in [(f"SMA_{sma_short}", sma_short), (f"SMA_{sma_med}", sma_med), (f"SMA_{sma_long}", sma_long)]:
                series = df_ind.get(name)
                if series is not None and len(series.dropna()) > 0:
                    val = float(series.dropna().iloc[-1])
                    if price > val:
                        bulls += 1
                        notes.append(f"Precio por encima de {name} ({window}) — sesgo alcista.")
                    else:
                        bears += 1
                        notes.append(f"Precio por debajo de {name} ({window}) — sesgo bajista.")

            # Ichimoku (Tenkan/Kijun/Senkou)
            ten = df_ind.get("ICH_Tenkan")
            kij = df_ind.get("ICH_Kijun")
            sa  = df_ind.get("ICH_Senkou_A")
            sb  = df_ind.get("ICH_Senkou_B")
            if ten is not None and kij is not None:
                tval = float(ten.dropna().iloc[-1]) if len(ten.dropna())>0 else None
                kval = float(kij.dropna().iloc[-1]) if len(kij.dropna())>0 else None
                if tval and kval:
                    if price > max(tval, kval):
                        bulls += 1
                        notes.append("Precio por encima de Tenkan/Kijun — momentum alcista.")
                    elif price < min(tval, kval):
                        bears += 1
                        notes.append("Precio por debajo de Tenkan/Kijun — momentum bajista.")
                    else:
                        notes.append("Precio entre Tenkan y Kijun — posible consolidación.")

            # Kumo (Senkou A/B): nube alcista si A > B
            if sa is not None and sb is not None and len(sa.dropna())>0 and len(sb.dropna())>0:
                a = float(sa.dropna().iloc[-1])
                b = float(sb.dropna().iloc[-1])
                if a > b and price > a:
                    bulls += 1
                    notes.append("En / sobre nube (SenkouA > SenkouB) — tendencia alcista respaldada por Kumo.")
                elif a < b and price < b:
                    bears += 1
                    notes.append("Por debajo de la nube (SenkouA < SenkouB) — tendencia bajista.")
                else:
                    notes.append("Nube neutra o precio dentro de la nube — señal débil.")

            # MACD (momentum)
            macd = df_ind.get("MACD")
            macd_sig = df_ind.get("MACD_signal")
            if macd is not None and macd_sig is not None and len(macd.dropna())>1 and len(macd_sig.dropna())>1:
                m = float(macd.dropna().iloc[-1])
                s = float(macd_sig.dropna().iloc[-1])
                if m > s:
                    bulls += 1
                    notes.append("MACD > señal — momentum alcista.")
                else:
                    bears += 1
                    notes.append("MACD < señal — momentum bajista.")

            # RSI (sobrecompra/sobreventa)
            rsi = df_ind.get("RSI")
            if rsi is not None and len(rsi.dropna())>0:
                r = float(rsi.dropna().iloc[-1])
                if r < 30:
                    bulls += 1
                    notes.append(f"RSI {r:.0f} — condición de sobreventa (posible rebote).")
                elif r > 70:
                    bears += 1
                    notes.append(f"RSI {r:.0f} — condición de sobrecompra (posible corrección).")
                else:
                    notes.append(f"RSI {r:.0f} — neutro.")

            # ADX (fuerza de la tendencia)
            adx = df_ind.get("ADX")
            adx_strength = None
            if adx is not None and len(adx.dropna())>0:
                a = float(adx.dropna().iloc[-1])
                adx_strength = a
                if a >= 25:
                    notes.append(f"ADX {a:.0f} — tendencia fuerte.")
                else:
                    notes.append(f"ADX {a:.0f} — tendencia débil / rango.")

            # Score
            score = bulls - bears
            if score >= 2:
                label = "buy"
                conf = "Alta" if adx_strength and adx_strength >= 25 else "Media"
            elif score <= -2:
                label = "sell"
                conf = "Alta" if adx_strength and adx_strength >= 25 else "Media"
            else:
                label = "neutral"
                conf = "Baja"

            # Resumen breve (3 bullets máximo)
            summary = []
            # Priorizar items sintéticos primero
            if label == "buy":
                summary.append("Sesgo técnico: ALCISTA")
            elif label == "sell":
                summary.append("Sesgo técnico: BAJISTA")
            else:
                summary.append("Sesgo técnico: NEUTRO / Esperar confirmación")

            # añadir 2 líneas explicativas más
            for n in notes[:7]:
                summary.append(n)

            return label, conf, summary

        # Generar y mostrar la señal (se coloca antes de plotting)
        signal_label, confidence, explanation = generate_simple_signals(df_ind, close_series, sma_short, sma_med, sma_long)

        # Mostrar UI amigable y NO-ASESORAMIENTO
        with st.container():
            st.markdown("### Señales Técnicas")
            if signal_label == "buy":
                st.success(f"⚒️ Señal: COMPRA  — Confianza: {confidence}")
            elif signal_label == "sell":
                st.error(f"🛑 Señal: VENTA  — Confianza: {confidence}")
            else:
                st.info(f"⚪ Señal: NEUTRA/ESPERAR  — Confianza: {confidence}")

            # bullets explicativos
            for b in explanation:
                st.markdown(f"- {b}")

            # Disclaimer breve
            st.caption("Nota: estas señales son condicionales y basadas en reglas simples. No constituyen asesoramiento financiero. Revísalas con más análisis antes de tomar decisiones.")


        #################################




        # Construir addplots
        apds = []

        # SMAs y Ichimoku lines (panel 0)
        apds += [
            mpf.make_addplot(df_ind.get(f"SMA_{sma_short}"), panel=0, color='#EDF67D', width=0.8, label=f"SMA short {sma_short}"),
            mpf.make_addplot(df_ind.get(f"SMA_{sma_med}"), panel=0, color='#CA7DF9', width=0.8,label=f"SMA med {sma_med}"),
            mpf.make_addplot(df_ind.get(f"SMA_{sma_long}"), panel=0, color='#40F7DF', width=0.8,label=f"SMA long {sma_long}"),
            mpf.make_addplot(df_ind.get("ICH_Tenkan"), panel=0, color='lime', width=0.9),
            mpf.make_addplot(df_ind.get("ICH_Kijun"),  panel=0, color='lightcoral', width=0.9),
            mpf.make_addplot(df_ind.get("ICH_Senkou_A"), panel=0, color='lightgreen', width=1.0, label='ICH Senkou A'),
            mpf.make_addplot(df_ind.get("ICH_Senkou_B"), panel=0, color='red', width=1.0, label='ICH Senkou B'),
            # Chikou (lagging) como línea en panel 0
            #mpf.make_addplot(df_ind.get("ICH_Chikou"), panel=0, color='cyan', width=0.1),
        ]

        # MACD (panel 1)
        apds += [
            mpf.make_addplot(df_ind.get("MACD"), panel=1, color='fuchsia', width=0.9, label='MACD'),
            mpf.make_addplot(df_ind.get("MACD_signal"), panel=1, color='gold', width=0.9,label='Signal'),
            mpf.make_addplot(df_ind.get("MACD_diff"), panel=1, type='bar', color='gray', alpha=0.6,label='MACD hist')
        ]

        # RSI + ADX en el mismo panel (panel 2)
        apds += [
            mpf.make_addplot(df_ind.get("RSI"), panel=2, color='yellow', width=0.9,label="RSI"),
            mpf.make_addplot(df_ind.get("ADX"), panel=2, color='lime', width=0.9,label="ADX"),
        ]

        # Stochastic + CCI en panel 3
        apds += [
            mpf.make_addplot(df_ind.get("STOCH_k"), panel=3, color='lime', width=0.9, label=''),
            mpf.make_addplot(df_ind.get("STOCH_d"), panel=3, color='magenta', width=0.9, label='Stoch %D'),
            mpf.make_addplot(df_ind.get("CCI"), panel=3, color='purple', width=0.9, label='CCI'),
        ]

        apds += [ mpf.make_addplot(df_ind.get("OBV"), panel=4, color='silver', label='OBV')]

        # Estilo
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds' if 'nightclouds' in mpf.available_styles() else 'yahoo',
                                    rc={'figure.facecolor': '#222222', 'axes.facecolor': '#222222'})

        # Panel ratios: main, macd, rsi/adx, stoch/cci
        panel_ratios = (6, 2, 2, 2, 2)  # añadir más si hay más paneles

        # Plotear y recuperar figuras/axes
        try:
            fig, axes = mpf.plot(
                dft,
                type='candle',
                style=mpf_style,
                addplot=apds,
                volume=('Volume' in dft.columns),
                panel_ratios=panel_ratios,
                figsize=(14, 10),
                title=f"{t} — Candles + SMAs + Ichimoku + MACD + RSI + Stoch + CCI",
                tight_layout=True,
                returnfig=True
            )

            # axes es lista; ax_price = axes[0]; otros ax = axes[1], axes[2], ...
            ax_price = axes[0]

            # Pintar la nube Ichimoku (SenkouA/SenkouB) sobre ax_price (solo donde ambas series no-nulas)
            #sa = df_ind.get("ICH_Senkou_A")
            #sb = df_ind.get("ICH_Senkou_B")
            #if sa is not None and sb is not None:
                #mask_valid = (~sa.isna()) & (~sb.isna())
                #if mask_valid.any():
                    #ax_price.fill_between(dft.index, sa, sb,
                                        #where=(mask_valid & (sa >= sb)),
                                        #interpolate=True, color='lightgreen', alpha=0.12)
                    #ax_price.fill_between(dft.index, sa, sb,
                                        #where=(mask_valid & (sa < sb)),
                                        #interpolate=True, color='lightcoral', alpha=0.12)

            # Intentar añadir leyendas en cada panel (siempre que existan handles)
            try:
                for ax in axes:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(handles, labels, loc='upper left', fontsize='small', framealpha=0.6)
            except Exception:
                pass

            # Mostrar en Streamlit
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.error(f"{t}: error al graficar con mplfinance: {e}")
            # fallback: gráfico simple de close
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(close_series.index, close_series.values, linewidth=1.6)
            ax.set_title(f"{t} — Precio (fallback)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("Ver datos (primeras filas)"):
            st.dataframe(dft.head())

# -------------------- Módulo: Optimización + Simulación --------------------

# Reutiliza download_ticker si ya existe en tu app; si no, usa este fallback:
def safe_download_prices(tickers, start, end, interval="1d"):
    try:
        # intentamos usar tu helper si existe
        df = download_ticker(tickers[0], start=start, end=end, interval=interval)
        # si existe download_ticker debería devolver df por ticker; pero para simplicidad usamos yf.download
    except Exception:
        pass
    # fallback a yfinance bulk (menos robusto, pero funciona)
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)["Close"]
    # asegurar DataFrame con columnas tickers
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.ffill().dropna(axis=1, how="all")

# Métricas y helpers (adaptadas de tu tesis)
def compute_metrics(returns_series, rf_annual=0.0):
    arr = np.asarray(returns_series)
    out = {'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'sortino': np.nan, 'max_drawdown': np.nan,
           'total_return': np.nan, 'ann_return_geom': np.nan}
    if arr.size == 0:
        return out

    n = len(arr)
    # total acumulado sobre el periodo
    total = np.prod(1 + arr) - 1
    out['total_return'] = total

    # annualizado geométrico (si n>0)
    if n > 0:
        out['ann_return'] = np.mean(arr) * 252  # aproximación
        out['ann_return_geom'] = (1 + total)**(252.0 / n) - 1
    # volatilidad (ann)
    ann_vol = np.std(arr, ddof=0) * math.sqrt(252)
    out['ann_vol'] = ann_vol

    # Sharpe (usando ann_return geométrica o aritmética; usamos geométrica por robustez)
    rf_ann = rf_annual if rf_annual is not None else 0.0
    if ann_vol > 0:
        out['sharpe'] = (out['ann_return_geom'] - rf_ann) / ann_vol
    else:
        out['sharpe'] = np.nan

    # Sortino
    downside = arr[arr < 0]
    if len(downside) > 1:
        dd_std = np.std(downside, ddof=0) * math.sqrt(252)
        out['sortino'] = (out['ann_return_geom'] - rf_ann) / dd_std if dd_std > 0 else np.nan
    else:
        out['sortino'] = np.nan

    # Max drawdown (sobre wealth)
    wealth = np.cumprod(1 + arr) if n>0 else np.array([])
    if wealth.size>0:
        peak = np.maximum.accumulate(wealth)
        drawdowns = (peak - wealth) / peak
        out['max_drawdown'] = drawdowns.max()
    else:
        out['max_drawdown'] = np.nan

    return out


def normalize_weights(s):
    s = s.fillna(0.0).astype(float)
    total = s.sum()
    if total == 0 or np.isnan(total):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s / total).fillna(0.0)

# Optimizadores clásicos (con restricciones 0..1 y suma 1)
def optimize_min_variance(returns_df):
    cov = returns_df.cov().values
    n = cov.shape[0]
    def fun(w): return w.dot(cov).dot(w)
    x0 = np.ones(n)/n
    bounds = [(0.0,1.0)] * n
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1.0},)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':10000})
    w = np.maximum(res.x, 0)
    if w.sum() > 0:
        w = w / w.sum()
    return pd.Series(w, index=returns_df.columns)

def optimize_max_sharpe(returns_df, rf_daily=0.0):
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    def neg_sharpe(w):
        port_ret = np.dot(mu, w)
        port_vol = np.sqrt(w.dot(cov).dot(w))
        if port_vol==0: return 1e9
        # annualize inside outside handled in metrics; for optimization it's fine
        return - (port_ret - rf_daily) / port_vol
    x0 = np.ones(n)/n
    bounds = [(0.0,1.0)]*n
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1.0},)
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
    w = np.maximum(res.x, 0)
    if w.sum()>0: w = w / w.sum()
    return pd.Series(w, index=returns_df.columns)

def optimize_max_return(returns_df):
    # asigna peso 1 al asset con mayor mean return in-sample
    mean_ret = (1 + returns_df).prod() - 1
    best = mean_ret.idxmax()
    s = pd.Series(0.0, index=returns_df.columns); s.loc[best] = 1.0
    return s

# Backtest OOS helper
def backtest_weights_on_returns(weights, returns_oos):
    w = weights.reindex(returns_oos.columns).fillna(0.0)
    if w.sum() <= 0: 
        return np.zeros(len(returns_oos))
    w = w / w.sum()
    port_ret = returns_oos.values.dot(w.values)
    return port_ret

# Montecarlo frontier quick (opcional)
def montecarlo_frontier(returns_df, n_sim=5000):
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    n = len(mu)
    res = []
    for i in range(n_sim):
        w = np.random.random(n)
        w = w / w.sum()
        r = np.dot(mu, w) * 252
        vol = np.sqrt(w.dot(cov).dot(w)) * math.sqrt(252)
        res.append((r, vol, w))
    df = pd.DataFrame([{'ret':r,'vol':v,'w':w} for r,v,w in res])
    return df

# Main render function
def render_optimization_module():
    st.title("Optimización de portafolio + Simulación")
    st.write("Optimiza pesos in-sample y evalúa en periodo OOS. Selecciona parámetros y haz run.")

    today_dt = pd.to_datetime(date.today())

    # UI inputs
    available = st.session_state.get('global_params', {}).get('available_tickers', CONFIG['DEFAULT_TICKERS'])
    tickers = st.multiselect("Select tickers:", options=available, default=available[:8])

    # fechas: start, opt_end (in-sample last day), test_end (OOS last day)
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("Start date", value=(today_dt - pd.Timedelta(days=365*2)).date())
    with col2:
        opt_end = st.date_input("Opt (in-sample) end date", value=(today_dt - pd.Timedelta(days=90)).date())
    with col3:
        test_end = st.date_input("Test (OOS) end date", value=today_dt.date())

    rf_pct = st.number_input("Tasa libre de riesgo anual (%)", value=9.0, min_value=0.0, max_value=100.0, step=0.1)
    # benchmark
    benchmark = st.text_input("Benchmark (ej. ^GSPC o ^MXX)", value='^MXX')

    rf_annual = rf_pct / 100.0
    trading_days = 252  # fijo

    # Validaciones básicas
    if pd.to_datetime(test_end) > today_dt:
        st.error("La fecha test_end no puede ser mayor a hoy.")
        return
    if pd.to_datetime(opt_end) >= pd.to_datetime(test_end):
        st.error("opt_end debe ser anterior a test_end y dejar espacio para OOS.")
        return
    min_oos_days = 20
    if (pd.to_datetime(test_end) - pd.to_datetime(opt_end)).days < min_oos_days:
        st.warning(f"Recomendado: al menos {min_oos_days} días OOS para evaluar. Ajusta las fechas si puedes.")

    if not tickers:
        st.info("Selecciona al menos un ticker.")
        return

    run = st.button("Ejecutar optimización y backtest")
    if not run:
        return

    # Descargar precios hasta test_end
    with st.spinner("Descargando precios..."):
        start_s = pd.to_datetime(start).strftime("%Y-%m-%d")
        test_end_s = pd.to_datetime(test_end).strftime("%Y-%m-%d")
        opt_end_s = pd.to_datetime(opt_end).strftime("%Y-%m-%d")
        prices = safe_download_prices(tickers, start=start_s, end=test_end_s, interval="1d")
    if prices.empty:
        st.error("No se descargaron precios. Revisa tickers/fechas.")
        return

    # Rendimientos diarios
    returns = prices.pct_change().dropna(how='all')
    if returns.empty:
        st.error("No hay rendimientos calculables con los datos descargados.")
        return

    # Definir in-sample (hasta opt_end) y OOS
    opt_end_dt = pd.to_datetime(opt_end_s)
    train_returns = returns.loc[:opt_end_dt].copy()
    test_start_idx = returns.index[returns.index > opt_end_dt]
    if len(test_start_idx)==0:
        st.error("No hay datos OOS posteriores a opt_end. Ajusta fechas.")
        return
    oos_returns = returns.loc[test_start_idx.min(): pd.to_datetime(test_end_s)].copy()

    st.write(f"In-sample rows: {len(train_returns)} — OOS rows: {len(oos_returns)}")

    # Si lookback (opcional) usar último año dentro del insample
    lookback_days = trading_days  # 1 año
    if len(train_returns) >= lookback_days:
        train_for_est = train_returns.iloc[-lookback_days:]
    else:
        train_for_est = train_returns.copy()

    # Filtrar activos con suficiente data
    min_obs = int(0.5 * len(train_for_est))
    valid_cols = [c for c in train_for_est.columns if train_for_est[c].dropna().shape[0] >= min_obs]
    if len(valid_cols) == 0:
        st.error("No hay tickers con suficientes observaciones en in-sample. Ajusta selección/fechas.")
        return
    train_for_est = train_for_est[valid_cols].copy()
    oos_returns = oos_returns[valid_cols].copy()

    # Estrategias a correr
    strategies = {}
    strategies['Equal'] = pd.Series(1.0/len(train_for_est.columns), index=train_for_est.columns)
    strategies['MinVar'] = optimize_min_variance(train_for_est)
    strategies['MaxSharpe'] = optimize_max_sharpe(train_for_est, rf_daily=rf_annual/252)
    strategies['MaxReturn'] = optimize_max_return(train_for_est)
    # Sortino/Sharpe via criteria (use optimizer on neg metrics) - we include SR as maxSharpe already; for Sortino use simple optimize on SOR criterion:
    def sor_obj(w, data):
        pr = np.dot(data.values, w)
        mean = pr.mean()
        downside = pr[pr < 0]
        if len(downside) > 1:
            dd_std = np.std(downside, ddof=0)
            return - (mean / dd_std)
        else:
            return 1e6
    def optimize_custom(objfn, data):
        n = data.shape[1]
        x0 = np.ones(n)/n
        bounds = [(0.0,1.0)]*n
        cons = ({'type':'eq','fun': lambda x: np.sum(x)-1.0},)
        res = minimize(lambda x: objfn(x, data), x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
        w = np.maximum(res.x, 0)
        if w.sum()>0: w = w / w.sum()
        return pd.Series(w, index=data.columns)
    try:
        strategies['Sortino'] = optimize_custom(sor_obj, train_for_est)
    except Exception:
        strategies['Sortino'] = strategies['Equal']

    # Normalizar y asegurar long-only
    for k in list(strategies.keys()):
        strategies[k] = normalize_weights(strategies[k]).reindex(train_for_est.columns).fillna(0.0)

    # Backtest OOS
    results = {}
    metrics = []
    for name, w in strategies.items():
        port_ret = backtest_weights_on_returns(w, oos_returns)
        results[name] = port_ret
        m = compute_metrics(port_ret, rf_annual=rf_annual)
        m['name'] = name
        metrics.append(m)

    
    # Descargar benchmark y alinear de forma segura con oos_returns.index
    try:
        bench_df = yf.download(benchmark, start=oos_returns.index.min().strftime("%Y-%m-%d"),
                            end=(oos_returns.index.max()+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            auto_adjust=True)["Close"]
        if isinstance(bench_df, pd.Series):
            bench_prices = bench_df
        else:
            bench_prices = bench_df.ffill().bfill()

        # Reindexar al índice OOS y forward-fill/backfill (NO poner ceros)
        bench_prices = bench_prices.reindex(oos_returns.index).ffill().bfill()

        # Calcular retornos diarios alineados con OOS (dropna del primero)
        bench_ret = bench_prices.pct_change().loc[oos_returns.index].fillna(0.0).values
        results['Benchmark'] = bench_ret
        bm = compute_metrics(bench_ret, rf_annual=rf_annual); bm['name']='Benchmark'; metrics.append(bm)
    except Exception as e:
        st.warning(f"No se pudo descargar/alinear benchmark {benchmark}: {e}")

    metrics_df = pd.DataFrame(metrics).set_index('name')
    # Mostrar métricas
    st.subheader("Resumen métricas (OOS)")
    st.dataframe(metrics_df.style.format({
        'ann_return': '{:.2%}', 'ann_vol': '{:.2%}', 'sharpe': '{:.3f}', 'sortino': '{:.3f}', 'max_drawdown': '{:.2%}', 'total_return': '{:.2%}', 'ann_return_geom': '{:.2%}'
    }))

    st.caption("""Notas: Las métricas presentadas reflejan el rendimiento fuera de muestra (OOS) de cada estrategia optimizada.  
               ann_return: retorno anualizado; ann_vol: volatilidad anualizada; sharpe: ratio de Sharpe; sortino: ratio de Sortino; max_drawdown: máxima caída desde un pico; total_return: retorno total en el periodo OOS; ann_return_geom: retorno anualizado geométrico.""")


    # Gráficos dark style y equity curves
    plt.style.use('dark_background')
    rc = {
        'axes.facecolor': '#222222', 'figure.facecolor':'#222222', 'axes.edgecolor': '#444444',
        'grid.color': '#333333', 'xtick.color': 'white', 'ytick.color': 'white',
    }
    plt.rcParams.update(rc)

    st.subheader("Simulación de estrategias - OOS (Rendimiento real)")
    fig, ax = plt.subplots(figsize=(14,6))
    from matplotlib.ticker import PercentFormatter
    for name, arr in results.items():
        cum_wealth = np.cumprod(1 + arr) - 1
        if name == 'Benchmark':
            ax.plot(oos_returns.index, cum_wealth, label=name, linewidth=2.2, linestyle='--', color='white')
        else:
            ax.plot(oos_returns.index, cum_wealth, label=name, linewidth=1.4)
    # Formatear eje Y como porcentaje (cum_wealth está en fracción, p.ej. 0.10 -> 10%)
    ax.axhline(0, color='red', linewidth=0.5, linestyle='--')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Equity curves (OOS)")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption("""Notas: la gráfica muestra la evolución del capital (equity curve) de cada estrategia durante el período fuera de muestra. 
               En otras palabras, refleja el rendimiento real que habrían tenido las estrategias optimizadas aplicadas a datos no vistos durante la optimización.  
               El rendimiento real puede diferir de las expectativas in-sample debido a la variabilidad del mercado y otros factores.  
               Las estrategias optimizadas no garantizan rendimientos futuros.""")


    # Pesos por estrategia (tabla)
    st.subheader("Pesos por estrategia (in-sample)")
    weights_df = pd.DataFrame({k: v for k,v in strategies.items()})
    st.dataframe(weights_df.style.format("{:.2%}"))

    #graficos de pastel de pesos
    n = len(strategies)
    cols = st.columns(3)  # 3 por fila
    i = 0
    for name, w in strategies.items():
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        # evitar wedges tiny y etiquetas largas; filtrar zeros
        w_nonzero = w[w > 0]
        if w_nonzero.empty:
            ax.text(0.5,0.5,"No weights", ha='center')
        else:
            ax.pie(w_nonzero.values, labels=w_nonzero.index, autopct=lambda p: f'{p:.1f}%', startangle=90, textprops={'fontsize':8})
            ax.set_title(name, fontsize=10)
        ax.axis('equal')
        with cols[i % 3]:
            st.pyplot(fig)
        plt.close(fig)
        i += 1


    # Frontier via MonteCarlo (visual)
    st.subheader("Frontera Eficiente (aproximada) - Método Monte Carlo")
    mc = montecarlo_frontier(train_for_est, n_sim=1500)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.scatter(mc['vol'], mc['ret'], s=8, alpha=0.3)
    # plot strategies
    for name, w in strategies.items():
        re = np.dot(train_for_est.mean().values, w) * 252
        vo = np.sqrt(w.values.dot(train_for_est.cov().values).dot(w.values)) * math.sqrt(252)
        ax2.scatter(vo, re, s=60, label=name)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("Volatility (ann.)")
    ax2.set_ylabel("Return (ann.)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize='small')
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption("""Notas: la frontera Monte Carlo se calcula con datos In-Sample; las métricas OOS muestran el comportamiento real fuera de muestra.             
    En otras palabras, es la frontera eficiente teórica. El comportamiento real está en las graficas OOS (out of sample).  
    Las estrategias optimizadas no garantizan rendimientos futuros.""")


    # Descargar pesos
    csv_out = "optimized_weights_summary.csv"
    df_for_export = weights_df.reset_index().rename(columns={'index':'Ticker'})
    df_for_export.to_csv(csv_out, index=False)
    st.download_button("Descargar pesos (CSV)", df_for_export.to_csv(index=False), file_name=csv_out, mime="text/csv")

    st.success("Optimización y backtest completados.")





# -------------------- Entrypoint --------------------
if __name__ == '__main__':
    main()