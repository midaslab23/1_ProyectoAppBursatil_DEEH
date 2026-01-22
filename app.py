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
        st.info("Módulo de optimización y simulación todavía no implementado")
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

# -------------------- Entrypoint --------------------
if __name__ == '__main__':
    main()