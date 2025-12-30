import pandas as pd
import folium
import streamlit as st
import numpy as np
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
import altair as alt

#
# ASETUKSET
#

LOC_DATA_PATH = "https://raw.githubusercontent.com/SanteriMikkola/Fysiikan_loppuprojekti/main/Location.csv"
ACC_DATA_PATH = "https://raw.githubusercontent.com/SanteriMikkola/Fysiikan_loppuprojekti/main/Linear Acceleration.csv"

# Leikataan pois alun seisoskelu ja siirretään aika alkamaan nollasta.
START_TIME = 100.0

# Segmentit ja värit karttaan
SEGMENTS = [
    ("hidas kävely", 0, 300, "red"),
    ("reipas kävely", 300, 570, "blue"),
    ("hölkkä", 570, None, "green"),
]

# Suodatin
FILTER_ORDER = 3
LOWPASS_CUTOFF_HZ = 4.0

# Piirto
MAX_POINTS_TIMEPLOT = 1800
PSD_MAX_HZ = 20
PSD_YMAX = 12000


#
# SIVUN ASETTELU + TITLE
#

st.set_page_config(
    page_title="Liikunta suorite",
    page_icon="🏃",
    layout="wide"
)

st.title("Liikunta suorite analyysi")


#
# FUNKTIOT
#

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    return c * r


def butter_lowpass_filter(x, fs, cutoff_hz=4.0, order=3):
    
    nyq = fs / 2
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, x)


@st.cache_data
def steps_fft_calc(t, signal, fmin=0.7, fmax=4.0):
    # Muutetaan numpy-taulukoiksi
    t = np.asarray(t)
    x = np.asarray(signal)
        
    dt = np.median(np.diff(t))

    N = len(x)

    # Reaalinen FFT
    freqs = np.fft.rfftfreq(N, d=dt)
    X = np.fft.rfft(x)
    power = np.abs(X) ** 2  # teho

    # Rajataan taajuudsalue
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0, np.nan

    idx_peak = np.argmax(power[band])
    f_peak = freqs[band][idx_peak]

    T = t[-1] - t[0]
    steps = f_peak * T

    return int(round(steps)), float(f_peak)


@st.cache_data
def psd(t, x):

    # Muutetaan numpy-taulukoiksi
    t = np.asarray(t)
    x = np.asarray(x)
    
    # Näytteenottoväli
    dt = np.median(np.diff(t))
    N = len(x)

    # Fourier-muunnos
    X = np.fft.fft(x, N)
    
    # Tehospektri
    psd = (X * np.conj(X)) / N

    # Taajuudet
    freq = np.fft.fftfreq(N, d=dt)

    # Vain positiiviset taajuudet
    L = np.arange(1, N // 2)

    df = pd.DataFrame({
        "freq_hz": freq[L],
        "power": psd[L].real
    })

    return df.sort_values("freq_hz").reset_index(drop=True)


def downsample(t, y, max_points=2000):
    
    n = len(t)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    return t[idx], y[idx]


# 
# DATA: LUKU + LEIKKAUS + AJAN NOLLAUS + NÄYTTEENOTTOTAAJUUS
#

df_loc = pd.read_csv(LOC_DATA_PATH)
df_acc = pd.read_csv(ACC_DATA_PATH)

# Leikataan alku pois molemmista datoista
df_loc = df_loc[df_loc["Time (s)"] >= START_TIME].copy().reset_index(drop=True)
df_acc = df_acc[df_acc["Time (s)"] >= START_TIME].copy().reset_index(drop=True)

# Siirretään aika alkamaan 0:sta
df_loc["Time (s)"] -= START_TIME
df_acc["Time (s)"] -= START_TIME

t_acc = df_acc["Time (s)"].to_numpy()
z_acc = df_acc["Linear Acceleration z (m/s^2)"].to_numpy()

# Näytteenottotaajuus
dt = np.median(np.diff(t_acc))
fs = 1.0 / dt if dt > 0 else np.nan


#
# ASKELET FFT:LLÄ + SUODATUKSELLA
#

# FFT-askeleet segmenteittäin
total_steps_fft = 0
for name, t0, t1, _color in SEGMENTS:
    t1 = t_acc[-1] if t1 is None else t1
    mask = (t_acc >= t0) & (t_acc < t1)
    steps_seg, f_peak = steps_fft_calc(t_acc[mask], z_acc[mask], fmin=0.7, fmax=4.0)
    total_steps_fft += steps_seg

# Suodatus
z_dc = z_acc - np.mean(z_acc)
z_filt = butter_lowpass_filter(z_dc, fs, cutoff_hz=LOWPASS_CUTOFF_HZ, order=FILTER_ORDER)

steps_filt = 0
for i in range(len(z_filt) - 1):
    if z_filt[i] * z_filt[i + 1] < 0:
        steps_filt += 0.5
steps_filt = int(round(steps_filt))


#
# MATKA + NOPEUS + ASKELPITUUS
#

df_loc = df_loc.sort_values("Time (s)").reset_index(drop=True)

df_loc["Distance_calc"] = 0.0
df_loc["time_diff"] = 0.0

for i in range(len(df_loc) - 1):
    lon1 = df_loc["Longitude (°)"].iloc[i]
    lon2 = df_loc["Longitude (°)"].iloc[i + 1]
    lat1 = df_loc["Latitude (°)"].iloc[i]
    lat2 = df_loc["Latitude (°)"].iloc[i + 1]

    df_loc.loc[i + 1, "Distance_calc"] = haversine(lon1, lat1, lon2, lat2) * 1000.0
    df_loc.loc[i + 1, "time_diff"] = df_loc["Time (s)"].iloc[i + 1] - df_loc["Time (s)"].iloc[i]

# Kokonaismatka
df_loc["total_dist"] = df_loc["Distance_calc"].cumsum()
total_km = float(df_loc["total_dist"].iloc[-1] / 1000.0)

# Kesto ja keskinopeus
duration_s = float(df_loc["Time (s)"].iloc[-1] - df_loc["Time (s)"].iloc[0]) if len(df_loc) > 1 else np.nan
avg_speed_ms = float(df_loc["total_dist"].iloc[-1] / duration_s) if duration_s and duration_s > 0 else np.nan

# Askelpituus (cm)
step_length_cm = (total_km * 100_000.0 / steps_filt) if steps_filt > 0 else np.nan


#
# METRIKAT
#

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Askelten määrä (FFT)", f"{int(total_steps_fft)} askelta")
col2.metric("Askelten määrä (suodatus)", f"{int(steps_filt)} askelta")
col3.metric("Kokonaismatka", f"{total_km:.2f} km")
col4.metric("Keskinopeus", f"{avg_speed_ms*3.6:.2f} km/h" if np.isfinite(avg_speed_ms) else "—")
col5.metric("Askelpituus", f"{int(step_length_cm)} cm" if np.isfinite(step_length_cm) else "—")


#
# SUODATETTU KIIHTYVYYS DATA
#

with st.expander("Suodatettu kiihtyvyysdata", expanded=True):
    st.subheader("Suodatetun kiihtyvyysdatan z-komponentti")

    # Downsample -> nopeampi piirtäminen "ison" data määrän vuoksi
    t_ds, z_ds = downsample(t_acc, z_filt, max_points=MAX_POINTS_TIMEPLOT)

    df_plot = pd.DataFrame({
        "t_min": t_ds / 60.0,
        "filt_az": z_ds
    })

    tmax = float(df_plot["t_min"].max())
    ticks = np.arange(0, tmax + 0.5, 0.5)

    chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=alt.X(
                "t_min:Q",
                title="Aika [min]",
                axis=alt.Axis(values=ticks.tolist(), format=".1f")
            ),
            y=alt.Y("filt_az:Q", title="suodatettu kiihtyvyys z (m/s²)"),
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(chart, width='stretch')


#
# TEHOSPEKTRI
#

with st.expander("Tehospektri", expanded=False):
    st.subheader("Tehospektri")

    df_psd = psd(t_acc, z_acc)
    df_show = df_psd[(df_psd["freq_hz"] >= 0) & (df_psd["freq_hz"] <= PSD_MAX_HZ)].copy()

    xticks = list(range(0, PSD_MAX_HZ + 1, 2))

    chart = (
        alt.Chart(df_show)
        .mark_line()
        .encode(
            x=alt.X(
                "freq_hz:Q",
                title="Taajuus [Hz]",
                scale=alt.Scale(domain=[0, PSD_MAX_HZ]),
                axis=alt.Axis(values=xticks)
            ),
            y=alt.Y(
                "power:Q",
                title="Teho",
                scale=alt.Scale(domain=[0, PSD_YMAX])
            ),
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(chart, width='stretch')


#
# KARTTA
#

with st.expander("Kartta", expanded=False):
    st.subheader("Reitti")

    # Keskipiste kartalle
    start_lat = float(df_loc["Latitude (°)"].mean())
    start_lon = float(df_loc["Longitude (°)"].mean())

    m = folium.Map(location=[start_lat, start_lon], zoom_start=15)
    t_end = float(df_loc["Time (s)"].iloc[-1]) if len(df_loc) else 0.0

    for name, t0, t1, color in SEGMENTS:
        t1 = t_end if t1 is None else t1

        seg = df_loc[(df_loc["Time (s)"] >= t0) & (df_loc["Time (s)"] < t1)]
        if len(seg) < 2:
            continue

        coords = seg[["Latitude (°)", "Longitude (°)"]].values.tolist()

        folium.PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.9,
            tooltip=name,
        ).add_to(m)

        # aloitus- ja lopetusmerkit
        folium.CircleMarker(
            coords[0], radius=4, color=color, fill=True, fill_opacity=1,
            tooltip=f"{name}: start"
        ).add_to(m)

        folium.CircleMarker(
            coords[-1], radius=4, color=color, fill=True, fill_opacity=1,
            tooltip=f"{name}: end"
        ).add_to(m)

    st_folium(m, width=None, height=650)
