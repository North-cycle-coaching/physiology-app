import streamlit as st
from fitparse import FitFile
import pandas as pd
import numpy as np
import altair as alt

st.title("Physiology Match Report")

uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

if uploaded_file is not None:
    fitfile = FitFile(uploaded_file)

    max_hr = st.number_input("Enter Max Heart Rate (bpm)", min_value=100, max_value=250)
    weight_kg = st.number_input("Enter Weight (kg)", min_value=30.0, max_value=150.0)
    cp = st.number_input("Enter Critical Power (W)", min_value=100, max_value=500)

    if st.button("Analyse"):
        threshold_hr = 0.9 * max_hr

        data_records = []
        for record in fitfile.get_messages('record'):
            data_point = {}
            for data in record:
                if data.name == 'timestamp':
                    data_point['timestamp'] = data.value
                elif data.name == 'heart_rate':
                    data_point['heart_rate'] = data.value
                elif data.name == 'power':
                    data_point['power'] = data.value
                elif data.name == 'altitude':
                    data_point['altitude'] = data.value
                elif data.name == 'cadence':
                    data_point['cadence'] = data.value
            if all(k in data_point for k in ['timestamp', 'heart_rate', 'power', 'altitude']):
                data_records.append(data_point)

        data_df = pd.DataFrame(data_records)
        data_df.dropna(inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        data_df['kj'] = data_df['power'].cumsum() / 1000
        data_df['kj_kg'] = data_df['kj'] / weight_kg
        data_df['above_90pct'] = data_df['heart_rate'] >= threshold_hr

        ride_start = data_df['timestamp'].iloc[0]

        intervals = []
        in_interval = False
        start_time = None

        for idx, row in data_df.iterrows():
            if row['above_90pct'] and not in_interval:
                in_interval = True
                start_time = row['timestamp']
            elif not row['above_90pct'] and in_interval:
                in_interval = False
                end_time = row['timestamp']
                duration = (end_time - start_time).total_seconds()
                if duration >= 30:
                    intervals.append((start_time, end_time))

        if in_interval:
            end_time = data_df['timestamp'].iloc[-1]
            duration = (end_time - start_time).total_seconds()
            if duration >= 30:
                intervals.append((start_time, end_time))

        def classify_terrain(elevation_change):
            if elevation_change > 5:
                return 'Climb'
            elif elevation_change < -5:
                return 'Descent'
            else:
                return 'Flat/Rolling'

        st.subheader(f"Total Matches Detected: {len(intervals)}")

        for i, (start, end) in enumerate(intervals, 1):
            interval_mask = (data_df['timestamp'] >= start) & (data_df['timestamp'] <= end)
            interval_data = data_df.loc[interval_mask]

            interval_power = interval_data['power']
            interval_avg_power = interval_power.mean()
            rolling_30 = interval_power.rolling(window=30, min_periods=1).mean()
            interval_np = (rolling_30 ** 4).mean() ** 0.25
            intensity_vs_cp = interval_np / cp * 100

            interval_hr = interval_data['heart_rate'].mean()
            interval_elev_change = interval_data['altitude'].iloc[-1] - interval_data['altitude'].iloc[0]
            terrain_type = classify_terrain(interval_elev_change)

            kj_accum = data_df.loc[data_df['timestamp'] <= start, 'kj'].iloc[-1]
            kjkg_accum = data_df.loc[data_df['timestamp'] <= start, 'kj_kg'].iloc[-1]

            before_90s_start = start - pd.Timedelta(seconds=90)
            before_90s = data_df[(data_df['timestamp'] >= before_90s_start) & (data_df['timestamp'] < start)]['power']
            peak_10s = before_90s.rolling(window=10).mean().max()

            after_60s_end = end + pd.Timedelta(seconds=60)
            post_data = data_df[(data_df['timestamp'] > end) & (data_df['timestamp'] <= after_60s_end)]
            after_60s_power = post_data['power'].mean()
            after_60s_hr = post_data['heart_rate'].iloc[-1] if not post_data.empty else interval_data['heart_rate'].iloc[-1]
            hr_drop = interval_data['heart_rate'].iloc[-1] - after_60s_hr

            if after_60s_power < 0.5 * cp and hr_drop > 10:
                recovery_type = "passive recovery"
            elif after_60s_power < 0.8 * cp:
                recovery_type = "active recovery"
            else:
                recovery_type = "continued effort"

            before_6m_start = before_90s_start - pd.Timedelta(seconds=360)
            before_6m = data_df[(data_df['timestamp'] >= before_6m_start) & (data_df['timestamp'] < before_90s_start)]['power'].mean()

            efficiency = interval_avg_power / interval_hr
            start_offset = (start - ride_start).total_seconds()
            end_offset = (end - ride_start).total_seconds()
            duration = (end - start).total_seconds()

            first_half = interval_power.iloc[:len(interval_power)//2].mean()
            second_half = interval_power.iloc[len(interval_power)//2:].mean()
            match_std = interval_power.std()

            if second_half > first_half + 20:
                match_shape = "build effort"
            elif first_half > second_half + 20:
                match_shape = "fade effort"
            elif match_std > 50:
                match_shape = "variable effort"
            else:
                match_shape = "steady state"

            with st.container():
                st.markdown(f"### 🔥 Match #{i}")
                st.caption(f"**Time in file**: {int(start_offset // 60):02d}:{int(start_offset % 60):02d} → {int(end_offset // 60):02d}:{int(end_offset % 60):02d}")
                st.caption(f"**Duration**: {int(duration)} sec | **Terrain**: {terrain_type}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Power", f"{interval_avg_power:.0f} W")
                col2.metric("Normalised Power", f"{interval_np:.0f} W")
                col3.metric("Intensity", f"{intensity_vs_cp:.0f}% of CP")

                col4, col5, col6 = st.columns(3)
                col4.metric("Avg HR", f"{interval_hr:.0f} bpm")
                col5.metric("kJ (at entry)", f"{kj_accum:.1f}")
                col6.metric("kJ/kg", f"{kjkg_accum:.2f}")

                # Match chart (now styled with cadence too if available)
                match_plot_df = interval_data[['timestamp', 'power', 'heart_rate']].copy()
                if 'cadence' in interval_data.columns:
                    match_plot_df['cadence'] = interval_data['cadence']

                match_plot_df = match_plot_df.melt('timestamp', var_name='Metric', value_name='Value')

                match_chart = alt.Chart(match_plot_df).mark_line().encode(
                    x='timestamp:T',
                    y='Value:Q',
                    color=alt.Color('Metric:N', scale=alt.Scale(domain=['power', 'heart_rate', 'cadence'],
                                                                 range=['#1f77b4', '#d62728', '#2ca02c']))
                ).properties(width=700, height=300)

                st.altair_chart(match_chart, use_container_width=True)

                # === Advanced Insight Section ===
                st.markdown("**🧭 Pre-Match Effort Breakdown**")

                pre_interval = data_df[(data_df['timestamp'] >= before_90s_start) & (data_df['timestamp'] < start)].copy()
                pre_interval['seconds'] = (pre_interval['timestamp'] - pre_interval['timestamp'].iloc[0]).dt.total_seconds()

                power_start = pre_interval['power'].iloc[0]
                hr_start = pre_interval['heart_rate'].iloc[0]
                cad_start = pre_interval['cadence'].iloc[0] if 'cadence' in pre_interval.columns else None

                power_end = pre_interval['power'].iloc[-1]
                hr_end = pre_interval['heart_rate'].iloc[-1]
                cad_end = pre_interval['cadence'].iloc[-1] if 'cadence' in pre_interval.columns else None

                power_peak = pre_interval['power'].max()
                hr_peak = pre_interval['heart_rate'].max()
                cad_peak = pre_interval['cadence'].max() if 'cadence' in pre_interval.columns else None

                surge_time = pre_interval['power'].rolling(5).mean().idxmax()
                surge_sec = int(pre_interval.loc[surge_time, 'seconds']) if not np.isnan(surge_time) else None

                hr_cross_90 = pre_interval[pre_interval['heart_rate'] >= threshold_hr]
                hr_cross_sec = int(hr_cross_90['seconds'].iloc[0]) if not hr_cross_90.empty else None

                cadence_comment = ""
                if cad_start and cad_peak:
                    if cad_peak - cad_start > 15:
                        cadence_comment = f"Cadence jumped from {int(cad_start)} to {int(cad_peak)} rpm, suggesting a standing effort or attack."
                    elif cad_peak - cad_start < -10:
                        cadence_comment = f"Cadence dropped sharply, indicating a seated grind or fatigue."
                    else:
                        cadence_comment = f"Cadence remained steady around {int(cad_start)} rpm."

                pre_commentary = (
                    f"In the 90 seconds before this match: HR was {int(hr_start)} bpm. "
                    f"Power was {'steady' if abs(power_end - power_start) < 20 else 'variable'} early, then surged at -{90 - surge_sec}s to {int(power_peak)}W. "
                    f"HR rose to {int(hr_end)} bpm before entry. "
                    f"{'HR crossed 90% max at -' + str(90 - hr_cross_sec) + 's. ' if hr_cross_sec else ''}"
                    f"{cadence_comment}"
                )

                st.markdown(pre_commentary)

                vis_df = pre_interval[['seconds', 'power', 'heart_rate']].copy()
                if 'cadence' in pre_interval.columns:
                    vis_df['cadence'] = pre_interval['cadence']

                vis_df = vis_df.melt(id_vars='seconds', var_name='Metric', value_name='Value')

                pre_chart = alt.Chart(vis_df).mark_line().encode(
                    x='seconds:Q',
                    y='Value:Q',
                    color=alt.Color('Metric:N', scale=alt.Scale(domain=['power', 'heart_rate', 'cadence'],
                                                                range=['#1f77b4', '#d62728', '#2ca02c']))
                ).properties(width=700, height=300)

                st.altair_chart(pre_chart, use_container_width=True)

                st.markdown("**💥 Match Shape Insight**")
                st.markdown(
                    f"This match followed a {match_shape}. "
                    f"Recovery was **{recovery_type}** — HR dropped {hr_drop:.0f} bpm, power {after_60s_power:.0f}W."
                )

                st.divider()


