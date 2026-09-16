import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from metrics.metrics import compute_all
from validation.validation import validate_image
from rating.rating import build_rating, pairwise_diff, CRITERIA, DEFAULT_WEIGHTS, REFERENCE

st.set_page_config(page_title="QualityChecker", page_icon="", layout="wide")
st.title("Рейтинг камер по качеству фото")

with st.sidebar:
    st.header("Настройки")
    method = st.radio(
        "Метод агрегации",
        ["weighted", "topsis"],
        format_func=lambda x: "Взвешенная сумма" if x == "weighted" else "TOPSIS",
    )
    use_ref = st.checkbox(
        "Абсолютная шкала (относительно эталона)",
        value=False,
        help="Если выключено — нормировка min–max по выборке. "
             "Если включено — балл не зависит от того, кто ещё загружен.",
    )
    st.caption("Веса показателей")
    weights = {}
    for k, v in DEFAULT_WEIGHTS.items():
        weights[k] = st.slider(k, 0.0, 0.5, v, 0.01)

with st.expander("Как пользоваться", expanded=False):
    st.markdown("""
    - Загружайте **оригиналы**, не через мессенджеры.
    - Снимайте **одну и ту же сцену** разными телефонами.
    - Без цифрового зума, одинаковое освещение.
    - Нужны и текстуры, и ровные участки.
    - Минимум 480 px по короткой стороне.
    """)

uploaded = st.file_uploader(
    "Загрузите фото (можно несколько)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded:
    rows = []
    st.subheader("Результаты по каждому фото")

    for file in uploaded:
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"{file.name}: не удалось прочитать")
            continue

        ok, msg, warnings = validate_image(img, file.name)
        if not ok:
            st.error(f"**{file.name}** — отклонено\n\n{msg}")
            continue

        m = compute_all(img)

        st.markdown(f"### {file.name}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"{m['original_size'][0]}×{m['original_size'][1]}",
                use_column_width=True,
            )
        with col2:
            st.write({
                "Резкость":            round(m["sharpness"], 3),
                "Яркостный шум":       round(m["noise"], 2),
                "Цветовой шум":        round(m["chroma_noise"], 2),
                "Динамический диапазон": round(m["dynamic_range"], 3),
                "Экспозиция":          round(m["exposure"], 3),
                "Контраст":            round(m["contrast"], 3),
                "Насыщенность":        round(m["colorfulness"], 2),
                "Света (0..1, больше=лучше)":  round(m["highlight_rec"], 3),
                "Тени  (0..1, больше=лучше)":  round(m["shadow_detail"], 3),
            })
            if m["was_resized"]:
                st.caption(
                    f"Анализировалось в {m['analyzed_size'][0]}×{m['analyzed_size'][1]}"
                )

        for wmsg in warnings:
            st.warning(f"{wmsg}")

        camera = st.text_input(
            "Название камеры",
            value=file.name.rsplit(".", 1)[0],
            key=f"cam_{file.name}",
        )
        rows.append({"camera": camera, **m})

    if rows:
        if len(rows) < 2:
            st.info("Для рейтинга нужно минимум 2 камеры. Пока балл условный.")

        rated = build_rating(rows, weights=weights, method=method, use_reference=use_ref)

        st.subheader("Итоговый рейтинг")
        df = pd.DataFrame(rated)
        show_cols = ["rank", "camera", "score"] + list(CRITERIA.keys())
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=df["camera"], y=df["score"],
            text=df["score"], textposition="outside",
            marker_color="#4C78A8",
        ))
        fig.update_layout(height=350, yaxis_title="Балл", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Профиль по показателям")
        keys = list(CRITERIA.keys())
        fig_radar = go.Figure()
        for _, row in df.iterrows():
            vals = []
            for k in keys:
                v = row[k]
                ref = REFERENCE[k]
                ratio = (v / ref) if CRITERIA[k] else (ref / max(v, 1e-6))
                vals.append(min(ratio, 1.5))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=keys + [keys[0]],
                name=row["camera"],
                fill="toself", opacity=0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        if len(rated) >= 2:
            st.subheader("Попарное сравнение")
            diffs = pairwise_diff(rated)
            st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)
            st.caption("Δ > 0 — первая камера лучше по показателю.")

        st.subheader("Экспорт")
        csv = df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV", csv, "camera_rank.csv", "text/csv")