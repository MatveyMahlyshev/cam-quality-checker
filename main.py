import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from metrics import compute_all
from validators import DefaultImageValidator
from rating import build_rating, pairwise_diff, CRITERIA, DEFAULT_WEIGHTS


st.set_page_config(page_title="QualityChecker", page_icon="📷", layout="wide")
st.title("Рейтинг камер по качеству фото")

validator = DefaultImageValidator()

WEIGHT_LABELS = {
    "sharpness":     "Резкость",
    "noise":         "Яркостный шум",
    "dynamic_range": "Динамический диапазон",
    "exposure":      "Экспозиция",
    "contrast":      "Контраст",
    "colorfulness":  "Насыщенность",
    "chroma_noise":  "Цветовой шум",
    "highlight_rec": "Сохранность светов",
    "shadow_detail": "Сохранность теней",
}

DISPLAY_LABELS = {
    "rank":          "Место",
    "camera":        "Камера",
    "score":         "Балл",
    "sharpness":     "Резкость",
    "noise":         "Яркостный шум",
    "dynamic_range": "Динамический диапазон",
    "exposure":      "Экспозиция",
    "contrast":      "Контраст",
    "colorfulness":  "Насыщенность",
    "chroma_noise":  "Цветовой шум",
    "highlight_rec": "Сохранность светов",
    "shadow_detail": "Сохранность теней",
}

METRIC_HELP = {
    "sharpness":     "Детализация изображения. Чем выше — тем резче фото.",
    "noise":         "Яркостный шум в однородных зонах. Чем ниже — тем чище кадр.",
    "dynamic_range": "Сколько ступеней яркости передаёт камера. Чем выше — тем шире диапазон.",
    "exposure":      "Насколько средняя яркость близка к середине. Чем выше — тем точнее экспозиция.",
    "contrast":      "Разброс яркости. Чем выше — тем контрастнее кадр.",
    "colorfulness":  "Насыщенность цвета. Чем выше — тем «сочнее» фото.",
    "chroma_noise":  "Цветовой шум в однородных зонах. Чем ниже — тем чище цвет.",
    "highlight_rec": "Сохранность текстуры в светах. Чем выше — тем лучше камера держит яркие участки.",
    "shadow_detail": "Сохранность текстуры в тенях. Чем выше — тем лучше проработаны тёмные участки.",
}

METHOD_HELP = {
    "weighted": (
        "Взвешенная сумма. Каждый показатель нормируется в 0–100 и складывается с весом. "
        "Прозрачно, видно вклад каждой метрики. Балл зависит от весов."
    ),
    "topsis": (
        "TOPSIS. Строится идеальная и антиидеальная камера, считается расстояние до них. "
        "Балл — близость к идеалу. Даёт крайние значения, когда одна камера доминирует."
    ),
}

SCALE_HELP = (
    "Выключено — относительная шкала: min–max по выборке. Лучшая камера 100, худшая 0. "
    "Балл зависит от состава выборки.\n\n"
    "Включено — абсолютная шкала: сравнение с эталоном. 100 = уровень эталона, "
    "балл не зависит от того, кто ещё загружен."
)


with st.sidebar:
    st.header("Настройки")

    method = st.radio(
        "Метод агрегации",
        ["weighted", "topsis"],
        format_func=lambda x: "Взвешенная сумма" if x == "weighted" else "TOPSIS",
        help=METHOD_HELP["weighted"] + "\n\n— — —\n\n" + METHOD_HELP["topsis"],
    )

    use_ref = st.checkbox(
        "Абсолютная шкала (относительно эталона)",
        value=False,
        help=SCALE_HELP,
    )

    st.caption("Веса показателей")
    weights = {}
    for k, v in DEFAULT_WEIGHTS.items():
        weights[k] = st.slider(
            WEIGHT_LABELS.get(k, k),
            0.0, 0.5, v, 0.01,
            help=METRIC_HELP[k],
        )


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


def dedupe_names(rows):
    seen = {}
    for r in rows:
        name = r["camera"]
        if name in seen:
            seen[name] += 1
            r["camera"] = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
    return rows


if uploaded:
    rows = []
    st.subheader("Результаты по каждому фото")

    for file in uploaded:
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"{file.name}: не удалось прочитать")
            continue

        result = validator.validate(img, file.name)
        if not result:
            st.error(f"**{file.name}** — отклонено\n\n{result.message}")
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

        for wmsg in result.warnings:
            st.warning(f"{wmsg}")

        camera = st.text_input(
            "Название камеры",
            value=file.name.rsplit(".", 1)[0],
            key=f"cam_{file.name}",
        )
        rows.append({"camera": camera, **m})

    if rows:
        rows = dedupe_names(rows)

        if len(rows) < 2:
            st.info("Для рейтинга нужно минимум 2 камеры. Пока балл условный.")

        rated = build_rating(rows, weights=weights, method=method, use_reference=use_ref)

        st.subheader("Итоговый рейтинг")

        df = pd.DataFrame(rated)
        show_cols = ["rank", "camera", "score"] + list(CRITERIA.keys())
        df_show = df[show_cols].rename(columns=DISPLAY_LABELS)

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            help=(
                "Место — позиция камеры в рейтинге (1 — лучшая). "
                "Балл — итоговая оценка после агрегации. "
                "Остальные столбцы — значения метрик в исходных единицах."
            ),
        )

        df_plot = df.reset_index(drop=True).copy()
        df_plot["score_num"] = pd.to_numeric(df_plot["score"], errors="coerce")

        fig = go.Figure(go.Bar(
            x=df_plot["camera"].astype(str).tolist(),
            y=df_plot["score_num"].tolist(),
            text=[f"{s:.2f}" if pd.notna(s) else "nan" for s in df_plot["score_num"]],
            textposition="outside",
            marker_color="#4C78A8",
        ))
        fig.update_layout(height=350, yaxis_title="Балл", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Столбцы — итоговые баллы камер. Чем выше, тем лучше результат."
            if method == "weighted"
            else "TOPSIS: 100 — близко к идеалу по всем показателям, 0 — антиидеал."
        )

        st.subheader("Профиль по показателям")
        keys = list(CRITERIA.keys())

        ranges = {k: (df[k].min(), df[k].max()) for k in keys}
        fig_radar = go.Figure()
        for _, row in df.iterrows():
            vals = []
            for k in keys:
                lo, hi = ranges[k]
                if hi - lo < 1e-9:
                    vals.append(0.5)
                    continue
                v = row[k]
                norm = (v - lo) / (hi - lo)
                if not CRITERIA[k]:
                    norm = 1 - norm
                vals.append(float(norm))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=[WEIGHT_LABELS[k] for k in keys] + [WEIGHT_LABELS[keys[0]]],
                name=str(row["camera"]),
                fill="toself",
                opacity=0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption(
            "1.0 — лучший результат по показателю внутри выборки, 0.0 — худший. "
            "Оси нормированы по min–max, форма профиля показывает сильные и слабые стороны камер."
        )

        if len(rated) >= 2:
            st.subheader("Попарное сравнение")

            diffs = pairwise_diff(rated)
            df_diffs = pd.DataFrame(diffs)

            DIFF_LABELS = {
                "A":                "Камера A",
                "B":                "Камера B",
                "Δ score":          "Δ Балл",
                "Δ sharpness":      "Δ Резкость",
                "Δ noise":          "Δ Яркостный шум",
                "Δ dynamic_range":  "Δ Динамический диапазон",
                "Δ exposure":       "Δ Экспозиция",
                "Δ contrast":       "Δ Контраст",
                "Δ colorfulness":   "Δ Насыщенность",
                "Δ chroma_noise":   "Δ Цветовой шум",
                "Δ highlight_rec":  "Δ Сохранность светов",
                "Δ shadow_detail":  "Δ Сохранность теней",
            }

            df_diffs = df_diffs.rename(columns=DIFF_LABELS)

            st.dataframe(
                df_diffs,
                use_container_width=True,
                hide_index=True,
                help=(
                    "Каждая строка — сравнение двух камер (A и B). "
                    "Δ > 0 — камера A лучше по показателю, Δ < 0 — лучше камера B."
                ),
            )

        st.subheader("Экспорт")
        csv = df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV", csv, "camera_rank.csv", "text/csv")
        st.caption("CSV содержит все показатели по каждой камере в исходных единицах.")