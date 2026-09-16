import cv2
import numpy as np
import pandas as pd
import streamlit as st

from metrics import compute_all
from validation import validate_image
from rating import build_rating

st.set_page_config(page_title="CameraRank", page_icon="📷", layout="wide")
st.title("📷 CameraRank — рейтинг камер по качеству фото")

with st.expander("ℹ️ Как пользоваться (прочитайте перед загрузкой)", expanded=False):
    st.markdown("""
    **Чтобы рейтинг был честным:**
    - Загружайте **оригиналы** (не через мессенджеры — они сжимают).
    - Снимайте **одну и ту же сцену** разными телефонами.
    - Освещение — одинаковое, **без цифрового зума**.
    - В кадре нужны и **текстуры** (листва, кирпич, текст), и **ровные участки** (стена, небо).

    **Минимум:** 480 px по короткой стороне.
    **Рекомендуется:** 1920 px по длинной стороне и выше.

    Форматы: JPEG, PNG.
    """)

uploaded = st.file_uploader(
    "Загрузите фото (можно несколько)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded:
    st.subheader("Результаты по каждому фото")
    rows = []

    for file in uploaded:
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"❌ {file.name}: не удалось прочитать файл")
            continue

        ok, msg, warnings = validate_image(img, file.name)

        if not ok:
            st.error(f"❌ **{file.name}** — отклонено\n\n{msg}")
            continue

        metrics = compute_all(img)

        st.markdown(f"### ✅ {file.name}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"{metrics['original_size'][0]}×{metrics['original_size'][1]}",
                use_column_width=True,
            )

        with col2:
            st.write({
                "Резкость": round(metrics["sharpness"], 4),
                "Шум (меньше = лучше)": round(metrics["noise"], 2),
                "Динамический диапазон": round(metrics["dynamic_range"], 3),
                "Экспозиция (0..1)": round(metrics["exposure"], 3),
            })
            if metrics["was_resized"]:
                aw = metrics["analyzed_size"]
                st.caption(
                    f"Анализировалось в размере {aw[0]}×{aw[1]} "
                    f"(приведено к 1920 по длинной стороне)."
                )

        if warnings:
            for wmsg in warnings:
                st.warning(f"⚠️ {wmsg}")

        camera = st.text_input(
            "Название камеры / телефона",
            value=file.name.rsplit(".", 1)[0],
            key=f"cam_{file.name}",
        )

        rows.append({"camera": camera, **metrics})

    if rows:
        st.subheader("🏆 Итоговый рейтинг")

        rated = build_rating(rows)
        df = pd.DataFrame(rated)[
            ["rank", "camera", "score", "sharpness", "noise", "dynamic_range", "exposure"]
        ]
        df.columns = ["Место", "Камера", "Балл", "Резкость", "Шум", "DR", "Экспозиция"]

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Камера")["Балл"])

        st.caption(
            "Балл — взвешенная сумма нормированных показателей по всей выборке. "
            "Шум инвертирован (меньше = лучше)."
        )