
import streamlit as st
import os
import cv2
from utils import ensure_dirs, get_embedding, save_embedding, add_student_row, load_students, load_all_embeddings, today_attendance_filename
import pandas as pd
import numpy as np
import threading
import subprocess
import time
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

ensure_dirs()
st.set_page_config(page_title="Face Attendance", layout="wide")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Enroll Student", "Reports"])




if page == "Enroll Student":
    st.header("Enroll Student")
    with st.form("enroll_form"):
        student_id = st.text_input("Student ID")
        name = st.text_input("Name")
        image = st.file_uploader("Photo (jpg/png)", type=['jpg','jpeg','png'])
        submitted = st.form_submit_button("Enroll & Save")
    if submitted:
        if not student_id or not name or image is None:
            st.warning("Fill all fields and upload a photo.")
        else:
            images_dir = "images"
            os.makedirs(images_dir, exist_ok=True)
            fname = f"{student_id}.jpg"
            path = os.path.join(images_dir, fname)
            
            with open(path, "wb") as f:
                f.write(image.getbuffer())
            
            emb = get_embedding(path)
            if emb is None:
                st.error("No face detected in uploaded photo. Try a different image (face should be clear).")
                os.remove(path)
            else:
                save_embedding(student_id, emb)
                add_student_row(student_id, name, path)
                st.success(f"Enrolled {name} (ID: {student_id}). Embedding saved.")

    st.markdown("---")
    st.subheader("Current Enrolled Students")
    students = load_students()
    if len(students) == 0:
        st.info("No students yet.")
    else:
        df = pd.DataFrame(students)
        st.dataframe(df)



    




elif page == "Reports":
    st.header("Attendance Reports")

    files = sorted([f for f in os.listdir("attendance") if f.endswith(".csv")])

    today = datetime.now().date()
    default_start = today - timedelta(days=30)

    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    selected_files = []
    for f in files:
        file_date = datetime.strptime(f.split("_")[1].split(".")[0], "%Y-%m-%d").date()
        
        if start_date <= file_date <= end_date:
            
            if file_date.day == 1 or file_date.weekday() != 5:
                selected_files.append(f)

    if not selected_files:
        st.info("No attendance records in this date range.")
        st.stop()

   
    df_list = [pd.read_csv(os.path.join("attendance", f)) for f in selected_files]
    df = pd.concat(df_list, ignore_index=True)

    
    students_df = pd.read_csv("Students.csv")
    student_options = students_df['student_id'].astype(str) + " - " + students_df['name']
    selected_students = st.multiselect("Select student(s)", student_options, default=student_options.tolist())
    selected_ids = [s.split(" - ")[0] for s in selected_students]

    df_filtered = df[df['id'].astype(str).isin(selected_ids)]

    total_days = len(selected_files)
    summary = {}

    # Pie charts 
    for sid in selected_ids:
        name = students_df[students_df['student_id'].astype(str) == sid]['name'].values[0]

        present_count = df_filtered[df_filtered['id'].astype(str) == sid].shape[0]
        absent_count = total_days - present_count

        fig, ax = plt.subplots()
        ax.pie([present_count, absent_count], labels=["Present", "Absent"], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{name} Attendance")
        st.pyplot(fig)

    #  Bar Chart
    if 'time' in df.columns:
        def bucket_time(t):
            try:
                hh, mm, ss = map(int, t.split(":"))
                sec = hh * 3600 + mm * 60 + ss
                class_start = 10 * 3600  # 10:00 AM

                if sec < class_start - 300:
                    return "Early"
                elif class_start - 300 <= sec <= class_start + 300:
                    return "On time"
                else:
                    return "Late"
            except:
                return "Unknown"

        df['punctuality'] = df['time'].astype(str).apply(bucket_time)
        counts = df['punctuality'].value_counts().reindex(["Early", "On time", "Late", "Unknown"], fill_value=0)

        fig2, ax2 = plt.subplots()
        ax2.bar(counts.index, counts.values)
        ax2.set_ylabel("Count")
        ax2.set_title("Punctuality Summary")
        st.pyplot(fig2)

   
    export_path = "attendance_report.csv"
    df.to_csv(export_path, index=False)

    st.download_button(
        "Download Attendance Report",
        data=open(export_path, "rb").read(),
        file_name="attendance_report.csv",
        mime="text/csv"
    )
