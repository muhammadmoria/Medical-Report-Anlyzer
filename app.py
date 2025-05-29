import streamlit as st
import os
import tempfile
import logging
import sys
import pandas as pd
from src.ocr import extract_text
from src.nlp import structure_data
from src.categorize import categorize_results
from src.table_formatter import format_results_for_table
from src.explain import explain_results_batch
from src.summary import generate_summary_bullet_points
from src.pdf_generator import generate_pdf_summary
from src.chatbot import MedicalChatbot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="[Stethoscope]",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@keyframes quantumParticles {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 200%; }
}
@keyframes quantumPulse {
    0% { box-shadow: 0 0 8px #00e6ff, 0 0 15px #ff4da6; }
    50% { box-shadow: 0 0 30px #00e6ff, 0 0 45px #ff4da6; }
    100% { box-shadow: 0 0 8px #00e6ff, 0 0 15px #ff4da6; }
}
@keyframes holoShimmer {
    0% { transform: skewX(-3deg) translateX(-3px); opacity: 0.9; }
    50% { transform: skewX(3deg) translateX(3px); opacity: 1; }
    100% { transform: skewX(-3deg) translateX(-3px); opacity: 0.9; }
}
@keyframes glowPulse {
    0% { transform: scale(1); filter: brightness(100%); }
    50% { transform: scale(1.2); filter: brightness(140%); }
    100% { transform: scale(1); filter: brightness(100%); }
}
.stApp {
    color: #f0f4ff;
    background: linear-gradient(135deg, #1a1a3d, #2a2a5a);
    font-family: 'Neon Glow', sans-serif;
    background-size: 200% 200%;
    animation: quantumParticles 12s linear infinite;
}
.stSidebar {
    background: linear-gradient(180deg, #2a2a5a, #3a3a7a);
    border-right: 2px solid #ff4da6;
    box-shadow: 0 0 25px rgba(0, 230, 255, 0.5);
    padding: 20px;
    border-radius: 0 12px 12px 0;
}
.quantum-panel {
    background: rgba(0, 230, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 15px 50px rgba(0, 230, 255, 0.4);
    border: 2px solid #b266ff;
    animation: quantumPulse 2s infinite;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.quantum-panel:hover {
    transform: translateY(-8px);
    box-shadow: 0 25px 70px rgba(0, 230, 255, 0.6);
}
.shiny-card {
    background: rgba(30, 30, 70, 0.9);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0, 230, 255, 0.8), 0 0 30px rgba(255, 77, 166, 0.6);
    border: 2px solid #00ff99;
    animation: glowPulse 2s infinite;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.shiny-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 0 30px rgba(0, 230, 255, 1), 0 0 40px rgba(255, 77, 166, 0.8);
}
.stButton>button {
    background: #00e6ff;
    color: #ffffff;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 0 15px rgba(0, 230, 255, 0.8);
    transition: box-shadow 0.3s ease;
    font-size: 14px;
    font-weight: 600;
    font-family: 'Orbitron', sans-serif;
}
.stButton>button:hover {
    box-shadow: 0 0 25px rgba(0, 230, 255, 1);
}
h1 {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(45deg, #00e6ff, #ff4da6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 15px rgba(255, 77, 166, 0.8);
    animation: holoShimmer 1.8s infinite;
}
h2, h3 {
    font-family: 'Orbitron', sans-serif;
    color: #00e5ff;
    text-shadow: 0 0 10px rgba(0, 230, 255, 0.6);
}
.shiny-card h3 {
    color: #b266ff;
    text-shadow: 0 0 10px rgba(178, 102, 255, 0.8);
}
.shiny-card p, .shiny-card div, .shiny-card li {
    color: #e0ccff;
}
.shiny-card b {
    color: #00e6ff;
}
.warning { color: #ff3366; font-weight: bold; }
.critical { color: #ff5252; font-weight: bold; }
.borderline { color: #ffca28; font-weight: bold; }
.normal { color: #00ff99; font-weight: bold; }
.stDataFrame {
    background: rgba(20, 20, 50, 0.9);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 0 20px rgba(0, 230, 255, 0.3);
}
.dataframe th {
    background: linear-gradient(45deg, #00e6ff, #b266ff) !important;
    color: #ffffff !important;
    font-weight: bold !important;
}
.dataframe tr:nth-child(even) { background: rgba(30, 30, 70, 0.9); }
.dataframe tr:nth-child(odd) { background: rgba(40, 40, 90, 0.9); }
.dataframe td { color: #f0f4ff; }
.bullet-point {
    margin-left: 30px;
    color: #e0ccff;
    font-size: 15px;
    line-height: 1.8;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 25px;
    background: rgba(20, 20, 50, 0.8);
    padding: 10px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0, 230, 255, 0.4);
}
.stTabs [data-baseweb="tab"] {
    background: rgba(0, 230, 255, 0.2);
    color: #f0f4ff;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 14px;
    transition: all 0.3s ease;
    font-family: 'Orbitron', sans-serif;
    border: 1px solid #b266ff;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(45deg, #ff4da6, #b266ff);
    color: #ffffff;
    box-shadow: 0 0 15px rgba(255, 77, 166, 0.9);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(178, 102, 255, 0.7);
    box-shadow: 0 0 20px rgba(255, 77, 166, 0.7);
}
.chat-message {
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 12px;
    transition: all 0.3s ease;
}
.chat-message.user { background: linear-gradient(45deg, #00e6ff, #b266ff); color: #ffffff; }
.chat-message.bot { background: rgba(30, 30, 70, 0.9); color: #f0f4ff; }
.stProgress .st-bo {
    background: linear-gradient(45deg, #ff4da6, #b266ff) !important;
}
.emoji-glow {
    display: inline-block;
    animation: glowPulse 1.5s infinite;
}
[role="tabpanel"] {
    animation: quantumFade 0.5s ease-in;
}
@keyframes quantumFade {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.collapsible {
    background: rgba(0, 230, 255, 0.2);
    color: #f0f4ff;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    cursor: pointer;
    border: 1px solid #b266ff;
}
.collapsible-content {
    background: rgba(30, 30, 70, 0.9);
    padding: 15px;
    border-radius: 10px;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Neon+Glow&display=swap" rel="stylesheet">
<script>
function toggleCollapsible(id) {
    var content = document.getElementById(id);
    if (content.style.display === "none") {
        content.style.display = "block";
    } else {
        content.style.display = "none";
    }
}
</script>
""", unsafe_allow_html=True)

st.sidebar.title("[Medical Insights] Health Report Analyzer 🩺✨")
st.sidebar.markdown("""
<p style='color:#00ff99'><b>Intelligent Medical Analysis</b> <span class='emoji-glow'>🏥💉</span></p>
<p style='color:#00e5ff'>Developed for the Hackathon by <b><a href="https://muhammadmoria.github.io/Portfolio--/">Muhammad Dawood</a></b> <span class='emoji-glow'>💻🚀</span><br>
This application leverages advanced AI to process, organize, and clarify medical test results, providing accessible and actionable insights for patients. <span class='emoji-glow'>💡🎯⚡</span><br>
<b>Driven by Groq LLM and Streamlit</b> <span class='emoji-glow'>⚡🔥</span></p>
""", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color:#00e5ff'>", unsafe_allow_html=True)

st.sidebar.subheader("📤 Upload Medical Report 🌡️🩺")
uploaded_files = st.sidebar.file_uploader("Choose a file 📂", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="global_uploader")

tab1, tab2, tab3 = st.tabs(["🏠 Home 🏡", "🩺 Analyze 🔍", "💬 Chatbot 🤖"])

with tab1:
    st.title("[Medical Insights] Health Report Analyzer 🌟🩺")
    st.markdown('<div class="quantum-panel">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#ffd700'>Project Overview 🎯✨</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#00ff99'>The Advanced Health Analytics platform is a pioneering tool designed to enhance comprehension of medical reports. Created for the Hackathon by <b><a href="https://muhammadmoria.github.io/Portfolio--/">Muhammad Dawood</a></b>, it utilizes cutting-edge AI to analyze, structure, and elucidate test results, offering clear and practical insights for patients. 🚀💉</p>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#ffd700'>Tech Ecosystem 🚀🔧</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='bullet-point'>
        <span class='emoji-glow'>🔥✨</span> <b>Streamlit</b>: Facilitates the development of an engaging web interface.<br>
        <span class='emoji-glow'>🔥💡</span> <b>Groq LLM</b>: Fuels sophisticated natural language processing and insight creation.<br>
        <span class='emoji-glow'>🔥📷</span> <b>OpenCV & Pytesseract</b>: Supports OCR for extracting text from images and PDFs.<br>
        <span class='emoji-glow'>🔥📊</span> <b>Pandas</b>: Enables data organization and table visualization.<br>
        <span class='emoji-glow'>🔥📄</span> <b>ReportLab</b>: Produces polished PDF summaries.<br>
        <span class='emoji-glow'>🔥🐍</span> <b>Python</b>: The essential programming language.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#ffd700'>Process Flow 🔄⚙️</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='bullet-point'>
        <span class='emoji-glow'>🔥📤</span> <b>Upload</b>: Users submit a medical report (PDF, PNG, or JPEG).<br>
        <span class='emoji-glow'>🔥📜</span> <b>Text Extraction</b>: OCR technology retrieves text from the report.<br>
        <span class='emoji-glow'>🔥🧠</span> <b>Data Structuring</b>: NLP organizes and categorizes the extracted data.<br>
        <span class='emoji-glow'>🔥🔍</span> <b>Evaluation</b>: Groq LLM analyzes and interprets test results.<br>
        <span class='emoji-glow'>🔥📋</span> <b>Overview</b>: A clear summary with risks and suggestions is provided.<br>
        <span class='emoji-glow'>🔥💾</span> <b>Export</b>: Users can download a comprehensive PDF report.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#ffd700'>Capabilities 🌐🔮</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='bullet-point'>
        <span class='emoji-glow'>🔥📁</span> <b>OCR Compatibility</b>: Supports various file formats (PDF, PNG, JPEG).<br>
        <span class='emoji-glow'>🔥💡</span> <b>AI-Driven Insights</b>: Delivers in-depth explanations of test outcomes.<br>
        <span class='emoji-glow'>🔥📈</span> <b>Flexible Outputs</b>: Features color-coded statuses and exportable PDFs.<br>
        <span class='emoji-glow'>🔥🎨</span> <b>Accessible Design</b>: Offers an intuitive interface with clear language.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#ffd700'>Team 👥🤝</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#00ff99'>Developed by <b><a href="https://muhammadmoria.github.io/Portfolio--/">Muhammad Dawood</a></b> with passion and expertise for the Hackathon! 🚀🎉</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.title("🏥 [Medical Insights] AI-Driven Health Report Analyzer 💥🔬🌟")
    st.markdown("<p style='color:#00ff99; font-size: 18px;'>Upload your medical report for cutting-edge AI insights! <span class='emoji-glow'>💡🎯⚡🚀</span></p>", unsafe_allow_html=True)
    st.sidebar.subheader("⏳ Processing Status 🔄")
    status_placeholder = st.sidebar.empty()

    if uploaded_files:
        with st.spinner("🔄 Analyzing your report... 🕒⏰"):
            try:
                if len(uploaded_files) > 1:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.warning("⚠️❌ Using only the first uploaded file for analysis.")
                    st.markdown('</div>', unsafe_allow_html=True)
                uploaded_file = uploaded_files[0]
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                raw_text = extract_text(tmp_file_path)
                if not raw_text:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.warning("⚠️📄 No text extracted from the file.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    raise ValueError("Text extraction failed")
                
                structured_data = structure_data(raw_text)
                if not structured_data:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.warning("⚠️🧠 No structured data extracted.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    raise ValueError("Data structuring failed")

                categorized_data = categorize_results(structured_data)
                if not categorized_data:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.warning("⚠️📊 No categorized data generated.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    raise ValueError("Categorization failed")

                test_results = [r for r in categorized_data if "test_name" in r or "Test" in r]
                metadata = [r for r in categorized_data if "test_name" not in r and "Test" not in r]

                if metadata:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#b266ff'>🧬 [Patient Profile] Data Overview 📊✨💉</h3>", unsafe_allow_html=True)
                    with st.expander("👁️‍🗨️ View Details 🔍"):
                        for item in metadata:
                            fields = "<br>".join(f"<b style='color:#00e6ff'>{k}</b>: <span style='color:#e0ccff'>{v}</span>" for k, v in item.items() if v)
                            st.markdown(fields, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if test_results:
                    table_data = format_results_for_table(test_results)
                    if table_data:
                        st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                        st.markdown("<h3 style='color:#b266ff'>🧪 Test Results 📊🔬</h3>", unsafe_allow_html=True)
                        df = pd.DataFrame(table_data)
                        def color_status(val):
                            if val == "Critical":
                                return 'color: #ff5252; font-weight: bold'
                            elif val == "Borderline":
                                return 'color: #ffca28; font-weight: bold'
                            elif val == "Normal":
                                return 'color: #00ff99; font-weight: bold'
                            return 'color: #e0ccff'
                        styled_df = df.style.map(color_status, subset=['status'])
                        st.dataframe(styled_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                        st.markdown("<h3>📘 Explanations 💡✨🔍</h3>", unsafe_allow_html=True)
                        explanation = explain_results_batch(test_results)
                        if explanation and explanation != "Unable to generate explanations due to an error.":
                            formatted_explanation = explanation.replace("**", "<b>").replace("**", "</b>")
                            formatted_explanation = formatted_explanation.replace("Critical", "<span class='critical'>Critical 🩺🚨</span>").replace("Borderline", "<span class='borderline'>Borderline 🩺⚠️</span>").replace("Normal", "<span class='normal'>Normal 🩺✅</span>")
                            st.markdown(f"<p>{formatted_explanation} 🌟</p>", unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="warning">⚠️❌ No explanations generated. 😕</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                        st.markdown("<h3>📝 Summary & Recommendations 🌿📋✨</h3>", unsafe_allow_html=True)
                        if explanation and test_results:
                            summary_bullets = generate_summary_bullet_points(explanation)
                            if summary_bullets and summary_bullets != "Unable to generate summary due to an error.":
                                formatted_summary = summary_bullets.replace("**Summary:**", "<b>✨ Summary: 🌟</b>")
                                formatted_summary = formatted_summary.replace("**Risks/Conditions:**", "<b>🚨 Risks/Conditions: ⚠️</b>")
                                formatted_summary = formatted_summary.replace("**Actions/Recommendations:**", "<b>✅ Actions/Recommendations: 💡</b>")
                                formatted_summary = formatted_summary.replace("* ", "<span class='emoji-glow'>🌟✨</span> ").replace("\n", "<br>")
                                st.markdown(f"<div class='bullet-point'>{formatted_summary} 🎉</div>", unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="warning">⚠️❌ No summary generated. 😕</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="warning">⚠️❌ No explanations available for summary. 😕</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                        st.markdown("<h3>📄 Download Summary 📥💾✨</h3>", unsafe_allow_html=True)
                        if test_results and explanation:
                            if st.button("📄 Generate PDF Summary 🌟🚀"):
                                pdf_bytes = generate_pdf_summary(categorized_data, explanation, summary_bullets)
                                st.download_button(
                                    label="💾 Save PDF Report 🎯📩",
                                    data=pdf_bytes,
                                    file_name="medical_summary.pdf",
                                    mime="application/pdf"
                                )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                        st.markdown('<p class="warning">⚠️❌ No test data found to display. 😕</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                    st.markdown('<p class="warning">⚠️❌ No test results found. 😕</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                os.unlink(tmp_file_path)
                logger.info(f"Temporary file deleted: {tmp_file_path}")
                status_placeholder.markdown("<p style='color:#00ff99'>✅🎉 Report processed successfully! 🚀</p>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown('<div class="shiny-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="warning">❌🚨 Error: {str(e)} 😕</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                logger.error(f"Error processing file: {str(e)}")
                status_placeholder.markdown("<p style='color:#ff5252'>❌🚨 Processing failed! 😕</p>", unsafe_allow_html=True)
    else:
        st.info("📢📄 Please upload a medical report using the sidebar to start analyzing! 🚀🌟")

with tab3:
    st.session_state["current_page"] = "chatbot"
    chatbot_obj = MedicalChatbot(uploaded_files)
    chatbot_obj.main()