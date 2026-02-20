import streamlit as st
from pathlib import Path
import sys
import logging
import os
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

sys.path.append(str(Path(__file__).parent))

import config
import utils
from core.parties import normalize_parties_list
from core.llm_setup import create_llm_handler, get_llm_display_name
from core.streaming import handle_stream_response
from core.cache import get_vectorstore
from query_system import ask_question

# Suppress warnings
logging.getLogger("PIL").setLevel(logging.CRITICAL)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

# ============================================
# MINIMALIST CSS - CLEAN & ELEGANT
# ============================================

# ============================================
# MODERN CSS - GLASSMORPHISM & PREMIUM LOOK
# ============================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Main Container */
.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Glassmorphism Cards */
div.stButton > button {
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 14px rgba(0,0,0,0.1);
}

/* Modern Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent;
    border-radius: 4px 4px 0 0;
    gap: 8px;
    padding-top: 10px;
    font-weight: 600;
}

/* Source Expander */
.source-box {
    background: rgba(255, 255, 255, 0.6);
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #0066FF;
    margin-bottom: 10px;
    font-size: 0.9rem;
}

/* Logo pulse animation */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.pulse-logo {
    animation: pulse 3s infinite ease-in-out;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# PREPARED PARTIES - İYİ NORMALIZATION
# ============================================

prepared_parties = utils.get_prepared_parties()
if not prepared_parties:
    st.error("❌ Hiç hazır veri yok!")
    st.info("💡 Çalıştır: `python src/prepare_data.py`")
    st.stop()

prepared_parties = normalize_parties_list(prepared_parties)


# ============================================
# LOCAL IMAGE LOADING - ROBUST
# ============================================


@st.cache_data
def load_logo_image_robust(party: str):
    """Local picture klasöründen logo yükle - robust"""
    filename = config.PARTY_LOGOS.get(party)
    if not filename:
        return None

    logo_path = config.PICTURE_DIR / filename
    if not logo_path.exists():
        return None

    try:
        from PIL import Image

        img = Image.open(logo_path)
        img.verify()
        img = Image.open(logo_path)
        return img
    except Exception:
        return None


def display_party_logo(party: str, width: int = 100, pulse: bool = False):
    """Logo göster"""
    img = load_logo_image_robust(party)
    if img:
        if pulse:
            st.markdown('<div class="pulse-logo">', unsafe_allow_html=True)
        st.image(img, width=width)
        if pulse:
            st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# SIDEBAR - ULTRA MINIMALIST
# ============================================

with st.sidebar:
    # Party selection
    st.markdown("### 🇹🇷 Partiler")

    selected_party = st.radio(
        "Seç",
        prepared_parties,
        index=0 if prepared_parties else None,
        format_func=lambda p: config.PARTY_INFO.get(p, {}).get("short", p),
        label_visibility="collapsed",
    )

    # Party logo + basic info
    party_info = config.PARTY_INFO.get(selected_party)

    if party_info:
        st.markdown("---")

        # Logo (small)
        col1, col2 = st.columns([1, 2])
        with col1:
            display_party_logo(selected_party, width=60)
        with col2:
            st.markdown(f"**{party_info['short']}**")
            st.caption(f"{party_info['name'][:25]}")

        st.markdown("---")

        # Minimal info - only essentials
        st.markdown(f"📊 **Kuruluş:** {party_info.get('founded', '?')}")
        st.markdown(f"🌐 [{party_info['website']}]({party_info['website']})")

    st.markdown("---")

    # Footer links
    st.caption(
        "**[GitHub](https://github.com/barancanercan/mizan-ai) • "
        "[LinkedIn](https://linkedin.com/in/barancanercan) • "
        "[Medium](https://barancanercan.medium.com)**"
    )


# ============================================
# LLM & VECTOR STORE SETUP (LAZY)
# ============================================


@st.cache_resource
def setup_llm():
    """LLM'i bir kez hazırlar ve cache'ler."""
    return create_llm_handler(prepared_parties[0] if prepared_parties else "CHP")


@st.cache_resource
def get_cached_vectorstore():
    """Vectorstore'u bir kez yükle ve cache'le"""
    return get_vectorstore()


# Lazy LLM Setup
llm_handler, llm_type = setup_llm()

# ============================================
# MAIN CONTENT - PAGE TITLE + TABS
# ============================================

st.title("Ulak-AI Türk Siyasi Partileri Bilgi Sistemi")
st.markdown("Açık Kaynak • Ücretsiz • 100% Türkçe" )

st.markdown("---")

# Tabs
tab_soru, tab_compare, tab_stats, tab_hakkinda = st.tabs(
    ["🔍 Soru Sor", "⚖️ Karşılaştır", "📊 Veri Merkezi", "ℹ️ Hakkında"]
)

# ============================================
# TAB 1: SORU SOR
# ============================================

with tab_soru:
    party_info = config.PARTY_INFO.get(selected_party)

    if party_info:
        # Minimal header - logo + name only
        col_logo, col_title = st.columns([1, 4], gap="medium")

        with col_logo:
            display_party_logo(selected_party, width=100, pulse=True)

        with col_title:
            st.markdown(f"## {party_info['short']}")
            st.markdown(f"**{party_info['name']}**")

    st.markdown("---")

    # Q&A SECTION - CLEAN
    col_q, col_status = st.columns([5, 1.5])

    with col_q:
        st.markdown("### Sorunuzu Yazın")

    with col_status:
        if llm_type == "gemini":
            st.success("⚡ Gemini 1.5 Flash")
        elif llm_type == "ollama":
            st.info("🔌 Ollama (Yedek)")
        else:
            st.error("⚠️ LLM Yok")

    # Question input
    question = st.text_input(
        "Sorunuzu yazın",
        placeholder="Örn: Genel başkanı nasıl seçilir?",
        label_visibility="collapsed",
        key="single_q",
    )

    # Action buttons - inline
    col_ask, col_clear = st.columns([4, 1])

    with col_ask:
        ask_btn = st.button("Cevap Al ✨", use_container_width=True, type="primary")

    with col_clear:
        if st.button("Temizle", use_container_width=True, key="clear_single"):
            st.rerun()

    st.markdown("---")

    # RESPONSE - MINIMAL
    if ask_btn:
        if not question.strip():
            st.warning("Lütfen bir soru yazın")
        elif llm_type == "none":
            st.error("LLM çalışmıyor! GEMINI_API_KEY veya Ollama ayarlayın")
        else:
            with st.spinner(f"{selected_party} araştırılıyor..."):
                try:
                    # Lazy Load Vector Store
                    vs = get_cached_vectorstore()

                    # Stream Generator
                    response_gen, source_docs, intent, web_results = ask_question(
                        question,
                        vs,
                        llm_handler,
                        selected_party,
                        llm_type,
                        stream=True,
                    )

                    def stream_container():
                        for chunk in response_gen:
                            yield handle_stream_response(chunk, llm_type)

                    st.write_stream(stream_container)

                    # Show intent analysis
                    if intent:
                        with st.expander("🎯 Niyet Analizi"):
                            st.write(f"**Tip:** {intent.intent_type}")
                            st.write(f"**Güven:** {intent.confidence:.2f}")
                            st.write(f"**Web Araması:** {'Evet' if intent.needs_web_search else 'Hayır'}")

                    # Source Expander
                    if source_docs:
                        with st.expander("📚 Kaynaklar ve Alıntılar"):
                            for i, doc in enumerate(source_docs):
                                page_num = doc.metadata.get("page", "?")
                                st.markdown(
                                    f"""
                                <div class="source-box">
                                    <b>Kaynak {i+1} - Sayfa {page_num}</b><br>
                                    <i>"{doc.page_content[:500]}..."</i>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                    
                    # Show web results if available
                    if web_results:
                        with st.expander("🌐 Web Sonuçları", expanded=True):
                            for i, r in enumerate(web_results[:3], 1):
                                st.markdown(f"**{i}. {r.title}**")
                                st.write(r.snippet)
                                st.markdown(f"[Kaynak]({r.url})")
                                st.markdown("---")

                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)[:150]}")

# ============================================
# TAB 2: KARŞILAŞTIRMA MODU
# ============================================

with tab_compare:
    st.markdown("### ⚖️ Partileri Karşılaştır")
    st.caption("Aynı soruyu seçtiğiniz birden fazla partiye aynı anda sorun.")

    target_parties = st.multiselect(
        "Karşılaştırılacak Partileri Seçin",
        prepared_parties,
        default=prepared_parties[:2] if len(prepared_parties) >= 2 else prepared_parties,
    )

    compare_question = st.text_input(
        "Ortak Sorunuz", placeholder="Örn: Gençlik kolları yapısı nasıldır?", key="comp_q"
    )

    if st.button("Karşılaştır ⚖️", type="primary", use_container_width=True):
        if not compare_question.strip():
            st.warning("Lütfen bir soru yazın")
        elif not target_parties:
            st.warning("En az bir parti seçin")
        else:
            vs = get_cached_vectorstore()
            cols = st.columns(len(target_parties))

            for i, p_code in enumerate(target_parties):
                with cols[i]:
                    st.markdown(f"#### {p_code}")
                    display_party_logo(p_code, width=50)

                    with st.spinner(f"{p_code} yanıt veriyor..."):
                        try:
                            resp, docs, _, _ = ask_question(
                                compare_question, vs, llm_handler, p_code, llm_type, stream=False
                            )
                            st.info(resp)
                        except Exception as e:
                            st.error(f"Hata: {str(e)[:50]}")

# ============================================
# TAB 3: VERİ MERKEZİ (DASHBOARD)
# ============================================

with tab_stats:
    st.markdown("### 📊 Veri Merkezi")

    col1, col2, col3 = st.columns(3)

    vs = get_cached_vectorstore()
    total_chunks = vs._collection.count()

    with col1:
        st.metric("Toplam Bilgi Parçası", f"{total_chunks:,}")
    with col2:
        st.metric("Aktif Partiler", len(prepared_parties))
    with col3:
        st.metric("Model Durumu", llm_type.upper())

    st.markdown("---")

    # DB Details
    st.markdown("#### Parti Bazlı Dağılım")
    # Histogram or metric list for parties (Mock data based on discovery)
    for p in prepared_parties:
        st.caption(f"✅ {p}: PDF Hazır, Vektör DB'de kayıtlı.")

# ============================================
# TAB 4: HAKKINDA
# ============================================

with tab_hakkinda:
    st.markdown(
        """
    ### 🇹🇷 Proje Hakkında
    Açık kaynak tüzük analiz sistemi.
    
    **Teknoloji**
    - **LLM:** Qwen2.5-7B
    - **Veritabanı:** ChromaDB (Unified)
    - **Vektör:** Turkish BGE-M3
    
    **Bağlantılar**
    - [GitHub](https://github.com/barancanercan/mizan-ai)
    """
    )

# ============================================
# FOOTER - MINIMAL & CENTERED
# ============================================

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 2rem;">
        <p style="font-size: 0.85rem; color: gray;">
        🇹🇷 Açık Kaynak • {len(prepared_parties)} Parti • Made with ❤️ by Baran Can Ercan for Betül Kurt 🍓 
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
