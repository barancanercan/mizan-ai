"""
mizan-ai - Query System
Hybrid RAG with Gemini + Ollama + Web Search
"""

import argparse
from typing import Any, List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
import utils
from core.parties import normalize_party_name, normalize_parties_list
from core.llm_setup import (
    setup_ollama_chain,
    create_llm_handler,
    get_llm_display_name,
)
from core.router_engine import create_router, IntentAnalysis
from core.duckduckgo_search import search_web, DuckDuckGoSearch
from core.streaming import handle_stream_response
from core.content_filter import should_answer
from core.search_agent import SearchAgent, create_search_agent
from core.political_context_agent import get_political_agent, PoliticalContext
from core.query_analyzer import get_query_analyzer, QueryAnalysis, QuestionType


def analyze_query_intent(question: str) -> IntentAnalysis:
    router = create_router(threshold=config.ROUTER_THRESHOLD)
    return router.analyze_intent(question)


def search_local_knowledge(
    vectorstore: Chroma,
    question: str,
    party: str,
    top_k: Optional[int] = None
) -> Tuple[str, List[Document], List[float]]:
    normalized_party = normalize_party_name(party)
    k = top_k or config.TOP_K

    # Political Context Agent ile sorguyu zenginleştir
    political_agent = get_political_agent()
    enhanced_query = political_agent.enhance_query_for_retrieval(question, normalized_party)
    utils.logger.info(f"Enhanced query: {enhanced_query[:100]}...")

    context, scores, docs = utils.search_similar_docs(
        vectorstore, enhanced_query, top_k=k, filter_metadata={"party": normalized_party}
    )

    return context, docs, scores


def search_online_knowledge(
    question: str,
    party: Optional[str] = None,
    max_results: Optional[int] = None
) -> List[Any]:
    """
    SearchAgent ile web araması yapar.
    """
    agent = create_search_agent()
    context = agent.search(question, party)
    return context.results


def synthesize_answer(
    local_context: str,
    web_results: List[Any],
    question: str,
    llm_handler: Any,
    llm_type: str,
    party: str,
    stream: bool = False,
) -> Tuple[Any, List[Any]]:
    """Parti bilgisi ile zenginleştirilmiş cevap üretir."""
    sources = []
    web_context = ""

    # Parti bilgisini al
    party_info = config.PARTY_INFO.get(party, {})
    party_name = party_info.get('name', party)

    if web_results:
        web_context = "\n\nGÜNCEL HABERLER:\n"
        for i, result in enumerate(web_results, 1):
            web_context += f"{i}. {result.title}\n   {result.snippet}\n   Kaynak: {result.url}\n"
        sources.append("web")

    if local_context:
        sources.append("yerel")

    # Web sonuçları varsa - farklı prompt kullan
    if web_results:
        system_instruction_web = f"""Görev: {party_name} hakkındaki soruyu yanıtla.

ÖNEMLİ KURALLAR:
1. "Bilgi bulunamadı" veya "açık bilgi yok" ASLA DEME
2. Web sonuçlarındaki İPUÇLARINI kullanarak ÇIKARIM yap
3. Haberlerde geçen her şehir/ilçe adı = o partinin belediyesi demektir

ÇIKARIM ÖRNEKLERİ:
- "Diyarbakır'da toplantı yaptı" → Diyarbakır Büyükşehir Belediyesi {party_name}'de
- "Silopi belediye başkanı ihraç" → Silopi Belediyesi {party_name}'de
- "Van'da kongre" → Van Büyükşehir Belediyesi {party_name}'de
- "Mardin mitingi" → Mardin Belediyesi {party_name}'de

CEVAP FORMATI:
Web sonuçlarından tespit edilen {party_name} belediyeleri:
1. [Şehir/İlçe adı] - [Nasıl tespit edildiği]
2. ...

Not: Bu liste web haberlerinden çıkarılmıştır, tam liste için resmi kaynaklara bakılmalıdır."""

        web_text = "**Güncel Web Sonuclari:**\n\n"
        for i, r in enumerate(web_results[:3], 1):
            web_text += f"**{i}. {r.title}**\n{r.snippet}\n[Kaynak]({r.url})\n\n"

        context_for_llm = f"GÜNCEL WEB SONUÇLARI:\n{web_context[:2000]}"
        if local_context:
            context_for_llm += f"\n\nPARTİ TÜZÜĞÜ:\n{local_context[:1500]}"

        prompt_input = {
            "context": context_for_llm,
            "question": f"{system_instruction_web}\n\nSORU: {question}"
        }

        try:
            if llm_type == "ollama":
                if stream:
                    return llm_handler.stream(prompt_input), sources
                return llm_handler.invoke(prompt_input), sources
            elif llm_type == "gemini":
                full_prompt = f"{system_instruction_web}\n\n{context_for_llm}\n\nSORU: {question}"
                if stream:
                    return llm_handler.stream(full_prompt), sources
                return llm_handler.invoke(full_prompt), sources
        except Exception as e:
            utils.logger.error(f"LLM hatası: {e}")

        def gen():
            yield web_text
        return gen(), sources

    # Fallback: sadece yerel bilgi
    if not local_context:
        return f"{party_name} hakkında bu konuda bilgi bulunamadı.", sources

    # Yerel context için explicit prompt (web sonucu yok)
    system_instruction_local = f"""Görev: {party_name} parti tüzüğünden soruyu yanıtla.

Aşağıdaki BİLGİLER bölümünde parti tüzüğünden alıntılar var. Bu alıntıları DİKKATLİCE oku ve soruyu yanıtla.
Cevabını Türkçe ver. Kısa ve öz ol. Madde numaralarını belirt."""

    prompt_input = {
        "context": f"PARTİ TÜZÜĞÜ ({party_name}):\n{local_context[:2500]}",
        "question": f"{system_instruction_local}\n\nSORU: {question}"
    }

    try:
        if llm_type == "ollama":
            if stream:
                return llm_handler.stream(prompt_input), sources
            return llm_handler.invoke(prompt_input), sources
        elif llm_type == "gemini":
            full_prompt = f"{system_instruction_local}\n\n{local_context[:2500]}\n\nSoru: {question}"
            if stream:
                return llm_handler.stream(full_prompt), sources
            return llm_handler.invoke(full_prompt), sources
    except Exception as e:
        utils.logger.error(f"LLM hatası: {e}")

    return "Yanıt üretilemedi.", sources


def ask_question(
    question: str,
    vectorstore: Chroma,
    llm_handler: Any,
    party: str,
    llm_type: str,
    stream: bool = False,
    use_router: bool = True,
) -> Tuple[Any, List[Document], IntentAnalysis, List[Any]]:
    """Sorguyu yanıtlar. Web sonuçlarını da döndürür."""

    should_respond, filter_message = should_answer(question)
    if not should_respond:
        intent = IntentAnalysis(
            intent_type="filtered", confidence=0.0,
            needs_web_search=False, parties_mentioned=[],
            reasoning=filter_message
        )
        if stream:
            def gen(): yield filter_message
            return gen(), [], intent, []
        return filter_message, [], intent, []

    normalized_party = normalize_party_name(party)

    # Query Analyzer ile sorguyu analiz et
    query_analyzer = get_query_analyzer()
    query_analysis = query_analyzer.analyze(question, normalized_party)

    # Alt soruları kontrol et - requires_web=True olan varsa web araması gerekli
    needs_web_from_analysis = any(
        sq.requires_web for sq in query_analysis.sub_questions
    )

    # DEBUG: Alt soru analizi
    utils.logger.info(f"=== QUERY ANALYSIS ===")
    utils.logger.info(f"Original: {question}")
    utils.logger.info(f"Sub-questions: {len(query_analysis.sub_questions)}")
    for i, sq in enumerate(query_analysis.sub_questions):
        utils.logger.info(f"  {i+1}. [{sq.question_type.value}] '{sq.text}' - web_required={sq.requires_web}")
    utils.logger.info(f"needs_web_from_analysis: {needs_web_from_analysis}")

    # Normal retrieval
    local_context, docs, scores = search_local_knowledge(vectorstore, question, normalized_party)
    first_score = scores[0] if scores else 0.0

    if use_router:
        intent = analyze_query_intent(question)
        intent.confidence = first_score
    else:
        intent = IntentAnalysis(
            intent_type="local", confidence=first_score,
            needs_web_search=False, parties_mentioned=[normalized_party],
            reasoning="Router devre dışı"
        )

    # Web araması kararı:
    # 1. QueryAnalyzer güncel bilgi gerektiğini tespit ettiyse
    # 2. Router web araması istiyorsa
    # 3. Retrieval skoru kötüyse (threshold üstü)
    needs_web = needs_web_from_analysis or intent.needs_web_search or first_score > config.SIMILARITY_THRESHOLD

    # DEBUG: Web araması kararı
    utils.logger.info(f"=== WEB SEARCH DECISION ===")
    utils.logger.info(f"needs_web_from_analysis: {needs_web_from_analysis}")
    utils.logger.info(f"intent.needs_web_search: {intent.needs_web_search}")
    utils.logger.info(f"first_score: {first_score} (threshold: {config.SIMILARITY_THRESHOLD})")
    utils.logger.info(f"FINAL needs_web: {needs_web}")

    web_results = []
    if needs_web:
        utils.logger.info("Web aramasi yapiliyor...")
        web_results = search_online_knowledge(question, normalized_party)
        utils.logger.info(f"📊 Web arama sonuçları: {len(web_results)} adet")
        for i, r in enumerate(web_results[:3], 1):
            utils.logger.info(f"  {i}. {r.title}")

    # Compound sorgu için zenginleştirilmiş prompt oluştur
    enhanced_question = question
    if query_analysis.is_compound:
        # Alt soruları açıkça belirt
        sub_q_list = "\n".join([f"- {sq.text}" for sq in query_analysis.sub_questions])
        enhanced_question = f"""{question}

Bu soruda şu alt sorular var, HEPSİNİ yanıtla:
{sub_q_list}"""
        utils.logger.info(f"Compound sorgu: {len(query_analysis.sub_questions)} alt soru")

    response_gen, sources = synthesize_answer(
        local_context, web_results, enhanced_question,
        llm_handler, llm_type, normalized_party, stream
    )

    return response_gen, docs, intent, web_results


def stream_response(response_gen: Any, llm_type: str) -> str:
    result = ""
    for chunk in response_gen:
        result += handle_stream_response(chunk, llm_type)
    return result


def single_party_mode(party: str):
    party = normalize_party_name(party)
    utils.print_header(f"🤖 {party} Soru-Cevap Sistemi")
    
    db_path = config.UNIFIED_VECTOR_DB
    if not db_path or not db_path.exists():
        utils.logger.error("❌ DB bulunamadı!")
        return

    embeddings = utils.load_embeddings()
    vectorstore = utils.load_vectorstore(db_path, embeddings)
    llm_handler, llm_type = create_llm_handler(party)

    if llm_type == "none":
        utils.logger.error("❌ LLM yok!")
        return

    utils.logger.info(f"✅ Sistem hazır!")
    
    while True:
        question = input(f"Sorunuz: ").strip()
        if question.lower() in ["q", "quit", "exit"]:
            break
        if not question:
            continue

        try:
            response_gen, source_docs, _, _ = ask_question(
                question, vectorstore, llm_handler, party, llm_type, stream=True
            )
            for chunk in response_gen:
                print(handle_stream_response(chunk, llm_type), end="", flush=True)
            print()
        except Exception as e:
            print(f"⚠️ Hata: {str(e)}")


def multi_party_mode():
    utils.print_header("🤖 Çoklu Parti Q&A")
    prepared_parties = normalize_parties_list(utils.get_prepared_parties())
    
    if not prepared_parties:
        utils.logger.error("❌ DB yok!")
        return

    embeddings = utils.load_embeddings()
    vectorstore = utils.load_vectorstore(config.UNIFIED_VECTOR_DB, embeddings)
    llm_handler, llm_type = create_llm_handler(prepared_parties[0])

    current_party = "CHP" if "CHP" in prepared_parties else prepared_parties[0]

    while True:
        question = input(f"[{current_party}] Sorunuz: ").strip()
        if question.lower() in ["q", "quit", "exit"]:
            break
        if question.startswith("/"):
            party_cmd = question[1:].upper()
            party_cmd = normalize_party_name(party_cmd)
            if party_cmd in prepared_parties:
                current_party = party_cmd
            continue

        try:
            response_gen, _, _, _ = ask_question(
                question, vectorstore, llm_handler, current_party, llm_type, stream=True
            )
            for chunk in response_gen:
                print(handle_stream_response(chunk, llm_type), end="", flush=True)
            print()
        except Exception as e:
            print(f"⚠️ Hata: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="mizan-ai")
    parser.add_argument("--party", type=str, help="Tek parti modu")
    args = parser.parse_args()

    if args.party:
        single_party_mode(args.party)
    else:
        multi_party_mode()


if __name__ == "__main__":
    main()
