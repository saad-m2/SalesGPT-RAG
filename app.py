# app.py
import os, streamlit as st, pandas as pd
from dotenv import load_dotenv
load_dotenv()

from core import ingest_pdf, hf_embed_batch, l2_normalize, ensure_collection, upsert_records, build_bm25
from core import hybrid_search, pitch_prompt, call_groq_chat, recall_at_k, qdrant_client
import ml

st.set_page_config(page_title="SalesGPT-RAG", layout="wide")
st.title("SalesGPT-RAG (Streamlit demo)")

page = st.sidebar.radio("Page", ["Ingest","Query","Insights","Eval"])

if page == "Ingest":
    st.header("Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    meta_industry = st.text_input("Metadata: industry", "unknown")
    if st.button("Ingest") and uploaded:
        all_records = []
        for f in uploaded:
            local = f"data/raw/{f.name}"
            os.makedirs("data/raw", exist_ok=True)
            with open(local,"wb") as out:
                out.write(f.read())
            recs = ingest_pdf(local, meta={"industry":meta_industry})
            all_records.extend(recs)
        st.write("Chunks:", len(all_records))
        texts = [r["text"] for r in all_records]
        vecs = hf_embed_batch(texts, batch_size=8)
        vecs = [l2_normalize(v) for v in vecs]
        ensure_collection(len(vecs[0]))
        upsert_records(all_records, vecs)
        bm25_path = build_bm25(all_records)
        st.success("Indexed to Qdrant Cloud. BM25 saved.")

elif page == "Query":
    st.header("Generate pitch / use-case")
    company = st.text_input("Company", "Siemens")
    industry = st.text_input("Industry", "Industrial Automation")
    goal = st.selectbox("Goal", ["pitch email","demo script","use-case brief"])
    focus = st.text_input("Focus (optional)", "energy efficiency")
    user_q = st.text_input("Query phrase (what to retrieve)", "energy efficiency PLC retrofit")
    if st.button("Generate"):
        with st.spinner("Retrieving..."):
            snippets = hybrid_search(user_q, top=8)
        st.subheader("Top evidence")
        for s in snippets:
            with st.expander(f"{s['doc_id']}:{s['chunk_id']}"):
                st.write(s["text"])
        prompt = pitch_prompt(company, industry, goal, snippets, focus=focus)
        try:
            data, took = call_groq_chat(prompt)
            out_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            out_text = f"Error from Groq call: {e}\n\n(See raw response if available.)"
            took = 0
        st.subheader("Generated Draft")
        st.write(out_text)
        st.caption(f"Model call took {took:.2f}s")

elif page == "Insights":
    st.header("Lead Insights")
    file = st.file_uploader("Upload CRM CSV for lead analysis", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            st.write("**CRM Data Preview:**")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train Clustering Model"):
                    with st.spinner("Training clustering model..."):
                        try:
                            ml.train(csv_path=file)
                            st.success("✅ Clustering model trained successfully!")
                        except Exception as e:
                            st.error(f"Error training model: {e}")
            
            with col2:
                if st.button("Predict Lead Segments"):
                    with st.spinner("Predicting segments..."):
                        try:
                            res = ml.predict(file)
                            st.write("**Predicted Segments:**")
                            st.dataframe(res.head())
                            
                            # Show segment distribution
                            if 'predicted_segment' in res.columns:
                                segment_counts = res['predicted_segment'].value_counts()
                                st.write("**Segment Distribution:**")
                                st.bar_chart(segment_counts)
                        except Exception as e:
                            st.error(f"Error predicting segments: {e}")
        except Exception as e:
            st.error(f"Error reading CRM file: {e}")

elif page == "Eval":
    st.header("Evaluation")
    st.write("Upload evaluation CSV with columns: query,relevant_docs")
    eval_file = st.file_uploader("Upload evaluation CSV", type=["csv"])
    if eval_file:
        try:
            df = pd.read_csv(eval_file)
            st.write("**Evaluation dataset preview:**")
            st.dataframe(df.head())
            
            k_values = st.multiselect("Select k values to evaluate:", [1, 3, 5, 10], default=[5, 10])
            
            if st.button("Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    results = {}
                    
                    for k in k_values:
                        try:
                            score = recall_at_k(eval_file, k=k)
                            results[f"Recall@{k}"] = score
                        except Exception as e:
                            st.error(f"Error computing Recall@{k}: {e}")
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    
                    # Create metrics display
                    cols = st.columns(len(results))
                    for i, (metric, score) in enumerate(results.items()):
                        with cols[i]:
                            st.metric(metric, f"{score:.3f}")
                    
                    # Show results table
                    results_df = pd.DataFrame(list(results.items()), columns=["Metric", "Score"])
                    st.dataframe(results_df)
                    
        except Exception as e:
            st.error(f"Error reading evaluation file: {e}")
