
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# --- 1. CONFIGURATION & MODELS ---
st.set_page_config(page_title="FIFA Agent", page_icon="⚽️")

@st.cache_resource
def load_models_and_db():
    # Load the embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="fifa_db")
    collection = client.get_or_create_collection(name="fifa_collection")

    # Initialize Groq
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_KEY"]
    groq_client = Groq()

    return model, collection, groq_client

model, collection, groq_client = load_models_and_db()

# --- 2. DATA LOADING & INDEXING ---
# This part only runs if the database is empty!
if collection.count() == 0:
    with st.status("Initializing Knowledge Base for the first time...", expanded=True) as status:
        st.write("Reading dataset...")
        df = pd.read_csv(r"Dataset/agent_knowledge_base.csv")
        for i, row in df.iterrows():
            embedding = model.encode(row["Profile"]).tolist()
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[row["Profile"]],
                metadatas=[{
                    "name": row["Name"],
                    "age": int(row["Age"]),
                    "growth": int(row["Potential_Growth"]),
                    "potential": int(row["Potential"]),
                    "is_wonder_kid": int(row["is_wonder_kid"])
                }]
            )
        status.update(label="Knowledge Base Ready!", state="complete", expanded=False)

# --- 3. RAG LOGIC ---
def get_scouting_report(user_query):
    # Vector Search
    query_vector = model.encode(user_query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )
  
    if results["documents"] and len(results["documents"][0]) > 0:
      context = "\n\n---\n\n".join(results["documents"][0])
    else:
      context = "NO DATA FOUND"

    system_prompt = f"""
    You are a specialized FIFA/Football Scouting Agent. 
    
    STRICT GUARDRAIL:
    - You ONLY answer questions related to football, FIFA players, scouting, or team tactics.
    - If the user's request is NOT about football (e.g., math, history, coding, or general chat), strictly respond with: 
      "I am sorry, but I am a dedicated FIFA Scouting Agent. I can only assist with football-related queries."
    - DO NOT answer the non-football question even if you know the answer.

    INSTRUCTIONS FOR FOOTBALL QUERIES:
    - Provide a list of the TOP 5 players that match the user's request.
    - Present the result as a numbered list (1 to 5).
    - Bold the **Player Name** at the start of each line.
    
    PLAYER DATA FROM DATABASE:
    {context}
    """

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7
    )
    return response.choices[0].message.content

# --- 4. STREAMLIT UI ---
st.title("PLAYER RECOMMENDATION AGENT")
st.markdown("Find the player you want using AI agent!")

user_input = st.text_input("Describe the target player", placeholder="e.g. A left-footed winger with high growth potential")

if user_input:
    with st.spinner("Agent is scouting the database..."):
        try:
            output = get_scouting_report(user_input)
            st.subheader("Scouting Report:")
            st.success(output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
