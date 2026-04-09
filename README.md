# ⚽ FIFA Scout AI: Next-Gen RAG Agent (FC 26 Edition)

An advanced AI-powered football scouting assistant that leverages **Retrieval-Augmented Generation (RAG)** to provide intelligent player recommendations from a dataset of 15,000+ players.

## 🚀 Key Features
- **Semantic Scouting:** Uses vector embeddings to understand player "profiles" instead of just keyword matching.
- **FC 26 Ready:** Integrated with the latest player stats, including synthetic "Potential" and "Wonderkid" metrics.
- **LLM Guardrails:** Specialized logic to ensure the agent stays focused on football and scouting intelligence.
- **Dynamic Context:** Delivers the Top 5 most relevant players with detailed reasoning for each selection.

## 🎯 Use Cases
This agent is designed for football enthusiasts and data-driven gamers who need more than just a filter tool.
- **Career Mode Enthusiasts:** Quickly find "hidden gems" or wonderkids with high growth potential for long-term builds without manually checking hundreds of scout reports.
- **Tactical Analysts:** Search for specific player profiles based on playstyles (e.g., "Find me a left-footed playmaker with high agility and vision").
- **Ultimate Team (FUT) Strategists:** Compare player profiles and attributes to find budget alternatives for high-end cards based on technical stats.
- **Football Data Curious:** Explore the FC 26 database through a natural language interface instead of traditional spreadsheet sorting.

## 🛠️ Technical Stack
- **Framework:** Streamlit
- **Vector DB:** ChromaDB (with Git LFS for persistent storage)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM:** Llama 3.1 8B via Groq (LPU Inference)

## 📁 Project Structure
- `app.py`: Main application logic and RAG pipeline.
- `Dataset/`: Source CSV data for the knowledge base.
- `fifa_db/`: Pre-indexed vector database (stored via Git LFS).
- `requirements.txt`: Python dependencies for deployment.

## 👷 Author
**Neha Anil** *Graduate Artificial Intelligence Engineer | Specialized in Data Science*
