import streamlit as st
from agent import SimpleResearchAgent
import os

# Hardcoded API keys
os.environ["SERPAPI_KEY"] = "be4114e1784ecaea4822b29c119fb45a0246c5e0581670a33b7dcccd24c5e9c2"
os.environ["GEMINI_API_KEY"] = "AIzaSyArW17vrJ96miwZ5SRNuFENZcJQULrtck0"

# Configure page
st.set_page_config(
    page_title="Web Scraping Agent",
    page_icon="🔍",
    layout="centered"
)

# Minimalist styling
st.markdown("""
<style>
    .stMarkdown a {
        color: #1E88E5 !important;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    .source-box {
        padding: 12px;
        margin-bottom: 12px;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.title("🔍 Web Scraping Agent")
st.write("Enter your question below to get research results from multiple sources.")

# Research form
query = st.text_input(
    "Research question:",
    placeholder="e.g. What are the latest advancements in AI?"
)

if st.button("Get Results"):
    if not query:
        st.warning("Please enter a question")
        st.stop()
    
    with st.spinner("Researching..."):
        try:
            agent = SimpleResearchAgent()
            result = agent.research(query)  # Always fetching 5 results now
            
            if result.answer.startswith(("Error", "No relevant")):
                st.error(result.answer)
            else:
                st.subheader("Research Findings")
                st.markdown(result.answer)
                
                # Confidence indicator
                st.caption(f"Confidence: {result.confidence*100:.0f}%")
                
                # Simple source display
                st.subheader("Sources")
                for i, source in enumerate(result.sources, 1):
                    with st.expander(f"Source {i}: {source.title}"):
                        st.markdown(f"[View Link]({source.url})")
                        if source.date:
                            st.caption(f"Published: {source.date}")
                        st.write(source.snippet)
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
st.caption("Powered by SerpAPI and Google Gemini")
