import os
import requests
import google.generativeai as genai
from typing import List
from dataclasses import dataclass
import json
import logging

# Configure logging for better tracking of errors and flow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    date: str = None

@dataclass
class ResearchResult:
    answer: str
    sources: List[SearchResult]
    confidence: float
    contains_news: bool = False

class SimpleResearchAgent:
    def __init__(self):
        """Initialize the research agent, including setting up the API and model."""
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel('gemini-1.5-pro-002')
            self.search_url = "https://serpapi.com/search"
            logger.info("Research agent initialized successfully.")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def search_web(self, query: str, num_results: int = 5, news: bool = False) -> List[SearchResult]:
        """Perform web search using SerpAPI and return top N results."""
        try:
            params = {
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "num": num_results,  # Ensure we request the correct number of results
                "hl": "en",
                "gl": "us"
            }

            if news:
                params["tbm"] = "nws"
                params["tbs"] = "qdr:d"  # Filter for recent news

            response = requests.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Debugging: Log the raw response data to ensure we're getting the correct number of results
            logger.debug(f"SerpAPI Response: {json.dumps(data, indent=2)}")

            # Return the number of results specified by num_results
            return [
                SearchResult(
                    title=result.get("title", "No title"),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", "No snippet"),
                    date=result.get("date") if news else None
                )
                for result in data.get("organic_results", [])[:num_results]  # Now returning exactly num_results
            ]
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return []

    def _parse_gemini_response(self, text: str) -> dict:
        """Parse Gemini's response for structured content."""
        try:
            if '{' in text and '}' in text:
                start = text.index('{')
                end = text.rindex('}') + 1
                return json.loads(text[start:end])
            return {"answer": text, "sources": [], "confidence": 0.7}
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini response, returning raw answer.")
            return {"answer": text, "sources": [], "confidence": 0.7}

    def analyze_content(self, query: str, search_results: List[SearchResult], is_news: bool = False) -> ResearchResult:
        """Analyze the search results and generate an answer using Gemini."""
        if not search_results:
            return ResearchResult(
                answer="No relevant results found.",
                sources=[],
                confidence=0.0,
                contains_news=is_news
            )
        
        try:
            # Combine information from multiple sources
            combined_content = "\n\n".join(
                f"Source {i}:\n"
                f"Title: {r.title}\n"
                f"URL: {r.url}\n"
                f"{'Date: ' + r.date + '\n' if r.date else ''}"
                f"Content: {r.snippet}"
                for i, r in enumerate(search_results, 1)
            )
            
            # Prepare the prompt for Gemini to generate an organized and concise answer
            prompt = f"""Analyze the following search results and provide a well-structured, concise answer that summarizes the key points:

            Query: {query}

            Search Results:
            {combined_content}

            Instructions:
            1. Combine information from multiple sources logically.
            2. Resolve any contradictions by prioritizing authoritative sources.
            3. Organize the answer in a clear, logical structure.
            4. Provide a summary that answers the query.
            5. Cite sources using [1][2] notation.
            6. Include a confidence score (0-1).
            7. Format response as JSON with:
                "answer": "your response",
                "sources": [list of source numbers used],
                "confidence": 0.85
            """
            
            response = self.model.generate_content(prompt)
            result = self._parse_gemini_response(response.text)
            
            # Extract sources from the result
            used_indices = [int(i) for i in result.get("sources", []) if str(i).isdigit()]
            used_sources = [
                search_results[i-1] 
                for i in used_indices 
                if 0 < i <= len(search_results)
            ] or search_results  # Default to all sources if no sources were found
            
            return ResearchResult(
                answer=result.get("answer", "No answer generated"),
                sources=used_sources[:5],
                confidence=min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
                contains_news=is_news
            )
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return ResearchResult(
                answer=f"Error processing information: {e}",
                sources=[],
                confidence=0.0,
                contains_news=is_news
            )

    def research(self, query: str, num_results: int = 5) -> ResearchResult:
        """Complete research workflow: search the web and analyze the results."""
        try:
            if not query or len(query.strip()) < 3:
                return ResearchResult(
                    answer="Please provide a more specific question.",
                    sources=[],
                    confidence=0.0
                )
            
            is_news = any(keyword in query.lower() 
                         for keyword in ["news", "headlines", "today's", "recent", "update"])
            
            search_results = self.search_web(query, num_results=num_results, news=is_news)
            return self.analyze_content(query, search_results, is_news)
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            return ResearchResult(
                answer="An error occurred during research.",
                sources=[],
                confidence=0.0
            )
