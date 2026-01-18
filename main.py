import json
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv(override=True)

client = OpenAI(
     api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


class Agent:
    """
    The base class for an agent that can interact with the OpenAI API.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = client
        self.model = model
        self.messages: list[Dict[str, Any]] = []


class SearchConfig(BaseModel):
    """
    A search configuration.
    """
    search_terms: list[str]


class WebSearchAgent(Agent):
    """
    A web search agent that uses the tools to search the web.
    """

    def __init__(self):
        super().__init__()
        print(">> [System] Initializing WebSearchAgent...")
        self._set_initial_prompt()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment variables for Google Custom Search.")
        self.service = build("customsearch", "v1", developerKey=self.google_api_key)

    def _set_initial_prompt(self):
        """
        Sets the initial prompt for the agent.
        """
        self.messages = [
            {
                "role": "developer",
                "content": f"""
                You are an expert in performing web searches.
                You will be given a research topic and you will need to derive a list of three and only three search terms that will be used to perform the search.
                The search terms should be derived from the research topic and should be as specific as possible.
                Focus on deriving impactful search terms that will help the user find the most relevant information.
                """
            }
        ]

    def run(self, research_topic: str):
        """
        Runs the agent.
        """
        print(f">> [WebSearchAgent] Received research topic: '{research_topic}'")
        print(">> [WebSearchAgent] Asking LLM to derive search terms...")
        
        self.messages.append(
            {"role": "user", "content": "Here's the research topic based on which you should derive search terms: " + research_topic + "\n\nReturn the search configuration as a JSON object that adheres to the following schema:\n" + json.dumps(SearchConfig.model_json_schema(), indent=2)}
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            response_format={"type": "json_object"},
        )

        json_output = response.choices[0].message.content
        search = SearchConfig.model_validate_json(json_output)
        
        print(f">> [WebSearchAgent] Generated search terms: {search.search_terms}")
        
        results = []

        for search_term in search.search_terms:
            print(f">> [WebSearchAgent] Executing Google Search for: '{search_term}'...")
            try:
                search_results = self.service.cse().list(
                    q=search_term,
                    cx=self.google_cse_id,
                    num=3, # Number of search results to return
                ).execute()

                if 'items' in search_results:
                    print(f"   -> Found {len(search_results['items'])} results for '{search_term}'")
                    for item in search_results['items']:
                        results.append({
                            "search_term": search_term,
                            "url": item.get('link'),
                            "title": item.get('title'),
                            "description": item.get('snippet'),
                        })
                else:
                    print(f"   -> No results found for '{search_term}'")
                    
            except HttpError as e:
                print(f"!! [Error] Google Custom Search failed for '{search_term}': {e}")
            except Exception as e:
                print(f"!! [Error] Unexpected error for '{search_term}': {e}")

        print(f">> [WebSearchAgent] Total results collected: {len(results)}")
        return results


class SummaryReportAgent(Agent):
    """
    A summary report agent that uses the tools to summarize the search results.
    """

    def __init__(self):
        super().__init__()
        print(">> [System] Initializing SummaryReportAgent...")
        self._set_initial_prompt()

    def _set_initial_prompt(self):
        """
        Sets the initial prompt for the agent.
        """
        self.messages = [
            {
                "role": "developer",
                "content": """
                You are a summary report agent.
                You will be given a list of search results (which include short descriptions) and you will need to summarize them into a report.
                The report should be in a format that is easy to understand and use.
                It's important that your report includes those URLs (next to the text they belong to) so that users can dive deeper.
                The report should be in Markdown format. Avoid extra explanations, annotations or text, just return the markdown report.
                """
            }
        ]

    def run(self, search_results: list[Dict[str, Any]]):
        """
        Runs the agent.
        """

        print(">> [SummaryReportAgent] Sending search results to LLM for summarization...")
        self.messages.append(
            {"role": "user", "content": "Please create a summary (and keep the links!) based on these search results: " + json.dumps(search_results, indent=2)}
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        report = response.choices[0].message.content
        
        print(">> [SummaryReportAgent] Received summary from LLM.")
        
        if report.startswith("```markdown"):
            report = report[11:]
        if report.endswith("```"):
            report = report[:-3]
        return report

def main_research_flow():
    """Main function to orchestrate the research and reporting process."""
    print(">> [System] Starting Research Flow...")
    research_topic = input("Enter the topic you want to research: ")

    search_agent = WebSearchAgent()
    results = search_agent.run(research_topic)

    if not results:
        print("!! [System] No search results found. Exiting.")
        return

    summary_report_agent = SummaryReportAgent()
    summary_report = summary_report_agent.run(results)

    output_file = research_topic.replace(" ", "_") + "_report.md"
    print(f">> [System] Saving report to file: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary_report)
    print(f">> [System] Research report saved successfully to '{output_file}'.")


if __name__ == "__main__":
    main_research_flow()
