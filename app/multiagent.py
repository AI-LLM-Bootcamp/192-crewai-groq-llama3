import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import tiktoken

load_dotenv()  # the .env file must include GROQ_API_KEY=... and TAVILY_API_KEY=...

# Original version: this example used llama3-70b-8192 (or mixtral-8x7b-32768).
# Groq has since retired both models, and Llama is now offered only to
# enterprise customers. Today we use OpenAI's open-weight gpt-oss models,
# which work with the same free Groq API key.
# Groq retires models often: if you see "model decommissioned" or
# "model not found", check https://console.groq.com/docs/models
# and pick a model marked "Production".
MODEL = "openai/gpt-oss-20b"  # or "openai/gpt-oss-120b", the most powerful one

# CrewAI counts the tokens of each request with the tiktoken library.
# The model name contains "gpt", so CrewAI asks tiktoken for its encoding,
# and this version of tiktoken does not know it: it would print a warning
# ("Could not automatically map openai/gpt-oss-20b to a tokeniser") on
# every call. We tell it to use cl100k_base, an encoding this version
# includes. It is only used to estimate tokens; nothing sent to Groq changes.
tiktoken.model.MODEL_TO_ENCODING[MODEL] = "cl100k_base"

# gpt-oss models reason before they answer. With the default effort,
# gpt-oss-20b can spend its tokens thinking and return an empty answer.
# Low reasoning effort leaves room for the answer itself. It goes through
# extra_body because this version of langchain-groq predates that option.
llm = ChatGroq(
    model=MODEL,
    model_kwargs={"extra_body": {"reasoning_effort": "low"}}
)

@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

tools = [TavilySearchResults(max_results=1), process_search_tool]

online_researcher = Agent(
    role="Online Researcher",
    goal="Research the topic online",
    backstory="""Your primary role is to function as an intelligent online research assistant. You possess the capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

blog_manager = Agent(
    role="Blog Manager",
    goal="Write the blog article",
    backstory="""You are a Blog Manager. The role of a Blog Manager encompasses several critical responsibilities aimed at transforming initial drafts provided by the online researcher into polished, SEO-optimized blog articles that engage and grow an audience.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

social_media_manager = Agent(
    role="Social Media Manager",
    goal="Write a tweet",
    backstory="""You are a Social Media Manager. The role of a Social Media Manager, particularly for managing Twitter content, involves transforming the drafts from the online researche into concise, engaging tweets that resonate with the audience and adhere to platform best practices.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

content_marketing_manager = Agent(
    role="Content Marketing Manager",
    goal="Manage the Content Marketing Team",
    backstory="""You are an excellent Content Marketing Manager. Your primary role is to supervise each publication from the 'blog manager' 
    and the tweets written by the 'social media manager' and approve them if they do not have profanity and are aligned with the initial report of the online researcher.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

task1 = Task(
    description="""Write me a report on the last Real Madrid-Barcelona soccer match. After the research online,pass the findings to the blog manager to generate a blog article. Once done, pass the finding to the social media 
    manager to write a tweet on the subject.""",
    expected_output="Report on the last Real Madrid-Barcelona soccer match",
    agent=online_researcher
)

task2 = Task(
    description="""Using the research findings of the online researcher, write a blog post of at least 3 paragraphs.""",
    expected_output="Blog Post on the last Real Madrid-Barcelona soccer match",    
    agent=blog_manager
)

task3 = Task(
    description="""Using the research findings of the online researcher, write a tweet.""",
    expected_output="Tweet on the last Real Madrid-Barcelona soccer match",
    agent=social_media_manager
)

task4 = Task(
    description="""Review the final output from both the blog manager and social media manager and approve them if they do not have profanity and are aligned with the initial report of the online researcher.""",
    expected_output="Final decision on the publication of the Blog Post and Tweet on the last Real Madrid-Barcelona soccer match",
    agent=content_marketing_manager
)

agents = [online_researcher, blog_manager, social_media_manager, content_marketing_manager]

crew = Crew(
    agents=agents,
    tasks=[task1, task2, task3, task4],
    verbose=2
)

result = crew.kickoff()

print(result)