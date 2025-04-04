from setuptools import setup, find_packages

setup(
    name="my_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langchain_anthropic",
        "tavily-python",
        "langchain_community",
        "langchain_openai",
        "streamlit"
    ],
) 