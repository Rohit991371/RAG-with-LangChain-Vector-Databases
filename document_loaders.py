import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import(
    TextLoader
)

from dotenv import load_dotenv

load_dotenv()