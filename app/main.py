
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
loader = TextLoader("data/dummy.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)


genai.configure(api_key="AIzaSyA3tr-tEqwVX_arTmMSe4nqjez7G_ZKE2Y")

# Set up the model
generation_config = {
  "temperature": 0.0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)




    
def make_rag_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the question , respond with 'no data found'.
  
  PASSAGE: '{relevant_passage}'
  QUESTION: '{query}'

  ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

def search(query):
    docs = db.similarity_search(query)
    return ''.join(doc.page_content for doc in docs[:3])

def get_response(chat):
    convo = model.start_chat(history=chat[:-1])
    convo.send_message(chat[-1]["parts"][-1]["text"])

    return convo.last.text



class UserCreate(BaseModel):
    user_id: int
    username: str


    
class ChatRequest(BaseModel):
    contents: list

name = []
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test/")
async def test():
    return {"test":"ok"}
@app.post("/create_user/")
async def create_user(user_data: UserCreate):
    user_id = user_data.user_id
    username = user_data.username
    return {
        "msg": "we got data succesfully",
        "user_id": user_id,
        "username": username,
    }


@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    query = chat_request.contents[-1]["parts"][-1]["text"]
    chat_request.contents[-1]["parts"][-1]["text"] = make_rag_prompt(query,search(query))
    response = get_response(chat_request.contents)

    return {"response": response}
 
 