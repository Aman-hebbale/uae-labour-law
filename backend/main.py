import os
from fastapi import FastAPI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Embeddings (Kept local for speed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load and Split PDF
file_path = "D:\\projects for dubai resume\\chat with pdf\\UAE_Labour_Law.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()

# For a 1B model, smaller chunks are better to avoid "distraction"
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Setup Vectorstore
# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="chroma_data/",
#     collection_name="uae_law_docs"
# )
CHROMA_DATA_PATH = "D:\\projects for dubai resume\\chat with pdf\\chroma_data"
COLLECTION_NAME = "uae_law_docs"
if os.path.exists(CHROMA_DATA_PATH):
    vectorstore = Chroma(
        persist_directory=CHROMA_DATA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DATA_PATH,
        collection_name=COLLECTION_NAME
    )
# print(f"Vectorstore has {vectorstore._collection.count()} documents.")
# k=2 is better for 1B models to keep the prompt focused
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Initialize Local 1B Model
# Point this to your quantized 1B .gguf file
model_path = "D:\\projects for dubai resume\\chat with pdf\\llama-1b-instr-Q4_K_M.gguf" 

llm = ChatLlamaCpp(
    model_path=model_path,
    temperature=0,           # CRITICAL: Keep at 0 for 1B models to prevent guessing
    max_tokens=256,         # 1B models give better answers when concise
    n_ctx=2048,             # 1B doesn't need huge context for simple chunks
    n_batch=512,
    verbose=False
)

# 5. The "Strict" Prompt Template
# 1B models need very clear boundaries (using ###) to separate data from instructions
template = """### Instructions:
You are a legal assistant specializing in UAE Labor Law. Use the provided context to answer the question. 
If the answer is not in the context, state that you do not know. Do not use outside knowledge.

### Context:
{context}

### Question:
{question}

### Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Build the Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()
@app.get("/ask")
async def ask_question(query: str):
    # 1. Search Pinecone for context (simplified)
    # 2. Feed context + query to Gemini
    response = rag_chain.invoke(query)
    return {"answer": response}


# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision

# questions = ["who is a worker?",
# "what is wage?"]

# questions = ["who is a worker?",
# "what is wage?",
# "What is classified as occupational injury?",
# "what is the official language of instructions?",
# "are employees responsible for court fee during a case?",
# "what calendar is used in the UAE?",
# "can a licensed employer charge a fee for providing employment?",
# "under what conditions can work permit of a non national be cancelled?",
# "what is the age limit for work in the UAE?",
# "how frequently can children take breaks during work?",
# "can children do overtime work?",
# "are men and women treated differently under the law?",
# "how are women compensated for maternity leave if they have worked for less than an year?",
# "can women take extra leave if they have complications during pregnancy?( not in the text but a common question)?",
# "can women take leave after exhausting their maternity leave?",
# "How many copies of an employment contract must be made?",
# "What is the maximum duration for a probation period in the UAE?",
# "How can the terms of employment be proven if no written contract exists?",
# "Can an employer place a worker on probation multiple times?",
# "What is the minimum age required for a person to enter into an apprenticeship contract?",
# "Who must represent an apprentice under the age of 18 when concluding a training contract?",
# "What must an employer provide to a trainee upon the completion of each training phase?",
# "How often must workers on a yearly or monthly wage basis be paid?",
# "What forms of evidence are admissible to prove the payment of wages?",
# "Can an employer force a worker to buy food from a specific shop?"
#             ]

# ground_truths = ["""Any male or female working, for wage of any kind, in the service orunder the management or control of an employer, albeit out of his sight. This
# term applies also to labourers and employees who are in an employer's serviceand are governed by the provisions of this Law.""",
# """Any consideration, in cash or in kind, given to a worker, in return for his service under an employment contract, whether on yearly, monthly,
# weekly, daily, hourly, piece meal, output or commission basis."""]

# ground_truths = ["""Any male or female working, for wage of any kind, in the service orunder the management or control of an employer, albeit out of his sight. This
# term applies also to labourers and employees who are in an employer's serviceand are governed by the provisions of this Law.""",
# """Any consideration, in cash or in kind, given to a worker, in return for his service under an employment contract, whether on yearly, monthly,
# weekly, daily, hourly, piece meal, output or commission basis.""",
# """Any of the work-related diseases listed in the schedule attached hereto or any other injury sustained by a worker during and
# by reason of carrying out his duties. Any accident sustained by a worker on his way to or back from work shall be considered an occupational injury, provided
# that the journey to and from work is made without any break, lingering or diversion from the normal route.""",
# """Arabic shall be the language to be used in all records, contracts, files, data, etc. """,
# """Actions initiated by employees or their beneficiaries under this Law shall be exempt from court fees at all stages of litigation and execution, and shall be dealt with in an
# expeditious manner.""",
# """The periods and dates referred to herein shall be calculated according to the Gregorian calendar.""",
# """No licensed employment agent or labour supplier shall demand or accept from any worker, whether before or after the latter's admission to employment, any
# commission or material reward in return for employment, or charge him for any expenses thereby incurred, except as may be prescribed or approved by the Ministry
# of Labour and Social Affairs.""",
# """The Ministry of Labour and Social Affairs may cancel a work permit granted to a nonNational in the following cases:
# 1. If the worker remains unemployed for more than three consecutive months.
# 2. If the worker no longer meets one or more of the conditions on the basis of
# which the permit was granted.
# 3. If it is satisfied that a particular National is qualified to replace the nonNational worker, in which case the latter shall remain in his job until the
# expiry date of his employment contract or of his employment permit,
# whichever is earlier. """,
# """It shall not be allowed to employ children under the age of 15. """,
# "breaks shall be arranged so that no child works for more than four consecutive hours",
# """Children shall under no circumstances be required to work overtime, or to remain at
# the workplace after their prescribed working hours, or be employed on a rest day. """,
# "A female wage shall be equal to that of a male if she performs the same work. ",
# "They are entitled to a 45 day maternity leave which includes both pre and postnatal leave with half pay",
# "The law does not specify this, so the answer should be that it is unknown based on the provided context.",
# "yes, they are considered as unpaid leave which is allowed for a maximum of 100 days",
# "An employment contract must be written in duplicate; one copy is for the worker and the other is for the employer.",
# "A worker may be employed on probation for a period not exceeding six months.",
# "In the absence of a written contract, adequate proof of its terms may be established by all admissible means of evidence.",
# "No. A worker shall not be placed on probation more than once with the same employer.",
# "A person must have completed at least 12 years of age to enter into an apprenticeship contract.",
# "Apprentices under the age of 18 must be represented by their natural guardians, legal trustees, or personal ad litem.",
# "The employer must issue the trainee a certificate on completion of each phase of training and a final certificate upon completion of the entire period.",
# "Workers employed on a yearly or monthly wage basis must be paid at least once a month.",
# "Evidence of wage payment must be in the form of documentary proof, admission, or oath.",
# "No. Workers shall not be required to purchase food or other commodities at any particular shop or of the employer's produce."]
# answers = []
# contexts = []

# # Inference
# for query in questions:
#     answers.append(rag_chain.invoke(query))
#     contexts.append([doc.page_content for doc in retriever.invoke(query)])

# # To dict
# data = {
# "question": questions,
# "answer": answers,
# "contexts": contexts,
# "reference": ground_truths
# }

# Convert dict to dataset
# dataset = Dataset.from_dict(data)

# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper

# llm = ChatLlamaCpp(
#     model_path="D:/Birmingham uni/sem2-assignment/llama-8b-Q4_K_M.gguf",
#     temperature=0,           # CRITICAL: Keep at 0 for 1B models to prevent guessing
#     max_tokens=2048,         # 1B models give better answers when concise
#     n_ctx=8192,             # 1B doesn't need huge context for simple chunks
#     n_batch=512,
#     verbose=False
# )

# langchain_llm = LangchainLLMWrapper(llm)
# langchain_embeddings = LangchainEmbeddingsWrapper(embeddings)

# you can also use custom LLMs and Embeddings here but make sure 
# they are subclasses of BaseRagasLLM and BaseRagasEmbeddings

# from ragas.run_config import RunConfig

# config = RunConfig(
#     max_workers=1,       # Forces sequential processing so the GPU isn't overwhelmed
#     timeout=1000,         # 5 minutes per query - local 8B can be slow
#     max_retries=2,       
# )

# result = evaluate(
#     dataset =dataset,
#     metrics=[
#         context_precision,
#         context_recall,
#         faithfulness,
#         answer_relevancy,
#     ],
#     llm=langchain_llm,
#     run_config=config,
#     embeddings=langchain_embeddings,
#     batch_size=1
# )

# df = result.to_pandas()
# df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
# if hasattr(llm, 'client'):
#     llm.client.close()