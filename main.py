import os
import shutil
import logging

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain.callbacks import FileCallbackHandler, StreamingStdOutCallbackHandler

from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent, load_tools

logging.getLogger("openai").setLevel(logging.DEBUG)



# 各フォルダのベクターストア化する処理

all_docs = []
for folder in folders:
  if folder == '.db':
    continue

  docs = []
  #  顧客, .db以外のフォルダのドキュメント化、ファイルコピーの処理
  if folder not in ['顧客', '.db']:
    unpersisted_folder_path = os.path.join(dir_path, folder, 'データベース化前')
    persisted_folder_path = os.path.join(dir_path, folder, 'データベース化済み')

    unpersisted_files = os.listdir(unpersisted_folder_path)
    persisted_files = os.listdir(persisted_folder_path)

    print(f'{folder}フォルダ処理中...')
    print(f'データベース化前ファイル: {unpersisted_files}')
    print(f'データベース化済みファイル: {persisted_files}')
    print('')

    if len(unpersisted_files) > 0:
      for file in unpersisted_files:
        if file not in persisted_files:
          print('ファイルドキュメント化、コピー中...')
          pages = Docx2txtLoader(os.path.join(unpersisted_folder_path, file)).load()
          docs.extend(pages)
          all_docs.extend(pages)

          unpersisted_file_path = os.path.join(unpersisted_folder_path, file)
          dst_folder = os.path.join(persisted_folder_path, file)
          shutil.copy(unpersisted_file_path, dst_folder)
          print('ファイルドキュメント化、コピー完了')
          print('')
  #  顧客フォルダのドキュメント化、ファイルコピーの処理
  elif folder == '顧客':
    prospect_customers_path = os.path.join(dir_path, folder, '見込み')
    existing_customers_path = os.path.join(dir_path, folder, '既存')
    prospect_customers = os.listdir(prospect_customers_path)
    existing_customers = os.listdir(existing_customers_path)
    customers = prospect_customers + existing_customers

    for customer in customers:
      if customer in prospect_customers:
        unpersisted_folder_path = os.path.join(prospect_customers_path, customer, 'データベース化前')
        persisted_folder_path = os.path.join(prospect_customers_path, customer, 'データベース化済み')
      else:
        unpersisted_folder_path = os.path.join(existing_customers_path, customer, 'データベース化前')
        persisted_folder_path = os.path.join(existing_customers_path, customer, 'データベース化済み')

      unpersisted_files = os.listdir(unpersisted_folder_path)
      persisted_files = os.listdir(persisted_folder_path)

      print(f'顧客/{customer}フォルダ処理中...')
      print(f'データベース化前ファイル: {unpersisted_files}')
      print('')
      print(f'データベース化済みファイル: {persisted_files}')
      print('')

      if len(unpersisted_files) > 0:
        for file in unpersisted_files:
          if file not in persisted_files:
            print('ファイルドキュメント化、コピー中...')
            pages = Docx2txtLoader(os.path.join(unpersisted_folder_path, file)).load()
            docs.extend(pages)
            all_docs.extend(pages)

            unpersisted_file_path = os.path.join(unpersisted_folder_path, file)
            dst_folder = os.path.join(persisted_folder_path, file)
            shutil.copy(unpersisted_file_path, dst_folder)
            print('ファイルドキュメント化、コピー完了')
            print('')
  # ベクターストア化の処理
  if len(docs) == 0:
    print(f'{folder}フォルダは最新の状態です。')
    print('')
    continue
  if len(docs) > 0:
    print(f'.{folder}_chromadb作成中...')

    text_splitter = CharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=50,
      separator='\n'
    )

    splitted_text = text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()

    vector_folder_path = os.path.join(dir_path, '.db', f'.{folder}_chromadb')
    print(vector_folder_path)
    if os.path.isdir(vector_folder_path):
      db = Chroma(persist_directory=vector_folder_path, embedding_function=embedding)
      db.add_documents(splitted_text)
      print('ベクターストア更新')
      print('')
    else:
      Chroma.from_documents(splitted_text, embedding, persist_directory=vector_folder_path)
      print('ベクターストア新規作成')
      print('')


# 全フォルダ共通のベクターストア化
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separator='\n'
)

splitted_text = text_splitter.split_documents(all_docs)

embedding = OpenAIEmbeddings()

vector_folders_path = os.path.join(dir_path, '.db', '.all_chromadb')

if len(all_docs) == 0:
  all_db = Chroma(persist_directory=vector_folders_path, embedding_function=embedding)
  print(f'all_chromadbは最新の状態です。')

if len(all_docs) > 0:
  print('.all_chromadb作成中...')
  print('')


  if os.path.isdir(vector_folders_path):
    all_db = Chroma(persist_directory=vector_folders_path, embedding_function=embedding)
    all_db.add_documents(splitted_text)
    print('ベクターストア更新')
    print('')
  else:
    all_db = Chroma.from_documents(splitted_text, embedding, persist_directory=vector_folders_path)
    print('ベクターストア新規作成')
    print('')
retriever = all_db.as_retriever()

# RAGシステムに会話履歴の記憶機能の実装
question_generator_template = '会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。'

question_generator_prompt = ChatPromptTemplate.from_messages(
    [
      ('system', question_generator_template),
        MessagesPlaceholder('chat_history'),
      ('human', '{input}'),
    ]
)

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0.5,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True
    )

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, question_generator_prompt
)

question_answer_template = """
あなたは優秀な質問応答アシスタントです。以下のcontextを使用して質問に答えてください。
また答えが分からない場合は、無理に答えようとせず「分からない」という旨を答えてください。
{context}
"""

question_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', question_answer_template),
          MessagesPlaceholder('chat_history'),
        ('human', '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

query = '多くの企業が力を入れているマーケティング施策は何ですか？'

ai_msg = rag_chain.invoke({'input': query, 'chat_history': chat_history})

chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg['answer'])])

# print(ai_msg['answer'])

query = '100文字以内に要約して'

ai_msg = rag_chain.invoke({'input': query, 'chat_history': chat_history})

# print(ai_msg['answer'])
chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg['answer'])])

# データベース化済みファイルの削除
def delete_persted_files(folders):
  for folder in folders:
    if folder not in ['顧客', '.db']:
      print(f'{folder}配下ファイル処理中...')
      persisted_folder_path = os.path.join(dir_path, folder, 'データベース化済み')
      files = os.listdir(persisted_folder_path)
      if len(files) == 0:
        print(f'{folder}配下にファイルはありません。')
        print('')
        continue
      if len(files) > 0:
        for file in files:
          print(f'[{file}]を削除中...')
          file_path = os.path.join(persisted_folder_path, file)
          os.remove(file_path)
          print(f'{file}を削除しました。')
          print('')
    elif folder == '.db':
      print(f'{folder}配下ファイル処理中...')
      db_path = os.path.join(dir_path, folder)
      db_folders = os.listdir(db_path)
      if len(db_folders) == 0:
        print(f'{folder}配下にフォルダはありません。')
        print('')
        continue
      if len(db_folders) > 0:
        for db_folder in db_folders:
          db_folder_path = os.path.join(db_path, db_folder)
          shutil.rmtree(db_folder_path)
          print(f'{db_folder}を削除しました。')
          print('')
    elif folder == '顧客':
      prospect_customers_path = os.path.join(dir_path, folder, '見込み')
      existing_customers_path = os.path.join(dir_path, folder, '既存')
      prospect_customers = os.listdir(prospect_customers_path)
      existing_customers = os.listdir(existing_customers_path)
      customers = prospect_customers + existing_customers
      for customer in customers:
        if customer in prospect_customers:
          persisted_folder_path = os.path.join(prospect_customers_path, customer, 'データベース化済み')
        else:
          persisted_folder_path = os.path.join(existing_customers_path, customer, 'データベース化済み')
        files = os.listdir(persisted_folder_path)
        if len(files) > 0:
          print(f'{customer}配下ファイル処理中...')
          for file in files:
            print(f'[{file}]を削除中...')
            file_path = os.path.join(persisted_folder_path, file)
            os.remove(file_path)
            print(f'{file}を削除しました。')
            print('')
        else:
          print(f'{customer}配下にファイルはありません。')
          print('')


# 各フォルダのデータベース化済みファイルを出力する処理
def output_persted_files(folders):
  for folder in folders:
    if folder not in ['顧客', '.db']:
      persisted_folder_path = os.path.join(dir_path, folder, 'データベース化済み')
      files = os.listdir(persisted_folder_path)
      print(f'{folder}のデータベース化済みファイル\n{files}')
      print('')
    if folder == '.db':
      db_path = os.path.join(dir_path, folder)
      db_folders = os.listdir(db_path)
      print(f'{folder}のデータベース化済みファイル\n{files}')
      print('')
    if folder == '顧客':
      prospect_customers_path = os.path.join(dir_path, folder, '見込み')
      existing_customers_path = os.path.join(dir_path, folder, '既存')
      prospect_customers = os.listdir(prospect_customers_path)
      existing_customers = os.listdir(existing_customers_path)
      customers = prospect_customers + existing_customers
      for customer in customers:
        if customer in prospect_customers:
          persisted_folder_path = os.path.join(prospect_customers_path, customer, 'データベース化済み')
          files = os.listdir(persisted_folder_path)
        else:
          persisted_folder_path = os.path.join(existing_customers_path, customer, 'データベース化済み')
          files = os.listdir(persisted_folder_path)
        print(f'{customer}のデータベース化済みファイル\n{files}')
        print('')

# 作成したRAGシステムにAIエージェント機能を搭載する
def create_rag_chain(folder):
  target_chromadb_path = os.path.join(dir_path, '.db', f'.{folder}_chromadb')

  embedding = OpenAIEmbeddings()
  db = Chroma(persist_directory=target_chromadb_path, embedding_function=embedding)
  retriever = db.as_retriever()

  question_generator_template = '会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。'

  question_generator_prompt = ChatPromptTemplate.from_messages(

      [

          ('system', question_generator_template),

          MessagesPlaceholder('chat_history'),

          ('human', '{input}'),

      ]

  )

  llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)

  history_aware_retriever = create_history_aware_retriever(

      llm, retriever, question_generator_prompt

  )

  question_answer_template = """

  あなたは優秀な質問応答アシスタントです。以下のcontextを使用して質問に答えてください。

  また答えが分からない場合は、無理に答えようとせず「分からない」という旨を答えてください。"

  {context}

  """

  question_answer_prompt = ChatPromptTemplate.from_messages(

      [

          ('system', question_answer_template),

          MessagesPlaceholder('chat_history'),

          ('human', '{input}'),

      ]

  )

  question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  return rag_chain


# ['採用', 'マーケティング', '営業', '全社', '顧客', '開発', '教育']
recruitment_doc_chain = create_rag_chain('採用')
recruitment_doc_chain_chat_history = []

marketing_doc_chain = create_rag_chain('マーケティング')
marketing_doc_chain_chat_history = []

sales_doc_chain = create_rag_chain('営業')
sales_doc_chain_chat_history = []

company_doc_chain = create_rag_chain('全社')
company_doc_chain_chat_history = []

customer_doc_chain = create_rag_chain('顧客')
customer_doc_chain_chat_history = []

development_doc_chain = create_rag_chain('開発')
development_doc_chain_chat_history = []

education_doc_chain = create_rag_chain('教育')
education_doc_chain_chat_history = []


def run_recruitment_doc_chain(param):
    ai_msg = recruitment_doc_chain.invoke({'input': param, 'chat_history': recruitment_doc_chain_chat_history})
    recruitment_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_marketing_doc_chain(param):
    ai_msg = marketing_doc_chain.invoke({'input': param, 'chat_history': marketing_doc_chain_chat_history})
    marketing_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_sales_doc_chain(param):
    ai_msg = sales_doc_chain.invoke({'input': param, 'chat_history': sales_doc_chain_chat_history})
    sales_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_company_doc_chain(param):
    ai_msg = company_doc_chain.invoke({'input': param, 'chat_history': company_doc_chain_chat_history})
    company_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_customer_doc_chain(param):
    ai_msg = customer_doc_chain.invoke({'input': param, 'chat_history': customer_doc_chain_chat_history})
    customer_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_development_doc_chain(param):
    ai_msg = development_doc_chain.invoke({'input': param, 'chat_history': development_doc_chain_chat_history})
    development_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']

def run_education_doc_chain(param):
    ai_msg = education_doc_chain.invoke({'input': param, 'chat_history': education_doc_chain_chat_history})
    education_doc_chain_chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg['answer'])])
    return ai_msg['answer']


recruitment_doc_tool = Tool.from_function(
    func=run_recruitment_doc_chain,
    name='自社の採用に関する情報を参照するTool',
    description='自社の採用に関する情報を参照したい場合に使う'
)

marketing_doc_tool = Tool.from_function(
    func=run_marketing_doc_chain,
    name='自社のマーケティングに関する情報を参照するTool',
    description='自社のマーケティングに関する情報を参照したい場合に使う'
)

sales_doc_tool = Tool.from_function(
    func=run_sales_doc_chain,
    name='自社の営業に関する情報を参照するTool',
    description='自社の営業に関する情報を参照したい場合に使う'
)

company_doc_tool = Tool.from_function(
    func=run_company_doc_chain,
    name='自社全体に関する情報を参照するTool',
    description='自社全体に関する情報を参照したい場合に使う'
)

customer_doc_tool = Tool.from_function(
    func=run_customer_doc_chain,
    name='自社の顧客に関する情報を参照するTool',
    description='自社の顧客に関する情報を参照したい場合に使う'
)

development_doc_tool = Tool.from_function(
    func=run_development_doc_chain,
    name='自社の開発に関する情報を参照するTool',
    description='自社の開発に関する情報を参照したい場合に使う'
)

education_doc_tool = Tool.from_function(
    func=run_education_doc_chain,
    name='自社の教育に関する情報を参照するTool',
    description='自社の教育に関する情報を参照したい場合に使う'
)

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)

tools = [
    recruitment_doc_tool,
    marketing_doc_tool,
    sales_doc_tool,
    company_doc_tool,
    customer_doc_tool,
    development_doc_tool,
    education_doc_tool,
    ]

agent_executor = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query_1 = '自社メンバーの育成に関する具体的なアクションプランを教えて'

agent_executor.run(query_1)