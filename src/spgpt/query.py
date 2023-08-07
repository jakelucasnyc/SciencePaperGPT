from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


def get_response_from_query(db:FAISS, query:str, temperature:float, k:int=8):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about the provided scientific publication,
        sourced from the pubication's pdf file on the internet: {docs}
        
        Only use the factual information from the publication to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    # template = """
    #     You are a helpful assistant that that can answer questions about the provided non-fiction book,
    #     sourced from a pdf file on the internet: {docs}
        
    #     Only use the factual information from the book to answer the question.
        
    #     If you feel like you don't have enough information to answer the question, say "I don't know".
        
    #     Your answers should be verbose and detailed.
    #     """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question} \nGive a verbose, detailed answer using basic vocabulary as much as possible."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs