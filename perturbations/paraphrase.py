from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from .prompt_template import PP_TEMPLATE


def llm_prompting(input, method=1, llm=None):
    """
    Generate a paraphrase from input text using a language model (LLM) based on a specified template.

    :param str input_text: The text to be processed.
    :param int method: The method number to be used for processing. Optional.
    :return: The processed text output from the language model.
    :rtype: str
    """
    pp_prompt = PromptTemplate(
        input_variables=["method", "sentence"],
        template=PP_TEMPLATE,
    )
    if llm is None:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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",)
    messages = [
        HumanMessage(content=pp_prompt.format(method=str(method), sentence=input))
    ]
    input_pp = llm(messages, stop="\n").content
    return input_pp.strip()
