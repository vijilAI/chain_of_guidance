import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .prompt_template import QUESTION_TEMPLATE


class BaseGenerator:
    """
    Base class for generating variations of outputs based on different methods.

    :param str variation_type: Type of variation method used (e.g., 'sampling', 'paraphrasing').
    :param Model model: The language model used for generating outputs.
    :param PromptTemplate question_prompt: Template for formatting the input question.
    """

    def __init__(self, model, variation_type):
        """
        Initializes the BaseGenerator with a specified model and variation type.

        :param Model model: The language model to use.
        :param str variation_type: The method of variation to apply ('sampling' or 'paraphrasing').
        """
        super(BaseGenerator, self).__init__()
        self.variation_type = variation_type
        self.model = model
        self.question_prompt = PromptTemplate(
            input_variables=["question"],
            template=QUESTION_TEMPLATE,
        )

    def apply(self, input, inputs_pert, outputs):
        """
        Applies additional processing to the generated outputs if needed.

        :param str input: The original input.
        :param list inputs_pert: Perturbed versions of the input.
        :param list outputs: Generated outputs before final processing.
        :return: Processed outputs.
        :rtype: list
        """
        return outputs

    def generate(self, input, input_perts):
        """
        Generates variations of outputs based on the specified variation type.

        :param str input: The original input to generate variations from.
        :param list input_perts: A list of perturbed inputs for variation generation.
        :return: A list of generated output variations.
        :rtype: list
        """
        outputs = []
        if self.variation_type == "sampling":
            for temperature in np.arange(0, 2, 0.2):
                self.model.model_kwargs["temperature"] = temperature
                chain = LLMChain(llm=self.model, prompt=self.question_prompt)
                output = chain.run(
                    {
                        "question": input,
                    }
                )
                outputs.append(output.strip())
        elif self.variation_type == "paraphrasing":
            chain = LLMChain(llm=self.model, prompt=self.question_prompt)
            input_perts = [input] + input_perts
            for input_pert in input_perts:
                output = chain.run(
                    {
                        "question": input_pert,
                    }
                )
                outputs.append(output.strip())
        else:
            NotImplementedError
        outputs = self.apply(input, input_perts, outputs)
        return outputs
