PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)
PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)
RESPONSE_HEADER_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
RESPONSE_HEADER_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)



def format_prompt(instruction, inp, response=""):
    if inp.strip():
        return PROMPT_WITH_INPUT.format(
            instruction=instruction, input=inp, response=response
        )

    return PROMPT_NO_INPUT.format(
        instruction=instruction, response=response
    )


def format_response_header(instruction, inp):
    if inp.strip():
        return RESPONSE_HEADER_WITH_INPUT.format(
            instruction=instruction, input=inp
        )

    return RESPONSE_HEADER_NO_INPUT.format(
        instruction=instruction
    )

