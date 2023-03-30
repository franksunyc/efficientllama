import torch
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from peft import PeftModel
# setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#llama_model_path = "/home/data/hdd-1/7B"
#llama_token_model = '/home/data/hdd-1/tokenizer.model'
llama_model_path = ""
if not llama_model_path:
    llama_model_path = "decapoda-research/llama-7b-hf"

# the path to the auxiliary model
efficient_llama_model_path = "auxiliary_model"

if not efficient_llama_model_path:
    raise Exception("Please input your auxiliary model")

llama_tokenizer = LLaMATokenizer.from_pretrained(llama_token_model)

if device != "cpu":
    # load the weights into gpu
    # load base quantized model
    llama_model = LLaMAForCausalLM.from_pretrained(llama_model_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
    # load fine-tuned weights
    llama_model = PeftModel.from_pretrained(llama_model, efficient_llama_model_path, torch_dtype=torch.float16)
else:
    # set the weights into cpu
    device_map = {"": device}
    # load base model
    # if working on cpu then we want to shrink the memory usage
    llama_model = LLaMAForCausalLM.from_pretrained(llama_model_path, device_map=device_map, low_cpu_mem_usage=True)
    # load fine-tuned weights
    llama_model = PeftModel.from_pretrained(llama_model, efficient_llama_model_path, device_map=device_map)

instruction_only_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{_instruction}

### Response:
"""

instruction_and_input_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{_instruction}

### Input:
{_input}

### Response:
"""

def get_model_input(_instruction: str, _input: str) -> str:
    """This function gernerates a template to feed into LLM
    input:
        _instruction: string
        _input: string

    return:
        string
    """
    if _input:
        return instruction_and_input_template.format(_instruction=_instruction, _input=_input)
    else:
        return instruction_only_template.format(_instruction=_instruction)


def run(_instruction: str,
        _input: str = None,
        temperature: float = 0.1,
        top_p: float = 0.75,
        num_beams: int = 4,
        max_len: int = 256
        ) -> str:
    """This function runs the model and return the output
    input:
        _instruction: string
        _input: string

        temperature: The value used to module the next token probabilities.
        top_p: If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation.
        num_beams: Number of beams for beam search. 1 means no beam search.
        max_len: max generation length

    return:
        string
    """
    #
    model_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams
    )

    # get the instruction input
    model_input = get_model_input(_instruction, _input)

    # tokenized input
    model_input = llama_tokenizer(model_input, return_tensors="pt")
    model_input_ids = model_input["input_ids"].to(device)

    # infer only, do not compute gradient
    with torch.no_grad():
        model_output = llama_model.generate(
            input_ids=model_input_ids,
            generation_config=model_config,
            max_new_tokens=max_len,
            return_dict_in_generate=True,
            output_scores=True
        )

    return llama_tokenizer.decode(model_output.sequences[0])

# understand domain specific words.
_instruction = "We car dealership. Generate a marketing email to our customer and give them a 20 off for new year. Also promot our   an."
_input = ""

# disambiguous
# _instruction = "Explain how 'mac' is used differently in these sentences?"
# _input = "['I love big mac.', 'My mac is broken']"

# Classification
# _instruction = "Categorize the given sentence into the following categories.: Finance, Romantic, Retail, Food, and None of the above. Assign multiple categories if needed."
# _input = "What if we go to Macy's and grab some lunch at Chick-fil-A?"

# Extract entities
# _instruction = "Should I target or consider the user who sent the following message as my audience for promoting our new laptop product? Explain why."
# _input = "I just got my salary. I'll just save it for future usage."
# _input = 'My macbook has just broken.'

# Extract entities
# _instruction = "What is the life stage of the user who sent the following messages? Explain why. Life Stages: in college, married, have a baby, new house."
# _input = "We need get the car seats for her."
# _input = 'I am a little bit nevers about going to Umass this summar.'

print(run(_instruction, _input))

