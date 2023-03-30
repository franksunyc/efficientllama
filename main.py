import torch
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from peft import PeftModel
# setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device",device)
#llama_model_path = "/home/data/hdd-1/7B"
#llama_token_model = '/home/data/hdd-1/tokenizer.model'
llama_model_path = ""
if not llama_model_path:
    llama_model_path = "decapoda-research/llama-7b-hf"

# the path to the auxiliary model
efficient_llama_model_path = "auxiliary_model"

if not efficient_llama_model_path:
    raise Exception("Please input your auxiliary model")

llama_tokenizer = LLaMATokenizer.from_pretrained(llama_model_path)

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
_instruction = "We car dealership. Generate a marketing email to our customer and give them a 20 off for new year."
_input = ""
print('understand domain specific words')
print(_instruction)
print(run(_instruction, _input))
print("")
# disambiguous
_instruction = "Explain how 'mac' is used differently in these sentences?"
_input = "['I love big mac.', 'My mac is broken']"
print('disambiguous')
print(_instruction)
print(run(_instruction, _input))
print("")

# Classification
_instruction = "Categorize the given sentence into the following categories.: Finance, Romantic, Retail, Food, and None of the above. Assign multiple categories if needed."
_input = "What if we go to Macy's and grab some lunch at Chick-fil-A?"
print('Classification')
print(_instruction)
print(run(_instruction, _input))
print("")
# Extract entities
_instruction = "Should I target or consider the user who sent the following message as my audience for promoting our new laptop product? Explain why."
_input = "I just got my salary. I'll just save it for future usage."
print('Extract entities')
print(_instruction)
print(run(_instruction, _input))
print("")
_input = 'My macbook has just broken.'
print("_instruction")
print(run(_instruction, _input))
print("")

# Extract entities
_instruction = "What is the life stage of the user who sent the following messages? Explain why. Life Stages: in college, married, have a baby, new house."
_input = "We need get the car seats for her."
print('Extract entities')
print(_instruction)
print(run(_instruction, _input))
print("")

_input = 'I am a little bit nevers about going to Umass this summar.'
print(_instruction)
print(run(_instruction, _input))
print("")

print(run(_instruction, _input))

# Extract entities
_instruction = "write a short title for the the following article"
_input = """Former President Donald Trump responded Thursday after a grand jury voted to indict him with a statement, calling it "Political Persecution and Election Interference at the highest level in history."
“This is Political Persecution and Election Interference at the highest level in history. From the time I came down the golden escalator at Trump Tower, and even before I was sworn in as your President of the United States, the Radical Left Democrats — the enemy of the hard-working men and women of this Country — have been engaged in a Witch-Hunt to destroy the Make America Great Again movement. You remember it just like I do: Russia, Russia, Russia; the Mueller Hoax; Ukraine, Ukraine, Ukraine; Impeachment Hoax 1; Impeachment Hoax 2; the illegal and unconstitutional Mar-a-Lago raid; and now this.
“The Democrats have lied, cheated and stolen in their obsession with trying to ‘Get Trump,’ but now they’ve done the unthinkable — indicting a completely innocent person in an act of blatant Election Interference. 
“Never before in our Nation’s history has this been done. The Democrats have cheated countless times over the decades, including spying on my campaign, but weaponizing our justice system to punish a political opponent, who just so happens to be a President of the United States and by far the leading Republican candidate for President, has never happened before. Ever.
“Manhattan DA Alvin Bragg, who was hand-picked and funded by George Soros, is a disgrace. Rather than stopping the unprecedented crime wave taking over New York City, he’s doing Joe Biden’s dirty work, ignoring the murders and burglaries and assaults he should be focused on. This is how Bragg spends his time!
“I believe this Witch-Hunt will backfire massively on Joe Biden. The American people realize exactly what the Radical Left Democrats are doing here. Everyone can see it. So our Movement, and our Party — united and strong — will first defeat Alvin Bragg, and then we will defeat Joe Biden, and we are going to throw every last one of these Crooked Democrats out of office so we can MAKE AMERICA GREAT AGAIN!”"""
print('title')
print(_instruction)
print(run(_instruction, _input))
print("")

_instruction = "下面这段对话主要讲了什么事情？"
_input = """
插件开发人员公开一个或多个 API 端点，并附有标准化的清单文件和 OpenAPI 规范。这些定义了插件的功能，允许 ChatGPT 使用文件并调用开发人员定义的 API。

AI 模型充当智能 API 调用者。给定 API 规范和何时使用 API 的自然语言描述，模型会主动调用 API 来执行操作。例如，如果用户问“我应该在巴黎住几晚？”，模型可能会选择调用酒店预订插件 API，接收 API 响应，并结合 API 数据生成面向用户的答案及其自然语言能力。

随着时间的推移，我们预计该系统将不断发展以适应更高级的用例。
"""print('title')
print(_instruction)
print(run(_instruction, _input))
print("")
