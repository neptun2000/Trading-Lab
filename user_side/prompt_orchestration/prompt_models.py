from openai import OpenAI
import anthropic
import os
from ..prompts.deep_research_prompt import create_deep_research_prompt
from ..prompts.daily_research_prompt import create_daily_prompt


def prompt_claude(text: str, model: str = "claude-sonnet-4-6") -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": text}],
    )

    if not message.content:
        raise RuntimeError("No content returned from Claude.")

    block = message.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise RuntimeError(f"Unexpected content type from Claude: {type(block)}")
    return block.text


def prompt_chatgpt(text: str, model: str = "gpt-4.1-mini") -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        temperature=0.0,
    )

    if not response.choices:
        raise RuntimeError("No choices returned from ChatGPT.")

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Output from ChatGPT was None.")

    return content


def prompt_deep_research(libb) -> str:
    model = libb._model_path.replace("user_side/runs/run_v1/", "")
    text = create_deep_research_prompt(libb)
    if model == "claude":
        return prompt_claude(text)
    elif model == "gpt-4.1":
        return prompt_chatgpt(text)
    else:
        raise RuntimeError(f"Unidentified model: {model}")


def prompt_daily_report(libb) -> str:
    model = libb._model_path.replace("user_side/runs/run_v1/", "")
    text = create_daily_prompt(libb)
    if model == "claude":
        return prompt_claude(text)
    elif model == "gpt-4.1":
        return prompt_chatgpt(text)
    else:
        raise RuntimeError(f"Unidentified model: {model}")