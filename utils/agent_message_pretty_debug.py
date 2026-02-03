from langchain_core.callbacks import BaseCallbackHandler

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def pretty_message(msg, previous_tool_calls=None):
    """
    Devuelve string legible para humanos y reemplaza Tool: X por tool call + resultado.
    Returns a human-friendly string and replace Tool: X by tool call + result
    """
    if hasattr(msg, "type"):
        role = msg.type.capitalize()
        content = getattr(msg, "content", "")
    else:
        role = "Message"
        content = str(msg)

    lines = content.splitlines()
    new_lines = []
    tool_idx = 0

    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("Tool:") and previous_tool_calls:
            # Obtener tool call correspondiente
            if tool_idx < len(previous_tool_calls):
                tool_call = previous_tool_calls[tool_idx]
                result = line_strip.replace("Tool:", "").strip()
                new_lines.append(f" {tool_call} → {result}")
                tool_idx += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    content_new = "\n".join(new_lines)
    return f"{BOLD}{role}:{RESET}\n{content_new}"


class PrettyDebug(BaseCallbackHandler):
    previous_tool_calls = []

    # ------------------ LLM START ------------------
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Detectar nueva invocación
        ai_tool_present = False

        for prompt in prompts:
            # Convertir a lista de líneas
            lines = []
            if isinstance(prompt, list):
                # Lista de strings
                for p in prompt:
                    if isinstance(p, str):
                        lines.extend(p.splitlines())
            elif isinstance(prompt, str):
                lines = prompt.splitlines()
            else:
                continue

            # Revisar todas las líneas
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("AI:") or line_strip.startswith("Tool:"):
                    ai_tool_present = True
                    break
            if ai_tool_present:
                break

        if not ai_tool_present:
            # No se detecta salida AI/Tool → nueva invocación
            self.previous_tool_calls = []

        print(f"{BLUE}{BOLD}\n========================")
        print("▶ LLM START")
        print(RESET)

        # Mostrar prompts con mapping Tool → resultado
        for prompt in prompts:
            if isinstance(prompt, list):
                for p in prompt:
                    print(pretty_message(p, previous_tool_calls=self.previous_tool_calls))
                    print()
            else:
                print(pretty_message(prompt, previous_tool_calls=self.previous_tool_calls))
                print()

    # ------------------ LLM END --------------------
    def on_llm_end(self, response, **kwargs):
        print(f"{GREEN}{BOLD}■ LLM END{RESET}")
        gen = response.generations[0][0]
        msg = gen.message

        # Contenido principal
        content = getattr(msg, "content", "")
        if content:
            print(f"{GREEN}Content:{RESET}\n{content}\n")

        # Detectar nuevas tool calls y guardarlas para la siguiente invocación
        tool_calls = getattr(msg, "tool_calls", [])
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            args_s = ", ".join(f"{k}={v}" for k, v in args.items())
            tool_call_str = f"▶ {name}({args_s})"
            self.previous_tool_calls.append(tool_call_str)
            print(f" ==> {tool_call_str}")

        # Razonamiento interno del agente
        reasoning = msg.additional_kwargs.get("reasoning", [])
        if reasoning:
            print(f"{MAGENTA}{BOLD}Reasoning steps:{RESET}")
            for i, step in enumerate(reasoning, 1):
                thought = step.get("thought", "")
                action = step.get("action", "")
                observation = step.get("observation", "")
                if thought:
                    print(f"  {i}. 💭 Thought: {thought}")
                if action:
                    print(f"     🛠️ Action: {action}")
                if observation:
                    print(f"     👀 Observation: {observation}")

        # Tokens
        usage = msg.additional_kwargs.get("response_metadata", {}).get("token_usage", {})
        if usage:
            print(f"{MAGENTA}Tokens:{RESET} input={usage.get('prompt_tokens')} "
                  f"output={usage.get('completion_tokens')} total={usage.get('total_tokens')}")

        print(f"{GREEN}========================{RESET}\n")