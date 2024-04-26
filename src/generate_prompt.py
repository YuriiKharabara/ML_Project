def generate_prompt_for_llm(elements, app_name):
    prompt = f"This is the app called {app_name}. This screen contains the following elements:\n\n"

    element_counts = {}
    for element in elements:
        if element['type'] in element_counts:
            element_counts[element['type']] += 1
        else:
            element_counts[element['type']] = 1

    for element_type, count in element_counts.items():
        prompt += f"- {count} {element_type}{'s' if count > 1 else ''}, "

    prompt = prompt.rstrip(", ") + ".\n\n"

    prompt += "Details of each element:\n"
    for i, element in enumerate(elements, 1):
        prompt += f"{i}. A {element['type']} located at coordinates {element['coords']} with color {element['color']} and containing the text: \"{element['text']}\".\n"

    return prompt
