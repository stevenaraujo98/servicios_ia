from ollama import chat
import json

def request_chat(model, prompt="hola"):
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            # {"role": "user", "content": content},
            # {"role": "user", "content": question},
        ],
        # stream=True,
    )

    return response.message.content

def calificate_objective(model_name, text):
    response = request_chat(model=model_name, prompt=text)
    only_json = response[response.find("{"):response.find("}") + 1]
    
    json_dict = json.loads(only_json)
    return json_dict["aprobado"], json_dict["verbos"], json_dict["detalle"], json_dict["sugerencias"], json_dict["opciones_sugerencia"]
