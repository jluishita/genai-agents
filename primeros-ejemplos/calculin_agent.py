from google import genai
from google.genai import types

"""
Ejemplo de definición de agente en arquitectura ReAct (Reason and Act)

No definimos un grafo de agente ni un flujo específico. La conversación puede tener cualquier estructura.

system_instructions establece la personalidad del agente y las instrucciones.

Se definen unas herramientas específicas: sumar y restar. El agente entiende cómo funcionan esas herramientas
en base al docstring que tengan esas funciones. Es importante especificar el formato de los argumentos de
entrada y de salida, para que el agente pueda usar esas herramientas correctamente.

Para poder usar el agente es necesario haber añadido la API Key de Gemini en las variables de entorno 
del sistema operativo.
"""

# 1. Definir la herramienta (Function Calling)
def sumar_numeros(a: float, b: float) -> float:
    """Suma dos números y devuelve el resultado."""
    print(f"[Ejecutando herramienta...] Sumando {a} + {b}")
    return a + b

def restar_numeros(a: float, b: float) -> float:
    """Resta dos números y devuelve el resultado."""
    print(f"[Ejecutando herramienta...] Restando {a} - {b}")
    return a - b

system_instructions = """
Eres un asistente matemático llamado 'Calculín'. 
Tu objetivo es ayudar a los usuarios con operaciones matemáticas elementales como las sumas o las restas, de forma muy entusiasta.
Reglas:
1. Siempre preséntate por tu nombre al principio.
2. Usa emojis relacionados con las matemáticas o la alegría en tus respuestas.
3. Para hacer las operaciones matemáticas, apóyate en las herramientas proporcionadas (funciones de Python).
4. Si te preguntan algo que no sea de matemáticas, di educadamente que solo sabes sumar y restar.
"""

# 2. Inicializar el cliente
client = genai.Client()

config = types.GenerateContentConfig(
    system_instruction=system_instructions,
    tools=[
        sumar_numeros,
        restar_numeros
    ],
    temperature=0.5
)

# 4. Crear el agente (iniciar la sesión de chat)
agente = client.chats.create(
    model="gemini-3.1-flash-lite-preview",
    config=config
)

# 5. Hacer que el agente tome la iniciativa y se presente
print("--- Iniciando el agente matemático ---")
saludo_inicial = agente.send_message("Hola, por favor preséntate y salúdame para empezar.")
print(f"🤖 Calculín: {saludo_inicial.text}\n")

# 6. Bucle interactivo para conversar con él
while True:
    pregunta = input("Tú: ")
    
    # Condición para salir del programa
    if pregunta.lower() in ['salir', 'adiós', 'exit', 'quit']:
        print("🤖 Calculín: ¡Hasta pronto! 👋")
        break
        
    # Enviar la pregunta y mostrar la respuesta
    respuesta = agente.send_message(pregunta)
    print(f"🤖 Calculín: {respuesta.text}\n")