# genai-agents
Examples of GenAI Agents

Para usar los agentes de la carpeta primeros-agentes:

1. Instalar google-genai en el entorno virtual

pip install google-genai

2. Obtener la API KEY de GEMINI en GOOGLE AI Studio (aistudio.google.com)
3. Añadir esta API KEY a las variables de entorno:
- En Ubuntu:
  echo 'export GEMINI_API_KEY="la-api-key-que-hayas-obtenido-en-aistudio.google.com"' >> ~/.bashrc
  source ~/.bashrc
4. Instalar LangGraph y la librería de integración de LangGraph para Google:
  pip install langgraph langchain-google-genai
5. Ir al directorio correspondiente y ejecutar el fichero Python desde terminal:

python3 calculin_agent.py
