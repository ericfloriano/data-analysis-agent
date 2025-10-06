import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from contextlib import redirect_stdout
import ast

# Importações para múltiplos LLMs
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --- Configuração da Página ---
st.set_page_config(page_title="Agente de Análise de CSV", page_icon="🤖", layout="wide")

# --- GERENCIADOR DE LLM COM FALLBACK ---
class LLMManager:
    """Gerencia a inicialização de LLMs com uma lógica de fallback em tempo real."""
    def __init__(self):
        self.providers = [
            {"name": "Groq", "model": "llama-3.1-8b-instant", "api_key": st.secrets.get("GROQ_API_KEY")},
            {"name": "OpenAI", "model": "gpt-4o", "api_key": st.secrets.get("OPENAI_API_KEY")}
        ]
        self.current_provider_index = 0

    def get_llm(self, use_fallback=False):
        """Obtém um LLM. Se use_fallback for True, tenta o próximo da lista."""
        if use_fallback:
            self.current_provider_index += 1
        
        if self.current_provider_index >= len(self.providers):
            st.error("Todos os provedores de LLM falharam.")
            return None, "Falha"

        provider = self.providers[self.current_provider_index]
        try:
            if not provider["api_key"]:
                st.warning(f"Chave de API para {provider['name']} não encontrada. Tentando o próximo...")
                return self.get_llm(use_fallback=True)

            llm = None
            if provider["name"] == "Groq":
                llm = ChatGroq(model_name=provider["model"], temperature=0, groq_api_key=provider["api_key"])
            elif provider["name"] == "OpenAI":
                llm = ChatOpenAI(model_name=provider["model"], temperature=0, openai_api_key=provider["api_key"])
            
            active_provider_name = f"{provider['name']} ({provider['model']})"
            # st.toast(f"Conectado ao LLM: {active_provider_name}", icon="🚀" if not use_fallback else "🛡️")
            return llm, active_provider_name
        except Exception as e:
            st.warning(f"Falha ao conectar com {provider['name']}: {e}. Tentando o próximo...")
            return self.get_llm(use_fallback=True)

# --- Funções de Lógica ---
def load_robust_csv(file):
    try:
        bytes_data = file.getvalue()
        string_io = io.StringIO(bytes_data.decode('utf-8'))
        return pd.read_csv(string_io, sep=None, engine='python')
    except UnicodeDecodeError:
        st.warning("Falha ao decodificar com UTF-8. Tentando com Latin-1...")
        string_io = io.StringIO(bytes_data.decode('latin-1'))
        return pd.read_csv(string_io, sep=None, engine='python')

def clean_column_names(df):
    new_columns = []
    for col in df.columns:
        try:
            col_tuple = ast.literal_eval(col)
            if isinstance(col_tuple, (tuple, list)):
                new_name = '_'.join(str(part).strip() for part in col_tuple if str(part).strip())
            else: new_name = str(col)
        except (ValueError, SyntaxError): new_name = str(col)
        cleaned_name = new_name.strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        cleaned_name = re.sub(r'[^\w_]', '', cleaned_name)
        new_columns.append(cleaned_name)
    df.columns = new_columns
    return df

@tool
def execute_python(code: str, plot_title: str = "Gráfico"):
    """
    Executa código Python para analisar o dataframe `df` ou criar visualizações.
    - Para retornar texto (tabelas, resumos), use o comando `print()`.
    - Para criar um gráfico, gere o código e forneça um `plot_title` descritivo.
    - NUNCA use `plt.show()` ou `plt.savefig()`.
    - O dataframe está disponível como `df`. Bibliotecas `pd`, `plt`, `sns` já estão importadas.
    """
    plt.switch_backend('Agg')
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec_globals = {'df': st.session_state.df, 'pd': pd, 'plt': plt, 'sns': sns}
            exec(code, exec_globals)
        
        text_output = buffer.getvalue()
        fig = plt.gcf()
        
        if fig.get_axes():
            title = plot_title if plot_title else "Gráfico Gerado"
            st.session_state.generated_plots.append({"title": title, "figure": fig})
            plt.close('all')
            return f"Sucesso! O gráfico '{title}' foi gerado e adicionado ao Dashboard de Gráficos."
        else:
            plt.close('all')
            # AJUSTE: Garante que mesmo sem print, se houver um resultado, ele seja retornado
            if 'result' in exec_globals:
                text_output += str(exec_globals['result'])
            return f"Código executado com sucesso.\n**Resultado:**\n```\n{text_output}\n```"
            
    except Exception as e:
        plt.close('all')
        return f"Erro ao executar o código: {e}"

# --- INICIALIZAÇÃO DO AGENTE (PROMPT FINAL E ROBUSTO) ---
system_prompt = """
Você é 'Agente-Analisador', um especialista em análise de dados em Português do Brasil. Sua única ferramenta é um executor de código Python.

**REGRAS FUNDAMENTAIS E OBRIGATÓRIAS:**
1.  **SEMPRE use a ferramenta `execute_python`** para qualquer análise de dados ou criação de gráficos.
2.  **PARA ANÁLISE DE TEXTO:** O código que você gera **DEVE** usar `print()` para que o resultado seja visível. Exemplo: `print(df.describe())`.
3.  **PARA GRÁFICOS:**
    * O código que você gera **DEVE** incluir um parâmetro `plot_title` claro e descritivo.
    * O código **NUNCA DEVE** conter `plt.show()` ou `plt.savefig()`.
4.  **RESPOSTA FINAL:** Após a ferramenta ser executada, sua resposta final para o usuário **DEVE** ser uma **interpretação amigável** do resultado. **SEMPRE inclua o resultado completo da ferramenta** (a tabela de dados ou a confirmação do gráfico) e adicione seus insights.
5.  **MEMÓRIA E IDIOMA:** Use o histórico da conversa para contexto e responda sempre em Português do Brasil.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt), MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad"),
])
tools = [execute_python]

# --- Inicialização do Estado da Sessão ---
st.session_state.setdefault('msgs', StreamlitChatMessageHistory(key="chat_messages"))
st.session_state.setdefault('generated_plots', [])

# --- UI DA SIDEBAR ---
with st.sidebar:
    st.header("1. Carregue seu Arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", accept_multiple_files=False, key="file_uploader_key")
    if st.button("Limpar Sessão"): st.session_state.clear(); st.rerun()
    st.header("2. Exemplos de Perguntas")
    st.markdown("- Faça um resumo estatístico dos dados.\n- Crie um gráfico de barras para a coluna '[...]'."
                "\n- Mostre a correlação entre as colunas com um heatmap.\n- Crie um histograma para a coluna '[...]'."
                "\n- Com base na nossa conversa, quais as conclusões?")
    if "active_provider" in st.session_state:
        st.divider(); st.markdown(f"**LLM Ativo:**\n`{st.session_state.active_provider}`")

# --- UI PRINCIPAL ---
st.title("🤖 Agente de IA para Análise de Dados")

# --- LÓGICA DE INICIALIZAÇÃO DO AGENTE ---
if uploaded_file and "agent_with_history" not in st.session_state:
    with st.spinner("Processando arquivo e preparando o agente..."):
        try:
            df = load_robust_csv(uploaded_file)
            df = clean_column_names(df)
            st.session_state.df = df
            
            st.session_state.llm_manager = LLMManager()
            llm, active_provider_name = st.session_state.llm_manager.get_llm()
            st.session_state.active_provider = active_provider_name
            
            if llm:
                agent = create_tool_calling_agent(llm, tools, prompt_template)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
                st.session_state.agent_with_history = RunnableWithMessageHistory(
                    agent_executor, lambda session_id: st.session_state.msgs,
                    input_messages_key="input", history_messages_key="chat_history",
                    agent_scratchpad_messages_key="agent_scratchpad"
                )
        except Exception as e: st.error(f"Erro ao carregar o arquivo: {e}")

# --- ESTRUTURA DE ABAS E INTERAÇÃO ---
if "df" in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat com Agente", "📊 Dashboard de Gráficos"])
    with tab1:
        st.write("### Pré-visualização dos Dados"); st.dataframe(st.session_state.df.head(11)); st.divider()
        for msg in st.session_state.msgs.messages:
            with st.chat_message("human" if msg.type == 'human' else "assistant"):
                text_parts = re.split(r'(\*\*Gráfico Gerado:\*\*\\n\[image data:image/png;base64,[^\]]+\])', msg.content)
                for part in text_parts:
                    if part.startswith('**Gráfico Gerado:**'):
                        base64_string = part.split('base64,')[1].replace(']', '')
                        st.image(base64.b64decode(base64_string))
                    else: st.markdown(part, unsafe_allow_html=True)

        if prompt := st.chat_input("Faça sua pergunta sobre o arquivo CSV..."):
            # A linha "add_user_message" foi REMOVIDA.
            # O sistema de memória (RunnableWithMessageHistory) agora é o único responsável
            # por adicionar a pergunta do usuário ao histórico.
            with st.spinner("O agente está pensando..."):
                try:
                    response = st.session_state.agent_with_history.invoke(
                        {"input": prompt},
                        {"configurable": {"session_id": "any_string"}}
                    )
                except Exception as e:
                    # ... (sua lógica de fallback continua aqui, sem alterações) ...
                    st.warning(f"Ocorreu um erro com o LLM atual: {e}. Acionando cliente LLM de CONTIGÊNCIA...")
                    try:
                        llm, active_provider_name = st.session_state.llm_manager.get_llm(use_fallback=True)
                        st.session_state.active_provider = active_provider_name
                        if llm:
                            agent = create_tool_calling_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
                            st.session_state.agent_with_history = RunnableWithMessageHistory(
                                agent_executor, lambda session_id: st.session_state.msgs,
                                input_messages_key="input", history_messages_key="chat_history",
                                agent_scratchpad_messages_key="agent_scratchpad"
                            )
                            st.info("Tentando novamente com o LLM de fallback...")
                            response = st.session_state.agent_with_history.invoke(
                                {"input": prompt}, {"configurable": {"session_id": "any_string"}}
                            )
                        else:
                            st.session_state.msgs.add_ai_message("Todos os provedores de LLM falharam.")
                    except Exception as final_e:
                        st.session_state.msgs.add_ai_message(f"Ocorreu um erro mesmo com o fallback: {final_e}")

            st.rerun()

    with tab2:
        st.write("### Gráficos Gerados na Sessão")
        if not st.session_state.generated_plots:
            st.info("Nenhum gráfico foi gerado ainda.")
        else:
            for plot in reversed(st.session_state.generated_plots):
                st.subheader(plot.get("title", "Gráfico")); st.pyplot(plot["figure"]); st.divider()
else:
    st.info("Por favor, carregue um arquivo CSV para começar.")