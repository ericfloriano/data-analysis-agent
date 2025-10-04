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

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --- Configuração da Página ---
st.set_page_config(page_title="Agente de Análise de CSV", page_icon="🤖", layout="wide")

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
            else:
                new_name = str(col)
        except (ValueError, SyntaxError):
            new_name = str(col)
        
        cleaned_name = new_name.strip().lower().replace(' ', '_')
        cleaned_name = re.sub(r'[^\w_]', '', cleaned_name)
        new_columns.append(cleaned_name)
        
    df.columns = new_columns
    return df

@tool
def execute_python(code: str, plot_title: str = "Gráfico"):
    """
    Executa código Python para analisar o dataframe `df` ou criar visualizações.
    - Para retornar texto (tabelas, resumos), use o comando `print()`.
    - Para criar um gráfico, gere o código do gráfico e forneça um `plot_title` descritivo.
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
            st.session_state.generated_plots.append({"title": plot_title, "figure": fig})
            plt.close('all')
            return f"Sucesso! O gráfico '{plot_title}' foi gerado e adicionado ao Dashboard de Gráficos."
        else:
            plt.close('all')
            return f"Código executado com sucesso.\n**Resultado:**\n```\n{text_output}\n```"
            
    except Exception as e:
        plt.close('all')
        return f"Erro ao executar o código: {e}"

# --- INICIALIZAÇÃO DO AGENTE ---
system_prompt = """
Você é um agente de IA especialista em análise de dados. Sua ferramenta principal é um executor de código Python.
O DataFrame `df` já está carregado e suas colunas já foram limpas e simplificadas.
As bibliotecas `pandas` (pd), `matplotlib.pyplot` (plt) e `seaborn` (sns) já estão importadas.

INSTRUÇÕES CRÍTICAS:
1.  **Use a Memória:** Você tem acesso ao histórico da conversa. Use-o para entender o contexto.
2.  **Seja um Executor:** Sempre use a ferramenta `execute_python` para responder.
3.  **Para Análise Textual:** Para retornar uma tabela ou texto, gere o código Python com `print()`.
4.  **Para Gráficos:** Se o usuário pedir um gráfico, gere o código que o cria e forneça um `plot_title` descritivo. NUNCA use `plt.show()` ou `plt.savefig()`.
5.  **Inferência de Colunas:** Se o usuário pedir "gráfico de satisfação", e a coluna for `satisfacao_trabalhador`, use o nome correto.
6.  **Idioma:** Responda 100% em Português do Brasil.
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
tools = [execute_python]

# --- Inicialização do LLM ---
try:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
except Exception as e:
    st.error(f"Erro ao inicializar o modelo da OpenAI: {e}. Verifique sua chave de API nos segredos.")
    st.stop()

# --- Inicialização do Estado da Sessão ---
st.session_state.setdefault('msgs', StreamlitChatMessageHistory(key="chat_messages"))
st.session_state.setdefault('generated_plots', [])

# --- UI DA SIDEBAR ---
with st.sidebar:
    st.header("1. Carregue seu Arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", accept_multiple_files=False, key="file_uploader_key")
    st.info("Atenção: Apenas um arquivo CSV pode ser carregado por vez.")
    if st.button("Limpar Sessão"):
        st.session_state.clear()
        st.rerun()
    st.header("2. Exemplos de Perguntas")
    st.markdown("- Faça um resumo estatístico dos dados.\n- Crie um gráfico de barras para a coluna '[nome_da_coluna]'."
                "\n- Mostre a correlação entre as colunas com um heatmap.\n- Crie um histograma para a coluna '[nome_da_coluna]'."
                "\n- Com base na nossa conversa, quais as conclusões?")

# --- UI PRINCIPAL ---
st.title("🤖 Agente de IA para Análise de Dados")

# --- LÓGICA DE INICIALIZAÇÃO DO AGENTE ---
if uploaded_file and "agent_with_history" not in st.session_state:
    with st.spinner("Processando arquivo e preparando o agente..."):
        try:
            df = load_robust_csv(uploaded_file)
            df = clean_column_names(df)
            st.session_state.df = df
            
            agent = create_tool_calling_agent(llm, tools, prompt_template)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            st.session_state.agent_with_history = RunnableWithMessageHistory(
                agent_executor, lambda session_id: st.session_state.msgs,
                input_messages_key="input", history_messages_key="chat_history",
                agent_scratchpad_messages_key="agent_scratchpad"
            )
            st.success("Agente pronto!")
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")

# --- ESTRUTURA DE ABAS E INTERAÇÃO ---
if "df" in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat com Agente", "📊 Dashboard de Gráficos"])

    with tab1:
        st.write("### Pré-visualização dos Dados")
        st.dataframe(st.session_state.df.head(11))
        st.divider()

        # Loop de exibição é a única parte que desenha as mensagens.
        for msg in st.session_state.msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

        # Lógica de input no final do script da aba, separada da exibição.
        if prompt := st.chat_input("Faça sua pergunta sobre o arquivo CSV..."):
            # O RunnableWithMessageHistory adiciona a pergunta do usuário ao histórico automaticamente
            
            # Não usamos st.spinner aqui para não bagunçar o layout.
            # O usuário verá o indicador "Running..." no canto da tela.
            try:
                response = st.session_state.agent_with_history.invoke(
                    {"input": prompt},
                    {"configurable": {"session_id": "any_string"}}
                )
                # O Runnable já salvou a resposta no histórico (msgs).
            except Exception as e:
                st.session_state.msgs.add_ai_message(f"Ocorreu um erro: {e}")
            
            # Apenas recarrega a página. O loop de exibição acima irá
            # desenhar a conversa atualizada com a nova pergunta e resposta.
            st.rerun()

    with tab2:
        st.write("### Gráficos Gerados na Sessão")
        if not st.session_state.generated_plots:
            st.info("Nenhum gráfico foi gerado ainda. Peça ao agente para criar um na aba de chat!")
        else:
            for plot in reversed(st.session_state.generated_plots):
                st.subheader(plot["title"])
                st.pyplot(plot["figure"])
                st.divider()

else:
    st.info("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")