import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
#from langchain_community.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def carregar_e_preparar_dados(arquivo_cabecalho, arquivo_itens):
    """
    Carrega os dois arquivos CSV, realiza o pré-processamento
    e os une em um único DataFrame.
    """
    try:
        # Carrega os dados especificando os separadores
        df_cabecalho = pd.read_csv(arquivo_cabecalho, sep=',', decimal='.')
        df_itens = pd.read_csv(arquivo_itens, sep=',', decimal='.')

        # --- Etapa Crítica: Identificar a coluna chave para o merge ---
        # Analisando os nomes prováveis, 'ChaveNF' é um candidato forte.
        # IMPORTANTE: Verifique nos seus arquivos qual é a coluna que conecta
        # o cabeçalho aos itens (pode ser 'ChaveNF', 'id_nota_fiscal', etc.)
        # e ajuste o nome da coluna em 'left_on' e 'right_on' se necessário.
        coluna_chave = None
        colunas_candidatas = ['NÚMERO']

        for col in colunas_candidatas:
            if col in df_cabecalho.columns and col in df_itens.columns:
                coluna_chave = col
                break

        if not coluna_chave:
            st.error("Erro: Não foi possível encontrar uma coluna em comum para unir os arquivos. Verifique se ambos possuem uma coluna como 'ChaveNF' ou 'ID_NF'.")
            return None

        # Converte as colunas de data para o formato datetime
        # O Pandas geralmente infere isso bem, mas podemos garantir.
        if 'DataEmissao' in df_cabecalho.columns:
            df_cabecalho['DataEmissao'] = pd.to_datetime(df_cabecalho['DataEmissao'])
        if 'DataEntrada' in df_cabecalho.columns:
            df_cabecalho['DataEntrada'] = pd.to_datetime(df_cabecalho['DataEntrada'])

        # Une os dois DataFrames em um só
        df_completo = pd.merge(df_itens, df_cabecalho, on=coluna_chave, how='left')

        return df_completo, coluna_chave

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar os arquivos: {e}")
        return None, None


def criar_agente(df, api_key):
    """
    Cria e retorna um agente LangChain para interagir com o DataFrame.
    """
    try:
        # Inicializa o LLM Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=api_key,
            temperature=0, # Baixa temperatura para respostas mais factuais
            convert_system_message_to_human=True # Necessário para agentes
        )

        # Cria o agente de DataFrame do Pandas
        # allow_dangerous_code=True é necessário para que o agente possa executar o código
        # gerado por ele mesmo para analisar os dados.
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, # Mostra os "pensamentos" do agente no terminal
            allow_dangerous_code=True,
            handle_parsing_errors=True # Ajuda a lidar com erros de formatação do LLM
        )
        return agent
    except Exception as e:
        st.error(f"Falha ao criar o agente. Verifique sua chave de API. Erro: {e}")
        return None

# --- Interface Web com Streamlit ---

st.set_page_config(page_title="Análise de Notas Fiscais com Gemini", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Notas Fiscais")
st.write("Faça perguntas em linguagem natural sobre os dados das notas fiscais.")

# --- Barra Lateral para Configuração ---
with st.sidebar:
    st.header("1. Configuração")
    gemini_api_key = st.text_input("Insira sua chave da API do Google Gemini", type="password")

    st.header("2. Carregar Arquivos CSV")
    uploaded_cabecalho = st.file_uploader("Carregue o CSV do Cabeçalho (NFs_Cabecalho)", type="csv")
    uploaded_itens = st.file_uploader("Carregue o CSV dos Itens (NFs_Itens)", type="csv")

# --- Lógica Principal ---

# Inicializa o agente no estado da sessão para não recriar a cada interação
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None

# Só prossiga se a chave e os arquivos forem fornecidos
if gemini_api_key and uploaded_cabecalho and uploaded_itens:
    with st.spinner("Processando arquivos e inicializando o agente..."):
        # Carrega os dados
        df_completo, chave = carregar_e_preparar_dados(uploaded_cabecalho, uploaded_itens)

        if df_completo is not None:
            # Cria o agente se ele ainda não foi criado
            if st.session_state.agent_executor is None:
                st.session_state.agent_executor = criar_agente(df_completo, gemini_api_key)
                st.success(f"Agente inicializado com sucesso! Dados unidos pela coluna '{chave}'.")
                st.dataframe(df_completo.head()) # Mostra um preview dos dados unidos

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Interação do Chat ---
if prompt := st.chat_input("Qual o fornecedor com maior valor total de notas?"):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Verifica se o agente está pronto
    if st.session_state.agent_executor:
        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    # Adiciona contexto para ajudar o agente
                    prompt_completo = f"""
                    Contexto: Você está analisando um conjunto de dados de notas fiscais.
                    O DataFrame contém informações tanto do cabeçalho da nota (fornecedor, data, valor total da nota)
                    quanto dos itens da nota (produto, quantidade, valor unitário).

                    Pergunta do usuário: {prompt}

                    Responda em português.
                    """
                    response = st.session_state.agent_executor.invoke({"input": prompt_completo})
                    answer = response['output']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar a sua pergunta: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.warning("Por favor, insira sua chave de API e carregue os arquivos na barra lateral para começar.")