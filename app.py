import streamlit as st
import os
import requests
import time
import hashlib
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import sqlite3
from contextlib import contextmanager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbot_app")

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configurações
MAX_HISTORY_TOKENS = 800  # Reduzido para evitar problemas com contexto muito longo
MAX_INPUT_LENGTH = 512  # Limitar tamanho de entrada para evitar erros 400
CACHE_EXPIRY = 3600  # Cache expira em 1 hora (segundos)
MAX_RETRIES = 3  # Número máximo de tentativas para chamadas à API
API_TIMEOUT = 30  # Timeout em segundos para chamadas à API (aumentado)
MAX_CONVERSATION_TURNS = 10  # Número máximo de turnos na conversa

# Configuração do banco de dados para cache
DB_PATH = "chat_cache.db"

@contextmanager
def get_db_connection():
    """Gerencia conexões com o banco de dados."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()

def init_db():
    """Inicializa o banco de dados para o cache."""
    try:
        with get_db_connection() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
            ''')
            conn.commit()
            logger.info("Banco de dados para cache inicializado")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        # Continua sem cache se não conseguir inicializar o banco de dados
        pass

class SecretManager:
    """Gerencia o acesso seguro aos segredos."""
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Obtém a chave da API de forma segura."""
        # Em produção, considere usar um serviço de gerenciamento de segredos
        # como AWS Secrets Manager, Google Secret Manager, HashiCorp Vault, etc.
        api_key = os.getenv("HF_API_KEY")
        
        # Validação básica da chave
        if not api_key:
            logger.error("Chave de API não encontrada")
            return None
            
        if len(api_key) < 8:  # Verificação simples
            logger.warning("Chave de API suspeita - muito curta")
            
        return api_key

class TokenManager:
    """Gerencia o tamanho do histórico de conversas."""
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimativa aproximada de tokens em um texto."""
        # Aproximação simples: cada 4 caracteres ~ 1 token
        return len(text) // 4

    @staticmethod
    def truncate_history(
        past_inputs: List[str], 
        past_responses: List[str], 
        max_tokens: int = MAX_HISTORY_TOKENS
    ) -> Tuple[List[str], List[str]]:
        """
        Trunca o histórico para não exceder o máximo de tokens.
        Preserva interações mais recentes.
        """
        inputs_copy = past_inputs.copy()
        responses_copy = past_responses.copy()
        
        # Limita o número de turnos para evitar contextos muito longos
        if len(inputs_copy) > MAX_CONVERSATION_TURNS:
            start_idx = len(inputs_copy) - MAX_CONVERSATION_TURNS
            inputs_copy = inputs_copy[start_idx:]
            responses_copy = responses_copy[start_idx:]
            logger.info(f"Histórico limitado a {MAX_CONVERSATION_TURNS} turnos")
            return inputs_copy, responses_copy
        
        # Estimativa de tokens totais
        total_tokens = sum(TokenManager.estimate_tokens(msg) for msg in inputs_copy + responses_copy)
        
        # Se estiver abaixo do limite, não é necessário truncar
        if total_tokens <= max_tokens:
            return inputs_copy, responses_copy
            
        # Remove interações mais antigas até estar abaixo do limite
        while total_tokens > max_tokens and inputs_copy and responses_copy:
            oldest_input = inputs_copy.pop(0)
            oldest_response = responses_copy.pop(0)
            total_tokens -= (TokenManager.estimate_tokens(oldest_input) + 
                            TokenManager.estimate_tokens(oldest_response))
            
        logger.info(f"Histórico truncado para {len(inputs_copy)} interações ({total_tokens} tokens est.)")
        return inputs_copy, responses_copy

    @staticmethod
    def truncate_text(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
        """Limita o tamanho de um texto para evitar erros de requisição."""
        if len(text) <= max_length:
            return text
        return text[:max_length]

class APICache:
    """Implementa cache para chamadas à API."""
    @staticmethod
    def compute_hash(data: str) -> str:
        """Calcula um hash único para a consulta."""
        return hashlib.md5(data.encode()).hexdigest()
        
    @staticmethod
    def get_cached_response(query: str) -> Optional[str]:
        """Recupera resposta em cache se existir e estiver válida."""
        try:
            query_hash = APICache.compute_hash(query)
            current_time = int(time.time())
            
            with get_db_connection() as conn:
                result = conn.execute(
                    "SELECT response, timestamp FROM cache WHERE query_hash = ?", 
                    (query_hash,)
                ).fetchone()
                
                if result and (current_time - result['timestamp']) < CACHE_EXPIRY:
                    logger.info("Cache hit!")
                    return result['response']
        except Exception as e:
            logger.error(f"Erro ao buscar no cache: {e}")
            # Continua sem usar cache se houver erro
                
        logger.info("Cache miss")
        return None
        
    @staticmethod
    def cache_response(query: str, response: str) -> None:
        """Armazena uma resposta no cache."""
        try:
            query_hash = APICache.compute_hash(query)
            current_time = int(time.time())
            
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (query_hash, response, timestamp) VALUES (?, ?, ?)",
                    (query_hash, response, current_time)
                )
                conn.commit()
                logger.info("Resposta armazenada em cache")
        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {e}")
            # Continua sem armazenar no cache se houver erro

class HuggingFaceClient:
    """Cliente para API do Hugging Face."""
    def __init__(self):
        self.api_key = SecretManager.get_api_key()
        # Blenderbot-400M-distill é mais estável para conversas simples
        self.api_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        
        if not self.api_key:
            st.error("❌ Chave de API não configurada corretamente")
            st.stop()
    
    def query(self, prompt: str) -> str:
        """
        Envia consulta para a API com tratamento de erros e retentativas.
        """
        # Limita o tamanho do prompt para evitar erro 400
        prompt = TokenManager.truncate_text(prompt)
        
        # Verifica cache primeiro
        cached_response = APICache.get_cached_response(prompt)
        if cached_response:
            return cached_response
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Para evitar problemas com a API do BlenderBot, usamos o formato mais simples
        payload = {
            "inputs": prompt,
            "options": {
                "wait_for_model": True  # Evita erros 503 por modelo não carregado
            }
        }
        
        # Implementação de retentativas com backoff exponencial
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Tentativa {attempt} de {MAX_RETRIES}")
                
                # Log da requisição para depuração (sem a chave de API)
                debug_headers = dict(headers)
                debug_headers["Authorization"] = "Bearer [REDACTED]"
                logger.debug(f"Enviando requisição: {json.dumps(payload)}")
                logger.debug(f"Headers: {debug_headers}")
                
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                # Log da resposta para depuração
                logger.debug(f"Status code: {response.status_code}")
                logger.debug(f"Resposta: {response.text[:200]}...")
                
                # Verificação específica para erro 400
                if response.status_code == 400:
                    error_msg = "Erro na formatação da requisição para a API."
                    logger.error(f"Erro 400: {response.text}")
                    # Tenta uma abordagem diferente no próximo retry
                    if attempt < MAX_RETRIES:
                        # Tenta simplificar o prompt para próxima tentativa
                        words = prompt.split()
                        if len(words) > 100:
                            prompt = " ".join(words[-100:])  # Usa apenas as últimas 100 palavras
                        continue
                    return "⚠️ Erro na requisição: formato inválido. Tente uma mensagem mais curta."
                
                # Verifica os erros HTTP
                response.raise_for_status()
                
                # Processa a resposta
                data = response.json()
                result = self._extract_response(data)
                
                # Se a resposta for muito curta ou vazia, pode ser um erro
                if not result or len(result) < 2:
                    logger.warning("Resposta muito curta ou vazia")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    result = "Desculpe, não consegui gerar uma resposta adequada. Poderia reformular?"
                
                # Armazena em cache
                APICache.cache_response(prompt, result)
                
                return result
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit excedido. Aguardando antes de tentar novamente.")
                    # Backoff exponencial
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:
                    logger.error(f"Erro no servidor: {e}")
                    if attempt < MAX_RETRIES:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        return "⚠️ Erro no servidor da API. Tente novamente mais tarde."
                else:
                    logger.error(f"Erro na requisição HTTP: {e}")
                    return f"⚠️ Erro na requisição: {e.response.status_code}"
                    
            except requests.exceptions.Timeout:
                logger.error("Timeout na requisição")
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return "⚠️ Tempo esgotado ao aguardar resposta da API."
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de conexão: {e}")
                return "⚠️ Erro de conexão com a API."
                
            except json.JSONDecodeError:
                logger.error("Resposta não é um JSON válido")
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return "⚠️ Resposta inválida da API."
                
            except Exception as e:
                logger.error(f"Erro desconhecido: {e}")
                return f"⚠️ Ocorreu um erro: {str(e)}"
        
        return "⚠️ Não foi possível obter uma resposta após várias tentativas."
    
    def _extract_response(self, data: Any) -> str:
        """Extrai texto da resposta da API, lidando com diferentes formatos."""
        try:
            if isinstance(data, dict):
                if "generated_text" in data:
                    return data["generated_text"].strip()
                if "error" in data:
                    logger.error(f"API retornou erro: {data['error']}")
                    return f"⚠️ Erro da API: {data['error']}"
                
            if isinstance(data, list) and data:
                if isinstance(data[0], dict) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                
            logger.warning(f"Formato de resposta inesperado: {json.dumps(data)[:200]}")
            
            # Tenta extrair qualquer texto disponível
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 5:
                        return value.strip()
                        
            if isinstance(data, list) and data and isinstance(data[0], dict):
                for key, value in data[0].items():
                    if isinstance(value, str) and len(value) > 5:
                        return value.strip()
        except Exception as e:
            logger.error(f"Erro ao extrair resposta: {e}")
            
        return "⚠️ Formato de resposta inesperado da API."

class ChatSession:
    """Gerencia a sessão de chat e o histórico."""
    def __init__(self):
        self.client = HuggingFaceClient()
        
        # Inicializa o histórico de conversas se não existir
        if "user_inputs" not in st.session_state:
            st.session_state.user_inputs = []
            st.session_state.bot_responses = []
    
    def add_message(self, user_input: str) -> str:
        """
        Adiciona uma mensagem do usuário e gera uma resposta.
        Retorna a resposta.
        """
        # Verifica se a entrada é válida
        if not user_input or len(user_input.strip()) == 0:
            return "⚠️ Por favor, digite uma mensagem válida."
            
        # Verifica se a entrada não é muito longa
        if len(user_input) > MAX_INPUT_LENGTH:
            user_input = user_input[:MAX_INPUT_LENGTH]
            logger.warning(f"Entrada truncada para {MAX_INPUT_LENGTH} caracteres")
        
        # Adiciona entrada do usuário
        st.session_state.user_inputs.append(user_input)
        
        # Prepara o contexto com histórico truncado
        truncated_inputs, truncated_responses = TokenManager.truncate_history(
            st.session_state.user_inputs[:-1],
            st.session_state.bot_responses
        )
        
        # Para limitar as chances de erros 400, simplificamos o formato do prompt
        # Enviando apenas a última mensagem e algumas interações anteriores
        # Isso aumenta a estabilidade da API
        if len(truncated_inputs) >= 3:
            # Usamos apenas as 3 últimas interações para o contexto
            truncated_inputs = truncated_inputs[-3:]
            truncated_responses = truncated_responses[-3:]
        
        # Monta a conversa como texto simplificado
        user_query = user_input
        
        # Se tivermos histórico, adicionamos contexto mínimo
        if truncated_inputs and truncated_responses:
            context = truncated_inputs[-1] + " " + truncated_responses[-1]
            # Limita o tamanho do contexto
            if len(context) > 200:
                context = context[-200:]
            user_query = context + " " + user_input
        
        # Gera resposta usando formato mais simples para reduzir erros
        try:
            reply = self.client.query(user_query)
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            reply = "⚠️ Ocorreu um erro ao processar sua mensagem. Tente novamente ou reformule."
        
        # Adiciona resposta ao histórico
        st.session_state.bot_responses.append(reply)
        
        return reply
        
    def get_history(self) -> List[dict]:
        """Retorna o histórico de conversas em formato estruturado."""
        history = []
        for i, user_msg in enumerate(st.session_state.user_inputs):
            history.append({"role": "user", "content": user_msg})
            if i < len(st.session_state.bot_responses):
                history.append({"role": "assistant", "content": st.session_state.bot_responses[i]})
        return history

class ChatUI:
    """Gerencia a interface do usuário do chat."""
    def __init__(self):
        st.set_page_config(
            page_title="Chat IA Contextual",
            page_icon="🤖",
            layout="centered",
            initial_sidebar_state="auto"
        )
        
        # Estilos CSS personalizados
        st.markdown("""
        <style>
        .user-message {
            background-color: #e6f7ff;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-bottom-right-radius: 5px;
            text-align: right;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f0f0f0;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            border-bottom-left-radius: 5px;
            margin-right: 20%;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            width: 100%;
        }
        .error-message {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def render_header(self):
        """Renderiza o cabeçalho da página."""
        st.markdown('<div class="chat-header">', unsafe_allow_html=True)
        st.title("🤖 Chat com IA (Hugging Face)")
        st.markdown("Converse com um modelo de chat mais fluido (BlenderBot)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    def render_conversation(self, history: List[dict]):
        """Renderiza a conversa com estilos CSS."""
        for message in history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">👤 {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                # Checagem se a resposta contém erro para destacar
                content = message["content"]
                if content.startswith("⚠️"):
                    st.markdown(
                        f'<div class="error-message">🤖 {content}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="bot-message">🤖 {content}</div>', 
                        unsafe_allow_html=True
                    )
    
    def render_input_form(self) -> Tuple[str, bool]:
        """Renderiza o formulário de entrada."""
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Você:", 
                placeholder="Digite sua mensagem aqui...",
                key="user_input"
            )
            col1, col2 = st.columns([4, 1])
            with col2:
                submitted = st.form_submit_button("Enviar")
                
        return user_input, submitted
        
    def render_sidebar(self):
        """Renderiza a barra lateral com informações e controles."""
        with st.sidebar:
            st.header("Sobre")
            st.info(
                "Este chat usa o modelo BlenderBot da Meta através da "
                "API da Hugging Face. As respostas são armazenadas em cache "
                "para melhorar o desempenho."
            )
            
            # Botão para limpar conversa com callback
            if st.button("Limpar Conversa", key="clear_btn"):
                # Limpeza explícita do histórico
                if "user_inputs" in st.session_state:
                    st.session_state.user_inputs = []
                if "bot_responses" in st.session_state:
                    st.session_state.bot_responses = []
                # Recarrega a página com st.rerun()
                st.rerun()
                
            st.subheader("Configurações")
            # Ajustando o nome para max_tokens para corresponder ao uso no código
            st.slider(
                "Contexto máximo (tokens)",
                min_value=100,
                max_value=1000,
                value=MAX_HISTORY_TOKENS,
                step=100,
                key="max_tokens"
            )
            
            # Adiciona informações de depuração se necessário
            if st.checkbox("Mostrar debug info", value=False):
                st.write(f"Versão Streamlit: {st.__version__}")
                st.write(f"Turnos na conversa: {len(st.session_state.get('user_inputs', []))}")
                st.write(f"Tamanho do histórico: {sum(len(m) for m in st.session_state.get('user_inputs', []) + st.session_state.get('bot_responses', []))} caracteres")

def main():
    """Função principal da aplicação."""
    try:
        # Inicializa o banco de dados
        init_db()
        
        # Inicializa componentes
        ui = ChatUI()
        session = ChatSession()
        
        # Renderiza interface
        ui.render_header()
        ui.render_sidebar()
        
        # Renderiza conversas anteriores
        history = session.get_history()
        ui.render_conversation(history)
        
        # Formulário de entrada
        user_input, submitted = ui.render_input_form()
        
        # Processa a entrada do usuário
        if submitted and user_input:
            with st.spinner("Pensando..."):
                try:
                    # Atualiza o valor do MAX_HISTORY_TOKENS com base no slider
                    global MAX_HISTORY_TOKENS
                    MAX_HISTORY_TOKENS = st.session_state.get("max_tokens", MAX_HISTORY_TOKENS)
                    
                    # Processa a mensagem do usuário
                    session.add_message(user_input)
                    
                    # Recarrega a página
                    st.rerun()
                except Exception as e:
                    logger.error(f"Erro ao processar mensagem: {e}")
                    st.error(f"Ocorreu um erro ao processar sua mensagem: {str(e)}")
                    
    except Exception as e:
        # Melhor tratamento de exceções para evitar falhas silenciosas
        logger.error(f"Erro na execução principal: {e}")
        st.error(f"Ocorreu um erro na aplicação: {str(e)}")
        st.info("Detalhes para suporte técnico foram registrados no log.")
        
        # Adiciona botão para reiniciar aplicação em caso de erro
        if st.button("Reiniciar Aplicação"):
            # Limpa todo o estado da sessão para um reset completo
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()