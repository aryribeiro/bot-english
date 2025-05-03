# Bot English (chatbot de IA c/ BlenderBot via Hugging Face)

Este é um aplicativo de chat interativo construído com **Streamlit**, utilizando o modelo **BlenderBot-400M-distill** da **Hugging Face**. Ele mantém o histórico de conversas e utiliza cache em banco de dados SQLite para otimizar o desempenho.

---

## 🚀 Funcionalidades

- Interface web com histórico de mensagens.
- Armazenamento local em SQLite para cache de conversas.
- Truncamento automático de contexto para evitar exceder limites de tokens.
- Mecanismo de retry com backoff exponencial.
- Configuração via variáveis de ambiente.

---

## ⚙️ Requisitos

- Python 3.8 ou superior
- Conta e token da Hugging Face (https://huggingface.co)

---

## 📦 Instalação Passo a Passo

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

2. Crie um ambiente virtual

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

Windows:

python -m venv venv
venv\Scripts\activate

3. Instale as dependências

pip install -r requirements.txt

4. Configure a chave da Hugging Face

Crie um arquivo .env na raiz do projeto com o seguinte conteúdo:

HF_API_KEY=sua_chave_da_huggingface

Você pode obter a chave em: https://huggingface.co/settings/tokens


▶️ Executando o App

Use o Streamlit para iniciar o app:

streamlit run app.py

Abra o navegador em http://localhost:8501.