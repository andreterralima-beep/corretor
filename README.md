# 📋 Corretor de Provas com IA

Aplicação web para corrigir provas de múltipla escolha usando visão computacional via Claude AI.

## Funcionalidades

- 📸 Tira foto do gabarito e extrai as respostas automaticamente com IA
- ✅ Permite revisar/editar o gabarito extraído
- ⚖️ Define o peso de cada questão
- 📝 Tira foto da prova do aluno e corrige automaticamente
- 📊 Exibe nota na escala 0–10 e detalhamento questão por questão

## Como rodar localmente

```bash
# 1. Clone o repo
git clone <seu-repo>
cd corretor-provas

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure a chave da API
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Rode
python app.py
```

Acesse `http://localhost:5000`

## Deploy no Render

1. Faça push deste repositório para o GitHub
2. Acesse [render.com](https://render.com) e crie um novo **Web Service**
3. Conecte seu repositório GitHub
4. O `render.yaml` já configura tudo automaticamente
5. Adicione a variável de ambiente `ANTHROPIC_API_KEY` com sua chave da Anthropic
6. Clique em **Deploy**

## Estrutura

```
corretor-provas/
├── app.py              # Backend Flask
├── templates/
│   └── index.html      # Frontend completo (single-page)
├── requirements.txt
├── render.yaml         # Config do Render
└── .gitignore
```
