import os
import base64
import json
import anthropic
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def encode_image(file):
    return base64.standard_b64encode(file.read()).decode("utf-8")

def extract_gabarito(image_b64, num_questoes):
    prompt = f"""Você está analisando a foto de um GABARITO de prova.
O gabarito tem {num_questoes} questões de múltipla escolha com alternativas A, B, C, D ou E.

Extraia as respostas corretas de cada questão.
Retorne SOMENTE um JSON válido, sem nenhum texto antes ou depois, no formato:
{{"gabarito": {{"1": "A", "2": "B", "3": "C", ...}}}}

Se não conseguir identificar uma questão, use null para ela.
"""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    text = response.content[0].text.strip()
    data = json.loads(text)
    return data["gabarito"]


def extract_respostas(image_b64, gabarito):
    """
    Lê as respostas do aluno na imagem.
    gabarito: dict {materia: {q_num: letra_correta}} OU dict plano {q_num: letra}
    Detecta automaticamente o formato e adapta o prompt.
    """

    # Verifica se é gabarito por matérias (dict de dicts) ou plano (dict simples)
    primeiro_valor = next(iter(gabarito.values())) if gabarito else None
    gabarito_por_materia = isinstance(primeiro_valor, dict)

    if gabarito_por_materia:
        # Formato: {materia: {q_num: letra}}
        secoes_desc = "\n".join(
            f"- {mat}: {len(qs)} questões (Q{', Q'.join(str(q) for q in sorted(qs, key=int))})"
            for mat, qs in gabarito.items()
        )
        ids_str = json.dumps(
            {mat: {q: None for q in qs} for mat, qs in gabarito.items()},
            ensure_ascii=False
        )
        prompt = f"""Você está analisando a foto de uma FOLHA DE RESPOSTAS de prova respondida por um aluno.

A folha está dividida em seções por MATÉRIA, dispostas em 3 colunas lado a lado.
Cada seção tem seu próprio conjunto de questões numeradas, com alternativas A, B, C, D ou E.

As seções e seus números de questão são:
{secoes_desc}

ATENÇÃO — erros comuns a evitar:
- Não confunda questões de matérias diferentes que ficam lado a lado na mesma linha visual
- Cada matéria tem sua própria numeração independente (ex: Matemática Q1 e Química Q1 são questões diferentes)
- A coluna do MEIO do layout pode conter uma matéria completamente diferente das colunas da esquerda e da direita
- Leia cada seção de cima para baixo, dentro de seu próprio bloco delimitado

Retorne SOMENTE um JSON válido, sem texto antes ou depois, no formato:
{{"respostas": {ids_str}}}

Substitua cada valor null pela letra que o aluno marcou (A, B, C, D ou E), ou deixe null se em branco/ilegível.
"""
    else:
        # Formato plano: {q_num: letra}
        num_questoes = len(gabarito)
        ids_str = json.dumps({q: None for q in gabarito}, ensure_ascii=False)
        prompt = f"""Você está analisando a foto de uma FOLHA DE RESPOSTAS de prova respondida por um aluno.
A prova tem {num_questoes} questões de múltipla escolha com alternativas A, B, C, D ou E.

ATENÇÃO — a folha pode ter questões dispostas em múltiplas colunas lado a lado.
Leia coluna por coluna, não linha por linha, para não misturar questões de colunas diferentes.

Retorne SOMENTE um JSON válido, sem texto antes ou depois, no formato:
{{"respostas": {ids_str}}}

Substitua cada valor null pela letra que o aluno marcou (A, B, C, D ou E), ou deixe null se em branco/ilegível.
"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    text = response.content[0].text.strip()
    # Remove possíveis blocos de markdown caso o modelo retorne ```json
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    data = json.loads(text)
    return data["respostas"]


# ─────────────────────────────────────────────
# ROTAS
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        num_questoes = int(request.form.get("num_questoes", 10))
        if not file:
            return jsonify({"erro": "Imagem não enviada"}), 400
        image_b64 = encode_image(file)
        gabarito = extract_gabarito(image_b64, num_questoes)
        return jsonify({"gabarito": gabarito})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/ler-prova", methods=["POST"])
def api_ler_prova():
    """Lê as respostas da prova e retorna para revisão manual antes de calcular a nota."""
    try:
        file = request.files.get("imagem")
        gabarito_str = request.form.get("gabarito")
        if not file:
            return jsonify({"erro": "Imagem não enviada"}), 400
        if not gabarito_str:
            return jsonify({"erro": "Gabarito não informado"}), 400
        gabarito = json.loads(gabarito_str)
        image_b64 = encode_image(file)
        respostas = extract_respostas(image_b64, gabarito)
        return jsonify({"respostas": respostas})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/calcular-nota", methods=["POST"])
def api_calcular_nota():
    """Calcula a nota com gabarito + respostas já revisadas/confirmadas pelo usuário."""
    try:
        gabarito = json.loads(request.form.get("gabarito"))
        respostas = json.loads(request.form.get("respostas"))
        pesos = json.loads(request.form.get("pesos", "{}"))
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        total_peso = 0
        acertos_peso = 0
        detalhes = {}

        for q, resp_correta in gabarito.items():
            peso = float(pesos.get(q, 1.0))
            resp_aluno = respostas.get(q)
            total_peso += peso
            acertou = bool(resp_aluno) and resp_aluno.upper() == resp_correta.upper()
            if acertou:
                acertos_peso += peso
            detalhes[q] = {
                "gabarito": resp_correta,
                "resposta": resp_aluno or "—",
                "acertou": acertou,
                "peso": peso
            }

        nota = round((acertos_peso / total_peso) * 10, 2) if total_peso > 0 else 0
        num_acertos = sum(1 for d in detalhes.values() if d["acertou"])

        return jsonify({
            "nome_aluno": nome_aluno,
            "nota": nota,
            "acertos": num_acertos,
            "total": len(gabarito),
            "detalhes": detalhes
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    """Rota legada: lê e calcula em uma única chamada (sem revisão manual)."""
    try:
        file = request.files.get("imagem")
        gabarito_str = request.form.get("gabarito")
        pesos_str = request.form.get("pesos")
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        if not file:
            return jsonify({"erro": "Imagem da prova não enviada"}), 400
        if not gabarito_str:
            return jsonify({"erro": "Gabarito não informado"}), 400

        gabarito = json.loads(gabarito_str)
        pesos = json.loads(pesos_str) if pesos_str else {}

        image_b64 = encode_image(file)
        respostas = extract_respostas(image_b64, gabarito)

        total_peso = 0
        acertos_peso = 0
        detalhes = {}

        for q, resp_correta in gabarito.items():
            peso = float(pesos.get(q, 1.0))
            resp_aluno = respostas.get(q)
            total_peso += peso
            acertou = bool(resp_aluno) and resp_aluno.upper() == resp_correta.upper()
            if acertou:
                acertos_peso += peso
            detalhes[q] = {
                "gabarito": resp_correta,
                "resposta": resp_aluno or "—",
                "acertou": acertou,
                "peso": peso
            }

        nota = round((acertos_peso / total_peso) * 10, 2) if total_peso > 0 else 0
        num_acertos = sum(1 for d in detalhes.values() if d["acertou"])

        return jsonify({
            "nome_aluno": nome_aluno,
            "nota": nota,
            "acertos": num_acertos,
            "total": len(gabarito),
            "detalhes": detalhes
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
