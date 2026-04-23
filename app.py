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

def extract_respostas(image_b64, num_questoes):
    prompt = f"""Você está analisando a foto de uma PROVA respondida por um aluno.
A prova tem {num_questoes} questões de múltipla escolha com alternativas A, B, C, D ou E.

Identifique qual alternativa o aluno marcou em cada questão.
Retorne SOMENTE um JSON válido, sem nenhum texto antes ou depois, no formato:
{{"respostas": {{"1": "A", "2": "B", "3": "C", ...}}}}

Se não conseguir identificar uma questão ou ela estiver em branco, use null para ela.
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
    return data["respostas"]

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

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
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

        num_questoes = len(gabarito)
        image_b64 = encode_image(file)
        respostas = extract_respostas(image_b64, num_questoes)

        # Calcular nota
        total_peso = 0
        acertos_peso = 0
        detalhes = {}

        for q, resp_correta in gabarito.items():
            peso = float(pesos.get(q, 1.0))
            resp_aluno = respostas.get(q)
            total_peso += peso
            acertou = resp_aluno and resp_aluno.upper() == resp_correta.upper()
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
            "total": num_questoes,
            "detalhes": detalhes
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
