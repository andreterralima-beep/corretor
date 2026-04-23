import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def processar_gabarito_especifico(image_file):
    # Converte a imagem para OpenCV
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {}

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Identifica respostas (Simulação base para edição no frontend)
    respostas = {}
    for i in range(1, 51):
        respostas[str(i)] = "A" 

    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        if not file:
            return jsonify({"erro": "Imagem não enviada"}), 400
        
        gabarito = processar_gabarito_especifico(file)
        return jsonify({"gabarito": gabarito})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        file = request.files.get("imagem")
        gabarito_str = request.form.get("gabarito")
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        if not file or not gabarito_str:
            return jsonify({"erro": "Dados incompletos"}), 400

        gabarito = json.loads(gabarito_str)
        respostas_aluno = processar_gabarito_especifico(file)

        acertos = 0
        detalhes = {}
        for q, resp_correta in gabarito.items():
            resp_aluno = respostas_aluno.get(q, "—")
            acertou = str(resp_aluno).upper() == str(resp_correta).upper()
            if acertou: 
                acertos += 1
            detalhes[q] = {
                "gabarito": resp_correta, 
                "resposta": resp_aluno, 
                "acertou": acertou
            }

        nota = round((acertos / len(gabarito)) * 10, 2) if gabarito else 0
        return jsonify({
            "nome_aluno": nome_aluno, 
            "nota": nota, 
            "acertos": acertos, 
            "total": len(gabarito), 
            "detalhes": detalhes
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
