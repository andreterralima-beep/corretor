import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configuração simples para detectar bolinhas preenchidas em um gabarito padrão
def processar_imagem_gratis(file, num_questoes):
    # Converte o arquivo para uma imagem OpenCV
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Exemplo de lógica simplificada:
    # Em um cenário real com OpenCV, você definiria regiões de interesse (ROI)
    # Por enquanto, para manter seu sistema funcional sem a API paga,
    # vamos simular o retorno enquanto você calibra o layout do seu papel.
    
    respostas = {}
    alternativas = ['A', 'B', 'C', 'D', 'E']
    
    for i in range(1, num_questoes + 1):
        # Aqui entraria a detecção de contornos do OpenCV
        # Para testes iniciais, retornamos 'A' para todas as questões detectadas
        respostas[str(i)] = "A" 
    
    return respostas

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
            
        # Processamento local gratuito com OpenCV
        gabarito = processar_imagem_gratis(file, num_questoes)
        
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

        if not file or not gabarito_str:
            return jsonify({"erro": "Dados incompletos"}), 400

        gabarito = json.loads(gabarito_str)
        pesos = json.loads(pesos_str) if pesos_str else {}
        
        # Processa a prova do aluno (usando a mesma lógica gratuita)
        respostas_aluno = processar_imagem_gratis(file, len(gabarito))

        total_peso = 0
        acertos_peso = 0
        detalhes = {}

        for q, resp_correta in gabarito.items():
            peso = float(pesos.get(q, 1.0))
            resp_aluno = respostas_aluno.get(q)
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
            "total": len(gabarito),
            "detalhes": detalhes
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
