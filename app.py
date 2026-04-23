import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def detectar_por_materias(image_file):
    # Ordem exata das matérias no seu papel (3 colunas)
    nomes_materias = [
        "Língua Portuguesa", "Matemática", "Biologia",
        "Química", "Física", "História",
        "Geografia", "Língua Inglesa", "Educação Física",
        "Arte", "Sociologia", "Filosofia"
    ]
    
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Detecta os blocos retangulares das disciplinas
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocos = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 180]
    
    # Ordena: Linha primeiro (y), depois Coluna (x)
    blocos.sort(key=lambda b: (b[1] // 100, b[0]))

    resultado = {}
    letras = ['A', 'B', 'C', 'D', 'E']

    for i, (bx, by, bw, bh) in enumerate(blocos):
        if i >= len(nomes_materias): break
        materia = nomes_materias[i]
        resultado[materia] = []
        
        roi = thresh[by:by+bh, bx:bx+bw]
        c_internos, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        sqs = []
        for ci in c_internos:
            ix, iy, iw, ih = cv2.boundingRect(ci)
            if 15 < iw < 45 and 0.8 < (iw/ih) < 1.2:
                sqs.append((ix, iy, ci))
        
        sqs.sort(key=lambda s: s[1]) # Ordena questões de cima para baixo

        for j in range(0, len(sqs), 5):
            linha = sqs[j:j+5]
            if len(linha) < 5: continue
            linha.sort(key=lambda s: s[0]) # Ordena A, B, C, D, E
            
            votos = []
            for (_, _, c) in linha:
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                votos.append(cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask)))
            
            res = letras[np.argmax(votos)] if max(votos) > 45 else "?"
            resultado[materia].append(res)
    return resultado

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get('imagem')
        return jsonify({"gabarito": detectar_por_materias(f)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f = request.files.get('imagem')
        gab_mestre = json.loads(request.form.get('gabarito'))
        nome = request.form.get('nome_aluno', 'Estudante')
        res_aluno = detectar_por_materias(f)

        acertos, total = 0, 0
        for mat, q_mestre in gab_mestre.items():
            q_aluno = res_aluno.get(mat, [])
            for idx, resp in enumerate(q_mestre):
                total += 1
                if idx < len(q_aluno) and str(q_aluno[idx]) == str(resp):
                    acertos += 1
        
        nota = round((acertos / total) * 10, 2) if total > 0 else 0
        return jsonify({"nome": nome, "nota": nota, "acertos": acertos, "total": total})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
