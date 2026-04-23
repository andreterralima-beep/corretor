import os
import cv2
import numpy as np
import json
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ──────────────────────────────────────────────
# ESTRUTURA DA PROVA (Jales Machado)
# ──────────────────────────────────────────────
ESTRUTURA_PROVA = {
    "col1": [
        ("Língua Portuguesa", 10),
        ("Química", 4),
        ("Geografia", 4),
        ("Arte", 2),
    ],
    "col2": [
        ("Matemática", 10),
        ("Física", 4),
        ("Língua Inglesa", 2),
        ("Sociologia", 2),
    ],
    "col3": [
        ("Biologia", 4),
        ("História", 4),
        ("Educação Física", 2),
        ("Filosofia", 2),
    ],
}

LETRAS = ["A", "B", "C", "D", "E"]


# ──────────────────────────────────────────────
# FUNÇÕES DE DETECÇÃO
# ──────────────────────────────────────────────

def extrair_boxes_global(region_img, x_offset, y_offset):
    """
    Usa threshold global (melhor para folhas com letras dentro das caixas
    + marcações hachuradas — ex.: coluna de Matemática).
    """
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h if h > 0 else 0
        if 28 < w < 75 and 22 < h < 65 and 0.6 < ratio < 1.7:
            roi = thresh[y : y + h, x : x + w]
            fill = cv2.countNonZero(roi) / (w * h) * 100
            boxes.append(
                {
                    "x": x + x_offset,
                    "y": y + y_offset,
                    "w": w,
                    "h": h,
                    "fill": fill,
                }
            )
    return boxes


def extrair_boxes_adapt(region_img, x_offset, y_offset):
    """
    Usa threshold adaptativo (melhor para colunas sem texto dentro das caixas
    — variações de iluminação e marcações hachuradas).
    """
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h if h > 0 else 0
        if 30 < w < 75 and 25 < h < 70 and 0.6 < ratio < 1.7:
            roi = thresh[y : y + h, x : x + w]
            fill = cv2.countNonZero(roi) / (w * h) * 100
            boxes.append(
                {
                    "x": x + x_offset,
                    "y": y + y_offset,
                    "w": w,
                    "h": h,
                    "fill": fill,
                }
            )
    return boxes


def dedup_por_x(linha, gap=15):
    """Remove boxes duplicadas com X muito próximo (RETR_LIST detecta
    bordas internas e externas), mantendo a de maior fill."""
    linha = sorted(linha, key=lambda b: b["x"])
    deduped, i = [], 0
    while i < len(linha):
        cluster = [linha[i]]
        j = i + 1
        while j < len(linha) and linha[j]["x"] - linha[i]["x"] < gap:
            cluster.append(linha[j])
            j += 1
        deduped.append(max(cluster, key=lambda b: b["fill"]))
        i = j
    return deduped


def agrupar_linhas(boxes, gap=20):
    """Agrupa boxes em linhas por proximidade de Y."""
    if not boxes:
        return []
    grupos, grupo_atual, prev_y = [], [], -100
    for b in sorted(boxes, key=lambda b: b["y"]):
        if b["y"] - prev_y > gap:
            if grupo_atual:
                grupos.append(dedup_por_x(grupo_atual)[:5])
            grupo_atual = [b]
        else:
            grupo_atual.append(b)
        prev_y = b["y"]
    if grupo_atual:
        grupos.append(dedup_por_x(grupo_atual)[:5])
    return grupos


def extrair_letra(linha):
    """Retorna a letra marcada na linha (ou '?' se ambíguo)."""
    if len(linha) < 5:
        return "?"
    marcadas = [i for i, b in enumerate(linha) if b["fill"] > 55]
    return LETRAS[marcadas[0]] if len(marcadas) == 1 else "?"


def detectar_respostas(image_file):
    """
    Lê a imagem do gabarito/prova e retorna as respostas detectadas
    junto com uma imagem de debug em base64.
    """
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {}, ""

    h, w = img.shape[:2]

    # ── Dividir em 3 colunas ──────────────────────────────────────────────
    # Os limites abaixo foram calibrados para a folha Jales Machado 1359×1600px.
    # Se a sua imagem tiver resolução diferente, ajuste as proporções:
    #   col1_end  ≈ 32% da largura
    #   col2_start≈ 33%   col2_end ≈ 64%
    #   col3_start≈ 64%
    col1_end   = int(w * 0.32)
    col2_start = int(w * 0.33)
    col2_end   = int(w * 0.64)
    col3_start = int(w * 0.64)

    # A coluna 2 (Matemática) tem letras dentro das caixas → threshold global
    # A parte inferior da col2 (Ing/Sociologia) usa threshold adaptativo
    col2_split_y = int(h * 0.41)   # separação Física/Inglesa dentro da col2

    col1_boxes  = extrair_boxes_adapt(img[:, :col1_end], 0, 0)
    col2a_boxes = extrair_boxes_global(img[:col2_split_y, col2_start:col2_end], col2_start, 0)
    col2b_boxes = extrair_boxes_adapt(img[col2_split_y:, col2_start:col2_end], col2_start, col2_split_y)
    col3_boxes  = extrair_boxes_adapt(img[:, col3_start:], col3_start, 0)

    linhas1 = agrupar_linhas(col1_boxes)
    linhas2 = agrupar_linhas(col2a_boxes + col2b_boxes)
    linhas3 = agrupar_linhas(col3_boxes)

    # ── Montar resultado ──────────────────────────────────────────────────
    resultado = {}
    for col_key, linhas in [("col1", linhas1), ("col2", linhas2), ("col3", linhas3)]:
        ptr = 0
        for nome, qtd in ESTRUTURA_PROVA[col_key]:
            resps = []
            for _ in range(qtd):
                if ptr < len(linhas):
                    resps.append(extrair_letra(linhas[ptr]))
                    ptr += 1
                else:
                    resps.append("?")
            resultado[nome] = resps

    # ── Imagem de debug com boxes coloridas ──────────────────────────────
    debug_img = img.copy()
    all_boxes = col1_boxes + col2a_boxes + col2b_boxes + col3_boxes
    for b in all_boxes:
        cor = (0, 200, 0) if b["fill"] > 55 else (0, 80, 200)
        thickness = 3 if b["fill"] > 55 else 1
        cv2.rectangle(
            debug_img,
            (b["x"], b["y"]),
            (b["x"] + b["w"], b["y"] + b["h"]),
            cor,
            thickness,
        )

    # Escala de 50 % para reduzir peso do base64
    scale = 0.5
    debug_small = cv2.resize(
        debug_img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    _, buf = cv2.imencode(".jpg", debug_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    debug_b64 = base64.b64encode(buf).decode()

    return resultado, debug_b64


# ──────────────────────────────────────────────
# ROTAS
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get("imagem")
        gabarito, debug_b64 = detectar_respostas(f)
        return jsonify({"gabarito": gabarito, "debug_img": debug_b64})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f = request.files.get("imagem")
        gab_mestre = json.loads(request.form.get("gabarito"))
        nome = request.form.get("nome_aluno", "Estudante")

        res_aluno, debug_b64 = detectar_respostas(f)

        acertos, total = 0, 0
        detalhes = {}
        for mat, q_mestre in gab_mestre.items():
            q_aluno = res_aluno.get(mat, [])
            mat_acertos = 0
            mat_resps = []
            for idx, resp_mestra in enumerate(q_mestre):
                total += 1
                resp_aluno = str(q_aluno[idx]) if idx < len(q_aluno) else "?"
                correto = resp_aluno == str(resp_mestra)
                if correto:
                    acertos += 1
                    mat_acertos += 1
                mat_resps.append(
                    {
                        "questao": idx + 1,
                        "gabarito": resp_mestra,
                        "aluno": resp_aluno,
                        "correto": correto,
                    }
                )
            detalhes[mat] = {"acertos": mat_acertos, "total": len(q_mestre), "questoes": mat_resps}

        nota = round((acertos / total) * 10, 2) if total > 0 else 0
        return jsonify(
            {
                "nome": nome,
                "nota": nota,
                "acertos": acertos,
                "total": total,
                "detalhes": detalhes,
                "debug_img": debug_b64,
            }
        )
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
