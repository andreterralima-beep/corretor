import os, cv2, numpy as np, json, base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ESTRUTURA = {
    "col1": [("Língua Portuguesa", 10), ("Química", 4), ("Geografia", 4), ("Arte", 2)],
    "col2": [("Matemática", 10), ("Física", 4), ("Língua Inglesa", 2), ("Sociologia", 2)],
    "col3": [("Biologia", 4), ("História", 4), ("Educação Física", 2), ("Filosofia", 2)],
}
LETRAS = ["A", "B", "C", "D", "E"]


def _detectar_modo(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_pixels = cv2.countNonZero(cv2.inRange(hsv, np.array([90,60,60]), np.array([140,255,255])))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return "azul_branco" if (blue_pixels > 5000 and np.mean(gray) > 160) else "hachura_escuro"


def _gap_medio(caixas):
    prev_y, linhas_tmp, g = -100, [], []
    for b in sorted(caixas, key=lambda b: b["y"]):
        if b["y"] - prev_y > 18:
            if g: linhas_tmp.append(sorted(g, key=lambda b: b["x"]))
            g = [b]
        else: g.append(b)
        prev_y = b["y"]
    if g: linhas_tmp.append(sorted(g, key=lambda b: b["x"]))
    gaps = []
    for l in linhas_tmp:
        xs = []
        for b in sorted(l, key=lambda b: b["x"]):
            if not xs or b["x"] - xs[-1] > 12: xs.append(b["x"])
        if len(xs) == 5:
            for i in range(4): gaps.append(xs[i+1] - xs[i])
    return float(np.median(gaps)) if gaps else 44.0


def _dedup(linha, gap=12):
    linha = sorted(linha, key=lambda b: b["x"])
    deduped, i = [], 0
    while i < len(linha):
        cluster = [linha[i]]
        j = i+1
        while j < len(linha) and linha[j]["x"] - linha[i]["x"] < gap:
            cluster.append(linha[j]); j += 1
        deduped.append(max(cluster, key=lambda b: b["blue"]))
        i = j
    return deduped


def _agrupar_linhas(boxes):
    if not boxes: return []
    grupos, grupo, prev_y = [], [], -100
    for b in sorted(boxes, key=lambda b: b["y"]):
        if b["y"] - prev_y > 18:
            if grupo:
                d = _dedup(grupo)
                if len(d) >= 3: grupos.append(d)
            grupo = [b]
        else: grupo.append(b)
        prev_y = b["y"]
    if grupo:
        d = _dedup(grupo)
        if len(d) >= 3: grupos.append(d)
    return grupos


def _letra_por_offset(linha, gap):
    if not linha: return "?"
    marcada = max(linha, key=lambda b: b["blue"])
    if marcada["blue"] < 10: return "?"
    x_ancora = min(b["x"] for b in linha)
    idx = round((marcada["x"] - x_ancora) / gap)
    return LETRAS[max(0, min(4, idx))]


def _extrair_caixas_azul_branco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([90,60,60]), np.array([140,255,255]))
    _, thresh_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    cnts, _ = cv2.findContours(thresh_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    caixas = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        ratio = bw/bh if bh > 0 else 0
        if 28 < bw < 60 and 18 < bh < 45 and 0.8 < ratio < 2.2:
            roi = mask_blue[y:y+bh, x:x+bw]
            blue = cv2.countNonZero(roi) / (bw*bh) * 100
            caixas.append({"x":x,"y":y,"w":bw,"h":bh,"blue":blue})
    return caixas


def _extrair_caixas_hachura(img):
    h, w = img.shape[:2]

    def adapt_boxes(region, x_off, y_off):
        gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        t = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_r,(5,5),0),255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,4)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = bw/bh if bh > 0 else 0
            if 30 < bw < 75 and 25 < bh < 70 and 0.6 < ratio < 1.7:
                roi = t[y:y+bh, x:x+bw]
                fill = cv2.countNonZero(roi)/(bw*bh)*100
                boxes.append({"x":x+x_off,"y":y+y_off,"w":bw,"h":bh,"blue":fill})
        return boxes

    def global_boxes(region, x_off, y_off):
        gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(gray_r, 180, 255, cv2.THRESH_BINARY_INV)
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
        cnts, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = bw/bh if bh > 0 else 0
            if 28 < bw < 75 and 22 < bh < 65 and 0.6 < ratio < 1.7:
                roi = t[y:y+bh, x:x+bw]
                fill = cv2.countNonZero(roi)/(bw*bh)*100
                boxes.append({"x":x+x_off,"y":y+y_off,"w":bw,"h":bh,"blue":fill})
        return boxes

    c1e = int(w*0.32); c2s = int(w*0.33); c2e = int(w*0.64); c2sp = int(h*0.41)
    caixas  = adapt_boxes(img[:, :c1e], 0, 0)
    caixas += global_boxes(img[:c2sp, c2s:c2e], c2s, 0)
    caixas += adapt_boxes(img[c2sp:, c2s:c2e], c2s, c2sp)
    caixas += adapt_boxes(img[:, c2e:], c2e, 0)
    return caixas


def detectar_respostas(image_file):
    raw = image_file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return {}, ""

    h, w = img.shape[:2]
    modo = _detectar_modo(img)
    caixas = _extrair_caixas_azul_branco(img) if modo == "azul_branco" else _extrair_caixas_hachura(img)
    gap = _gap_medio(caixas)

    if modo == "azul_branco":
        c1e, c2s, c2e = int(w*0.34), int(w*0.34), int(w*0.67)
    else:
        c1e, c2s, c2e = int(w*0.32), int(w*0.33), int(w*0.64)

    l1 = _agrupar_linhas([b for b in caixas if b["x"] < c1e])
    l2 = _agrupar_linhas([b for b in caixas if c2s <= b["x"] < c2e])
    l3 = _agrupar_linhas([b for b in caixas if b["x"] >= c2e])

    resultado = {}
    for col_key, linhas in [("col1",l1),("col2",l2),("col3",l3)]:
        ptr = 0
        for nome, qtd in ESTRUTURA[col_key]:
            resps = []
            for _ in range(qtd):
                if ptr < len(linhas): resps.append(_letra_por_offset(linhas[ptr], gap)); ptr += 1
                else: resps.append("?")
            resultado[nome] = resps

    debug = img.copy()
    thr = 55 if modo == "hachura_escuro" else 10
    for b in caixas:
        cor = (0,200,0) if b["blue"] > thr else (0,80,200)
        esp = 3 if b["blue"] > thr else 1
        cv2.rectangle(debug,(b["x"],b["y"]),(b["x"]+b["w"],b["y"]+b["h"]),cor,esp)

    debug_small = cv2.resize(debug,(int(w*0.5),int(h*0.5)),interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", debug_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return resultado, base64.b64encode(buf).decode()


@app.route("/")
def index(): return render_template("index.html")

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
        f          = request.files.get("imagem")
        gab_mestre = json.loads(request.form.get("gabarito"))
        nome       = request.form.get("nome_aluno", "Estudante")
        res_aluno, debug_b64 = detectar_respostas(f)
        acertos, total = 0, 0
        detalhes = {}
        for mat, q_mestre in gab_mestre.items():
            q_aluno = res_aluno.get(mat, [])
            mat_ac  = 0
            questoes = []
            for idx, gab in enumerate(q_mestre):
                total += 1
                aluno = str(q_aluno[idx]) if idx < len(q_aluno) else "?"
                ok = aluno == str(gab)
                if ok: acertos += 1; mat_ac += 1
                questoes.append({"questao":idx+1,"gabarito":gab,"aluno":aluno,"correto":ok})
            detalhes[mat] = {"acertos":mat_ac,"total":len(q_mestre),"questoes":questoes}
        nota = round((acertos/total)*10, 2) if total > 0 else 0
        return jsonify({"nome":nome,"nota":nota,"acertos":acertos,"total":total,
                        "detalhes":detalhes,"debug_img":debug_b64})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=True)
