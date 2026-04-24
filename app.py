import os, cv2, numpy as np, json, base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ESTRUTURA = {
    "col1": [("Língua Portuguesa",10),("Química",4),("Geografia",4),("Arte",2)],
    "col2": [("Matemática",10),("Física",4),("Língua Inglesa",2),("Sociologia",2)],
    "col3": [("Biologia",4),("História",4),("Educação Física",2),("Filosofia",2)],
}
LETRAS = ["A","B","C","D","E"]


# ── Detecção de modo ─────────────────────────────────────────────────────────
def _detectar_modo(img):
    """
    Retorna:
      'azul_branco'  – fundo branco, marcação azul sólida (caneta esferográfica azul)
      'preto_cinza'  – fundo cinza, marcação preta (caneta/piloto preto)
      'hachura'      – fundo escuro, marcações hachuradas
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_px = cv2.countNonZero(cv2.inRange(hsv, np.array([90,60,60]), np.array([140,255,255])))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean    = float(np.mean(gray))

    if blue_px > 5000 and mean > 160:
        return "azul_branco"
    if mean >= 110:          # fundo cinza/claro com tinta preta
        return "preto_cinza"
    return "hachura"         # fundo escuro com hachura


# ── Utilitários compartilhados ────────────────────────────────────────────────
def _dedup(linha, gap=12):
    linha = sorted(linha, key=lambda b: b["x"]); deduped, i = [], 0
    while i < len(linha):
        cluster = [linha[i]]; j = i+1
        while j < len(linha) and linha[j]["x"] - linha[i]["x"] < gap:
            cluster.append(linha[j]); j += 1
        deduped.append(max(cluster, key=lambda b: b["blue"])); i = j
    return deduped

def _agrupar(boxes, min_caixas=3):
    if not boxes: return []
    grupos, grupo, prev_y = [], [], -100
    for b in sorted(boxes, key=lambda b: b["y"]):
        if b["y"] - prev_y > 18:
            if grupo:
                d = _dedup(grupo)
                if len(d) >= min_caixas: grupos.append(d)
            grupo = [b]
        else: grupo.append(b)
        prev_y = b["y"]
    if grupo:
        d = _dedup(grupo)
        if len(d) >= min_caixas: grupos.append(d)
    return grupos

def _gap_medio(caixas):
    prev_y, tmp, g = -100, [], []
    for b in sorted(caixas, key=lambda b: b["y"]):
        if b["y"] - prev_y > 18:
            if g: tmp.append(sorted(g, key=lambda b: b["x"]))
            g = [b]
        else: g.append(b)
        prev_y = b["y"]
    if g: tmp.append(sorted(g, key=lambda b: b["x"]))
    gaps = []
    for l in tmp:
        xs = []
        for b in sorted(l, key=lambda b: b["x"]):
            if not xs or b["x"] - xs[-1] > 12: xs.append(b["x"])
        if len(xs) == 5:
            for i in range(4): gaps.append(xs[i+1] - xs[i])
    return float(np.median(gaps)) if gaps else 44.0

def _letra_por_offset(linha, gap, fill_min, ratio_min):
    if not linha: return "?"
    fills   = [b["blue"] for b in linha]
    max_f   = max(fills); max_i = fills.index(max_f)
    outros  = [f for i,f in enumerate(fills) if i != max_i]
    med     = float(np.mean(outros)) if outros else 0
    ratio   = max_f / med if med > 0 else 99
    if max_f >= fill_min and ratio >= ratio_min:
        x0  = min(b["x"] for b in linha)
        idx = round((linha[max_i]["x"] - x0) / gap)
        return LETRAS[max(0, min(4, idx))]
    return "?"


# ── Extratores por modo ───────────────────────────────────────────────────────
def _caixas_azul_branco(img):
    """Detecta bordas pretas e mede fill azul dentro de cada caixa."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([90,60,60]), np.array([140,255,255]))
    _, ti = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ti = cv2.morphologyEx(ti, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    cnts,_ = cv2.findContours(ti, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    caixas = []
    for c in cnts:
        x,y,bw,bh = cv2.boundingRect(c); ratio = bw/bh if bh>0 else 0
        if 28<bw<60 and 18<bh<45 and 0.8<ratio<2.2:
            roi = mask_blue[y:y+bh,x:x+bw]
            blue = cv2.countNonZero(roi)/(bw*bh)*100
            caixas.append({"x":x,"y":y,"w":bw,"h":bh,"blue":blue})
    return caixas, 10.0, 1.8   # fill_min, ratio_min


def _caixas_preto_cinza(img):
    """
    Detecta caixas com threshold adaptativo (normaliza fundo cinza)
    e mede o fill de tinta PRETA (thresh=100) dentro de cada caixa.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    adapt = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray,(5,5),0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    cnts,_ = cv2.findContours(adapt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    caixas = []
    for c in cnts:
        x,y,bw,bh = cv2.boundingRect(c); ratio = bw/bh if bh>0 else 0
        if 28<bw<65 and 18<bh<50 and 0.8<ratio<2.2:
            roi = mask_dark[y:y+bh,x:x+bw]
            fill = cv2.countNonZero(roi)/(bw*bh)*100
            caixas.append({"x":x,"y":y,"w":bw,"h":bh,"blue":fill})
    return caixas, 65.0, 2.5   # fill_min, ratio_min


def _caixas_hachura(img):
    """Threshold adaptativo + global para marcações hachuradas em fundo escuro."""
    h, w = img.shape[:2]

    def adapt_boxes(region, xo, yo):
        g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        t = cv2.adaptiveThreshold(cv2.GaussianBlur(g,(5,5),0),255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,4)
        t = cv2.morphologyEx(t,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        cnts,_ = cv2.findContours(t,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in cnts:
            x,y,bw,bh=cv2.boundingRect(c); ratio=bw/bh if bh>0 else 0
            if 30<bw<75 and 25<bh<70 and 0.6<ratio<1.7:
                roi=t[y:y+bh,x:x+bw]
                fill=cv2.countNonZero(roi)/(bw*bh)*100
                boxes.append({"x":x+xo,"y":y+yo,"w":bw,"h":bh,"blue":fill})
        return boxes

    def global_boxes(region, xo, yo):
        g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _,t = cv2.threshold(g,180,255,cv2.THRESH_BINARY_INV)
        t = cv2.morphologyEx(t,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        cnts,_ = cv2.findContours(t,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in cnts:
            x,y,bw,bh=cv2.boundingRect(c); ratio=bw/bh if bh>0 else 0
            if 28<bw<75 and 22<bh<65 and 0.6<ratio<1.7:
                roi=t[y:y+bh,x:x+bw]
                fill=cv2.countNonZero(roi)/(bw*bh)*100
                boxes.append({"x":x+xo,"y":y+yo,"w":bw,"h":bh,"blue":fill})
        return boxes

    c1e=int(w*0.32); c2s=int(w*0.33); c2e=int(w*0.64); c2sp=int(h*0.41)
    caixas  = adapt_boxes(img[:,:c1e], 0, 0)
    caixas += global_boxes(img[:c2sp,c2s:c2e], c2s, 0)
    caixas += adapt_boxes(img[c2sp:,c2s:c2e], c2s, c2sp)
    caixas += adapt_boxes(img[:,c2e:], c2e, 0)
    return caixas, 55.0, 1.4   # fill_min, ratio_min


# ── Pipeline principal ────────────────────────────────────────────────────────
def detectar_respostas(image_file):
    raw = image_file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return {}, ""

    h, w = img.shape[:2]
    modo = _detectar_modo(img)

    if   modo == "azul_branco":  caixas, fill_min, ratio_min = _caixas_azul_branco(img)
    elif modo == "preto_cinza":  caixas, fill_min, ratio_min = _caixas_preto_cinza(img)
    else:                        caixas, fill_min, ratio_min = _caixas_hachura(img)

    gap = _gap_medio(caixas)

    # Limites de coluna por modo
    if modo == "hachura":
        c1e, c2s, c2e = int(w*0.32), int(w*0.33), int(w*0.64)
    else:
        c1e, c2s, c2e = int(w*0.34), int(w*0.34), int(w*0.67)

    l1 = _agrupar([b for b in caixas if b["x"] < c1e])
    l2 = _agrupar([b for b in caixas if c2s <= b["x"] < c2e])
    l3 = _agrupar([b for b in caixas if b["x"] >= c2e])

    resultado = {}
    for col_key, linhas in [("col1",l1),("col2",l2),("col3",l3)]:
        ptr = 0
        for nome, qtd in ESTRUTURA[col_key]:
            resps = []
            for _ in range(qtd):
                if ptr < len(linhas):
                    resps.append(_letra_por_offset(linhas[ptr], gap, fill_min, ratio_min))
                    ptr += 1
                else: resps.append("?")
            resultado[nome] = resps

    # Imagem de debug
    debug = img.copy()
    for b in caixas:
        marcada = b["blue"] >= fill_min
        cor = (0,200,0) if marcada else (0,80,200)
        esp = 3 if marcada else 1
        cv2.rectangle(debug,(b["x"],b["y"]),(b["x"]+b["w"],b["y"]+b["h"]),cor,esp)

    ds = cv2.resize(debug,(int(w*0.5),int(h*0.5)),interpolation=cv2.INTER_AREA)
    _,buf = cv2.imencode(".jpg",ds,[cv2.IMWRITE_JPEG_QUALITY,75])
    return resultado, base64.b64encode(buf).decode()


# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get("imagem")
        gabarito, debug_b64 = detectar_respostas(f)
        return jsonify({"gabarito":gabarito,"debug_img":debug_b64})
    except Exception as e:
        return jsonify({"erro":str(e)}),500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f          = request.files.get("imagem")
        gab_mestre = json.loads(request.form.get("gabarito"))
        nome       = request.form.get("nome_aluno","Estudante")
        res_aluno, debug_b64 = detectar_respostas(f)
        acertos,total = 0,0
        detalhes = {}
        for mat,q_mestre in gab_mestre.items():
            q_aluno=res_aluno.get(mat,[]); mat_ac=0; questoes=[]
            for idx,gab in enumerate(q_mestre):
                total+=1; aluno=str(q_aluno[idx]) if idx<len(q_aluno) else "?"
                ok=aluno==str(gab)
                if ok: acertos+=1; mat_ac+=1
                questoes.append({"questao":idx+1,"gabarito":gab,"aluno":aluno,"correto":ok})
            detalhes[mat]={"acertos":mat_ac,"total":len(q_mestre),"questoes":questoes}
        nota=round((acertos/total)*10,2) if total>0 else 0
        return jsonify({"nome":nome,"nota":nota,"acertos":acertos,"total":total,
                        "detalhes":detalhes,"debug_img":debug_b64})
    except Exception as e:
        return jsonify({"erro":str(e)}),500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=True)
