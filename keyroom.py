from PIL import Image
import random

matrix_casa = [
    ['soffitta_sinistra', 'camera_oscura', 'soffitta_destra'],
    ['corridoio_sinistro', 'balcone', 'corridoio_destro'],
    ['anta_sinistra', 'portone', 'anta_destra']
]

import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
import time

# ── flux2 ──────────────────────────────────────────────────────────────────────
from lycoris import create_lycoris_from_weights
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize
from deep_translator import GoogleTranslator
from tqdm import tqdm
import torch
import gc
from safetensors import safe_open

QUALITY_SUFFIX = ", realistic skin texture, 8k ultra detailed, sharp focus, photorealistic, studio lighting, high resolution"

def flux2(prompt_list, steps_var, output_dir, lora1, lora2, refs_list=None):
    dtype = torch.bfloat16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[VRAM] Libera prima del caricamento: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    print("Caricamento pipeline FLUX.2 klein 9B...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )

    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = None

    def is_lokr_lora(path):
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            return any("lokr_w" in k for k in keys)
        except Exception:
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
                return any("lokr_w" in k for k in sd.keys())
            except Exception:
                return False

    def load_lora(pipe, lora_path, adapter_name, weight=1.0):
        if is_lokr_lora(lora_path):
            print(f"  → Formato LoKr rilevato, uso LyCORIS per '{adapter_name}'")
            try:
                wrapper, _ = create_lycoris_from_weights(weight, lora_path, pipe.transformer)
                wrapper.merge_to()
                print(f"  → LoKr/LyCORIS merged nel transformer ✓")
                return "lycoris"
            except ImportError:
                print("  ✗ lycoris-lora non installata!")
                return False
            except Exception as e:
                print(f"  ✗ Errore LyCORIS: {e}")
                return False
        else:
            print(f"  → Formato standard LoRA, uso Diffusers per '{adapter_name}'")
            try:
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                print(f"  → LoRA standard caricata ✓")
                return "diffusers"
            except Exception as e:
                print(f"  ✗ Errore Diffusers load_lora_weights: {e}")
                return False

    adapter_names   = []
    adapter_weights = []

    if lora1 != 'no_lora':
        lora1_path = os.path.join("./lora", lora1)
        if os.path.exists(lora1_path):
            result = load_lora(pipe, lora1_path, "lora1", weight=1.0)
            if result == "diffusers":
                adapter_names.append("lora1")
                adapter_weights.append(1.0)
        else:
            print(f"  ✗ File non trovato: {lora1_path}")

    if lora2 != 'no_lora':
        lora2_path = os.path.join("./lora", lora2)
        if os.path.exists(lora2_path):
            result = load_lora(pipe, lora2_path, "lora2", weight=1.0)
            if result == "diffusers":
                adapter_names.append("lora2")
                adapter_weights.append(1.0)
        else:
            print(f"  ✗ File non trovato: {lora2_path}")

    if adapter_names:
        if len(adapter_names) == 1:
            pipe.set_adapters(adapter_names[0], adapter_weights=adapter_weights[0])
        else:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    print("Quantizzazione transformer...")
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)

    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    gc.collect()
    torch.cuda.empty_cache()

    REF_SIZE = 512

    for i, (nome, prompt) in enumerate(tqdm(prompt_list, desc="generazione immagini")):
        output_path = os.path.join(output_dir, f"{nome}.png")
        if os.path.exists(output_path):
            print(f"  → '{nome}' già esiste, skip.")
            continue

        images = []
        if refs_list and i < len(refs_list):
            ref_path = refs_list[i]
            if ref_path and os.path.exists(ref_path):
                img_ref = Image.open(ref_path).convert("RGB")
                w, h = img_ref.size
                if w >= h:
                    rw, rh = REF_SIZE, (REF_SIZE * h) // w
                else:
                    rw, rh = (REF_SIZE * w) // h, REF_SIZE
                img_ref = img_ref.resize((rw, rh), Image.BICUBIC)
                images.append(img_ref)

        if not prompt:
            prompt = "a beautiful portrait, photorealistic, high detail"

        try:
            prompt_en = GoogleTranslator(source='it', target='en').translate(prompt)
        except Exception as e:
            prompt_en = prompt

        gen_params = {
            "prompt":              prompt_en,
            "height":              1024,
            "width":               1024,
            "guidance_scale":      1.0,
            "num_inference_steps": steps_var,
            "generator":           torch.Generator(device="cpu").manual_seed(0)
        }

        if len(images) == 1:
            gen_params["image"] = images[0]
        elif len(images) > 1:
            gen_params["image"] = images

        image = pipe(**gen_params).images[0]
        image.save(output_path)
        print(f"[OK] Immagine salvata: {output_path}")

        gc.collect()
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def parse_prompt_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        entries = [
            line.strip().replace('"', '').replace('[', '').replace(']', '')
            for line in f.read().split(']')
        ]
    prompt_list = []
    for entry in entries:
        if not entry:
            continue
        nome, prompt = entry.split(':', 1)
        prompt_list.append((nome.strip(), prompt.strip()))
    return prompt_list


os.makedirs("./locations", exist_ok=True)
os.makedirs("./girls", exist_ok=True)

if not os.path.exists("prompt_location.txt"):
    print("file prompt_location.txt non trovato")
    sys.exit(1)

prompt_list_locations = parse_prompt_file("prompt_location.txt")

tutti_esistenti = all(
    any(os.path.exists(f"locations/{nome}{ext}") for ext in ['.png', '.jpg', '.jpeg'])
    for nome, _ in prompt_list_locations
)

if tutti_esistenti:
    print(f"Trovati {len(prompt_list_locations)} locations già generate. Skip flux2.")
else:
    flux2(prompt_list=prompt_list_locations, steps_var=8, output_dir="./locations", lora1='no_lora', lora2='no_lora')

if not os.path.exists("girls.txt"):
    print("file girls.txt non trovato")
    sys.exit(1)

REFS_DIR = "./riferimenti girls"
if not os.path.exists(REFS_DIR):
    print(f"cartella '{REFS_DIR}' non trovata")
    sys.exit(1)

refs = sorted([
    os.path.join(REFS_DIR, f)
    for f in os.listdir(REFS_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])[:10]

prompt_list_girls_raw = parse_prompt_file("girls.txt")[:10]
prompt_list_girls = [
    (os.path.splitext(os.path.basename(refs[i]))[0], prompt + QUALITY_SUFFIX)
    for i, (_, prompt) in enumerate(prompt_list_girls_raw)
    if i < len(refs)
]

nomi_refs = [os.path.splitext(os.path.basename(r))[0] for r in refs]

tutti_esistenti = all(
    any(os.path.exists(f"girls/{nome}{ext}") for ext in ['.png', '.jpg', '.jpeg'])
    for nome in nomi_refs
)

if tutti_esistenti:
    print(f"Tutte le {len(nomi_refs)} immagini girls sono già generate. Skip flux2.")
else:
    flux2(prompt_list=prompt_list_girls, steps_var=8, output_dir="./girls", lora1='no_lora', lora2='no_lora', refs_list=refs)


from rembg import remove

if os.path.exists("locations/balcone.png"):
    balcone = Image.open("locations/balcone.png")
else:
    print("balcone.png non trovato!")
    exit()

girls_files = [
    f for f in os.listdir("girls")
    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.endswith("_trsp.png")
]

tutti_trsp_esistenti = all(
    os.path.exists(f"girls/{os.path.splitext(f)[0]}_trsp.png")
    for f in girls_files
)

if tutti_trsp_esistenti:
    print(f"Trovati {len(girls_files)} file _trsp.png già generati. Skip rembg.")
else:
    for f in os.listdir("girls"):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if f.endswith("_trsp.png"):
            continue
        full_path = os.path.join("girls", f)
        img = Image.open(full_path)
        image_alfa = remove(img).convert("RGBA")
        nome_base = os.path.splitext(f)[0]
        image_alfa.save(f"girls/{nome_base}_trsp.png")


prompt_file = "prompt_rigenera.txt"
output_dir  = "./image_rigenera"

lora_map = {
    0: ('no_lora',                    'no_lora'),
    1: ('POVblowjobV1A.safetensors',  'no_lora'),
    2: ('FKmissionary.safetensors',   'no_lora'),
}

nome_map = {
    0: "bacio",
    1: "pompa",
    2: "monta",
}

with open(prompt_file, "r", encoding="utf-8") as f:
    raw = f.read()

prompts = [p.strip() for p in raw.split("_") if p.strip()]
os.makedirs(output_dir, exist_ok=True)

ESTENSIONI_VALIDE = {'.jpg', '.jpeg', '.png', '.webp'}
girls = [
    g for g in os.listdir("riferimenti girls")
    if os.path.splitext(g)[1].lower() in ESTENSIONI_VALIDE
]
totale = len(girls) * len(prompts)
mancanti = []

indice_girls = 1
for girl in girls:
    girl_name = os.path.splitext(girl)[0]
    for i, prompt_text in enumerate(prompts):
        nome_prompt = nome_map.get(i, f"prompt_{i}")
        nome_output = f"{indice_girls}_{girl_name}_{nome_prompt}"
        if not os.path.exists(f"{output_dir}/{nome_output}.png"):
            mancanti.append((indice_girls, girl, i, prompt_text))
    indice_girls += 1

print(f"\n{'='*60}")
print(f"Immagini totali:    {totale}")
print(f"Già generate:       {totale - len(mancanti)}")
print(f"Da generare:        {len(mancanti)}")
print(f"{'='*60}\n")

contatore = 0
for (idx_girl, girl, i, prompt_text) in tqdm(mancanti, desc="generazione immagini rigenera"):
    contatore += 1
    ref_path  = f"riferimenti girls/{girl}"
    girl_name = os.path.splitext(girl)[0]
    lora1, lora2 = lora_map.get(i, ('no_lora', 'no_lora'))
    nome_prompt  = nome_map.get(i, f"prompt_{i}")
    nome_output  = f"{idx_girl}_{girl_name}_{nome_prompt}"
    flux2(
        prompt_list=[(nome_output, prompt_text)],
        steps_var=8,
        output_dir=output_dir,
        lora1=lora1,
        lora2=lora2,
        refs_list=[ref_path]
    )


# ── Configurazione gioco ────────────────────────────────────────────────────
dir_location = "locations"

frecciasu       = "./freccie/frecciasu.png"
frecciagiu      = "./freccie/frecciagiu.png"
frecciasinistra = "./freccie/frecciasinistra.png"
frecciadestra   = "./freccie/frecciadestra.png"

FRECCIA_SIZE = (80, 80)

pos_riga = 1
pos_col  = 1

current_image = None
img_su = img_giu = img_sinistra = img_destra = None

def carica_frecce():
    global img_su, img_giu, img_sinistra, img_destra
    try:
        img_su       = ImageTk.PhotoImage(Image.open(frecciasu).resize(FRECCIA_SIZE, Image.LANCZOS))
        img_giu      = ImageTk.PhotoImage(Image.open(frecciagiu).resize(FRECCIA_SIZE, Image.LANCZOS))
        img_sinistra = ImageTk.PhotoImage(Image.open(frecciasinistra).resize(FRECCIA_SIZE, Image.LANCZOS))
        img_destra   = ImageTk.PhotoImage(Image.open(frecciadestra).resize(FRECCIA_SIZE, Image.LANCZOS))
        print("✅ Frecce caricate")
    except FileNotFoundError as e:
        print(f"⚠️ Freccia non trovata: {e}")

locations = ['soffitta_sinistra', 'camera_oscura', 'soffitta_destra', 'corridoio_sinistro',
             'corridoio_destro', 'anta_sinistra', 'portone', 'anta_destra']

selected = random.sample(locations, 4)
select_keygreen, select_keyyellow, select_keyblue, select_keymagenta = selected

key_positions = {}

key_files = {
    'green':   'keys/keygreen.png',
    'yellow':  'keys/keyoro.png',
    'blue':    'keys/keyblue.png',
    'magenta': 'keys/keymagenta.png',
}

def resize_key(path, size=50):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    if w >= h:
        img = img.resize((size, size * h // w), Image.LANCZOS)
    else:
        img = img.resize((size * w // h, size), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

baci_rimasti  = 3
pompe_rimaste = 2
amore_rimasto = 1
vita          = 100
rianima       = 1
time_dead     = 0

import cv2
import numpy as np

ESTENSIONI_VALIDE = {'.jpg', '.jpeg', '.png', '.webp'}

def get_girls_list():
    return sorted([
        os.path.splitext(g)[0]
        for g in os.listdir("riferimenti girls")
        if os.path.splitext(g)[1].lower() in ESTENSIONI_VALIDE
    ])

import pygame
pygame.mixer.init()

# ── caricamento suoni (vicino agli altri sound) ───────────────────────────
sound_bacio = pygame.mixer.Sound("kiss.wav")
sound_pompa = pygame.mixer.Sound("pompa.mp3")
sound_amore = pygame.mixer.Sound("amore.wav")

def mostra_immagine_cv2(img_path, titolo="immagine"):
    if not os.path.exists(img_path):
        print(f"  ✗ Immagine non trovata: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(titolo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mostra_menu_ragazza(indice, x_root, y_root):
    global baci_rimasti, pompe_rimaste, amore_rimasto, vita

    girls = get_girls_list()
    if indice < 0 or indice >= len(girls):
        return

    girl_name = girls[indice]
    idx_file  = indice + 1

    def get_img_path(tipo):
        return os.path.join(output_dir, f"{idx_file}_{girl_name}_{tipo}.png")

    popup = tk.Toplevel(window)
    popup.title("")
    popup.geometry(f"+{x_root}+{y_root}")
    popup.configure(bg="#1a1a1a")
    popup.resizable(False, False)

    tk.Label(popup, text=f"Ragazza {indice + 1} — {girl_name.capitalize()}",
             font=("Courier", 12, "bold"), fg="#ff3333", bg="#1a1a1a").pack(pady=6)

    info = tk.Label(popup,
                    text=f"💋 {baci_rimasti}  💦 {pompe_rimaste}  ❤️ {amore_rimasto}",
                    font=("Courier", 10), fg="#aaaaaa", bg="#1a1a1a")
    info.pack(pady=2)

    def usa_bacio(p):
        global baci_rimasti, vita
        if baci_rimasti <= 0 or vita >= 100:
            return
        baci_rimasti -= 1
        vita = min(100, vita + 5)
        sound_bacio.play()                                        # ← aggiunto
        mostra_immagine_cv2(get_img_path("bacio"), titolo="💋 Bacio")
        disegna_vita()
        p.destroy()

    def usa_pompa(p):
        global pompe_rimaste, vita
        if pompe_rimaste <= 0 or vita >= 100:
            return
        pompe_rimaste -= 1
        vita = min(100, vita + 50)
        sound_pompa.play()                                        # ← aggiunto
        mostra_immagine_cv2(get_img_path("pompa"), titolo="💦 Pompa")
        disegna_vita()
        p.destroy()

    def usa_amore(p):
        global amore_rimasto, vita
        if amore_rimasto <= 0 or vita >= 100:
            return
        amore_rimasto -= 1
        vita = min(100, vita + 100)
        sound_amore.play()                                        # ← aggiunto
        mostra_immagine_cv2(get_img_path("monta"), titolo="❤️ Amore")
        disegna_vita()
        p.destroy()

    tk.Button(popup, text=f"💋 Bacio  ({baci_rimasti} rimasti)",
              font=("Courier", 11), bg="#2a2a2a", fg="white",
              activebackground="#ff3333", relief="flat", cursor="hand2",
              state="normal" if baci_rimasti > 0 else "disabled",
              command=lambda p=popup: usa_bacio(p)).pack(fill="x", padx=16, pady=3)

    tk.Button(popup, text=f"💦 Pompa  ({pompe_rimaste} rimaste)",
              font=("Courier", 11), bg="#2a2a2a", fg="white",
              activebackground="#ff3333", relief="flat", cursor="hand2",
              state="normal" if pompe_rimaste > 0 else "disabled",
              command=lambda p=popup: usa_pompa(p)).pack(fill="x", padx=16, pady=3)

    tk.Button(popup, text=f"❤️ Amore  ({amore_rimasto} rimasto)",
              font=("Courier", 11), bg="#2a2a2a", fg="white",
              activebackground="#ff3333", relief="flat", cursor="hand2",
              state="normal" if amore_rimasto > 0 else "disabled",
              command=lambda p=popup: usa_amore(p)).pack(fill="x", padx=16, pady=3)

    tk.Button(popup, text="✕ Chiudi", font=("Courier", 9),
              bg="#111", fg="#555", relief="flat",
              command=popup.destroy).pack(pady=6)

serrature_aperte = set()
gioco_bloccato   = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CARICA FRAME ZOMPIE ────────────────────────────────────────────────────
arrays_zompie1 = []
arrays_zompie2 = []
arrays_zompie3 = []
arrays_zompie4 = []
arrays_zompie5 = []

def carica_frame(znome):
    imgz = Image.open(f"./frames_zompie/{znome}")
    w, h = imgz.size
    Y = (512 * h) // w
    return imgz.resize((512, Y), Image.BICUBIC)

files = sorted(os.listdir("./frames_zompie"), key=lambda f: int(f.split('.')[0].replace('Z', '')))

for znome in files:
    z = int(znome.split('.')[0].replace('Z', ''))
    if z <= 191:
        arrays_zompie1.append(carica_frame(znome))
    elif z <= 383:
        arrays_zompie2.append(carica_frame(znome))
    elif z <= 575:
        arrays_zompie3.append(carica_frame(znome))
    elif z <= 696:
        arrays_zompie4.append(carica_frame(znome))
    else:
        arrays_zompie5.append(carica_frame(znome))

print(f"Caricati: {len(arrays_zompie1)}, {len(arrays_zompie2)}, {len(arrays_zompie3)}, {len(arrays_zompie4)}, {len(arrays_zompie5)} frames")

# ── STATO ONDATE ZOMPIE ────────────────────────────────────────────────────
ondata_corrente   = 1
zompie_per_ondata = 2    # 2 → 4 → 8 → cap a 10
zompie_in_campo   = []   # lista di dict, uno per ogni zompie
_zompie_id_counter = 0
_loop_after_id     = None   # id del SINGOLO loop centrale

ZOMPIE_MAX        = 2   # mai più di 10 zompie contemporaneamente
ZOMPIE_DELAY_BASE = 42   # ms per frame alla ondata 1
ZOMPIE_DELAY_MIN  = 18   # limite minimo velocità

# variabili legacy per compatibilità con il resto del codice
zompie_k              = 0
zompie_array_corrente = []
zompie_x              = 0
zompie_y              = 0
zompie_after_id       = None
zompie_photo          = None

def _get_delay():
    d = int(ZOMPIE_DELAY_BASE * (0.85 ** ((ondata_corrente - 1) // 2)))
    return max(d, ZOMPIE_DELAY_MIN)

# ── LOOP CENTRALE (un solo after per tutti i zompie) ──────────────────────
def _loop_zompie():
    """Aggiorna TUTTI i zompie in un unico tick — zero loop paralleli."""
    global _loop_after_id, vita

    if not zompie_in_campo:
        _loop_after_id = None
        return

    canvas.delete("zompie")   # cancella tutto in un colpo solo

    photos = []   # lista temporanea per evitare garbage collection

    for z in zompie_in_campo:
        frames = z['frames']
        z['k'] = (z['k'] + 1) % len(frames)
        frame  = frames[z['k']]
        tag    = f"zompie_{z['id']}"

        photo = ImageTk.PhotoImage(frame)
        photos.append(photo)
        z['photo'] = photo   # mantieni riferimento nel dict
        canvas.create_image(z['x'], z['y'], anchor="sw",
                            image=photo, tags=("zompie", tag))

        # danno
        meta = len(frames) // 2
        if z['k'] >= meta:
            z['danno_tick'] = z.get('danno_tick', 0) + 1
            if z['danno_tick'] >= 24:
                z['danno_tick'] = 0
                vita = max(0, vita - 10)
                print(f"🩸 Zompie {z['id']} vicino! Vita: {vita}%")
                disegna_vita()
        else:
            z['danno_tick'] = 0

    canvas.tag_raise("pistola")

    _loop_after_id = canvas.after(_get_delay(), _loop_zompie)

def _avvia_loop():
    """Avvia il loop centrale se non è già in esecuzione."""
    global _loop_after_id
    if _loop_after_id is None:
        _loop_zompie()

def stoppa_zompie():
    global _loop_after_id, zompie_array_corrente, zompie_after_id
    if _loop_after_id is not None:
        canvas.after_cancel(_loop_after_id)
        _loop_after_id = None
    zompie_in_campo.clear()
    zompie_array_corrente = []
    zompie_after_id = None
    canvas.delete("zompie")

def _aggiungi_zompie(frames, x_spawn):
    """Aggiunge un singolo zompie alla lista e (ri)avvia il loop."""
    global _zompie_id_counter, zompie_array_corrente, zompie_x, zompie_y, zompie_k
    _zompie_id_counter += 1
    zid = _zompie_id_counter
    x   = max(80, min(x_spawn, CANVAS_W - 80))

    z = {
        'id':         zid,
        'frames':     frames,
        'k':          0,
        'x':          x,
        'y':          CANVAS_H - 100,
        'photo':      None,
        'danno_tick': 0,
    }
    zompie_in_campo.append(z)

    # aggiorna variabili legacy
    zompie_array_corrente = frames
    zompie_x = x
    zompie_y = CANVAS_H - 100
    zompie_k = 0

    _avvia_loop()
    return zid

def avvia_ondata():
    """Lancia la ondata corrente con cap a ZOMPIE_MAX."""
    stoppa_zompie()
    n = min(zompie_per_ondata, ZOMPIE_MAX)
    print(f"🧟 ONDATA {ondata_corrente} — {n} zompie (delay={_get_delay()}ms)")

    step = CANVAS_W // (n + 1)
    for i in range(n):
        pool    = random.choice([arrays_zompie1, arrays_zompie2,
                                 arrays_zompie3, arrays_zompie4,
                                 arrays_zompie5])
        x_spawn = step * (i + 1) + random.randint(-30, 30)
        _aggiungi_zompie(pool, x_spawn)

def avvia_prossima_ondata():
    global ondata_corrente, zompie_per_ondata
    ondata_corrente   += 1
    zompie_per_ondata *= 2
    effettivi = min(zompie_per_ondata, ZOMPIE_MAX)
    print(f"⬆️  Ondata {ondata_corrente} — {effettivi} zompie (max {ZOMPIE_MAX})")
    avvia_ondata()

# ── DISSOLVI un singolo zompie ─────────────────────────────────────────────
def dissolvi_zompie_singolo(zid):
    """Rimuove il zompie dal loop e lo dissolve in background con un after separato."""
    z = next((x for x in zompie_in_campo if x['id'] == zid), None)
    if z is None:
        return

    # Rimuovi subito dalla lista — il loop centrale non lo disegnerà più
    zompie_in_campo[:] = [x for x in zompie_in_campo if x['id'] != zid]

    tag    = f"zompie_{zid}"
    frames = z['frames']
    frame  = frames[z['k'] % len(frames)].copy().convert("RGBA")
    pixels = frame.load()
    w, h   = frame.size
    total  = w * h
    coords = [(cx, cy) for cx in range(w) for cy in range(h)]
    random.shuffle(coords)
    steps  = 20
    chunk  = total // steps
    state  = {'step': 0, 'photo': None, 'x': z['x'], 'y': z['y']}

    def step_dissolvi():
        global zompie_uccisi
        s = state['step']
        if s >= steps:
            canvas.delete(tag)
            zompie_uccisi += 1
            disegna_kill()
            if not zompie_in_campo and _loop_after_id is None:
                print("✅ Ondata eliminata!")
                canvas.after(1500, avvia_prossima_ondata)
            return

        start = s * chunk
        end   = start + chunk if s < steps - 1 else total
        for i in range(start, end):
            px, py = coords[i]
            r, g, b, a = pixels[px, py]
            pixels[px, py] = (r, g, b, 0)

        state['photo'] = ImageTk.PhotoImage(frame)
        canvas.delete(tag)
        canvas.create_image(state['x'], state['y'], anchor="sw",
                            image=state['photo'], tags=tag)   # NO tag "zompie" → il loop non lo tocca
        canvas.tag_raise("pistola")
        state['step'] += 1
        canvas.after(40, step_dissolvi)

    step_dissolvi()

def dissolvi_zompie():
    """Alias legacy — dissolve il primo zompie in campo."""
    if zompie_in_campo:
        dissolvi_zompie_singolo(zompie_in_campo[0]['id'])

# ── RILEVAMENTO COLLISIONI ─────────────────────────────────────────────────
def _trova_zompie_colpito(mx, my):
    for z in zompie_in_campo:
        frames = z['frames']
        if not frames:
            continue
        frame  = frames[z['k'] % len(frames)]
        zw, zh = frame.size
        if z['x'] <= mx <= z['x'] + zw and z['y'] - zh <= my <= z['y']:
            return z['id']
    return None

# ── PISTOLA FPS ────────────────────────────────────────────────────────────
GUN_W, GUN_H = 400, 400

gun_img     = Image.open("gun.png").convert("RGBA")
gunfire_img = Image.open("gunfire.png").convert("RGBA")
gun_img     = gun_img.resize((GUN_W, GUN_H), Image.LANCZOS)
gunfire_img = gunfire_img.resize((GUN_W, GUN_H), Image.LANCZOS)

gun_x        = 512
gun_y        = 1024
sparo_attivo = False
y_mirino     = 80

def aggiorna_pistola(event):
    global gun_x, gun_y
    gun_x = event.x
    gun_y = event.y + y_mirino
    if not sparo_attivo:
        canvas.delete("pistola")
        canvas.create_image(gun_x, gun_y, anchor="center",
                            image=gun_photo, tags="pistola")
        canvas.tag_raise("pistola")

# ── DISSOLVI un singolo zompie ─────────────────────────────────────────────
def dissolvi_zompie_singolo(zid):
    """Rimuove il zompie dal loop e lo dissolve in background con un after separato."""
    z = next((x for x in zompie_in_campo if x['id'] == zid), None)
    if z is None:
        return

    # Rimuovi subito dalla lista — il loop centrale non lo disegnerà più
    zompie_in_campo[:] = [x for x in zompie_in_campo if x['id'] != zid]

    tag    = f"zompie_{zid}"
    frames = z['frames']
    frame  = frames[z['k'] % len(frames)].copy().convert("RGBA")
    pixels = frame.load()
    w, h   = frame.size
    total  = w * h
    coords = [(cx, cy) for cx in range(w) for cy in range(h)]
    random.shuffle(coords)
    steps  = 20
    chunk  = total // steps
    state  = {'step': 0, 'photo': None, 'x': z['x'], 'y': z['y']}

    def step_dissolvi():
        global zompie_uccisi
        s = state['step']
        if s >= steps:
            canvas.delete(tag)
            zompie_uccisi += 1
            disegna_kill()
            if not zompie_in_campo and _loop_after_id is None:
                print("✅ Ondata eliminata!")
                canvas.after(1500, avvia_prossima_ondata)
            return

        start = s * chunk
        end   = start + chunk if s < steps - 1 else total
        for i in range(start, end):
            px, py = coords[i]
            r, g, b, a = pixels[px, py]
            pixels[px, py] = (r, g, b, 0)

        state['photo'] = ImageTk.PhotoImage(frame)
        canvas.delete(tag)
        canvas.create_image(state['x'], state['y'], anchor="sw",
                            image=state['photo'], tags=tag)   # NO tag "zompie" → il loop non lo tocca
        canvas.tag_raise("pistola")
        state['step'] += 1
        canvas.after(40, step_dissolvi)

    step_dissolvi()

def dissolvi_zompie():
    """Alias legacy — dissolve il primo zompie in campo."""
    if zompie_in_campo:
        dissolvi_zompie_singolo(zompie_in_campo[0]['id'])

# ── RILEVAMENTO COLLISIONI ─────────────────────────────────────────────────
def _trova_zompie_colpito(mx, my):
    for z in zompie_in_campo:
        frames = z['frames']
        if not frames:
            continue
        frame  = frames[z['k'] % len(frames)]
        zw, zh = frame.size
        if z['x'] <= mx <= z['x'] + zw and z['y'] - zh <= my <= z['y']:
            return z['id']
    return None

sound_sparo = pygame.mixer.Sound("sparo.wav")

def spara(event):
    global sparo_attivo
    if sparo_attivo:
        return

    zid = _trova_zompie_colpito(event.x, event.y)
    if zid is not None:
        print(f"💥 ZOMPIE {zid} COLPITO!")
        dissolvi_zompie_singolo(zid)

    sound_sparo.play()

    sparo_attivo = True
    canvas.delete("pistola")
    canvas.create_image(gun_x, gun_y, anchor="center",
                        image=gunfire_photo, tags="pistola")
    canvas.tag_raise("pistola")

    def fine_sparo():
        global sparo_attivo
        sparo_attivo = False
        canvas.delete("pistola")
        canvas.create_image(gun_x, gun_y, anchor="center",
                            image=gun_photo, tags="pistola")
        canvas.tag_raise("pistola")

    canvas.after(80, fine_sparo)

import os
import time
from PIL import Image, ImageTk # Assicurati di avere Pillow installato

# --- CARICAMENTO DATI INIZIALI ---
if os.path.exists("win.txt"):
    with open("win.txt", "r") as f:
        try:
            zompie_winner = int(f.read().strip())
        except:
            zompie_winner = 10
else:
    zompie_winner = 10

if os.path.exists("l.txt"):
    with open("l.txt", "r") as f:
        try:
            livel = int(f.read().strip())
        except:
            livel = 1
else:
    livel = 1


def on_canvas_click(event):
    global vita, rianima, time_dead, gioco_bloccato,zompie_winner,livel
    mx, my = event.x, event.y
    stanza = matrix_casa[pos_riga][pos_col]

    def prova_sparo_zompie():
        zid = _trova_zompie_colpito(mx, my)
        if zid is not None:
            print("💥 ZOMPIE COLPITO!")
            dissolvi_zompie_singolo(zid)
            spara(event)
            return True
        return False

    # ── CAMERA OSCURA ─────────────────────────────────────────────────────
    if stanza == "camera_oscura":
        if prova_sparo_zompie():
            return
        key_map_oscura = {
            'green':   (select_keygreen,   key_green_var),
            'yellow':  (select_keyyellow,  key_yellow_var),
            'blue':    (select_keyblue,    key_blue_var),
            'magenta': (select_keymagenta, key_magenta_var),
        }
        for color, (select_stanza, var) in key_map_oscura.items():
            if stanza != select_stanza or var.get():
                continue
            if color not in key_positions or key_positions[color] is None:
                continue
            kx, ky = key_positions[color]
            if abs(mx - kx) <= 20 and abs(my - ky) <= 20:
                var.set(1)
                key_positions[color] = None
                aggiorna_view()
                return
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        if abs(mx - cx) <= 200 and abs(my - cy) <= 200:
            if vita <= 0 and rianima > 0 and time_dead > 0:
                secondi_passati = time.time() - time_dead
                if secondi_passati <= 10:
                    vita = 100
                    rianima -= 1
                    time_dead = 0
                    print(f"✨ Rianimato! Vita: {vita}%  Rianima rimasti: {rianima}")
                    disegna_vita()
                else:
                    print(f"⏰ Troppo tardi! Passati {secondi_passati:.1f}s (max 10s).")
            elif vita > 0:
                print("ℹ️ Sei ancora vivo, non serve rianimare.")
            elif rianima <= 0:
                print("💀 Nessuna rianimazione disponibile.")
        return

    # ── BALCONE ───────────────────────────────────────────────────────────
    if stanza == "balcone" and hasattr(canvas, 'girls_boxes'):
        if prova_sparo_zompie():
            return
        for i, (x1, y1, x2, y2) in enumerate(canvas.girls_boxes):
            if x1 <= mx <= x2 and y1 <= my <= y2:
                mostra_menu_ragazza(i, event.x_root, event.y_root)
                return

    # ── CHIAVI ────────────────────────────────────────────────────────────
    key_map = {
        'green':   (select_keygreen,   key_green_var),
        'yellow':  (select_keyyellow,  key_yellow_var),
        'blue':    (select_keyblue,    key_blue_var),
        'magenta': (select_keymagenta, key_magenta_var),
    }
    for color, (select_stanza, var) in key_map.items():
        if stanza != select_stanza or var.get():
            continue
        if color not in key_positions or key_positions[color] is None:
            continue
        kx, ky = key_positions[color]
        if abs(mx - kx) <= 20 and abs(my - ky) <= 20:
            if prova_sparo_zompie():
                return
            var.set(1)
            key_positions[color] = None
            aggiorna_view()
            return

    # ── SERRATURE ─────────────────────────────────────────────────────────
    serrature = {
        'green':   (key_green_var,   "corridoio_sinistro"),
        'yellow':  (key_yellow_var,  "corridoio_destro"),
        'blue':    (key_blue_var,    "anta_destra"),
        'magenta': (key_magenta_var, "anta_sinistra"),
    }
    video_map = {
        'green':   os.path.join(BASE_DIR, "videos", "verde.mp4"),
        'yellow':  os.path.join(BASE_DIR, "videos", "gialla.mp4"),
        'blue':    os.path.join(BASE_DIR, "videos", "blue.mp4"),
        'magenta': os.path.join(BASE_DIR, "videos", "magenta.mp4"),
    }
    for color, (var, stanza_serratura) in serrature.items():
        if var.get() and stanza == stanza_serratura:
            if prova_sparo_zompie():
                return
            if color not in serrature_aperte:
                serrature_aperte.add(color)
                if color in video_map:
                    gioco_bloccato = True
                    os.startfile(video_map[color])
            return

 
    # ── PORTONE: VITTORIA ─────────────────────────────────────────────────
    if stanza == "portone":
        if prova_sparo_zompie():
            return
            
        if (key_green_var.get() and key_yellow_var.get() and
                key_blue_var.get() and key_magenta_var.get()):
            
            if zompie_uccisi >= zompie_winner:
                print("Apertura portone - HAI VINTO!")
                gioco_bloccato = True
                
                # Video finale
                video_path = os.path.join(BASE_DIR, "videos", "f.mp4")
                if os.path.exists(video_path):
                    os.startfile(video_path)

                # Aggiorna statistiche
                livel += 1
                zompie_winner = zompie_winner * livel

                # Salvataggio su file
                with open("win.txt", "w") as f:
                    f.write(str(zompie_winner))
                with open("l.txt", "w") as f:
                    f.write(str(livel))

                # --- Visualizzazione Vittoria su Canvas ---
                w = window.winfo_width()
                h = window.winfo_height()
                canvas.delete("all") 
                canvas.create_rectangle(0, 0, w, h, fill="white")
                
                # Usiamo x=10 invece di 3 per dare un minimo di respiro dal bordo fisico
                # anchor="w" fa sì che il testo inizi dal punto X indicato
                canvas.create_text(10, h//2 - 50, 
                                   text="WINNER", 
                                   fill="green", 
                                   font=("Arial", 60, "bold"),
                                   anchor="w") 
                
                canvas.create_text(10, h//2 + 50, 
                                   text=f"NEXT LEVEL: {livel}", 
                                   fill="darkgreen", 
                                   font=("Arial", 40, "bold"),
                                   anchor="w")
                
                window.update()
                time.sleep(3) 

                # --- PULIZIA ASSETS PER NUOVO LIVELLO ---
                def svuota_cartella(cartella):
                    if os.path.exists(cartella):
                        for file in os.listdir(cartella):
                            path_file = os.path.join(cartella, file)
                            try:
                                if os.path.isfile(path_file):
                                    os.remove(path_file)
                            except Exception as e:
                                print(f"Errore eliminazione {path_file}: {e}")

                svuota_cartella("location")
                svuota_cartella("girls")
                svuota_cartella("image_rigenera")
                
                # Resetta eventuali stati globali prima del restart
                serrature_aperte.clear()
                
                # Riavvio script
                print("Rigenerazione mondo in corso...")
                os.system("python keyroom.py")
                window.destroy()
            else:
                mancano = zompie_winner - zompie_uccisi
                print(f"🚫 Portone bloccato! Devi eliminare ancora {mancano} zompie!")
        else:
            print("🔑 Ti mancano ancora delle chiavi!")
        return

    spara(event)


# ── AGGIORNA VIEW ──────────────────────────────────────────────────────────
def aggiorna_view():
    global current_image, keygreen, keyyellow, keyblue, keymagenta

    stoppa_zompie()

    stanza = matrix_casa[pos_riga][pos_col]
    label_stanza.config(text=stanza.replace("_", " ").upper())
    label_pos.config(text=f"Posizione: [{pos_riga}][{pos_col}]")

    try:
        img = Image.open(f"{dir_location}/{stanza}.png")
        img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        current_image = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, anchor="center", image=current_image)

        if stanza != "balcone":
            print(f"lo zompie si trova nella stanza: {stanza}")
            if random.randint(1, 5) == 1:
                avvia_ondata()   # ← era avvia_zompie()
        else:
            stoppa_zompie()

        # ── BALCONE ───────────────────────────────────────────────────────
        if stanza == "balcone":
            trsp_files = sorted([
                f for f in os.listdir("girls")
                if f.endswith("_trsp.png")
            ])
            if trsp_files:
                if not hasattr(canvas, 'trsp_images'):
                    canvas.trsp_images = []
                canvas.trsp_images.clear()
                if not hasattr(canvas, 'girls_boxes'):
                    canvas.girls_boxes = []
                canvas.girls_boxes.clear()

                target_h = int(CANVAS_H * 0.90)
                resized_imgs = []
                total_w = 0
                for trsp_file in trsp_files:
                    try:
                        trsp_img = Image.open(f"girls/{trsp_file}").convert("RGBA")
                        bbox = trsp_img.getbbox()
                        if bbox:
                            trsp_img = trsp_img.crop(bbox)
                        orig_w, orig_h = trsp_img.size
                        scale = target_h / orig_h
                        new_w = int(orig_w * scale)
                        trsp_img = trsp_img.resize((new_w, target_h), Image.LANCZOS)
                        resized_imgs.append(trsp_img)
                        total_w += new_w
                    except Exception as e:
                        print(f"Errore caricamento {trsp_file}: {e}")
                        resized_imgs.append(None)

                if total_w > CANVAS_W:
                    scale_down = CANVAS_W / total_w
                    scaled_imgs = []
                    total_w = 0
                    for img2 in resized_imgs:
                        if img2:
                            nw = int(img2.width * scale_down)
                            nh = int(img2.height * scale_down)
                            img2 = img2.resize((nw, nh), Image.LANCZOS)
                            total_w += nw
                        scaled_imgs.append(img2)
                    resized_imgs = scaled_imgs

                start_x = (CANVAS_W - total_w) // 2
                x_pos = start_x
                for img2 in resized_imgs:
                    if img2 is None:
                        continue
                    trsp_photo = ImageTk.PhotoImage(img2)
                    canvas.trsp_images.append(trsp_photo)
                    y_pos = CANVAS_H - img2.size[1] // 4
                    canvas.create_image(x_pos, y_pos, anchor="sw", image=trsp_photo)
                    canvas.girls_boxes.append((
                        x_pos,
                        y_pos - img2.size[1],
                        x_pos + img2.width,
                        y_pos
                    ))
                    x_pos += img2.width

        # ── KEYS ──────────────────────────────────────────────────────────
        key_map = {
            'green':   (select_keygreen,   keygreen,   key_green_var),
            'yellow':  (select_keyyellow,  keyyellow,  key_yellow_var),
            'blue':    (select_keyblue,    keyblue,    key_blue_var),
            'magenta': (select_keymagenta, keymagenta, key_magenta_var),
        }
        for color, (select_stanza, kimg, var) in key_map.items():
            if stanza == select_stanza and not var.get():
                if color not in key_positions or key_positions[color] is None:
                    key_positions[color] = (
                        random.randint(64, CANVAS_W - 64),
                        random.randint(64, CANVAS_H - 64)
                    )
                kx, ky = key_positions[color]
                canvas.create_image(kx, ky, anchor="center", image=kimg, tags=f"key_{color}")
                print(f"Key {color} posizionata in [{kx},{ky}]")

        # ── TESTI ─────────────────────────────────────────────────────────
        canvas.create_text(
            CANVAS_W // 2, CANVAS_H - 60,
            text=stanza.replace("_", " ").upper(),
            font=("Courier", 22, "bold"), fill="#ff3333"
        )
        canvas.create_text(
            CANVAS_W // 2, CANVAS_H - 30,
            text=f"[{pos_riga}][{pos_col}]",
            font=("Courier", 11), fill="#888888"
        )

        canvas.delete("pistola")
        canvas.create_image(gun_x, gun_y, anchor="center",
                            image=gun_photo, tags="pistola")
        canvas.tag_raise("pistola")

    except FileNotFoundError:
        stoppa_zompie()
        canvas.delete("all")
        canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill="#0d0d0d")
        canvas.create_text(CANVAS_W // 2, CANVAS_H // 2 - 30,
                           text="[ IMMAGINE NON TROVATA ]",
                           font=("Courier", 16), fill="#ff0000")
        canvas.create_text(CANVAS_W // 2, CANVAS_H // 2 + 10,
                           text=f"{dir_location}/{stanza}.png",
                           font=("Courier", 11), fill="#555555")
        canvas.create_text(CANVAS_W // 2, CANVAS_H - 40,
                           text=stanza.replace("_", " ").upper(),
                           font=("Courier", 22, "bold"), fill="#ff3333")

# ── NAVIGAZIONE ────────────────────────────────────────────────────────────
def vai_su():
    global pos_riga
    if pos_riga > 0:
        pos_riga -= 1
        aggiorna_view()

def vai_giu():
    global pos_riga
    if pos_riga < len(matrix_casa) - 1:
        pos_riga += 1
        aggiorna_view()

def vai_sinistra():
    global pos_col
    if pos_col > 0:
        pos_col -= 1
        aggiorna_view()

def vai_destra():
    global pos_col
    if pos_col < len(matrix_casa[0]) - 1:
        pos_col += 1
        aggiorna_view()

def key_press(event):
    global gioco_bloccato
    if gioco_bloccato:
        gioco_bloccato = False
        print("▶️ Gioco ripreso!")
        return
    if event.keysym == "Up":    vai_su()
    if event.keysym == "Down":  vai_giu()
    if event.keysym == "Left":  vai_sinistra()
    if event.keysym == "Right": vai_destra()

pygame.mixer.music.load("music.mp3")
pygame.mixer.music.set_volume(0.4)   # ← volume basso per non coprire gli spari
pygame.mixer.music.play(-1)          # ← -1 = loop infinito

window = tk.Tk()
window.title("Keys House")

# Questo attiva il vero full screen (senza bordi né barra del titolo)
window.attributes('-fullscreen', True)

# Se vuoi permettere all'utente di uscire dal full screen con il tasto ESC
window.bind("<Escape>", lambda event: window.attributes("-fullscreen", False))

window.configure(bg="#1a1a1a")
window.bind("<KeyPress>", key_press)

# ── ORA si possono creare i PhotoImage ────────────────────────────────────
gun_photo     = ImageTk.PhotoImage(gun_img)
gunfire_photo = ImageTk.PhotoImage(gunfire_img)

CANVAS_W = 1024
CANVAS_H = 1024

keygreen   = resize_key(key_files['green'])
keyyellow  = resize_key(key_files['yellow'])
keyblue    = resize_key(key_files['blue'])
keymagenta = resize_key(key_files['magenta'])

key_green_var   = tk.BooleanVar(value=False)
key_yellow_var  = tk.BooleanVar(value=False)
key_blue_var    = tk.BooleanVar(value=False)
key_magenta_var = tk.BooleanVar(value=False)

print(f"Key green:   {select_keygreen}")
print(f"Key yellow:  {select_keyyellow}")
print(f"Key blue:    {select_keyblue}")
print(f"Key magenta: {select_keymagenta}")

carica_frecce()

window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=3)
window.columnconfigure(2, weight=1)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=5)
window.rowconfigure(2, weight=1)

frame_top = tk.Frame(window, bg="#1a1a1a")
frame_top.grid(row=0, column=1, pady=10)

VITA_W = 300
VITA_H = 24
canvas_vita = tk.Canvas(frame_top, width=VITA_W, height=VITA_H,
                         bg="#1a1a1a", highlightthickness=0)
canvas_vita.grid(row=0, column=0, padx=10)

_timer_id = None

def controlla_morte():
    global time_dead, _timer_id
    if vita <= 0 and time_dead == 0:
        time_dead = time.time()
        print("💀 Personaggio morto! Hai 10 secondi per rianimare.")
        aggiorna_barra_morte()

def aggiorna_barra_morte():
    global _timer_id
    if vita > 0 or time_dead == 0:
        return
    secondi_passati = time.time() - time_dead
    frazione = min(secondi_passati / 10.0, 1.0)
    canvas_vita.delete("all")
    canvas_vita.create_rectangle(0, 0, VITA_W, VITA_H, outline="#ff0000", width=2, fill="#1a1a1a")
    fill_w = int(frazione * (VITA_W - 4))
    if fill_w > 0:
        canvas_vita.create_rectangle(2, 2, 2 + fill_w, VITA_H - 2, fill="#222222", outline="")
    rimanenti = max(0, 10 - int(secondi_passati))
    canvas_vita.create_text(VITA_W // 2, VITA_H // 2,
                             text=f"💀 RIANIMA! {rimanenti}s",
                             font=("Courier", 11, "bold"), fill="#ff4444")
    if frazione >= 1.0:
        game_over()
    else:
        _timer_id = canvas_vita.after(100, aggiorna_barra_morte)

def game_over():
    global _timer_id
    if _timer_id:
        canvas_vita.after_cancel(_timer_id)
        _timer_id = None
    canvas_vita.delete("all")
    canvas_vita.create_rectangle(0, 0, VITA_W, VITA_H, outline="#ff0000", width=2, fill="#0d0d0d")
    canvas_vita.create_text(VITA_W // 2, VITA_H // 2,
                             text="💀  GAME  OVER  💀",
                             font=("Courier", 11, "bold"), fill="#ff0000")
    window.after(2000, window.destroy)

zompie_uccisi = 0

# ── questi due canvas vanno creati UNA VOLTA nel layout ──────────────────
VITA_W, VITA_H = 300, 24
canvas_vita = tk.Canvas(frame_top, width=VITA_W, height=VITA_H,
                        bg="#1a1a1a", highlightthickness=0)
canvas_vita.grid(row=0, column=0, padx=10)

KILL_W, KILL_H = 120, 24
canvas_kill = tk.Canvas(frame_top, width=KILL_W, height=KILL_H,
                        bg="#1a1a1a", highlightthickness=0)
canvas_kill.grid(row=0, column=2, padx=10)

def disegna_kill():
    global zompie_winner, livel, zompie_uccisi
    
    canvas_kill.delete("all")
    
    # Sfondo e bordo
    canvas_kill.create_rectangle(0, 0, KILL_W, KILL_H,
                                 outline="#ff4444", width=2, fill="#1a1a1a")
    
    # Punto centrale del widget
    centro_x = KILL_W // 2
    centro_y = KILL_H // 2

    # 1. Parte sinistra: Kills (ancorata a destra del centro)
    canvas_kill.create_text(centro_x - 5, centro_y,
                            text=f"💀 {zompie_uccisi}/{zompie_winner}",
                            font=("Courier", 11, "bold"), 
                            fill="#ff4444",
                            anchor="e") # 'e' sta per East (ancora a destra)

    # 2. Parte destra: Livello (ancorata a sinistra del centro)
    canvas_kill.create_text(centro_x + 5, centro_y,
                            text=f" LV: {livel}",
                            font=("Courier", 11, "bold"), 
                            fill="#00ff00",
                            anchor="w") # 'w' sta per West (ancora a sinistra)

def disegna_vita():
    global _timer_id
    if _timer_id and vita > 0:
        canvas_vita.after_cancel(_timer_id)
        _timer_id = None
    if vita > 0:
        canvas_vita.delete("all")
        fill_w = int((vita / 100) * (VITA_W - 4))
        canvas_vita.create_rectangle(0, 0, VITA_W, VITA_H, outline="#00bfff", width=2, fill="#1a1a1a")
        if fill_w > 0:
            canvas_vita.create_rectangle(2, 2, 2 + fill_w, VITA_H - 2, fill="#ff2222", outline="")
        canvas_vita.create_text(VITA_W // 2, VITA_H // 2,
                                text=f"❤ {vita}%",
                                font=("Courier", 11, "bold"), fill="white")
    controlla_morte()

btn_su = tk.Button(
    frame_top,
    image=img_su if img_su else None,
    text="" if img_su else "▲  SU",
    command=vai_su,
    bg="#1a1a1a", relief="flat", cursor="hand2",
    borderwidth=0, activebackground="#2a2a2a"
)
btn_su.grid(row=0, column=1, padx=10)

disegna_vita()
disegna_kill()

canvas = tk.Canvas(
    window, width=CANVAS_W, height=CANVAS_H,
    bg="#0d0d0d", highlightthickness=2, highlightbackground="#ff0000"
)
canvas.grid(row=1, column=1, padx=20, pady=10)
canvas.bind("<Button-1>", on_canvas_click)   # click sinistro → azioni + sparo
canvas.bind("<Motion>",   aggiorna_pistola)  # movimento mouse → muovi pistola
canvas.config(cursor="none") 

label_stanza = tk.Label(window, text="", font=("Courier", 1), fg="#1a1a1a", bg="#1a1a1a")
label_stanza.grid(row=3, column=1)
label_pos = tk.Label(window, text="", font=("Courier", 1), fg="#1a1a1a", bg="#1a1a1a")
label_pos.grid(row=4, column=1)

btn_sinistra = tk.Button(
    window,
    image=img_sinistra if img_sinistra else None,
    text="" if img_sinistra else "◀  SX",
    command=vai_sinistra,
    bg="#1a1a1a", relief="flat", cursor="hand2",
    borderwidth=0, activebackground="#2a2a2a"
)
btn_sinistra.grid(row=1, column=0, padx=10)

btn_destra = tk.Button(
    window,
    image=img_destra if img_destra else None,
    text="" if img_destra else "DX  ▶",
    command=vai_destra,
    bg="#1a1a1a", relief="flat", cursor="hand2",
    borderwidth=0, activebackground="#2a2a2a"
)
btn_destra.grid(row=1, column=2, padx=10)

frame_bottom = tk.Frame(window, bg="#1a1a1a")
frame_bottom.grid(row=2, column=1, pady=10)

btn_giu = tk.Button(
    frame_bottom,
    image=img_giu if img_giu else None,
    text="" if img_giu else "▼  GIU",
    command=vai_giu,
    bg="#1a1a1a", relief="flat", cursor="hand2",
    borderwidth=0, activebackground="#2a2a2a"
)
btn_giu.pack(side="left", padx=10)

tk.Checkbutton(frame_bottom, text="🗝 Green",
               bg='green', fg='white', selectcolor='#004400',
               variable=key_green_var, state="disabled").pack(side="left", padx=8)

tk.Checkbutton(frame_bottom, text="🗝 Yellow",
               bg='yellow', fg='black', selectcolor='#888800',
               variable=key_yellow_var, state="disabled").pack(side="left", padx=8)

tk.Checkbutton(frame_bottom, text="🗝 Blue",
               bg='blue', fg='white', selectcolor='#000088',
               variable=key_blue_var, state="disabled").pack(side="left", padx=8)

tk.Checkbutton(frame_bottom, text="🗝 Magenta",
               bg='magenta', fg='white', selectcolor='#880088',
               variable=key_magenta_var, state="disabled").pack(side="left", padx=8)

# Avvio
aggiorna_view()
window.mainloop()