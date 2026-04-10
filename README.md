🗝️ KeysHouse: AI Survival Game
KeysHouse è un'esperienza di gioco innovativa basata su Intelligenza Artificiale, alimentata dal modello Flux.1 [dev]. Immergiti in una casa generata proceduralmente dall'IA, dove il tuo obiettivo è trovare le chiavi per fuggire mentre ti difendi da orde di zombie.

Tutte le ambientazioni (location) e gli sprite di gioco vengono generati dinamicamente dall'IA per offrire un'esperienza visiva ogni volta unica.

🚀 Requisiti e Installazione
Segui questi passaggi per configurare correttamente l'ambiente di gioco. Si consiglia l'uso di una GPU NVIDIA con supporto CUDA 12.8.

1. Clonazione del Repository
Apri il terminale e clona il progetto:

Bash
git clone https://github.com/asprho-arkimete/keyshouse.git
cd keyshouse
2. Estrazione delle Risorse Video
Individua il file videos.part1.exe (o i file divisi) nella cartella principale ed eseguilo per estrarre i contenuti video necessari.

3. Setup dei Frame Zombie
Scarica l'archivio zombie dalla sezione Releases del repository. Estrai il contenuto per ottenere la cartella frames_zombie e posizionala nella root del progetto.

4. Download Modelli e LoRA
Scarica i modelli LoRA necessari seguendo i link contenuti nel file lora_links.txt.

Al primo avvio, il gioco scaricherà automaticamente il modello Flux.1 Dev (9B) se non presente.

5. Configurazione Ambiente Python
Crea e attiva un ambiente virtuale per gestire le dipendenze:

Bash
python -m venv vroom
# Su Windows:
vroom\Scripts\activate
# Su Linux/Mac:
source vroom/bin/activate
6. Installazione Dipendenze
Installa PyTorch con supporto CUDA e tutte le librerie necessarie:

Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
🎨 Personalizzazione (Bonus)
Prima di iniziare, puoi personalizzare i volti dei personaggi (bonus) nel gioco:

Scegli le foto delle tue modelle o attrici preferite.

Inseriscile nella cartella riferimenti_girls.
Nota: La cartella è vuota per motivi di privacy, permettendoti di creare la tua esperienza personalizzata.

🕹️ Come Giocare
Lo scopo è semplice ma impegnativo:

Esplora: Muoviti nelle stanze generate dall'IA Flux.

Trova: Recupera le 4 chiavi nascoste per sbloccare l'uscita.

Sopravvivi: Difenditi dagli attacchi degli zombie durante la ricerca.

Avvio del Gioco
Per iniziare la tua avventura, esegui:

Bash
python keyroom.py

 
