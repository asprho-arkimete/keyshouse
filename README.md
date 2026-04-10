🗝️ KeysHouse: AI Survival Horror
KeysHouse è un survival game sperimentale che sfrutta la potenza di FLUX.2 [klein] 9B per generare l'intera esperienza visiva. Lo scopo del gioco è esplorare una casa generata proceduralmente dall'IA, trovare le 4 chiavi nascoste e fuggire prima di essere sopraffatti dagli zombie.

Tutte le immagini delle location e gli sprite di gioco vengono renderizzati in tempo reale dal modello FLUX, rendendo ogni partita visivamente unica.

🛠️ Procedura di Installazione
Segui attentamente questi passaggi per configurare l'ambiente di gioco (richiede GPU NVIDIA con supporto CUDA).

1. Clonazione del Progetto
Bash
git clone https://github.com/asprho-arkimete/keyshouse.git
cd keyshouse
2. Estrazione Risorse Video
Estrai il file videos.part1.exe presente nella cartella principale per generare i contenuti video necessari al gioco.

3. Setup Asset Zombie
Scarica l'archivio zombie dalla sezione Releases e posiziona la cartella estratta frames_zombie nella directory principale del progetto.

4. Modelli LoRA
Scarica i modelli LoRA necessari utilizzando i link forniti nel file lora_links.txt.

5. Creazione Ambiente Virtuale
Bash
python -m venv vroom
# Attivazione:
vroom\Scripts\activate
6. Installazione Dipendenze e PyTorch
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
🎭 Personalizzazione e Bonus
Prima di avviare il gioco, puoi personalizzare i personaggi femminili (sistema bonus):

Scegli le foto delle tue attrici o modelle preferite.

Inseriscile nella cartella riferimenti_girls.
Nota: La cartella originale è vuota per motivi di privacy; spetta all'utente popolarla per attivare le generazioni personalizzate.

🕹️ Gameplay
Modello IA: Al primo avvio verrà inizializzato FLUX.2 [klein] 9B per la creazione delle ambientazioni.

Obiettivo: Trova le 4 chiavi nella stanza generata dall'IA.

Sopravvivenza: Difenditi dagli attacchi zombie mentre cerchi la via d'uscita.

Avvio del Gioco
Bash
python keyroom.py

 
