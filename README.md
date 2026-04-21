# SRT Français ➜ Doublage audio français dans une vidéo (Kokoro TTS + GPU)

Ce dépôt contient un script Python pour transformer des sous-titres `.srt` en audio français (TTS) puis injecter cet audio dans une vidéo `.mp4`.

Le script principal est:

- `srt_to_french_audio_kokoro.py`

---

## 1) Objectif

Automatiser un pipeline simple:

1. Lire un fichier SRT en français.
2. Générer un segment audio par sous-titre avec **Kokoro TTS**.
3. Mélanger les segments dans une seule piste WAV 24 kHz.
4. Remplacer la piste audio de la vidéo via **FFmpeg**.

Le script supporte:

- **GPU NVIDIA (CUDA)** quand disponible.
- **Fallback CPU** si aucun GPU n'est détecté.
- Sous-titres en `utf-8` avec fallback `latin-1`.

---

## 2) Prérequis

## Python

- Python 3.10+ recommandé.

## Outils système

- `ffmpeg` et `ffprobe` accessibles dans le `PATH`.
- Sous Windows: installer `espeak-ng-X64.msi` (requis par l'écosystème Kokoro / misaki).

## Dépendances Python

### Option GPU NVIDIA (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "kokoro>=0.9.4" soundfile pysrt "misaki[fr]" numpy
```

### Option CPU uniquement

```bash
pip install torch torchvision torchaudio
pip install "kokoro>=0.9.4" soundfile pysrt "misaki[fr]" numpy
```

---

## 3) Utilisation rapide

```bash
python srt_to_french_audio_kokoro.py \
  --video "C:\\Users\\Mohamed\\Desktop\\formation packt\\test.mp4" \
  --srt "C:\\Users\\Mohamed\\Desktop\\formation packt\\subtitles_fr.srt" \
  --out "C:\\Users\\Mohamed\\Desktop\\formation packt\\test_audio_fr.mp4" \
  --voix "ff_siwis" \
  --vitesse 1.0
```

### Paramètres CLI

- `--video` : chemin de la vidéo source (`.mp4` conseillé).
- `--srt` : chemin du fichier sous-titres français.
- `--out` : chemin du fichier vidéo de sortie.
- `--voix` : voix Kokoro (défaut: `ff_siwis`).
- `--vitesse` : vitesse de parole (float, défaut: `1.0`).

---

## 4) Comment le pipeline fonctionne

## Étape A — Lecture et nettoyage SRT

- Le script lit le SRT en `utf-8`, sinon tente `latin-1`.
- Il supprime les balises (`<i>`, etc.) et remplace les retours ligne par des espaces.
- Chaque sous-titre valide devient un segment: texte + timestamp de début + fichier WAV temporaire.

## Étape B — Synthèse TTS Kokoro

- Le script crée `KPipeline(lang_code="f")` pour le français.
- Pour chaque segment, Kokoro génère un flux audio, concaténé puis sauvegardé en WAV 24 kHz.
- En cas d'erreur sur un segment, le script écrit un court silence (robustesse globale).

## Étape C — Mixage audio global

- Une piste `numpy` est allouée selon la durée de la vidéo.
- Chaque segment est inséré à son temps de départ en échantillons.
- Les chevauchements sont additionnés (mix), puis la piste est normalisée pour éviter la saturation.

## Étape D — Mux vidéo final

- `ffmpeg` copie la vidéo (`-c:v copy`) et encode la nouvelle piste audio en AAC (`-c:a aac`).
- Le résultat est écrit dans `--out`.

---

## 5) Performance GPU (NVIDIA)

- Si `torch.cuda.is_available()` renvoie `True`, Kokoro tourne sur GPU.
- Le gain dépend de:
  - la taille du SRT,
  - la longueur des segments,
  - la VRAM disponible,
  - les performances CPU/disque (écriture WAV temporaires).

Pour vérifier rapidement CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 6) Dépannage

## `ffmpeg` introuvable

- Installer FFmpeg puis ajouter son dossier `bin` au `PATH`.
- Vérifier:

```bash
ffmpeg -version
ffprobe -version
```

## Erreur Kokoro / voix non trouvée

- Vérifier la version installée de `kokoro`.
- Tester avec la voix par défaut `ff_siwis`.

## Sous Windows: erreurs audio / event loop

- Le script applique déjà `WindowsSelectorEventLoopPolicy`.
- Vérifier aussi l'installation de `espeak-ng-X64.msi`.

## Audio trop fort / saturation

- Le script normalise déjà la piste finale.
- Vous pouvez réduire `--vitesse` ou éditer le script pour appliquer un gain global supplémentaire.

---

## 7) Limitations connues

- Le script remplace la piste audio par la voix synthétique (pas de mix avec l'audio d'origine).
- Le timing repose sur l'heure de début des sous-titres (pas de time-stretch automatique).
- Pas de batch natif multi-fichiers (1 vidéo / 1 SRT par exécution).

---

## 8) Idées d'amélioration

- Ajouter un mode "mix original + voix TTS" avec contrôle de volume.
- Ajouter un mode batch (dossier d'entrée/sortie).
- Ajouter des tests unitaires (nettoyage texte, parsing SRT, mixage).
- Ajouter un `requirements.txt` / `pyproject.toml` pour l'installation reproductible.

---

## 9) Fichier principal

- `srt_to_french_audio_kokoro.py`: pipeline complet SRT ➜ TTS ➜ mix ➜ mux.
