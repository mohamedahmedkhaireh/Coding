"""SRT Français -> Audio Français dans la vidéo (GPU NVIDIA).

Moteur : Kokoro TTS (82M paramètres, rapide sur GPU)

Installation (une seule fois):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install "kokoro>=0.9.4" soundfile pysrt "misaki[fr]"
  + installer espeak-ng-X64.msi (Windows)
Ce script:
1) lit un fichier SRT en français,
2) génère un WAV par sous-titre avec Kokoro TTS,
3) mélange les WAV à leurs positions temporelles d'origine,
4) remplace la piste audio de la vidéo par la piste générée.

Exemple d'utilisation:
    python srt_to_french_audio_kokoro.py \
        --video "C:\\videos\\test.mp4" \
        --srt "C:\\videos\\subtitles_fr.srt" \
        --out "C:\\videos\\test_audio_fr.mp4" \
        --voix "ff_siwis" \
        --vitesse 1.0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import TypedDict

import numpy as np
import pysrt
import soundfile as sf
import torch

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class Segment(TypedDict):
    """Représente un segment audio à synthétiser depuis un sous-titre."""

    texte: str
    debut: float
    fichier: str


def nettoyer_texte(texte: str) -> str:
    """Nettoie le texte d'un sous-titre.

    - supprime les balises HTML/ASS simples (<...>)
    - remplace les sauts de ligne par des espaces
    - retire les espaces superflus en début/fin
    """
    texte = re.sub(r"<[^>]+>", "", texte)
    return texte.replace("\n", " ").strip()


def srt_vers_secondes(t) -> float:
    """Convertit un timestamp pysrt.SubRipTime en secondes flottantes."""
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0


def obtenir_duree_video(chemin_video: str) -> float:
    """Retourne la durée de la vidéo (en secondes) via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        chemin_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def assembler_audio_avec_liste(segments: list[Segment], duree: float, tmp: str) -> str:
    """Assemble la piste audio via un fichier WAV mixé segment par segment.

    Évite la limite Windows de longueur de commande (WinError 206).
    """
    print("   Méthode : mix progressif segment par segment...")

    # Fréquence native des sorties Kokoro.
    sample_rate = 24000
    total_samples = int((duree + 5) * sample_rate)
    piste = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        fichier = seg["fichier"]
        if not os.path.exists(fichier) or os.path.getsize(fichier) < 200:
            continue

        try:
            audio, sr = sf.read(fichier, dtype="float32")
            if sr != sample_rate:
                # Kokoro retourne normalement 24kHz; sécurité minimale.
                pass

            # Convertir stéréo -> mono si besoin
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            debut_sample = int(seg["debut"] * sample_rate)
            fin_sample = debut_sample + len(audio)
            if fin_sample > len(piste):
                fin_sample = len(piste)
                audio = audio[: fin_sample - debut_sample]

            piste[debut_sample:fin_sample] += audio
        except Exception:
            continue

    # Normaliser pour éviter saturation
    max_val = np.max(np.abs(piste))
    if max_val > 0.95:
        piste = piste * (0.95 / max_val)

    # Sauvegarder la piste finale en WAV
    piste_wav = os.path.join(tmp, "piste_fr.wav")
    sf.write(piste_wav, piste, sample_rate)
    print(f"   -> Piste audio assemblée ({len(segments)} segments).")
    return piste_wav


def generer_segments_depuis_srt(chemin_srt: str, tmp: str) -> list[Segment]:
    """Charge un SRT et retourne la liste des segments à générer.

    Le fichier est lu en UTF-8, avec fallback latin-1 pour les SRT hérités.
    """
    try:
        subs = pysrt.open(chemin_srt, encoding="utf-8")
    except Exception:
        subs = pysrt.open(chemin_srt, encoding="latin-1")

    segments: list[Segment] = []
    for i, sub in enumerate(subs):
        texte = nettoyer_texte(sub.text)
        if texte:
            segments.append(
                {
                    "texte": texte,
                    "debut": srt_vers_secondes(sub.start),
                    "fichier": os.path.join(tmp, f"seg_{i:04d}.wav"),
                }
            )
    return segments


def main() -> None:
    """Point d'entrée CLI.

    Étapes:
    0. Détection du device + chargement du pipeline Kokoro
    1. Lecture/normalisation des sous-titres
    2. Synthèse de chaque segment
    3. Mixage de tous les segments et mux final dans le MP4
    """
    parser = argparse.ArgumentParser(description="Injecte un doublage FR Kokoro à partir d'un SRT.")
    parser.add_argument("--video", required=True, help="Chemin de la vidéo d'entrée")
    parser.add_argument("--srt", required=True, help="Chemin du fichier SRT français")
    parser.add_argument("--out", required=True, help="Chemin du MP4 de sortie")
    parser.add_argument("--voix", default="ff_siwis", help="Voix Kokoro (défaut: ff_siwis)")
    parser.add_argument("--vitesse", type=float, default=1.0, help="Vitesse de synthèse")
    args = parser.parse_args()

    print("=" * 55)
    print("  SRT Français -> Audio GPU (Kokoro TTS)")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = "cuda"
        print(f"\n  GPU : {gpu_name}")
    else:
        device = "cpu"
        print("\n  Pas de GPU, utilisation CPU")

    print(f"\n  Vidéo  : {args.video}")
    print(f"  SRT    : {args.srt}")
    print(f"  Sortie : {args.out}")
    print(f"  Voix   : {args.voix}")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Vidéo introuvable: {args.video}")
    if not os.path.exists(args.srt):
        raise FileNotFoundError(f"SRT introuvable: {args.srt}")

    print("\n[0/3] Chargement du modèle Kokoro...")
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="f", device=device)
    print(f"   -> Modèle chargé sur {device.upper()}")

    with tempfile.TemporaryDirectory() as tmp:
        print("\n[1/3] Lecture du SRT français...")
        segments = generer_segments_depuis_srt(args.srt, tmp)
        print(f"   -> {len(segments)} sous-titres trouvés.")

        print(f"\n[2/3] Génération audio (Kokoro sur {device.upper()})...")
        total = len(segments)
        echoues = 0

        for i, seg in enumerate(segments):
            try:
                gen = pipeline(seg["texte"], voice=args.voix, speed=args.vitesse)
                audio_data = None
                for _, _, audio in gen:
                    audio_data = audio if audio_data is None else np.concatenate([audio_data, audio])

                if audio_data is not None and len(audio_data) > 0:
                    sf.write(seg["fichier"], audio_data, 24000)
                else:
                    sf.write(seg["fichier"], np.zeros(100), 24000)
                    echoues += 1
            except Exception:
                sf.write(seg["fichier"], np.zeros(100), 24000)
                echoues += 1

            if total and ((i + 1) % 50 == 0 or (i + 1) == total):
                pct = int((i + 1) / total * 100)
                print(f"   {i + 1}/{total} ({pct}%) segments générés...")

        print(f"   -> {total - echoues}/{total} segments réussis.")

        print("\n[3/3] Assemblage et intégration dans la vidéo...")
        duree = obtenir_duree_video(args.video)
        piste_wav = assembler_audio_avec_liste(segments, duree, tmp)

        cmd_video = [
            "ffmpeg",
            "-y",
            "-i",
            args.video,
            "-i",
            piste_wav,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            args.out,
        ]
        r = subprocess.run(cmd_video, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Erreur FFmpeg:\n{r.stderr[-800:]}")

    print("\n" + "=" * 55)
    print("  TERMINÉ !")
    print(f"  -> {args.out}")
    print("=" * 55)


if __name__ == "__main__":
    main()
