Put one 16-bit mono 16 kHz WAV here as `caller.wav`.

Use a real recording from the queue you are targeting, not a clean studio read —
narrowband PSTN audio with background noise is what the endpointer actually has
to survive, and it is where the number moves.

    ffmpeg -i whatever.m4a -ac 1 -ar 16000 -sample_fmt s16 caller.wav

Two properties matter: at least a second of genuine trailing silence after the
last word (otherwise the endpointer never fires inside the clip), and speech
that ends on a complete clause. To test the dangling-phrase case from the
turn-taking spec, record a second clip ending mid-sentence and compare.

Once it's here:

    python3 -m spike.transcribe audio/caller.wav
    python3 -m spike.endpoint_eval --wav audio/caller.wav --words audio/caller.words.json

No Deepgram key? Type the transcript into `audio/caller.txt` and pass `--text`
instead. Words get spread across the voiced audio — approximate, and enough to
exercise the semantic gate.
