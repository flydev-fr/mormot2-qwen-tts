LOCALCHAT PORTABLE - WINDOWS 64 BITS
===================================

1. Extraire tout le ZIP dans un dossier (pas d'execution depuis le ZIP).
2. Lancer Download-Model.cmd pour telecharger Qwen3-14B Q4_K_M (~9 Go).
   La revision est figee et l'empreinte SHA-256 est verifiee. Si le transfert
   est interrompu, relancer le meme fichier pour reprendre.
3. Lancer Chat-GPU.cmd (Vulkan) ou Chat-CPU.cmd.
4. Lancer Benchmark.cmd pour mesurer CPU et GPU, ou Benchmark-CPU.cmd /
   Benchmark-GPU.cmd pour un seul backend.

Aucun Delphi, Python, serveur ou cle API n'est necessaire. Le paquet utilise
PowerShell et curl fournis avec Windows 10/11. Pour le GPU, il faut un pilote
Vulkan fonctionnel. Le modele occupe 9 Go sur disque, plus la memoire utilisee
par le moteur et le contexte ; il a ete teste sur une Radeon 16 Go. Si le GPU
ne peut pas le charger, utiliser le CPU ou le modele 4B ci-dessous.

Le modele teste est le 14B (14 milliards de parametres), pas un 16B.

CHAT
----
/exit : quitter. /reset : oublier la conversation.
Le chargement et chaque reponse affichent leur duree en ms, s ou min.
Le debit "overall" du chat inclut le traitement du prompt et le streaming.
Le benchmark separe prefill et decode et exclut le chargement du modele.

RESULTATS A PARTAGER
-------------------
Chaque benchmark cree un dossier results/date-heure contenant :
- summary.csv : medianes, par modele et backend ;
- results.csv : toutes les mesures ;
- system-info.json : CPU, GPU, pilote, RAM si disponible, parametres et
  empreinte du binaire. Aucun nom de PC, utilisateur ou numero de serie ;
- les journaux bruts .log.
Envoyer ce petit dossier de resultats pour comparer les PC.
reference/ contient les mesures initiales (Ryzen 9950X3D / RX 9060 XT).
Les parametres par defaut sont 512 tokens en entree, 128 en sortie,
3 repetitions apres chauffe. Ce test mesure le debit, pas la qualite.
Si le GPU echoue, les mesures CPU deja terminees restent disponibles.

AUTRES COMMANDES (dans PowerShell, depuis le dossier extrait)
-----------------------------------------------------------
Modele plus petit (~2,5 Go) :
  .\tools\chat\download-model.ps1 -Model 4B
  .\tools\chat\portable.ps1 -Action Chat -Model 4B -Backend vulkan
  .\tools\chat\portable.ps1 -Action Benchmark -Model 4B -Backend both

Un prompt sans mode interactif :
  .\tools\chat\portable.ps1 -Action Chat -Backend cpu -Prompt "Bonjour"

Verifier un modele deja telecharge :
  .\tools\chat\download-model.ps1 -Model 14B -VerifyOnly

SOURCES ET LICENCES
------------------
Les sources Pascal des demos et des bibliotheques utilisees sont dans source/.
Les licences et notices sont conservees dans LICENSE, NOTICE et licenses/.
build-info.json identifie les revisions et empreintes des executables.
Les poids ne sont pas inclus dans le ZIP ; ils sont telecharges separement
depuis Hugging Face. Modeles Qwen3 : licence Apache-2.0.
  https://huggingface.co/unsloth/Qwen3-14B-GGUF
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
