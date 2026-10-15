# TD GPGPU n°1 – Exercice 4 : somme de deux matrices

## Compilation
```
cd Ex4
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<votre archi>   # 86 par défaut (RTX 30xx)
cmake --build build
```
(l'exécutable `matsum` est créé à la racine du dossier, comme pour Ex3)

## Utilisation
```
./matsum -l 4096 -h 4096                    # L x H, blocks 16x16 par défaut
./matsum -l 4096 -h 4096 -tx 32 -ty 8       # autre taille de block
./matsum -l 8192 -h 8192 -mb 64             # grille limitée à ~64 blocks (boucles)
./matsum -l 8192 -h 8192 -bench             # tableau comparatif automatique
```

## Ce que fait le programme
1. Lit L et H, crée 2 matrices L×H aléatoires (float, graine fixe).
2. Somme sur CPU (ChronoCPU).
3. Somme sur GPU : alloc / H→D / kernel / D→H / free chronométrés séparément (ChronoGPU).
4. Compare CPU et GPU.

## Points à analyser (question 2)
- **Petites matrices** : le CPU gagne (coût fixe des transferts + lancement).
- **Grandes matrices** : le kernel est très rapide, mais le *total* GPU est dominé par
  les transferts PCIe (opération mémoire-bound : 1 addition pour 12 octets déplacés).
- **Taille de block** : 1x1 très mauvais (1 thread/warp -> 31/32 de l'unité inutilisée) ;
  à partir de 128–256 threads/block les temps se stabilisent. Attention aux blocks "larges
  en x" (32x8) vs "hauts" (8x32) : x est contigu en mémoire, donc un warp doit couvrir
  des x consécutifs (accès coalescés).
- **Grille limitée (-mb)** : le résultat reste correct grâce aux boucles grid-stride,
  mais avec trop peu de blocks le GPU est sous-utilisé -> kernel plus lent.
