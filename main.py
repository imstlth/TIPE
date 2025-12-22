from multiprocessing import Process, Manager
import os
import numpy as np
import time
import random
from PIL import Image
import pandas

# On s'aide un peu de https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders
# Avec numpy, le produit matriciel s'écrit A @ B
# la transposée A.T et le modulo np.mod(mat, modulo).
# F256 = F2[X]/(P) avec P un polynôme de F2[X] irréductible de degré 8
# (P) = P F2[X]
# En l'occurence, grâce à des gens intelligents sur Internet,
# X^8 + X^7 + X^2 + X + 1 (ligne 54)
# est irréductible sur F2[X].
# Il a aussi une propriété ultra cool : notre F256* = {X^k}
# On représente les polynômes de F2[X] par des nombres binaires.
# Pour un polynome P, si P >= 0b100000000 = 2**8
# cela veut dire que son degré est supérieur ou égal à 8
# Python permet d'effecteur des opérations bit par bit
# ^ = XOR = + dans F2 (très important)

# Il y a beaucoup de commentaires mais le code tient en 10 lignes
def div_P(poly):
    "Calcule le reste de la div de poly (en binaire) par P"
    if poly < 2**8:
        return poly
    # On effectue d'abord la div par X^8
    quotient = poly >> 8  # Ça équivaut à se donner le quotient par X^8
    reste = poly - (quotient * 2**8)
    # Pour chaque puissance 8+n : X^8 X^n = (P - X^8) X^n
    # Ce qui revient à augmenter les puissances de P-X^8 de n puis à l'ajouter au reste du polynôme.
    # Puisque en python, + = - = ^, on "shift" P ^ 2**8 (py, ^ vu comme -) par n puis on ajoute (^) ça au reste
    n = 0
    # P0 = X^8 + X^7 + X^2 + X + 1 = 0b110000111
    # Comme le seul endroit où on l'utilise, on l'utilise - X^8, on pose direct :
    P0 = 0b110000111 ^ 2**8
    for coeff in bin(quotient)[:1:-1]:
        # Si coeff = 1, tout ce passe bien, si = 0, ^0 ne fait rien
        reste ^= (P0 << n) * int(coeff)
        n += 1
    # Comme rien n'empêche que les n soient très grands et donc que le reste ait toujours un degré ≥ 8.
    return div_P(reste)


########
# F256 #
########

class F256:

    # Attention ! F256 prend en entrée un polynôme et non pas une puissance de X
    def __init__(self, polynome):
        self.poly = div_P(polynome)

    def __add__(self, other):
        return F256(self.poly ^ other.poly)

    # Nous sommes en caractéristique 2 donc
    def __sub__(self, other):
        return F256(self.poly ^ other.poly)

    def __neg__(self):
        return self

    def __int__(self):
        return self.poly

    def __str__(self):
        # Finalement, on affiche juste le polynôme et non pas la puissance de X
        return str(self.poly)

    def __repr__(self):
        return str(self.poly)

    def __eq__(self, other):
        return self.poly == other.poly


# D'un POV purement Python, on a besoin d'avoir défini F256 avant de definir create_table
# Mais pour définir la méthode __mul__ de F256, on a besoin de create_table.
def create_tables():
    table_log = {}
    table_exp = {}
    for puissance in range(2**8 - 1):
        # 2 représente X ici
        table_log[F256(2**puissance).poly] = puissance
        table_exp[puissance] = F256(2**puissance)
    return table_log, table_exp

# La table des puissances
table_log, table_exp = create_tables()

def mulF256(self, other):
    try:
        # nouvelle multiplication -> on additionne les puissances de X
        # Il faut faire ce test avant car 0 n'est pas une puissance de X
        if 0 in [self.poly, other.poly]:
            return F256(0)
        return table_exp[(table_log[self.poly] + table_log[other.poly]) % 255]
    except:
        raise Exception("Il faut multiplier par un autre nombre de F256 !")

def powF256(self, scalaire):
    if scalaire == 0:
        return F256(1)
    elif self.poly == 0:
        return self
    else:
        return table_exp[(table_log[self.poly] * scalaire) % 255]

def truedivF256(self, other):
    if other.poly == 0:
        raise ZeroDivisionError()
    elif self.poly == 0:
        return self
    else:
        return table_exp[(table_log[self.poly] - table_log[other.poly]) % 255]

setattr(F256, "__mul__", mulF256)
setattr(F256, "__pow__", powF256)
setattr(F256, "__truediv__", truedivF256)



#################################
# NOUVELLE APPROCHE POLYNÔMIALE #
#################################

#########################################
# FONCTIONS GÉNÉRALES SUR LES POLYNÔMES #
#########################################

def clean(poly):
    copie = poly.copy()
    if copie != []:
        while copie[-1] == F256(0):
            copie.pop()
            if copie == []:
                break
    return copie

def prod(P, Q):
    n1, n2 = len(P), len(Q)
    PQ = [F256(0)] * (n1 + n2 - 1)
    for k in range(n1):
        Pk = P[k]
        for i in range(n2):
            PQ[k + i] += Pk * Q[i]
    return clean(PQ)


# On est en caractéristique 2 donc pour soustraire des polynômes, on les additionne
def somme(P, Q):
    n0 = max(len(P), len(Q))
    P += [F256(0)] * (n0 - len(P))
    Q += [F256(0)] * (n0 - len(Q))
    resultat = [P[i] + Q[i] for i in range(n0)]
    return clean(resultat)

def generateur(t):
    G = [F256(1)]
    for i in range(2 * t):
        G = prod(G, [table_exp[i], F256(1)])
    return G

def div_euclid(P, Q):
    P = clean(P)
    Q = clean(Q)
    degP, degQ = len(P) - 1, len(Q) - 1
    if degP < degQ:
        return [[], P]
    if degQ == -1:
        raise ZeroDivisionError()
    quotient = []
    c_dom_Q = Q[-1]
    for i in range(degP, degQ - 1, -1):
        # Il est possible lors de la division euclidienne qu'en retirant sub à P,
        # on élimine aussi le terme d'après donc on skip le terme nul si ça arrive
        if i < len(P):
            mult = P[i] / c_dom_Q
            sub = [F256(0)] * (i - degQ) + [ mult * c_Q for c_Q in Q ]
            P = somme(P, sub)
            quotient.insert(0, mult)
    return [quotient, P]


#######################
# ENCODAGE INDIVIDUEL #
#######################

def encodage_v2(message, k, t, results, i):
    P = [F256(0) for i in range(2 * t + 1)]
    P[-1] = F256(1)
    controle = div_euclid(prod(message, P), generateur(t))[1]
    transmis = somme(prod(message, P), controle)
    transmis += [F256(0)] * (k + 2*t - len(transmis))  # On fait en sorte qu'il est exactement k+2t coefficients
    results[i] = transmis

def evaluation(P, a):
    s = F256(0)
    for i in range(len(P)):
        s += P[i] * (a**i)
    return s


#######################
# DECODAGE INDIVIDUEL #
#######################

def identite(n):
    mat = np.full((n, n), F256(0), dtype=F256)
    for i in range(n):
        mat[i, i] = F256(1)
    return mat

def algo_du_pivot(mat):
    cop = mat.copy()
    n = cop.shape[0]
    n_pivots_prec = 0
    inv = identite(n)
    if n != cop.shape[1]:
        raise Exception("Elle est pas carrée !")
    for col in range(n):
        pivot = None
        # À chaque fois qu'on fait un pivot, on regarde plus que les lignes en dessous
        for ligne in range(n_pivots_prec, n):
            pivot = cop[ligne, col]
            if pivot != F256(0):
                # On passe à la suite
                break
        if pivot == F256(0):
            # Dans ce cas, la matrice n'est pas inversible
            return False
        # On échange les lignes "ligne" et "n_pivots_prec"
        # Il faut mettre des doubles crochets bien échanger les lignes
        cop[[ligne]], cop[[n_pivots_prec]] = cop[[n_pivots_prec]], cop[[ligne]] #type:ignore
        inv[[ligne]], inv[[n_pivots_prec]] = inv[[n_pivots_prec]], inv[[ligne]] #type:ignore
        # Opération Li <- Li - coeff/pivot Lj
        #  et Lj <- coeff/pivot Lj
        for i in range(n):
            if cop[i, col] != F256(0):
                if i == n_pivots_prec:
                    cop[i] /= pivot
                    inv[i] /= pivot
                    pivot = F256(1)
                else:
                    coeff = cop[i, col] / pivot
                    cop[i] -= np.multiply(coeff, cop[n_pivots_prec])
                    inv[i] -= np.multiply(coeff, inv[n_pivots_prec])
        n_pivots_prec += 1
    return inv

def syndromes(recu, t):
    return [evaluation(recu, table_exp[i]) for i in range(2 * t)]

def det_poly_correcteur(t, synds):
    for nu in range(t, 0, -1):
        mat = np.full((nu, nu), F256(0), dtype=F256)
        for ligne in range(nu):
            for col in range(nu):
                mat[ligne, col] = synds[nu + ligne - col - 1]
        inv = algo_du_pivot(mat)
        if inv is not False:  # inv = False si la matrice n'est pas inversible
            mat_col = np.array(synds[nu:2 * nu]).reshape((nu, 1))
            lambdas = (inv @ mat_col).reshape((1, nu)).tolist()
            return [F256(1)] + lambdas[0]
    raise Exception("Ya jamais de solution, c'est pas normal")

def det_ir_xr(poly_lambda):
    ir, xr = [], []
    for puissance in range(255):
        if evaluation(poly_lambda, table_exp[puissance]) == F256(0):
            ir.append((-puissance) % 255)
            xr.append(table_exp[(-puissance) % 255])
    return ir, xr

def det_E(ir, xr, synds, k, t):
    # if len(xr) < nu:
    #     print("Le nombre d'erreur est supérieur au seuil max, on ne peut pas décoder")
    #     exit()
    nu = len(xr)
    mat_xr = np.full((nu, nu), F256(0), dtype=F256)
    for ligne in range(nu):
        for col in range(nu):
            mat_xr[ligne, col] = xr[col] ** ligne
    mat_col = np.array(synds[:nu]).reshape((nu, 1))
    inv_xr = algo_du_pivot(mat_xr)
    yr = (inv_xr @ mat_col).reshape((1, nu)).tolist()[0]
    E = []
    for i in range(k + 2 * t):
        if i in ir:
            E.append(yr.pop())
        else:
            E.append(F256(0))
    return E

def decodage_v2(recu, k, t, results, i):
    synds = syndromes(recu, t)
    correct = True
    for s in synds:
        if s != F256(0):
            correct = False
            break
    if correct:
        results[i] = recu[-k:]
    else:
        poly_correcteur = det_poly_correcteur(t, synds)
        ir, xr = det_ir_xr(poly_correcteur)
        E = det_E(ir, xr, synds, k, t)
        s = somme(recu, E)[-k:]
        s += [F256(0)] * (k - len(s))
        results[i] = s


################
# MULTIPROCESS #
################

# Execute sur chaque bloc une fonction f qui respecte le bon format
# Utile pour l'encodage et le décodage
def multiprocess(blocs, f, t, n_process, silent):
    n_blocs = len(blocs)
    blocs_traites = [None] * n_blocs
    # Gestion des processus en parallèle
    process = [None] * n_process
    manager = Manager()
    results = manager.list([None] * n_process)
    coords_list = [None] * n_process

    for g in range(n_blocs // n_process):
        if g == 0:
            debut = time.time()
        # Lancement de chaque processus
        for i in range(n_process):
            empla = n_process * g + i
            coords_list[i] = blocs[empla][1]
            k_i = coords_list[i][2] * coords_list[i][3]
            process[i] = Process(target=f, args=(blocs[empla][0].tolist(), k_i, t, results, i))
            process[i].start()
        for i in range(n_process):
            process[i].join()
            empla = n_process * g + i
            # Stockage des résultats
            blocs_traites[empla] = (np.array(results[i]), coords_list[i]) #type:ignore
        if g == 0:
            fin = time.time()
            if not silent:
                print("Temps estimé :", int((fin - debut) * (n_blocs // n_process)), "secondes.") #type:ignore

    # S'il reste quelques processus on les fait (n_process peut ne pas diviser n_blocs)
    m = n_blocs % n_process
    for g in range(m):
        for i in range(m):
            empla = n_blocs - i - 1
            coords_list[i] = blocs[empla][1]
            k_i = coords_list[i][2] * coords_list[i][3]
            process[i] = Process(target=f, args=(blocs[empla][0].tolist(), k_i, t, results, i))
            process[i].start()
        for i in range(m):
            process[i].join()
            empla = n_blocs - i - 1
            blocs_traites[empla] = (np.array(results[i]), coords_list[i]) #type:ignore
    return blocs_traites


##################
# ENCODAGE IMAGE #
##################

# Récupère une image d'une URL et nous renvoit une matrice d'éléments de F256 la représentant.
def extract_img(url):
    # img = Image.open(urlopen(url))
    img = Image.open(url)
    largeur, hauteur = img.size
    pixels = list(img.getdata()) # type: ignore
    img.close()
    # RGB fait tripler la largeur de l'image
    Fpixels = np.full((hauteur, largeur * 3), F256(0), dtype=F256)
    for i in range(len(pixels)):
        x = i % largeur
        y = i // largeur
        for c in range(3):
            Fpixels[y, 3 * x + c] = F256(pixels[i][c])
    return Fpixels

# NOTE:
# Ce n'est pas un problème que la taille des blocs soit différente
# alors qu'ils ont le même t.

# Prend une matrice de F256 et la divise en blocs.
# Les blocs peuvent ne pas quadriller parfaitement.
# Les blocs sont des blocs de composantes de couleur et non de pixels
def diviser_blocs(img_src, blocs_x, blocs_y):
    img_y, img_x = img_src.shape
    x, y = 0, 0
    blocs = []
    while True:
        # Les coordonnées et dimensions réelles du bloc (tenant compte des bords)
        coords = (
            x, y,
            blocs_x if x + blocs_x <= img_x else img_x - x,
            blocs_y if y + blocs_y <= img_y else img_y - y
        )
        bloc_xy = img_src[y:y+coords[3], x:x+coords[2]]
        blocs.append((bloc_xy.flatten(), coords))
        # Coin en bas à droite
        if x + blocs_x >= img_x and y + blocs_y >= img_y:
            break
        # Côté de droite
        if x + blocs_x < img_x:
            x += blocs_x
        else:
            x, y = 0, y + blocs_y
    return blocs

def encoder_blocs(blocs, t, n_process, silent):
    return multiprocess(blocs, encodage_v2, t, n_process, silent)


##################
# DÉCODAGE IMAGE #
##################

# HACK:
# On n'a pas besoin de créer d'autres types de bruit.
# En effet, dans un vrai système de type Reed-Solomon,
# les blocs sont astucieusement choisis/choisis aléatoirement
# afin que le bruit se répartissent le plus possible sur les blocs.
# Dans le but de ne pas avoir une concentration excessive d'erreur sur un seul bloc
# ce qui empêcherait toute correction.

# Changer un certain % de valeurs de couleurs aléatoirement
def bruit(blocs_encodes, pourcent):
    # On calcule dans un premier temps la taille réelle du message transmis
    img_trans_size = 0
    chg = 0
    array_bruite = []
    for bloc, coords in blocs_encodes:
        img_trans_size += len(bloc)
        array_bruite.append((bloc.copy(), coords))
    # On s'assure de modifier le bon pourcentage en:
    # ne modifiant qu'une seule valeur de rgb
    # en vérifiant qu'on n'a pas déjà modifié le pixel
    while chg < img_trans_size * pourcent:
        b_rand = random.randint(0, len(blocs_encodes)-1)
        val_rand = random.randint(0, len(blocs_encodes[b_rand][0])-1)
        if array_bruite[b_rand][0][val_rand] == blocs_encodes[b_rand][0][val_rand]:
            array_bruite[b_rand][0][val_rand] = F256(random.randint(0, 255))
            chg += 1
    return array_bruite

def decoder_blocs(blocs, t, n_process, silent):
    return multiprocess(blocs, decodage_v2, t, n_process, silent)

# On récupère une image en bloc et on la reconstruit selon les bonnes dimensions toujours en F256
def recreer_img(blocs, img_x, img_y):
    img = np.full((img_y, img_x), 0, dtype=F256)
    for bloc, coords in blocs:
        img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]] = bloc.reshape((coords[3], coords[2]))
    return img

# On récupère une image en F256 et on fait une image PIL
def PIL_img(img):
    img_y, img_x = img.shape
    if img_x % 3 != 0:
        raise Exception("R, G et B ne sont pas à la suite")
    rgb_img = np.full((img_y, img_x // 3, 3), 0)
    for y in range(img_y):
        for x in range(img_x // 3):
            for c in range(3):
                rgb_img[y, x, c] = img[y, 3 * x + c].poly
    PIL_img = Image.fromarray(rgb_img.astype("uint8"), "RGB")
    return PIL_img

# On récupère une image résultat en F256 et l'image de base (en F256)
# et on renvoit le % d'erreur en couleur et en pixel
def erreur(final, initial):
    compte_val = 0
    compte_pixel = 0
    if (len(final), len(final[0])) != (len(initial), len(initial[0])):
        raise Exception("Les tailles ne correspondent pas ! (force à toi)")
    for y in range(len(final)):
        pixel = 0
        erreur_pixel = False
        for x in range(len(final[0])):
            if final[y][x] != initial[y][x]:
                compte_val += 1
                if not erreur_pixel:
                    erreur_pixel = True
                    compte_pixel += 1
            pixel += 1
            if pixel == 3:
                pixel = 0
                erreur_pixel = False
    size = len(final) * len(final[0])
    return (compte_val / size, compte_pixel / size)

# Surveille le temps pris par une tache à une précision donnée
def monitor(fn, args, texte, precis):
    print(texte)
    a = time.time()
    output = fn(*args)
    b = time.time()
    print(int((b - a) * precis) / precis)
    print()
    return output


##############
# MODE EXCEL #
##############

if input("Mode excel - défaut non : ") == "oui":

    silent = input("Mode silencieux - défaut non : ") == "oui"
    notif = input("Notification (visuelle) - défaut oui : ") != "non"
    notif_son = input("Notification (sonore) - défaut oui : ") != "non"

    mode_overwrite = input("Mode overwrite (danger) - défaut non : ") == "oui"
    if mode_overwrite == True:
        print("Sûr d'entrer en mode overwrite (toutes les anciennes sorties vont être supprimées) ?! Tu peux faire Ctrl-C pour annuler ")
        input()
    n_process = input("n_process - défaut = 16 : ")
    if n_process == "":
        n_process = 16
    else:
        n_process = int(n_process)

    entree = pandas.read_excel("/home/caracole/H4/TIPE/excel/entree.xlsx")
    sortie = pandas.read_excel("/home/caracole/H4/TIPE/excel/sortie.xlsx")

    curseur = sortie.shape[0]
    if mode_overwrite or sortie.empty:
        headers = pandas.read_excel("/home/caracole/H4/TIPE/excel/headers.xlsx")
        sortie = pandas.DataFrame(columns=headers.columns)
        curseur = 0

    if silent:
        long = int(os.popen("stty size", "r").read().split()[1])
        print("\r[" + " " * (long - 10) + "] 1/" + str(entree.shape[0]), end="")

    for i, row in entree.iterrows():
        if not silent:
            print()
            print(f"Image {i+1}/{entree.shape[0]}") #type:ignore
            print()
        taille, nom, blocs_x, blocs_y, t, pourcent_bruit = row.taille, row.nom, row.blocs_x, row.blocs_y, row.t, row.bruit
        t_max = int((255 - blocs_x * blocs_y)/2)
        # pour modifier
        # tableau.at[i, "bruit"] = 0
        # pour lire
        # row.bruit
        # pour enregistrer
        # tableau.to_excel("/home/caracole/H4/TIPE/excel/sortie.xlsx")
        img_url = f"/home/caracole/H4/TIPE/images sources/{taille}/{nom}.jpg"

        img_raw = extract_img(img_url)
        width, height = len(img_raw[0]), len(img_raw)  # Attention ! width fait 3* la largeur de la vraie img
        img_blocs = diviser_blocs(img_raw, blocs_x, blocs_y)

        debut_enc = time.time()
        blocs_encodes = encoder_blocs(img_blocs, t, n_process, silent)
        fin_enc = time.time()
        blocs_bruites = bruit(blocs_encodes, pourcent_bruit / 100) #type:ignore

        debut_dec = time.time()
        blocs_decodes = decoder_blocs(blocs_bruites, t, n_process, silent)
        fin_dec = time.time()

        img_decodee = recreer_img(blocs_decodes, width, height)
        final = PIL_img(img_decodee)
        end = f"{nom} {taille} - {pourcent_bruit} - {blocs_x}x{blocs_y} - {t} sur {t_max}.jpg"
        chemin = f"/home/caracole/H4/TIPE/resultats automatiques/{end}"
        final.save(chemin, quality=95)

        e_pourcent = erreur(img_decodee, img_raw)

        pos = curseur + i #type:ignore
        sortie.at[pos, "nom"] = nom
        sortie.at[pos, "taille"] = taille
        sortie.at[pos, "img_x"] = width
        sortie.at[pos, "img_y"] = height
        sortie.at[pos, "n_pixels"] = width * height
        sortie.at[pos, "blocs_x"] = blocs_x
        sortie.at[pos, "blocs_y"] = blocs_y
        sortie.at[pos, "t"] = t
        sortie.at[pos, "t_max"] = t_max
        sortie.at[pos, "bruit"] = pourcent_bruit
        sortie.at[pos, "enc"] = fin_enc - debut_enc
        sortie.at[pos, "dec"] = fin_dec - debut_dec
        sortie.at[pos, "e_val"] = e_pourcent[0]
        sortie.at[pos, "e_pixel"] = e_pourcent[1]

        if silent:
            long = int(os.popen("stty size", "r").read().split()[1])
            n = int(((i+1)/entree.shape[0]) * (long-10)) #type:ignore
            print("\r[" + "#" * n + " " * (long - 10 - n) + f"] {i+1}/{entree.shape[0]}", end="") #type:ignore

    sortie.to_excel("/home/caracole/H4/TIPE/excel/sortie.xlsx")
    if notif_son:
        os.system('paplay /home/caracole/Musique/notif.mp3 &')
    if notif:
        os.system('dunstify -I "/usr/share/icons/Papirus/32x32/apps/python.svg" -t 10000 -a "TIPE" "fin" "Le mode excel est terminé"')

    print("Enregistré dans /home/caracole/H4/TIPE/excel/sortie.xlsx")

    exit()


######
# UI #
######

print("C'est étrange mais en boucle, les performances diminuent nettement à chaque itération")

test_boucle = input("En boucle - défaut = non : ")
boucle = 1 if test_boucle == "oui" else -1
if boucle == 1:
    print("""Après chaque affichage de l'image décodée,
il faut appuyer 1 fois sur entrée pour passer à l'image suivante
car des messages s'affichent tout seuls.

Si tu entres 'q' la boucle s'arrête.""")

print()
while boucle != 0:

    precis_temps = input("Nb chiffres après , - défaut = 2 : ")
    if precis_temps == "":
        precis_temps = 100
    else:
        precis_temps = 10 ** int(precis_temps)
    name = input("bounty, classe HX2, HX2 tableau, logo HX2, soleil ou vincent - défaut = bounty : ")
    if name == "":
        name = "bounty"
    size = input("small, mid, big ou originale - défaut = small : ")
    if size == "":
        size = "small"
    img_url = f"/home/caracole/H4/TIPE/images sources/{size}/{name}.jpg"
    blocs_x = int(input("blocs_x : "))
    blocs_y = int(input(f"blocs_y <= {int(255/blocs_x)} : "))
    t = int(input(f"t <= {int((255 - blocs_x * blocs_y)/2)} : "))
    niveau_bruit = float(input("niveau_bruit : "))
    n_process = input("n_process - défaut = 16 : ")
    if n_process == "":
        n_process = 16
    else:
        n_process = int(n_process)

    img_raw = extract_img(img_url)
    width, height = len(img_raw[0]), len(img_raw)  # Attention ! width fait 3* la largeur de la vraie img
    img_blocs = diviser_blocs(img_raw, blocs_x, blocs_y)
    blocs_encodes = monitor(encoder_blocs, (img_blocs, t, n_process, False), "Encodage des blocs", precis_temps)

    print(f"On crée du bruit à {niveau_bruit}%")
    blocs_bruites = bruit(blocs_encodes, niveau_bruit / 100)

    if input("Afficher l'image avec du bruit ?") == "oui":
        vraie_img_blocs = [ (bloc[-coords[2]*coords[3]:], coords) for bloc, coords in blocs_bruites ]
        vraie_img_bruit = recreer_img(vraie_img_blocs, width, height)
        bruit_PIL = PIL_img(vraie_img_bruit)
        bruit_PIL.show()

    texte_decodage = f"Décodage avec des blocs/mots de taille (au max) k = {blocs_x * blocs_y} et t = {t}:"
    blocs_decodes = monitor(decoder_blocs, (blocs_bruites, t, n_process, False), texte_decodage, precis_temps)
    img_decodee = recreer_img(blocs_decodes, width, height)
    final = PIL_img(img_decodee)
    final.show()

    e_pourcent = erreur(img_decodee, img_raw)
    print("% d'erreur (rgb, pixel) :", (e_pourcent[0] * 100, e_pourcent[1] * 100))

    if input() == "q":
        break

    boucle += 1


####################
# ANCIENNE VERSION #
####################

############################
# Version finale et aboutie (normalement) de l'implémentation du premier algorithme
# c'est uniquement le 1er algorithme.
# Résultats peu concluants
# Même en optimisant énormément les fonctions les plus appelés, le programme prend une éternité.
# Maintenant algo_du_pivot s'execute aux alentours de 0.01 secondes même pour des matrices de taille 180.
# decodage aussi a été accéléré.
# La fonction de décodage d'une image a été améliorer et décode les blocs en parallèle (jusqu'à 10 en même temps - dépend de l'image)
# Sur mon PC (a priori vu les temps, il est 2x plus rapide que replit),
# rien que le décodage d'Albert Einstein en 125*196 avec 1% (!!) d'erreur prend ~30min.
# Je n'ai réussi qu'une seule fois à achever le décodage, mais à cause d'erreurs dans la suite du programme,
# je n'ai pas pû voir si le résultat était correct.

def encodage_v1(u):
    k = u.shape[0]
    a = np.zeros(255, dtype=F256)
    for i in range(255):
        c = F256(0)
        for j in range(k):
            c += u[j] * table_exp[(i * j) % 255]
        a[i] = c
    return a

def combi_random(jusqua, k, iter_max):
    for i in range(iter_max):
        yield random.sample(range(jusqua), k)

# Pour la suite, il est nécéssaire que decodage ne renvoit pas de données
# mais modifie une liste en argument à l'emplacement prévu
def decodage_v1(w, k, lim_pointe, lim_max, liste, emplacement):
    udict = {}
    # Pour chaque façon de prendre k équations parmi les 255
    for kcombi in combi_random(255, k, lim_max):
        w_k = np.array([w[x] for x in kcombi])
        matrice_systeme = np.zeros((k, k), dtype=F256)
        for i in range(len(kcombi)):
            for puissance in range(k):
                matrice_systeme[i, puissance] = table_exp[(kcombi[i] * puissance) % 255]
        u = algo_du_pivot(matrice_systeme) @ w_k
        # On convertit u en un object "hashable"
        hash_u = tuple([int(i) for i in u])
        if hash_u not in udict:
            udict[hash_u] = 0
        udict[hash_u] += 1
        # Dès que l'on a plus de "limite" équations menant au même vecteur on s'arrête
        max_compte = 0
        max_u = None
        for u in udict:
            if udict[u] > max_compte:
                max_compte = udict[u]
                max_u = u
        if max_compte >= lim_pointe:
            liste[emplacement] = np.array(max_u)
            return
    raise Exception("pas trouvé")
