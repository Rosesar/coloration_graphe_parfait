import streamlit as st
import pandas as pd
from graphviz import Graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib.cm as cm
st.write("<center><h1>Coloration d\'un graphe parfait d\'ordre 9</h1></center>", unsafe_allow_html=True)

##########AFFICHAGE DU GRAPHE 
def afficher_graphe(matrice_adjacence):
    n = len(matrice_adjacence)
    dot = Graph()  # Use Graph() instead of Digraph()

    # Ajouter tous les sommets au graphe
    for i in range(n):
        dot.node(str(i))

    # Ajouter toutes les arêtes au graphe
    for i in range(n):
        for j in range(i+1, n):  # Iterate only over upper triangle of the adjacency matrix
            if matrice_adjacence[i][j] == 1:
                # Add edge from i to j
                dot.edge(str(i), str(j))

    return dot
###############################caracterisation 1#####################################
def supprimer_sommet(sommet, matrice_adjacence):
    
    ''' supprime un sommet et toutes ses arêtes dans la matrice d'adjacence 
    en mettant à zéro les entrées correspondantes dans la matrice.'''
    n = len(matrice_adjacence)
    for i in range(n):
        matrice_adjacence[i][sommet] = 0
        matrice_adjacence[sommet][i] = 0 # symetrie de la  matrice 
    return matrice_adjacence


def trouver_sommets_simpliciaux_et_les_supprimer(matrice_adjacence):
    # Un sommet x d’un graphe G est dit simplicial si son voisinage NG(x) est une clique.
    n = len(matrice_adjacence)
    sommets_simpliciaux = []
    for i in range(n):
     for sommet in range(n):
            # Trouver les voisins du sommet
         voisins = [voisin for voisin in range(n) if matrice_adjacence[sommet][voisin] == 1]
         if len(voisins) >= 1: # au moins un seul voisin
             # Créer une sous-matrice avec les voisins du sommet
             sous_matrice = [[matrice_adjacence[i][j] for j in voisins] for i in voisins]
             # Vérifier si la sous-matrice est une clique
             #Si c'est une clique, le sommet est ajouté à la liste des sommets simpliciaux et supprimé du graphe.
             est_clique = all(all(sous_matrice[i][j] == 1 for j in range(len(voisins)) if j != i) for i in range(len(voisins)))
             if est_clique:
                 sommets_simpliciaux.append(sommet)
                 supprimer_sommet(sommet, matrice_adjacence)
                 break

    return sommets_simpliciaux



def est_triangule(matrice_adjacence):
    """
    Vérifie si un graphe est triangulé en supprimant les sommets simpliciaux.
    """
    n = len(matrice_adjacence)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
           for j in range(n):
                M[i][j] = matrice_adjacence[i][j]

               
    st.markdown('<span style="color: green;">$G$ est triangulé?:</span>', unsafe_allow_html=True)
    sommets_simpliciaux = trouver_sommets_simpliciaux_et_les_supprimer(matrice_adjacence)
    st.write(r"Les sommets simpliciaux de $G$ :",sommets_simpliciaux)
    if  matrice_adjacence == [[0]*n for _ in range(n)]:    
        st.write(r"$G$ est triangulé!")
    else:
        st.write(r"$G$ n'est pas triangulé.")
        st.write("<p style='color:red;'>Ainsi, Le graphe n'est pas scindé et n'est pas d\' intervalle!</p>", unsafe_allow_html=True)
    st.markdown('<span style="color: green;">$\overline{G}$ est triangulé?:</span>', unsafe_allow_html=True)
    # Création du complémentaire de la matrice d'adjacence
    complementaire = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if M[i][j] == 0:
                     complementaire[i][j] = 1
                else:
                     complementaire[i][j] = 0
    sommets_simpliciaux2 = trouver_sommets_simpliciaux_et_les_supprimer(complementaire)
    st.write(r"Les sommets simpliciaux de $\bar{G}$ :",sommets_simpliciaux2)
    # Vérifier si le complémentaire est triangulé
    if complementaire == [[0]*n for _ in range(n)]:
        st.write(r"$\bar{G}$ est triangulé aussi!")
        st.write("<p style='color:red;'>Ainsi, Le graphe est scindé!</p>", unsafe_allow_html=True)

    else :
        st.write(r"$\bar{G}$ n'est pas triangulé.")
        st.write("<p style='color:red;'>Ainsi, Le graphe n'est pas scindé !</p>", unsafe_allow_html=True)
##############################caractérisation 2############################            
def find_k33_subgraph(G):
    nodes = list(G.nodes())
    for combo_a in combinations(nodes, 3):
        remaining_nodes = [n for n in nodes if n not in combo_a]
        for combo_b in combinations(remaining_nodes, 3):
            if all(G.has_edge(a, b) for a in combo_a for b in combo_b):
                if all(not G.has_edge(a, b) for a in combo_a for b in combo_a if a != b):
                    if all(not G.has_edge(a, b) for a in combo_b for b in combo_b if a != b):
                        k33_subgraph = G.subgraph(combo_a + combo_b)
                        return k33_subgraph, combo_a, combo_b
    return None, None, None
        
############################### caractérisation 2 ############################################
def contient_C5_induit(matriceadj):
    n = len(matriceadj)
    i=0
    for i in range(n):
        excluded_values = [i]
        for j in [x for x in range(n) if x not in excluded_values]:
            if matriceadj[i][j] == 1:
                excluded_values.append(j)
                for k in[x for x in range(n) if x not in excluded_values]:
                    if matriceadj[j][k] == 1:
                        excluded_values.append(k)
                        for l in[x for x in range(n) if x not in excluded_values]:
                            if matriceadj[k][l] == 1:
                                excluded_values.append(l)
                                for h in[x for x in range(n) if x not in excluded_values]:
                                    if matriceadj[i][h] == 1 and matriceadj[l][h] == 1:
                                        if matriceadj[i][k] == 0 and matriceadj[j][l] == 0 and matriceadj[i][l] == 0 and matriceadj[j][h] == 0 and matriceadj[k][h] == 0 :
                                           #print(excluded_values, ' with ', h)
                                           return True
                                excluded_values.pop()
                        excluded_values.pop()
                excluded_values.pop()
        excluded_values.pop()

    return False
  
    

def contient_C7_induit(matriceadj):
    n = len(matriceadj)
    i=0
    for i in range(n):
        excluded_values = [i]
        for j in [x for x in range(n) if x not in excluded_values]:
            if matriceadj[i][j] == 1:
                excluded_values.append(j)
                for k in[x for x in range(n) if x not in excluded_values]:
                    if matriceadj[j][k] == 1:
                        excluded_values.append(k)
                        for l in[x for x in range(n) if x not in excluded_values]:
                            if matriceadj[k][l] == 1:
                                excluded_values.append(l)
                                for m in[x for x in range(n) if x not in excluded_values]:
                                    if matriceadj[l][m] == 1:
                                        excluded_values.append(m)
                                        for n in[x for x in range(n) if x not in excluded_values]:
                                            if matriceadj[m][n] == 1:
                                                excluded_values.append(n)                              
                                                for h in[x for x in range(n) if x not in excluded_values]:
                                                    if matriceadj[i][h] == 1 and matriceadj[n][h] == 1:
                                                        if matriceadj[i][k] == 0 and matriceadj[i][l] == 0 and matriceadj[i][m] == 0 and matriceadj[i][n] == 0 and matriceadj[j][l] == 0 and matriceadj[j][m] == 0 and matriceadj[j][n] == 0 and matriceadj[j][h] == 0 and matriceadj[k][m] == 0 and matriceadj[k][n] == 0 and matriceadj[k][h] == 0 and matriceadj[l][n] == 0 and matriceadj[l][h] == 0 and matriceadj[m][h] == 0:
                                                           #print(excluded_values, ' with ', h)
                                                           return True
                                                excluded_values.pop()
                                        excluded_values.pop()
                                excluded_values.pop()
                        excluded_values.pop()
                excluded_values.pop()
        excluded_values.pop()

    return False
    

    
def contient_C9_induit(matriceadj):
    n = len(matriceadj)
    i=0
    for i in range(n):
        excluded_values = [i]
        for j in [x for x in range(n) if x not in excluded_values]:
            if matriceadj[i][j] == 1:
                excluded_values.append(j)
                for k in[x for x in range(n) if x not in excluded_values]:
                    if matriceadj[j][k] == 1:
                        excluded_values.append(k)
                        for l in[x for x in range(n) if x not in excluded_values]:
                            if matriceadj[k][l] == 1:
                                excluded_values.append(l)
                                for m in[x for x in range(n) if x not in excluded_values]:
                                    if matriceadj[l][m] == 1:
                                        excluded_values.append(m)
                                        for n in[x for x in range(n) if x not in excluded_values]:
                                            if matriceadj[m][n] == 1:
                                                excluded_values.append(n) 
                                                for o in[x for x in range(n) if x not in excluded_values]:
                                                    if matriceadj[n][o] == 1:
                                                        excluded_values.append(o) 
                                                        for p in[x for x in range(n) if x not in excluded_values]:
                                                            if matriceadj[o][p] == 1:
                                                                excluded_values.append(p)                                              
                                                                for h in[x for x in range(n) if x not in excluded_values]:
                                                                    if matriceadj[i][h] == 1 and matriceadj[p][h] == 1:
                                                                        if matriceadj[i][k] == 0 and matriceadj[i][l] == 0 and matriceadj[i][m] == 0 and matriceadj[i][n] == 0  and matriceadj[i][o] == 0 and matriceadj[i][p] == 0 and matriceadj[j][l] == 0 and matriceadj[j][m] == 0 and matriceadj[j][n] == 0  and matriceadj[j][o] == 0 and matriceadj[j][p] == 0 and matriceadj[j][h] == 0 and matriceadj[k][m] == 0 and matriceadj[k][n] == 0  and matriceadj[k][o] == 0 and matriceadj[k][p] == 0  and matriceadj[k][h] == 0 and matriceadj[l][n] == 0 and matriceadj[l][o] == 0 and matriceadj[l][p] == 0 and matriceadj[l][h] == 0 and matriceadj[m][o] == 0 and matriceadj[m][p] == 0 and matriceadj[m][h] == 0 and matriceadj[n][p] == 0 and matriceadj[n][h] == 0:
                                                                           #print(excluded_values, ' with ', h)
                                                                           return True
                                                                excluded_values.pop()
                                                        excluded_values.pop()
                                                excluded_values.pop()
                                        excluded_values.pop()
                                excluded_values.pop()
                        excluded_values.pop()
                excluded_values.pop()
        excluded_values.pop()

    return False
    

    
def complementaire(matriceadj):
    n=len(matriceadj)
    matriceComp=[[0]*n for i in range (n)]
    for i in range(n):
        for j in range(n):
            if i!=j:
                if matriceadj[i][j]==0:
                    matriceComp[i][j]=1
                else:
                    matriceComp[i][j]=0
    return matriceComp   


############################### caractérisation 3 ############################################
def trouver_sous_graphe_k33(G):
    nodes = list(G.nodes())
    for combo_a in combinations(nodes, 3):
        remaining_nodes = [n for n in nodes if n not in combo_a]
        for combo_b in combinations(remaining_nodes, 3):
            if all(G.has_edge(a, b) for a in combo_a for b in combo_b):
                if all(not G.has_edge(a, b) for a in combo_a for b in combo_a if a != b):
                    if all(not G.has_edge(a, b) for a in combo_b for b in combo_b if a != b):
                        k33_subgraph = G.subgraph(combo_a + combo_b)
                        return k33_subgraph, combo_a, combo_b
    return None, None, None

def matrice_adjacence_vers_graphe(adj_matrix):
    return nx.from_numpy_array(np.array(adj_matrix))

def colorier_sous_graphe_k33(k33_subgraph):
    coloring = nx.coloring.greedy_color(k33_subgraph, strategy="largest_first")
    pos = nx.spring_layout(k33_subgraph)
    plt.figure(figsize=(8, 6))
    nx.draw(k33_subgraph, pos, with_labels=True, node_color=list(coloring.values()), cmap=plt.cm.Set1)
    plt.title("Coloration du sous-graphe K3,3")
    st.pyplot(plt)
    return coloring

def colorier_reste_graphe(G, k33_subgraph, k33_coloring):
    coloring = k33_coloring.copy()
    remaining_nodes = set(G.nodes()) - set(k33_coloring.keys())

    for node in remaining_nodes:
        neighbors = set(G.neighbors(node))
        neighbor_colors = {coloring.get(neighbor) for neighbor in neighbors if neighbor in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[node] = color

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=[coloring[n] for n in G.nodes()], cmap=plt.cm.Set1)
    plt.title("Coloration finale du graphe entier")
    st.pyplot(plt)
############################### caractérisation 4 ############################################
# Function to display the graph
def dessiner_graphe(graph, titre=""):
    """Display the graph with a title."""
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    node_colors = nx.get_node_attributes(graph, "color")
    if node_colors:
        unique_colors = list(set(node_colors.values()))
        color_map = {
            c: cm.get_cmap("tab10")(i / len(unique_colors))
            for i, c in enumerate(unique_colors)
        }
        colors = [color_map[node_colors[node]] for node in graph.nodes()]
    else:
        colors = "lightblue"
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=500,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(titre)
    st.pyplot(plt.gcf())
    plt.close()

# Function to check if the graph is a clique
def est_clique(graph):
    """Check if the graph is a clique (every vertex is connected to every other vertex)."""
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v and not graph.has_edge(u, v):
                return False
    return True

# Function to find a 2-pair
def est_2_paire(graph, u, v):
    if graph.has_edge(u, v):
        return False
    chemins = list(nx.all_simple_paths(graph, source=u, target=v))
    chemins_filtrés = chemins.copy()
    for chemin in chemins:
        for autre_chemin in chemins:
            if chemin != autre_chemin:
                intersection = set(chemin) & set(autre_chemin)
                if intersection - {u, v}:
                    if len(chemin) > len(autre_chemin) and chemin in chemins_filtrés:
                        chemins_filtrés.remove(chemin)
                    elif (
                        len(autre_chemin) > len(chemin)
                        and autre_chemin in chemins_filtrés
                    ):
                        chemins_filtrés.remove(autre_chemin)

    for chemin in chemins_filtrés:
        if (len(chemin) - 1) % 2 != 0:
            return False

    return True

def trouver_2_paire(graph):
    """Find a 2-pair (u, v) in the graph."""
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v and est_2_paire(graph, u, v):
                return u, v
    return None

def contracter_2_paire(graph, u, v):
    """Contract the pair (u, v) in the graph."""
    graph = graph.copy()
    graph = nx.contracted_nodes(graph, u, v, self_loops=False)
    return graph

# Function for graph coloring
def coloration_parfaite(graph):
    graph_copy = graph.copy()
    sequence_contraction = []
    while not est_clique(graph_copy):
        paire = trouver_2_paire(graph_copy)
        if not paire:
            st.write("Aucun paire d\'amis trouvé , le graphe n'\est pas parfait!")
            return None
        st.write(f"paire d\'amis trouvé : {paire}")
        sequence_contraction.append(paire)
        graph_copy = contracter_2_paire(graph_copy, paire[0], paire[1])
        dessiner_graphe(graph_copy, f"Graphe après contraction de : {paire}")

    clique = list(graph_copy.nodes())
    couleur_clique = {sommet: i + 1 for i, sommet in enumerate(clique)}

    nx.set_node_attributes(graph_copy, couleur_clique, "color")
    dessiner_graphe(graph_copy, "Coloration finale du clique")

    colorations = {}
    for u, v in reversed(sequence_contraction):
        couleur = couleur_clique.get(u, couleur_clique.get(v))
        colorations[u] = colorations[v] = couleur

    for sommet in graph.nodes():
        if sommet not in colorations:
            colorations[sommet] = couleur_clique[sommet]

    nx.set_node_attributes(graph, colorations, "color")
    dessiner_graphe(graph, "Coloration finale du graphe")
    return colorations

# Helper function to extract edges from adjacency matrix
def extract_edges_from_adjacency_matrix(matrix):
    edges = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):  # Only iterate over upper triangle for undirected graph
            if matrix[i, j] == 1:
                edges.append((i + 1, j + 1))  # Adjust indices to be 1-based
    return edges
#########################caractérisation 5#######################
def afficher_graphe(matrice_adjacence):
    n = len(matrice_adjacence)
    dot = Graph()  # Use Graph() instead of Digraph()

    # Ajouter tous les sommets au graphe
    for i in range(n):
        dot.node(str(i))

    # Ajouter toutes les arêtes au graphe
    for i in range(n):
        for j in range(i+1, n):  # Iterate only over upper triangle of the adjacency matrix
            if matrice_adjacence[i][j] == 1:
                # Add edge from i to j
                dot.edge(str(i), str(j))

    return dot
# DSATUR function
def Dsatur(matrice_adj):
    n = len(matrice_adj)
    couleurs = [-1] * n
    saturation = [0] * n
    degrees = [sum(row) for row in matrice_adj]
    sommet_non_colorie = set(range(n))

    def couleur_min_voisinage(sommet):
        couleurs_voisinage = {couleurs[i] for i in range(n) if matrice_adj[sommet][i] == 1 and couleurs[i] != -1}
        couleur = 0
        while couleur in couleurs_voisinage:
            couleur += 1
        return couleur

    sommet_actuel = degrees.index(max(degrees))
    couleurs[sommet_actuel] = 0
    sommet_non_colorie.remove(sommet_actuel)

    for voisin in range(n):
        if matrice_adj[sommet_actuel][voisin] == 1 and voisin in sommet_non_colorie:
            saturation[voisin] += 1

    while sommet_non_colorie:
        sommet_actuel = max(sommet_non_colorie, key=lambda sommet: (saturation[sommet], degrees[sommet]))
        couleurs[sommet_actuel] = couleur_min_voisinage(sommet_actuel)
        sommet_non_colorie.remove(sommet_actuel)
        for voisin in range(n):
            if matrice_adj[sommet_actuel][voisin] == 1 and voisin in sommet_non_colorie:
                saturation[voisin] += 1

    return couleurs, max(couleurs) + 1
####### MISE EN PAGE
C1 =r"Le graphe $G$ est triangulé  $\iff$ $G$ admet un ordre d\'élimination simplicial."
C2 =r"Le graphe $G$ est planaire  $\iff$ $G$ ne contient pas un sous-graphe induit $K_{3,3}$."
C3 =r"Le graphe $G$ est parfait $\iff$ $G$ ne contient ni trou ni anti-trou impair."
C4 = r"Coloration de graphe $G$ utilisant une heuristique basé sur la coloration de $K_{3,3}$."
C5 =r"Coloration de graphe $G$ utilisant une algorithme des graphes parfaits."
C6 =r"Coloration de graphe $G$ utilisant l\'algoritme Dsatur."
# Définir les options de choix
options = ["G est triangulé?","G contient un sous graphe k33?","G est parfait?", "Heuristique de coloration basé sur K3,3","Coloration graphe parfait","Coloration Dsatur"]

# Demander à l'utilisateur de choisir une option
choix = st.sidebar.radio("Choisir une application:", options)

if choix=="G est triangulé?":
    st.subheader('Reconnaissance d\'un graphe triangulé  :') 
    st.markdown(f"> {C1}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
   # Titre de la page
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        # Afficher le graphe
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        graph = afficher_graphe(matrice_adjacence.values.tolist())
        st.graphviz_chart(graph.source)
        est_triangule(matrice_adjacence.values.tolist())
        
    else:
        st.write("Aucun fichier CSV n'a été téléchargé.")


###CARACTERISATION 2 ###
elif choix== "G contient un sous graphe k33?":
    st.subheader('Vérification si $G$ contient un $K_{3,3}$:') 
    st.markdown(f"> {C2}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
    # Vérifier si un fichier a été téléchargé
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        # Afficher le graphe
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        adj_matrix = matrice_adjacence.to_numpy()
        # Convert matrix to graph
        G = matrice_adjacence_vers_graphe(adj_matrix)
        mapping = {i: i + 1 for i in range(len(adj_matrix))}
        G = nx.relabel_nodes(G, mapping)
        # Find K3,3 subgraph
        k33_subgraph, set_a, set_b = find_k33_subgraph(G)
        if k33_subgraph:
            st.success("sous-graph $K_{3,3}$ trouvé !")
            st.write(f"L\'ensemble A: {set_a}")
            st.write(f"L\'ensemble B: {set_b}")
            st.write("les arêtes de K3,3:", list(k33_subgraph.edges()))

            # Visualization of the graph
            st.header("Visualisation graphique")
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(G)  # Layout for visualization
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
            nx.draw(k33_subgraph, pos, with_labels=True, node_color='orange', edge_color='orange', node_size=500, font_size=10)
            st.pyplot(plt)
    else:
        st.write("Aucun fichier CSV n'a été téléchargé.")        
        




###caractérisation 3###
elif choix=="G est parfait?":
    st.subheader('Reconnaissance d\'un graphe parfait ordre 9:') 
    st.markdown(f"> {C3}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
    # Vérifier si un fichier a été téléchargé
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        # Afficher le graphe
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        graph = afficher_graphe(matrice_adjacence.values.tolist())
        st.graphviz_chart(graph.source)
        M=matrice_adjacence.values.tolist()
        matriceComp=complementaire(M)
        
        if contient_C5_induit(M) == False:
            st.write(r"$G$ ne contient pas de $C_5$")
        else: 
            st.write(r"$G$ contient un $C_5$")

        if contient_C7_induit(M) == False:
            st.write(r"$G$ ne contient pas de $C_7$")
        else: 
            st.write(r"$G$ contient un $C_7$")

        if contient_C9_induit(M) == False:
            st.write(r"$G$ ne contient pas un $C_9$")
        else: 
            st.write(r"$G$ contient un $C_9$")
        
        if contient_C5_induit(matriceComp) == False:
            st.write(r"$\bar{G}$ ne contient pas de $C_5$")
        else: 
            st.write(r"$\bar{G}$ contient un $C_5$")

        if contient_C7_induit(matriceComp) == False:
            st.write(r"$\bar{G}$ ne contient pas de $C_7$")
        else: 
            st.write(r"$\bar{G}$ contient un $C_7$")

        if contient_C9_induit(matriceComp) == False:
            st.write(r"$\bar{G}$ ne contient pas un $C_9$")
        else: 
            st.write(r"$\bar{G}$ contient un $C_9$") 

        if contient_C5_induit(M) == False and contient_C7_induit(M) == False and contient_C9_induit(M) == False and contient_C5_induit(matriceComp) == False and contient_C7_induit(matriceComp) == False and  contient_C9_induit(matriceComp) == False:
            st.write("<p style='color:red;'> Le graphe est parfait!</p>", unsafe_allow_html=True)
        else:
            st.write("<p style='color:red;'> Le graphe n'est pas parfait!</p>", unsafe_allow_html=True)
                
    else:
        st.write("Aucun fichier CSV n'a été téléchargé.")


###CARACTERISATION 4 ###
elif choix == "Heuristique de coloration basé sur K3,3":
    # Titre de la page
    st.subheader("Heuristique de coloration basé sur K3,3")
    st.markdown(f"> {C4}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Cette application permet de rechercher un sous-graphe $K_{3,3}$ et de colorier le graphe $G$ en basant sur la coloration de $K_{3,3}$.")
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        # Afficher le graphe
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        adj_matrix = matrice_adjacence.to_numpy()
        # Convert matrix to graph
        G = matrice_adjacence_vers_graphe(adj_matrix)
        mapping = {i: i + 1 for i in range(len(adj_matrix))}
        G = nx.relabel_nodes(G, mapping)

        # Find K3,3 subgraph
        st.title("Recherche et coloration d'un sous-graphe K3,3")
        k33_subgraph, set_a, set_b = trouver_sous_graphe_k33(G)
        if k33_subgraph:
            st.success("Sous-graphe K3,3 trouvé!")
            st.write("Ensemble A:", set_a)
            st.write("Ensemble B:", set_b)

            # Step 1: Color the K3,3 subgraph
            st.subheader("Coloration du sous-graphe K3,3")
            k33_coloring = colorier_sous_graphe_k33(k33_subgraph)

            # Step 2: Color the rest of the graph
            st.subheader("Coloration du reste du graphe")
            colorier_reste_graphe(G, k33_subgraph, k33_coloring)
        else:
            st.warning("Aucun sous-graphe K3,3 trouvé.")
        
    else:
            st.write("Aucun fichier CSV n'a été téléchargé.")


###CARACTERISATION 5 ###
elif choix=="Coloration graphe parfait":
    st.subheader("Coloration graphe parfait")
    st.markdown(f"> {C5}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        # Afficher le graphe
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        adj_matrix = matrice_adjacence.to_numpy()
        edges = extract_edges_from_adjacency_matrix(adj_matrix)
        G = nx.Graph()
        G.add_edges_from(edges)
        st.write("### Initial Graph:")
        dessiner_graphe(G, "Initial Graph")
        # Perform perfect graph coloring
        st.write("### Coloring Process:")
        result = coloration_parfaite(G)
        if result:
            st.write("### Final Coloring:")
            st.write(result)
        
    else:
        st.write("Aucun fichier CSV n'a été téléchargé.")
    
    
    
    
    
    
###caractérisation 6###    
else:
    st.subheader("Coloration Dsatur")
    st.markdown(f"> {C6}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        matrice_adjacence = pd.read_csv(file)
        st.write("La matrice d’adjacence importée:")
        st.write(matrice_adjacence)  # Afficher le DataFrame uniquement si un fichier est téléchargé
        st.write(r"Soit $G=(V, E)$ le graphe correspondant à la matrice d'adjacence:")
        graph = afficher_graphe(matrice_adjacence.values.tolist())
        st.graphviz_chart(graph.source)
        M=matrice_adjacence.values.tolist()
        # Step 2: Apply DSATUR algorithm
        coloration, nbr_chromatique = Dsatur(M)
        st.success(f"Nombre chromatique calculé: {nbr_chromatique}")
        # Step 3: Create and visualize the graph
        G = nx.Graph()

        # Construct the graph from the adjacency matrix M
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                if M[i][j] == 1:
                    G.add_edge(i + 1, j + 1)  # Use 1-based indexing for nodes

        # Map colors to nodes based on the DSATUR result
        colors = [coloration[node - 1] for node in G.nodes()]  # Adapt the coloring for 1-based nodes
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph with the calculated colors
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            G, pos,
            with_labels=True,
            labels={node: node for node in G.nodes()},
            node_color=colors,
            node_size=600,
            cmap=plt.cm.rainbow,
            font_weight='bold'
        )
        plt.title(f"Graphe coloré avec {nbr_chromatique} couleurs", fontsize=14, fontweight="bold")
        plt.axis("off")
        st.pyplot(fig)
    else:
        st.write("Aucun fichier CSV n'a été téléchargé.")
        
          
        
    









        








