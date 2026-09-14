import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import plotly.express as px

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(8,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(8,8))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree, color='#2C92D5')
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(25,10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "top",
    );
    plt.show()
    
def plot_dendrogram_centroids(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "bottom",
    );
    plt.show()

#Fonction d'affichage avec plotly
def plot_plotly(data, title):
    fig = px.line(title=title)
    for i in data:
        fig.add_scatter(x = data.index, y = data[i], name = i)
    fig.show()
        
#Création d'une fonction de carte de corrélation
def plot_heatmap(data):
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 9, 9
    corr_map = data.corr()
    mask = np.triu(np.ones_like(corr_map, dtype=bool))
    sns.heatmap(corr_map, mask = mask, center = 0, cmap='RdYlGn', linewidths=1, annot=True, fmt='.2f', vmin=-1,vmax=1)
    plt.title('Carte de corrélation', fontsize=15, fontweight='bold')
    plt.show()
    
def interet(montant,taux,duree):
    pret = montant
    total_interet = 0
    d = duree
    while(d >= 1):
        interet = (taux*montant)/100
        total_interet += interet
        montant = montant - interet
        d -= 1
        pret += interet
    mensuel = pret/duree
    return 'Le coût total du prêt est:', pret,' les intérêts représentent', total_interet, mensuel,' par an à payer'

def namestr(obj, namespace):
    ''' function that returns variable name in namespace '''
    return [name for name in namespace if namespace[name] is obj]
    
    
def data_synthese(df_0):   
    
    ''' fonction retournant une synthèse d'informations sur le data passé en paramètre :
    nombre de lignes et de colonnes'''
    data=df_0.copy()
    print('---------------------------DATA SET INFO-----------------------------------------------')
    print('Data : {}'.format(namestr(data, globals())))
    print('Number of variables : {}'.format(len(data.columns)))
    print('Number of observations : {}'.format(len(data)))
    print("")
    print("")
    print('---------------------------MISSING VALUES-----------------------------------------------')
    print('The dataset contains cell missing : {}'.format(data.isna().sum().sum()))
    print('The dataset contains cell missing in % : {:.2%}'.format(data.isna().sum().sum()/(data.size)))
    print("")
    
    
    print("")
    print("")
    print('---------------------------DUPLICATED VALUES--------------------------------------------')

    # Removing duplicates if there exist
    n_dupli = sum(data.duplicated(keep='first'))
    print('The dataset contains duplicates : {}'.format(n_dupli))
    print('The dataset contains duplicates in % : {:.2%}'.format(n_dupli/(data.size)))

    
    print("")
    print("")
    print('---------------------------OUTLIERS VALUES-----------------------------------------------')
    df_num=data.select_dtypes(exclude=['object',"category"])   #object=variables qualitatives
    Q1 = df_num.quantile(0.25)  
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1 
    is_outliers = (df_num < (Q1 - 1.5 * IQR)) |(df_num > (Q3 + 1.5 * IQR))
    print('The dataset contains cell outliers : {}'.format(is_outliers.sum().sum()))
    print('The dataset contains cell outliers in % : {:.2%}'.format(is_outliers.sum().sum()/df_num.size))
    print("")
    print("The dataset contains line with outliers",is_outliers.any(axis=1).sum().sum())
    print('The dataset contains line with outliers in % : {:.2%}'.format(is_outliers.any(axis=1).sum().sum()/len(df_num)))


    outliers_missing=pd.DataFrame({'count_outliers':is_outliers.sum(axis=0), 
                           'outliers_rate':is_outliers.sum(axis=0)/len(data)*100,
                           'count_missing':(data.isna()).sum(axis=0), 
                           'missing_rate':(data.isna()).sum(axis=0)/data.size*100,
                           "type": data.dtypes,
                            "unique":[len(data[col].unique()) for col in list(data.columns)]})                
    display(outliers_missing)
    print("")
    print("")
    #modifier data
    data["nb_outliers"]=is_outliers.sum(axis=1)
    data["nb_missing"] =(data.isna()).sum(axis=1)
    print("")
    print("verifications")
    print("nombres de cellules avec des outliers", data["nb_outliers"].sum())
    print("nombres de lignes avec des outliers",len(data[data["nb_outliers"]>0]))
    print("nombres de cellules avec des missing", data["nb_missing"].sum())
    print("nombres de ligne avec des missing",len(data[data["nb_missing"]>0]))   

    return data

def pie(df, col):
    plt.figure(figsize=(8,8))

    plt.title('Répartition de la variable: {}'.format(col) , size=20)
    wedges, texts, autotexts = plt.pie(df[col].value_counts().values, 
                                       labels = df[col].value_counts().index,
                                       autopct='%1.1f%%', textprops={'fontsize': 16})

    ax = plt.gca()

    ax.legend(wedges, df[col].value_counts().index,
              title="Note",
              loc="center left",
              fontsize=14,bbox_to_anchor=(1, 0, 0.5, 1))
    return None    