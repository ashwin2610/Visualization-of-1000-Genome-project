import numpy as np
import pandas as pd
import csv

from bokeh.models import ColumnDataSource, MultiSelect, Button, HoverTool
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from copy import deepcopy
from webbrowser import open_new_tab


import proj3D

#overwriting 3D.html
f = open('3D.html','w')
f.close()


#selecting data file
def get_data():
    root=Tk()
    root.withdraw()

    filename = askopenfilename(parent=root)

    root.destroy()

    with open(filename) as f:
        reader = csv.reader(f, delimiter = ',')
        data = [(column[1:]) for column in reader]

    return data



#selecting classes from data file
def get_classes(classes_selected):
    class_data = [entry for entry in data if entry[1] in classes_selected]
    class_data = shuffle(class_data)

    global X,y,id


    #Indexing not completely generalized
    id = [col[0] for col in class_data]
    X = [col[2::] for col in class_data]
    y = [col[1] for col in class_data]




#runs pca on selected class
def pca(X):
    X_transformed = PCA(n_components=3).fit_transform(X)
    return X_transformed[:,0], X_transformed[:,1], X_transformed[:,2]


#runs lda on 2 selected classes
def lda(X, y):
    X = np.reshape(X, (np.shape(X)[0],np.shape(X)[1]))
    X = X.astype(np.float64)
    y = np.reshape(y, np.shape(y)[0])
    logit = LogisticRegression(C=10**10, max_iter = 1000)
    logit.fit(X, y)

    U = logit.coef_
    u = np.reshape(U[0], (10,1))
    u =  u/np.sqrt(np.sum(u**2))
    axis1 = np.dot(X,u)[:,0]

    P = X - np.matmul(np.dot(X,u), np.transpose(u))


    logit = LogisticRegression(C=10**10, max_iter = 1000)
    logit.fit(P, y)

    V = logit.coef_
    v = np.reshape(V[0], (10,1))
    v =  v/np.sqrt(np.sum(v**2))
    axis2 = np.dot(P,v)[:,0]

    Q = P - np.matmul(np.dot(P,v), np.transpose(v))


    logit = LogisticRegression(C=10**10, max_iter = 1000)
    logit.fit(Q, y)

    W = logit.coef_
    w = np.reshape(W[0], (10,1))
    w =  w/np.sqrt(np.sum(w**2))
    axis3 = np.dot(Q,w)[:,0]


    return axis1, axis2, axis3



#runs t-SNE on selected classes
def t_sne(X,n):
    X_transformed = TSNE(n_components=n, init='pca', learning_rate=500).fit_transform(X)
    if n == 2:
        return X_transformed[:,0], X_transformed[:,1]
    else:
        return X_transformed[:,0], X_transformed[:,1], X_transformed[:,2]




#updates plot with data from new class(es) when select button is pressed
def update():
    if "Select All" in multiselect.value:
        multiselect.value = opt

    classes_selected = multiselect.value


    get_classes(classes_selected)


    if len(classes_selected) == 1:
        axis1, axis2, axis3 = pca(X)
        title = "PCA of " + classes_selected[0]

    elif len(classes_selected) == 2:
        axis1, axis2, axis3 = lda(X,y)
        title = "LogisticLDA of " + classes_selected[0] + " and " + classes_selected[1]

    else:
        axis1, axis2 = t_sne(X,2)
        axis3 = np.zeros((len(axis1),1))

        if len(classes_selected) < 5:
            title = "t-SNE of"
            for i in  range(len(classes_selected)):
                title += " " + classes_selected[i]
                if i < len(classes_selected) - 1:
                    title += ", "

        else:
            title = "t-SNE of " + str(len(classes_selected)) + " classes"


    #Updating source
    source.data['axis1'], source.data['axis2'], source.data['axis3'] = axis1, axis2, axis3
    source.data['colors'] = [coloring[i] for i in y]
    source.data['X'], source.data['y'], source.data['id'] = X,y,id


    fig.title.text = title

    global original
    original = deepcopy(dict(source.data))



#Selected data points form lasso selection are made available
def lasso(attr, old, new):
    inds = np.array(new['1d']['indices'])

    global X_selected, y_selected, id_selected
    X_selected = [source.data['X'][i] for i in inds]
    y_selected = [source.data['y'][i] for i in inds]
    id_selected = [source.data['id'][i] for i in inds]



#Selected data points are focussed on by re-running necessary algorithm
def focus_selected():
    classes_focus = []
    for i in y_selected:
        if i not in classes_focus:
            classes_focus.append(i)

    multiselect.value = classes_focus
    no_classes = len(classes_focus)

    if no_classes == 1:
        axis1, axis2, axis3 = pca(X_selected)
        title = "PCA of " + classes_focus[0]

    elif no_classes == 2:
        axis1, axis2, axis3 = lda(X_selected,y_selected)
        title = "LogisticLDA of " + classes_focus[0] + " and " + classes_focus[1]

    else:
        axis1, axis2 = t_sne(X_selected,2)
        axis3 = np.zeros((len(axis1),1))

        if len(classes_focus) < 5:
            title = "t-SNE of"
            for i in  range(len(classes_focus)):
                title += " " + classes_focus[i]
                if i < len(classes_focus) - 1:
                    title += ", "

        else:
            title = "t-SNE of " + str(len(classes_focus)) + " classes"


    colors = [coloring[i] for i in y_selected]

    source.data['axis1'], source.data['axis2'], source.data['axis3'], source.data['colors'] = axis1, axis2, axis3, colors
    source.data['X'], source.data['y'], source.data['id'] = X_selected,y_selected,id_selected


    fig.title.text = title



#resets points to original selection on pressing reset button
def reset_points():
    source.data = deepcopy(original)



#gives 3D view when view 3D button is pressed
def project():
    #To check if there is more than 2 classes, since t-SNE has to be run separately for 2D and 3D
    if fig.title.text[0] == 't':
        ax1, ax2, ax3 = t_sne(source.data['X'], 3)
        source3D = ColumnDataSource(data=dict(x=ax1, y=ax2, z=ax3, country=source.data['y'], color=source.data['colors']))

    else:
        source3D = ColumnDataSource(data=dict(x=source.data['axis1'], y=source.data['axis2'], z=source.data['axis3'], country=source.data['y'], color=source.data['colors']))

    #Creates 3D plot using proj3D.py
    surface = proj3D.Surface3d(x="x", y="y", z="z", country="country", color="color", data_source=source3D)


    #html data created, then written into 3D.html
    html = file_html(surface, CDN, "3D view")

    f = open("3D.html", "w")
    f.write(html)
    f.close()


    open_new_tab("3D.html")



#Used to download data of selected points/classes, along with 3 axes of projection
#Specific only to this dataset
def download():
    if fig.title.text[0] == 't':
        ax1, ax2, ax3 = t_sne(source.data['X'], 3)

    else:
        ax1, ax2, ax3 = source.data['axis1'], source.data['axis2'], source.data['axis3']


    table = {'IID': source.data['id'],
             'POPULATION': source.data['y'],
             'PC1' : np.asarray(source.data['X'])[::,0],
             'PC2':np.asarray(source.data['X'])[::,1],
             'PC3': np.asarray(source.data['X'])[::,2],
             'PC4': np.asarray(source.data['X'])[::,3],
             'PC5': np.asarray(source.data['X'])[::,4],
             'PC6': np.asarray(source.data['X'])[::,5],
             'PC7': np.asarray(source.data['X'])[::,6],
             'PC8': np.asarray(source.data['X'])[::,7],
             'PC9': np.asarray(source.data['X'])[::,8],
             'PC10': np.asarray(source.data['X'])[::,9],
             'axis1': ax1,
             'axis2': ax2,
             'axis3': ax3
             }
    df = pd.DataFrame.from_dict(data=table, orient='columns')
    myFormats = [
    ('CSV','*.csv'),
    ]

    root=Tk()
    root.withdraw()
    fileName =asksaveasfilename(parent=root,filetypes=myFormats ,title="Save the file as...")
    df.to_csv(fileName, sep=',')








data = get_data()



#options for classes to be selected
opt = []
for i in data[1::]:
    if i[1] not in opt:
        opt.append(i[1])

palette = ['#997378', '#d9368d', '#ff00ee', '#633366', '#311a33', '#cbace6',
           '#7340ff', '#000e66', '#006dcc', '#4d5a66', '#00c2f2', '#39e6ac',
           '#408062', '#1fe600', '#e1ffbf', '#8da629', '#303300', '#ffee00',
           '#996600', '#332d26', '#cc5200', '#ffd9bf', '#e50000', '#401010',
           '#e57373', 'f2ceb6']

coloring = {opt[i]:palette[i] for i in range(len(opt))}



#Widgets defined
multiselect = MultiSelect(title="Select Classes", options=["Select All"]+opt, size=10, width=300)

button = Button(label="Select", button_type="success", width=300)
focus = Button(label="Focus", button_type="primary", width=100)
reset = Button(label="Reset", button_type="danger", width=100)
view3d = Button(label="View 3D", button_type="warning", width=100)
download_file = Button(label='Download', button_type="success", width=100)



#Any changes to be reflected dynamically in the plot must be made to 'source'
source = ColumnDataSource(data={'axis1':[], 'axis2':[], 'axis3':[], 'colors':[], 'X':[], 'y':[], 'id':[]})




#Creating empty plot
fig = figure(title="Empty plot", tools="pan,wheel_zoom,lasso_select")
p = fig.circle(x='axis1', y='axis2', fill_color='colors', source=source,
               fill_alpha=0.8, size=10, line_alpha=0,
               selection_alpha=1, selection_fill_color='colors',)
fig.add_tools(HoverTool(tooltips=[('Population', '@y')]))


p.nonselection_glyph = None

plot=column(row(fig,column(multiselect,button)),row(focus,reset,view3d,download_file))


#functions called on pressing each button
button.on_click(update)

focus.on_click(focus_selected)
reset.on_click(reset_points)
view3d.on_click(project)

download_file.on_click(download)


#lasso function is called whenever points are selected
source.on_change('selected', lasso)


curdoc().add_root(plot)
