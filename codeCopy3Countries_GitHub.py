import json
import csv
from parallel_sync import wget
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


#the paragraph below saves the openFDA database on disk
#it uses the download.json file that can be found here: https://api.fda.gov/download.json

with open('download.json','r') as f:
	mesData = json.load(f)
	
mesResults = mesData['results']
mesDrugs = mesResults['drug']
mesEvents = mesDrugs['event']
mesPartitions = mesEvents['partitions']

os.chdir('data')
for partition in mesPartitions:
	url = partition['file']
	print(url)
	file_name = url.split('/')[-1]
	dir_name = url.split('/')[-2]
	if not os.path.isfile(os.path.join(dir_name,file_name[0:-4])):
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)
		
		os.chdir(dir_name)
		monfichier = wget.download(os.getcwd(),url,filenames = file_name)
		monzip = zipfile.ZipFile(file_name,'r')
		monzip.extractall()
		os.remove(file_name)
		os.chdir('..')


os.chdir('..')	
	




#First attempt to read the two fields of interest and store them in lists
#country = []
#adverseEvent = []
#os.chdir('data')
#for partition in mesPartitions:
#	url = partition['file']
#	file_name = url.split('/')[-1]
#	dir_name = url.split('/')[-2]
#	os.chdir(dir_name)
#	with open(file_name[0:-4],'r') as f:
#		mesData = json.load(f)
#		mesResults = mesData['results']
#		for i in range(len(mesResults)):
#			try:
#				country.append(mesResults[i]['primarysource']['reportercountry'])
#				print(country[-1])
#			except (KeyError, TypeError):
#				country.append(float('nan'))				
#			try: 
#				adverseEvent.append(mesResults[i]['patient']['reaction'][0]['reactionmeddrapt'])
#				print(adverseEvent[-1])
#			except (KeyError, TypeError):
#				adverseEvent.append(float('nan'))
#						
#	os.chdir('..')
#
#os.chdir('..')
	





#I download the fields needed to answer question 1. 
#I planned to make use of safetyreportid, safetyreportversion, companynumb, but I did not have time in the end.
#It needs to be done in a more finalised version
#There are several reaction (reactionmeddrapt) fields per record, and we create one separate line for each in the output file

all_files = []
os.chdir('data')

countryCodes = pd.read_csv('CountryCodes.csv')
countryCodes.columns = ["country","code","2","3"]
np.where(countryCodes.index == 'DE')[0][0]

for partition in mesPartitions:
	url = partition['file']
	file_name = url.split('/')[-1]
	dir_name = url.split('/')[-2]
	all_files.append(os.path.join(dir_name,file_name[0:-4]))

all_fields = ["reportercountry","reactionmeddrapt","safetyreportid","safetyreportversion","companynumb"]
with open("Country_AdverseEvent_version2.csv", "w") as output_file:
	writer = csv.DictWriter(output_file,all_fields)
	writer.writeheader()
	for filename in all_files:
	#for filename in [all_files[0]]:
		#filename = '2016q4/drug-event-0022-of-0023.json'
		print(filename)
		with open(filename,"r") as in_file:
			mesData = json.load(in_file)
			mesResults = mesData['results']
			for i in range(len(mesResults)):
			#for i in range(1):
				try:
					country = mesResults[i]['primarysource']['reportercountry']
				except (KeyError, TypeError):
					country = 0				
				try: 
					adverseEvent = []
					for lind in range(len(mesResults[i]['patient']['reaction'])):
						adverseEvent.append(mesResults[i]['patient']['reaction'][lind]['reactionmeddrapt'])
				except (KeyError, TypeError):
					adverseEvent = 0
				try: 
					safetyreportid = mesResults[i]['safetyreportid']
				except (KeyError, TypeError):
					safetyreportid = '0'
				try: 
					safetyreportversion = mesResults[i]['safetyreportversion']
				except (KeyError, TypeError):
					safetyreportversion = '0'
				try: 
					companynumb = mesResults[i]['companynumb']
				except (KeyError, TypeError):
					companynumb = '0'
				#									
				aecrire = dict()
				if country != 0:
					if adverseEvent != 0:
						temp = countryCodes[countryCodes["code"] == country]["code"]
						if len(temp) > 0:
							res = countryCodes[countryCodes["code"] == country]["country"]
							country = res.values[0]
						if safetyreportversion == '0':
							res = safetyreportid.split('-')
							if len(res) == 2:
								aecrire["safetyreportid"] = res[0]
								aecrire["safetyreportversion"] = res[1]
						if safetyreportversion != '0':
							aecrire["safetyreportid"] = safetyreportid
							aecrire["safetyreportversion"] = safetyreportversion
						aecrire["reportercountry"] = country
						aecrire["companynumb"] = companynumb
						for lind in range(len(mesResults[i]['patient']['reaction'])):
							aecrire["reactionmeddrapt"] = adverseEvent[lind]
							writer.writerow(aecrire)
								


os.chdir('..')



#Now, I need to analyse the data and create relevant displays
#I load the data
os.chdir('data')
aregarder = pd.read_csv('Country_AdverseEvent_version2.csv')
aregarder.columns = ["reportercountry","reactionmeddrapt","safetyreportid","safetyreportversion","companynumb"]
aregarder["reportercountry"] = aregarder["reportercountry"].str.upper()
aregarder = aregarder[aregarder["reportercountry"] != 'COUNTRY NOT SPECIFIED']
aregarder["reportercountry"] = aregarder["reportercountry"].str.replace(r"UNITED STATES OF AMERICA","UNITED STATES")
os.chdir('..')

#I isolate the top 20 countries in the global table
topCountries = aregarder.groupby('reportercountry').size().nlargest(20)
topCountries

#I create a bar plot showing the most frequent reactions for the top 5 countries
def plot_important_words(scores, words, name):
    y_pos = np.arange(len(words))
    pairs = [(a,b) for a,b in zip(words, scores)]
    pairs = sorted(pairs, key=lambda x: x[1])
    words = [a[0] for a in pairs]
    scores = [a[1] for a in pairs]
    fig = plt.figure(figsize=(10, 10))  
    plt.barh(y_pos,scores, align='center', alpha=0.5)
    plt.title(name, fontsize=20)
    plt.yticks(y_pos, words, fontsize=8)
    plt.xlabel('Number of reports', fontsize=20)
    plt.show()


def lanceLePlot(numCountry):
	aplotter = aregarder[aregarder["reportercountry"] == topCountries.index[numCountry]]
	aplotter["reactionmeddrapt"] = aplotter["reactionmeddrapt"].str.lower()
	topReactions = aplotter.groupby('reactionmeddrapt').size().nlargest(20)
	scores = []
	words = []
	for i in range(10):
		scores.append(topReactions[i])
		words.append(topReactions.index[i])
	plot_important_words(scores, words, 'Reactions in '+topCountries.index[numCountry])


lanceLePlot(0)
lanceLePlot(1)
lanceLePlot(2)
lanceLePlot(3)
lanceLePlot(4)





#plots a map of the rank of the death outcome. Known caveats discussed in ppt, to be improved. Also need to add a colorbar.
def maWorldMap(monkeyword,montitle):
	shapename = 'admin_0_countries'
	countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)
	#
	# some nice colors
	earth_colors = ['orangered','tomato','coral','darkorange','orange','lightyellow','black']
	#
	ax = plt.axes(projection=ccrs.PlateCarree())
	for country in shpreader.Reader(countries_shp).records():
		print(country.attributes['NAME_LONG'])
		aanalyser = aregarder[aregarder["reportercountry"] == country.attributes['NAME_LONG'].upper()]
		#aanalyser = aregarder[aregarder["reportercountry"] == "FRANCE"]
		print(len(aanalyser))
		lindiceici = 6
		if len(aanalyser) > 0:
			topReactions = aanalyser.groupby('reactionmeddrapt').size().nlargest(20)
			try:
				lindiceici = np.where(topReactions.index.str.upper() == monkeyword)[0][0]
			except IndexError:
				lindiceici = 6
		if lindiceici > 6:
			lindiceici = 6
		print(lindiceici)
		ax.add_geometries(country.geometry, ccrs.PlateCarree(),facecolor=earth_colors[lindiceici],label=country.attributes['NAME_LONG'])
	#
	monc1 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[0])
	monc2 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[1])
	monc3 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[2])
	monc4 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[3])
	monc5 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[4])
	monc6 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[5])
	monc7 = Line2D([0], [0], linestyle = 'none', marker = 's', markersize = 10, markerfacecolor = earth_colors[6])
	plt.legend((monc1,monc2,monc3,monc4,monc5,monc6,monc7),('1','2','3','4','5','6','7'),numpoints = 1, loc = 'best')
	plt.title(montitle)
	#
	plt.show()

maWorldMap('DEATH','Rank of death occurence')
maWorldMap('DRUG INEFFECTIVE','Rank of drug_ineffective occurence')



# For the top 5 reactions, let's create a bar plot that shows the number of countries for which this reaction is in the top 5
topReactions = aregarder.groupby('reactionmeddrapt').size().nlargest(5)

compteur = topReactions*0
for reaction in topReactions.index:
	for country in shpreader.Reader(countries_shp).records():
		print(country.attributes['NAME_LONG'],reaction)
		aanalyser = aregarder[aregarder["reportercountry"] == country.attributes['NAME_LONG'].upper()]
		#aanalyser = aregarder[aregarder["reportercountry"] == "FR"]
		topReactionsCountry = aanalyser.groupby('reactionmeddrapt').size().nlargest(5)
		lindiceici = 10
		if len(topReactionsCountry) > 0:
			try:
				lindiceici = np.where(topReactionsCountry.index == reaction)[0][0]
			except IndexError:
				lindiceici = 10
		if lindiceici < 10:
			compteur[reaction] += 1			

lecompteur = [chiffre for chiffre in compteur]
lemot = [mot for mot in topReactions.index]
plot_important_words(lecompteur, lemot, 'Country count for reaction in top 5')




###################################################################################
#machine learning part below
###################################################################################


#############################################################################################
#############################################################################################
#Let's use the workflow of Emmanuel Ameisen, which I find amazing
#I also copied some explanations from Emmanuel Ameisen's code 
#https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
#############################################################################################
#############################################################################################

##########################################################################################################################################################################################################################
#The question that I am trying to answer might not have a lot of impact in terms of business, better questions would probably make more sense to focus on, but it is linked to question 1
#Question: if the country name is missing from a number of records, can we find it from the list of reactions?
#For reasons of computational time, I narrowed down the question to: if the country name is missing from a number of records, and we know it is either France or UK, or Japan, can we find it from the list of reactions?
#A quick litterature search also shows that not much machine learning work has been done on the openFDA database, while machine learning is highly succesful on other types of data in the pharmacy industry
##########################################################################################################################################################################################################################

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression

import random
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import gensim

from nltk.tokenize import RegexpTokenizer

from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

import numpy as np
import itertools
from sklearn.metrics import confusion_matrix



#After a few tries, I decided to keep the data from three countries only: France, Japan, and England
#Otherwise, my old computer did not want to run

aregarder = pd.read_csv('Country_AdverseEvent.csv')
aregarder.columns = ["reportercountry","reactionmeddrapt","safetyreportid","safetyreportversion","companynumb"]
aregarder["reactionmeddrapt"] = aregarder["reactionmeddrapt"].str.upper()
aregarder["reportercountry"] = aregarder["reportercountry"].str.replace(r"UNITED STATES OF AMERICA","UNITED STATES")

topCountry = aregarder.groupby('reportercountry').size().nlargest(5)


#dataUSA = aregarder[aregarder["reportercountry"] == "UNITED STATES"] 
dataFrance = aregarder[aregarder["reportercountry"].str.upper() == "FRANCE"] 
dataUK = aregarder[aregarder["reportercountry"].str.upper() == "UNITED KINGDOM"] 
dataJapan = aregarder[aregarder["reportercountry"].str.upper() == "JAPAN"] 
#dataCanada = aregarder[aregarder["reportercountry"].str.upper() == "CANADA"] 


#Here below, we select data from UK, France, and Japan. No more for computational time reasons.
#We group the reactionmedrapt by groups of 10 in order to create some context, if we have too few words in one 'sentence', such as one word only, the machine learning algorithm cannot work
stepN = 10
sizeSubSample = -1

	
listtemp = []
for k,v in dataFrance.iteritems():
	listtemp.append([k,v])

#
listtemp2 = listtemp[1][1][0:sizeSubSample]
listtemp3France = [' '.join(listtemp2[i:i+stepN]) for i in range(1,len(listtemp2),stepN)]


listtemp = []
for k,v in dataUK.iteritems():
	listtemp.append([k,v])

#
listtemp2 = listtemp[1][1][0:sizeSubSample]
listtemp3UK = [' '.join(listtemp2[i:i+stepN]) for i in range(1,len(listtemp2),stepN)]


listtemp = []
for k,v in dataJapan.iteritems():
	listtemp.append([k,v])

#
listtemp2 = listtemp[1][1][0:sizeSubSample]
listtemp3Japan = [' '.join(listtemp2[i:i+stepN]) for i in range(1,len(listtemp2),stepN)]


#Here, below, we put the data in the format requested by Emmanuel Ameisen
list_corpus = []
list_corpus.extend(listtemp3France)
list_corpus.extend(listtemp3UK)
list_corpus.extend(listtemp3Japan)


list_labels_France = [2 for i in range(len(listtemp3France))]
list_labels_UK = [3 for i in range(len(listtemp3UK))]
list_labels_Japan = [4 for i in range(len(listtemp3Japan))]

list_labels = []
list_labels.extend(list_labels_France)
list_labels.extend(list_labels_UK)
list_labels.extend(list_labels_Japan)

#Then, we run the classical train_test_split and calculate the embedding
X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)


#First plot, as Emmanuel Ameisen explains: Now that we've created embeddings, let's visualize them and see if we can identify some structure. 
#In a perfect world, our embeddings would be so distinct that are two classes would be perfectly separated.

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
	lsa = TruncatedSVD(n_components=2)
	lsa.fit(test_data)
	lsa_scores = lsa.transform(test_data)
	color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
	color_column = [color_mapper[label] for label in test_labels]
	#colors = ['orange','blue','red','green','purple']
	colors = ['blue','red','green']
	if plot:
		plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
		#orange_patch = mpatches.Patch(color='orange', label='USA')
		blue_patch = mpatches.Patch(color='blue', label='France')
		red_patch = mpatches.Patch(color='red', label='UK')
		green_patch = mpatches.Patch(color='green', label='Japan')
		#purple_patch = mpatches.Patch(color='purple', label='Germany')
		#plt.legend(handles=[orange_patch, blue_patch, red_patch, green_patch, purple_patch], prop={'size': 10})
		plt.legend(handles=[blue_patch, red_patch, green_patch], prop={'size': 10})


fig = plt.figure(figsize=(10, 10))          
plot_LSA(X_train_counts, y_train)
plt.show()
#Obviously, we are not in a perfect world... 

#Fitting LogisticRegression classifier, as Emmanuel Ameisen recommends.
#After Checking scikit-Learn tutorial, 'working with text data', as well as other pages, we might want to try multinomialNB, SVM, Decision trees, random forest, Gradient boosted trees, and each with GridSearchCV and compare the metrics
#Or even something like autokeras or TPOT  
clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)

#Evaluation
def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
#accuracy = 0.821, precision = 0.821, recall = 0.821, f1 = 0.821
#Quite good! Why are they so close to each other, though? Needs to be checked...

#Nice plot of the confusion matrix from Emmanuel Ameisen
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    #
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    #
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    #
    return plt

cm = confusion_matrix(y_test, y_predicted_counts)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['France','UK','Japan'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)

#bar plot of the most important words
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    #
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

importance = get_most_important_features(count_vectorizer, clf, 10)

#Some modifications were needed below, which should be checked carefully
def plot_important_words(France_scores, France_words, UK_scores, UK_words, Japan_scores, Japan_words, name):
    y_pos = np.arange(len(France_words))
    #top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    #top_pairs = sorted(top_pairs, key=lambda x: x[1])
    #
    #bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    #bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    #
    #top_words = [a[0] for a in top_pairs]
    #top_scores = [a[1] for a in top_pairs]
    #
    #bottom_words = [a[0] for a in bottom_pairs]
    #bottom_scores = [a[1] for a in bottom_pairs]
    #
    fig = plt.figure(figsize=(15, 10))  
    #
    plt.subplot(131)
    plt.barh(y_pos,France_scores, align='center', alpha=0.5)
    plt.title('France', fontsize=20)
    plt.yticks(y_pos, France_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    #
    plt.subplot(132)
    plt.barh(y_pos,UK_scores, align='center', alpha=0.5)
    plt.title('UK', fontsize=20)
    plt.yticks(y_pos, UK_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    #
    plt.subplot(133)
    plt.barh(y_pos,Japan_scores, align='center', alpha=0.5)
    plt.title('Japan', fontsize=20)
    plt.yticks(y_pos, Japan_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    #
    plt.subplots_adjust(wspace=0.8)
    plt.show()

France_scores = [a[0] for a in importance[0]['tops']]
France_words = [a[1] for a in importance[0]['tops']]
UK_scores = [a[0] for a in importance[1]['tops']]
UK_words = [a[1] for a in importance[1]['tops']]
Japan_scores = [a[0] for a in importance[2]['tops']]
Japan_words = [a[1] for a in importance[2]['tops']]

plot_important_words(France_scores, France_words, UK_scores, UK_words, Japan_scores, Japan_words, "Most important words for relevance")


#Same as above, with TfidfVectorizer instead of CountVectorizer
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    #
    train = tfidf_vectorizer.fit_transform(data)
    #
    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

fig = plt.figure(figsize=(10, 10))          
plot_LSA(X_train_tfidf, y_train)
plt.show()

clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 
                                                                       recall_tfidf, f1_tfidf))
#accuracy = 0.822, precision = 0.822, recall = 0.822, f1 = 0.822
#slightly better than before, the improvement is comparable to the improvement obtained by Emmanuel Ameisen with his dataset

cm2 = confusion_matrix(y_test, y_predicted_tfidf)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm2, classes=['France','UK','Japan'], normalize=False, title='Confusion matrix')
plt.show()
print("TFIDF confusion matrix")
print(cm2)
print("BoW confusion matrix")
print(cm)

importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)

France_scores = [a[0] for a in importance[0]['tops']]
France_words = [a[1] for a in importance[0]['tops']]
UK_scores = [a[0] for a in importance[1]['tops']]
UK_words = [a[1] for a in importance[1]['tops']]
Japan_scores = [a[0] for a in importance[2]['tops']]
Japan_words = [a[1] for a in importance[2]['tops']]

plot_important_words(France_scores, France_words, UK_scores, UK_words, Japan_scores, Japan_words, "Most important words for relevance")

###################################################################################################################################################################################################
#same as above with word2vec. 
#This part does not make sense here because Word2Vec is defined to predict the next word of a sentence, and is calibrated on wikipedia
#However, I applied it for completeness
#We could maybe calibrate a model similar to word2vec on our present dataset.
#The computational times that I found in Mikolov et al., 2013 look discouraging (2daysx140CPU). Certainly not achievable on my personal machine, maybe achievable on a cluster?
####################################################################################################################################################################################################

#File below can be downloaded here: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
#line below take a lot of time, once model in memory, comment for further runs
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


tokenizer = RegexpTokenizer(r'\w+')

clean_questions = pd.DataFrame(np.array(list_corpus), columns = ["text"])
clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
embeddings = get_word2vec_embeddings(word2vec, clean_questions)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, 
	                                                                                        test_size=0.2, random_state=40)

fig = plt.figure(figsize=(10, 10))          
plot_LSA(embeddings, list_labels)
plt.show()

clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', random_state=40)
clf_w2v.fit(X_train_word2vec, y_train_word2vec)
y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 
                                                                       recall_word2vec, f1_word2vec))
#accuracy = 0.644, precision = 0.652, recall = 0.644, f1 = 0.645
#No surprise, the performances are dreadful. The model should really be recalibrated for our purposes!


#

cm_w2v = confusion_matrix(y_test_word2vec, y_predicted_word2vec)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['France','UK'], normalize=False, title='Confusion matrix')
plt.show()
print("Word2Vec confusion matrix")
print(cm_w2v)
print("TFIDF confusion matrix")
print(cm2)
print("BoW confusion matrix")
print(cm)



X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                random_state=40)
vector_store = word2vec
def word2vec_pipeline(examples):
    global vector_store
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_list = []
    for example in examples:
        example_tokens = tokenizer.tokenize(example)
        vectorized_example = get_average_word2vec(example_tokens, vector_store, generate_missing=False, k=300)
        tokenized_list.append(vectorized_example)
    return clf_w2v.predict_proba(tokenized_list)

c = make_pipeline(count_vectorizer, clf)


#Here below, we do not see 'exp', it is the same in the original notebook, which means that nothing is explained by lime
#This should be discussed with the author
#The explanation part, I like less
def explain_one_instance(instance, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(instance, word2vec_pipeline, num_features=6)
    return exp

def visualize_one_exp(features, labels, index, class_names = ["France","UK"]):
    exp = explain_one_instance(features[index], class_names = class_names)
    print('Index: %d' % index)
    print('True class: %s' % class_names[labels[index]-2])
    exp.show_in_notebook(text=True)

visualize_one_exp(X_test_data, y_test_data, 1)
visualize_one_exp(X_test_data, y_test_data, 11)



random.seed(40)

def get_statistical_explanation(test_set, sample_size, word2vec_pipeline, label_dict):
    sample_sentences = random.sample(test_set, sample_size)
    explainer = LimeTextExplainer()
    #
    labels_to_sentences = defaultdict(list)
    contributors = defaultdict(dict)
    #
    # First, find contributing words to each class
    for sentence in sample_sentences:
        probabilities = word2vec_pipeline([sentence])
        curr_label = probabilities[0].argmax()
        labels_to_sentences[curr_label].append(sentence)
        exp = explainer.explain_instance(sentence, word2vec_pipeline, num_features=6, labels=[curr_label])
        listed_explanation = exp.as_list(label=curr_label)
        
        for word,contributing_weight in listed_explanation:
            if word in contributors[curr_label]:
                contributors[curr_label][word].append(contributing_weight)
            else:
                contributors[curr_label][word] = [contributing_weight]    
    #
    # average each word's contribution to a class, and sort them by impact
    average_contributions = {}
    sorted_contributions = {}
    for label,lexica in contributors.items():
        curr_label = label
        curr_lexica = lexica
        average_contributions[curr_label] = pd.Series(index=curr_lexica.keys())
        for word,scores in curr_lexica.items():
            average_contributions[curr_label].loc[word] = np.sum(np.array(scores))/sample_size
        detractors = average_contributions[curr_label].sort_values()
        supporters = average_contributions[curr_label].sort_values(ascending=False)
        sorted_contributions[label_dict[curr_label]] = {
            'detractors':detractors,
             'supporters': supporters
        }
    return sorted_contributions

label_to_text = {
    0: 'France',
    1: 'UK',
    2: 'Japan',
}
sorted_contributions = get_statistical_explanation(X_test_data, 100, word2vec_pipeline, label_to_text)

France_words = sorted_contributions['France']['supporters'][:10].index.tolist()
France_scores = sorted_contributions['France']['supporters'][:10].tolist()
UK_words = sorted_contributions['UK']['supporters'][:10].index.tolist()
UK_scores = sorted_contributions['UK']['supporters'][:10].tolist()
Japan_words = sorted_contributions['Japan']['supporters'][:10].index.tolist()
Japan_scores = sorted_contributions['Japan']['supporters'][:10].tolist()

France_words = France_words[::-1]
France_scores = France_scores[::-1]
UK_words = UK_words[::-1]
UK_scores = UK_scores[::-1]
Japan_words = Japan_words[::-1]
Japan_scores = Japan_scores[::-1]

plot_important_words(France_scores, France_words, UK_scores, UK_words, Japan_scores, Japan_words, "Most important words for relevance")




