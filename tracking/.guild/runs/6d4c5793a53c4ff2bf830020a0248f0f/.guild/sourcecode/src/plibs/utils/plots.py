
from matplotlib.colors import Normalize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [40, 30]
import seaborn as sns
import scipy

import re


def make_confusion_matrix(cf,
						  group_names=None,
						  categories='auto',
						  count=True,
						  percent=True,
						  cbar=True,
						  xyticks=True,
						  xyplotlabels=True,
						  sum_stats=True,
						  figsize=None,
						  cmap='Blues',
						  title=None,
						  percent_relative_to='y_true', # possible values (y_true, y_hat, absolute)
						  normalize = True
						  ):
	'''
	This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
	Arguments
	---------
	cf:            confusion matrix to be passed in
	group_names:   List of strings that represent the labels row by row to be shown in each square.
	categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
	count:         If True, show the raw number in the confusion matrix. Default is True.
	normalize:     If True, show the proportions for each category. Default is True.
	cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
				   Default is True.
	xyticks:       If True, show x and y ticks. Default is True.
	xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
	sum_stats:     If True, display summary statistics below the figure. Default is True.
	figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
	cmap:          Colormap of the values displayed from matplotlib.plt.cm. Default is 'Blues'
				   See http://matplotlib.org/examples/color/colormaps_reference.html
				   
	title:         Title for the heatmap. Default is None.
	'''


	# CODE TO GENERATE TEXT INSIDE EACH SQUARE
	blanks = ['' for i in range(cf.size)]

	if group_names and len(group_names)==cf.size:
		group_labels = ["{}\n".format(value) for value in group_names]
	else:
		group_labels = blanks

	if count:
		group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
	else:
		group_counts = blanks

	cf_percentage = None
	if percent:
		if percent_relative_to=='y_true':
			cf_percentage = []
			for r in cf:
				persum = r.sum()
				per = [ (ri/persum) for ri in r]
				cf_percentage.append(per)
				#group_percentages.extend(per)
			cf_percentage = np.array(cf_percentage)			
			#group_percentages = cf_percentage.flatten()

			group_percentages = ["{0:.2%}".format(value) for value in cf_percentage.flatten()]
		elif percent_relative_to=='y_hat':
			group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
		else: # absolute
			group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
	else:
		group_percentages = blanks

	box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
	box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


	# CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
	if sum_stats:
		#Accuracy is sum of diagonal divided by total observations
		accuracy  = np.trace(cf) / float(np.sum(cf))

		#if it is a binary confusion matrix, show some more stats
		if len(cf)==2:
			#Metrics for Binary Confusion Matrices
			precision = cf[1,1] / sum(cf[:,1])
			recall    = cf[1,1] / sum(cf[1,:])
			f1_score  = 2*precision*recall / (precision + recall)
			stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
				accuracy,precision,recall,f1_score)
		else:
			stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
	else:
		stats_text = ""


	# SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
	if figsize==None:
		#Get default figure size if not set
		figsize = plt.rcParams.get('figure.figsize')

	if xyticks==False:
		#Do not show categories if xyticks is False
		categories=False


	# MAKE THE HEATMAP VISUALIZATION
	plt.figure(figsize=figsize)
	rs = sns.heatmap( (cf_percentage if not (cf_percentage is None) and normalize else cf), annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

	if xyplotlabels:
		plt.ylabel('True label')
		plt.xlabel('Predicted label' + stats_text)
	else:
		plt.xlabel(stats_text)
	
	if title:
		plt.title(title)
		
	return plt

def to_include(value, rules):
	for r in rules:
		if re.search(r, value):
			return True
	return False

def my_plot_confusionmatrix(true_labels, predicted_labels, figsize=(12,10), sort_first=[]):
	
	categs =  sorted(list(set(true_labels)), key=lambda x: '1.' + x if to_include(x, sort_first) else '2.' + x)

	return make_confusion_matrix(
		confusion_matrix(true_labels, 
						 predicted_labels, 
						 labels=categs), 
						  categories = categs,
						  figsize=figsize, cbar=False)


def plot_prec_recall_curve(testy, yhat_probs):
	yhat = yhat_probs[:, 1] if np.array(yhat_probs).ndim > 1 else yhat_probs # keep only true class 
	precision, recall, thresholds = precision_recall_curve(testy, yhat)
	# scoring for each threshold
	scores = (2 * precision * recall) / (precision + recall) # F1
	ix = argmax(scores)
	print('Best Threshold=%f, Score=%.3f' % (thresholds[ix], scores[ix]))
	no_skill = sum(testy) / len(testy)
	plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
	plt.plot(recall, precision, marker='.', label='Classifier')
	plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
	# axis labels
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	# show the plot
	plt.show()
	
def plot_roc_curve(testy, yhat_probs, score_func=(lambda tpr, fpr: np.sqrt(tpr * (1-fpr)))):	
	yhat = yhat_probs[:, 1] if np.array(yhat_probs).ndim > 1 else yhat_probs # keep only true class 
	fpr, tpr, thresholds = roc_curve(testy, yhat)
	# calculate the g-mean for each threshold
	#scores = np.sqrt(tpr * (1-fpr))
	scores = score_func(tpr, fpr)
	
	ix = argmax(scores)
	print('Best Threshold=%f, G-Mean=%.3f, TPR=%.3f, FPR=%.3f' % (thresholds[ix], scores[ix], tpr[ix], fpr[ix] ))
	# plot the roc curve for the model
	plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
	plt.plot(fpr, tpr, marker='.', label='Classifier')
	plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
	# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()
	# show the plot
	plt.show()    
	

def plot_prec_recall_curves(testy, yhat_probs, target='recall', target_reached=(lambda p,r: r > 0.95) ):
	yhat = yhat_probs[:, 1] if np.array(yhat_probs).ndim > 1 else yhat_probs # keep only true class 
	precision, recall, treshold = precision_recall_curve(testy, yhat)
  
	spot = None
	if not (target is None):
		spots=[]
		spots_val=[]
		for p, r, t in zip(precision, recall, treshold):
			if target_reached(p,r):
				spots.append((r if target=='recall' else p, t))
				spots_val.append( p if target=='recall' else r ) # add the other metric
		spot = spots[argmax(spots_val)]	

	# Plot the output.
	plt.plot(treshold, precision[:-1], c ='r', label ='PRECISION')
	plt.plot(treshold, recall[:-1], c ='b', label ='RECALL')
	if spot:
		plt.scatter(spot[1], spot[0], marker='o', color='black', label=f'Best Th: {spot[1]:.3f} [{spot[0]:.3f}]')
	plt.grid()
	plt.legend()
	plt.title('Precision-Recall Curve')  


# ==========================================
