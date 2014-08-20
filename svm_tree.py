from sklearn import svm
from Bio import SeqIO
import numpy as np


class SVMTree:
	def __init__(self):
		self.table = {'A' : [2**0.5/4,0.5,0],
			'C' : [2**0.5/4,-0.5,0],
			'G': [-(2**0.5/4),0,0.5],
			'T' : [-(2**0.5/4),0,-0.5]}
		self.alphabet = {'A' : 0,'C' : 1,'G' : 2,'T' : 3}
		self.parse_rawdata()

		self.sequences = self.get_sequences(self.rawData)
		self.labels = self.get_labels(self.rawData)
		self.variables = self.variable_sites()
		self.vectors = self.fasta_to_vectors(self.sequences)
		

	def parse_rawdata(self):
		raw = list(SeqIO.parse("all_yers.fas","fasta"))
		self.rawData = [record for index,record in enumerate(raw)]		

	def get_sequences(self,data):
		return np.array([str(record.seq).upper() for record in data])

	def get_labels(self,data):		
		ids = [record.id for record in data]
		return np.array([id[:id.rfind('_')] for id in ids])
		
	def str_to_vector(self,sequence):
		vector = []
		for bp in np.array([char for char in sequence])[self.variables]:
				vector += self.table[bp]
		return np.array(vector)

	
	def fasta_to_vectors(self,data):
		vectors = [[] for i in range(len(data))]
		for index, sequence in enumerate(data):
			for bp in np.array([char for char in sequence])[self.variables]:
				vectors[index] += self.table[bp]
		return np.array(vectors)

	
	def variable_sites(self):	
		sites = np.zeros((len(self.sequences[0])))
		for i in range(len(self.sequences[0])):
			counts = np.zeros((len(self.alphabet)))
			for j in range(len(self.sequences)):
				counts[self.alphabet[self.sequences[j][i]]] += 1
			sites[i] = max(counts)/len(self.sequences)
		variables = np.array([i for i,value in enumerate(sites) if value < 0.8])
		return variables


	def training_sets(self,size,number):
		return np.array([vector for index,vector in enumerate(self.vectors) if index % size != number]),np.array([label for index,label in enumerate(self.labels) if index % size != number])

	def test_sets(self,size,number):
		return np.array([vector for index,vector in enumerate(self.vectors) if index % size == number]),np.array([label for index,label in enumerate(self.labels) if index % size == number])

	def train(self):
		X = self.vectors
		y = self.labels
		
		self.svc = svm.SVC()
		self.svc.fit(X,y)

	def test(self):
		return self.svc.score(self.test_vectors,self.test_labels)

	def predict(self,str):
		vec = self.str_to_vector(str)
		return self.svc.predict(vec)

	def crossvalidate(self,size,C_,gamma_):
		svc = svm.SVC(C = C_, gamma = gamma_)		
		score = 0.0
		for i in range(size):
			X,y = self.training_sets(size,i)
			Xt,yt = self.test_sets(size,i)
			svc.fit(X,y)
			score += svc.score(Xt,yt)
		return score/size

	def grid_search(self,n):
		max_score = 0.0
		max_C = 0
		max_gamma = 0
		for i in range(-4,15):
			for j in range(15,-4,-1):
				gamma = 2**i
				C = 2**j
				score = self.crossvalidate(n,C,gamma)
				if score > max_score:
					max_score = score
					max_C = C
					max_gamma = gamma
		return max_score,max_C,max_gamma		