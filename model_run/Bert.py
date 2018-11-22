import pickle
import ujson, sys, numpy

class Bert:
    def __init__(self, bertfile):
        self.filename = bertfile
        self.sentences = []
        self.tokens = []
        self.vector_buffer = []
        self.token_vectors = numpy.array()


    def load(self):
        with open(self.filename) as infile:
            for line in infile:
                self.sentences.append(ujson.loads(line))

        for sentence in self.sentences:
            for token_data in sentence['features']:
                self.tokens.append(token_data['token'])
                self.vector_buffer.append(
                    numpy.array(token_data['layers'][0]['values'])
                )
        
        self.token_vectors = numpy.array(self.vector_buffer)


    def indices_of(self, s):
        return [ i for i, w in enumerate(self.tokens) if w == s ]

    
    def nearest(self, i):
        sorted_tokens = sorted(
            zip(
                self.token_vectors.dot(self.token_vectors[i,:]), 
                self.tokens
            ), 
            reverse=True
        )
        return [s for x, s in sorted_tokens[:100]]