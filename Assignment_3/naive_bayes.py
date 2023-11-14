import numpy as np 

class NaiveBayes:
    def __init__(self):
        self.prior = None 
        self.num_classes = None
        self.vocab_size = None
        self.count_matrix = None
        self.inv_count_matrix = None # To store when certain word is not present in a class
        self.words_per_class = None
        self.class_proportions = None # Prior probabilities of each class
        
    def find_instances(self, X_i: np.ndarray):
        instances = np.zeros((self.vocab_size, 1))
        instances = np.sum(X_i, axis=0)
        return instances
    
    def find_inv_instances(self, X_i: np.ndarray):
        inv_instances = np.zeros((self.vocab_size, 1))
        inv_instances = np.sum(X_i == 0, axis=0) # Find number of times a word is not present in a class
        return inv_instances
        # Number of times a word is not present in a class
        # inv_instances = np.zeros((self.vocab_size, 1))

    
    def find_total_words(self, X: np.ndarray):
        # Find total words in a class
        total_words = np.sum(X)
        return total_words
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.num_classes = np.max(y) + 1 # Zero-indexed
        # self.count_matrix = np.ones((self.num_classes, X.shape[1])) # Laplace smoothing
        self.count_matrix = np.zeros((self.num_classes, X.shape[1]))
        # self.words_per_class = np.ones((self.num_classes, 1)) * X.shape[1] # Laplace smoothing
        self.words_per_class = np.zeros((self.num_classes, 1))
        self.inv_count_matrix = np.zeros((self.num_classes, X.shape[1]))
        self.vocab_size = X.shape[1]
        self.class_proportions = np.zeros((self.num_classes, 1))
        for i in range(self.num_classes):
            self.class_proportions[i] = np.sum(y == i) / y.shape[0]
            y_class = (np.argwhere(y == i))[:, 0]
            X_class = X[y_class] # Features for class i
            instances = self.find_instances(X_class)
            inv_instances = self.find_inv_instances(X_class)
            total_words = self.find_total_words(X_class)
            self.count_matrix[i, :] += instances.reshape((self.count_matrix[i, :].shape))
            self.inv_count_matrix[i, :] += inv_instances.reshape((self.inv_count_matrix[i, :].shape))
            self.words_per_class[i] += total_words
    
    def find_probabilities(self, X: np.ndarray):
        probabilities = np.zeros((X.shape[0], self.num_classes))
        X_absent = X == 0
        for i in range(X.shape[0]):
            log_probs_present = np.sum(np.log((1+self.count_matrix)/(self.vocab_size+self.words_per_class)) * X[i, :], axis=1) # Power of X is multiplied after log
            log_probs_absent = np.sum(np.log(1-(1+self.count_matrix)/(self.vocab_size+self.words_per_class)) * X_absent[i, :], axis=1)
            log_probs = log_probs_present + log_probs_absent + np.log(self.class_proportions).reshape(log_probs_present.shape)
            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / np.sum(probs)
            probabilities[i, :] = probs
        return probabilities

    def predict(self, X: np.ndarray):
        probabilities = self.find_probabilities(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
    
    def evaluate_acc(self, X: np.ndarray, y: np.ndarray):
        predictions = self.predict(X).reshape((y.shape))
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy



        
        


    # def predict(self, X: np.ndarray):

                
if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    text = ["Hello my name is james james",
    "my python notebook haha programming",
    "james trying to hello james create create a big dataset",
    "python programming is fun haha",
    "python programming is fun haha"]
    coun_vect = CountVectorizer()
    # Print which word is assigned which index
    mat = coun_vect.fit_transform(text)
    print(coun_vect.get_feature_names_out())
    X = mat.toarray()
    y = np.array([0, 1, 0, 1, 0])
    nb = NaiveBayes()
    nb.fit(X, y)
    # nb.find_probabilities(X)
    print(nb.predict(X))

        



        
