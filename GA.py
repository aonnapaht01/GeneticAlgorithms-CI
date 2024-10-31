import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path="wdbc"):
    data = pd.read_csv(file_path, header=None)
    columns = ["ID", "Diagnosis"] + [f"Feature{i}" for i in range(1, 31)]
    data.columns = columns
    data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})

    features = data.iloc[:, 2:].values
    labels = data["Diagnosis"].values
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features, labels


class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.random.randn(output_dim)

    def predict(self, x):
        self.hidden_layer = np.tanh(np.dot(x, self.w1) + self.b1)
        output_layer = 1 / (1 + np.exp(-np.dot(self.hidden_layer, self.w2) - self.b2))
        return output_layer


class GeneticOptimizer:
    def __init__(self, population_size, input_dim, hidden_dim, output_dim):
        self.size = population_size
        self.individuals = [SimpleMLP(input_dim, hidden_dim, output_dim) for _ in range(population_size)]

    def assess_fitness(self, X, y):
        fitness_scores = []
        for model in self.individuals:
            predictions = np.round([model.predict(x) for x in X]).flatten()
            accuracy = np.mean(predictions == y)
            fitness_scores.append(accuracy)
        return fitness_scores

    def select_best(self, fitness_scores):
        selected_indices = np.argsort(fitness_scores)[-self.size // 2:][::-1]
        self.individuals = [self.individuals[i] for i in selected_indices]

    def perform_crossover(self):
        new_generation = []
        for _ in range(len(self.individuals) * 2):
            parent1, parent2 = np.random.choice(self.individuals, 2)
            offspring = SimpleMLP(self.individuals[0].w1.shape[0], self.individuals[0].w1.shape[1], 1)
            offspring.w1 = (parent1.w1 + parent2.w1) / 2
            offspring.w2 = (parent1.w2 + parent2.w2) / 2
            offspring.b1 = (parent1.b1 + parent2.b1) / 2
            offspring.b2 = (parent1.b2 + parent2.b2) / 2
            new_generation.append(offspring)
        self.individuals = new_generation[:self.size]

    def apply_mutation(self, mutation_rate=0.01):
        for model in self.individuals:
            if np.random.rand() < mutation_rate:
                model.w1 += np.random.randn(*model.w1.shape) * mutation_rate
                model.w2 += np.random.randn(*model.w2.shape) * mutation_rate
                model.b1 += np.random.randn(*model.b1.shape) * mutation_rate
                model.b2 += np.random.randn(*model.b2.shape) * mutation_rate


def generate_confusion_matrix(y_true, y_pred): #Create a confusion matrix
    matrix = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

def display_confusion_matrix(cm, labels): #Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    threshold = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()  # Display the confusion matrix


def train_mlp_using_ga(
    features,
    labels,
    hidden_layer_size=5,
    population_count=5,
    num_generations=5,
    k_fold=10,
):

    #Train MLP using Genetic Algorithm with k-fold cross-validation

    fold_size = len(features) // k_fold
    avg_accuracies = []
    gen_accuracies = []
    
    # Specify a custom list of colors for each fold
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'brown', 'gray']

    for fold in range(k_fold):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_train = np.concatenate([features[:start], features[end:]], axis=0)
        y_train = np.concatenate([labels[:start], labels[end:]], axis=0)
        X_test, y_test = features[start:end], labels[start:end]

        ga = GeneticOptimizer(population_count, features.shape[1], hidden_layer_size, 1)
        fold_gen_accuracies = []

        for generation in range(num_generations):
            fitness_scores = ga.assess_fitness(X_train, y_train)
            ga.select_best(fitness_scores)
            ga.perform_crossover()
            ga.apply_mutation()

            best_fit = max(fitness_scores)
            fold_gen_accuracies.append(best_fit)
            print(f"Fold {fold + 1}, Generation {generation + 1}, Best Fitness: {best_fit}")

        best_mlp = ga.individuals[np.argmax(fitness_scores)]
        test_preds = np.round([best_mlp.predict(x) for x in X_test]).flatten()
        accuracy = np.mean(test_preds == y_test)
        avg_accuracies.append(accuracy)

        cm = generate_confusion_matrix(y_test, test_preds.astype(int))
        display_confusion_matrix(cm, labels=["Benign", "Malignant"])

        gen_accuracies.append(fold_gen_accuracies)

    print("Mean Cross-Validation Accuracy:", np.mean(avg_accuracies))

    plt.figure(figsize=(10, 6))
    for i, fold_accuracy in enumerate(gen_accuracies):
        plt.plot(fold_accuracy, marker="x", color=colors[i], label=f"Fold {i + 1}")
    plt.title("Accuracy per Generation Across Folds")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.show()  # Display the accuracy plot

features, labels = load_and_preprocess_data()
train_mlp_using_ga(features, labels)
