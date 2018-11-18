import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from timeit import default_timer as timer
import pickle
import multiprocessing
from itertools import product
from gensim.models.doc2vec import Doc2Vec, FAST_VERSION
from SVMclassifier import *

assert FAST_VERSION > -1
################## DATA PROCESSING #########################

saved_models = os.path.join(root_dir, 'models')
if not os.path.exists(saved_models):
    os.makedirs(saved_models)

imdb_data_dir = os.path.join(root_dir, 'aclImdb')


def get_imdb_docs():
    data_dirs = [os.path.join(imdb_data_dir, 'test', 'pos')]
    data_dirs.append(os.path.join(imdb_data_dir, 'test', 'neg'))
    data_dirs.append(os.path.join(imdb_data_dir, 'train', 'neg'))
    data_dirs.append(os.path.join(imdb_data_dir, 'train', 'pos'))
    data_dirs.append(os.path.join(imdb_data_dir, 'train', 'unsup'))
    return absoluteFilePaths(data_dirs)


# generator considering a paragraph to be a whole review
def document_TaggedDocs_stream(tokens_only=False):
    imdb_docs = enumerate(list(get_imdb_docs()))
    for index, imbd_review in imdb_docs:
        with smart_open.smart_open(imbd_review, encoding="iso-8859-1") as review_file:
            str_content = review_file.read().replace('\n', ' ')
            # For training data, add tags
            yield TaggedDocument(simple_preprocess(str_content), [index])


# generator considering a paragraph to be a phrase
def phrase_TaggedDocs_stream():
    imdb_docs = list(get_imdb_docs())
    current_tag = 0
    for imbd_review in imdb_docs:
        with smart_open.smart_open(imbd_review, encoding="iso-8859-1") as review_file:
            str_content = review_file.read().replace('\n', ' ').replace('!', '.').replace('?', '.').split('.')
            for phrase in str_content:
                # For training data, add tags
                yield TaggedDocument(simple_preprocess(phrase), [current_tag])
                current_tag += 1


def persist_training_set(tagging_type):
    saved_train = os.path.join(saved_models, tagging_type.__name__.split('_')[0] + '.pkl')
    if not os.path.isfile(saved_train):
        print("Training set of " + tagging_type.__name__.split('_')[0] + " TaggedDocuments objects is computed now...")
        with open(saved_train, 'wb') as output:
            train_corpus = list(tagging_type())
            pickle.dump(train_corpus, output, pickle.HIGHEST_PROTOCOL)
    else:
        print(
            "Training set of " + tagging_type.__name__.split('_')[0] + " TaggedDocuments objects already saved on disk")
        train_corpus = pickle.load(open(saved_train, 'rb'))

    return train_corpus


class ModelChoice(object):

    def __init__(self, dm, epochs, window, hs, negative, tag_granularity):
        self.dm = dm
        self.epochs = epochs
        self.window = window
        self.hs = hs
        self.negative = negative
        self.tag_granularity = tag_granularity

    def __str__(self):
        mode_type = 'dm' if self.dm == 1 else 'dmbow'
        return 'model_' + mode_type + \
               '_epochs' + str(self.epochs) + \
               '_window' + str(self.window) + \
               '_hs' + str(self.hs) + \
               '_ns' + str(self.negative) + \
               '_' + self.tag_granularity.__name__.split('_')[0]


################ CREATE AND TRAIN A DOC2VEC MODEL ###################
def create_model(choice):
    saved_model = os.path.join(saved_models, str(choice))

    if not os.path.isfile(saved_model):
        print("The " + str(choice) + " Doc2Vec model was not trained before, training it now ...")
        start = timer()
        train_corpus = persist_training_set(choice.tag_granularity)
        model = Doc2Vec(dm=choice.dm,
                        vector_size=100,
                        window=choice.window,
                        negative=choice.negative,
                        hs=choice.hs,
                        min_count=2,
                        alpha=0.025,
                        sample=0,
                        workers=multiprocessing.cpu_count())

        model.build_vocab(train_corpus)
        print("The vocabulary of the model was built")
        # training of model
        model.train(train_corpus, total_examples=model.corpus_count, epochs=choice.epochs)
        model.save(saved_model)
        end = timer()
        print("The training process finished  and took " + str(round(end - start)) + " seconds")
    else:
        print("The " + path_leaf(saved_model) + " Doc2Vec model is already trained.")

    return Doc2Vec.load(saved_model)


################ CREATE EMBEDDINGS FOR ALL THE REVIEW DOCUMENTS ###################

def create_embedding_dataset(choice, update=False):
    embedding_path = os.path.join(data_root_dir, 'embeddings_' + str(choice))
    if os.path.exists(embedding_path) and not update:
        print("The embeddings for the model" + str(choice) + " are already created")
        return

    doc_dataset = split_review_data()['train']
    model = create_model(choice)
    emb_file = open(embedding_path, 'w', encoding='UTF-8')
    for (path, r_class) in doc_dataset:
        with open(path, 'r', encoding='UTF-8') as rev_file:
            str_content = rev_file.read().replace('\n', ' ').split()
            embed_str = '+1 ' if r_class == 'positive' else '-1 '
            embedding_vector = model.infer_vector(str_content)
            embed_str += ''.join(
                [str(index + 1) + ":" + str(dim_val) + " " for index, dim_val in enumerate(embedding_vector)])
            emb_file.write(embed_str + "\n")

    emb_file.close()


################ TRAIN AND TEST THE SVM MODEL USING THOSE EMBEDDINGS###################


if __name__ == "__main__":
    dm_choice = [1, 0]
    epochs_choice = [10, 20]
    tag_methods = [document_TaggedDocs_stream]

    for dm, epochs, tag_method in product(dm_choice, epochs_choice, tag_methods):
        window = 5 if dm == 1 else 15
        create_embedding_dataset(ModelChoice(dm=dm,
                                             epochs=epochs,
                                             hs=1,
                                             window= window,
                                             negative=0,
                                             tag_granularity=tag_method), update=True)

        create_embedding_dataset(ModelChoice(dm=dm,
                                             epochs=epochs,
                                             hs=0,
                                             window= window,
                                             negative=5,
                                             tag_granularity=tag_method), update=True)
