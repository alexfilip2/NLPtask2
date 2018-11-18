import operator
import os
import smart_open
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
import ntpath
from nltk.stem import PorterStemmer


root_dir = os.path.join(os.getcwd(), os.pardir, 'NLPtask2')
svm_light_learn = os.path.join(root_dir, 'SVMlight', 'svm_learn')
svm_light_classify = os.path.join(root_dir, 'SVMlight', 'svm_classify')
pos_stem_dir = os.path.join(root_dir, 'POS')
neg_stem_dir = os.path.join(root_dir, 'NEG')
pos_rev_dir = os.path.join(root_dir, 'POS')
neg_rev_dir = os.path.join(root_dir, 'NEG')

data_root_dir = os.path.join(root_dir, 'SVMdataset')
if not os.path.exists(data_root_dir):
    os.makedirs(data_root_dir)


# stem the initial review dataset using the Porter stemming algorithm
def stem_all_reviews():
    stemmer = PorterStemmer()
    if not os.path.exists(pos_stem_dir):
        os.makedirs(pos_stem_dir)
    else:
        assert len(sorted(os.listdir(pos_stem_dir))) == len(sorted(os.listdir(pos_rev_dir)))
        print("All the positive reviews are already stemmed.")
    if not os.path.exists(neg_stem_dir):
        os.makedirs(neg_stem_dir)
    else:
        assert len(sorted(os.listdir(neg_stem_dir))) == len(sorted(os.listdir(neg_rev_dir)))
        print("All the negative reviews are already stemmed.")
        return
    print("Start stemming of the review dataset...")
    pos_reviews = sorted(os.listdir(pos_rev_dir))
    neg_reviews = sorted(os.listdir(neg_rev_dir))

    for POS_review, NEG_review in zip(pos_reviews, neg_reviews):
        new_pos_stemmed = open(os.path.join(pos_stem_dir, 'stemmed' + '_' + POS_review), "w", encoding='UTF-8')
        new_neg_stemmed = open(os.path.join(neg_stem_dir, 'stemmed' + '_' + NEG_review), "w", encoding='UTF-8')

        with open(os.path.join(pos_rev_dir, POS_review), "r", encoding='UTF-8') as file:
            for line in file:
                for word in line.split():
                    new_pos_stemmed.write(stemmer.stem(word) + "\n")

        with open(os.path.join(neg_rev_dir, NEG_review), "r", encoding='UTF-8') as file:
            for line in file:
                for word in line.split():
                    new_neg_stemmed.write(stemmer.stem(word) + "\n")

        new_pos_stemmed.close()
        new_neg_stemmed.close()



# get all the full paths to the files in the directory in the form of a generator
def absoluteFilePaths(directory_list):
    for directory in directory_list:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def grouper(train_corpus, chunk_size):
    return [train_corpus[x:x + chunk_size] for x in range(0, len(train_corpus), chunk_size)]


def split_review_data(test_fold_id=-1, train_test_ratio=10, data_val_ratio=10, val_flag=False):
    train, test, validation = [], [], []
    pos_reviews, neg_reviews = list(absoluteFilePaths([pos_stem_dir])), list(absoluteFilePaths([neg_stem_dir]))
    assert len(os.listdir(pos_stem_dir)) == len(os.listdir(neg_stem_dir)), (
        'Incomplete review dataset, check PosNeg/PosPos directories')
    limit = len(os.listdir(pos_stem_dir))
    for index, POS_review, NEG_review in zip(range(limit), pos_reviews, neg_reviews):
        if index < (limit / data_val_ratio):
            if val_flag:
                validation.append((POS_review, 'positive'))
                validation.append((NEG_review, 'negative'))
                continue
        if (index % train_test_ratio) == test_fold_id:
            test.append((POS_review, 'positive'))
            test.append((NEG_review, 'negative'))
        else:
            train.append((POS_review, 'positive'))
            train.append((NEG_review, 'negative'))

    train.sort(key=operator.itemgetter(1))
    test.sort(key=operator.itemgetter(1))
    validation.sort(key=operator.itemgetter(1))

    return {'train': train, 'test': test, 'val': validation}


def tag_by_document(file_path, start_tag, tokens_only=False):
    with smart_open.smart_open(file_path, encoding="iso-8859-1") as review_file:
        str_content = review_file.read().replace('\n', '')
        if tokens_only:
            # returns a list of lower case tokens resulted from the input string
            return simple_preprocess(str_content)
        else:
            # For training data, add tags
            return TaggedDocument(simple_preprocess(str_content), [start_tag])


def stream_doc_words(dirs_list, start_tag, tokens_only=False):
    abs_path_generator = absoluteFilePaths(dirs_list)
    tag = start_tag
    for file_path in abs_path_generator:
        with smart_open.smart_open(file_path, encoding="iso-8859-1") as review_file:
            str_content = review_file.read().replace('\n', '')
            if tokens_only:
                # returns a list of lower case tokens resulted from the input string
                yield simple_preprocess(str_content)
            else:
                # For training data, add tags
                tag += 1
                yield TaggedDocument(simple_preprocess(str_content), [tag])
