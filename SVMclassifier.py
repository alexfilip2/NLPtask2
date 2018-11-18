from Tools import *
import subprocess
from Doc2VecTutorial import *
from itertools import product

def split_embeddings(choice, split_id=-1, train_test_ratio=10, data_val_ratio=10, val_flag=True):
    train = open(os.path.join(data_root_dir, 'train'), "w", encoding='UTF-8')
    test = open(os.path.join(data_root_dir, 'test'), "w", encoding='UTF-8')
    val = open(os.path.join(data_root_dir, 'val'), "w", encoding='UTF-8')

    limit = len(os.listdir(pos_stem_dir)) + len(os.listdir(neg_stem_dir))
    emb_file = os.path.join(data_root_dir, 'embeddings_' + str(choice))
    if not os.path.exists(emb_file):
        print("The embedding dataset for " + str(choice) + "doesn't exist")
        return
    with open(emb_file, "r", encoding='UTF-8') as dataset:
        for line_nr, line in enumerate(dataset):
            if line_nr < (limit / data_val_ratio):
                if val_flag:
                    val.write(line)
                    continue
            if (line_nr % train_test_ratio) == split_id:
                test.write(line)
            else:
                train.write(line)
    train.close()
    test.close()
    val.close()


def train():
    # create model file
    train_file = os.path.join(data_root_dir, 'train')
    model_path = os.path.join(data_root_dir, 'model')
    # create the model file if not existing
    model_file = open(model_path, "w", encoding='UTF-8')
    # call the executable for SVM training
    subprocess_stdout = open(os.path.join(root_dir, 'intermediate_results'), 'a', encoding='UTF-8')
    subprocess.call((svm_light_learn + " -z c -m 100 " + train_file + " " + model_path), shell=True,
                    stdout=subprocess_stdout)
    subprocess_stdout.close()
    model_file.close()


def evaluate(choice, split_id):
    model_path = os.path.join(data_root_dir, 'model')
    test_file = os.path.join(data_root_dir, 'test')
    results_path = os.path.join(data_root_dir, 'results_split' + str(split_id) + '_' + str(choice))
    # call the executable for SVM testing
    subprocess_stdout = open(os.path.join(root_dir, 'intermediate_results'), 'a', encoding='UTF-8')
    subprocess.call((svm_light_classify + " " + test_file + " " + model_path + " " + results_path), shell=True,
                    stdout=subprocess_stdout)
    subprocess_stdout.close()


def cross_validation_SVM(choice, nr_of_folds=10):
    subprocess_stdout = open(os.path.join(root_dir, 'intermediate_results'), 'a', encoding='UTF-8')
    print("Cross validation accurracy for " + str(choice), file=subprocess_stdout)
    subprocess_stdout.close()
    for iter in range(nr_of_folds):
        split_embeddings(choice=choice, split_id=iter)
        train()
        evaluate(choice=choice, split_id=iter)


def summary_results(nr_of_folds=10):
    interm_results = open(os.path.join(root_dir, 'intermediate_results'), 'r', encoding='UTF-8')
    acc = 0
    fold = 0
    for line in interm_results:
        if line.split()[0] == 'Cross':
            print("On the embeddings based on " + line.split()[4])
        if line.split()[0] == 'Accuracy':
            fold += 1
            acc += float(line.split()[4].split('%')[0])
        if fold == nr_of_folds:
            print('the accuracy for the SVM classiffier is ' + str(acc / nr_of_folds))

            acc = 0
            fold = 0


if __name__ == "__main__":

    interm_results = open(os.path.join(root_dir, 'intermediate_results'), 'w', encoding='UTF-8')
    interm_results.close()
    dm_choice = [1, 0]
    epochs_choice = [10, 20]
    tag_methods = [document_TaggedDocs_stream]
    for dm, epochs, tag_method in product(dm_choice, epochs_choice, tag_methods):
        window = 5 if dm == 1 else 15
        cross_validation_SVM(ModelChoice(dm=dm,
                                             epochs=epochs,
                                             hs=1,
                                             window = window,
                                             negative=0,
                                             tag_granularity=tag_method))

        cross_validation_SVM(ModelChoice(dm=dm,
                                             epochs=epochs,
                                             hs=0,
                                             window = window,
                                             negative=5,
                                             tag_granularity=tag_method))

    summary_results()

