import sys, os, random
import time
import logging
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__file__)

import torch
import torch.nn as nn
import torch.optim as OPT
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import data
from few_shot_code.SupportSetManagerLargeHybrid import SupportSetManagerLargeHybrid
from few_shot_code.MatchingCnn import MatchPair
import few_shot_code.ArgumentProcessorMnet as args_mnet
from few_shot_code.MatchingNetWithSupportPolicy import MatchingCnnWithSuppPolicy
from few_shot_code.DataProcessing.NlcDatasetSingleFile import NlcDatasetSingleFile
from few_shot_code.load_data_on_the_fly import get_iter_by_task_id_on_the_fly

from torchtext import data
from few_shot_code.DataProcessing.MTLField import MTLField

parser = args_mnet.get_parser()
args = parser.parse_args()

with open(os.path.join(args.save_path, "args.txt"), "w", encoding="utf-8") as file:
    file.write("\n".join(sys.argv[1:]))

batch_size = args.batch_size
args.seed = 12345678

torch.manual_seed(args.seed)
if args.cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)
    LOGGER.info("use cuda for training")
else:
    args.device = torch.device("cpu")
    LOGGER.info("use cpu for training")


def load_train_test_files(listfilename, test_suffix='.dev'):
    filein = open(listfilename, 'r')
    file_tuples = []
    for line in filein:
        array = line.strip().split('\t')
        line = array[0]
        trainfile = line + '.train'
        devfile = line + '.dev'
        testfile = line + test_suffix
        file_tuples.append((trainfile, devfile,  testfile))
    filein.close()
    return file_tuples

datasets = []
list_dataset = []
file_tuples = load_train_test_files(args.filelist)
TEXT = MTLField(lower=True)

print("loading data into memory")
for (trainfile, devfile, testfile) in tqdm(file_tuples):
    LABEL1 = data.Field(sequential=False)
    train1, dev1, test1 = NlcDatasetSingleFile.splits(
        TEXT, LABEL1, path=args.workingdir, train=trainfile,
        validation=devfile, test=testfile)
    datasets.append((TEXT, LABEL1, train1, dev1, test1))
    list_dataset.append(train1)
    list_dataset.append(dev1)
    list_dataset.append(test1)


dataset_iters = []
print("generating data iterators")
for (TEXT, LABEL, train, dev, test) in tqdm(datasets):
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=args.device)
    train_iter.repeat = False
    dataset_iters.append((train_iter, dev_iter, test_iter))

# print information about the data
num_batch_total = 0
for i, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    num_batch_total += len(train) / batch_size

TEXT.build_vocab(list_dataset)
# TEXT.build_vocab(list_dataset)

# build the vocabulary
for taskid, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    LABEL.build_vocab(train, dev, test)
    LABEL.vocab.itos = LABEL.vocab.itos[1:]
    for k, v in LABEL.vocab.stoi.items():
        LABEL.vocab.stoi[k] = v - 1

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    # print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    print('len(LABEL.vocab)', len(LABEL.vocab)),
    #print LABEL.vocab.itos
    print(len(LABEL.vocab.itos))
    #if taskid == 0:
    #    print LABEL.vocab.stoi
    #print len(LABEL.vocab.stoi)

config = args
config.n_embed = len(TEXT.vocab)
config.d_embed = args.emsize
config.d_proj = 100
config.d_hidden = args.nhid
config.projection = False
config.num_tasks = len(datasets)

ss_manager = SupportSetManagerLargeHybrid(datasets, config, config.sample_per_class)

config.n_labels = []
for (TEXT, LABEL, train, dev, test) in datasets:
    config.n_labels.append(len(LABEL.vocab))
print(config.n_labels)

config.n_cells = 1
config.maxpool = True

model = MatchingCnnWithSuppPolicy(config, args.emsize, config.d_hidden, num_tasks=config.num_tasks, normal_init=True)

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
opt = OPT.Adam(model.parameters(), lr=args.lr)

# sys.exit(0)

iterations = 0
start = time.time()
best_dev_acc = -1
best_dev_epoch = -1
best_test_acc = -1
best_test_epoch = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

overall_log_template = 'Best dev acc: {:>8.6f}, epoch: {:>5.0f}; Best test acc: {:>8.6f}, epoch: {:>5.0f}'
# os.makedirs(args.save_path, exist_ok=True)
iter_per_sample = 1
n_correct, n_total = 0, 0
num_batch_total = np.ceil(num_batch_total).astype(int)
iterations = np.ceil(num_batch_total * args.epochs / iter_per_sample)
iterations = iterations.astype(int)
for t in range(1):
    taskid = random.randint(0, config.num_tasks - 1)
    (train_iter, dev_iter, test_iter) = dataset_iters[taskid]
    train_iter.init_epoch()
    model.train()

    for num_iter in range(1):
        batch = next(iter(train_iter))
        sys.stdout.write('%d\r'%t)
        sys.stdout.flush()
        opt.zero_grad()

        if args.training_policy == 'first':
            supp_text = ss_manager.select_support_set(taskid, ss_manager.FIXED_FIRST)
        elif args.training_policy == 'random':
            supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM)
        elif args.training_policy == 'hybrid':
            if t % 2 == 0:
                supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM_SUB)
            else:
                supp_text = ss_manager.select_support_set(taskid, ss_manager.RANDOM)
        answer = model(MatchPair(batch.text, supp_text))
        print(batch.text.shape, "batch text shape")
        print(supp_text.shape, "supp text shape")
        print(answer.shape, "answer shape")
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        loss = criterion(answer, batch.label)
        loss.backward()
        opt.step()

    #print(t, iter_per_sample, num_batch_total)
    #if (t + 1) % num_batch_total == 0:
    if (t + 1) % min(10000 / iter_per_sample, num_batch_total) == 0:
        # if config.lrDecay > 0:
        #     scheduler.step(loss)
        # if config.lrDecay > 0:
        #    # cur_lr = opt.lr
        #    # cur_lr = cur_lr * 0.9
        #    # opt.setScale(cur_lr * config.lr > 0.0001 and cur_lr or 0.0001 / config.lr)
        #    # print('lr: ', opt.getScale() * config.lr)
        #     print(opt.param_groups["lr"])
        #     opt.param_groups["lr"] = opt.param_groups["lr"]*.9

        print(log_template.format(time.time() - start,
                                  (t + 1) / num_batch_total, (t + 1), (t + 1), num_batch_total * args.epochs,
                                  100., loss.item(), ' ' * 8,
                                  n_correct / n_total * 100, ' ' * 12))

        model.eval()
        avg_dev_acc = 0.0
        avg_test_acc = 0.0
        print("start testing")
        for taskid, (train_iter, dev_iter, test_iter) in enumerate(dataset_iters):
            n_dev_correct, dev_loss = 0, 0
            n_dev_total = 0
            n_test_correct, test_loss = 0, 0
            n_test_total = 0
            if args.testing_policy == 'first':
                supp_text = ss_manager.select_support_set(taskid, ss_manager.FIXED_FIRST)
            elif args.testing_policy == 'mean':
                supp_emb = ss_manager.get_average_as_support(taskid, model)
            elif args.testing_policy == 'mean_std':
                supp_emb, supp_std = ss_manager.get_average_and_std_as_support(taskid, model)
            for set_idx, set_iter in enumerate([dev_iter, test_iter]):
                set_iter.init_epoch()
                for dev_batch_idx, dev_batch in enumerate(set_iter):
                    if args.testing_policy == 'mean':
                        answer = model(MatchPair(dev_batch.text, supp_emb), y_mode = 'emb')
                    elif args.testing_policy == 'mean_std':
                        answer = model(MatchPair(dev_batch.text, supp_emb), y_mode='emb', std=supp_std)
                    else:
                        answer = model(MatchPair(dev_batch.text, supp_text))
                    if set_idx == 0:
                        n_dev_correct += (
                        torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                        n_dev_total += dev_batch.batch_size
                        dev_loss = criterion(answer, dev_batch.label)
                    else:
                        n_test_correct += (
                            torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                        n_test_total += dev_batch.batch_size
                        test_loss = criterion(answer, dev_batch.label)

            dev_acc = 100. * n_dev_correct / n_dev_total
            test_acc = 100. * n_test_correct / n_test_total
            avg_dev_acc += dev_acc
            avg_test_acc += test_acc


        avg_dev_acc /= config.num_tasks
        avg_test_acc /= config.num_tasks
        print('Iteration Results:\t{:>4.2f}\t{:>4.2f}'.format(avg_dev_acc, avg_test_acc))

        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            best_dev_epoch = (t + 1) / num_batch_total

            best_test_acc = avg_test_acc
            best_test_epoch = (t + 1) / num_batch_total

        print(overall_log_template.format(best_dev_acc, best_dev_epoch, best_test_acc, best_test_epoch))

result = list()
for taskid, (train_iter, dev_iter, test_iter) in enumerate(dataset_iters):
    n_dev_correct, dev_loss = 0, 0
    n_dev_total = 0
    n_test_correct, test_loss = 0, 0
    n_test_total = 0
    if args.testing_policy == 'first':
        supp_text = ss_manager.select_support_set(taskid, ss_manager.FIXED_FIRST)
    elif args.testing_policy == 'mean':
        supp_emb = ss_manager.get_average_as_support(taskid, model)
    elif args.testing_policy == 'mean_std':
        supp_emb, supp_std = ss_manager.get_average_and_std_as_support(taskid, model)
    for set_idx, set_iter in enumerate([dev_iter, test_iter]):
        set_iter.init_epoch()
        for dev_batch_idx, dev_batch in enumerate(set_iter):
            if args.testing_policy == 'mean':
                answer = model(MatchPair(dev_batch.text, supp_emb), y_mode = 'emb')
            elif args.testing_policy == 'mean_std':
                answer = model(MatchPair(dev_batch.text, supp_emb), y_mode='emb', std=supp_std)
            else:
                answer = model(MatchPair(dev_batch.text, supp_text))
            if set_idx == 0:
                n_dev_correct += (
                torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                n_dev_total += dev_batch.batch_size
                dev_loss = criterion(answer, dev_batch.label)
            else:
                n_test_correct += (
                    torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                n_test_total += dev_batch.batch_size
                test_loss = criterion(answer, dev_batch.label)

    dev_acc = n_dev_correct / n_dev_total
    test_acc =  n_test_correct / n_test_total

    result.append(dev_acc.item())

with open(os.path.join(args.save_path, "statistics.txt"), "w+", encoding="utf-8") as file:
    file.writelines([workspace_id.replace(".train", "") +"\t" + str(acc) +"\n"  for (workspace_id, _, _), acc in zip(file_tuples, result)])

