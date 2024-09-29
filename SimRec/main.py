import os
import time
import torch
import json
import argparse
from functools import partial

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true', 'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true' or s == 'True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="../data_preprocessing/Beauty/Beauty.txt")
    parser.add_argument('--item_frequency', default="../data_preprocessing/Beauty/Beauty-item_freq.txt")

    parser.add_argument('--similarity_indices', default="../data_preprocessing/Beauty/Beauty-similarity-indices-thenlper_gte-large.pt", type=str)
    parser.add_argument('--similarity_values', default="../data_preprocessing/Beauty/Beauty-similarity-values-thenlper_gte-large.pt", type=str)
    parser.add_argument('--similarity_threshold', default=0.9, type=float, help="zero similarities that are lower than the threshold (cosine similarity is between [-1,1])")

    parser.add_argument('--temperature', default=1, type=float, help="softmax temperature for training")
    parser.add_argument('--lambd', default=0.5, type=float, help="control the weight of the 'distilation' loss")
    parser.add_argument('--lambd_scheduling', default="LINEAR", choices=["LINEAR", "NONE"])
    parser.add_argument('--lambd_warmup_steps', default=1000, type=int, help="control the number of warmup steps for the lambda scheduler")
    parser.add_argument('--lambd_steps', default=70000, type=int, help="control the number of lambda steps for the lambda binary scheduler")

    parser.add_argument('--train_dir', default='train_beauty')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default='false', type=str2bool)
    parser.add_argument('--state_dict_path', type=str)
    args = parser.parse_args()
    return args

        
def main(args):
    print(args)

    item_freq_df = pd.read_csv(args.item_frequency, delimiter=' ', names=['id', 'freq'])
    item_freq_df['id'] += 1 # id 0 is reserved for PAD
    item_freq = pd.Series(item_freq_df.freq.values, index=item_freq_df.id).to_dict()

    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # global dataset
    dataset = data_partition(args.dataset)


    if args.similarity_indices is None or args.similarity_values is None:
        raise Exception("args.similarity_indices or args.similarity_values are None")
    similarity_indices = torch.load(args.similarity_indices, map_location=args.device)
    similarity_values = torch.load(args.similarity_values, map_location=args.device)
    if args.similarity_threshold < 1:
        similarity_values[similarity_values <= args.similarity_threshold] = -float('inf')
    else:
        # make the self similarity maximal
        similarity_indices = torch.arange(similarity_indices.shape[0], device=args.device).reshape(-1, 1)
        similarity_values = torch.ones_like(similarity_indices)
    
    # handle padding index (0) by increasing the index values
    similarity_indices += 1

    similarity_indices = torch.concat([torch.arange((similarity_indices.shape[1]), device=args.device).unsqueeze(dim=0), similarity_indices], dim=0)
    similarity_values = torch.concat([torch.full((1, similarity_values.shape[1]), fill_value=-float('inf'), device=args.device), similarity_values], dim=0)


    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    training_steps = num_batch * args.num_epochs
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(f'average sequence length: {cc / len(user_train):.2f}')
    print(f"{training_steps} training steps in total")

    lambda_scheduler = NoneSchedule(args.lambd)
    if args.lambd_scheduling == 'LambdaScheduling.LINEAR':
        lambda_scheduler = LinearScheduleWithWarmup(args.lambd, args.lambd_warmup_steps, min(training_steps, args.lambd_steps))

    f = open(os.path.join(args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SimRec(usernum, itemnum, args).to(args.device)
    
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_total_params} model parameters")

    model.train() 
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        test_eval_results = [evaluate_test(model, dataset, args) for _ in range(5)]
            
        test_id_hr = test_eval_results[-1][-2]
        test_id_ndcg = test_eval_results[-1][-1]

        test_hr_list = [e[0][1] for e in test_eval_results]
        test_ndcg_list = [e[0][0] for e in test_eval_results]

        test_hr = np.array(test_hr_list).mean()
        test_ndcg = np.array(test_ndcg_list).mean()
        print(f'test (NDCG@10: {test_hr:.4f}, HR@10: {test_ndcg:.4f})')
    

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    cross_entropy_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    fname = f'SimRec.epoch={args.num_epochs}.pth'
    best_val_hr = -float('inf')
    best_results = {}
    global_step = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: 
            break 
        for step in range(num_batch):
            lambd = lambda_scheduler.get_lambd()
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            pos_logits, neg_logits, logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != PAD_IDX)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            pos = torch.tensor(pos, device=args.device, dtype=torch.long).view(-1) # (batch_size x max_len)
            pad_indices = pos == PAD_IDX
            pos = pos[~pad_indices]
            logits_flat = logits.view(-1, itemnum + 1) # (batch_size x max_len ,num_items + 1)
            logits_flat = logits_flat[~pad_indices]

            targets_distribution = create_similarity_distirbution(similarity_indices, similarity_values, args.temperature, pos) # (batch_size x max_len, num_items + 1)

            loss = lambd * cross_entropy_criterion(logits_flat / args.temperature, targets_distribution) + (1 - lambd) * loss
                
            if args.l2_emb > 0:
                for param in model.item_emb.parameters(): 
                    loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            lambda_scheduler.step()
            global_step += 1
        print(f"loss in epoch {epoch}: {loss.item()}")
    
        if epoch % 20 == 0 or epoch == args.num_epochs:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            test_eval_results = [evaluate_test(model, dataset, args) for _ in range(5)]
            val_eval_results = [evaluate_valid(model, dataset, args) for _ in range(5)]
            
            test_id_hr = test_eval_results[-1][-2]
            test_id_ndcg = test_eval_results[-1][-1]
            valid_id_hr = val_eval_results[-1][-2]
            valid_id_ndcg = val_eval_results[-1][-1]
            
            test_hr_list = [e[0][1] for e in test_eval_results]
            test_ndcg_list = [e[0][0] for e in test_eval_results]
            val_hr_list = [e[0][1] for e in val_eval_results]
            val_ndcg_list = [e[0][0] for e in val_eval_results]

            # we take the average of different test/val runs
            test_hr = np.array(test_hr_list).mean()
            test_ndcg = np.array(test_ndcg_list).mean()
            val_hr = np.array(val_hr_list).mean()
            val_ndcg = np.array(val_ndcg_list).mean()

            (train_ndcg, train_hr), train_id_hr, train_id_ndcg = evaluate_train(model, dataset, args)
            s = '\nepoch:%d, time: %f(s), train (NDCG@10: %.4f, HR@10: %.4f), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'\
                    % (epoch, T, train_ndcg, train_hr, val_ndcg, val_hr, test_ndcg, test_hr)
            print(s)
            f.write(s + '\n')
            f.flush()
            if val_hr > best_val_hr:
                torch.save(model.state_dict(), os.path.join(args.train_dir, fname))
                torch.save(test_id_hr, os.path.join(args.train_dir, "test_id_hr.pt"))
                torch.save(test_id_ndcg, os.path.join(args.train_dir, "test_id_ndcg.pt"))
                torch.save(valid_id_hr, os.path.join(args.train_dir, "valid_id_hr.pt"))
                torch.save(valid_id_ndcg, os.path.join(args.train_dir, "valid_id_ndcg.pt"))
                torch.save(train_id_hr, os.path.join(args.train_dir, "train_id_hr.pt"))
                torch.save(train_id_ndcg, os.path.join(args.train_dir, "train_id_ndcg.pt"))
                print(f"New best val HR@10: {val_hr}. Saving checkpoint.")
                best_val_hr = max(best_val_hr, val_hr)
                best_results = {
                    'test/best_id_to_HR@10': test_id_hr,
                    'test/best_id_to_NDCG@10': test_id_ndcg,
                    'val/best_id_to_HR@10': valid_id_hr,
                    'val/best_id_to_NDCG@10': valid_id_ndcg,
                    'train/best_id_to_HR@10': train_id_hr,
                    'train/best_id_to_NDCG@10': train_id_ndcg,
                    "test/best_NDCG@10": test_ndcg,
                    "test/best_HR@10": test_hr,
                    "val/best_NDCG@10": val_ndcg,
                    "val/best_HR@10": val_hr,
                    "train/best_NDCG@10": train_ndcg,
                    "train/best_HR@10": train_hr,
                }
            t0 = time.time()
            model.train()
    f.close()
    sampler.close()
    print("Done")


if __name__ == '__main__':
    args = parse_args()
    main(args)