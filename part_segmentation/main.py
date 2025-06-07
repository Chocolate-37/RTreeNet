from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import torch.cuda
import matplotlib.pyplot as plt
import time
import sklearn.metrics as skl_metrics
import argparse
import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from solution import solution

# Define a Bounds function to ensure the solution vector is within specified bounds
def Bounds(s, Lb, Ub):
    temp = s.copy()
    for i in range(len(s)):
        if temp[i] < Lb[i]:
            temp[i] = Lb[i]
        elif temp[i] > Ub[i]:
            temp[i] = Ub[i]
    return temp

# ACPA (Coati Optimization Algorithm) combined with PSO (Particle Swarm Optimization)
def ACPA(objf, lb, ub, dim, SearchAgents_no, Max_pos, PSO_iters, args):
    xposbest = np.zeros([1, dim])
    fvalbest = float("inf")

    best_instance_iou = 0  # Initialize best instance IOU

    # If lb and ub are not lists, convert them to lists of dim dimension
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Randomly initialize solution vectors for the number of search agents
    X = np.asarray([X * (ub - lb) + lb for X in np.random.uniform(0, 1, (SearchAgents_no, dim))])
    X_P1 = np.zeros(X.shape)
    F_P1 = np.zeros([1, SearchAgents_no])
    X_P2 = np.zeros(X.shape)
    F_P2 = np.zeros([1, SearchAgents_no])
    fit = np.zeros([1, SearchAgents_no])

    # Initial evaluation
    for i in range(SearchAgents_no):
        fit[0, i] = objf(X[i, :])
        print('Code running initial evaluation objf') #SearchAgents_no = 4

    convergence_curve = np.zeros(Max_pos)

    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    best_lr_for_iou = None  # Initialize best learning rate for IOU

    for t in range(Max_pos):
        best = np.min(fit[0, :])
        if t == 0:
            fbest = best
            ibest = np.argmin(fit[0, :])
            Xbest = X[ibest, :]
        elif best < fbest:
            fbest = best
            ibest = np.argmin(fit[0, :])
            Xbest = X[ibest, :]

        # COA exploration phase
        for i in range(int(SearchAgents_no / 2)):
            iguana = Xbest
            I = round(1 + random.random())

            X_P1[i, :] = X[i, :] + random.random() * (iguana - I * X[i, :])
            X_P1[i, :] = Bounds(X_P1[i, :], lb, ub)

            L = X_P1[i, :]
            F_P1[0, i] = objf(L)
            print('Code running in exploration phase objf') #SearchAgents_no / 2 = 2

            if F_P1[0, i] < fit[0, i]:
                X[i, :] = X_P1[i, :]
                fit[0, i] = F_P1[0, i]

        # COA exploitation phase
        for i in range(int(SearchAgents_no / 2), SearchAgents_no):
            iguana = lb + random.random() * (ub - lb)
            L = iguana
            F_HL = objf(L)
            print('Code running in exploitation phase 1 objf') #SearchAgents_no / 2 = 2
            I = round(1 + random.random())

            if fit[0, i] > F_HL:
                X_P1[i, :] = X[i, :] + random.random() * (iguana - I * X[i, :])
            else:
                X_P1[i, :] = X[i, :] + random.random() * (X[i, :] - iguana)
            X_P1[i, :] = Bounds(X_P1[i, :], lb, ub)

            L = X_P1[i, :]
            F_P1[0, i] = objf(L)
            print('Code running in exploitation phase 2 objf') #SearchAgents_no / 4 = 1

            if F_P1[0, i] < fit[0, i]:
                X[i, :] = X_P1[i, :]
                fit[0, i] = F_P1[0, i]

        # PSO phase
        pso = PSO(objf=lambda lr: objf(lr), D=dim, N=SearchAgents_no, M=PSO_iters, p_low=lb[0], p_up=ub[0], v_low=-0.1, v_high=0.1)
        print('Code running to pso = PSO()')
        pso.g_best, pso.g_bestFit = Xbest, fbest   # Use COA's best solution as PSO's initial global best solution
        pso.update()
        print('Code running to pso.update()')
        Xbest, fbest = pso.g_best, pso.g_bestFit

        convergence_curve[t] = fbest
        if t % 1 == 0:
            print(f"At iteration {t}, the best fit is {fbest}")

        # Update the best learning rate and instance IOU
        if fbest > best_instance_iou:
            best_instance_iou = fbest
            best_lr_for_iou = float(X[np.argmin(fit[0, :])])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "ACPA"
    s.objfname = objf.__name__
    s.best = fvalbest
    s.bestIndividual = Xbest

    return s, best_instance_iou, best_lr_for_iou

# Define the PSO class as described earlier
class PSO:
    def __init__(self, objf, D, N, M, p_low, p_up, v_low, v_high, w=1., c1=2., c2=2.):
        self.objf = objf
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.D = D
        self.N = N
        self.M = M
        self.p_range = [p_low, p_up]
        self.v_range = [v_low, v_high]
        self.x = np.zeros((self.N, self.D))
        self.v = np.zeros((self.N, self.D))
        self.p_best = np.zeros((self.N, self.D))
        self.g_best = np.zeros((1, self.D))[0]
        self.p_bestFit = np.zeros(self.N)
        self.g_bestFit = float('Inf')

        # Initialize particle positions and velocities
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0], self.p_range[1])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            self.p_best[i] = self.x[i]
            fit = self.fitness(self.p_best[i])
            print('Code running in PSO development phase 1 objf') # SearchAgents_no = 4 +1 = 5
            self.p_bestFit[i] = fit
            if fit < self.g_bestFit:
                self.g_best = self.p_best[i]
                self.g_bestFit = fit

    def fitness(self, x):
        return self.objf(x) # Evaluate particle fitness based on objective function

    # Update particle positions and velocities according to PSO algorithm
    def update(self):
        for i in range(self.N):
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
            self.x[i] = self.x[i] + self.v[i]
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0]:
                    self.x[i][j] = self.p_range[0]
                if self.x[i][j] > self.p_range[1]:
                    self.x[i][j] = self.p_range[1]
            _fit = self.fitness(self.x[i])
            print('Code running to PSO update objf') #SearchAgents_no = 4
            if _fit < self.p_bestFit[i]:
                self.p_best[i] = self.x[i]
                self.p_bestFit[i] = _fit
            if _fit < self.g_bestFit:
                self.g_best = self.x[i]
                self.g_bestFit = _fit

    # Run PSO optimization algorithm
    def optimize(self):
        best_fit = []
        w_range = None
        if isinstance(self.w, tuple):
            w_range = self.w[1] - self.w[0]
            self.w = self.w[1]
        time_start = time.time()
        for i in range(self.M):
            self.update()
            if w_range:
                self.w -= w_range / self.M
            print(f"\rIter: {i}/{self.M} fitness: {self.g_bestFit:.4f}", end='')
            best_fit.append(self.g_bestFit)
        time_end = time.time()
        print(f'\nAlgorithm takes {time_end - time_start} seconds')
        plt.plot(best_fit)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Optimization Process")
        plt.show()
        return self.g_best, self.g_bestFit



classes_str = ['tree']

best_acc = 0  # Initialize best accuracy to 0
best_class_iou = 0  # Initialize best class iou to 0
best_instance_iou = 0  # Initialize best instance iou to 0
best_train_iou = 0  # Initialize best training iou to 0
epoch_objf = 0
best_precision = 0  # Added new
best_recall = 0  # Added new
best_f1_score = 0  # Added new

def _init_():
    if not os.path.exists('checkpoints'): # Check if 'checkpoints' folder exists, create if not
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):  # Check if 'checkpoints/args.exp_name' folder exists, create if not
        os.makedirs('checkpoints/' + args.exp_name)


def weight_init(m):
    if isinstance(m, torch.nn.Linear): # Check if current module m is a linear layer, if so, initialize weights with Xavier normal distribution and bias with constant
        torch.nn.init.xavier_normal_(m.weight)  # Initialize weights with Xavier normal distribution
        if m.bias is not None:  # If bias exists, initialize it with constant
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):  # Check if current module m is a 2D conv layer, if so, initialize weights with Xavier normal distribution and bias with constant
        torch.nn.init.xavier_normal_(m.weight)   # Initialize weights with Xavier normal distribution
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)   # If bias exists, initialize it with constant
    elif isinstance(m, torch.nn.Conv1d):    # Check if current module m is a 1D conv layer, if so, initialize weights with Xavier normal distribution and bias with constant
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)  # If bias exists, initialize it with constant
    elif isinstance(m, torch.nn.BatchNorm2d):  # Check if current module m is a 2D BatchNorm layer, if so, initialize weights and bias with constant
        torch.nn.init.constant_(m.weight, 1)  # Initialize weights with constant
        torch.nn.init.constant_(m.bias, 0)   # Initialize bias with constant
    elif isinstance(m, torch.nn.BatchNorm1d):  # Check if current module m is a 1D BatchNorm layer, if so, initialize weights and bias with constant
        torch.nn.init.constant_(m.weight, 1)  # Initialize weights with constant
        torch.nn.init.constant_(m.bias, 0)   # Initialize bias with constant


def train(args, io):
    # ============= Global variables ===================
    global best_acc
    global best_class_iou
    global best_instance_iou
    global best_train_iou
    global epoch_objf
    global best_precision  # Added new
    global best_recall    # Added new
    global best_f1_score # Added new

    # ============= Model ===================
    num_part = 2 # pre:50 now:2 Set variable num_part to 50, representing the number of parts in the model
    device = torch.device("cuda" if args.cuda else "cpu") # Determine whether to use GPU or CPU

    model = models.__dict__[args.model](num_part).to(device) # Create a model object with the model name specified by args.model, using num_part as a parameter passed to the model constructor
    io.cprint(str(model)) # Print model information using io.cprint() function  Print statement for PointMLP network structure

    model.apply(weight_init) # Initialize model parameters
    model = nn.DataParallel(model) # Wrap the model with nn.DataParallel to support multi-GPU parallel training
    print("Let's use", torch.cuda.device_count(), "GPUs!") # Print the number of GPUs being used

    '''Resume or not'''
    if args.resume: # Check if args.resume is True, indicating whether to resume training from previous checkpoint
        state_dict = torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name, # Load the state dictionary of the previously saved best model
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys(): # Iterate through the keys of the model parameter dictionary
            if 'module' not in k:
                from collections import OrderedDict # Import OrderedDict class
                new_state_dict = OrderedDict() # Create an ordered dictionary new_state_dict to store renamed model parameters
                for k in state_dict: # Iterate through the keys of the previously loaded model parameter dictionary
                    new_state_dict['module.' + k] = state_dict[k] # Add the model parameter dictionary keys to new_state_dict
                state_dict = new_state_dict # Assign the renamed model parameter dictionary to state_dict variable
            break
        model.load_state_dict(state_dict) # Load model parameters using the renamed model parameter dictionary

        print("Resume training model...") # Print message indicating resuming training from previous model
        print(torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name).keys()) # Print the keys contained in the loaded model parameter file
    else:
        print("Training from scratch...") # Print message indicating starting training from scratch

    # =========== Dataloader =================
    train_data = PartNormalDataset(npoints=2048, split='trainval', normalize=True)#PRE:False # Create training dataset object train_data
    print("The number of training data is:%d", len(train_data))

    test_data = PartNormalDataset(npoints=2048, split='test', normalize=True)#PRE:False # Create test dataset object test_data
    print("The number of test data is:%d", len(test_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True) # Create DataLoader object train_loader for training data
    print('Code running to train_loader')
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False) # Create DataLoader object test_loader for test data
    print('Code running to test_loader')

    def objf(lr):
        """
        Objective function, which is used to assess the learning rate. Train the model on several epochs and return the
        loss on the validation set. Parameter: LR (float): The learning rate to be assessed. num_classes
        (int): The number of categories in the dataset. Returns: float: Verify loss.
        """
        opt = optim.Adam(model.parameters(), lr=float(lr), betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        global best_acc           # pre:best_acc   now: global   Best accuracy initialized to 0
        global best_class_iou     # Best class iou initialized to 0
        global best_instance_iou  # Best instance iou initialized to 0
        global best_train_iou     # Best train iou initialized to 0
        global epoch_objf         #objf global epoch
        global best_precision     # New added
        global best_recall        # New added
        global best_f1_score      # New added

        num_part = 2  # pre:50 now:2
        num_classes = 1  # num_classes = 16  # Category count, default is 1 (binary classification task)
        num_loops = 2 # num_loops = 3

        for loop in range(num_loops):
            train_metrics , train_loss , count = train_epoch(train_loader, model, opt, num_part = num_part, num_classes = num_classes)
            outstr1 = 'Train %d, loss: %f, acc: %f, ins_iou: %f, precision: %f, recall: %f, f1_score: %f' % (epoch_objf+1, train_loss * 1.0 / count, train_metrics['accuracy'], train_metrics['shape_avg_iou'], train_metrics['precision'],train_metrics['recall'], train_metrics['f1_score'])

            #outstr1 = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch_objf+1, train_loss * 1.0 / count, train_metrics['accuracy'], train_metrics['shape_avg_iou'])
            print(f'Code entering objf train_epoch epoch_objf and loop {epoch_objf} {loop}')
            io.cprint('Learning rate: %f' % lr)  # Update learning rate
            io.cprint(outstr1) # Print training indicators

            # Update and save best train IoU
            if train_metrics['shape_avg_iou'] > best_train_iou:
                best_train_iou = train_metrics['shape_avg_iou']
                io.cprint('Max Train IoU:%.5f' % best_train_iou)

            # Evaluate model on validation set
            test_metrics, total_per_cat_iou,test_loss = test_epoch(test_loader, model, num_part = num_part, num_classes = num_classes)
            print('Code entering objf test_epoch')
            outstr2 = 'Test %d, loss: %f, acc: %f, ins_iou: %f, precision: %f, recall: %f, f1_score: %f' % (
            epoch_objf + 1, test_loss * 1.0 / count, test_metrics['accuracy'], test_metrics['shape_avg_iou'], test_metrics['precision'],
            test_metrics['recall'], test_metrics['f1_score'])

            io.cprint(outstr2)

            # 1. when get the best accuracy, save the model:  # 1. When the best accuracy is obtained, save the model
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                io.cprint('Max Acc:%.5f' % best_acc)  # Print best accuracy
                state = {
                    'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    'optimizer': opt.state_dict(), 'epoch': epoch_objf, #pre:epoch
                    'test_acc': best_acc}  # Save model parameters, optimizer status, and current epoch number to 'checkpoints/%s/best_acc_model.pth' file
                torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)

            # 2. when get the best instance_iou, save the model:  # 2. When the best instance iou is obtained, save the model
            if test_metrics['shape_avg_iou'] > best_instance_iou:
                best_instance_iou = test_metrics['shape_avg_iou']
                io.cprint('Max instance iou:%.5f' % best_instance_iou)  # Print best instance iou
                state = {
                    'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    'optimizer': opt.state_dict(), 'epoch': epoch_objf, 'test_instance_iou': best_instance_iou} #pre:epoch
                torch.save(state,
                           'checkpoints/%s/best_insiou_model.pth' % args.exp_name)  # Save model parameters, optimizer status, and current epoch number to 'checkpoints/%s/best_insiou_model.pth' file

            # 3. when get the best class_iou, save the model: # 3. When the best class iou is obtained, save the model
            # first we need to calculate the average per-class iou # First we need to calculate the average per-class iou
            class_iou = 0
            for cat_idx in range(1):  # pre:16  # Loop to calculate the sum of iou for each class
                class_iou += total_per_cat_iou[cat_idx]
            avg_class_iou = class_iou / 1  # pre:16  # Calculate average per-class iou (here default is 1 class)
            if avg_class_iou > best_class_iou:  # If average per-class iou is greater than current best class iou
                best_class_iou = avg_class_iou  # Update best class iou
                # print the iou of each class:  # Print the iou of each class:
                for cat_idx in range(1):  # range(16)   # Loop through each category
                    io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # Print category iou
                io.cprint('Max class iou:%.5f' % best_class_iou)  # Print best class iou
                state = {
                    'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    'optimizer': opt.state_dict(), 'epoch': epoch_objf, 'test_class_iou': best_class_iou} #pre:epoch
                torch.save(state,
                           'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)  # Save model parameters, optimizer status, and current epoch number to 'checkpoints/%s/best_clsiou_model.pth' file

            # 4. when get the best precision, recall, f1_score, save the model:   #New added
            if test_metrics['precision'] > best_precision:
                best_precision = test_metrics['precision']
                io.cprint('Max Precision:%.5f' % best_precision)

            if test_metrics['recall'] > best_recall:
                best_recall = test_metrics['recall']
                io.cprint('Max Recall:%.5f' % best_recall)

            if test_metrics['f1_score'] > best_f1_score:
                best_f1_score = test_metrics['f1_score']
                io.cprint('Max F1 Score:%.5f' % best_f1_score)

            epoch_objf = epoch_objf + 1
        return float(test_metrics['loss'])  # Ensure return is a single value
    # ------------ end of objf code ------------


    # ============= Optimizer ================
    if args.use_acpa:
        print("Use ACPA")
        acpa_result, best_instance_iou, best_lr_for_iou = ACPA(objf=lambda lr: objf(lr), lb=1e-7, ub=1e-3, dim=1,
                                                              SearchAgents_no=60, Max_pos=1, PSO_iters=1, args=args) #Calculate times [ SearchAgents_no * 4 + SearchAgents_no / 2 ] * loops
        best_lr = best_lr_for_iou
        # Report best acc, ins_iou, cls_iou
        io.cprint('Final Max Acc:%.5f' % best_acc)  # Print final best accuracy
        io.cprint('Final Max instance iou:%.5f' % best_instance_iou)  # Print final best instance iou
        io.cprint('Final Max class iou:%.5f' % best_class_iou)  # Print final best class iou
        io.cprint('Final Max Precision:%.5f' % best_precision)
        io.cprint('Final Max Recall:%.5f' % best_recall)
        io.cprint('Final Max F1 Score:%.5f' % best_f1_score)
        opt = optim.Adam(model.parameters(), lr=best_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # Save last model
        state = {
            'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou}
        torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))  # Save model parameters, optimizer status, and last epoch number to 'checkpoints/%s/model_ep%d.pth' file
        print('Code running end')
    else:
        print("Use Adam")


def train_epoch(train_loader, model, opt, num_part, num_classes): # Initialize training loss, sample count, accuracy and shape iou
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    all_preds = []
    all_targets = []
    metrics = defaultdict(lambda: list())   # Dictionary for storing metrics
    model.train()  # Set model to training mode

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9): # Iterate through training dataset
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float()) # Convert data to Variable and move to GPU
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        # target: b,n
        # Get model output
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50  # Get model output
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0]) # Calculate loss

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward and parameter update
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum # Calculate total shape iou and sample count
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

        # all_preds + all_targets
        all_preds.append(pred_choice.cpu().numpy())
        all_targets.append(target.cpu().numpy())

        # Note: We do not need to calculate per_class iou during training

    # Calculate and save training metrics
    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count
    # ###########（Precision, Recall, F1-score）#################
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics['precision'] = skl_metrics.precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['recall'] = skl_metrics.recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['f1_score'] = skl_metrics.f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    # ############################################################

    return metrics, train_loss, count


def test_epoch(test_loader, model, num_part, num_classes):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    all_preds = []
    all_targets = []
    final_total_per_cat_iou = np.zeros(1).astype(np.float32)  # pre:16
    final_total_per_cat_seen = np.zeros(1).astype(np.int32)  # pre:16
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader),
                                                            smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
            Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
            target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        # sum:
        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

        # all_preds + all_targets
        all_preds.append(pred_choice.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    for cat_idx in range(1):  # pre:16
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[
                cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count
    metrics['loss'] = test_loss * 1.0 / count  # New added

    # ###########（Precision, Recall, F1-score）#################
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics['precision'] = skl_metrics.precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['recall'] = skl_metrics.recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    metrics['f1_score'] = skl_metrics.f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    # ############################################################

    return metrics, final_total_per_cat_iou, test_loss


def test(args, io):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=True) # default：normalize=False
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # Try to load models
    num_part = 2 #pre:50 now:2
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 2 #pre:50 now:2
    num_classes = 1 #num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((1)).astype(np.float32) #pre:16
    total_per_cat_seen = np.zeros((1)).astype(np.int32)#pre:16

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(1): #pre:16
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(1):#pre:16
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 1#pre：16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='RTreeNet')
    parser.add_argument('--exp_name', type=str, default='your_exp_name', metavar='N',  #default='demo1'
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',#default=32
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=5, metavar='batch_size', #default=32
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', #default=350
                        help='number of episode to train')
    parser.add_argument('--use_acpa', type=bool, default=True,
                        help='Use ACPA') # Use ACPA optimizer
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',   #pre:0.003 now:0.0001
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name

    _init_()

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    # Record start time
    start_time = time.time()

    if not args.eval:
        train(args, io)
        print('Code running to train(args, io)')
    else:
        test(args, io)

    # Record end time
    end_time = time.time()
    # Calculate and print running time
    io.cprint(f"Code running time: {end_time - start_time} seconds")








